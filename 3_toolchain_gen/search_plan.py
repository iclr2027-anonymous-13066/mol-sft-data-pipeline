"""Ref-free, property-driven constructive planner for the toolchain builder (S1b).

Δ-guided decoration with a scaffold guard, built on the **``suggest_edits`` tool**.
Starting from the described scaffold seed (``scaffold_smiles`` — which embeds
``scaffold_smarts`` by construction), each planner step is the exact two-tool loop
the agent runs at inference:

  1. measure the REAL property vector of the current molecule with *measure_fn*
     (the live ``analyze_properties`` tool);
  2. call :func:`molkit.utils.suggest_edits.suggest_edits` (the ``suggest_edits``
     tool) to rank the mmpdb-derived moves by predicted reduction in the normalised
     distance to the target property box — each candidate already an
     ``edit_fragment``-ready ``{from_smiles, to_smiles, anchors}`` triple;
  3. commit one candidate — via ``edit_fragment`` (the ``replace_fragment`` engine)
     — whose product still embeds the strict scaffold SMARTS (the *scaffold guard*);
  4. re-measure and iterate until every constraint is satisfied or the budget hits.

The search strategy over the ``suggest_edits`` candidates is one of:

* **greedy** (``plan_search``) — commit the single top-ranked guard-passing
  candidate each step; ~1 measurement/step, cheap bulk.
* **beam** (``beam_search``) — expand every beam node into its top-``expand_k``
  candidates, MEASURE them all, keep the ``beam_width`` with the smallest REAL gap;
  escapes the local minima greedy commits to, at ~``beam_width``×``expand_k``
  measurements/round.
* **hybrid** (``hybrid_search``) — greedy first, beam-rescue only the failures.

The committed move is chosen by ``suggest_edits``' *predicted* Δ (greedy) or the
*real measured* gap (beam), but termination and reported success are always driven
by the real measured vector — so error in the ADMET Δ's never yields a false
"satisfied".

Each emitted decorate step carries BOTH tool calls the agent makes:

    step["suggest"] = {"arguments": {mol_smiles, constraints, top_k},
                       "candidates": [<the suggest_edits output list>]}
    step["tool"]    = "edit_fragment"
    step["arguments"] = {mol_smiles, from_smiles, to_smiles, anchors}
    step["result"]  = <product SMILES>

so the builder emits ``suggest_edits`` immediately before the ``edit_fragment`` it
informs (see :meth:`ToolChainBuilder._suggest_step` / ``_edit_step``).

The returned plan carries ``seed`` plus phase-tagged ``steps``; this function is
called with the completed scaffold as its seed, so every step it emits is tagged
``decorate`` (a stray stereo step keeps its ``finalize`` tag).
"""
from __future__ import annotations

import asyncio
import logging
import os
import random
from typing import Awaitable, Callable, List, Optional, Sequence, Union

import numpy as np
from rdkit import Chem, RDLogger

from molkit.utils.suggest_edits import suggest_edits as _suggest_edits
from molkit.utils.molecule_edit_utils import replace_fragment as _replace_fragment

RDLogger.DisableLog("rdApp.*")
logger = logging.getLogger(__name__)

# The 16 indexed properties, in the fixed order used by build_mmp_moves
# (PHYS_DEFAULT + ADMET_DEFAULT).
PROP_ORDER = ["MW", "logP", "HBD", "HBA", "TPSA", "rotB", "rings_total", "QED",
              "MR", "heavy_atoms", "formal_charge", "logD", "logS", "BBBP",
              "HIA", "Mutag"]

# The five admet_ai model outputs. Under GPU contention the ADMET backend
# intermittently fails as a UNIT — returning None for every ADMET property at once
# while physchem (RDKit) is unaffected — so a measurement is "incomplete" iff a
# requested ADMET prop came back None, and worth retrying.
_ADMET = frozenset({"logD", "logS", "BBBP", "HIA", "Mutag"})

# Per-property normalisation for the multi-objective gap (z-score standardisation:
# "1 gap unit = 1 std of natural variation"). MIRROR of suggest_edits._SCALE /
# agentic_eval._PROP_SCALE — keep the three in sync.
_SCALE = {"MW": 86.0, "logP": 1.26, "HBD": 0.85, "HBA": 1.62, "TPSA": 26.0,
          "rotB": 1.93, "rings_total": 1.08, "QED": 0.15, "MR": 24.0,
          "heavy_atoms": 6.24, "formal_charge": 1.0, "logD": 1.36, "logS": 1.56,
          "BBBP": 0.125, "HIA": 0.1, "Mutag": 0.19}

DEFAULT_MOVES_DIR = os.environ.get(
    "MMP_MOVES_DIR", "data/mmp_moves")


# ---------------------------------------------------------------------------
# compatibility shim: the builder used to build a MoveLibrary and pass it in.
# suggest_edits owns its own (disk-backed) move cache now, so the library is a
# no-op handle kept only so ``lib=get_move_library()`` call sites stay valid.
# ---------------------------------------------------------------------------
class MoveLibrary:  # pragma: no cover - thin compat handle
    """Deprecated. Kept so ``lib=...`` call sites don't break; suggest_edits owns
    the (disk-cached) move set now (see :mod:`molkit.utils.suggest_edits`)."""

    def __init__(self, moves_dir: str = DEFAULT_MOVES_DIR, **_kw) -> None:
        self.moves_dir = moves_dir


def get_move_library(moves_dir: str = DEFAULT_MOVES_DIR, **kw) -> MoveLibrary:
    """Deprecated no-op handle (suggest_edits owns the move cache)."""
    return MoveLibrary(moves_dir, **kw)


# ---------------------------------------------------------------------------
# gap helpers (dict-based, over the target properties only)
# ---------------------------------------------------------------------------
def _canon(smiles: Optional[str]) -> Optional[str]:
    if not smiles:
        return None
    m = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(m) if m is not None else None


def _to_constraints(targets: dict) -> dict:
    """suggest_edits ``constraints`` = ``{prop: [lo, hi]}`` (None for an open bound),
    which is exactly the ``extract_properties`` shape — normalise to plain lists."""
    return {k: [v[0], v[1]] for k, v in targets.items()}


def _norm_gap(props: dict, targets: dict) -> float:
    """Total normalised (z-score) distance of *props* outside the target box.

    ``inf`` if any target property is missing / unmeasured (an unusable candidate)."""
    total = 0.0
    for k, (lo, hi) in targets.items():
        v = props.get(k)
        if v is None:
            return float("inf")
        try:
            v = float(v)
        except (TypeError, ValueError):
            return float("inf")
        s = _SCALE.get(k, 1.0)
        g = max(0.0,
                (lo - v) if lo is not None else 0.0,
                (v - hi) if hi is not None else 0.0)
        total += g / s
    return total


def _all_satisfied(props: dict, targets: dict) -> bool:
    for k, (lo, hi) in targets.items():
        v = props.get(k)
        if v is None:
            return False
        try:
            v = float(v)
        except (TypeError, ValueError):
            return False
        if lo is not None and v < lo:
            return False
        if hi is not None and v > hi:
            return False
    return True


async def measure_with_retry(measure_fn, smi: str, props: list, *,
                             retries: int = 8, backoff: float = 0.35) -> dict:
    """Call *measure_fn*, retrying while requested ADMET props come back all-None.

    Works around the transient, load-dependent ADMET backend failure (see
    ``_ADMET``): the values are deterministic once they return, so a bounded retry
    with linear backoff recovers a complete measurement in the common case and
    gives up (returning the last response, ADMET possibly None) otherwise.
    Physchem-only requests never trigger a retry.
    """
    want = [p for p in props if p in _ADMET]
    last: dict = {}
    for attempt in range(retries + 1):
        res = await measure_fn(smi, props)
        if isinstance(res, dict):
            last = res
            if not want or all(res.get(p) is not None for p in want):
                return res
        if attempt < retries:
            await asyncio.sleep(backoff * (1 + attempt * 0.5))
    return last


# ---------------------------------------------------------------------------
# suggest_edits oracle + candidate application (shared by greedy and beam)
# ---------------------------------------------------------------------------
async def _suggest(cur: str, targets: dict, props: dict, *, top_k: int,
                   max_cut: int, context_aware: bool,
                   scaffold_smarts: Optional[GuardSmarts] = None) -> List[dict]:
    """The ``suggest_edits`` tool, ranked for *cur* against *targets*, given the
    already-measured *props* (so ADMET isn't re-computed). When *scaffold_smarts* is
    set, only scaffold-preserving edits are returned. Returns the candidate list
    (each an edit_fragment-ready dict) — [] on any failure."""
    constraints = _to_constraints(targets)
    try:
        return await asyncio.to_thread(
            _suggest_edits, cur, constraints, top_k,
            props=dict(props), max_cut=max_cut,
            context_aware=context_aware, scaffold_smarts=scaffold_smarts or None)
    except Exception as exc:  # noqa: BLE001 - measurement gap / bad SMILES → no moves
        logger.debug("suggest_edits failed on %s: %s", cur, exc)
        return []


# What has to survive every edit. A scaffold task passes ONE big SMARTS; a
# functional-group task passes one per required group, and each is checked on its own
# — joining them with '.' would demand atom-disjoint matches and reject molecules
# where two groups legitimately share an atom (measured: 13.1% of two-group rows).
GuardSmarts = Union[str, Sequence[str], None]


def _guard_mols(smarts: GuardSmarts) -> List[Chem.Mol]:
    """Parse the guard spec into query mols. Unparseable entries are dropped."""
    if not smarts:
        return []
    items = [smarts] if isinstance(smarts, str) else list(smarts)
    out = []
    for s in items:
        if not s:
            continue
        m = Chem.MolFromSmarts(s)
        if m is not None:
            out.append(m)
    return out


# Passed as `visited` when a search runs with dedup off: nothing has been seen, ever.
_NO_SEEN: frozenset = frozenset()


def _apply(cur: str, cand: dict, guard: Optional[List[Chem.Mol]], visited: set) -> Optional[str]:
    """Apply *cand* via the ``edit_fragment`` (``replace_fragment``) engine, pinned
    with its anchors. Returns the guard-passing, unvisited product SMILES, or None.

    ``edit_fragment`` returns ``products[0]``; with the candidate's anchors the swap
    is pinned to one site so that product is fully determined — exactly what the
    stored ``result`` records.
    """
    # `suggest_edits` built this product to rank and guard-check the candidate, and
    # since it started reporting it there is no reason to run the same reaction again.
    # On a depth-5 expansion that reaction was ~40% of the worker's MAIN-THREAD time,
    # which is the scarce resource — a worker is GIL-bound at ~1.3 cores, so main-thread
    # work is what caps how much of a machine one process can use. Falls back when the
    # key is absent (restoration candidates, or an older suggest_edits).
    ready = cand.get("product")
    if ready is None:
        anchors = {int(k): int(v) for k, v in cand["anchors"].items()}
        prods = _replace_fragment(cur, cand["from_smiles"], cand["to_smiles"],
                                  anchors=anchors)
        if not prods:
            return None
        ready = prods[0]
    prod = _canon(ready)
    if prod is None or prod in visited:
        return None
    if guard:
        m = Chem.MolFromSmiles(prod)
        if m is None or not all(m.HasSubstructMatch(g) for g in guard):
            return None
    return prod


def _make_step(cur: str, targets: dict, top_k: int, candidates: List[dict],
               chosen: dict, product: str,
               scaffold_smarts: Optional[GuardSmarts] = None) -> dict:
    """A decorate step carrying the ``suggest_edits`` call (+ its full candidate
    list) and the ``edit_fragment`` call that commits *chosen*. The suggest call
    records the EXACT arguments the search used (top_k, scaffold_smarts) so the
    emitted SFT trajectory matches the generation-time call."""
    anchors = {str(k): int(v) for k, v in chosen["anchors"].items()}
    suggest_args = {"mol_smiles": cur,
                    "constraints": _to_constraints(targets),
                    "top_k": top_k}
    if scaffold_smarts:
        # A single guard is recorded as a bare STRING, not a one-element list: the
        # scaffold task has always emitted it that way and the trajectory text is what
        # the model learns, so wrapping it would silently change every scaffold chain.
        guards = ([scaffold_smarts] if isinstance(scaffold_smarts, str)
                  else [g for g in scaffold_smarts if g])
        if guards:
            suggest_args["scaffold_smarts"] = guards[0] if len(guards) == 1 else guards
    return {
        "phase": "decorate",
        "suggest": {
            "arguments": suggest_args,
            "candidates": candidates,
        },
        "tool": "edit_fragment",
        "arguments": {"mol_smiles": cur, "from_smiles": chosen["from_smiles"],
                      "to_smiles": chosen["to_smiles"], "anchors": anchors},
        "result": product,
    }


# ---------------------------------------------------------------------------
# planners
# ---------------------------------------------------------------------------
MeasureFn = Callable[[str, list], Awaitable[dict]]


async def _forward_walk(
    seed_smiles: str,
    scaffold_smarts: GuardSmarts,
    targets: dict[str, list],
    measure_fn: MeasureFn,
    *,
    max_steps: int = 12,
    top_k: int = 4,
    max_cut: int = 3,
    context_aware: bool = True,
    measure_retries: int = 8,
    order_fn: Optional[Callable[[List[dict]], List[dict]]] = None,
) -> Optional[dict]:
    """One-molecule-at-a-time forward walk shared by the single-path planners.

    Each step: measure → ``suggest_edits`` → commit ONE guard-passing candidate →
    re-measure, for up to *max_steps* steps (~1 measurement/step).

    *order_fn* re-orders the ``suggest_edits`` candidate list before the walk takes
    the first applicable candidate from it. ``None`` keeps the tool's own ranking
    (predicted_gap ascending) — the greedy top-1 rule of :func:`plan_search`;
    :func:`random_search` passes a shuffler. The candidate list STORED in the step
    is always the tool's own (unreordered) output, so the emitted trajectory shows
    the ranking the agent would see at inference regardless of the pick rule.
    """
    seed = _canon(seed_smiles)
    if seed is None:
        return None
    guard = _guard_mols(scaffold_smarts)
    req_props = list(targets)
    incomplete = {"n": 0}
    n_measured = {"n": 0}
    # Search size, separately from measurement cost: how many candidate edits the
    # suggester offered, and how many of them became a real molecule. The greedy walk
    # takes the FIRST applicable candidate, so it builds one product per step while
    # having been offered top_k — the gap between the two columns is what a wider pick
    # rule has to spend measurements on.
    n_cand = {"n": 0}
    n_built = {"n": 0}

    async def measure(smi: str) -> Optional[dict]:
        res = await measure_with_retry(measure_fn, smi, req_props, retries=measure_retries)
        n_measured["n"] += 1
        if not isinstance(res, dict):
            return None
        if any(res.get(p) is None for p in req_props if p in _ADMET):
            incomplete["n"] += 1
        return res

    def _counts() -> dict:
        return dict(incomplete=incomplete["n"], n_measured=n_measured["n"],
                    n_candidates=n_cand["n"], n_products=n_built["n"])

    steps: list[dict] = []
    cur = seed
    visited = {seed}

    props = await measure(cur)
    if props is None:
        return _finish(seed, [], seed, satisfied=False, steps_taken=0, **_counts())
    best_gap, best_prefix, best_mol = _norm_gap(props, targets), 0, cur
    if _all_satisfied(props, targets):
        return _finish(seed, [], seed, satisfied=True, steps_taken=0, **_counts())

    for _step in range(max_steps):
        cands = await _suggest(cur, targets, props, top_k=top_k, max_cut=max_cut,
                               context_aware=context_aware, scaffold_smarts=scaffold_smarts)
        n_cand["n"] += len(cands)
        committed = None
        for cand in (order_fn(cands) if order_fn is not None else cands):
            prod = _apply(cur, cand, guard, visited)
            if prod is not None:
                committed = (cand, prod)
                break
        if committed is None:
            break   # stuck: no improving / guard-passing candidate
        cand, product = committed
        n_built["n"] += 1
        steps.append(_make_step(cur, targets, top_k, cands, cand, product,
                                scaffold_smarts=scaffold_smarts))
        cur = product
        visited.add(cur)
        props = await measure(cur)
        if props is None:
            break
        g = _norm_gap(props, targets)
        if g < best_gap - 1e-9:
            best_gap, best_prefix, best_mol = g, len(steps), cur
        if _all_satisfied(props, targets):
            return _finish(seed, steps, cur, satisfied=True, steps_taken=len(steps),
                           **_counts())

    return _finish(seed, steps[:best_prefix], best_mol, satisfied=False,
                   steps_taken=len(steps), final_gap=best_gap, **_counts())


async def plan_search(
    seed_smiles: str,
    scaffold_smarts: GuardSmarts,
    targets: dict[str, list],
    measure_fn: MeasureFn,
    *,
    lib: Optional[MoveLibrary] = None,          # accepted for compat, ignored
    max_steps: int = 12,
    top_k: int = 4,
    max_cut: int = 3,
    context_aware: bool = True,
    measure_retries: int = 8,
    **_ignored,
) -> Optional[dict]:
    """Greedy ``suggest_edits``-guided ref-free decoration plan (module docstring).

    Each step: measure → ``suggest_edits`` → commit the single **top-ranked**
    (smallest ``predicted_gap``) guard-passing candidate → re-measure. Returns a plan
    dict (best molecule found, its edit prefix, phase-tagged), or ``None`` if
    *seed_smiles* is unparseable.
    """
    return await _forward_walk(
        seed_smiles, scaffold_smarts, targets, measure_fn, max_steps=max_steps,
        top_k=top_k, max_cut=max_cut, context_aware=context_aware,
        measure_retries=measure_retries, order_fn=None)


async def random_search(
    seed_smiles: str,
    scaffold_smarts: GuardSmarts,
    targets: dict[str, list],
    measure_fn: MeasureFn,
    *,
    lib: Optional[MoveLibrary] = None,          # accepted for compat, ignored
    max_steps: int = 12,
    top_k: int = 4,
    max_cut: int = 3,
    context_aware: bool = True,
    measure_retries: int = 8,
    rng_seed: int = 0,
    **_ignored,
) -> Optional[dict]:
    """Ablation baseline: the greedy walk with the ``predicted_gap`` RANKING removed.

    Identical to :func:`plan_search` — same ``suggest_edits`` call, same top_k
    candidate pool, same scaffold guard, same measurement budget — except the
    committed candidate is drawn uniformly at random from the pool instead of being
    its top-ranked member. It isolates how much of greedy's success comes from the
    predicted Δ ranking rather than from mmpdb move applicability alone.

    *rng_seed* seeds the per-instance shuffle, so a run is reproducible.
    """
    rng = random.Random(rng_seed)

    def shuffled(cands: List[dict]) -> List[dict]:
        out = list(cands)
        rng.shuffle(out)
        return out

    return await _forward_walk(
        seed_smiles, scaffold_smarts, targets, measure_fn, max_steps=max_steps,
        top_k=top_k, max_cut=max_cut, context_aware=context_aware,
        measure_retries=measure_retries, order_fn=shuffled)


async def beam_search(
    seed_smiles: str,
    scaffold_smarts: GuardSmarts,
    targets: dict[str, list],
    measure_fn: MeasureFn,
    *,
    lib: Optional[MoveLibrary] = None,          # accepted for compat, ignored
    beam_width: int = 4,
    expand_k: int = 4,
    max_rounds: int = 8,
    top_k: int = 4,
    max_cut: int = 3,
    context_aware: bool = True,
    measure_retries: int = 8,
    measure_batch_fn: Optional[Callable] = None,
    dedup: bool = True,
    traj_select: str = "best",
    traj_rng_seed: int = 0,
    **_ignored,
) -> Optional[dict]:
    """Beam-search ref-free decoration over ``suggest_edits`` candidates: keep the
    *beam_width* best partial molecules; each round expand every beam node into up to
    *expand_k* guard-passing candidates, **measure them all** (batched when possible),
    and keep the *beam_width* with the smallest real normalised gap. Selection uses
    the REAL measured vector (not the predicted Δ), so it escapes greedy's local
    minima — at ~*beam_width*×*expand_k* measurements per round instead of 1.

    ``measure_batch_fn`` (optional): ``(smiles_list, props) -> {smiles: dict}``. When
    given, a round's candidates are measured in ONE batched call. These
    search-internal measurements are throwaway (only the builder's final-chain steps
    are stored), so batching never affects stored output.

    ``dedup`` (default on) drops a product already seen anywhere else in the search
    rather than making it a node again. It costs no coverage — the same molecule has
    the same subtree — but it does make the measurement count a function of how much
    the branches happen to collide, which is not a property of the budget. Turn it OFF
    (:func:`exhaustive_search` does) when the count has to be the tree's own size.

    Same plan shape / seed contract as :func:`plan_search`.
    """
    seed = _canon(seed_smiles)
    if seed is None:
        return None
    guard = _guard_mols(scaffold_smarts)
    req_props = list(targets)
    incomplete = {"n": 0}
    n_measured = {"n": 0}
    # See _forward_walk: candidates offered vs products actually built. Here the two
    # track each other closely (every guard-passing product becomes a measured node),
    # so their ratio reads as the guard/dedup rejection rate of the expansion.
    n_cand = {"n": 0}
    n_built = {"n": 0}

    def _account(res: dict) -> dict:
        n_measured["n"] += 1
        if isinstance(res, dict) and any(res.get(p) is None for p in req_props if p in _ADMET):
            incomplete["n"] += 1
        return res if isinstance(res, dict) else {}

    def _counts() -> dict:
        return dict(incomplete=incomplete["n"], n_measured=n_measured["n"],
                    n_candidates=n_cand["n"], n_products=n_built["n"])

    async def measure(smi: str) -> Optional[dict]:
        res = await measure_with_retry(measure_fn, smi, req_props, retries=measure_retries)
        return _account(res) if isinstance(res, dict) else _account({})

    async def measure_many(smis: list) -> list:
        res = await measure_batch_fn(smis, req_props)
        return [_account(res.get(s) if isinstance(res, dict) else {}) for s in smis]

    props0 = (await measure_many([seed]))[0] if measure_batch_fn else await measure(seed)
    if not props0:
        return _finish(seed, [], seed, satisfied=False, steps_taken=0, **_counts())
    if _all_satisfied(props0, targets):
        return _finish(seed, [], seed, satisfied=True, steps_taken=0, **_counts())

    beam = [{"mol": seed, "props": props0, "gap": _norm_gap(props0, targets), "steps": []}]
    visited = {seed} if dedup else set()
    best = beam[0]

    for _round in range(max_rounds):
        # 1. Expand every beam node via suggest_edits; collect the guard-passing
        #    products (keep each candidate's full suggest list). With `dedup` on they
        #    are also deduped against every product seen so far.
        cand: list = []
        seen: set = set()
        for node in beam:
            # suggest exactly the expand_k candidates beam explores per node, so the
            # emitted trajectory shows the SAME top_k the search actually used (#7).
            cands = await _suggest(node["mol"], targets, node["props"],
                                   top_k=expand_k, max_cut=max_cut,
                                   context_aware=context_aware, scaffold_smarts=scaffold_smarts)
            n_cand["n"] += len(cands)
            kept = 0
            for c in cands:
                if kept >= expand_k:
                    break
                prod = _apply(node["mol"], c, guard, (visited | seen) if dedup else _NO_SEEN)
                if prod is None or (dedup and prod in seen):
                    continue
                seen.add(prod)
                cand.append((node, cands, c, prod))
                kept += 1
            n_built["n"] += kept
        if not cand:
            break
        if dedup:
            visited |= seen
        # 2. Measure all candidate products (the expensive part).
        products = [p for _n, _cl, _c, p in cand]
        if measure_batch_fn is not None:
            vs = await measure_many(products)
        else:
            vs = await asyncio.gather(*(measure(p) for p in products))
        # 3. Build children; early-exit on satisfaction; keep the best beam_width.
        children: list = []
        for (node, cands, chosen, product), props in zip(cand, vs):
            if not props or _norm_gap(props, targets) == float("inf"):
                continue
            step = _make_step(node["mol"], targets, expand_k, cands, chosen, product,
                              scaffold_smarts=scaffold_smarts)
            g = _norm_gap(props, targets)
            child = {"mol": product, "props": props, "gap": g,
                     "steps": node["steps"] + [step]}
            children.append(child)
            if g < best["gap"]:
                best = child
            if _all_satisfied(props, targets):
                # The search stops here, so this child's path is the ONLY path it
                # completed — every traj_select agrees on it.
                return _finish(seed, child["steps"], product, satisfied=True,
                               steps_taken=len(child["steps"]), **_counts())
        if not children:
            break
        children.sort(key=lambda n: n["gap"])
        beam = children[:beam_width]

    traj_node = _select_traj_node(beam, targets, traj_select, traj_rng_seed)
    return _finish(seed, best["steps"], best["mol"],
                   satisfied=_all_satisfied(best["props"], targets),
                   steps_taken=len(best["steps"]), final_gap=best["gap"],
                   traj_steps=(traj_node or {}).get("steps"), **_counts())


async def exhaustive_search(
    seed_smiles: str,
    scaffold_smarts: GuardSmarts,
    targets: dict[str, list],
    measure_fn: MeasureFn,
    *,
    lib: Optional[MoveLibrary] = None,          # accepted for compat, ignored
    top_k: int = 4,
    max_depth: int = 3,
    max_cut: int = 3,
    context_aware: bool = True,
    measure_retries: int = 8,
    measure_batch_fn: Optional[Callable] = None,
    dedup: bool = False,
    traj_select: str = "best",
    traj_rng_seed: int = 0,
    **_ignored,
) -> Optional[dict]:
    """Full enumeration of the ``suggest_edits`` candidate tree to *max_depth*.

    This is :func:`beam_search` with the pruning AND the dedup removed — every node of
    every level is kept and expanded, so the search visits the tree's own size and
    returns the shallowest molecule that satisfies the box (or the smallest-gap one
    seen). It is the upper bound the pruned strategies are measured against: with
    ``top_k=4, max_depth=3`` it measures ≤ 4+16+64 = 84 molecules per instance
    against greedy's ≤ 4, so it is a benchmark, not a production planner.

    THE MEASUREMENT COUNT. Levels run in order, 1..``max_depth``. A level is measured
    IN FULL before any of it is tested for satisfaction, so finding the answer at depth
    d still charges the whole of level d — the count is then

        n_measured = 1 (the seed) + Σ_{i=1..d} |level i|

    where ``|level i|`` is the sum over level-(i-1) nodes of however many candidates
    ``suggest_edits`` returned for that node, capped at ``top_k``. That is ``top_k**i``
    exactly when every node returns a full ``top_k``, which is what makes the count
    readable as a budget: at ``top_k=4, max_depth=5`` the ceiling is
    1 + 4 + 16 + 64 + 256 + 1024 = 1365.

    Two things still put it under that ceiling, and both are real rather than
    bookkeeping. A node whose ``suggest_edits`` returns fewer than ``top_k`` eligible
    moves branches by that smaller number (and a level where every node returns nothing
    ends the search early). And a candidate whose ``replace_fragment`` builds nothing,
    or whose product fails the guard, has no molecule to measure — those show up as the
    shortfall of ``n_products`` against ``n_candidates``.

    What does NOT reduce it any more is branch collision: ``dedup=False`` means a
    product already built elsewhere in the tree is measured and expanded again. That
    costs budget for no coverage (the same molecule has the same subtree), which is
    exactly why the pruned searches keep the dedup — but it is the only way the count
    reports the size of the tree the strategy is defined by, rather than how much this
    particular molecule's branches happened to overlap. Repeats are cheap in wall time:
    the harness memoises both tools per instance, so a re-visited node re-measures from
    cache. It is the COUNTER, not the compute, that this restores.

    ``dedup=True`` puts the collision pruning back, trading the readable count for the
    repeated subtrees. Nothing asks for it today — every caller here reports the count —
    and whether the search SATISFIES is the same either way.
    """
    return await beam_search(
        seed_smiles, scaffold_smarts, targets, measure_fn,
        beam_width=1 << 30,               # no pruning: keep every child
        expand_k=top_k, max_rounds=max_depth, top_k=top_k, max_cut=max_cut,
        context_aware=context_aware, measure_retries=measure_retries,
        measure_batch_fn=measure_batch_fn, dedup=dedup,
        traj_select=traj_select, traj_rng_seed=traj_rng_seed)


async def hybrid_search(
    seed_smiles: str,
    scaffold_smarts: GuardSmarts,
    targets: dict[str, list],
    measure_fn: MeasureFn,
    *,
    lib: Optional[MoveLibrary] = None,          # accepted for compat, ignored
    max_steps: int = 12,
    beam_width: int = 4,
    expand_k: int = 4,
    top_k: int = 4,
    max_cut: int = 3,
    context_aware: bool = True,
    measure_retries: int = 8,
    measure_batch_fn: Optional[Callable] = None,
    **_ignored,
) -> Optional[dict]:
    """Greedy first, then beam-rescue: run the cheap greedy search; if it already
    satisfies every constraint, keep it (most instances, ~1 measurement/step);
    otherwise run beam search and take it when it satisfies or lands closer to the
    box. ``n_measured`` / ``n_candidates`` / ``n_products`` are the TOTAL spent
    (greedy + beam)."""
    greedy = await plan_search(seed_smiles, scaffold_smarts, targets, measure_fn,
                               max_steps=max_steps, top_k=top_k, max_cut=max_cut,
                               context_aware=context_aware, measure_retries=measure_retries)
    if greedy is None or greedy.get("search_satisfied"):
        return greedy
    beam = await beam_search(seed_smiles, scaffold_smarts, targets, measure_fn,
                             beam_width=beam_width, expand_k=expand_k, max_rounds=max_steps,
                             top_k=top_k, max_cut=max_cut, context_aware=context_aware,
                             measure_retries=measure_retries, measure_batch_fn=measure_batch_fn)
    if beam is None:
        return greedy
    take_beam = (beam.get("search_satisfied")
                 or beam.get("final_gap", float("inf")) < greedy.get("final_gap", float("inf")))
    chosen = dict(beam if take_beam else greedy)
    for k in ("n_measured", "n_candidates", "n_products"):
        chosen[k] = greedy.get(k, 0) + beam.get(k, 0)
    return chosen


def _sat_ratio(props: dict, targets: dict) -> float:
    """Fraction of the target constraints *props* satisfies. 0.0 when unmeasured."""
    if not targets:
        return 0.0
    n = 0
    for k, (lo, hi) in targets.items():
        v = props.get(k) if isinstance(props, dict) else None
        if v is None:
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if (lo is None or v >= lo) and (hi is None or v <= hi):
            n += 1
    return n / len(targets)


def _select_traj_node(nodes: list, targets: dict, how: str, rng_seed: int):
    """Pick the path whose steps are REPORTED as the search's trajectory.

    ``nodes`` are the leaves the search ended on — the surviving beam for
    ``beam_search``, every leaf for ``exhaustive_search`` (its beam is unbounded).

    * ``best``       the answer's own path; the caller passes None and _finish falls
                     back to ``steps``.
    * ``random``     one leaf uniformly at random. For a beam this is the honest
                     per-path number: the beam measured every leaf and then reported
                     its best, so the best path's efficiency is a max over N paths,
                     not the efficiency of running the strategy once.
    * ``sat_argmax`` the leaf satisfying the most constraints (ties -> smallest gap).
                     For the exhaustive tree "the path it took" is otherwise undefined.

    Deterministic: the RNG is seeded per (instance, arm) by the caller.
    """
    if not nodes or how == "best":
        return None
    if how == "random":
        import random as _random
        return _random.Random(rng_seed).choice(nodes)
    if how == "sat_argmax":
        return max(nodes, key=lambda nd: (_sat_ratio(nd.get("props") or {}, targets),
                                          -float(nd.get("gap", float("inf")))))
    return None


def _finish(seed: str, steps: list, final_mol: str, *, satisfied: bool,
            steps_taken: int, incomplete: int = 0, n_measured: int = 0,
            n_candidates: int = 0, n_products: int = 0,
            final_gap: float = float("inf"), traj_steps: list = None) -> dict:
    """Assemble the plan dict (decorate-only from the given seed).

    ``traj_steps`` is the path to REPORT as this search's trajectory, which is not
    always the path it answers with: a beam or an exhaustive search explores many
    root-to-leaf paths and returns the best of them, so reporting that one as "the
    trajectory" makes its per-edit efficiency look like the best of N paths rather
    than like a path. ``beam_search``'s ``traj_select`` chooses which path fills this
    (see there); ``None`` means "the answer's own path", which is what every
    single-path planner has.
    """
    return {
        # The path the trajectory metrics (edit efficiency, productive-edit ratio)
        # are read off. Equal to `steps` unless traj_select asked for another path.
        "traj_steps": traj_steps if traj_steps is not None else steps,
        "final_gap": final_gap,
        "seed": seed,
        "ref": None,
        "ref_iso": final_mol,
        "final_mol": final_mol,
        "scaffold_smiles": seed,
        "steps": steps,
        "n_edit": len(steps),
        "n_attach": 0,
        "n_form_bond": 0,
        "n_scaffold_steps": 0,
        "n_decorate_steps": len(steps),
        "has_stereo_step": False,
        "search_satisfied": satisfied,
        "search_steps_taken": steps_taken,
        "measure_incomplete": incomplete,
        "n_measured": n_measured,
        # Search size (not measurement cost): candidate edits the suggester offered,
        # and how many of them the search actually turned into a molecule.
        "n_candidates": n_candidates,
        "n_products": n_products,
    }
