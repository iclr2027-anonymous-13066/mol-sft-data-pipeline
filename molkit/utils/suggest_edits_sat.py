#!/usr/bin/env python
"""``suggest_edits`` with a SWAPPABLE ranking objective.

:func:`molkit.utils.suggest_edits.suggest_edits` scores every mmpdb move by one
fixed objective: the predicted reduction in the z-score-normalised distance to the
target property box (``Σ_p box_distance_p / std_p``). This module keeps that tool's
machinery bolt-for-bolt and makes the objective a parameter, so the ranking rule can be
ablated on its own.

``score="count"``      (default) the number of constraints SATISFIED after the edit —
                       a discrete objective. ``gate`` decides eligibility and
                       ``tie_break`` decides what happens inside a tied group.
``score="minmax"``     ``Σ_p box_distance_p / maxdev_p``, where ``maxdev_p`` is the
                       LARGEST box distance property *p* can reach anywhere in the
                       population. Every property then contributes in [0, 1], so the
                       sum is bounded by the number of constraints and no property can
                       dominate by having wide natural units.
``score="worst"``      ``max_p box_distance_p / std_p`` (the Chebyshev / L-inf
                       distance) — minimise the WORST property instead of the sum.
                       Every constraint has to hold for success, so the binding one is
                       the worst; a sum lets a big win on one axis pay for a loss on the
                       axis that is actually blocking.
``score="gapstd"``     the SHIPPED objective moved off the mean. The tool ranks by the
                       normalised box distance of the mean prediction,
                       ``Σ_p gap_p(v_p + D_p)/std_p``; this scores that same distance at
                       the two ENDPOINTS of the move's own ±S interval and averages
                       them, ``Σ_p [gap_p(v_p + D_p + S_p) + gap_p(v_p + D_p − S_p)]
                       / (2·std_p)``. The mean point itself is NOT in the sum. Because
                       the box distance is convex, that average is never below the
                       mean-point distance, so the objective is the shipped one plus a
                       penalty that grows with the move's spread and vanishes once BOTH
                       endpoints land inside the box: at equal predicted gap the
                       reliable move wins, and a move whose mean lands in the box only
                       because it is a coin flip does not.
``score="prob"``       the EXPECTED number of satisfied constraints, taking the move's
                       own Δ spread seriously: ``Σ_p P(lo_p ≤ v_p + Δ_p ≤ hi_p)`` with
                       ``Δ_p ~ N(mean, std)`` from the move set's ``(D, S)``. Every
                       other objective here (and the shipped tool) uses ``D`` and
                       throws ``S`` away, so this is the only one that prefers a
                       reliable move over a high-variance one with a better mean.

Why a separate module rather than re-ranking suggest_edits' output
-----------------------------------------------------------------
The tool only ever returns candidates it already judged gap-REDUCING, in gap order. So
re-ranking its top-K can only reorder the gap-feasible prefix, and the edits a different
objective most wants — the ones that satisfy one more constraint, or fix the worst
property, while the total normalised distance gets *worse* — are not in that list at
all. Scoring has to happen over the whole move set to be tested honestly, which is what
this does: the same ~280k moves, the same vectorised pass, only a different score.

Everything that is not the objective is IMPORTED from ``suggest_edits`` rather than
reimplemented — the move index and its disk cache, the from-fragment queries, anchor-site
enumeration, the clean-swap product builder, the guard split, the context-Δ refinement,
the restoration scan, and every budget constant. That is deliberate: a difference in
outcome between two objectives is then attributable to the objective and nothing else.
``suggest_edits`` itself is not modified or monkey-patched.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from rdkit import Chem, RDLogger
from scipy.special import ndtr           # standard-normal CDF, vectorised

from molkit.utils.molecule_edit_utils import replace_fragment
from molkit.utils.suggest_edits import (
    MOVES_DIR, PROP_ORDER, _ANCHOR_VARIANTS, _GUARD_MATCH_CAP, _INT_PROPS, _PIDX,
    _REPL_BUDGET, _SCALE, _SITE_TRY, _anchor_sites, _best_per_from, _by_from,
    _context_delta, _cut_core, _gap, _load_all, _measure, _products_by_anchor,
    _query, _restore_scan, _split_sites, _TOSET)

RDLogger.DisableLog("rdApp.*")

SCORES = ("count", "minmax", "worst", "prob", "gapstd")
GATES = ("strict", "lex")
TIE_BREAKS = ("random", "gap")

# Population range used by ``score="minmax"``. Percentiles, not the true min/max: the
# extremes of a million molecules are outliers (one 2000-Da peptide would shrink every
# other molecule's MW term by an order of magnitude), so the 1st/99th percentile is the
# stable version of "as far out as this property realistically goes".
_RANGE_PCT = (1.0, 99.0)
_RANGE_CACHE: Dict[str, dict] = {}
_SIGMA_FLOOR = 1e-6          # moves with no recorded spread are treated as exact


def population_range(moves_dir: str = MOVES_DIR) -> Dict[str, Tuple[float, float]]:
    """``{prop: (p1, p99)}`` over the move set's own property table, cached to disk.

    Read from ``<moves_dir>/sample.props`` — the same table the Δ's were derived from,
    so the normalisation and the deltas describe one population. ~1M rows, so the
    percentiles are computed once and memoised to ``.prop_range.json``.
    """
    hit = _RANGE_CACHE.get(moves_dir)
    if hit is not None:
        return hit
    cache = os.path.join(moves_dir, ".prop_range.json")
    if os.path.exists(cache):
        try:
            with open(cache) as fh:
                got = {k: tuple(v) for k, v in json.load(fh).items()}
            _RANGE_CACHE[moves_dir] = got
            return got
        except Exception:  # noqa: BLE001 - corrupt cache -> recompute
            pass
    path = os.path.join(moves_dir, "sample.props")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found; score='minmax' needs the property "
                                f"table the move set was built from")
    arr = np.genfromtxt(path, delimiter="\t", skip_header=1,
                        usecols=range(1, len(PROP_ORDER) + 1), dtype=float)
    lo = np.nanpercentile(arr, _RANGE_PCT[0], axis=0)
    hi = np.nanpercentile(arr, _RANGE_PCT[1], axis=0)
    got = {p: (float(lo[i]), float(hi[i])) for i, p in enumerate(PROP_ORDER)}
    try:
        with open(cache + ".tmp", "w") as fh:
            json.dump({k: list(v) for k, v in got.items()}, fh, indent=2)
        os.replace(cache + ".tmp", cache)
    except Exception:  # noqa: BLE001 - a read-only moves dir just recomputes
        pass
    _RANGE_CACHE[moves_dir] = got
    return got


def _maxdev(keys: List[str], lo: np.ndarray, hi: np.ndarray,
            moves_dir: str = MOVES_DIR) -> np.ndarray:
    """Largest box distance each constrained property can reach in the population.

    ``max(lo_p - pool_lo_p, pool_hi_p - hi_p)`` — how far outside its own box the
    property can get without leaving the space of real molecules. Floored so a box that
    already spans the whole population cannot divide by ~0.
    """
    rng = population_range(moves_dir)
    out = []
    for j, p in enumerate(keys):
        p_lo, p_hi = rng[p]
        below = max(0.0, float(lo[j]) - p_lo) if lo[j] > -1e17 else 0.0
        above = max(0.0, p_hi - float(hi[j])) if hi[j] < 1e17 else 0.0
        out.append(max(below, above, 1e-9))
    return np.array(out)


def _round_int_props(dq: np.ndarray, keys: List[str]) -> np.ndarray:
    """Round the Δ of integer-valued properties to whole numbers (on a copy).

    HBA/HBD/rings/rotB/heavy_atoms/formal_charge only take integer values on a real
    molecule, so a predicted ΔHBA of 0.4 cannot be realised: counting a constraint as
    satisfied at 4.4 would score a molecule that cannot exist. Rounding also matches the
    ``delta`` the tool REPORTS (it emits ints for these), so the score agrees with the
    number a caller reads. Distance-based objectives need no such treatment — they
    measure how far away it is, not whether it is inside.
    """
    out = np.array(dq, dtype=float, copy=True)
    for j, p in enumerate(keys):
        if p in _INT_PROPS:
            out[:, j] = np.rint(out[:, j])
    return out


class _Objective:
    """The swappable part: score a batch of predicted post-edit vectors.

    ``value(post, sigma)`` returns one float per row, HIGHER IS BETTER, and
    ``current`` is that same quantity for the unedited molecule, so eligibility is
    always "beats the molecule you already have".
    """

    def __init__(self, score: str, keys: List[str], v: np.ndarray, lo: np.ndarray,
                 hi: np.ndarray, scale: np.ndarray, gate: str,
                 moves_dir: str = MOVES_DIR) -> None:
        self.score, self.keys, self.gate = score, keys, gate
        self.v, self.lo, self.hi, self.scale = v, lo, hi, scale
        self.discrete = score in ("count",)
        self.norm = _maxdev(keys, lo, hi, moves_dir) if score == "minmax" else None
        self.current = self.value(v[None, :], None)[0]
        # For `count`, `current` is the exact number satisfied now; `prob` is compared
        # against that same exact count, since the unedited molecule has no Δ spread.
        self.sat_now = int(np.sum((v >= lo) & (v <= hi)))

    def value(self, post: np.ndarray, sigma: Optional[np.ndarray]) -> np.ndarray:
        if self.score == "count":
            return ((post >= self.lo[None, :]) & (post <= self.hi[None, :])
                    ).sum(axis=1).astype(np.float64)
        if self.score == "minmax":
            return -(_gap(post, self.lo, self.hi) / self.norm[None, :]).sum(axis=1)
        if self.score == "worst":
            return -(_gap(post, self.lo, self.hi) / self.scale[None, :]).max(axis=1)
        if self.score == "gapstd":
            # The shipped objective's own quantity, but scored at the two ENDS of the
            # move's ±sigma interval instead of at its mean. The mean point does not
            # enter the sum: a move is judged by how far outside the box it lands when
            # its Δ comes in one std high and one std low.
            if sigma is None:
                # The unedited molecule has no Δ and so no spread — its ±sigma endpoints
                # are both itself. This is `current`, the bar every move has to clear,
                # and it is the molecule's true normalised gap.
                return -(_gap(post, self.lo, self.hi) / self.scale[None, :]).sum(axis=1)
            s = np.abs(sigma)
            g_hi = _gap(post + s, self.lo, self.hi) / self.scale[None, :]
            g_lo = _gap(post - s, self.lo, self.hi) / self.scale[None, :]
            return -(0.5 * (g_hi + g_lo)).sum(axis=1)
        if self.score == "prob":
            if sigma is None:                     # the unedited molecule: no spread
                return ((post >= self.lo[None, :]) & (post <= self.hi[None, :])
                        ).sum(axis=1).astype(np.float64)
            s = np.maximum(sigma, _SIGMA_FLOOR)
            upper = np.where(self.hi[None, :] > 1e17, 1.0,
                             ndtr((self.hi[None, :] - post) / s))
            lower = np.where(self.lo[None, :] < -1e17, 0.0,
                             ndtr((self.lo[None, :] - post) / s))
            return np.clip(upper - lower, 0.0, 1.0).sum(axis=1)
        raise ValueError(f"unknown score {self.score!r}")

    def eligible(self, val: np.ndarray, gap_red: np.ndarray) -> np.ndarray:
        """Which moves are allowed to be returned at all."""
        if self.score == "count" and self.gate == "lex":
            # More constraints satisfied, or the same number and strictly closer — the
            # faithful counterpart of the shipped tool's "must reduce the gap".
            return (val > self.current) | ((val == self.current) & (gap_red > 1e-9))
        # Everything else, including count/strict: the objective itself must improve.
        return val > self.current + (0.0 if self.discrete else 1e-9)


def suggest_edits_sat(mol_smiles: str, constraints: Dict[str, list], top_k: int = 4, *,
                      props: Optional[dict] = None, max_cut: int = 3,
                      context_aware: bool = True, return_delta: bool = True,
                      scaffold_smarts: Optional[Union[str, List[str]]] = None,
                      score: str = "count", gate: str = "strict",
                      tie_break: str = "random",
                      rng_seed: Optional[int] = None) -> List[dict]:
    """Rank mmpdb moves by *score* and return the top ``top_k`` as ``edit_fragment``-ready
    dicts.

    Signature-compatible with :func:`molkit.utils.suggest_edits.suggest_edits`, and
    returns the same keys plus ``objective`` (the score's own value for that candidate),
    ``predicted_n_satisfied`` and ``n_satisfied_now``.

    ``tie_break`` only applies to the discrete ``count`` objective, where large groups
    of moves share a score:

    ``random``  (default) draw uniformly from the tied group — the returned ``top_k``
                is then a uniform random sample of the best-scoring moves, with the
                distance to the box playing NO part in the choice.
    ``gap``     break ties by predicted normalised gap reduction, i.e. among the moves
                that satisfy the most constraints, prefer the one that also lands
                closest.

    ``rng_seed`` seeds the random tie-break so a run is reproducible.
    """
    if score not in SCORES:
        raise ValueError(f"score must be one of {SCORES}, got {score!r}")
    if gate not in GATES:
        raise ValueError(f"gate must be one of {GATES}, got {gate!r}")
    if tie_break not in TIE_BREAKS:
        raise ValueError(f"tie_break must be one of {TIE_BREAKS}, got {tie_break!r}")
    base = Chem.MolFromSmiles(mol_smiles)
    if base is None:
        raise ValueError(f"Invalid SMILES: {mol_smiles}")
    guard_smarts = ([scaffold_smarts] if isinstance(scaffold_smarts, str)
                    else list(scaffold_smarts or []))
    guard_smarts = [g for g in guard_smarts if g]
    guards: List[Chem.Mol] = []
    for gs in guard_smarts:
        g = Chem.MolFromSmarts(gs)
        if g is None:
            raise ValueError(f"Invalid scaffold_smarts: {gs}")
        guards.append(g)
    guard_key = "\x00".join(guard_smarts)
    keys = list(constraints)
    unknown = [k for k in keys if k not in _PIDX]
    if unknown:
        raise ValueError(f"unknown properties {unknown}; choose from {PROP_ORDER}")

    lo = np.array([(-1e18 if constraints[k][0] is None else constraints[k][0]) for k in keys])
    hi = np.array([(1e18 if constraints[k][1] is None else constraints[k][1]) for k in keys])
    scale = np.array([_SCALE[k] for k in keys])
    cols = [_PIDX[k] for k in keys]

    meas = dict(props) if props else _measure(mol_smiles, keys)
    missing = [k for k in keys if meas.get(k) is None]
    if missing:
        raise ValueError(f"could not measure {missing} for the input molecule "
                         f"(ADMET backend down? pass props=... to override)")
    v = np.array([float(meas[k]) for k in keys])
    cur_g = _gap(v, lo, hi)
    current_gap = float(np.sum(cur_g / scale))
    if current_gap <= 1e-9:
        return []                                          # already inside the box

    metas, D, S, ctx_index = _load_all(MOVES_DIR, max_cut)
    if not metas:
        return []

    obj = _Objective(score, keys, v, lo, hi, scale, gate)
    sat_now = obj.sat_now

    # 1. One vectorised pass over every move.
    Dq = _round_int_props(D[:, cols], keys)
    post0 = v[None, :] + Dq
    val0 = obj.value(post0, S[:, cols])
    red0 = ((cur_g[None, :] - _gap(post0, lo, hi)) / scale).sum(axis=1)
    ok0 = obj.eligible(val0, red0)

    # Ordering key. For a discrete objective the score is an integer shared by a huge
    # number of moves, so the tie-break decides most of the ranking; both variants stay
    # strictly inside ±0.5 of the integer, which keeps the objective itself primary.
    if obj.discrete:
        if tie_break == "random":
            rs = np.random.default_rng(rng_seed)
            tie = rs.random(val0.shape[0]) - 0.5
        else:
            tie = 0.5 * np.tanh(red0)
    else:
        tie = np.zeros_like(val0)
    score0 = np.where(ok0, val0 + tie, -np.inf)

    # 2. From-fragment-first walk, best fragment first — identical in shape to the
    #    shipped tool's, so the same fragments are screened the same way.
    by_from = _by_from(MOVES_DIR, max_cut, metas)
    fkeys, fbest = _best_per_from(MOVES_DIR, max_cut, score0)
    pos = np.nonzero(np.isfinite(fbest))[0]
    froms = [fkeys[i] for i in pos[np.argsort(-fbest[pos], kind="stable")]]
    in_canon = Chem.MolToSmiles(base)

    gmatches = [[frozenset(m) for m in
                 base.GetSubstructMatches(g, uniquify=True, maxMatches=_GUARD_MATCH_CAP)]
                for g in guards]

    window = max(40, 4 * top_k)
    repl_left = _REPL_BUDGET
    collected: List[dict] = []
    per_rule: dict = {}

    def _refine(i: int, anch: dict, frag) -> dict:
        """Context-refined Δ and objective value for move *i* applied at this site."""
        _mv_f, _mv_to, _sup, of, ot, sign = metas[i]
        dvec, svec = D[i], S[i]
        if context_aware and (of, ot) in ctx_index:
            core, lbl2dummy = _cut_core(base, anch, frag)
            if core is not None:
                cdvec, csvec, _crad = _context_delta(
                    core, lbl2dummy, ctx_index[(of, ot)], sign)
                if cdvec is not None:
                    dvec, svec = cdvec, csvec
        postc = v + _round_int_props(dvec[cols][None, :], keys)[0]
        val = float(obj.value(postc[None, :], svec[cols][None, :])[0])
        gap = float(np.sum(_gap(postc, lo, hi) / scale))
        nsat = int(np.sum((postc >= lo) & (postc <= hi)))
        return {"dvec": dvec, "svec": svec, "val": val, "gap": gap, "nsat": nsat,
                "tie": float(tie[i])}

    # A molecule that does not contain the guard yet needs the RESTORATION objective —
    # buying the substructure back — before any property objective is meaningful.
    absent = [g for g, ms in zip(guards, gmatches) if not ms]
    restore = bool(absent)
    if restore:
        for i, anch, frag in _restore_scan(base, absent, guard_key,
                                           metas, by_from, _TOSET[(MOVES_DIR, max_cut)],
                                           max(2 * top_k, 8), all_guards=guards):
            c = _refine(i, anch, frag)
            c.update({"mv_from": metas[i][0], "mv_to": metas[i][1], "anch": anch,
                      "verified": True})
            collected.append(c)
        froms = ()

    for mv_from in froms:
        if len(collected) >= window:
            break
        q = _query(mv_from)
        if q is None or not base.HasSubstructMatch(q[0]):
            continue
        safe, checked = _split_sites(_anchor_sites(base, mv_from), gmatches)
        if not safe and not checked:
            continue
        idxs = by_from[mv_from]
        for i in idxs[np.argsort(-score0[idxs], kind="stable")]:
            if not np.isfinite(score0[i]) or len(collected) >= window:
                break
            mv_f, mv_to = metas[i][0], metas[i][1]
            rule = (mv_f, mv_to)
            if per_rule.get(rule, 0) >= _ANCHOR_VARIANTS:
                continue
            tried = 0
            pmap = None
            for sites, presumed_ok in ((safe, True), (checked, False)):
                for anch, frag in sites:
                    if (per_rule.get(rule, 0) >= _ANCHOR_VARIANTS
                            or len(collected) >= window or tried >= _SITE_TRY):
                        break
                    tried += 1
                    if not presumed_ok:
                        if pmap is None:
                            if repl_left <= 0:
                                break
                            repl_left -= 1
                            pmap = _products_by_anchor(base, in_canon, mv_f, mv_to)
                        prod = pmap.get(tuple(sorted(anch.items())))
                        if prod is None:
                            continue
                        if guards and not all(prod.HasSubstructMatch(g) for g in guards):
                            continue
                    c = _refine(int(i), anch, frag)
                    c.update({"mv_from": mv_f, "mv_to": mv_to, "anch": anch,
                              "verified": not presumed_ok})
                    collected.append(c)
                    per_rule[rule] = per_rule.get(rule, 0) + 1

    # 3. Re-rank the window on the CONTEXT-REFINED objective and emit top_k. The tie
    #    value is carried over from the vectorised pass so a candidate keeps its place
    #    in its tied group rather than being re-drawn here.
    collected.sort(key=lambda c: -(c["val"] + (c["tie"] if obj.discrete else 0.0)))
    out: List[dict] = []
    emitted_rules: set = set()
    for allow_repeat in (False, True):
        if len(out) >= top_k:
            break
        for c in collected:
            if len(out) >= top_k:
                break
            # Re-apply the gate on the refined values (context can move a candidate out
            # of eligibility), except in restoration mode where supplying the missing
            # substructure is the objective and normally costs property score.
            if not restore:
                red = current_gap - c["gap"]
                if not obj.eligible(np.array([c["val"]]), np.array([red]))[0]:
                    break
            rule = (c["mv_from"], c["mv_to"])
            if c.get("emitted"):
                continue
            if not allow_repeat and rule in emitted_rules:
                continue
            if not c["verified"]:
                prods = replace_fragment(mol_smiles, c["mv_from"], c["mv_to"],
                                         anchors=c["anch"])
                if not prods:
                    c["emitted"] = True
                    continue
                if guards:
                    pm = Chem.MolFromSmiles(prods[0])
                    if pm is None or not all(pm.HasSubstructMatch(g) for g in guards):
                        c["emitted"] = True
                        continue
            emitted_rules.add(rule)
            c["emitted"] = True
            delta = {}
            for p, j in zip(keys, cols):
                avg = (int(round(c["dvec"][j])) if p in _INT_PROPS
                       else round(float(c["dvec"][j]), 3))
                delta[p] = {"avg": avg, "std": round(float(c["svec"][j]), 3)}
            cand = {
                "from_smiles": c["mv_from"], "to_smiles": c["mv_to"],
                "anchors": {str(k): int(val) for k, val in c["anch"].items()},
                "predicted_gap": round(c["gap"], 4),
                "objective": round(float(c["val"]), 4),
                "score": score,
                "predicted_n_satisfied": int(c["nsat"]),
                "n_satisfied_now": sat_now,
                "n_constraints": len(keys),
            }
            if return_delta:
                cand["delta"] = delta
            out.append(cand)
    return out
