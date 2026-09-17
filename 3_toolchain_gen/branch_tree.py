"""The WHOLE exhaustive tree per instance, with each node's own satisfied-leaf ratio.

``branch_features.py`` labels only the four candidates at the seed, and on the
"does this branch reach the box" target the siblings almost never disagree — 400
usable sibling pairs over 832 instances. That is one layer of a tree that has
``4 + 4^2 + ... + 4^depth`` of them, and the layer where the answer is least visible:
near the root every branch still leads to a mixture of satisfied and unsatisfied
leaves, and the branches only separate further down.

So this module expands the full tree and stores, for every node:

``n_leaf`` / ``n_sat_leaf`` / ``sat_ratio``  how many of the paths through this node end
        inside the property box. A SOFT target, which is what the mixing near the root
        calls for — a binary "can it ever work" is ~1 everywhere up there and carries
        no gradient, while the ratio still separates a node with 90% good descendants
        from one with 10%.
``sat_any`` / ``min_depth``  the binary and the shallowest satisfying descendant, for
        the questions where those are the honest target.
``self_sat``  the node itself is already inside the box.

Two rules make the ratio mean what it says:

* **No early exit.** ``search_plan.exhaustive_search`` stops at the first satisfying
  molecule, which is right for a planner and wrong here: the unexplored remainder of
  the level would silently count as unsatisfied.
* **Path-local dedup only.** The planner also refuses to revisit a molecule seen on
  ANY branch, so a subtree's size would depend on the order its siblings were expanded.
  Here a molecule is only blocked from repeating along its own ancestry.
  ``--no-dedup`` drops even that: the children are exactly what ``suggest_edits``
  returned. Two rules that produce the same molecule then appear as two children with
  the same ``sat_ratio``, which is the honest weighting when the policy being estimated
  picks uniformly from the rule LIST rather than from the distinct products — and it is
  what a model sees at inference, where the tool hands back the raw list.

Analysis (``--analyze``) is conditional on the PARENT, never pooled: each comparison is
between siblings that share a state, and everything is reported per depth, so
"the root is blurry and the leaves are sharp" is a hypothesis the numbers can answer
rather than an assumption. It also reports the intraclass correlation — the share of
the variance in ``sat_ratio`` that is between parents rather than within — which bounds
how much any sibling-level feature could explain in principle.

Usage::

    python -m 3_toolchain_gen.branch_tree \\
        --input /data/.../benchmark_fg/generation_benchmark-00000.jsonl \\
        --limit 300 --top-k 4 --tree-depth 4 --num-procs 96 --gpus 0,1,2,3,4,5,6,7 \\
        --output data/analysis/branch_tree/fg
    python -m 3_toolchain_gen.branch_tree --analyze --output <same dir>
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# one instance -> one tree
# ---------------------------------------------------------------------------
async def _expand_tree(rec: dict, idx: int, cfg: dict) -> list:
    import asyncio

    from .branch_features import candidate_features
    from .builder import guard_smarts, seed_for
    from .constraints import extract_properties
    from .local_tools import local_call_async, local_call_batch
    from . import search_plan
    from .search_plan import (_all_satisfied, _apply, _canon, _guard_mols, _norm_gap,
                              measure_with_retry)

    scaffold = seed_for(rec)
    smarts = guard_smarts(rec)
    targets = extract_properties(rec)
    if not scaffold or not smarts or not targets:
        return []
    scaffold = _canon(scaffold)
    if scaffold is None:
        return []
    props_list = list(targets)
    guard = _guard_mols(smarts)
    memo: dict = {}

    def _complete(res) -> bool:
        return isinstance(res, dict) and all(res.get(p) is not None for p in props_list)

    async def measure_one(smi: str, props: list) -> dict:
        if smi in memo:
            return dict(memo[smi])
        res = await local_call_async(
            "analyze_properties", {"mol_smiles": smi, "property_names": props})
        res = res if isinstance(res, dict) else {}
        if _complete(res):
            memo[smi] = dict(res)
        return res

    async def measure_many(smis: list) -> dict:
        uniq = [s for s in dict.fromkeys(smis) if s not in memo]
        got = (await asyncio.to_thread(local_call_batch, "analyze_properties", uniq,
                                       property_names=props_list) if uniq else {})
        out = {}
        for s in dict.fromkeys(smis):
            res = (got or {}).get(s)
            if not isinstance(res, dict) or not _complete(res):
                res = memo.get(s) or await measure_with_retry(
                    measure_one, s, props_list, retries=cfg["measure_retries"])
            res = res if isinstance(res, dict) else {}
            if _complete(res):
                memo[s] = dict(res)
            out[s] = res
        return out

    seed_props = await measure_with_retry(measure_one, scaffold, props_list,
                                          retries=cfg["measure_retries"])
    # measure_with_retry gives UP after its retries and hands back the last, possibly
    # incomplete, response. A partial dict is truthy, so it used to sail through here —
    # and then suggest_edits refuses to rank ("could not measure ... ADMET backend
    # down?"), search_plan._suggest swallows that, and the instance silently produced a
    # ROOT-ONLY tree indistinguishable from "this molecule has no moves". On the first
    # depth-5 run that was 2,760 of the 5,000 scaffold instances, all of them under a
    # contended GPU. An incomplete seed is a measurement failure: say so and skip.
    if not seed_props or not _complete(seed_props):
        missing = [p for p in props_list if seed_props.get(p) is None] if seed_props else props_list
        logger.warning("instance %d (%s): seed measurement incomplete after %d retries, "
                       "missing %s — skipping (raise --measure-retries or run when the "
                       "GPUs are free)", idx, rec.get("id"), cfg["measure_retries"], missing)
        return []
    root = {"nid": 0, "parent": None, "depth": 0, "smiles": scaffold,
            "props": seed_props, "path": (scaffold,), "rule": None, "feat": {},
            "self_sat": int(_all_satisfied(seed_props, targets)),
            "gap": round(_norm_gap(seed_props, targets), 4)}
    if root["self_sat"]:
        return []                      # trivial: nothing to choose
    nodes = [root]
    frontier = [root]
    from molkit.utils.suggest_edits import _SCALE

    for depth in range(cfg["tree_depth"]):
        # 1. expand every frontier node (a satisfied node is a leaf: the search would
        #    have stopped there, so its subtree is not part of any plan).
        pending = []
        for node in frontier:
            if node["self_sat"]:
                continue
            try:
                cands = await search_plan._suggest(
                    node["smiles"], targets, node["props"], top_k=cfg["top_k"],
                    max_cut=cfg["max_cut"], context_aware=True, scaffold_smarts=smarts)
            except Exception as exc:  # noqa: BLE001 - one node must not kill the tree
                logger.warning("instance %d node %d suggest failed: %s", idx,
                               node["nid"], exc)
                cands = []
            state_cache: dict = {}
            seen: set = set()
            here = []
            for rank, cand in enumerate(cands):
                # Dedup against this node's OWN ancestry and its own siblings only.
                # With --no-dedup nothing is deduped: every candidate suggest_edits
                # returned becomes a child, including two rules that land on the same
                # molecule (measured: 12.2% of scaffold candidates). Their sat_ratio
                # comes out identical, and the parent's ratio then weights that
                # product by how often a uniform policy over the ACTUAL rule list
                # would pick it — which is what Q(s,a) is defined over.
                visited = set() if cfg.get("no_dedup") else (set(node["path"]) | seen)
                prod = _apply(node["smiles"], cand, guard, visited)
                if prod is None:
                    continue
                seen.add(prod)
                feat = candidate_features(node["smiles"], node["props"], targets, cand,
                                          rank, cfg["tree_depth"] - depth, _SCALE,
                                          state_cache, max_cut=cfg["max_cut"],
                                          guard_smarts=smarts)
                feat["_delta"] = {p: float((cand.get("delta") or {}).get(p, {})
                                           .get("avg") or 0.0) for p in props_list}
                here.append((prod, feat, cand))
            _add_sibling_features(here, node)
            _add_history_features(here, node, targets, props_list)
            for prod, feat, _c in here:
                pending.append((node, prod, feat))
        if not pending:
            break
        # 2. one batched measurement for the whole level
        got = await measure_many([p for _n, p, _f in pending])
        nxt = []
        for parent, prod, feat in pending:
            props = got.get(prod) or {}
            if not props or _norm_gap(props, targets) == float("inf"):
                continue
            child = {"nid": len(nodes), "parent": parent["nid"], "depth": depth + 1,
                     "smiles": prod, "props": props,
                     "pred_delta": feat.get("_delta"), "parent_props": parent["props"],
                     "path": parent["path"] + (prod,), "rule": feat.get("rule"),
                     "feat": feat,
                     "self_sat": int(_all_satisfied(props, targets)),
                     "gap": round(_norm_gap(props, targets), 4)}
            nodes.append(child)
            nxt.append(child)
        frontier = nxt

    # 3. propagate the satisfied-leaf counts upward. A node that is itself satisfied is
    #    a leaf and counts as one satisfied leaf; a node with no children likewise
    #    contributes itself.
    kids: dict = {}
    for n in nodes:
        if n["parent"] is not None:
            kids.setdefault(n["parent"], []).append(n["nid"])
    by_id = {n["nid"]: n for n in nodes}
    for n in sorted(nodes, key=lambda z: -z["depth"]):
        ch = kids.get(n["nid"], [])
        if n["self_sat"] or not ch:
            n["n_leaf"] = 1
            n["n_sat_leaf"] = n["self_sat"]
            n["min_depth"] = 0 if n["self_sat"] else None
        else:
            n["n_leaf"] = sum(by_id[c]["n_leaf"] for c in ch)
            n["n_sat_leaf"] = sum(by_id[c]["n_sat_leaf"] for c in ch)
            md = [by_id[c]["min_depth"] for c in ch if by_id[c]["min_depth"] is not None]
            n["min_depth"] = (1 + min(md)) if md else None
        n["sat_ratio"] = (n["n_sat_leaf"] / n["n_leaf"]) if n["n_leaf"] else 0.0
        n["sat_any"] = int(n["n_sat_leaf"] > 0)
        n["n_children"] = len(ch)

    out = []
    for n in nodes:
        row = {"index": idx, "id": rec.get("id"), "nid": n["nid"],
               "parent": n["parent"], "depth": n["depth"], "smiles": n["smiles"],
               "rule": n["rule"], "gap": n["gap"], "self_sat": n["self_sat"],
               "n_leaf": n["n_leaf"], "n_sat_leaf": n["n_sat_leaf"],
               "sat_ratio": round(n["sat_ratio"], 4), "sat_any": n["sat_any"],
               "min_depth": n["min_depth"], "n_children": n["n_children"]}
        row.update({k: v for k, v in (n["feat"] or {}).items()
                    if k not in ("rule", "_delta")})
        # The per-property scalars a downstream model wants one-per-property rather
        # than aggregated into st_worst_z / st_n_violated: the state's own measured
        # values, and the move's predicted delta (mean/std per property). Both are
        # already computed above; only the dump dropped them.
        if n["parent"] is not None:
            row["parent_props"] = n.get("parent_props")
            row["pred_delta"] = n.get("pred_delta")
        row["props"] = n.get("props")
        out.append(row)
    return out



def _add_sibling_features(here: list, node: dict) -> None:
    """Features that only exist because the OTHER candidates are on the table.

    The pairwise model differences away anything linear and shared, so a set-level
    quantity has to be non-linear in the set to survive — a margin to the runner-up
    does, a mean does not. These say how DECISIVE the list is, which is exactly what a
    model needs to know before deciding whether to trust its pick or go and measure.
    """
    if not here:
        return
    probs = [f.get("c_prob", 0.0) for _p, f, _c in here]
    n = len(probs)
    mean = sum(probs) / n
    var = sum((x - mean) ** 2 for x in probs) / n
    sd = math.sqrt(var)
    best = max(probs)
    spread = best - min(probs)
    n_best = sum(1 for x in probs if x >= best - 1e-9)
    # Fragment similarity to the other candidates: if the four are near-duplicates the
    # choice barely matters, and that is worth knowing separately from which is best.
    sims = _pairwise_sim([c.get("to_smiles", "") for _p, _f, c in here])
    for k, (_p, f, _c) in enumerate(here):
        others = probs[:k] + probs[k + 1:]
        f["set_n_cands"] = n
        f["set_prob_spread"] = round(spread, 4)
        f["c_prob_margin"] = round(probs[k] - (max(others) if others else probs[k]), 4)
        f["c_prob_z"] = round((probs[k] - mean) / sd, 4) if sd > 1e-9 else 0.0
        f["c_is_unique_best"] = int(probs[k] >= best - 1e-9 and n_best == 1)
        f["cand_similarity"] = sims[k]


def _pairwise_sim(smis: list) -> list:
    """Mean Morgan-fingerprint Tanimoto of each fragment against the other candidates."""
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    fps = []
    for smi in smis:
        m = Chem.MolFromSmiles(smi.replace("[*:1]", "[H]").replace("[*:2]", "[H]")
                               .replace("[*:3]", "[H]")) if smi else None
        fps.append(gen.GetFingerprint(m) if m is not None else None)
    out = []
    for i, fi in enumerate(fps):
        vals = [DataStructs.TanimotoSimilarity(fi, fj)
                for j, fj in enumerate(fps) if j != i and fi is not None and fj is not None]
        out.append(round(sum(vals) / len(vals), 4) if vals else 0.0)
    return out


def _add_history_features(here: list, node: dict, targets: dict,
                          props_list: list) -> None:
    """What the walk has already learned on THIS instance.

    The most useful of these is the calibration error: if the edit that produced the
    current molecule promised a Δ and the measurement disagreed, the same move set is
    likely to be off again here, and the model should discount its predictions and
    verify instead. None of it needs lookahead — it is all in the memory block the
    assistant is already reading.
    """
    from molkit.utils.suggest_edits import _SCALE
    from rdkit import Chem

    step = node.get("depth", 0)
    pred = node.get("pred_delta") or {}
    prev = node.get("parent_props") or {}
    err = 0.0
    n_err = 0
    for p in props_list:
        a, b = prev.get(p), (node.get("props") or {}).get(p)
        if a is None or b is None or p not in pred:
            continue
        sc = float(_SCALE.get(p, 1.0)) or 1.0
        err += abs((float(b) - float(a)) - pred[p]) / sc
        n_err += 1
    improved = 0
    if prev:
        from .search_plan import _norm_gap
        improved = int(_norm_gap(node["props"], targets)
                       < _norm_gap(prev, targets) - 1e-9)
    cur = Chem.MolFromSmiles(node["smiles"])
    for _p, f, c in here:
        f["h_step_index"] = step
        f["h_pred_error_last"] = round(err / n_err, 4) if n_err else 0.0
        f["h_has_history"] = int(bool(prev))
        f["h_gap_improved_last"] = improved
        f["h_same_rule_as_prev"] = int(bool(node.get("rule"))
                                       and node.get("rule") == f.get("rule"))
        # Is this fragment already present? Adding a second copy of a motif is a
        # different kind of move from introducing a new one.
        rep = 0
        to = (c.get("to_smiles") or "").replace("[*:1]", "").replace("[*:2]", "")
        if cur is not None and len(to) > 3:
            q = Chem.MolFromSmarts(to)
            if q is not None and q.GetNumAtoms() > 2:
                try:
                    rep = int(cur.HasSubstructMatch(q))
                except Exception:  # noqa: BLE001
                    rep = 0
        f["h_motif_repeat"] = rep


# ---------------------------------------------------------------------------
# worker
# ---------------------------------------------------------------------------
def _worker_main(worker_id: int, records_path: str, gpus: list, cfg: dict,
                 out_path: str, ret_q, cursor, progress=None) -> None:
    import asyncio
    import signal

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    with open(records_path) as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(v, "1")
    os.environ.setdefault("MALLOC_ARENA_MAX", "2")
    # An EMPTY --gpus means "score ADMET on the CPU", and above about a dozen workers
    # per card that is the faster setting, not a fallback. The ADMET forward pass is
    # not what costs — profiling puts it outside the top twenty entries of a
    # prediction, behind descriptor normalisation — but every worker holds its own
    # CUDA context, and at 48 workers per card the driver spends its time switching
    # between them: measured at 192 workers over 4 cards, GPU utilisation sat at
    # 0-44% while the workers themselves were SLEEPING at 18% CPU, and the whole run
    # flatlined near 64 cores no matter how many processes were added. On the CPU a
    # worker is 51 vs 69 mol/s on its own, but nothing serialises it, and the model
    # loads in 3.8 s instead of 53. Predictions are the same to the precision the
    # pipeline stores: over 1,000 molecules x 4 ADMET properties, zero values differ
    # at the 3 decimals `_safe_round` keeps (max raw difference 7.3e-7).
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[worker_id % len(gpus)]) if gpus else ""
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    from . import fast_norm, http_client, local_tools
    from .branch_features import _fg_patterns
    http_client.set_local_mode(True)
    # Before the ADMET model is touched: over half of a prediction's wall time is
    # descriptor normalisation, not inference. See fast_norm — the swap is exact.
    fast_norm.patch()
    try:
        from molkit.utils import suggest_edits as se
        metas, _D, _S, _c = se._load_all(se.MOVES_DIR, cfg["max_cut"])
        se._by_from(se.MOVES_DIR, cfg["max_cut"], metas)
    except Exception as exc:  # noqa: BLE001
        logger.warning("move-index warmup skipped: %s", exc)
    local_tools.preload(with_admet=True)
    _fg_patterns()

    def take_next() -> Optional[int]:
        with cursor.get_lock():
            i = cursor.value
            if i >= len(records):
                return None
            cursor.value = i + 1
        return i

    rotate = int(cfg.get("shard_max_bytes") or 0)
    part = [0]

    async def main() -> int:
        lock = asyncio.Lock()
        written = 0

        async def consume():
            nonlocal written
            while True:
                idx = take_next()
                if idx is None:
                    return
                try:
                    rows = await _expand_tree(records[idx], idx, cfg)
                except Exception as exc:  # noqa: BLE001
                    logger.exception("instance %d failed: %s", idx, exc)
                    rows = []
                finally:
                    if progress is not None:
                        with progress.get_lock():
                            progress.value += 1
                if not rows:
                    continue
                async with lock:
                    # One instance's whole tree lands in ONE part, contiguously: that
                    # is what lets a closed part be converted to decision states on its
                    # own, without the merged dump ever existing.
                    path = out_path if not rotate else f"{out_path[:-6]}.p{part[0]:05d}.jsonl"
                    with open(path, "a") as fh:
                        for r in rows:
                            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                    written += len(rows)
                    if rotate and os.path.getsize(path) >= rotate:
                        # Rename, then bump: the converter only ever sees a name it can
                        # take, and a part still being appended to never carries it.
                        os.rename(path, f"{out_path[:-6]}.done{part[0]:05d}.jsonl")
                        part[0] += 1

        await asyncio.gather(*(consume() for _ in range(max(1, cfg["per_proc"]))))
        return written

    n_written = asyncio.run(main())
    if rotate:
        # Seal the tail part so the converter picks it up like any other.
        tail = f"{out_path[:-6]}.p{part[0]:05d}.jsonl"
        if os.path.exists(tail):
            os.rename(tail, f"{out_path[:-6]}.done{part[0]:05d}.jsonl")
    ret_q.put((worker_id, n_written))


# ---------------------------------------------------------------------------
# streaming: nodes -> decision states, without the nodes ever landing whole
# ---------------------------------------------------------------------------
class _StateStreamer:
    """Convert sealed worker parts into decision states and delete them as they go.

    The expansion produces roughly 4.5x more bytes of tree than of training rows —
    measured on the depth-5 2M-pool probe, 2.0 MB of nodes per instance against 0.43 MB
    of states — and nothing downstream of ``build_dataset`` ever reads the tree. At
    50k instances keeping it was merely wasteful; extrapolated to the 2M scaffold + fg
    pools it is 7.9 TB against 9.0 TB free, i.e. the run cannot finish. So with
    ``--stream-states`` the dump is never assembled: workers rotate their shard every
    ``--shard-max-bytes``, and each sealed part is converted here and unlinked, which
    also retires the load-everything-and-sort merge that used to end the run (23 M row
    dicts in one list, for a file that was about to be thrown away).

    A part is only ever picked up after the worker RENAMED it to ``.doneNNNNN``, so a
    file still being appended to is never read. Instance trees are written whole under
    the worker's lock, so an instance never straddles two parts.

    Also keeps the per-depth node/outcome tally the summary printed, counted on the
    way past rather than by reading the finished dump with pandas.
    """

    def __init__(self, out_dir: Path, states_path: Path, instances: list, *,
                 tag: str, min_candidates: int = 2, drop_all_zero: bool = True):
        from . import tree_states as ts
        self._ts = ts
        self.out_dir = Path(out_dir)
        self.tag = tag
        self.kw = {"min_candidates": min_candidates, "drop_all_zero": drop_all_zero}
        self.targets = ts.load_targets(instances)
        self.fh = open(states_path, "w")
        self.stats: dict = __import__("collections").Counter()
        self.n_states = 0
        self.n_nodes = 0
        self.n_instances = 0
        self.by_depth: dict = __import__("collections").defaultdict(
            lambda: [0, 0, 0, 0.0])          # depth -> [nodes, self_sat, sat_any, ratio]

    def _tally(self, rows: list) -> None:
        for r in rows:
            d = self.by_depth[r["depth"]]
            d[0] += 1
            d[1] += int(r.get("self_sat") or 0)
            d[2] += int(r.get("sat_any") or 0)
            d[3] += float(r.get("sat_ratio") or 0.0)
        self.n_nodes += len(rows)

    def drain(self) -> int:
        """Convert every sealed part now on disk. Returns how many it consumed."""
        parts = sorted(self.out_dir.glob(".shard_*.done*.jsonl"))
        for p in parts:
            try:
                for idx, rows in self._ts.iter_instances(str(p)):
                    self.n_instances += 1
                    self._tally(rows)
                    for rec in self._ts.states_from_rows(
                            idx, rows, self.tag, self.targets, self.stats, **self.kw):
                        self.fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        self.n_states += 1
            except Exception as exc:  # noqa: BLE001 - a bad part must not kill the run
                logger.exception("state conversion failed on %s: %s", p, exc)
                continue
            p.unlink(missing_ok=True)
        return len(parts)

    def finish(self) -> None:
        """Drain, then sweep up parts no worker got to seal.

        A worker that dies mid-part leaves its ``.pNNNNN`` open and the seal never
        happens, so `drain` — which deliberately only looks at sealed names — would
        walk past finished trees and delete nothing. Called after every worker has
        exited, an unsealed part is simply a sealed one nobody renamed, and skipping it
        would be exactly the silent partial loss this file already learned about once.
        """
        self.drain()
        for p in sorted(self.out_dir.glob(".shard_*.p[0-9]*.jsonl")):
            logger.warning("recovering unsealed part %s (a worker died before sealing)",
                           p.name)
            p.rename(p.with_suffix("").with_suffix(".done_recovered.jsonl"))
        self.drain()
        self.fh.close()

    def report(self) -> None:
        print(f"\n{self.n_nodes} nodes over {self.n_instances} instances "
              f"-> {self.n_states} decision states (nodes not kept)")
        print("\nnodes and outcome by depth")
        print(f"{'depth':>6}{'nodes':>9}{'self_sat%':>11}{'sat_any%':>10}{'sat_ratio':>11}")
        for d in sorted(self.by_depth):
            n, ss, sa, sr = self.by_depth[d]
            print(f"{d:>6}{n:>9}{100 * ss / n:>10.1f}%{100 * sa / n:>9.1f}%{sr / n:>11.3f}")
        for k in sorted(self.stats):
            print(f"   {k:<34}{self.stats[k]:,}")


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------
def analyze(out_dir: Path, top_n: int = 14, summary_only: bool = False) -> None:
    import numpy as np
    import pandas as pd

    path = out_dir / "tree_nodes.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} not found — run the expansion first")
    df = pd.read_json(path, lines=True)
    ch = df[df.parent.notna()].copy()          # every node except the roots is a choice
    # nid is numbered per instance, so a bare parent id collides across instances: the
    # sibling set is (instance, parent), and grouping on parent alone silently merges
    # unrelated molecules into one "sibling set".
    ch["pid"] = ch["index"].astype(str) + ":" + ch["parent"].astype(int).astype(str)
    print(f"\n{len(df)} nodes over {df['index'].nunique()} instances "
          f"({len(ch)} parent->child edges)")
    print("\nnodes and outcome by depth")
    print(f"{'depth':>6}{'nodes':>9}{'self_sat%':>11}{'sat_any%':>10}"
          f"{'sat_ratio':>11}{'ratio sd':>10}")
    for d, g in df.groupby("depth"):
        print(f"{d:>6}{len(g):>9}{g.self_sat.mean() * 100:>10.1f}%"
              f"{g.sat_any.mean() * 100:>9.1f}%{g.sat_ratio.mean():>11.3f}"
              f"{g.sat_ratio.std():>10.3f}")

    # How much of the outcome is decided ABOVE the node? The share of variance in
    # sat_ratio that lies between parents rather than between siblings of one parent.
    print("\nvariance of sat_ratio: between parents vs within a sibling set (ICC)")
    for d, g in ch.groupby("depth"):
        if g.pid.nunique() < 5:
            continue
        gm = g.sat_ratio.mean()
        grp = g.groupby("pid").sat_ratio
        ssb = float((grp.count() * (grp.mean() - gm) ** 2).sum())
        ssw = float(((g.sat_ratio - g.pid.map(grp.mean())) ** 2).sum())
        icc = ssb / (ssb + ssw) if (ssb + ssw) > 0 else float("nan")
        print(f"   depth {d}: ICC {icc:.3f}  -> {icc * 100:.0f}% of the outcome is "
              f"already set by which parent you are under, "
              f"{100 - icc * 100:.0f}% by which sibling you pick")

    if summary_only:
        # The per-feature Spearman below is a Python-level loop over every sibling set
        # for every feature: minutes per depth on a depth-5 dump, and redundant with
        # --rank. The expansion run only needs the shape summary above.
        return
    feats = [c for c in ch.columns
             if c.startswith(("c_", "r_", "fg_n_", "site_"))
             and pd.api.types.is_numeric_dtype(ch[c]) and ch[c].nunique() > 1]

    def within_parent_rho(g, col):
        """Spearman of feature vs sat_ratio inside each sibling set, then pooled.

        Three summaries, because they disagree and the disagreement is informative: a
        sibling set holds 2-4 nodes, so rho hits +-1 on ~20% of them by chance alone,
        and the Fisher-z pooling (which is the textbook choice) weights those extremes
        heavily — on c_damage it reads -0.74 where the mean of the per-set rhos is
        -0.45. The ORDER of the features is the same either way; the magnitude is not,
        so the mean is reported as the headline and the z as a second column."""
        rs, ws = [], []
        for _, s in g.groupby("pid"):
            if len(s) < 2 or s[col].nunique() < 2 or s.sat_ratio.nunique() < 2:
                continue
            r = s[col].rank().corr(s.sat_ratio.rank())
            if r is None or math.isnan(r):
                continue
            rs.append(float(r))
            ws.append(len(s) - 1)
        if not rs:
            return float("nan"), float("nan"), float("nan"), 0
        z = float(np.average([math.atanh(max(min(v, 0.999), -0.999)) for v in rs],
                             weights=ws))
        return float(np.mean(rs)), float(np.median(rs)), math.tanh(z), len(rs)

    for d in sorted(ch.depth.unique()):
        g = ch[ch.depth == d]
        rows = []
        for c in feats:
            mean_r, med_r, z_r, n = within_parent_rho(g, c)
            if n >= 30 and not math.isnan(mean_r):
                rows.append((abs(mean_r), mean_r, med_r, z_r, n, c))
        if not rows:
            continue
        rows.sort(reverse=True)
        print(f"\n=== depth {d}: within-parent Spearman(feature, sat_ratio) "
              f"[{g.pid.nunique()} sibling sets] ===")
        print(f"   {'feature':<18}{'mean':>8}{'median':>8}{'Fisher-z':>10}{'n sets':>8}")
        for _, mean_r, med_r, z_r, n, c in rows[:top_n]:
            bar = "#" * int(abs(mean_r) * 60)
            print(f"   {c:<18}{mean_r:>+8.3f}{med_r:>+8.3f}{z_r:>+10.3f}{n:>8d}  {bar}")
    with open(out_dir / "tree_report.json", "w") as fh:
        json.dump({"n_nodes": len(df), "n_instances": int(df["index"].nunique())},
                  fh, indent=2)


# ---------------------------------------------------------------------------
# which features decide the argmax
# ---------------------------------------------------------------------------
def rank_argmax(out_dir: Path, top_n: int = 0, min_pairs: int = 300,
                max_pairs: int = 120_000, extra_diagnostics: bool = False) -> None:
    """Rank every feature by how much it decides WHICH SIBLING IS BEST.

    The unit is a sibling PAIR: at each node the agent picks one child, so what has to
    be modelled is the comparison, not the value. For siblings i, j of one parent with
    ``sat_ratio_i > sat_ratio_j`` the training row is the feature DIFFERENCE
    ``x_i - x_j`` with label 1, plus ``-(x_i - x_j)`` with label 0 so the fit is
    symmetric and the boundary passes through the origin. Logistic regression on that
    is the pairwise form of McFadden's conditional logit, and it has the property this
    question needs: a feature that is constant inside a sibling set differences to
    exactly zero, so instance difficulty cannot leak in. Every ``st_*`` column drops out
    on its own.

    Importance is measured by LOCO — leave one covariate out and REFIT. Zeroing a
    coefficient in the fitted model instead is much cheaper and was tried first; it is
    biased, because the surviving coefficients were fitted in the presence of the
    removed feature and cannot re-absorb what it carried. On depth 4 that read
    ``c_pred_gap`` at +3.48pp where the honest refit says -0.06pp: its information is
    fully covered by c_gap_reduction and c_worst_after, which come from the same gap
    arithmetic. Permutation importance sits in between (+3.39pp there) because it feeds
    the model feature combinations that never occur. Only LOCO answers "what is lost if
    this column did not exist".
    """
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold

    from .branch_features import FEATURE_DOC

    path = out_dir / "tree_nodes.jsonl"
    df = pd.read_json(path, lines=True)
    ch = df[df.parent.notna()].copy()
    ch["pid"] = ch["index"].astype(str) + ":" + ch["parent"].astype(int).astype(str)
    feats = [c for c in ch.columns
             if c.startswith(("c_", "r_", "fg_n_", "site_", "st_", "ctx_", "h_",
                              "set_", "cand_"))
             and pd.api.types.is_numeric_dtype(ch[c]) and ch[c].nunique() > 1]

    print(_LEGEND)
    for depth in sorted(ch.depth.unique()):
        g = ch[ch.depth == depth]
        g = g[g.groupby("pid").sat_ratio.transform("nunique") > 1].copy()
        if g.pid.nunique() < 20:
            continue
        D, inst = [], []
        for _pid, s in g.groupby("pid"):
            X = s[feats].fillna(0).to_numpy(float)
            y = s.sat_ratio.to_numpy()
            for i in range(len(s)):
                for j in range(len(s)):
                    if y[i] > y[j] + 1e-9:
                        D.append(X[i] - X[j])
                        inst.append(s["index"].iloc[0])
        if len(D) < min_pairs:
            continue
        D = np.asarray(D)
        inst = np.asarray(inst)
        if len(D) > max_pairs:          # bound the LOCO cost on the deep, huge levels
            keep = np.random.default_rng(0).choice(len(D), max_pairs, replace=False)
            D, inst = D[keep], inst[keep]
        sd = D.std(axis=0)
        sd[sd == 0] = 1.0
        Ds = D / sd
        X = np.vstack([Ds, -Ds])
        y = np.r_[np.ones(len(Ds)), np.zeros(len(Ds))]
        groups = np.r_[inst, inst]
        Xn = g[feats].fillna(0).to_numpy(float) / sd
        gi = g["index"].to_numpy()
        folds = list(GroupKFold(n_splits=min(5, len(np.unique(inst)))).split(X, y, groups))

        # Ties are broken at RANDOM, with the same draw for every score. A stable sort
        # would fall back on row order, which is the tool's own ranking, and the
        # integer/binary columns are tied inside most sibling sets (c_fix_worst in
        # 100% of them at depth 4) — that credited them with the tool's accuracy.
        tie = np.random.default_rng(0).random(len(g))
        mx = g.groupby("pid").sat_ratio.max()

        def top1(sc):
            pick = (g.assign(_s=sc, _t=tie)
                    .sort_values(["pid", "_s", "_t"], ascending=[True, False, True],
                                 kind="mergesort").groupby("pid").head(1))
            return float((pick.sat_ratio.to_numpy()
                          >= pick.pid.map(mx).to_numpy() - 1e-9).mean())

        def oof(cols):
            idx = [feats.index(c) for c in cols]
            sc = np.zeros(len(g))
            for tr, te in folds:
                m = LogisticRegression(max_iter=3000, C=0.5,
                                       fit_intercept=False).fit(X[tr][:, idx], y[tr])
                hold = np.isin(gi, np.unique(groups[te]))
                sc[hold] = Xn[hold][:, idx] @ m.coef_[0]
            return top1(sc), sc

        base, _sc = oof(feats)
        rnd = float((g.sat_ratio >= g.pid.map(mx) - 1e-9)
                    .groupby(g.pid).mean().mean())
        full = LogisticRegression(max_iter=3000, C=0.5,
                                  fit_intercept=False).fit(X, y)
        rows = []
        for k, c in enumerate(feats):
            loco = base - oof([f for f in feats if f != c])[0]
            solo = top1(np.sign(full.coef_[0][k]) * Xn[:, k])
            varies = float(g.groupby("pid")[c].nunique().gt(1).mean())
            rows.append((loco, full.coef_[0][k], solo, varies, c))
        rows.sort(key=lambda t: -t[0])
        print(f"\n{'=' * 108}")
        print(f"depth {depth} | {g.pid.nunique():,} sibling sets | {len(Ds):,} ordered "
              f"pairs | full ranker {base * 100:.1f}% vs random {rnd * 100:.1f}%")
        print("=" * 108)
        print(f"{'feature':<19}{'LOCO':>9}{'solo':>8}{'varies':>8}{'coef':>8}{'tier':>6}"
              f"  설명")
        for loco, co, solo, varies, c in rows:
            if top_n and len([r for r in rows if r[0] > loco]) >= top_n:
                break
            tier, doc = FEATURE_DOC.get(c, (2, ""))
            print(f"{c:<19}{loco * 100:>+8.2f}p{solo * 100:>7.1f}%{varies * 100:>7.0f}%"
                  f"{co:>+8.2f}{tier:>6}  {doc}")


_LEGEND = """
지표 설명
  LOCO    이 feature를 빼고 랭커를 **다시 학습**했을 때 top-1이 떨어지는 폭(pp).
          그 feature의 고유 기여 — 다른 feature로 대체 불가능한 몫만 남는다.
          음수면 빼는 편이 나았다는 뜻(과적합/잡음).
  solo    이 feature 하나만으로 형제 중 argmax를 고를 때의 정확도. 동점은 무작위로 깬다.
  varies  형제 집합 중 이 feature 값이 갈리는 비율. 낮으면 대부분의 결정에서 침묵한다.
  coef    표준화된 pairwise 로지스틱 가중치(전체 데이터 적합). 부호가 방향.
  tier    1 = 프롬프트에 이미 있음 / 2 = SMILES에서 계산 필요 / 3 = 도구가 노출 안 함
          0 = 그 시점 정보가 아님(제외 대상)
  random  형제 중 무작위로 고를 때의 top-1. 모든 값은 이것과 비교해야 한다.
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="data/training_data/instances/"
                                       "benchmark_fg/generation_benchmark-00000.jsonl")
    ap.add_argument("--output", default="data/analysis/branch_tree")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--scan-limit", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--tree-depth", type=int, default=4,
                    help="levels expanded below the seed. Nodes = sum(top_k**d) and "
                         "the suggest_edits calls are the cost: 21 per instance at "
                         "depth 3, 85 at 4, 341 at 5.")
    ap.add_argument("--max-cut", type=int, default=3)
    ap.add_argument("--measure-retries", type=int, default=8)
    ap.add_argument("--no-dedup", action="store_true",
                    help="keep every candidate suggest_edits returned, even when two "
                         "rules produce the same molecule or a product repeats an "
                         "ancestor. Branching factor becomes exactly top_k; duplicate "
                         "siblings carry identical sat_ratio by construction.")
    ap.add_argument("--num-procs", type=int, default=96)
    ap.add_argument("--per-proc", type=int, default=2)
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7",
                    help="cards to spread the ADMET model over, round-robin by worker. "
                         "EMPTY ('') scores ADMET on the CPU instead, which is what you "
                         "want past ~12 workers per card — see _worker_main.")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--stream-states", default=None, metavar="STATES.JSONL",
                    help="write decision states directly and never keep the nodes. "
                         "The tree is ~4.5x the bytes of the states it produces and "
                         "nothing downstream reads it, so at 2M-instance scale this "
                         "is the difference between 7.9 TB and 1.7 TB. Workers rotate "
                         "their shard every --shard-max-bytes and each sealed part is "
                         "converted and deleted while the run continues, so peak disk "
                         "is one rotation per worker, not the whole expansion.")
    ap.add_argument("--shard-max-bytes", type=int, default=256 * 1024 * 1024,
                    help="rotate a worker's shard once it passes this (--stream-states "
                         "only). Peak node bytes on disk is about num_procs x this.")
    ap.add_argument("--instances", action="append", default=None,
                    help="instance JSONL(s) to join the property box from, for "
                         "--stream-states. Defaults to --input.")
    ap.add_argument("--tag", default=None,
                    help="the `source` field on streamed states (default: output dir "
                         "name, which is how build_dataset labelled scaffold vs fg).")
    ap.add_argument("--min-candidates", type=int, default=2)
    ap.add_argument("--keep-all-zero", action="store_true",
                    help="keep sibling sets where every candidate has sat_ratio 0 "
                         "(dropped by default; see build_dataset.py).")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--rank", action="store_true",
                    help="rank features by what decides the argmax (pairwise "
                         "conditional logit), per depth.")
    args = ap.parse_args(argv)

    out_dir = Path(args.output)
    if args.rank:
        rank_argmax(out_dir)
        return
    if args.analyze:
        analyze(out_dir)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    dump = out_dir / "tree_nodes.jsonl"
    states_path = Path(args.stream_states) if args.stream_states else None
    if states_path is not None:
        states_path.parent.mkdir(parents=True, exist_ok=True)
        if states_path.exists() and not args.overwrite:
            raise SystemExit(f"{states_path} exists; pass --overwrite to replace it.")
    if dump.exists():
        if not args.overwrite:
            raise SystemExit(f"{dump} exists; pass --overwrite to replace it.")
        dump.unlink()

    from .compare_strategies import sample_instances
    records = sample_instances(args.input, args.limit, args.seed, args.scan_limit)
    if not records:
        raise SystemExit("no usable instances found")
    cfg = {"top_k": args.top_k, "tree_depth": args.tree_depth, "max_cut": args.max_cut,
           "measure_retries": args.measure_retries, "per_proc": args.per_proc,
           "no_dedup": bool(args.no_dedup),
           "shard_max_bytes": args.shard_max_bytes if states_path is not None else 0}
    logger.info("expanding %d instances to depth %d (top_k=%d)", len(records),
                args.tree_depth, args.top_k)

    num_procs = max(1, min(args.num_procs, len(records)))
    ctx = mp.get_context("spawn")
    ret_q, progress, cursor = ctx.Queue(), ctx.Value("L", 0), ctx.Value("l", 0)
    shards = [out_dir / f".shard_{w:03d}.jsonl" for w in range(num_procs)]
    for s in shards:
        s.unlink(missing_ok=True)
    records_path = out_dir / ".records.jsonl"
    with open(records_path, "w") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    gpus = [int(g) for g in args.gpus.split(",") if g.strip() != ""]

    streamer = None
    if states_path is not None:
        streamer = _StateStreamer(out_dir, states_path, args.instances or [args.input],
                                  tag=args.tag or out_dir.name,
                                  min_candidates=args.min_candidates,
                                  drop_all_zero=not args.keep_all_zero)

    procs = []
    t0 = time.perf_counter()
    try:
        for w in range(num_procs):
            p = ctx.Process(target=_worker_main,
                            args=(w, str(records_path), gpus, cfg, str(shards[w]),
                                  ret_q, cursor, progress), daemon=False)
            p.start()
            procs.append(p)
        try:
            from tqdm import tqdm
            bar = tqdm(total=len(records), desc="instances", unit="inst",
                       dynamic_ncols=True)
        except Exception:  # pragma: no cover
            bar = None
        import queue as _queue
        done, last = [], 0
        while len(done) < num_procs:
            try:
                while True:
                    done.append(ret_q.get_nowait())
            except _queue.Empty:
                pass
            n = min(int(progress.value), len(records))
            if bar is not None and n != last:
                bar.n = last = n
                bar.refresh()
            if streamer is not None:
                # Convert while the workers are still running, so peak disk is one
                # rotation's worth of nodes rather than the whole expansion.
                got = streamer.drain()
                if got and bar is not None:
                    bar.set_postfix_str(f"states {streamer.n_states:,}", refresh=False)
            if len(done) < num_procs:
                if not any(p.is_alive() for p in procs):
                    break
                time.sleep(0.5)
        if bar is not None:
            bar.close()
        for p in procs:
            p.join()
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join(timeout=5)

    records_path.unlink(missing_ok=True)

    if streamer is not None:
        streamer.finish()
        print(f"\n{streamer.n_states} decision states -> {states_path}  "
              f"({time.perf_counter() - t0:.0f}s)")
        streamer.report()
        return

    # Merge the shards WITHOUT holding them. The old code built one list of every row
    # and sorted it — 23 M dicts on a depth-5 half, so the run died of memory long
    # before the sort mattered. The file it produced is reproduced exactly here: workers
    # draw instances from a shared cursor, so each shard's blocks are already in
    # increasing index order, and a k-way merge over one block per shard restores the
    # (index, nid) order the rest of the tooling documents while holding only k blocks.
    n_rows = 0
    streams = [(_iter_shard_blocks(s), s) for s in shards if s.exists()]
    heads = []
    for it, s in streams:
        nxt = next(it, None)
        if nxt is not None:
            heads.append([nxt[0], nxt[1], it])
    with open(dump, "w") as out:
        while heads:
            k = min(range(len(heads)), key=lambda i: heads[i][0])
            _idx, rows, it = heads[k]
            rows.sort(key=lambda r: r["nid"])
            for r in rows:
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
                n_rows += 1
            nxt = next(it, None)
            if nxt is None:
                heads.pop(k)
            else:
                heads[k][0], heads[k][1] = nxt
    for _it, s in streams:
        s.unlink(missing_ok=True)
    print(f"\n{n_rows} nodes -> {dump}  ({time.perf_counter() - t0:.0f}s)")
    analyze(out_dir, summary_only=True)


def _iter_shard_blocks(path: Path):
    """Yield (index, rows) for each contiguous instance block in a worker shard."""
    cur, buf = None, []
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if cur is not None and r["index"] != cur:
                yield cur, buf
                buf = []
            cur = r["index"]
            buf.append(r)
    if buf:
        yield cur, buf


if __name__ == "__main__":
    main()
