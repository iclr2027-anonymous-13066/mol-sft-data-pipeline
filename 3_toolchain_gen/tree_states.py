"""One tree's nodes -> rule-selection DECISION STATES. The single implementation.

``build_dataset.py`` used to hold this and read a finished ``tree_nodes.jsonl``. At
the sizes the 2M pools imply that file is the problem, not the input: the full
scaffold + fg expansion is 7.9 TB of dump for 1.7 TB of states, and the states are
the only part anything downstream reads. So the logic lives here, where BOTH the
after-the-fact converter and the branch-search streaming writer can call it and there
is no second copy of the label rule to drift.

What a state is, and what is dropped, is unchanged — see ``build_dataset.py``'s
docstring, which remains the reference for the label decision.
"""
from __future__ import annotations

import collections
import json

META = {"index", "id", "nid", "parent", "depth", "smiles", "rule", "gap", "self_sat",
        "n_leaf", "n_sat_leaf", "sat_ratio", "sat_any", "min_depth", "n_children",
        "props", "parent_props", "pred_delta"}
GLOBAL_PREFIXES = ("st_", "set_", "h_")


def split_row(row: dict):
    """(global features, candidate features) for one node."""
    g, c = {}, {}
    for k, v in row.items():
        if k in META:
            continue
        (g if k.startswith(GLOBAL_PREFIXES) else c)[k] = v
    return g, c


def iter_instances(path):
    """Yield (index, [rows]) for each contiguous run of one instance's nodes.

    Contiguity is what makes this streamable, and it holds for both inputs: the merged
    dump is written sorted by (index, nid), and a worker shard gets one instance's
    whole tree in a single locked write.
    """
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


def load_targets(paths):
    """id -> the instance's property box, for the per-property columns built later."""
    out = {}
    for p in paths or []:
        with open(p) as fh:
            for line in fh:
                if not line.strip():
                    continue
                d = json.loads(line)
                if d.get("id") and d.get("properties"):
                    out[d["id"]] = d["properties"]
    return out


def states_from_rows(idx: int, rows: list, tag: str, targets: dict,
                     stats: collections.Counter, *, min_candidates: int = 2,
                     drop_all_zero: bool = True, max_depth=None):
    """Yield the decision-state records for ONE instance's node list."""
    by_nid = {r["nid"]: r for r in rows}
    sets = collections.defaultdict(list)
    for r in rows:
        if r["parent"] is not None:
            sets[r["parent"]].append(r)
    for pid, kids in sets.items():
        stats[f"{tag}/sets"] += 1
        if len(kids) < min_candidates:
            stats[f"{tag}/dropped_too_few"] += 1
            continue
        if drop_all_zero and max(k["sat_ratio"] for k in kids) <= 0:
            stats[f"{tag}/dropped_all_zero"] += 1
            continue
        parent = by_nid.get(pid)
        if parent is None:
            stats[f"{tag}/dropped_no_parent"] += 1
            continue
        if max_depth is not None and parent["depth"] > max_depth:
            stats[f"{tag}/dropped_deep"] += 1
            continue
        # A "global" key is only global if it really is constant across this set.
        # h_* is 96% constant, not 100% — where it differs the key is demoted into the
        # candidate features rather than silently taking the first sibling's value.
        splits = [split_row(k) for k in kids]
        glob = dict(splits[0][0])
        demoted = {key for key in glob
                   if any(g.get(key) != glob[key] for g, _ in splits[1:])}
        if demoted:
            stats[f"{tag}/demoted_global_keys"] += len(demoted)
            for key in demoted:
                glob.pop(key, None)
        cands = []
        for k, (gk, cf) in zip(kids, splits):
            cf = dict(cf)
            for key in demoted:
                cf[key] = gk.get(key)
            cands.append({"smiles": k["smiles"], "rule": k["rule"],
                          "sat_ratio": k["sat_ratio"], "sat_any": k["sat_any"],
                          "min_depth": k["min_depth"],
                          "pred_delta": k.get("pred_delta"),
                          "features": cf})
        vals = [c["sat_ratio"] for c in cands]
        stats[f"{tag}/kept"] += 1
        stats[f"{tag}/candidate_rows"] += len(cands)
        stats[f"{tag}/ranking_pairs"] += sum(1 for a in vals for b in vals if a > b)
        if max(vals) - min(vals) <= 1e-9:
            stats[f"{tag}/kept_flat_nonzero"] += 1
        yield {"source": tag, "index": idx, "id": parent.get("id"),
               "targets": targets.get(parent.get("id")),
               "depth": parent["depth"], "state_smiles": parent["smiles"],
               "state_props": parent.get("props"), "state_gap": parent["gap"],
               "n_candidates": len(cands),
               "label_spread": round(max(vals) - min(vals), 4),
               "global": glob, "candidates": cands}


def convert_file(path: str, out_fh, tag: str, targets: dict,
                 stats: collections.Counter, **kw) -> int:
    """Stream one node file into *out_fh* as states. Returns the states written."""
    n = 0
    for idx, rows in iter_instances(path):
        for rec in states_from_rows(idx, rows, tag, targets, stats, **kw):
            out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n
