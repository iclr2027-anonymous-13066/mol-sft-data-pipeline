# -*- coding: utf-8 -*-
"""Which functional group A is spending itself on, and how many the molecule has.

WHY THIS FILE EXISTS AT ALL. v27 wants a STATE observation naming a functional group,
and the rule-selection model has no state feature for one: `FeatureSpec.from_rows`
builds the global block out of the 14 properties (`st_val__`, `st_has_lo__`,
`st_has_hi__`, `st_lo_z__`, `st_hi_z__`) plus three scalars and the `h_*` history, and
nothing else. There is no `st_fg_count`. The only functional-group attention in the
network is EDIT-side, `r_dfg__<name>` -- what that candidate does to that group's count
-- and `encode` sets `rm[i, j] = True` for all 61 of them on every candidate, so the
gate is defined for every group on every round rather than only where an edit moves one.

So the group is CHOSEN by an edit-side gate and the number PRINTED is a state fact:

    gate(name)  = max over live candidates of that candidate's r_dfg__<name> gate
    count(name) = matches of the catalog SMARTS in the CURRENT molecule

The max-over-candidates aggregation is the one `render_bucket._gates` already uses for
the A-gate contrast, and it costs little here: A's rule gate is ~90% a property of the
round rather than the candidate (0.134 spread across options against 1.296 across
columns), so the per-candidate maximum IS the round's gate.

WHY A SECOND DUMP. `evidence_v10` keeps only each candidate's top-6 gated features, and
an `r_dfg__` column reaches that cut on 9.6% of rounds -- the full 61-wide vector is
needed and one forward per round gets it. `gate_vectors` renormalises within the mask
before turning alpha into a multiplier, so 1.0 means "attended as usual" and these
numbers are on the same scale as the cached ones.

    PYTHONPATH=. python 6_edit_reasoning/fg_gates.py --splits test --limit 200
"""
from __future__ import annotations

import argparse
import glob
import importlib
import json
import os
import sys
import time
import types


class _Stub(types.ModuleType):
    """Same stub as occlude_rules: `molkit/__init__` pulls an LLM client in
    transitively and the fragment catalog needs none of it. `aiohttp` joins it because
    `3_toolchain_gen.__init__` imports the HTTP tool client, and that is the only
    reason `fg_names()` returns an empty list in this env -- the checkpoint's own
    `spec.fg` is the authoritative 61 names and is used instead."""

    def __getattr__(self, k):
        if k.startswith("__") and k.endswith("__"):
            raise AttributeError(k)
        return type(k, (), {})


_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
for _m in ("anthropic", "aiohttp"):
    try:
        importlib.import_module(_m)
    except ModuleNotFoundError:
        sys.modules[_m] = _Stub(_m)

rs = importlib.import_module("5_rule_selection.reasoning")
va = importlib.import_module("5_rule_selection.variants")
rp = importlib.import_module("6_edit_reasoning.recipe")
ev_mod = importlib.import_module("6_edit_reasoning.evidence")

TENSORS = "data/analysis/rule_selection/runs/d5-50k/tensors_molbert"
CTX = "data/analysis/rule_selection/runs/d5-50k/ctx_molbert"

_PATTERNS = None


def patterns():
    """`(name, mol)` for the 61 broadest_only catalog groups, in catalog order.

    Imported from `molkit.utils.fragments` rather than through
    `3_toolchain_gen.branch_features` so the tool-client import chain stays out of
    the way; the pattern set and the `uniquify=True` count are the same ones
    `_fg_counts` and the benchmark grader use.
    """
    global _PATTERNS
    if _PATTERNS is None:
        from rdkit import Chem

        from molkit.utils.fragments import fr_catalog
        out = []
        for key, meta in fr_catalog(broadest_only=True).items():
            if not meta.get("smarts"):
                continue
            p = Chem.MolFromSmarts(meta["smarts"])
            if p is not None:
                out.append((meta.get("name") or key, p))
        _PATTERNS = out
    return _PATTERNS


def counts(smiles: str) -> dict:
    from rdkit import Chem
    m = Chem.MolFromSmiles(smiles or "")
    if m is None:
        return {}
    out = {}
    for name, patt in patterns():
        n = len(m.GetSubstructMatches(patt, uniquify=True))
        if n:
            out[name] = n
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default=None)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dst", default="fg_gates")
    ap.add_argument("--top", type=int, default=8, help="groups stored per round")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args(argv)

    work = a.work or rp.work_dir(rp.DEFAULT)
    sc = rs.Scorer(ev_mod.CKPT, TENSORS, CTX, a.device)
    rnames = sc.spec.rule_names
    # the group order comes from the CHECKPOINT's spec, not from fg_names()
    fg = [n for n in sc.spec.fg if f"r_dfg__{n}" in set(rnames)]
    print(f"# scorer ready  R={len(rnames)}  fg columns={len(fg)}  "
          f"catalog patterns={len(patterns())}", flush=True)
    os.makedirs(f"{work}/{a.dst}", exist_ok=True)
    if a.shard == 0:
        with open(f"{work}/{a.dst}/fg_names.json", "w") as fh:
            json.dump({"fg": fg, "patterns": [n for n, _ in patterns()]}, fh)

    for split in a.splits:
        shards = [sh for k, sh in
                  enumerate(sorted(glob.glob(f"{work}/records/{split}/*.jsonl")))
                  if k % a.nshards == a.shard]
        out_path = (f"{work}/{a.dst}/{split}.jsonl" if a.nshards == 1 else
                    # %03d, like base_dump: %02d cannot address a hundredth shard, and
                    # the scaffold corpus has 100. Readers glob `part*.jsonl`, so the
                    # width is free to change.
                    f"{work}/{a.dst}/{split}.part{a.shard:03d}.jsonl")
        t0, n = time.time(), 0
        with open(out_path, "w") as out:
            for sh in shards:
                if a.limit and n >= a.limit:
                    break
                rows = ev_mod._rows_of_shard(sh)
                sc.ensure_ctx_rows(rows)
                for row in rows:
                    if a.limit and n >= a.limit:
                        break
                    res = sc.run(row)
                    gv = va.gate_vectors(res, sc.spec)
                    live = [i for i, on in enumerate(res["cmask"]) if on]
                    if not live:
                        continue
                    cnt = counts(row.get("state_smiles"))
                    # one number per group: the round's gate, then the molecule's count
                    ranked = sorted(
                        ((max(gv["cand"][i].get(f"r_dfg__{nm}", -1e30) for i in live),
                          nm) for nm in fg), key=lambda t: (-t[0], t[1]))
                    # TWO lists, because the gate does not rank what is THERE.
                    # `r_dfg__x` is what an edit does to x's count, so A gates the
                    # groups the candidates are MOVING: the single highest-gated group
                    # is one the molecule actually has on 22.5% of rounds, and the whole
                    # top-8 contains no present group at all on 33.3%. A state line has
                    # to name something the molecule HAS, so `fg` carries every present
                    # group with its gate -- the renderer ranks that -- and `fg_top` is
                    # the unrestricted ranking, kept for diagnostics and in case a later
                    # arm wants to speak about an absent group.
                    out.write(json.dumps({
                        "group_id": row["group_id"], "depth": row.get("depth"),
                        "smiles": row.get("state_smiles"),
                        "fg": [{"name": nm, "gate": round(float(g), 6),
                                "count": int(cnt.get(nm, 0))}
                               for g, nm in ranked if cnt.get(nm, 0) > 0],
                        "fg_top": [{"name": nm, "gate": round(float(g), 6),
                                    "count": int(cnt.get(nm, 0))}
                                   for g, nm in ranked[:a.top]],
                        "n_present": sum(1 for v in cnt.values() if v),
                    }) + "\n")
                    n += 1
                    if n % 5000 == 0:
                        print(f"# {split} {n} rounds  {n / (time.time() - t0):.1f}/s",
                              flush=True)
        print(f"# {split}: {n} rounds -> {out_path}  "
              f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
