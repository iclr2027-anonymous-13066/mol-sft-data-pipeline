# -*- coding: utf-8 -*-
"""ONE unmasked forward per round: the per-candidate q, the rule feature matrix, the gate.

    PYTHONPATH=. python 6_edit_reasoning/base_dump.py --splits train \
        --device cpu --nshards 96 --shard 0

WHY THIS EXISTS. The nat arms need three things from the rule-selection checkpoint and
they do not cost the same:

    q per candidate       the DROP ORDER            evidence.py, one GPU pass, 60 r/s
    col_v                 the observation VALUES    occlude_rules.py, a byproduct
    gate / gate_spread    two of v36's six criteria gate_dump.py, one forward
    col_q/col_m/cell_*    the other four criteria   occlude_rules.py, ~715 forwards

The first three all fall out of ONE unmasked forward. `evidence.py` and `gate_dump.py`
and `occlude_rules.py` each run that forward -- and each re-embeds the same five
molecules with MolBERT to do it, which is what the pass actually costs. Measured on this
box, one worker at 8 threads: 11.8 rounds/s for the single forward against 0.6 for
occlude_rules' masking sweep. At 3.6M rounds that is the difference between three hours
and three days.

So this writes all three at once, in the formats the existing readers already parse:

    {work}/{dst}/{split}.part{k}.jsonl      col_idx / col_v / q_all / pick / n_cand
    {work}/{dst}/rule_names.json            the index -> name table, copied (see below)
    {work}/gates_full/{split}.part{k}.jsonl rule_gate + rule_gate_spread + state_gate
    {work}/gates_spread -> gates_full       a symlink; both readers take their own key
    {work}/{q_dst}/{split}/<shard>.jsonl    candidates[].q, the shape nat_span reads

WHAT IT DOES NOT WRITE is `col_m`, `col_q`, `cell_m`, `cell_q` -- the occlusion drops.
Four of `CRITERIA_V36`'s six are re-ranks of those, so **n1, n2 and n4 still need
`occlude_rules.py`**. n3 and n6 do not: they print the whole live pool, and
`bucket_span._live_cols` reads `col_v` alone. That is the whole reason to build this
first -- n3's corpus can be generated while the question of whether n1 beats it is still
being trained.

RULE_NAMES.JSON IS COPIED, NOT DERIVED. `labels`, `pools` and `prop_of` are properties of
the checkpoint's feature spec, not of the corpus, and they are not reproduced by
`occlude_rules.py`'s own writer -- the file in the study run was enriched by hand. Copying
it is the honest thing to do: a second hand-built copy could silently disagree, and every
arm's column pool is defined by it.
"""
from __future__ import annotations

import argparse
import glob
import importlib
import json
import os
import shutil
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

rs = importlib.import_module("5_rule_selection.reasoning")
va = importlib.import_module("5_rule_selection.variants")
rp = importlib.import_module("6_edit_reasoning.recipe")
ev_mod = importlib.import_module("6_edit_reasoning.evidence")

TENSORS = "data/analysis/rule_selection/runs/d5-50k/tensors_molbert"
CTX = "data/analysis/rule_selection/runs/d5-50k/ctx_molbert"
# the study run's table, the one every rendered corpus so far was built against
NAMES_SRC = ("data/analysis/reasoning_arms/v231318b2cfa1/"
             "occlusion_rules2/rule_names.json")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default=None)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=2,
                    help="torch threads. MEASURED: 2 is faster per CORE than 8 (0.7 vs "
                         "0.6 rounds/s per worker on occlude_rules), because the model "
                         "is small enough that thread launch dominates. Run many "
                         "workers at 2 threads, not few at 16.")
    ap.add_argument("--dst", default="base_cols")
    ap.add_argument("--gates-dst", default="gates_full")
    ap.add_argument("--q-dst", default="evidence_q")
    ap.add_argument("--names-src", default=NAMES_SRC)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args(argv)

    torch.set_num_threads(max(1, a.threads))
    work = a.work or rp.work_dir(rp.DEFAULT)
    sc = rs.Scorer(ev_mod.CKPT, TENSORS, CTX, a.device)
    rn, gn = sc.spec.rule_names, sc.spec.global_names
    print(f"# scorer ready  R={len(rn)}  G={len(gn)}  device={a.device}  "
          f"threads={a.threads}  work={work}", flush=True)

    for d in (a.dst, a.gates_dst):
        os.makedirs(f"{work}/{d}", exist_ok=True)
    if a.shard == 0:
        # Worker 0 alone writes the shared tables: several processes opening the same
        # path truncate each other.
        shutil.copyfile(a.names_src, f"{work}/{a.dst}/rule_names.json")
        with open(f"{work}/{a.gates_dst}/names.json", "w") as fh:
            json.dump({"rule_names": rn, "global_names": gn}, fh)
        # `_gates_full` reads `rule_gate` and `_gates_spread` reads `rule_gate_spread`;
        # both keys are in the same record, so one directory serves both readers and the
        # 20 GB is not written twice.
        link = f"{work}/gates_spread"
        if not os.path.exists(link):
            os.symlink(a.gates_dst, link)

    for split in a.splits:
        shards = [sh for k, sh in
                  enumerate(sorted(glob.glob(f"{work}/records/{split}/*.jsonl")))
                  if k % a.nshards == a.shard]
        os.makedirs(f"{work}/{a.q_dst}/{split}", exist_ok=True)
        col_p = (f"{work}/{a.dst}/{split}.jsonl" if a.nshards == 1 else
                 f"{work}/{a.dst}/{split}.part{a.shard:03d}.jsonl")
        gat_p = (f"{work}/{a.gates_dst}/{split}.jsonl" if a.nshards == 1 else
                 f"{work}/{a.gates_dst}/{split}.part{a.shard:03d}.jsonl")
        t0, n = time.time(), 0
        with open(col_p, "w") as fc, open(gat_p, "w") as fg:
            for sh in shards:
                if a.limit and n >= a.limit:
                    break
                rows = ev_mod._rows_of_shard(sh)
                sc.ensure_ctx_rows(rows)
                # q rides in the per-shard layout nat_span already reads, so
                # `--evidence <q_dst>` needs no change anywhere downstream.
                with open(f"{work}/{a.q_dst}/{split}/{os.path.basename(sh)}",
                          "w") as fq:
                    for row in rows:
                        if a.limit and n >= a.limit:
                            break
                        res = sc.run(row)
                        r, rm, cm = res["r_raw"], res["r_mask"], res["cmask"]
                        live = [i for i, on in enumerate(cm) if on]
                        nc = len(live)
                        if nc < 2:
                            continue
                        pick = int(row["committed"])
                        q = [float(x) for x in res["q"][:nc]]
                        cols = sorted({j for i in range(nc)
                                       for j in range(len(rn)) if rm[i, j]})
                        fc.write(json.dumps({
                            "group_id": row["group_id"], "depth": row.get("depth"),
                            "pick": pick, "n_cand": nc,
                            "col_idx": [int(j) for j in cols],
                            "col_v": [[round(float(r[i, j]), 6) if rm[i, j] else None
                                       for j in cols] for i in range(nc)],
                            "q_all": [round(x, 6) for x in q],
                        }) + "\n")
                        fq.write(json.dumps({
                            "group_id": row["group_id"], "depth": row.get("depth"),
                            "candidates": [{"index": i, "q": round(q[i], 6)}
                                           for i in range(nc)],
                        }) + "\n")
                        gv = va.gate_vectors(res, sc.spec)
                        rg = [0.0] * len(rn)
                        lo = [None] * len(rn)
                        for i in live:
                            d = gv["cand"][i]
                            for j, nm in enumerate(rn):
                                v = d.get(nm)
                                if v is None:
                                    continue
                                if v > rg[j]:
                                    rg[j] = v
                                if lo[j] is None or v < lo[j]:
                                    lo[j] = v
                        sp = [(rg[j] - lo[j]) if lo[j] is not None else 0.0
                              for j in range(len(rn))]
                        st = gv["state"]
                        fg.write(json.dumps({
                            "group_id": row["group_id"], "depth": row.get("depth"),
                            "n_cand": nc,
                            "rule_gate": [round(float(x), 5) for x in rg],
                            "rule_gate_spread": [round(float(x), 5) for x in sp],
                            "state_gate": [round(float(st.get(nm, 0.0)), 5)
                                           for nm in gn],
                        }) + "\n")
                        n += 1
                        if n % 5000 == 0:
                            print(f"# {split} {n} rounds  "
                                  f"{n/(time.time()-t0):.1f}/s", flush=True)
        print(f"# {split}: {n} rounds -> {col_p}  {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
