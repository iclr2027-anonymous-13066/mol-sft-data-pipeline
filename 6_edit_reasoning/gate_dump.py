# -*- coding: utf-8 -*-
"""A's WHOLE gate vector, because the evidence dump only kept the top of it.

`evidence_v10` stores each candidate's top-6 gated features plus `table4` -- about 14.6
of the 167 rule columns per round. That is enough while a block draws from all of P0,
where A gates a varying column on 99.6% of rounds, and NOT enough as soon as the pool is
split: over the three survivors A gates a varying PROPERTY column on 97.7% of rounds but
a varying STRUCTURAL one on only 53.8%, and the 46.2% shortfall is the dump's truncation
rather than anything about A. With the full vector every one of the 91 structural columns
has a gate and the shortfall goes away.

One forward per round, no masking, so this is ~500x cheaper than occlude_rules and runs
on CPU at ~210 rounds/s per worker -- the whole corpus in under a minute across 16
workers, without touching a GPU someone else is training on.

WHAT IS STORED. The round-level gate, `max` over the live candidates, which is what
`render_bucket._gates` already computes from the truncated dump and what
`choose_gate_axis` consumes: A's rule gate is ~90% a property of the round rather than
the candidate (0.134 spread across options against 1.296 across columns), so the maximum
IS the round's gate. The 88 state gates ride along because they cost nothing and the
state-observation designs need them.

`gate_vectors` renormalises within the mask before turning alpha into a multiplier, so
1.0 means "attended as usual" and these numbers are on the same scale as the cached ones.
`--verify` checks that against the old dump instead of trusting it.

    PYTHONPATH=. python 6_edit_reasoning/gate_dump.py --splits test --limit 500 --verify
"""
from __future__ import annotations

import argparse
import base64 as _b64
import glob
import importlib
import json
import os
import sys
import time
import types


class _Stub(types.ModuleType):
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

import numpy as np                                             # noqa: E402
import torch                                                   # noqa: E402

rs = importlib.import_module("5_rule_selection.reasoning")
va = importlib.import_module("5_rule_selection.variants")
rp = importlib.import_module("6_edit_reasoning.recipe")
ev_mod = importlib.import_module("6_edit_reasoning.evidence")

TENSORS = "data/analysis/rule_selection/runs/d5-50k/tensors_molbert"
CTX = "data/analysis/rule_selection/runs/d5-50k/ctx_molbert"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default=None)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dst", default="gates_full")
    ap.add_argument("--threads", type=int, default=2,
                    help="torch threads per worker. 2 beats 1 and does not beat 4: the "
                         "model is tiny and the shard loop is the wall.")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--embed", action="store_true",
                    help="also store the pre-MLP representations `u_hat` and `x`")
    ap.add_argument("--per-cand", action="store_true",
                    help="also store the full [candidate][rule] gate matrix as "
                         "`rule_gate_cand`. ~4x the file, no extra compute.")
    ap.add_argument("--verify", action="store_true",
                    help="cross-check against evidence_v10's cached gates")
    # ── the main-corpus path ─────────────────────────────────────────────────
    ap.add_argument("--records-glob", default="",
                    help="read record shards from HERE instead of <work>/records/"
                         "<split>. `rows_from_sft_record` is what builds a row, and it "
                         "reads the SFT record itself, so a training corpus feeds this "
                         "directly -- no rounds/ extraction in between.")
    ap.add_argument("--uhat-only", action="store_true",
                    help="write only `u_hat`, not the gate vectors. The 167 rule and 88 "
                         "state gates are ~255 numbers a round and the signature study "
                         "needs none of them: at the main corpus's 5.5M rounds that is "
                         "the difference between hundreds of GB and 7.")
    ap.add_argument("--ctx-extra", default="",
                    help="directory of persisted MolBERT rows (or $RS_CTX_EXTRA). "
                         "Without it every run re-embeds what the last one computed.")
    ap.add_argument("--overwrite", action="store_true",
                    help="redo shards whose output already exists. Off by default, so "
                         "a killed run resumes at shard granularity.")
    a = ap.parse_args(argv)

    torch.set_num_threads(max(1, a.threads))
    work = a.work or rp.work_dir(rp.DEFAULT)
    sc = rs.Scorer(ev_mod.CKPT, TENSORS, CTX, a.device,
                   ctx_extra=a.ctx_extra)
    rn, gn = sc.spec.rule_names, sc.spec.global_names
    print(f"# scorer ready  R={len(rn)}  G={len(gn)}  device={a.device}", flush=True)
    os.makedirs(f"{work}/{a.dst}", exist_ok=True)
    if a.shard == 0:
        with open(f"{work}/{a.dst}/names.json", "w") as fh:
            json.dump({"rule_names": rn, "global_names": gn}, fh)

    old = {}
    if a.verify:
        for p in sorted(glob.glob(f"{work}/evidence_v10/{a.splits[0]}/*.jsonl")):
            for line in open(p):
                r = json.loads(line)
                g = {}
                for c in (r.get("candidates") or []):
                    for x in (c.get("features") or []):
                        g[x["feature"]] = max(g.get(x["feature"], -1e30), x["gate"])
                old[(r["group_id"], r["depth"])] = g
        print(f"# verify: {len(old):,} cached rounds loaded", flush=True)

    worst = 0.0
    ncmp = 0
    # ONE OUTPUT FILE PER INPUT SHARD when a glob is given. The corpus run is hours
    # long on hardware someone else also wants; a single append-only file means a kill
    # costs the whole pass, and a per-shard file costs one shard. An existing output is
    # skipped, so restarting IS the resume.
    splits = a.splits if not a.records_glob else ["_"]
    for split in splits:
        if a.records_glob:
            shards = [sh for k, sh in enumerate(sorted(glob.glob(a.records_glob)))
                      if k % a.nshards == a.shard]
        else:
            shards = [sh for k, sh in
                      enumerate(sorted(glob.glob(f"{work}/records/{split}/*.jsonl")))
                      if k % a.nshards == a.shard]
        out_path = (f"{work}/{a.dst}/{split}.jsonl" if a.nshards == 1 else
                    f"{work}/{a.dst}/{split}.part{a.shard:02d}.jsonl")
        t0, n = time.time(), 0
        if a.records_glob:
            odir = a.dst if os.path.isabs(a.dst) else f"{work}/{a.dst}"
            os.makedirs(odir, exist_ok=True)
            # THE OUTPUT NAME MUST IDENTIFY THE SOURCE, NOT JUST ITS BASENAME. The
            # corpus keeps four directories that each number their shards from zero, so
            # 124 sources carry only 96 distinct basenames and 52 of them collide. Two
            # workers then opened the SAME `.part`, interleaved their lines into it, and
            # whichever renamed second died on a file the first had already moved --
            # six workers lost, and every colliding output either overwritten or mixed.
            # Prefixing the parent directory makes the name unique and readable.
            todo = []
            for sh in shards:
                op = os.path.join(odir, os.path.basename(os.path.dirname(sh))
                                  + "__" + os.path.basename(sh))
                if os.path.exists(op) and not a.overwrite:
                    continue
                todo.append((sh, op))
            print(f"# {len(todo)} of {len(shards)} shards to do "
                  f"({len(shards)-len(todo)} already written)", flush=True)
            for si, (sh, op) in enumerate(todo, 1):
                rows = ev_mod._rows_of_shard(sh)
                sc.ensure_ctx_rows(rows)
                m = 0
                with open(op + ".part", "w") as out:
                    for row in rows:
                        res = sc.run(row)
                        live = [i for i, on in enumerate(res["cmask"]) if on]
                        if not live:
                            continue
                        rec = {"group_id": row["group_id"], "depth": row.get("depth"),
                               "n_cand": len(live),
                               "u_hat": _b64.b64encode(np.ascontiguousarray(
                                   res["u_hat"][live], np.float16).tobytes()).decode(),
                               "embed_dtype": "float16",
                               "embed_shape": {"u_hat": list(np.shape(res["u_hat"][live]))}}
                        out.write(json.dumps(rec) + "\n")
                        m += 1
                os.replace(op + ".part", op)
                n += m
                el = time.time() - t0
                print(f"# {si}/{len(todo)}  {os.path.basename(sh)}  {m} rounds  "
                      f"{n:,} total  {n/max(el,1e-9):.0f}/s  "
                      f"eta {el/si*(len(todo)-si)/60:.0f}m", flush=True)
            print(f"# done: {n:,} rounds -> {odir}  {time.time()-t0:.0f}s", flush=True)
            continue
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
                    # round-level gate = max over live candidates, the aggregation
                    # render_bucket._gates already applies to the truncated dump
                    rg = [0.0] * len(rn)
                    # ... and the SPREAD of the same gate across the live candidates.
                    # `rule_gate` answers "where is A looking this round"; the spread
                    # answers "where does A look DIFFERENTLY depending on which edit it
                    # is scoring", which is a different question and the only one of the
                    # two that is about the candidates at all. It is small -- the gate
                    # is ~90% a property of the round (0.134 across options against
                    # 1.296 across columns) -- but small is not zero, and the level has
                    # already been measured to carry no axis-selection signal (+0.038
                    # on-path lift against a random column's +0.050).
                    #
                    # Stored as max-min rather than the whole 4 x 167 matrix: that is
                    # what both criteria need, and the matrix would be ~700 MB against
                    # 180 MB for this.
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
                    if a.verify:
                        for nm, v in (old.get((row["group_id"], row.get("depth")))
                                      or {}).items():
                            if nm in sc.spec.rule_names:
                                worst = max(worst, abs(rg[rn.index(nm)] - v))
                                ncmp += 1
                    rec = {
                        "group_id": row["group_id"], "depth": row.get("depth"),
                        "n_cand": len(live),
                        "rule_gate": [round(float(x), 5) for x in rg],
                        "rule_gate_spread": [round(float(x), 5) for x in sp],
                        "state_gate": [round(float(st.get(nm, 0.0)), 5) for nm in gn],
                    }
                    if a.embed:
                        # THE PRE-MLP REPRESENTATIONS. The scoring MLP reads
                        # `x = [c_proj | u_g | u_hat]`, so anything `q` knows about a
                        # candidate is in there -- but only `u_hat` VARIES across the
                        # candidates; the other two thirds are the context projection
                        # and the pooled state, identical for every edit in the round.
                        # So `x` is stored as its three parts, not as N copies of two
                        # round-constant vectors: 4x128 + 2x128 instead of 4x384, and
                        # the consumer rebuilds x by concatenation.
                        #
                        # ROUNDING IN NUMPY, NOT IN PYTHON. `[round(float(v), 4) for v
                        # in row]` is 2,048 interpreter-level calls a round and was the
                        # whole cost of `--embed`; `np.round(...).tolist()` is one.
                        # FLOAT16, BASE64. `tolist()` on a rounded float32 goes through
                        # float64, so `json` writes ~18 characters for a number that
                        # carries 4 -- 20 KB a round and most of the wall clock. These
                        # are LayerNorm outputs in [-8, 8]; fp16 holds them to ~3e-3,
                        # far finer than anything downstream resolves. 10x smaller and
                        # the encode is one memcpy.
                        for k_, v_ in (("u_hat", res["u_hat"][live]),
                                       ("u_g", res["u_g"]), ("c_proj", res["c_proj"])):
                            rec[k_] = _b64.b64encode(
                                np.ascontiguousarray(v_, np.float16).tobytes()).decode()
                        rec["embed_dtype"] = "float16"
                        rec["embed_shape"] = {"u_hat": list(np.shape(res["u_hat"][live])),
                                              "u_g": list(np.shape(res["u_g"])),
                                              "c_proj": list(np.shape(res["c_proj"]))}
                    if a.per_cand:
                        # THE WHOLE [candidate][rule] MATRIX, which `rg` and `sp` are the
                        # max and the range of. It is the one shape a criterion can use to
                        # ask "which observation is A looking at FOR THIS EDIT", and no
                        # artefact on disk had it: `gates_full` keeps the round-level max
                        # and `evidence_ac` keeps each candidate's top-6, so the full pool
                        # and the per-candidate axis were never available together.
                        #
                        # IT IS BIGGER AND STILL CHEAP. ~700 MB against 180 MB for the
                        # folded form, and no extra compute at all -- `gv["cand"]` is
                        # already built above and is being thrown away. A missing entry
                        # is a column A did not gate for that candidate; 0.0 is the
                        # renormalised "not attended", which is what `gate_vectors`
                        # means by absence.
                        rec["rule_gate_cand"] = [
                            [round(float(gv["cand"][i].get(nm, 0.0)), 5) for nm in rn]
                            for i in live]
                    out.write(json.dumps(rec) + "\n")
                    n += 1
                    if n % 5000 == 0:
                        print(f"# {split} {n} rounds  {n/(time.time()-t0):.0f}/s",
                              flush=True)
        print(f"# {split}: {n} rounds -> {out_path}  {time.time()-t0:.0f}s", flush=True)
    if a.verify:
        print(f"# verify: {ncmp:,} cached gate values compared, "
              f"worst absolute difference {worst:.2e}", flush=True)


if __name__ == "__main__":
    main()
