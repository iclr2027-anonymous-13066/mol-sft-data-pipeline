# -*- coding: utf-8 -*-
"""One GPU pass: A and A_c for every prepared round, cached to disk.

The `model` arm's prompt is a rendering of the trained rule-selection checkpoint's
attention. The attention does not change when the prompt wording does, so it is
extracted once and stored. After this pass, iterating on the `model` prompt is pure
CPU string formatting plus LLM calls — no MolBERT, no checkpoint, no GPU.

Cost, measured on this node (one B200):

    MolBERT context embedding   302 mol/s warm  ->  5 molecules per round
                                                    (state + 4 products)
    RuleSelector forward        228 rounds/s    ->  not the wall
    net                         ~60 rounds/s    ->  100k rounds in ~28 min

The 28 mol/s figure from an earlier probe was a cold start: it counted the checkpoint
and MolBERT load. Warm, embedding is not the bottleneck, which is why this pass exists
for reuse across prompt versions rather than for speed.

None of this corpus's molecules are in the training context table (measured: 0 of 256),
so every one is embedded here with the SAME encoder the table was built with — there is
no representation mismatch, only work.

What is stored per round is the rendered evidence (`reasoning.evidence`), not the raw
matrices: the state lines and contrast lines with their gate multipliers. It is ~30x
smaller than the full A_g/A_r/A_c, and it stores MORE lines than any prompt prints (12
state, 12 contrast) so the line counts stay a render-time knob — that is what makes
prompt iteration free after this pass.

Usage::

    PYTHONPATH=. CUDA_VISIBLE_DEVICES=4 python 6_edit_reasoning/evidence.py \
        --splits train val --workers 8
"""
from __future__ import annotations

import argparse
import glob
import importlib
import json
import os
import sys
import time

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

rp = importlib.import_module("6_edit_reasoning.recipe")
rs = importlib.import_module("5_rule_selection.reasoning")
ts = importlib.import_module("5_rule_selection.toolchain_states")

CKPT = ("data/analysis/rule_selection/runs/ablation_24/"
        "molbert-cand-norank-prod/best.pt")
TENSORS = "data/analysis/rule_selection/runs/d5-50k/tensors_molbert"
CTX = "data/analysis/rule_selection/runs/d5-50k/ctx_molbert"


def _rows_of_shard(path: str) -> list:
    """`rows_from_sft_record` output for one prepared record shard.

    The RuleSelector needs its own feature layout (`row['global']`,
    `row['candidates'][i]['features']`) plus each candidate's post-edit molecule, and
    `toolchain_states` is the code that builds both. Reading the permuted RECORD rather
    than the round table keeps one source of truth for the candidate order.
    """
    out = []
    for rec in ts.iter_records([path], limit=0):
        gid = (rec.get("metadata") or {}).get("group_id") or ""
        for row in ts.rows_from_sft_record(rec):
            if row.get("committed", -1) < 0:
                continue
            row["group_id"] = gid
            out.append(row)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default=None, help="default: recipe.work_dir(DEFAULT)")
    ap.add_argument("--splits", nargs="+", default=["train", "val"],
                    help="test needs no evidence: the model writes its own span there")
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--tensors", default=TENSORS)
    ap.add_argument("--ctx-dir", default=CTX)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=8,
                    help="CPU processes parsing record shards ahead of the GPU")
    ap.add_argument("--batch-shards", type=int, default=4,
                    help="record shards embedded per MolBERT call — bigger is a longer "
                         "warm batch and fewer fixed costs")
    ap.add_argument("--limit-shards", type=int, default=0)
    # SPLIT THE SHARD LIST ACROSS PROCESSES. One GPU per stride slot, so eight
    # processes cover the split once with no overlap and no race on the output
    # files (each shard is written by exactly one worker).
    ap.add_argument("--shard-mod", default="",
                    help="i/N -- take shards where index %% N == i")
    ap.add_argument("--overwrite", action="store_true")
    # THE CAP IS THE EXPERIMENT NOW. evidence_ac was written at 12/6/12, so every
    # measurement over it -- A's gate as a selection criterion, the candidate pools,
    # the flat-rate census -- was conditioned on a shortlist A had already drawn.
    # Raising these to the full vocabulary (81 state, 167 rule) is what makes
    # "is A's own shortlist better than a random one" answerable at all.
    ap.add_argument("--top-state", type=int, default=12)
    ap.add_argument("--top-rule", type=int, default=6)
    ap.add_argument("--top-contrast", type=int, default=12)
    ap.add_argument("--gate-floor", type=float, default=None,
                    help="override evidence()'s AVERAGE_GATE cut; 0 keeps everything")
    ap.add_argument("--dst-name", default="evidence",
                    help="subdirectory of --work to write into. A richer dump goes to a "
                         "NEW name so a generate job reading `evidence/` keeps working.")
    args = ap.parse_args(argv)

    work = args.work or rp.work_dir(rp.DEFAULT)
    t0 = time.time()
    sc = rs.Scorer(args.ckpt, args.tensors, args.ctx_dir, args.device)
    print(f"# scorer loaded in {time.time()-t0:.1f}s  encoder={sc.encoder}  "
          f"ctx={sc.ctx.shape}  prod_dims={sc.prod_dims}", flush=True)

    import concurrent.futures as cf
    for split in args.splits:
        src = sorted(glob.glob(f"{work}/records/{split}/*.jsonl"))
        if args.limit_shards:
            src = src[:args.limit_shards]
        if args.shard_mod:
            i_s, n_s = (int(x) for x in args.shard_mod.split("/"))
            src = [p for k, p in enumerate(src) if k % n_s == i_s]
        dst = f"{work}/{args.dst_name}/{split}"
        os.makedirs(dst, exist_ok=True)
        todo = [p for p in src
                if args.overwrite or not os.path.exists(
                    os.path.join(dst, os.path.basename(p)))]
        print(f"\n# {split}: {len(todo)} of {len(src)} shards to do", flush=True)
        n_tot = 0
        t_split = time.time()
        with cf.ProcessPoolExecutor(max_workers=args.workers) as ex:
            for i in range(0, len(todo), args.batch_shards):
                grp = todo[i:i + args.batch_shards]
                rows_by = list(ex.map(_rows_of_shard, grp))
                flat = [r for rr in rows_by for r in rr]
                if not flat:
                    for p in grp:
                        open(os.path.join(dst, os.path.basename(p)), "w").close()
                    continue
                tE = time.time()
                n_new = sc.ensure_ctx_rows(flat)
                tE = time.time() - tE
                tS = time.time()
                for p, rows in zip(grp, rows_by):
                    with open(os.path.join(dst, os.path.basename(p)), "w") as fh:
                        for row in rows:
                            res = sc.run(row)
                            # MORE lines than any prompt prints. `evidence`'s own
                            # defaults are 6 state / 7 contrast, and `arms.ev_model`
                            # asks for 5 / 9 — so the default silently capped the
                            # contrast block at 7 and the measured-best layout was
                            # unreachable. Storing 12/6 means the state and contrast
                            # counts stay a render-time knob, which is the whole point
                            # of caching this pass.
                            ev = rs.evidence(row, res, sc.spec,
                                             top_state=args.top_state,
                                             top_rule=args.top_rule,
                                             top_contrast=args.top_contrast,
                                             gate_floor=args.gate_floor,
                                             pick_override=row["committed"])
                            # A AND A_c, IN FULL. The first version of this pass kept
                            # only `state`, `contrast`, q and score, and `arms.ev_model`
                            # could therefore do nothing with A or A_c except sort the
                            # pick-vs-rival contrast by gate. Three things it threw away:
                            #
                            #   A_c_row      the candidate-attention row of the pick. The
                            #                docstring in reasoning.evidence records A_c
                            #                as flat at 0.24 vs a uniform 0.25 on ITS
                            #                corpus; on this one `rival_rule` says
                            #                "candidate-attention" on 96.9% of rounds, so
                            #                it is selective here and was discarded for a
                            #                measurement that does not transfer.
                            #   features     the per-candidate top features by A's rule
                            #                gate, for EVERY candidate rather than two.
                            #   mass_*       A is ONE softmax over the state tokens and
                            #                all N*Rt candidate tokens together, so the
                            #                share it puts on each candidate is a
                            #                cross-candidate salience the per-candidate
                            #                renormalisation inside reasoning.evidence
                            #                removes. Never computed anywhere before.
                            #
                            # mass is reported both raw and per token: a candidate whose
                            # instance defines more features draws more mass mechanically,
                            # and only the per-token figure is comparable across them.
                            A_g, A_r, A_p = res["A_g"], res["A_r"], res["A_p"]
                            rmask = res["r_mask"]
                            n = int(res["cmask"].sum())
                            m_raw, m_tok, m_prod = [], [], []
                            for i in range(n):
                                nt = float(rmask[i].sum()) + float(A_p.shape[1])
                                mr = float(A_r[i].sum()) + float(A_p[i].sum())
                                m_raw.append(mr)
                                m_tok.append(mr / max(nt, 1.0))
                                m_prod.append(float(A_p[i].sum()) / max(mr, 1e-12))
                            fh.write(json.dumps({
                                "group_id": row["group_id"], "depth": row.get("depth"),
                                "smiles": row["state_smiles"],
                                "pick": ev["pick"], "rival": ev["rival"],
                                "rival_rule": ev["rival_rule"],
                                "argmax": ev["argmax"],
                                "pick_is_argmax": ev["pick_is_argmax"],
                                "state": ev["state"], "contrast": ev["contrast"],
                                # v9. The RICH contrast for EVERY possible rival, not
                                # just A_c's. A_c's own choice is index-skewed (#1 on
                                # 34.3% of rounds vs 20.9-23.4%), which taught the v7/v8
                                # student that #1 is the option to reject -- it chose #1
                                # on 12.6% and 21.6% of rounds against a 25% truth
                                # share. Balancing the rival index needs the contrast
                                # for an arbitrary pair, and rebuilding one from the
                                # cached per-candidate top-6 gate lists is far too thin
                                # (0 or 1 usable line on 76% of pairs against 9+ here).
                                # The GPU forward is already done, so each extra rival
                                # costs only the numpy diff loop.
                                # v10: the 4-way table. See reasoning.table4 -- every
                                # span version so far has argued PAIRWISE, which is the
                                # right shape for set_acc and the wrong one for beam_acc,
                                # and that is exactly the split the five results show.
                                "table4": rs.table4(row, res, sc.spec, top_k=8,
                                                    pick=row["committed"]),
                                "contrast_by_rival": {
                                    str(j): rs.evidence(
                                        row, res, sc.spec, top_state=1, top_rule=1,
                                        top_contrast=12,
                                        pick_override=row["committed"],
                                        rival_override=j)["contrast"]
                                    for j in range(int(res["cmask"].sum()))
                                    if j != ev["pick"]},
                                "A_c": [[float(x) for x in r_] for r_ in
                                        res["A_c"][:n, :n]],
                                "A_c_row": ev["A_c_row"],
                                "self_weight": ev["self_weight"],
                                "uniform": ev["uniform"],
                                "mass_state": float(A_g.sum()),
                                "mass_cand": m_raw,
                                "mass_cand_per_token": m_tok,
                                "mass_prod_frac": m_prod,
                                "candidates": [{"index": c["index"], "rule": c["rule"],
                                                "q": c["q"], "score": c["score"],
                                                "features": c["features"]}
                                               for c in ev["candidates"]],
                            }, ensure_ascii=False) + "\n")
                n_tot += len(flat)
                el = time.time() - t_split
                print(f"  shards {i+len(grp):>4}/{len(todo)}  rounds {n_tot:>7,}  "
                      f"embed {n_new:>6} mol in {tE:>5.1f}s  "
                      f"score {time.time()-tS:>5.1f}s  "
                      f"[{n_tot/max(el,1e-9):.0f} rounds/s]", flush=True)
        print(f"# {split}: {n_tot:,} rounds in {time.time()-t_split:.0f}s -> {dst}",
              flush=True)


if __name__ == "__main__":
    main()
