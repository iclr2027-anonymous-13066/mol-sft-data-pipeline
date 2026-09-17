"""states.jsonl -> fixed-shape memmapped tensors, so training is pure compute.

A depth-5 dump over the 10K produces millions of decision states; re-parsing JSON
every epoch on 8 ranks would dominate the step time. This encodes once with a
FeatureSpec fitted on the TRAIN split only, and writes:

    g.npy      [S, G]        global (state) features, normalised
    gmask.npy  [S, G]        False where the feature is meaningless for this instance
    r.npy      [S, N, R]     rule features per candidate, normalised
    rmask.npy  [S, N, R]
    y.npy      [S, N]        the label, sat_ratio (-1 on padding)
    cmask.npy  [S, N]        candidate padding mask
    ctx.npy    [S]           row into the context-embedding matrix (the STATE molecule)
    cctx.npy   [S, N]        row into the same matrix for each CANDIDATE's post-edit
                             molecule; -1 on padding, and -1 for a product the context
                             encoder has no row for
    depth.npy  [S]           the state's depth, for per-depth metrics
    source.npy [S]           0 = scaffold, 1 = fg
    group.npy  [S]           instance key: states of one instance never straddle splits
    split.npy  [S]           0 train / 1 val / 2 test

Splitting is by INSTANCE, not by state: two states from the same search tree share a
seed and most of their history, so a state-level split would leak.

`--split-key` chooses what an "instance" means for that purpose, and the right answer
differs per condition:

``instance`` (default) one search tree = one unit. Matches how the model is used — the
              same molecule recurs under different property boxes, and adapting to the
              box is the job. But measured on the first depth-5 set, the FG half has
              only 261 distinct seed molecules in train (its seed is BUILT from the 1-2
              required groups, out of 61 patterns), so 99% of FG val instances stand on
              a molecule seen in training. The scaffold half is 1%: its seeds are Murcko
              scaffolds of distinct references.
``seed``      the seed MOLECULE is the unit and never straddles the split. The strict
              generalisation test, and the honest one for the FG half.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np

from .features import FeatureSpec


def _iter(path):
    with open(path) as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def _split_of(group_key: str, frac_val: float, frac_test: float, seed: int) -> int:
    h = hashlib.blake2b(f"{seed}:{group_key}".encode(), digest_size=8).digest()
    u = int.from_bytes(h, "big") / float(1 << 64)
    if u < frac_test:
        return 2
    if u < frac_test + frac_val:
        return 1
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", action="append", default=[],
                    help="states.jsonl whose rows are split by the usual hash "
                         "(repeatable)")
    ap.add_argument("--train-states", action="append", default=[],
                    help="states.jsonl whose rows are ALL train, whatever the hash says "
                         "(repeatable). For training on one corpus and evaluating on "
                         "another — see --eval-states.")
    ap.add_argument("--eval-states", action="append", default=[],
                    help="states.jsonl to take ONLY the val/test rows from; its train "
                         "rows are dropped (repeatable). Paired with --train-states this "
                         "trains on a new corpus while keeping an existing run's held-out "
                         "splits, which is only honest because the hash is a pure "
                         "function of (seed, split_key): pass the same --seed / "
                         "--val-frac / --test-frac / --split-key the original used and "
                         "the same instances land in val and test, exactly.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--spec", default="",
                    help="an existing spec.json to reuse instead of fitting one. The "
                         "feature SET and the normalisation both come from it, so the "
                         "tensors stay shape- and scale-compatible with the run that "
                         "produced it — which is what makes a checkpoint trained here "
                         "comparable to one trained there. Without it the spec is fitted "
                         "on this run's train rows, and a small corpus can silently drop "
                         "a rule column it happens not to contain.")
    ap.add_argument("--ctx-dir", default="", help="context_embed output (index.json)")
    ap.add_argument("--exclude-smiles", action="append", default=[],
                    help="file of SMILES (one per line); DROP every decision state whose "
                         "own molecule is listed. Written by context_embed as "
                         "rejected_smiles.txt — MolBERT refuses anything over 128 tokens "
                         "and would otherwise contribute a zero context vector, which is "
                         "not a neutral input since the gate is conditioned on it. "
                         "Repeatable, and it must be given the SAME list for every "
                         "encoder or the splits stop being comparable.")
    ap.add_argument("--fit-sample", type=int, default=50000,
                    help="train rows used to fit the normalisation")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--test-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-key", choices=["instance", "seed"], default="instance",
                    help="what never straddles the split: the search tree, or the seed "
                         "molecule it starts from")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    drop = set()
    for path in args.exclude_smiles:
        with open(path) as fh:
            drop |= {ln.strip() for ln in fh if ln.strip()}
    if drop:
        print(f"# excluding states whose molecule is one of {len(drop):,} listed SMILES")

    ctx_row = {}
    if args.ctx_dir:
        with open(os.path.join(args.ctx_dir, "index.json")) as fh:
            idx = json.load(fh)
        ctx_row = {s: i for i, s in enumerate(idx["smiles"])}
        print(f"# context: {len(ctx_row):,} molecules, dim {idx['dim']} ({idx['encoder']})")

    # pass 1: count, and collect a TRAIN-only sample for the spec
    def split_key(row: dict) -> str:
        if args.split_key == "seed":
            return "seed:" + str(row.get("state_smiles") or f"{row['source']}:{row['index']}")
        return f"{row['source']}:{row['index']}"

    # (path, policy): "hash" splits as usual, "train" forces every row to train,
    # "eval" keeps only the rows the hash puts in val/test.
    sources = ([(p, "hash") for p in args.states]
               + [(p, "train") for p in args.train_states]
               + [(p, "eval") for p in args.eval_states])
    if not sources:
        raise SystemExit("give at least one of --states / --train-states / --eval-states")

    def rows_with_split():
        """Every kept row, with the split it belongs in and which corpus it came from.

        The corpus index is carried because ``split_key`` is only unique WITHIN one
        states file: it is "<source>:<index>", and every branch_tree run numbers its
        instances from 0 with the same two tags. Two corpora combined here therefore
        collide on it — `scaffold:5` of a 2M slice is a different molecule from
        `scaffold:5` of the 50K set, and with one forced to train and the other in val
        they would share a group id across the split boundary. It is deliberately NOT
        mixed into the split hash: that hash has to keep reproducing the other run's
        val/test exactly, so it stays a function of (seed, split_key) alone.
        """
        for ci, (path, policy) in enumerate(sources):
            for row in _iter(path):
                if drop and row.get("state_smiles") in drop:
                    yield None, None, None
                    continue
                sp = _split_of(split_key(row), args.val_frac, args.test_frac, args.seed)
                if policy == "train":
                    sp = 0
                elif policy == "eval" and sp == 0:
                    continue            # its train rows belong to the other corpus
                yield row, sp, ci

    n = 0
    n_dropped = 0
    counts = [0, 0, 0]
    sample = []
    for row, sp, _ci in rows_with_split():
        if row is None:
            n_dropped += 1
            continue
        if sp == 0 and len(sample) < args.fit_sample:
            sample.append(row)
        counts[sp] += 1
        n += 1
    if drop:
        print(f"# dropped {n_dropped:,} states on excluded molecules "
              f"({n_dropped/max(n + n_dropped, 1):.4%})")
    print(f"# {n:,} states  (train {counts[0]:,} / val {counts[1]:,} / test {counts[2]:,})")
    if args.spec:
        spec = FeatureSpec.load(args.spec)
        print(f"# spec from {args.spec} (not refitted)")
    else:
        print(f"# fitting the spec on {len(sample):,} train rows")
        spec = FeatureSpec.from_rows(sample)
        spec.fit(sample)
    spec.save(os.path.join(args.out_dir, "spec.json"))
    G, R, N = len(spec.global_names), len(spec.rule_names), spec.max_candidates
    print(f"# G={G} global, R={R} rule features, N<={N} candidates")

    gc, gsd, rc, rsd = spec.norm_vectors()
    mm = lambda name, shape, dt: np.lib.format.open_memmap(
        os.path.join(args.out_dir, name), mode="w+", dtype=dt, shape=shape)
    A = {"g": mm("g.npy", (n, G), np.float32), "gmask": mm("gmask.npy", (n, G), bool),
         "r": mm("r.npy", (n, N, R), np.float32), "rmask": mm("rmask.npy", (n, N, R), bool),
         "y": mm("y.npy", (n, N), np.float32), "cmask": mm("cmask.npy", (n, N), bool),
         "ctx": mm("ctx.npy", (n,), np.int32),
         "cctx": mm("cctx.npy", (n, N), np.int32),
         "depth": mm("depth.npy", (n,), np.int16),
         "source": mm("source.npy", (n,), np.int8), "group": mm("group.npy", (n,), np.int64),
         "split": mm("split.npy", (n,), np.int8)}

    groups = {}
    miss_ctx = 0
    miss_prod = 0
    n_prod = 0
    i = -1
    for row, sp, ci in rows_with_split():
        if row is None:
            continue
        i += 1
        g, gm, r, rm, y, cm = spec.encode(row)
        A["g"][i] = (g - gc) / gsd
        A["gmask"][i] = gm
        A["r"][i] = (r - rc) / rsd
        A["rmask"][i] = rm
        A["y"][i] = y
        A["cmask"][i] = cm
        A["group"][i] = groups.setdefault((ci, split_key(row)), len(groups))
        A["split"][i] = sp
        A["depth"][i] = int(row.get("depth") or 0)
        # prefix, not equality: a re-run lands in its own directory (e.g.
        # "scaffold_redo") and must still count as the scaffold condition.
        A["source"][i] = 0 if str(row.get("source", "")).startswith("scaffold") else 1
        j = ctx_row.get(row.get("state_smiles"), -1)
        miss_ctx += j < 0
        A["ctx"][i] = j
        # the post-edit molecule of each candidate. context_embed already fingerprints
        # these — its unique_smiles() walks candidates as well as states — so this is a
        # lookup into the SAME matrix, not a second encoder pass.
        cc = np.full(N, -1, dtype=np.int32)
        for k, c in enumerate((row.get("candidates") or [])[:N]):
            jj = ctx_row.get(c.get("smiles"), -1)
            cc[k] = jj
            n_prod += 1
            miss_prod += jj < 0
        A["cctx"][i] = cc
        if i % 100000 == 0:
            print(f"  {i:,}/{n:,}", flush=True)
    for v in A.values():
        v.flush()

    counts = np.bincount(np.asarray(A["split"]), minlength=3).tolist()
    meta = {"n": n, "G": G, "R": R, "N": N, "n_groups": len(groups),
            "split_counts": {"train": counts[0], "val": counts[1], "test": counts[2]},
            "missing_context": int(miss_ctx),
            "missing_product_context": int(miss_prod),
            "n_products": int(n_prod), "ctx_dir": args.ctx_dir,
            "excluded_states": int(n_dropped),
            "exclude_smiles": list(args.exclude_smiles),
            "split_key": args.split_key,
            "states": args.states, "train_states": args.train_states,
            "eval_states": args.eval_states, "spec_from": args.spec or None,
            "seed": args.seed}
    with open(os.path.join(args.out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"# {json.dumps(meta)}")
    if miss_ctx:
        print(f"# [warn] {miss_ctx:,} states have no context embedding "
              f"(ctx=-1 -> zero vector). Re-run context_embed on THIS states file.")
    if miss_prod:
        print(f"# [warn] {miss_prod:,}/{n_prod:,} candidate products have no embedding "
              f"({miss_prod/max(n_prod,1):.4%}); they get a zero vector when "
              f"--product-ctx is on.")


if __name__ == "__main__":
    main()
