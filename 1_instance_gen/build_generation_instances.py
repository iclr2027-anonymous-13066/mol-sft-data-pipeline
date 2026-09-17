#!/usr/bin/env python3
r"""
build_generation_instances.py
==============================
Instance generator for the generation task.

Draws a reference molecule at random from the pool dump (train_pool_props.parquet) and
builds "ref-anchored property constraints" from that molecule's measured properties,
written out in the benchmark schema (generation.jsonl). Each instance also records the
joint hit count: how many pool molecules satisfy every constraint at once.

Recipe
------
1. Pick one reference molecule at random, from the whole pool.
2. Number of integer properties ki ~ U[1, max-int]; of continuous ones kc ~ U[1, max-cont].
3. Choose ki integer and kc continuous properties at random, excluding HIA and formal_charge.
4. Integer properties: pick both / single (min|max) / exact at random around the ref value;
   for single, which side is also random. Bounds sit a small integer offset from the ref
   (0..int-offset) and always contain it, so the instance stays feasible.
5. Continuous properties: build a q-window in quantile space.
     q ~ U[q-min, q-max], straddle α ~ U[0,1]
     window (in quantiles) = [pc - alpha*q, pc + (1-alpha)*q]   (pc = the ref's percentile)
   If one end reaches the edge of the distribution (0 or 1), that bound is dropped and the
   constraint becomes one-sided by itself. Because one-sidedness is never chosen
   explicitly, the pass rate stays bounded by q, and the ref is always inside.
6. Round the bounds in the direction that preserves feasibility (min down, max up), then
   measure the joint hit rate of those final bounds against the whole pool (or a sample)
   and store it as hit_count / hit_rate.

Output schema (benchmark fields + hit information)
  {"id","task_type","properties":[{"property","min"?,"max"?}...],"ref_smiles",
   "hit_count","hit_rate","pool_size"}

This script needs no HTTP and no heavy dependencies: parquet + numpy only.

Parallelism
------
The per-instance bottleneck is the joint hit-mask, which scans the whole 9.75M-row pool.
Instances are independent, so `--workers` spreads them over several processes. The pool,
the sorted arrays and the SMILES are loaded once in the parent and shared read-only with
the children through fork copy-on-write, so nothing is copied. Reproducibility: chunk c
seeds its RNG with `SeedSequence(seed, spawn_key=(c,))`, which is independent of
`--workers` (a pure performance knob) and depends only on (seed, chunk-size, n).

Usage
-----
  python 1_instance_gen/build_generation_instances.py \
      --pool data/pool/train_pool_props.parquet \
      --output data/training_data/instances/generation_200k.jsonl \
      --n 200000 --q-min 0.1 --q-max 0.5 --seed 0 --workers 64
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import shutil
import time

import numpy as np
import pyarrow.parquet as pq

# Exclude the properties that carry no discriminative signal (HIA ~ 1.0, formal_charge ~ 0).
INT_PROPS = ["HBD", "HBA", "rotB", "rings_total", "heavy_atoms"]
CONT_PROPS = ["MW", "logP", "logD", "logS", "TPSA", "QED", "BBBP", "Mutag", "MR"]
# Ordering used to keep the output readable.
PROP_ORDER = ["MW", "logP", "logD", "logS", "TPSA", "QED", "BBBP", "Mutag", "MR",
              "HBD", "HBA", "rotB", "rings_total", "heavy_atoms"]
# Decimal places for rounding continuous properties (feasibility-preserving: min down, max up).
DECIMALS = {"MW": 1, "logP": 2, "logD": 2, "logS": 2, "TPSA": 1,
            "QED": 3, "BBBP": 3, "Mutag": 3, "MR": 1}

# Global context the children inherit (share) through fork. Filled in by main().
_G: dict = {}


def floor_to(x: float, d: int) -> float:
    f = 10 ** d
    return np.floor(x * f) / f


def ceil_to(x: float, d: int) -> float:
    f = 10 ** d
    return np.ceil(x * f) / f


# ---------------------------------------------------------------------------
# Instance construction (uses the global _G context)
# ---------------------------------------------------------------------------
def _pct_of(p: str, x: float) -> float:
    return float(np.searchsorted(_G["SORTED"][p], x)) / _G["N"]


def _qval(p: str, f: float) -> float:
    a = _G["SORTED"][p]
    return float(a[int(np.clip(f, 0.0, 1.0) * (len(a) - 1))])


def _build_int(p: str, x0: float, rng, int_offset: int):
    x0 = int(round(x0))
    mode = rng.choice(["both", "single", "exact"])
    off = lambda: int(rng.integers(0, int_offset + 1))
    if mode == "exact":
        lo = hi = x0
    elif mode == "both":
        lo, hi = x0 - off(), x0 + off()
    else:  # single
        if rng.random() < 0.5:
            lo, hi = x0 - off(), None          # min-only
        else:
            lo, hi = None, x0 + off()          # max-only
    if lo is not None:
        lo = max(lo, int(np.floor(_G["PMIN"][p])))
    if hi is not None:
        hi = min(hi, int(np.ceil(_G["PMAX"][p])))
    return lo, hi


def _build_cont(p: str, x0: float, rng, q_min: float, q_max: float):
    q = float(rng.uniform(q_min, q_max))
    alpha = float(rng.uniform(0.0, 1.0))
    pc = _pct_of(p, x0)
    lo_q, hi_q = pc - alpha * q, pc + (1.0 - alpha) * q
    d = DECIMALS[p]
    lo = None if lo_q <= 0.0 else floor_to(_qval(p, lo_q), d)
    hi = None if hi_q >= 1.0 else ceil_to(_qval(p, hi_q), d)
    # Both ends dropping cannot happen while q < 1. Safety net:
    if lo is None and hi is None:
        hi = ceil_to(_qval(p, hi_q), d)
    # If rounding collapsed a two-sided window to lo == hi, widen it by one step while
    # keeping the ref inside.
    if lo is not None and hi is not None and hi <= lo:
        hi = round(lo + 10 ** (-d), d)
    # Guarantee the ref stays feasible (rounding edge cases).
    if lo is not None and x0 < lo:
        lo = floor_to(x0, d)
    if hi is not None and x0 > hi:
        hi = ceil_to(x0, d)
    return lo, hi


def _hit_mask(constraints):
    HCOL = _G["HCOL"]
    m = np.ones(_G["hN"], dtype=bool)
    for p, lo, hi in constraints:
        v = HCOL[p]
        if lo is not None:
            m &= v >= lo
        if hi is not None:
            m &= v <= hi
    return m


def _gen_records(start_i: int, count: int, seed: int, spawn_key: int):
    """Build the records (dicts) for one chunk and return them as a list."""
    rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(spawn_key,)))
    N = _G["N"]
    HCOL = _G["HCOL"]
    hN = _G["hN"]
    full = _G["HCOL_is_full"]
    COL = _G["COL"]
    smiles = _G["smiles"]
    max_int = _G["max_int"]
    max_cont = _G["max_cont"]
    int_offset = _G["int_offset"]
    q_min = _G["q_min"]
    q_max = _G["q_max"]
    id_prefix = _G["id_prefix"]

    cand = _G["CAND"]                       # None = the whole pool; otherwise the candidate row indices
    n_cand = N if cand is None else len(cand)

    out = []
    for j in range(count):
        i = start_i + j
        r = int(rng.integers(n_cand))
        if cand is not None:
            r = int(cand[r])
        ki = int(rng.integers(1, max_int + 1))
        kc = int(rng.integers(1, max_cont + 1))
        int_sel = list(rng.choice(INT_PROPS, ki, replace=False))
        cont_sel = list(rng.choice(CONT_PROPS, kc, replace=False))

        constraints = []  # (prop, lo, hi)
        for p in int_sel:
            lo, hi = _build_int(p, COL[p][r], rng, int_offset)
            constraints.append((p, lo, hi))
        for p in cont_sel:
            lo, hi = _build_cont(p, COL[p][r], rng, q_min, q_max)
            constraints.append((p, lo, hi))

        # Joint hit against the final bounds. The ref is always feasible, so the true
        # pool count is >= 1.
        mask = _hit_mask(constraints)
        hits = int(mask.sum())
        if full:                                # whole pool: exact (>= 1, since the ref is in it)
            hit_count = hits
            hit_rate = hits / hN
        else:                                   # sample: scaled to the pool, floored at 1 by the ref
            hit_count = max(int(round((hits / hN) * N)), 1)
            hit_rate = hit_count / N

        # Output properties in readable order; a None bound omits the key entirely.
        cmap = {p: (lo, hi) for p, lo, hi in constraints}
        props_out = []
        for p in PROP_ORDER:
            if p not in cmap:
                continue
            lo, hi = cmap[p]
            entry = {"property": p}
            if lo is not None:
                entry["min"] = float(lo)
            if hi is not None:
                entry["max"] = float(hi)
            props_out.append(entry)

        out.append({
            "id": f"{id_prefix}_{i}",
            "task_type": "generation",
            "properties": props_out,
            "ref_smiles": smiles[r],
            "hit_count": hit_count,
            "hit_rate": hit_rate,
            "pool_size": N,
        })
    return out


def _worker(chunk):
    """Write the shard file for (chunk_id, start_i, count) and return (chunk_id, path, count)."""
    chunk_id, start_i, count = chunk
    recs = _gen_records(start_i, count, _G["seed"], chunk_id)
    shard = f"{_G['output']}.part_{chunk_id:06d}"
    with open(shard, "w") as f:
        f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs))
    return chunk_id, shard, count


def main() -> None:
    ap = argparse.ArgumentParser(
        description="ref-anchored generation instance builder (q-window).",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", default="data/pool/train_pool_props.parquet",
                    help="the property dump parquet")
    ap.add_argument("--output", default="data/training_data/instances/generation.jsonl",
                    help="output jsonl path")
    ap.add_argument("--n", type=int, default=10000, help="number of instances to build")
    ap.add_argument("--q-min", type=float, default=0.1, help="lower bound on the continuous-property window pass rate q")
    ap.add_argument("--q-max", type=float, default=0.5, help="upper bound on the continuous-property window pass rate q")
    ap.add_argument("--max-int", type=int, default=len(INT_PROPS),
                    help=f"maximum number of integer properties (<= {len(INT_PROPS)})")
    ap.add_argument("--max-cont", type=int, default=len(CONT_PROPS),
                    help=f"maximum number of continuous properties (<= {len(CONT_PROPS)})")
    ap.add_argument("--int-offset", type=int, default=2,
                    help="largest offset from the ref for integer both/single bounds (drawn from 0..offset)")
    ap.add_argument("--smiles-col", default="smiles", help="name of the SMILES column")
    ap.add_argument("--exclude-smiles", action="append", default=[], metavar="FILE",
                    help="file of SMILES to exclude from the reference candidates, one per line; "
                         "may be given several times. The exclusion applies to reference "
                         "sampling ONLY — hit_count and hit_rate are still computed against "
                         "the full pool, so the numbers stay comparable with an existing "
                         "instance set.")
    ap.add_argument("--id-prefix", default="generation", help="id prefix")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hitrate-sample", type=int, default=0,
                    help="pool sample size for estimating the joint hit rate (0 = exact over the "
                         "whole pool, slow). With a sample, hit_count is an estimate scaled "
                         "back up to the full pool.")
    ap.add_argument("--workers", type=int, default=min(64, max(1, (os.cpu_count() or 2) - 2)),
                    help="worker processes (default min(64, CPU-2)); 1 runs in a single process. "
                         "The hit-mask is memory-bandwidth bound, so too many workers makes "
                         "it slower, not faster (~48-64 is best for float64; float32 keeps "
                         "gaining up to ~96).")
    ap.add_argument("--chunk-size", type=int, default=500,
                    help="instance chunk size handed to each worker. Reproducibility depends only "
                         "on (seed, chunk-size, n), never on --workers.")
    ap.add_argument("--dtype", choices=["float64", "float32"], default="float64",
                    help="dtype of the pool property arrays. float32 halves the hit-mask memory "
                         "traffic and runs ~2x faster, at the cost of slight drift in the "
                         "quantile bounds and hit_count versus float64 (~1%% relative, "
                         "statistically harmless). Recommended for large runs.")
    args = ap.parse_args()

    dtype = np.float32 if args.dtype == "float32" else np.float64

    max_int = min(args.max_int, len(INT_PROPS))
    max_cont = min(args.max_cont, len(CONT_PROPS))

    # -- load the pool: 14 properties + smiles --
    cols = INT_PROPS + CONT_PROPS + [args.smiles_col]
    print(f"# loading pool: {args.pool}")
    t = time.time()
    tbl = pq.read_table(args.pool, columns=cols)
    smiles = tbl.column(args.smiles_col).to_pylist()
    COL = {p: tbl.column(p).to_numpy().astype(dtype) for p in INT_PROPS + CONT_PROPS}
    del tbl
    N = len(smiles)
    print(f"# pool rows={N:,}  loaded in {time.time()-t:.1f}s")

    # Sorted arrays for the continuous properties (for the quantile function) + observed min/max.
    SORTED = {p: np.sort(COL[p]) for p in CONT_PROPS}
    PMIN = {p: float(COL[p].min()) for p in INT_PROPS + CONT_PROPS}
    PMAX = {p: float(COL[p].max()) for p in INT_PROPS + CONT_PROPS}

    # Restrict the reference candidates (--exclude-smiles): drop molecules already used as
    # references by another instance set, to build a new, non-overlapping one.
    CAND = None
    if args.exclude_smiles:
        excl = set()
        for path in args.exclude_smiles:
            with open(path) as f:
                for line in f:
                    s = line.rstrip("\n")
                    if s:
                        excl.add(s)
        print(f"# exclude list: {len(excl):,} unique SMILES from {len(args.exclude_smiles)} file(s)")
        keep = np.fromiter((s not in excl for s in smiles), dtype=bool, count=N)
        CAND = np.flatnonzero(keep)
        print(f"# ref candidates: {len(CAND):,} / {N:,} rows "
              f"({N - len(CAND):,} excluded)")
        if len(CAND) == 0:
            raise SystemExit("ERROR: exclusion removed every pool row.")

    # The pool used for the hit rate: whole or sampled.
    if args.hitrate_sample and args.hitrate_sample < N:
        seed_rng = np.random.default_rng(args.seed)
        hidx = seed_rng.choice(N, args.hitrate_sample, replace=False)
        HCOL = {p: COL[p][hidx] for p in INT_PROPS + CONT_PROPS}
        hN = args.hitrate_sample
        HCOL_is_full = False
        print(f"# hit-rate via sample of {hN:,} (count scaled up to the full {N:,})")
    else:
        HCOL = COL
        hN = N
        HCOL_is_full = True
        print(f"# hit-rate via full pool ({N:,}) — exact but slow")

    # The global context the children will share through fork.
    _G.update(dict(
        COL=COL, SORTED=SORTED, PMIN=PMIN, PMAX=PMAX, smiles=smiles, N=N, CAND=CAND,
        HCOL=HCOL, hN=hN, HCOL_is_full=HCOL_is_full,
        max_int=max_int, max_cont=max_cont, int_offset=args.int_offset,
        q_min=args.q_min, q_max=args.q_max, id_prefix=args.id_prefix,
        seed=args.seed, output=args.output,
    ))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    # Split into chunks; reproducibility depends only on chunk-size, n and seed.
    chunk_size = max(1, args.chunk_size)
    num_chunks = math.ceil(args.n / chunk_size)
    chunks = []
    for c in range(num_chunks):
        start_i = c * chunk_size
        count = min(chunk_size, args.n - start_i)
        chunks.append((c, start_i, count))

    workers = max(1, min(args.workers, num_chunks))
    print(f"# generating {args.n:,} instances via {workers} workers "
          f"({num_chunks} chunks × {chunk_size})")

    t0 = time.time()
    done = 0
    results = []  # (chunk_id, shard_path)

    if workers == 1:
        for ch in chunks:
            cid, shard, cnt = _worker(ch)
            results.append((cid, shard))
            done += cnt
            el = time.time() - t0
            print(f"  {done:,}/{args.n:,}  ({done/el:.1f} inst/s)")
    else:
        # fork: the children share the already-loaded _G (pool, sorted arrays, smiles)
        # copy-on-write.
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=workers) as pool:
            for cid, shard, cnt in pool.imap_unordered(_worker, chunks):
                results.append((cid, shard))
                done += cnt
                el = time.time() - t0
                print(f"  {done:,}/{args.n:,}  ({done/el:.1f} inst/s)", flush=True)

    # Concatenate the shards in id order (by chunk_id), then delete them.
    results.sort(key=lambda x: x[0])
    print(f"# merging {len(results)} shards → {args.output}")
    with open(args.output, "w") as fout:
        for _, shard in results:
            with open(shard, "r") as fin:
                shutil.copyfileobj(fin, fout, 1 << 20)
            os.remove(shard)

    print(f"\nSaved {args.n:,} instances → {args.output}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
