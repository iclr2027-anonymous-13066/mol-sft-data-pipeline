#!/usr/bin/env python
"""Keep only the tool chains that will actually produce an AUTHORED round.

Building the "+authored" delta means generating a second stage-4 corpus and then
throwing away everything except the segments rendered with an empty
``suggest_edits`` response. Measured on the 2M chains: with
AUTHORED_EDIT_FRACTION=0.5 / AUTHORED_MAX_FAILING=2, only ~28% of chains contain
even one such round, so a full second pass spends ~72% of its GPU time producing
records that the delta step discards.

This pre-filter answers the same question stage 4 will, BEFORE any LLM runs, and
writes a reduced tool-chain directory. Point ``INPUT_ROOT`` at the result:

    bash run_scripts/pipeline/4a_filter_authored_chains.sh
    INPUT_ROOT=<filtered> OUTPUT_ROOT=<...>_authored \\
    AUTHORED_EDIT_FRACTION=0.5 AUTHORED_MAX_FAILING=2 \\
        bash run_scripts/pipeline/4_sftdata_gen.sh

WHY IT REPRODUCES THE DECISION EXACTLY
--------------------------------------
It does not re-implement the rule. It imports the pipeline's own
``build_segments``, ``_out_of_range`` and ``_should_author`` and evaluates the
same predicate on the same objects, including the same deterministic
``md5(task_id|segment_index|authored)`` draw. A hand-rolled approximation gets
this wrong: a naive "every suggest_edits step is a round" reading of the chain
gives 46.6% eligibility where the segmenter gives 24.3%, because the segmenter
merges lead-ins and drops rounds without a usable prior state.

Chains are copied VERBATIM — the chain is the unit stage 4 consumes, and a chain
that survives still needs all of its rounds so the memory block of its authored
segment carries the right history.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")))


def _load_pipeline_bits():
    """Import the generator/segmenter helpers through importlib.

    The package directory starts with a digit, so a plain ``from 4_… import``
    is a syntax error; the module has to come in by name.
    """
    import importlib

    gen = importlib.import_module("4_sftdata_gen.generator")
    seg = importlib.import_module("4_sftdata_gen.segmenter")
    sch = importlib.import_module("4_sftdata_gen.schema")
    return gen, seg, sch


def _process_file(job) -> tuple:
    """Filter ONE shard. Returns (basename, Counter). Runs in a worker process.

    Work is split per FILE, matching the other stage-4 passes: files are
    independent, the kept lines are written by the worker itself so nothing large
    crosses the process boundary, and one slow file cannot stall the others.
    """
    path, out_dir, fraction, max_failing, dry_run = job
    gen, seg_mod, sch = _load_pipeline_bits()
    cfg = type("C", (), {"authored_edit_fraction": fraction,
                         "authored_max_failing": max_failing})()
    stat = Counter()
    keep_lines = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            stat["chains"] += 1
            try:
                rec = json.loads(line)
                inp = sch.GeneratorInput(**rec)
            except Exception:
                stat["unparseable"] += 1
                continue
            meta = inp.metadata or {}
            targets = meta.get("target_properties") or {}
            key = str(meta.get("task_id") or (inp.user_prompt or "")[:96])
            try:
                segments = seg_mod.build_segments(inp)
            except Exception:
                stat["unsegmentable"] += 1
                continue
            n_auth = 0
            for s in segments:
                if not s.lead_steps:
                    continue
                prior = s.prior_rounds[-1] if s.prior_rounds else None
                n_fail = len(gen._out_of_range(prior, targets))
                if gen._should_author(cfg, key, s.segment_index, n_fail):
                    n_auth += 1
            if n_auth:
                stat["kept"] += 1
                stat["authored_rounds"] += n_auth
                keep_lines.append(line)
            else:
                stat["dropped"] += 1
    if keep_lines and not dry_run:
        dst = os.path.join(out_dir, os.path.basename(path))
        with open(dst, "w") as f:
            f.write("\n".join(keep_lines) + "\n")
    return os.path.basename(path), stat


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True, help="tool-chain dir (stage-3 output)")
    ap.add_argument("--output-dir", required=True, help="filtered tool-chain dir")
    ap.add_argument("--fraction", type=float, required=True,
                    help="AUTHORED_EDIT_FRACTION the stage-4 run will use")
    ap.add_argument("--max-failing", type=int, default=2,
                    help="AUTHORED_MAX_FAILING the stage-4 run will use")
    ap.add_argument("--num-proc", type=int, default=16,
                    help="worker processes, one file at a time each (default 16). "
                         "Measured 4.2s per 200MB/20k-chain shard on one core, so 100 "
                         "shards is ~7 min serial and well under a minute at 16.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    shards = sorted(glob.glob(os.path.join(args.input_dir, "toolchains_*chunk_*.jsonl")))
    if not shards:
        sys.exit(f"no toolchains_*chunk_*.jsonl under {args.input_dir}")
    if not args.dry_run:
        os.makedirs(args.output_dir, exist_ok=True)

    nproc = max(1, min(args.num_proc, len(shards)))
    print(f"filtering {len(shards)} shard(s) with {nproc} process(es)"
          f"{' (dry run)' if args.dry_run else ''}", flush=True)

    jobs = [(p, args.output_dir, args.fraction, args.max_failing, args.dry_run)
            for p in shards]
    stat = Counter()
    t0 = time.time()
    done = 0

    def _report(name, s):
        nonlocal done
        done += 1
        stat.update(s)
        rate = s["kept"] / max(1, s["chains"])
        el = time.time() - t0
        eta = el / done * (len(shards) - done)
        # Progress per file: without it a 7-minute serial run looks like a hang,
        # which is exactly how this script first got reported as "too slow".
        print(f"  [{done:>3}/{len(shards)}] {name}  "
              f"chains {s['chains']:>6}  kept {s['kept']:>6} ({rate:5.1%})  "
              f"elapsed {el:5.1f}s  eta {eta:5.1f}s", flush=True)

    if nproc == 1:
        for job in jobs:
            _report(*_process_file(job))
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=nproc) as ex:
            futs = [ex.submit(_process_file, j) for j in jobs]
            for fut in as_completed(futs):
                _report(*fut.result())

    n = stat["chains"]
    print("=" * 70)
    print(f"  chains read              : {n}")
    print(f"  KEPT (>=1 authored round): {stat['kept']}  ({stat['kept']/max(1,n):.1%})")
    print(f"  dropped (contribute 0)   : {stat['dropped']}  ({stat['dropped']/max(1,n):.1%})")
    if stat["unparseable"]:
        print(f"  unparseable              : {stat['unparseable']}")
    if stat["unsegmentable"]:
        print(f"  unsegmentable            : {stat['unsegmentable']}")
    print(f"  authored rounds expected : {stat['authored_rounds']}")
    print(f"  wall clock               : {time.time()-t0:.1f}s")
    if stat["kept"]:
        print(f"\n  stage-4 cost reduced to ~{stat['kept']/max(1,n):.0%} "
              f"({n/max(1,stat['kept']):.1f}x faster)")
    if args.dry_run:
        print("\n  DRY RUN — nothing written.")
    else:
        print(f"\n  wrote {args.output_dir}")
        print("  NOTE the .done markers and any other files were NOT copied; point")
        print("       INPUT_ROOT at this dir and let stage 4 start clean.")


if __name__ == "__main__":
    main()
