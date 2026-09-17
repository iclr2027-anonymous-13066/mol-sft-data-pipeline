#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Augment a per-row JSONL with Murcko-scaffold descriptions (1:1, order-preserving).

Unlike ``describe_scaffolds.py`` (a bulk de-duplicating generator keyed on SMILES),
this driver keeps **exactly one output row per input row**, in the original order,
and carries **all original keys forward unchanged**. It is meant for benchmark /
evaluation files where each line is a distinct record (e.g. duplicate SMILES or
duplicate ids must NOT collapse).

For each input record it:
  1. reads the SMILES from --smiles-key (default: ref_smiles),
  2. runs the deterministic analysis + dimension/eval SMARTS (`analyze_one`/`build_item`),
  3. polishes the description with the vLLM pool + faithfulness validate/revise loop
     (identical logic to describe_scaffolds.describe_one),
  4. writes {**original_record, **scaffold_fields} to the output JSONL.

Reuses the real pipeline machinery from describe_scaffolds.py so behaviour matches
the training-data generator exactly.

--no-describe skips step 3 entirely: no vLLM pool is contacted and ``description``
stays empty, so the output carries only the DETERMINISTIC structural constraint
(``scaffold_smiles`` / ``scaffold_smarts`` / ``dimension_smarts`` / ``eval_query`` and
the ring-system analysis). Use it when the consumer only needs the constraint — e.g.
stage 3, whose scaffold guard reads ``scaffold_smarts`` and never the prose — and no
GPU is available. ``template_draft`` (the deterministic draft the LLM would have
polished) is still written, so a description can be filled in later without redoing
the analysis.

Output is either a single file (--output) or a folder of fixed-size shards (--out-dir
+ --shard-size, default 10000/shard): `<prefix>-00000.jsonl`, `-00001.jsonl`, ...
Sharded writing is streaming, order-preserving, atomic per shard, and resumable
(already-complete shards are skipped on restart).

Usage:
  PY=python
  # sharded (10K per file) into a new folder:
  $PY 2_substructure_gen/augment_jsonl_with_scaffold.py \
      --input   data/training_data/instances/generation_200k.jsonl \
      --out-dir data/training_data/instances/generation_200k_scaffold \
      --shard-size 10000 --smiles-key ref_smiles
  # single file:
  $PY 2_substructure_gen/augment_jsonl_with_scaffold.py \
      --input data/training_data/instances/generation_benchmark.jsonl \
      --out-dir data/training_data/instances/generation_benchmark_scaffold \
      --smiles-key ref_smiles
"""
from __future__ import annotations

import argparse
import asyncio
import gc
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm

# Reuse the exact pipeline machinery (same directory).
from describe_scaffolds import (
    VLLMPool, build_base_urls, analyze_one, DEFAULT_SERVERS,
)
from scaffold_describer import normalize_text, validate_text


def load_records(path: str, smiles_key: str) -> list[dict]:
    """Read every JSONL line in order. Each record keeps a 0-based _row index."""
    recs: list[dict] = []
    with open(path) as fh:
        for ln, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            obj.setdefault("_row", ln)
            recs.append(obj)
    miss = sum(1 for r in recs if not r.get(smiles_key))
    if miss:
        print(f"# [warn] {miss}/{len(recs)} record(s) have no {smiles_key!r} value.",
              file=sys.stderr)
    return recs


def _scan_complete_shards(out_dir: str, prefix: str, shard_size: int, n: int) -> int:
    """Count the leading shards K (0..K-1) that were written *completely*, for resume.

    A shard k counts as done only when its line count matches exactly what is expected
    (shard_size, or the remainder for the last shard). Counting stops at the first
    incomplete or missing shard — generation restarts from there."""
    nshards = (n + shard_size - 1) // shard_size
    k = 0
    while k < nshards:
        p = os.path.join(out_dir, f"{prefix}-{k:05d}.jsonl")
        if not os.path.exists(p):
            break
        expected = min(shard_size, n - k * shard_size)
        try:
            c = sum(1 for _ in open(p))
        except OSError:
            break
        if c != expected:
            break
        k += 1
    return k


async def polish(pool: VLLMPool, server_idx: int, an, it: dict, validate_passes: int) -> dict:
    """Mirror describe_scaffolds.describe_one's polishing body; fill it['description']."""
    try:
        if an is None or it.get("analysis_error"):
            it["description"] = ""                       # analysis failed: leave empty
        elif not it["has_scaffold"]:
            it["description"] = it["template_draft"]     # acyclic: deterministic draft
        elif it.get("scaffold_kind") == "functional_group":
            it["description"] = it["template_draft"]     # FG framework: deterministic draft
        else:
            desc = normalize_text(
                await pool.describe(server_idx, an, draft=it["template_draft"]))
            viol = validate_text(desc, an)
            for _ in range(max(0, validate_passes)):
                if not viol:
                    break
                rev = normalize_text(await pool.revise(server_idx, an, desc, viol))
                rviol = validate_text(rev, an)
                if len(rviol) < len(viol):
                    desc, viol = rev, rviol
                else:
                    break
            it["description"] = desc
            if viol:
                it["description_violations"] = viol
    except Exception as exc:  # noqa: BLE001
        it["description"] = ""
        it["describe_error"] = str(exc)[:200]
    return it


async def _make_pool(args: argparse.Namespace):
    """Build + probe the vLLM pool. ``None`` when no server is reachable."""
    base_urls = build_base_urls(args.servers, args.host, args.base_ports)
    pool = VLLMPool(base_urls=base_urls, per_server_concurrency=args.concurrency_per_server,
                    timeout=args.timeout, max_retries=args.max_retries,
                    reasoning_effort=args.reasoning_effort, temperature=args.temperature,
                    max_tokens=args.max_tokens, max_sentences=args.max_sentences, model=args.model)
    await pool.discover_models()
    dropped = pool.prune_dead()
    if dropped:
        print(f"# [warn] dropped {dropped} unreachable server(s).")
    if pool.n_servers == 0:
        print("error: no live vLLM server. Try --servers/--host.", file=sys.stderr)
        return None
    uniq = sorted({m for m in pool.detected_models if m}) or [args.model]
    print(f"# Live servers: {pool.n_servers} (cap {pool.n_servers * args.concurrency_per_server}); "
          f"model(s): {uniq}")
    return pool


async def main_async(args: argparse.Namespace) -> int:
    smiles_key = args.smiles_key
    recs = load_records(args.input, smiles_key)
    n = len(recs)
    print(f"# Loaded {n} record(s) from {args.input} (smiles key = {smiles_key!r}).")

    # --- vLLM pool (same defaults as describe_scaffolds) ---
    # --no-describe: analysis only, so never touch the pool (nothing to serve).
    pool = None
    if not args.no_describe:
        pool = await _make_pool(args)
        if pool is None:
            return 3
    else:
        print("# --no-describe: deterministic analysis only, description stays empty.")

    # --- output mode: sharded directory (10K/file) or single file ---
    sharded = bool(args.out_dir)
    shard_size = args.shard_size
    if sharded:
        out_dir = os.path.abspath(args.out_dir)
        prefix = args.shard_prefix or os.path.splitext(os.path.basename(args.input))[0]
        nshards = (n + shard_size - 1) // shard_size
    else:
        out_dir = os.path.dirname(os.path.abspath(args.output)) or "."
        prefix, nshards = None, 0
    os.makedirs(out_dir, exist_ok=True)
    if getattr(args, "copy_readmes", True):
        # Sharded dataset runs carry their schema documentation beside the output.
        _here = os.path.dirname(os.path.abspath(__file__))
        for _src, _dst in (("DATASET_README.md", "README.md"),
                           ("DATASET_README.en.md", "README.en.md")):
            try:
                sp = os.path.join(_here, _src)
                if os.path.exists(sp):
                    import shutil
                    shutil.copyfile(sp, os.path.join(out_dir, _dst))
            except OSError:
                pass

    # Resume: skip the leading shards that are already complete.
    next_shard = _scan_complete_shards(out_dir, prefix, shard_size, n) if sharded else 0
    resume_off = min(next_shard * shard_size, n) if sharded else 0
    if resume_off:
        print(f"# Resuming: {next_shard} complete shard(s) present -> skip first {resume_off} row(s).")

    # --- streaming queue: rows flow through analyse -> polish -> ordered shard flush ---
    n_slots = (pool.n_servers * args.concurrency_per_server) if pool is not None \
        else max(8, (args.analyzer_procs or min(32, os.cpu_count() or 8)))
    queue: asyncio.Queue = asyncio.Queue(maxsize=n_slots * 3)
    _DONE = object()
    loop = asyncio.get_event_loop()
    pbar = tqdm(total=n - resume_off, desc="Describe", unit="mol", smoothing=0.02)
    stats = {"described": 0, "no_scaffold": 0, "analysis_fail": 0, "verified": 0,
             "with_violations": 0}

    results: list = [None] * n if not sharded else []   # full buffer only in single-file mode
    pending: dict = {}                                   # sharded: idx -> serialised JSON line, not yet flushed
    shard_done: dict = defaultdict(int)                  # sharded: shard -> rows completed
    flush_lock = asyncio.Lock()

    def _write_shard(tmp: str, dst: str, lines: list) -> None:
        """Run on a thread: join + file write. File I/O releases the GIL, so the event
        loop keeps running."""
        with open(tmp, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        os.replace(tmp, dst)

    async def _flush_shards() -> None:
        """Once the leading shards are contiguously full, write them in input order,
        atomically (.tmp then rename).

        Serialisation (json.dumps) already happened, spread out, as each row was stored;
        only the join and the write are handed to a thread here, so the event loop —
        which dispatches requests to every server — never stalls."""
        nonlocal next_shard
        async with flush_lock:
            while next_shard < nshards:
                s = next_shard * shard_size
                e = min(s + shard_size, n)
                if shard_done.get(next_shard, 0) < (e - s):
                    break
                p = os.path.join(out_dir, f"{prefix}-{next_shard:05d}.jsonl")
                lines = [pending.pop(i) for i in range(s, e)]   # pop on the event loop: a cheap reference move
                shard_done.pop(next_shard, None)
                await loop.run_in_executor(None, _write_shard, p + ".tmp", p, lines)
                pbar.write(f"# shard {next_shard:05d} written ({e - s} rows) -> {os.path.basename(p)}")
                next_shard += 1

    async def _store(idx: int, out: dict) -> None:
        if sharded:
            # Serialising per row (~0.05 ms) removes the burst where a flush used to dumps
            # 20,000 rows at once and froze the event loop for ~1 s — which is what dropped
            # every GPU to 0% at that moment.
            pending[idx] = json.dumps(out, ensure_ascii=False)
            shard_done[idx // shard_size] += 1
            await _flush_shards()
        else:
            results[idx] = out

    async def consumer(server_idx: int) -> None:
        while True:
            payload = await queue.get()
            if payload is _DONE:
                return
            idx, an, it = payload
            if pool is not None:
                it = await polish(pool, server_idx, an, it, args.validate_passes)
            # merge: original record first (all keys preserved), then scaffold fields
            rec = recs[idx]
            out = {k: v for k, v in rec.items() if k != "_row"}
            out.update(it)
            await _store(idx, out)
            if it.get("analysis_error"):
                stats["analysis_fail"] += 1
            elif not it.get("has_scaffold"):
                stats["no_scaffold"] += 1
            else:
                stats["described"] += 1
                if it.get("scaffold_verified"):
                    stats["verified"] += 1
                if it.get("description_violations"):
                    stats["with_violations"] += 1
            pbar.update(1)

    consumers = ([asyncio.create_task(consumer(i))
                  for i in range(pool.n_servers) for _ in range(args.concurrency_per_server)]
                 if pool is not None
                 else [asyncio.create_task(consumer(0)) for _ in range(n_slots)])

    # --- analysis producer: the CPU work (analyze_one) runs on a process pool ---
    # This used to await one row at a time on the main event loop, capped at ~100 rows/s
    # (one core, GIL), so adding LLM servers starved here instead of helping. A
    # ProcessPoolExecutor spreads it over several cores and lifts the producer ceiling
    # above the LLM capacity.
    n_analyzers = args.analyzer_procs or min(32, os.cpu_count() or 8)
    ppe = ProcessPoolExecutor(max_workers=n_analyzers)
    idx_iter = iter(range(resume_off, n))

    # Move the long-lived objects built so far (recs = up to millions of records, the
    # pool, the clients) into the GC's permanent generation. Without this, the flood of
    # short-lived objects from the consumers triggers gen-2 collections that walk all of
    # recs every time (~0.7 s at 2M rows), freezing the event loop and dropping every GPU
    # to 0% at that moment. After the freeze that scan costs essentially nothing.
    gc.collect()
    gc.freeze()

    async def feeder() -> None:
        """Take the next row index, analyse it on the process pool, and put the result on
        the queue in order.

        next(idx_iter) is only ever called between awaits, so on a single-threaded event
        loop it is race-free."""
        while True:
            try:
                idx = next(idx_iter)
            except StopIteration:
                return
            smi = str(recs[idx].get(smiles_key) or "")
            an, it = await loop.run_in_executor(ppe, analyze_one, smi)
            await queue.put((idx, an, it))

    try:
        feeders = [asyncio.create_task(feeder()) for _ in range(n_analyzers * 2)]
        await asyncio.gather(*feeders)
        for _ in consumers:
            await queue.put(_DONE)
        await asyncio.gather(*consumers)
    finally:
        ppe.shutdown(wait=False)
        pbar.close()

    # --- final write ---
    written = 0
    if sharded:
        await _flush_shards()
        for k in range(nshards):
            p = os.path.join(out_dir, f"{prefix}-{k:05d}.jsonl")
            if os.path.exists(p):
                written += sum(1 for _ in open(p))
        dest = f"{out_dir}/{prefix}-NNNNN.jsonl  ({nshards} shard(s), {shard_size}/shard)"
    else:
        with open(args.output, "w") as fh:
            for out in results:
                if out is None:
                    continue
                fh.write(json.dumps(out, ensure_ascii=False) + "\n")
                written += 1
        dest = args.output

    print(f"\n# Done. rows_in={n} rows_out={written} (this run: "
          f"described={stats['described']} verified={stats['verified']} "
          f"no_scaffold={stats['no_scaffold']} analysis_fail={stats['analysis_fail']} "
          f"remaining_violations={stats['with_violations']})")
    print(f"# Wrote: {dest}")
    return 0


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="input JSONL, one record per line.")
    # Output: either a single file (--output) or sharded into a directory (--out-dir).
    ap.add_argument("--output", default=None, help="single output JSONL path; everything goes in one file.")
    ap.add_argument("--out-dir", default=None,
                    help="shard output directory, created if missing. Results are split into "
                         "--shard-size chunks as `<prefix>-00000.jsonl`, `-00001.jsonl`, ... "
                         "Input order is preserved and the run is resumable.")
    ap.add_argument("--shard-size", type=int, default=10000, help="records per shard (default 10000).")
    ap.add_argument("--shard-prefix", default=None,
                    help="shard filename prefix (default: the input filename stem).")
    ap.add_argument("--smiles-key", default="ref_smiles", help="the key holding the SMILES (default ref_smiles).")
    ap.add_argument("--servers", default=DEFAULT_SERVERS, help="vLLM endpoints as comma-separated 'host:portspec'.")
    ap.add_argument("--host", default=None)
    ap.add_argument("--base-ports", default="8080-8087")
    ap.add_argument("--concurrency-per-server", type=int, default=5)
    ap.add_argument("--analyzer-procs", type=int, default=0,
                    help="worker processes for analyze_one (CPU). 0 = auto (min(32, cpu_count)).")
    ap.add_argument("--model", default=None)
    ap.add_argument("--reasoning-effort", default="low")
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--max-sentences", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--max-retries", type=int, default=4)
    ap.add_argument("--validate-passes", type=int, default=3)
    ap.add_argument("--no-describe", action="store_true",
                    help="skip the LLM description (no vLLM connection). Fills the deterministic "
                         "structure fields only and leaves description as an empty string.")
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    if bool(args.output) == bool(args.out_dir):
        sys.exit("error: give exactly one of --output (single file) or --out-dir (sharded folder).")
    if args.out_dir and args.shard_size <= 0:
        sys.exit("error: --shard-size must be > 0.")
    rc = asyncio.run(main_async(args))
    sys.exit(rc)


if __name__ == "__main__":
    main()
