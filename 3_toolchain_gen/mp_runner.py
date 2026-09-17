"""Multiprocess (server-free) tool-chain building.

The HTTP path scales by fanning out to many tool servers; the in-process path
(:mod:`local_tools`) removes the servers but a single Python process is
GIL-bound (~1.8 inst/s). This module gives the in-process path its parallelism:
``num_procs`` worker processes, each with its own ADMET model, each building a
disjoint round-robin shard of the records.

Why this shape
--------------
Benchmarking showed the workload is **CPU-bound**, not GPU-bound — the heavy
cost is RDKit featurisation inside ``analyze_properties`` plus the
RDKit planning/edit tools; the ADMET neural net on the GPU is a short burst
(one GPU sat ~50% utilised while all cores were pegged). So:

* throughput scales with **process count** until the CPU cores saturate — peak
  at roughly ``num_procs ≈ cores / 6`` (e.g. ~40 processes on ~250 cores),
  above which oversubscription *reduces* throughput;
* one GPU is plenty for the ADMET bursts of a whole node's worth of processes;
  ``--gpus`` round-robins processes over the listed devices only for memory/
  scheduling headroom, not for throughput;
* per-process async concurrency barely helps (the model is serial within a
  process), so keep ``per_proc`` low (2–4) and let processes be the unit of
  parallelism.

Output ordering
---------------
Work is sharded by PART, not by instance: part *p* is the contiguous block of
instances ``[p*PART_SIZE, (p+1)*PART_SIZE)``, and the parent deals the outstanding
parts to workers in contiguous blocks (:func:`_deal_parts`). Each worker builds its
parts in ascending index and writes each — in original instance order — to a
single global-index file
``toolchains_generation_chunk_<cccccc>.jsonl``. So reading the chunk files in
sorted order reproduces the **original instance order** exactly, regardless of
which worker built a chunk or in what order chunks finished. Chunks are written
atomically (``.tmp`` then ``os.replace``), so ``--resume`` simply skips chunk
files that already exist, and a crash never leaves a partial chunk.
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

logger = logging.getLogger(__name__)


# Output layout — two levels, so the final file size (CHUNK_SIZE, what the user
# wants: N records per file) is DECOUPLED from the parallel work unit (PART_SIZE):
#   * PART_SIZE = contiguous instances per *part* — the unit of work handed to a
#     worker AND the granularity of incremental, crash-safe writes during the run.
#     Small -> many parts -> even load balancing across many workers + frequent
#     flushes.
#   * CHUNK_SIZE = instances per *final* file. After the build, PART_SIZE-sized
#     parts are concatenated (in order) into CHUNK_SIZE-sized final files, so
#     R = CHUNK_SIZE // PART_SIZE consecutive parts make one final file.
# During the run only part files exist; the final chunk files appear at the merge
# step. Reading final files sorted reproduces the original instance order.

def _part_name(part_idx: int) -> str:
    return f"toolchains_generation_part_{part_idx:07d}.jsonl"


def _chunk_name(chunk_idx: int) -> str:
    """Global, zero-padded FINAL filename — sorts into original instance order."""
    return f"toolchains_generation_chunk_{chunk_idx:06d}.jsonl"


def _existing_parts(out_dir: Path) -> set:
    """Part indices whose part file already exists."""
    done = set()
    for path in out_dir.glob("toolchains_generation_part_*.jsonl"):
        try:
            done.add(int(path.stem.rsplit("_", 1)[1]))
        except (ValueError, IndexError):
            pass
    return done


def _existing_chunks(out_dir: Path) -> set:
    """Final chunk indices already merged (their source parts are gone)."""
    done = set()
    for path in out_dir.glob("toolchains_generation_chunk_*.jsonl"):
        try:
            done.add(int(path.stem.rsplit("_", 1)[1]))
        except (ValueError, IndexError):
            pass
    return done


def _done_parts(out_dir: Path, parts_per_chunk: int) -> set:
    """Parts already accounted for (for --resume): those with a part file, plus
    those already folded into a merged final chunk (part p -> chunk p // R)."""
    done = set(_existing_parts(out_dir))
    for c in _existing_chunks(out_dir):
        done.update(range(c * parts_per_chunk, (c + 1) * parts_per_chunk))
    return done


def _count_records(fp) -> int:
    """Fast newline count for a JSONL file (records == lines; no blank lines)."""
    n = 0
    with open(fp, "rb") as f:
        while True:
            buf = f.read(1 << 22)
            if not buf:
                break
            n += buf.count(b"\n")
    return n


def _file_plan(input_path: str, limit: Optional[int]) -> tuple[list, int]:
    """One-pass plan of the input files: ``[(path, start_global_index), ...]`` and
    the total record count.

    Counting once here (in the parent, files counted concurrently) lets each worker
    read ONLY the files assigned to it — instead of every worker streaming the whole
    dataset to find its scattered parts, which on a network FS (GPFS) meant
    num_procs × 12 GB of reads and a multi-minute stall before any building started.
    """
    from concurrent.futures import ThreadPoolExecutor

    p = Path(input_path)
    files = sorted(p.glob("*.jsonl")) if p.is_dir() else [p]
    with ThreadPoolExecutor(max_workers=min(32, len(files) or 1)) as ex:
        counts = list(ex.map(_count_records, files))
    plan: list = []
    total = 0
    for fp, n in zip(files, counts):
        if limit is not None and total >= limit:
            break
        plan.append((str(fp), total))
        total += n
    return plan, (min(total, limit) if limit is not None else total)


def _deal_parts(remaining: list, num_procs: int) -> list:
    """Split the *remaining* part indices into ``num_procs`` CONTIGUOUS, near-equal
    blocks — one work assignment per worker.

    Contiguous (not round-robin) on purpose: it balances by part count *and* keeps
    each worker's reads local to the one or two input files its block spans, which
    is the property the old per-file assignment was protecting (a worker must never
    stream the whole dataset off GPFS). Empty blocks are dropped.
    """
    n = len(remaining)
    if n == 0 or num_procs <= 0:
        return []
    k = min(num_procs, n)
    base, extra = divmod(n, k)
    blocks, i = [], 0
    for w in range(k):
        size = base + (1 if w < extra else 0)
        blocks.append(remaining[i:i + size])
        i += size
    return blocks


def _load_worker_parts(file_plan: list, my_parts: list, part_size: int,
                       limit: Optional[int], end_gi: int) -> dict:
    """Read ONLY the input files spanning this worker's assigned parts and group
    their records into global PART_SIZE blocks.

    ``my_parts`` is a contiguous block of part indices (see :func:`_deal_parts`), so
    the records it needs form the contiguous global range
    ``[min*part_size, (max+1)*part_size)`` and only the files overlapping that range
    are opened. Global index ``gi = file_start + local_line`` (from
    :func:`_file_plan`) keeps part indices — and therefore the merged chunk order —
    identical to the original instance order.
    Returns ``{part_idx: [(gi, record), ...]}``.
    """
    import json

    if not my_parts:
        return {}
    want = set(my_parts)
    lo = min(my_parts) * part_size
    hi = (max(my_parts) + 1) * part_size          # exclusive
    if limit is not None:
        hi = min(hi, limit)

    parts: dict = {}
    for fi, (fp, start_gi) in enumerate(file_plan):
        file_end = file_plan[fi + 1][1] if fi + 1 < len(file_plan) else end_gi
        if file_end <= lo or start_gi >= hi:      # no overlap with this worker's range
            continue
        gi = start_gi
        with open(fp) as fh:
            for line in fh:
                if gi >= hi:
                    break
                if gi >= lo:
                    line = line.strip()
                    if line and (gi // part_size) in want:
                        parts.setdefault(gi // part_size, []).append((gi, json.loads(line)))
                gi += 1
    return parts


async def _build_worker_parts(
    builder, parts: dict, out_dir: Path, per_proc: int, progress=None,
) -> tuple[int, int]:
    """Build this worker's parts and write each, in original instance order, to its
    part file the moment it completes (atomic ``.tmp`` -> ``os.replace``).

    Within a part ``asyncio.gather`` preserves order, so record j is the build of
    the j-th instance. Parts are written even if empty (marks them done for
    --resume). Only one part's results are held at a time.
    """
    import json

    sem = asyncio.Semaphore(max(1, per_proc))

    async def one(idx, rec):
        async with sem:
            try:
                return await builder.build(rec, idx)
            except Exception as exc:  # pragma: no cover - per-instance guard
                logger.exception("instance %d failed: %s", idx, exc)
                return None
            finally:
                if progress is not None:
                    with progress.get_lock():
                        progress.value += 1

    success = total = 0
    for pidx in sorted(parts):
        items = parts[pidx]
        results = await asyncio.gather(*(one(gi, rec) for gi, rec in items))
        kept = [r for r in results if r is not None]   # order preserved
        tmp = out_dir / (_part_name(pidx) + ".tmp")
        with open(tmp, "w") as fh:
            for gi in kept:
                fh.write(json.dumps(gi.model_dump(), ensure_ascii=False) + "\n")
        os.replace(tmp, out_dir / _part_name(pidx))     # atomic publish
        total += len(kept)
        success += sum(1 for r in kept if r.metadata.get("success"))
    return success, total


def _merge_parts(out_dir: Path, parts_per_chunk: int) -> int:
    """Concatenate PART_SIZE parts into CHUNK_SIZE final files, in order.

    Called once after the build completes (so every part that should exist does).
    Final chunk *c* = parts ``[c*R, (c+1)*R)`` concatenated in index order → the
    original instance order. Written atomically, then the source parts are deleted.
    Idempotent: skips chunks whose final file already exists (and cleans up any
    leftover parts for them), so an interrupted merge resumes cleanly. Returns the
    number of final files present.
    """
    parts = _existing_parts(out_dir)
    if not parts:
        return len(_existing_chunks(out_dir))
    max_chunk = max(parts) // parts_per_chunk
    for c in range(max_chunk + 1):
        final = out_dir / _chunk_name(c)
        member_parts = [p for p in range(c * parts_per_chunk, (c + 1) * parts_per_chunk)
                        if (out_dir / _part_name(p)).exists()]
        if final.exists():
            for p in member_parts:      # already merged -> drop orphaned parts
                (out_dir / _part_name(p)).unlink(missing_ok=True)
            continue
        if not member_parts:
            continue
        tmp = out_dir / (_chunk_name(c) + ".tmp")
        with open(tmp, "w") as out:
            for p in member_parts:
                with open(out_dir / _part_name(p)) as fh:
                    for line in fh:
                        out.write(line)
        os.replace(tmp, final)          # atomic publish of the final file
        for p in member_parts:
            (out_dir / _part_name(p)).unlink(missing_ok=True)
        logger.info("merged %d parts → %s", len(member_parts), final.name)
    return len(_existing_chunks(out_dir))


def _terminate_workers(procs: list) -> None:
    """Tear down all worker processes: SIGTERM, brief grace, then SIGKILL any still
    alive (a worker mid-GPU-call may not honour SIGTERM at once). Idempotent — safe
    on both Ctrl-C and normal completion."""
    for p in procs:
        if p.is_alive():
            p.terminate()
    for p in procs:
        p.join(timeout=5)
    for p in procs:
        if p.is_alive():
            p.kill()
    for p in procs:
        p.join(timeout=5)


def _worker_main(worker_id, my_parts, gpus, args_dict, file_plan,
                 out_dir_str, limit, part_size, end_gi, per_proc, ret_q,
                 progress=None, size_q=None):
    # Ignore Ctrl-C in workers: the PARENT catches SIGINT and tears workers down
    # cleanly (SIGTERM/SIGKILL). Without this, Ctrl-C to the process group leaves
    # workers running (each in its own asyncio/GPU state) after the parent exits.
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Cap BLAS / OpenMP threads to 1 per worker BEFORE numpy/torch/rdkit import.
    # With many workers, uncapped OpenBLAS/MKL spawn one thread PER CORE each, so N
    # workers × ~cores threads massively oversubscribe the CPU and thrash — measured
    # (2026-07-24): 64 uncapped workers failed to even finish startup, while capped
    # they ran fine (~8-10 inst/s). One BLAS thread/worker + workers-as-the-unit-of-
    # parallelism is the correct config for this CPU-bound workload. Honour any
    # caller-provided override. MALLOC_ARENA_MAX bounds glibc arena RSS under many procs.
    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(_v, "1")
    os.environ.setdefault("MALLOC_ARENA_MAX", "2")

    # Pin the GPU BEFORE anything imports torch (molkit pulls it in lazily).
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[worker_id % len(gpus)])
    # Keep the CUDA caching allocator from fragmenting over a long (2M) run: beam
    # measures variable-size ADMET batches, so the allocator otherwise reserves
    # for the largest and creeps upward. expandable_segments reclaims cleanly and
    # bounds per-worker GPU memory (the slow "GPU mem keeps rising" symptom). Must
    # be set before torch is imported; honour any caller-provided value.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    from . import http_client, local_tools
    from .builder import ToolChainBuilder

    http_client.set_local_mode(True)
    local_tools.preload(with_admet=True)

    out_dir = Path(out_dir_str)

    # The parent dealt out this worker's parts (already-done parts excluded for
    # --resume). Reading is scoped to the files those parts span, so a worker
    # touches a small slice of the input instead of scanning all of it.
    parts = _load_worker_parts(file_plan, my_parts, part_size, limit, end_gi)
    if size_q is not None:
        size_q.put(sum(len(v) for v in parts.values()))  # pending count for tqdm

    builder = ToolChainBuilder(args=SimpleNamespace(**args_dict))
    success, total = asyncio.run(
        _build_worker_parts(builder, parts, out_dir, per_proc, progress=progress))
    ret_q.put((worker_id, success, total))


def run_multiprocess(
    args, input_path: str, out_dir: str, limit: Optional[int],
    chunk_size: int, num_procs: int, per_proc: int, gpus: list[int],
    part_size: Optional[int] = None,
) -> tuple[int, int]:
    """Build tool chains across *num_procs* in-process workers.

    Returns ``(success, total)`` aggregated over workers. Work is sharded into
    PART_SIZE units (contiguous blocks per worker) for load balancing + incremental
    crash-safe writes; after the build, ``R = CHUNK_SIZE // PART_SIZE`` consecutive
    parts are merged, in order, into each final ``CHUNK_SIZE``-instance file. So
    the output is CHUNK_SIZE records per file, in original instance order, while
    all workers stay busy and results stream to disk throughout the run.
    """
    import queue as _queue
    import time as _time

    # PART_SIZE (work unit) is decoupled from CHUNK_SIZE (final file size). Default
    # to ~cover the dataset in many small parts; cap at CHUNK_SIZE and make
    # CHUNK_SIZE an integer multiple of it so parts concatenate cleanly into files.
    if not part_size or part_size <= 0:
        part_size = min(chunk_size, 1000)
    part_size = max(1, min(part_size, chunk_size))
    parts_per_chunk = max(1, chunk_size // part_size)
    eff_chunk = parts_per_chunk * part_size   # effective final size (multiple of part)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    size_q = ctx.Queue()
    progress = ctx.Value("L", 0)      # instances finished, incremented by workers
    args_dict = vars(args) if not isinstance(args, dict) else dict(args)

    # Plan the input files ONCE (parent), so each worker reads only its assigned
    # files instead of every worker scanning the whole dataset.
    logger.info("planning input files...")
    file_plan, n_total = _file_plan(input_path, limit)
    n_files = len(file_plan)

    # Deal the OUTSTANDING parts across workers. Work used to be assigned per input
    # FILE, which left a brutal tail: once every worker had finished its one file,
    # the run collapsed onto the 1-2 workers that happened to own a second file
    # (measured 2026-07-31 on the 1M segment1 run: 16.7 inst/s for 96% of the run,
    # then ~0.7 inst/s for the last 40 parts — 16 h for 4% of the data). Dealing
    # parts instead means the tail is bounded by ONE part (part_size instances) on
    # the slowest worker, and --resume redistributes leftovers over all workers.
    n_parts = (n_total + part_size - 1) // part_size
    done_parts = _done_parts(Path(out_dir), parts_per_chunk) if args_dict.get("resume") else set()
    remaining = [p for p in range(n_parts) if p not in done_parts]
    blocks = _deal_parts(remaining, num_procs)
    if not blocks:
        logger.info("nothing to build: all %d parts already done", n_parts)
        n_final = _merge_parts(Path(out_dir), parts_per_chunk)
        logger.info("merge done: %d final chunk files", n_final)
        return 0, 0
    if len(blocks) < num_procs:
        logger.info("only %d parts outstanding: spawning %d workers instead of %d",
                    len(remaining), len(blocks), num_procs)
        num_procs = len(blocks)

    logger.info(
        "multiprocess build: %d workers, per_proc=%d, gpus=%s | %d files, %d records, "
        "part_size=%d, %d/%d parts outstanding (~%d parts/worker), final chunk=%d "
        "(%d parts/file) → %s",
        num_procs, per_proc, gpus, n_files, n_total, part_size, len(remaining),
        n_parts, max(len(b) for b in blocks), eff_chunk, parts_per_chunk, out_dir)

    # Ctrl-C / kill handling: workers ignore SIGINT (set in _worker_main); the
    # parent catches SIGINT (KeyboardInterrupt) and SIGTERM and tears every worker
    # down (SIGTERM→SIGKILL) so none are left orphaned. Restored in finally.
    import signal
    _prev_term = signal.getsignal(signal.SIGTERM)

    def _on_sigterm(_sig, _frm):
        raise KeyboardInterrupt
    try:
        signal.signal(signal.SIGTERM, _on_sigterm)
    except (ValueError, OSError):  # not main thread — best effort
        pass

    procs: list = []
    results: list = []
    bar = None
    interrupted = False
    try:
        for w in range(num_procs):
            p = ctx.Process(
                target=_worker_main,
                args=(w, blocks[w], gpus, args_dict, file_plan, out_dir,
                      limit, part_size, n_total, per_proc, q, progress, size_q),
                daemon=False,
            )
            p.start()
            procs.append(p)

        # Workers load only their assigned files, then report their pending count —
        # the sum gives the tqdm total.
        logger.info("workers loading assigned files...")
        total_pending = sum(size_q.get() for _ in range(num_procs))

        # Single aggregate progress bar driven by the shared counter; drain results
        # as workers finish (a multiprocessing.Queue must be emptied before join()).
        try:
            from tqdm import tqdm
            bar = tqdm(total=total_pending, desc="tool chains", unit="mol",
                       smoothing=0.05, dynamic_ncols=True)
        except Exception:  # pragma: no cover - tqdm always present, defensive
            bar = None
        while len(results) < num_procs:
            try:
                while True:
                    results.append(q.get_nowait())
            except _queue.Empty:
                pass
            if bar is not None:
                bar.n = min(int(progress.value), total_pending)
                bar.refresh()
            if len(results) < num_procs:
                _time.sleep(0.5)
        if bar is not None:
            bar.n = min(int(progress.value), total_pending)
            bar.refresh()

        for p in procs:
            p.join()

        # Build complete: fold PART_SIZE parts into CHUNK_SIZE final files, in order.
        logger.info("merging parts into %d-instance chunk files...", eff_chunk)
        n_final = _merge_parts(Path(out_dir), parts_per_chunk)
        logger.info("merge done: %d final chunk files", n_final)
    except KeyboardInterrupt:
        interrupted = True
        logger.warning("interrupted — terminating %d worker processes...", len(procs))
        raise
    finally:
        if bar is not None:
            bar.close()
        _terminate_workers(procs)      # idempotent: cleans up on interrupt AND normal exit
        try:
            signal.signal(signal.SIGTERM, _prev_term)
        except (ValueError, OSError):
            pass
        if interrupted:
            # Parts already flushed are kept; --resume picks up from them next run.
            logger.warning("workers terminated. Re-run with --resume to continue.")

    success = sum(r[1] for r in results)
    total = sum(r[2] for r in results)
    return success, total
