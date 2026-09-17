#!/usr/bin/env python3
"""Re-chunk a directory of JSONL shards into larger (or smaller) fixed-size chunks.

Reads every ``*.jsonl`` in --in-dir (sorted), concatenates the records in order,
and writes them back out in chunks of --records-per-chunk, preserving the
``<prefix>chunk_NNNN.jsonl`` naming. Verifies the total record count is preserved.

In-place (--in-place) rewrites the same directory atomically: it writes the new
chunks to a temp dir, checks the count matches, then swaps and deletes the old
shards. Otherwise it writes to --out-dir.

Usage:
    # 1k -> 10k, in place:
    python 3_toolchain_gen/rechunk_jsonl.py \
        --in-dir /data/.../toolchain/generation_200k_scaffold_noref \
        --records-per-chunk 10000 --in-place
    # to a new dir:
    python 3_toolchain_gen/rechunk_jsonl.py --in-dir <src> --out-dir <dst> -n 10000
"""
import argparse
import glob
import os
import re
import shutil


def _chunk_prefix(paths):
    """Infer the '<prefix>chunk_' filename stem from the first shard."""
    base = os.path.basename(paths[0])
    m = re.match(r"(.*chunk_)\d+\.jsonl$", base)
    return m.group(1) if m else "toolchains_generation_chunk_"


def _shards(d):
    return sorted(glob.glob(os.path.join(d, "*.jsonl")))


def _count(d):
    n = 0
    for f in _shards(d):
        with open(f) as fh:
            for line in fh:
                if line.strip():
                    n += 1
    return n


def rechunk(in_dir, out_dir, per_chunk, prefix):
    os.makedirs(out_dir, exist_ok=True)
    written = idx = 0
    fo = None

    def _open(i):
        return open(os.path.join(out_dir, f"{prefix}{i:04d}.jsonl"), "w")

    for f in _shards(in_dir):
        with open(f) as fh:
            for line in fh:
                if not line.strip():
                    continue
                if fo is None or written == per_chunk:
                    if fo is not None:
                        fo.close(); idx += 1
                    fo = _open(idx); written = 0
                fo.write(line if line.endswith("\n") else line + "\n")
                written += 1
    if fo is not None:
        fo.close()
    return idx + 1 if fo is not None else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("-n", "--records-per-chunk", type=int, default=10000)
    ap.add_argument("--in-place", action="store_true")
    args = ap.parse_args()

    shards = _shards(args.in_dir)
    if not shards:
        raise SystemExit(f"no *.jsonl shards in {args.in_dir}")
    prefix = _chunk_prefix(shards)
    total_in = _count(args.in_dir)

    if args.in_place:
        tmp = args.in_dir.rstrip("/") + ".rechunk_tmp"
        if os.path.exists(tmp):
            shutil.rmtree(tmp)
        n_chunks = rechunk(args.in_dir, tmp, args.records_per_chunk, prefix)
        total_out = _count(tmp)
        if total_out != total_in:
            shutil.rmtree(tmp)
            raise SystemExit(f"ABORT: record count changed {total_in} -> {total_out}; kept original.")
        # swap: remove old shards, move new chunks in
        for f in _shards(args.in_dir):
            os.remove(f)
        for f in _shards(tmp):
            shutil.move(f, os.path.join(args.in_dir, os.path.basename(f)))
        shutil.rmtree(tmp)
        print(f"[in-place] {args.in_dir}: {len(shards)} shards -> {n_chunks} chunks "
              f"of {args.records_per_chunk} ({total_in} records preserved)")
    else:
        if not args.out_dir:
            raise SystemExit("--out-dir required unless --in-place")
        n_chunks = rechunk(args.in_dir, args.out_dir, args.records_per_chunk, prefix)
        total_out = _count(args.out_dir)
        status = "OK" if total_out == total_in else f"WARNING count {total_in}->{total_out}"
        print(f"{args.in_dir} ({len(shards)} shards, {total_in} rec) -> {args.out_dir} "
              f"({n_chunks} chunks of {args.records_per_chunk}) [{status}]")


if __name__ == "__main__":
    main()
