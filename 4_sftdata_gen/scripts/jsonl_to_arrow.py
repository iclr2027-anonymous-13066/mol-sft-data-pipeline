#!/usr/bin/env python
"""Pre-convert a directory of *.jsonl SFT shards into one UNCOMPRESSED Arrow-IPC
sidecar so the DataModule can memory-map it zero-copy (see
data.dataset._ArrowSource) instead of parsing hundreds of JSONL files on every
run/rank. Loading is then instant and per-row JSON is decoded lazily.

Writes ``<dir>/_records.arrow`` with two columns:
  * ``json``     — large_binary, one raw record (the stripped JSONL line) per row
  * ``group_id`` — string, ``metadata.group_id`` (used for subsample + val split
                   without decoding the full record)

Uncompressed (so it can be mmap'd zero-copy) and written as a single record batch
(so the column is one chunk → O(1) row access). The training loader auto-detects
the sidecar and prefers it when it is newer than every .jsonl in the dir; re-run
this after regenerating the data.

Usage:
  python 4_sftdata_gen/scripts/jsonl_to_arrow.py \
    data/training_data/sftdata/generation_2m_scaffold \
    --num-proc 48
"""
import argparse
import glob
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor

import orjson
import pyarrow as pa
import pyarrow.feather as feather


def _convert_file(path):
    js, gids = [], []
    with open(path, "rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                gid = (orjson.loads(line).get("metadata") or {}).get("group_id")
            except Exception:
                continue  # malformed line — skip (matches loader)
            js.append(line)
            gids.append("" if gid is None else str(gid))
    return js, gids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", help="data dir(s) containing *.jsonl shards")
    ap.add_argument("--num-proc", type=int, default=min(48, os.cpu_count() or 8))
    ap.add_argument("--out-name", default="_records.arrow")
    args = ap.parse_args()

    for d in args.dirs:
        files = sorted(glob.glob(os.path.join(d, "*.jsonl")))
        if not files:
            print(f"[skip] no .jsonl in {d}")
            continue
        t = time.perf_counter()
        js_all, gid_all = [], []
        ctx = mp.get_context("fork")
        with ProcessPoolExecutor(max_workers=min(args.num_proc, len(files)),
                                 mp_context=ctx) as ex:
            for js, gids in ex.map(_convert_file, [str(f) for f in files]):
                js_all.extend(js)
                gid_all.extend(gids)
        tbl = pa.table({
            "json": pa.array(js_all, type=pa.large_binary()),
            "group_id": pa.array(gid_all, type=pa.string()),
        })
        out = os.path.join(d, args.out_name)
        tmp = out + ".tmp"
        # Uncompressed so the loader can memory-map it zero-copy.
        feather.write_feather(tbl, tmp, compression="uncompressed")
        os.replace(tmp, out)
        print(f"[ok] {d}: {len(js_all)} records -> {out} "
              f"({os.path.getsize(out) / 1e6:.0f} MB, {time.perf_counter() - t:.1f}s)")


if __name__ == "__main__":
    main()
