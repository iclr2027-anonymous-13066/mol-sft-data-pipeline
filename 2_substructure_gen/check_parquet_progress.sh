#!/usr/bin/env bash
set -u

PYTHON="${PYTHON:-python}"
INPUT="${INPUT:-data/benchmark/train_pool_2m.parquet}"
SAMPLES=2
WATCH_SECONDS=0

usage() {
    echo "Usage: $0 [--samples N] [--watch SECONDS] [--input INPUT.parquet]"
    echo
    echo "  One-time check:  $0"
    echo "  Background watch: nohup $0 --watch 60 > parquet_progress.log 2>&1 &"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --samples)
            [ "$#" -ge 2 ] || { echo "error: --samples requires a value" >&2; exit 2; }
            SAMPLES="$2"
            shift 2
            ;;
        --watch)
            [ "$#" -ge 2 ] || { echo "error: --watch requires seconds" >&2; exit 2; }
            WATCH_SECONDS="$2"
            shift 2
            ;;
        --input)
            [ "$#" -ge 2 ] || { echo "error: --input requires a path" >&2; exit 2; }
            INPUT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "error: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

case "$SAMPLES" in
    ''|*[!0-9]*) echo "error: --samples must be a positive integer" >&2; exit 2 ;;
esac
[ "$SAMPLES" -gt 0 ] || { echo "error: --samples must be greater than zero" >&2; exit 2; }

case "$WATCH_SECONDS" in
    ''|*[!0-9]*) echo "error: --watch must be a non-negative integer" >&2; exit 2 ;;
esac

[ -x "$PYTHON" ] || { echo "error: Python not executable: $PYTHON" >&2; exit 1; }

inspect_once() {
    "$PYTHON" - "$INPUT" "$SAMPLES" <<'PY'
import glob
import json
import os
import sys
from datetime import datetime

import pyarrow.parquet as pq

input_path = sys.argv[1]
sample_count = int(sys.argv[2])
candidate_files = glob.glob("/tmp/substructure_parquet_*/part-*.parquet")

print(f"[{datetime.now().isoformat(timespec='seconds')}]")
if not candidate_files:
    print("No intermediate Parquet chunks found under /tmp/substructure_parquet_*.")
    raise SystemExit(1)

newest = max(candidate_files, key=os.path.getmtime)
work_dir = os.path.dirname(newest)
part_paths = sorted(glob.glob(os.path.join(work_dir, "part-*.parquet")))

readable = []
completed_rows = 0
for path in part_paths:
    try:
        metadata = pq.ParquetFile(path).metadata
    except Exception:
        continue
    readable.append(path)
    completed_rows += metadata.num_rows

if not readable:
    print(f"Temporary directory: {work_dir}")
    print("No fully readable chunk yet; the first batch may still be writing.")
    raise SystemExit(1)

total_rows = None
try:
    total_rows = pq.ParquetFile(input_path).metadata.num_rows
except Exception:
    pass

latest_path = readable[-1]
rows = pq.read_table(latest_path).to_pylist()

empty_descriptions = sum(not str(row.get("description") or "").strip() for row in rows)
parse_failures = sum(row.get("parse_ok") is False for row in rows)
violations = sum(bool(row.get("description_violations")) for row in rows)
analysis_errors = sum(bool(row.get("analysis_error")) for row in rows)

print(f"Temporary directory: {work_dir}")
print(f"Latest readable chunk: {os.path.basename(latest_path)}")
if total_rows:
    pct = completed_rows * 100.0 / total_rows
    print(
        f"Completed rows on disk: {completed_rows:,} / {total_rows:,} "
        f"({pct:.3f}%, {len(readable)} chunks)"
    )
else:
    print(f"Completed rows on disk: {completed_rows:,} ({len(readable)} chunks)")
print(
    f"Latest chunk checks ({len(rows):,} rows): "
    f"empty_description={empty_descriptions}, parse_failure={parse_failures}, "
    f"description_violations={violations}, analysis_error={analysis_errors}"
)
print(f"Top-level keys ({len(rows[0])}): {', '.join(rows[0].keys())}")

for index, row in enumerate(rows[-sample_count:], 1):
    print(f"\n--- sample {index} ---")
    print(json.dumps(row, ensure_ascii=False, indent=2, default=str))
PY
}

if [ "$WATCH_SECONDS" -eq 0 ]; then
    inspect_once
    exit $?
fi

[ "$WATCH_SECONDS" -gt 0 ] || { echo "error: --watch must be greater than zero" >&2; exit 2; }
while true; do
    inspect_once || true
    echo
    sleep "$WATCH_SECONDS"
done
