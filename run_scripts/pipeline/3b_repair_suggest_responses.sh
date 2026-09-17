#!/usr/bin/env bash
# run_scripts/pipeline/3b_repair_suggest_responses.sh
#
# Recompute the ``suggest_edits`` responses stored in FINISHED tool-chain files
# (3_toolchain_gen/repair_suggest_responses.py), across every chunk file of
# every chain set, with --num-proc 100.
#
# NOT NEEDED FOR A FRESH RUN. This is a one-off BACKFILL for chain sets generated
# BEFORE suggest_edits._measure was fixed. Stage 3 run with the current tool records
# correct responses to begin with, so on freshly generated 2M chains this pass
# rewrites nothing — run it only to migrate an older chain set, or as a no-op audit.
#
# Why: stage 3 recorded each suggest step's expected_response by re-calling the
# live tool WITHOUT the properties the planner had already measured, so the tool
# re-measured the molecule itself — and its MW was the AVERAGE mass while the rest
# of the pipeline (analyze_properties, the constraint boxes, the mmpdb Δ table)
# uses the monoisotopic mass. Near a target-box boundary that re-ranked the
# candidates, so the committed edit could drop out of the recorded top_k list
# (~0.3% of edit rounds) or the list could come back empty (~0.2% of chains).
# suggest_edits._measure is fixed now; this pass rewrites the responses of chains
# generated before the fix so the whole dataset is uniform.
#
# The chains themselves do NOT change — molecules, edits, checkpoints and
# constraint verdicts are untouched, because the planner passed its own measured
# props and never went through the buggy path. So NO stage-3 re-run is needed.
#
# PREREQUISITE: stage 3 must be FINISHED, including its part→chunk merge. This
# script rewrites the same files it reads, and only reads toolchains_*chunk_*.jsonl,
# so a directory that still holds toolchains_generation_part_*.jsonl is refused.
#
# Usage:
#   bash run_scripts/pipeline/3b_repair_suggest_responses.sh              # report only
#   APPLY=1 bash run_scripts/pipeline/3b_repair_suggest_responses.sh      # rewrite in place
#
# Env overrides:
#   INPUT_ROOT   tool-chain root (default data/training_data/toolchain)
#   BASE_NAMES   space-separated chain-set subdirs (default generation_2m_scaffold)
#   NUM_PROC     worker processes (default 100)
#   LIMIT        first N chunk files per set only (smoke test; default all)
#   OUT_ROOT     write repaired copies under here instead of in place
#
# Cost (measured): ~0.32 s per suggest step on one core; work is split per FILE, so
# a 20k-record chunk is ~2 h of single-core time and 100 chunks over 100 procs run
# in one wave (~2 h). Each worker holds its own copy of the mmpdb move index —
# ~1.1 GB RSS — so 100 procs needs ~110 GB of RAM free.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_DIR"

INPUT_ROOT="${INPUT_ROOT:-data/training_data/toolchain}"
BASE_NAMES="${BASE_NAMES:-generation_2m_scaffold}"
NUM_PROC="${NUM_PROC:-100}"
APPLY="${APPLY:-0}"          # 0 = report only (matches the usage note above)
LIMIT="${LIMIT:-}"
OUT_ROOT="${OUT_ROOT:-}"

if [ -z "${PYTHON:-}" ]; then
    for c in python \
             python; do
        [ -x "$c" ] && PYTHON="$c" && break
    done
    PYTHON="${PYTHON:-$(command -v python3)}"
fi

# ~1.1 GB of move index per worker: warn before the OOM killer finds out.
avail_gb="$(awk '/MemAvailable/ {printf "%d", $2/1024/1024}' /proc/meminfo 2>/dev/null || echo 0)"
need_gb=$(( NUM_PROC * 11 / 10 ))
echo "Using Python: $PYTHON"
echo "  workers   : $NUM_PROC (needs ~${need_gb} GB; ${avail_gb} GB available)"
echo "  mode      : $([ "$APPLY" = "1" ] && echo 'APPLY (rewriting files in place)' || echo 'report only')"
if [ "$avail_gb" -gt 0 ] && [ "$need_gb" -gt "$avail_gb" ]; then
    echo "[warn] ${NUM_PROC} workers want ~${need_gb} GB but only ${avail_gb} GB is available;" >&2
    echo "       lower NUM_PROC (each worker loads its own ~1.1 GB move index)." >&2
fi

for base in $BASE_NAMES; do
    dir="$INPUT_ROOT/$base"
    if [ ! -d "$dir" ]; then
        echo ""
        echo "[$base] not found, skipping: $dir"
        continue
    fi

    shopt -s nullglob
    chunks=("$dir"/toolchains_*chunk_*.jsonl)
    parts=("$dir"/toolchains_generation_part_*.jsonl)
    shopt -u nullglob

    echo ""
    if [ ${#chunks[@]} -eq 0 ]; then
        if [ ${#parts[@]} -gt 0 ]; then
            echo "[$base][error] ${#parts[@]} unmerged part file(s) and no chunk file(s) —" >&2
            echo "              stage 3 has not finished its part→chunk merge. Wait for it." >&2
            exit 1
        fi
        echo "[$base] no toolchains_*chunk_*.jsonl, skipping."
        continue
    fi
    if [ ${#parts[@]} -gt 0 ]; then
        echo "[$base][error] ${#chunks[@]} chunk file(s) mixed with ${#parts[@]} unmerged part" >&2
        echo "              file(s) — stage 3 is still running. Let it finish first." >&2
        exit 1
    fi

    echo "[$base] repairing ${#chunks[@]} chunk file(s) in $dir"
    cmd=("$PYTHON" "$PROJECT_DIR/3_toolchain_gen/repair_suggest_responses.py"
         --dir "$dir" --num-proc "$NUM_PROC")
    [ "$APPLY" = "1" ] && cmd+=(--apply)
    [ -n "$LIMIT" ] && cmd+=(--limit "$LIMIT")
    [ -n "$OUT_ROOT" ] && cmd+=(--out-dir "$OUT_ROOT/$base")
    "${cmd[@]}"
done

echo ""
if [ "$APPLY" = "1" ]; then
    echo "Done. suggest_edits responses recomputed."
    echo "Next: stage 4 (bash run_scripts/pipeline/4_sftdata_gen.sh)."
else
    echo "Report only. Re-run with APPLY=1 to rewrite the files."
fi
