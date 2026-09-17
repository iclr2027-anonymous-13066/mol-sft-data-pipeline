#!/usr/bin/env bash
# run_scripts/pipeline/4a_filter_authored_chains.sh
#
# ONLY for the "+authored" arm. Shrink the stage-4 INPUT to the tool chains that
# will actually produce an authored round, before spending any GPU on them.
#
# Runs BEFORE 4_sftdata_gen.sh (it filters stage-3 output, not stage-4 output),
# which is why it sorts as 4a. The plain corpus does not need it.
#
# ── WHY ─────────────────────────────────────────────────────────────────────
# The authored corpus exists only to supply the delta: the segments rendered with
# an empty suggest_edits response. Measured with AUTHORED_EDIT_FRACTION=0.5 /
# AUTHORED_MAX_FAILING=2, just ~28% of chains contain even one such round, so a
# full second stage-4 pass spends ~72% of its wall clock generating records that
# 4e_make_authored_delta.sh then discards. On 2M chains that is days of GPU.
#
# Chains cannot be pruned any finer than this. A round's span quotes the per-round
# intent lines of earlier rounds, and those come from _populate_tool_reasons — an
# LLM call. So a surviving chain still needs all of its rounds generated; only
# chains that contribute nothing can go.
#
# ── IT REPRODUCES THE DECISION EXACTLY ──────────────────────────────────────
# The filter imports the pipeline's own build_segments / _out_of_range /
# _should_author and evaluates the same predicate, including the same
# deterministic md5(task_id|segment_index|authored) draw. Verified end to end on
# 80 chains: the filter predicted 26 authored rounds, the full unfiltered stage-4
# run produced 26, and stage 4 over the filtered input produced the same 26 while
# writing 89 records instead of 242.
#
# ── THE ONE WAY TO GET THIS WRONG ───────────────────────────────────────────
# FRACTION and MAX_FAILING here MUST equal the AUTHORED_EDIT_FRACTION and
# AUTHORED_MAX_FAILING you then pass to 4_sftdata_gen.sh. Different values and the
# chains kept are not the chains stage 4 selects — the delta silently shrinks with
# no error anywhere.
#
# Usage:
#   bash run_scripts/pipeline/4a_filter_authored_chains.sh                  # write
#   DRY_RUN=1 bash run_scripts/pipeline/4a_filter_authored_chains.sh        # count only
#
# Env overrides:
#   INPUT_ROOT   tool-chain root                (default …/training_data/toolchain)
#   BASE_NAMES   chain-set subdir to filter     (default generation_2m_scaffold)
#   OUT_SUFFIX   suffix of the filtered subdir  (default _authoredonly)
#   FRACTION     = AUTHORED_EDIT_FRACTION       (default 0.5)
#   MAX_FAILING  = AUTHORED_MAX_FAILING         (default 2)
#   NUM_PROC     worker processes, one shard each (default 16). Measured 4.2s
#                per 200MB/20k-chain shard on one core: 100 shards is ~7 min
#                serial, well under a minute at 16. Progress prints per shard.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_DIR"

INPUT_ROOT="${INPUT_ROOT:-data/training_data/toolchain}"
BASE_NAMES="${BASE_NAMES:-generation_2m_scaffold}"
OUT_SUFFIX="${OUT_SUFFIX:-_authoredonly}"
FRACTION="${FRACTION:-0.5}"
MAX_FAILING="${MAX_FAILING:-2}"
NUM_PROC="${NUM_PROC:-16}"
DRY_RUN="${DRY_RUN:-0}"

PYTHON_BIN="${PYTHON_BIN:-python}"
if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="python"
fi

extra=()
[ "$DRY_RUN" = "1" ] && extra+=(--dry-run)

for base in $BASE_NAMES; do
    src="$INPUT_ROOT/$base"
    dst="$INPUT_ROOT/${base}${OUT_SUFFIX}"
    if [ ! -d "$src" ]; then
        echo "[$base] input dir not found, skipping: $src"
        continue
    fi
    echo "=============================================================="
    echo "  in       : $src"
    echo "  out      : $dst"
    echo "  fraction : $FRACTION   max_failing : $MAX_FAILING   num_proc : $NUM_PROC"
    echo "=============================================================="
    "$PYTHON_BIN" "$PROJECT_DIR/4_sftdata_gen/scripts/filter_authored_chains.py" \
        --input-dir "$src" --output-dir "$dst" \
        --fraction "$FRACTION" --max-failing "$MAX_FAILING" \
        --num-proc "$NUM_PROC" \
        ${extra[@]+"${extra[@]}"}

    if [ "$DRY_RUN" != "1" ]; then
        cat <<EOF

Next — generate with the SAME two values:

  INPUT_ROOT=$INPUT_ROOT \\
  BASE_NAMES=${base}${OUT_SUFFIX} \\
  OUTPUT_ROOT=data/training_data/sftdata_authored \\
  AUTHORED_EDIT_FRACTION=$FRACTION AUTHORED_MAX_FAILING=$MAX_FAILING \\
      bash run_scripts/pipeline/4_sftdata_gen.sh

NOTE 4d_qc_authored.sh will report a HIGH authored-round share on this corpus
     (measured 57.8% vs 24.3% unfiltered) — every chain in it was kept because it
     has one. That is expected; read the defect rates, not the share.
EOF
    fi
done
