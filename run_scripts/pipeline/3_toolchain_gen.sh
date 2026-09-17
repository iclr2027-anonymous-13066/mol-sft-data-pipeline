#!/usr/bin/env bash
# run_scripts/pipeline/3_toolchain_gen.sh
#
# Build ground-truth tool chains for the scaffold-constrained generation task.
#
# Direct-scaffold-seed (default): the seed IS the completed scaffold. The chain is
# a `checkpoint → (edit → checkpoint)*` alternation where every checkpoint is the
# 3-way match_substructure ∥ analyze_properties ∥ label_atom_indices
# (match verifies scaffold_smarts, analyze reports properties, label grounds the
# next edit). Only property-tuning decorate edits are kept; the scaffold-building
# edits are dropped. DIRECT_SCAFFOLD_SEED=0 selects the legacy checkpoint FORMAT
# instead (2-way checkpoints + a separate label_atom_indices before every edit);
# it plans the same decorate steps, and any chain still carrying a scaffold-BUILD
# step from an old planner is skipped loudly rather than emitting a removed tool.
#
# The trajectory is planned purely from the description + properties via the
# Δ-guided forward search (SEARCH_MODE below); ref_smiles is never used.
#
# Prerequisites:
#   NONE by default — tools run IN-PROCESS across NUM_PROCS worker processes
#   (no tool servers needed; each worker loads its own admet_ai model). Stored
#   results are byte-identical to the tool-server path.
#   To instead use the HTTP tool servers, set NUM_PROCS=1 LOCAL_TOOLS=0; then all
#   5 tool servers must be running on 10000-10004:
#       python -m molkit.tools.tool_server --all
#
# Usage (2M run, the default): reads the shards written by stage 2 and writes
# chunk files of 20k records each. Work is assigned per PART_SIZE-record PART (not
# per input file), so NUM_PROCS is independent of the shard count; on --resume the
# outstanding parts are re-dealt over all workers.
#
# Override defaults with env vars, e.g. a smoke run over one shard:
#   INPUT=/path/to/one_shard.jsonl LIMIT=200 SEARCH_MODE=greedy NUM_PROCS=1 \
#     bash run_scripts/pipeline/3_toolchain_gen.sh
#
# Throughput (measured on the previous 2M run): hybrid ~9.5 inst/s ≈ 2.4 days;
# greedy is 3-4x faster (fewer real measurements) at a lower satisfied rate. beam
# (the default) sits between the two.
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# ---------------------------------------------------------------------------
# Config (override via env vars)
# ---------------------------------------------------------------------------
INPUT="${INPUT:-data/training_data/instances/generation_2m_scaffold_segment1}"
OUTPUT_DIR="${OUTPUT_DIR:-data/training_data/toolchain}"
# CHUNK_SIZE = records per FINAL output file (chunk_<cccccc>.jsonl, in original
# instance order). PART_SIZE = the parallel WORK unit + incremental-write
# granularity, DECOUPLED from the file size: workers build PART_SIZE-sized parts
# round-robin (so all NUM_PROCS stay busy and results stream to disk throughout
# the run, crash-safe), then parts are merged in order into CHUNK_SIZE files at the
# end. So you get 20000 records/file AND full parallelism. Keep PART_SIZE well
# below N/NUM_PROCS (e.g. 1000: 2M→2000 parts over 120 workers ≈ 17 each). CHUNK_SIZE
# is rounded down to a multiple of PART_SIZE. (PART_SIZE only applies in MP mode.)
CHUNK_SIZE="${CHUNK_SIZE:-20000}"   # records per final file
PART_SIZE="${PART_SIZE:-1000}"      # MP work unit / incremental-write granularity
NUM_WORKERS="${NUM_WORKERS:-48}"   # single-process (HTTP) async concurrency; ignored in local multiprocess mode
LIMIT="${LIMIT:-}"
DIRECT_SCAFFOLD_SEED="${DIRECT_SCAFFOLD_SEED:-1}"   # 1=direct scaffold seed + 3-way ckpts (default); 0=legacy
# Keep beam_width = beam_expand = suggest_top_k so the number of
# suggest_edits candidates shown in the SFT data == the number the search actually
# explored (beam explores beam_expand candidates/node; suggest returns that many).
SEARCH_MODE="${SEARCH_MODE:-beam}"   # greedy | beam | hybrid
BEAM_WIDTH="${BEAM_WIDTH:-4}"
BEAM_EXPAND="${BEAM_EXPAND:-4}"
SUGGEST_TOP_K="${SUGGEST_TOP_K:-4}"  # greedy's suggest_edits top_k (beam uses BEAM_EXPAND)
# Decoration DEPTH budget: max beam rounds (beam) / max greedy edits (greedy,
# hybrid) before the search returns its best molecule. This is an upper bound, not
# a fixed depth — the search stops early once every target is satisfied or no
# guard-passing candidate remains. Used to be left unset here, which silently took
# the CLI default of 12; pinned to 7 so the pipeline depth is explicit.
SEARCH_MAX_STEPS="${SEARCH_MAX_STEPS:-7}"

# ---------------------------------------------------------------------------
# Local in-process execution (DEFAULT) — no tool servers needed
# ---------------------------------------------------------------------------
# Tools run IN-PROCESS (byte-identical stored results, no HTTP round-trip, no
# ADMET retry storm). NUM_PROCS worker processes build disjoint shards in
# parallel, each with its own admet_ai model, round-robined over GPUS.
#
# Tuning (re-measured 2026-07-24 on the 288-core node, hybrid w4×e4 search, GPUs
# 0-3, AFTER mp_runner started capping BLAS threads to 1/worker):
#   * The workload is CPU-BOUND, not GPU-bound. With BLAS capped, GPUs 0-3 sit at
#     only ~50% util; the earlier "GPU-bound" reading was uncapped-BLAS CPU thrash.
#   * Throughput PLATEAUS at NUM_PROCS ≈ 32-48 (~9-10 instances/s ≈ ~100 ADMET
#     mol/s). NP=32 and NP=64 give the same steady rate — more procs do NOT help.
#   * NP ≥ 96 is COUNTERPRODUCTIVE: 96 workers loading admet_ai at once thrash
#     startup (didn't finish loading in 150 s on a shared node) with no throughput
#     gain. So do NOT scale procs with GPU count for this path.
#   * BLAS capping is essential and now automatic (mp_runner sets OMP/MKL/OPENBLAS/
#     NUMEXPR/VECLIB=1 per worker). Uncapped, even 64 procs failed to start.
#   * PER_PROC 4 is fine (8 was slightly worse).
#
# So: ~48 procs over GPUs 0-3 is the sweet spot; GPUs only spread ADMET-model
# memory (1 GPU's compute would suffice). On a heavily-shared node, drop toward 32.
# 2M @ ~9.5 inst/s (hybrid) ≈ ~2.4 days; greedy is ~3-4× faster (fewer measurements).
#
# Set NUM_PROCS=1 for a single process; add LOCAL_TOOLS=0 to instead POST to the
# running tool servers (the old HTTP path).
# Work is assigned per PART: the parent deals the outstanding PART_SIZE-record parts
# to workers in contiguous blocks (each worker opens only the 1-2 input files its
# block spans), so NUM_PROCS is NOT bounded by the input shard count and the tail is
# bounded by one part. Assignment used to be per input FILE, which collapsed the end
# of a run onto the 1-2 workers owning an extra file (measured 2026-07-31 on the 1M
# segment1 run: 16.7 inst/s → ~0.7 inst/s for the last 40 of 1000 parts). If a run
# still ends up straggling, just Ctrl-C and re-run: --resume re-deals what is left.
NUM_PROCS="${NUM_PROCS:-48}"       # local worker processes (>1 implies local mode)
PER_PROC="${PER_PROC:-4}"          # async concurrency within each worker (keep 2-4)
GPUS="${GPUS:-4,5,6,7}"            # comma-separated GPU ids to round-robin workers over
LOCAL_TOOLS="${LOCAL_TOOLS:-1}"    # 1=in-process tools (default); 0=HTTP tool servers


# ---------------------------------------------------------------------------
# Python resolution — prefer a caller-provided PYTHON, otherwise fall back to
# the conda env the Makefile uses for ML workloads (MOLKIT_ENV), otherwise
# the first `python3` on PATH.
# ---------------------------------------------------------------------------
if [ -z "${PYTHON:-}" ]; then
    MOLKIT_ENV="${MOLKIT_ENV:-$CONDA_ROOT/envs/benchmark}"
    if [ -x "$MOLKIT_ENV/bin/python" ]; then
        PYTHON="$MOLKIT_ENV/bin/python"
    elif [ -x "python" ]; then
        PYTHON="python"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON="$(command -v python3)"
    else
        echo "ERROR: no usable Python found. Set PYTHON=/path/to/python." >&2
        exit 1
    fi
fi
echo "Using Python: $PYTHON"

cd "$PROJECT_DIR"

# Base output-subfolder name: explicit OUTPUT_NAME, else derived from the input
# (dir name, or file stem — matching the CLI default).
BASE_NAME="${OUTPUT_NAME:-}"
if [ -z "$BASE_NAME" ]; then
    BASE_NAME="$(basename "$INPUT")"; BASE_NAME="${BASE_NAME%.jsonl}"
fi

# build + execute the toolchain run.
run_build() {
    local -a cmd=("$PYTHON" -m 3_toolchain_gen
        --input "$INPUT"
        --output "$OUTPUT_DIR"
        --output-name "$BASE_NAME"
        --chunk-size "$CHUNK_SIZE"
        --part-size "$PART_SIZE"
        --num-workers "$NUM_WORKERS"
        --num-procs "$NUM_PROCS"
        --per-proc "$PER_PROC"
        --gpus "$GPUS"
        --search-mode "$SEARCH_MODE"
        --search-max-steps "$SEARCH_MAX_STEPS"
        --beam-width "$BEAM_WIDTH"
        --beam-expand "$BEAM_EXPAND"
        --suggest-top-k "$SUGGEST_TOP_K"
        --resume)
    if [ "$DIRECT_SCAFFOLD_SEED" = "0" ]; then
        cmd+=(--no-direct-scaffold-seed)
    fi
    # NUM_PROCS>1 implies local mode; otherwise honour LOCAL_TOOLS for 1 process.
    if [ "$NUM_PROCS" = "1" ] && [ "$LOCAL_TOOLS" != "0" ]; then
        cmd+=(--local-tools)
    fi
    if [ -n "$LIMIT" ]; then
        cmd+=(--limit "$LIMIT")
    fi
    echo ""
    echo "=== $SEARCH_MODE -> $OUTPUT_DIR/${BASE_NAME} ==="
    "${cmd[@]}"
}

echo "Building substructure-generation tool chains..."
echo "  input : $INPUT"
echo "  output: $OUTPUT_DIR"

run_build

echo ""
echo "Tool chains written under: $OUTPUT_DIR"
