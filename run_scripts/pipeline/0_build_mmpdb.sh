#!/usr/bin/env bash
# run_scripts/pipeline/0_build_mmpdb.sh
#
# Build a data-driven molecule-edit move set from a property-tagged pool via mmpdb.
# This is the FIRST stage of the SFT data pipeline: it mines matched molecular pairs
# (MMPs) and their per-property change statistics, then writes the move sets used by
# downstream tool-chain generation:
#   attach_library.json  — single-attachment R-group fragments (also the novelty
#                          source-universe of building blocks)
#   single_cut.json      — 1-attachment substituent swaps A->B for edit_fragment
#   double_cut.json      — 2-attachment swaps
#   triple_cut.json      — 3-attachment swaps
# Each swap row carries its environment radius (0..5) + context and a full Δ-vector
# (mean/std for every indexed property). mmpdb indexes all radii in one pass, so
# radius 0 (context-free) through 5 (context-specific) are all extracted together.
#
# Optional follow-up (human-readable export of the mmpdb SQLite db):
#   "$PYTHON" 0_build_mmpdb/export_mmpdb.py <OUT_DIR>/sample.mmpdb --out-dir <OUT_DIR>
#
# Prerequisites:
#   mmpdb (`pip install mmpdb`) + RDKit in the active env (default: benchmark).
#
# Usage:
#   bash run_scripts/pipeline/0_build_mmpdb.sh
#
# Override defaults with env vars, e.g.:
#   SRC=/data/.../train_pool_props.parquet OUT_DIR=/data/.../mmp_moves \
#   N_SAMPLE=50000 NUM_JOBS=8 \
#     bash run_scripts/pipeline/0_build_mmpdb.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# ---------------------------------------------------------------------------
# Config (override via env vars)
# ---------------------------------------------------------------------------
SRC="${SRC:-data/pool/train_pool_props.parquet}"
OUT_DIR="${OUT_DIR:-data/mmp_moves}"
N_SAMPLE="${N_SAMPLE:-2200000}"
NUM_JOBS="${NUM_JOBS:-48}"
MIN_SUPPORT="${MIN_SUPPORT:-10}"
MAX_RADIUS="${MAX_RADIUS:-5}"
SEED="${SEED:-0}"
# SHARDS>1 → parallel build: partition by constant + concurrent `index --properties`
# per shard, pooled at extraction. Parallelises the property-stats step (the ~80%
# serial bottleneck of a single `index`); ~5x faster at 8 shards. 1 = single-DB.
SHARDS="${SHARDS:-128}"
# Max concurrent shard `index` processes (each single-threaded + holds its shard in
# RAM). 0 = min(SHARDS, 32). Keep within free cores AND memory on a shared node.
MAX_PARALLEL="${MAX_PARALLEL:-48}"
# Worker processes for the sharded swap EXTRACTION (pooling per-transform across all
# shards). Each holds every shard's rule_smiles map in RAM, so scale with cores AND
# memory. Only used when SHARDS>1.
EXTRACT_JOBS="${EXTRACT_JOBS:-16}"
# SKIP_BUILD=1 → reuse the existing sample.mmpdb (or shard.*.mmpdb) in OUT_DIR and
# only re-extract the move sets (fast; for tuning MIN_SUPPORT / MAX_RADIUS).
SKIP_BUILD="${SKIP_BUILD:-0}"

# ---------------------------------------------------------------------------
# Python resolution — prefer a caller-provided PYTHON, otherwise the conda env
# that has mmpdb + rdkit for this pipeline (benchmark), otherwise python3.
# ---------------------------------------------------------------------------
if [ -z "${PYTHON:-}" ]; then
    if [ -x "python" ]; then
        PYTHON="python"
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

echo "Building mmpdb move set..."
echo "  src     : $SRC"
echo "  out-dir : $OUT_DIR"
echo "  n-sample: $N_SAMPLE"
echo "  shards  : $SHARDS"

EXTRA_ARGS=()
[ "$SKIP_BUILD" = "1" ] && EXTRA_ARGS+=(--skip-build)

"$PYTHON" 0_build_mmpdb/build_mmp_moves.py \
    --src "$SRC" \
    --out-dir "$OUT_DIR" \
    --n-sample "$N_SAMPLE" \
    --num-jobs "$NUM_JOBS" \
    --min-support "$MIN_SUPPORT" \
    --max-radius "$MAX_RADIUS" \
    --shards "$SHARDS" \
    --max-parallel "$MAX_PARALLEL" \
    --extract-jobs "$EXTRACT_JOBS" \
    --seed "$SEED" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

echo "Done. Move sets under: $OUT_DIR"
