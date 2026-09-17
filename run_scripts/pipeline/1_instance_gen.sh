#!/usr/bin/env bash
# run_scripts/pipeline/1_instance_gen.sh
#
# Generate training instances for the generation task.
#
# From a property-tagged pool this samples a random ref molecule and builds
# "ref-anchored" property constraints in the benchmark schema. Each instance also
# records the pool joint hit count / hit-rate for the chosen constraints.
# HTTP/external services are NOT required (parquet + numpy only).
#
# POOL defaults to the SFT-TRAIN half of the held-out pool split
# (pool_split/train_pool_sfttrain.parquet, 7,145,958 rows). That half already has
# the benchmark scaffolds removed (see pool_split/train_pool_split_meta.json,
# rule_1), so instances drawn from it cannot leak the benchmark by construction —
# do NOT point POOL back at the undivided data/pool/train_pool_props.parquet.
#
# HIA and formal_charge are present as columns but never become constraints: their
# distributions are near-constant, so build_generation_instances excludes them. The
# 14 remaining properties are exactly what analyze_properties exposes.
#
# This produces the `generation_*.jsonl` that `2_substructure_gen.sh` consumes as INPUT.
#
# Usage (2M run, the default):
#   bash run_scripts/pipeline/1_instance_gen.sh
#
# Override defaults with env vars, e.g. a small smoke run:
#   OUTPUT=/data/.../instances/generation_5k.jsonl \
#   N=5000 Q_MIN=0.1 Q_MAX=0.5 SEED=0 \
#     bash run_scripts/pipeline/1_instance_gen.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# ---------------------------------------------------------------------------
# Config (override via env vars)
# ---------------------------------------------------------------------------
POOL="${POOL:-data/training_data/instances/pool_split/train_pool_sfttrain.parquet}"
OUTPUT="${OUTPUT:-data/training_data/instances/generation_2m.jsonl}"
N="${N:-2000000}"
Q_MIN="${Q_MIN:-0.1}"
Q_MAX="${Q_MAX:-0.5}"
ID_PREFIX="${ID_PREFIX:-generation}"
SEED="${SEED:-1}"
# Parallelism: the per-instance bottleneck is the hit-mask scan over the whole
#   9.75M-row pool, which is memory-bandwidth bound.
#   WORKERS    : worker processes (default 64). Bandwidth saturates, so raising it
#                too far makes the run slower, not faster.
#   CHUNK_SIZE : chunk size handed to each worker. Reproducibility depends only on
#                (SEED, CHUNK_SIZE, N) — never on WORKERS.
#   DTYPE      : float64 (default, exact values) | float32 (half the traffic, ~2x
#                faster, with slight drift in bound / hit_count).
WORKERS="${WORKERS:-64}"
CHUNK_SIZE="${CHUNK_SIZE:-20000}"
DTYPE="${DTYPE:-float64}"
# EXCLUDE_SMILES: file(s) of SMILES to exclude from the reference candidates
# (space-separated for several). Use it to build a new instance set whose reference
# molecules do not overlap an existing one.
#   EXCLUDE_SMILES=".../generation_2m_scaffold.ref_smiles.txt" \
#   OUTPUT=.../generation_2m_v3.jsonl SEED=3 bash .../1_instance_gen.sh
EXCLUDE_SMILES="${EXCLUDE_SMILES:-}"

EXCLUDE_ARGS=()
for f in $EXCLUDE_SMILES; do
    EXCLUDE_ARGS+=(--exclude-smiles "$f")
done

# ---------------------------------------------------------------------------
# Python resolution — prefer a caller-provided PYTHON, otherwise the conda env
# that has pyarrow + numpy for this pipeline (molkit), otherwise python3.
# ---------------------------------------------------------------------------
if [ -z "${PYTHON:-}" ]; then
    MOLKIT_ENV="${MOLKIT_ENV:-$CONDA_ROOT/envs/molkit}"
    if [ -x "$MOLKIT_ENV/bin/python" ]; then
        PYTHON="$MOLKIT_ENV/bin/python"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON="$(command -v python3)"
    else
        echo "ERROR: no usable Python found. Set PYTHON=/path/to/python." >&2
        exit 1
    fi
fi
echo "Using Python: $PYTHON"

cd "$PROJECT_DIR"

echo "Generating generation-task instances..."
echo "  pool    : $POOL"
echo "  output  : $OUTPUT"
echo "  n       : $N"
echo "  workers : $WORKERS  (chunk=$CHUNK_SIZE, dtype=$DTYPE)"

"$PYTHON" 1_instance_gen/build_generation_instances.py \
    --pool "$POOL" \
    --output "$OUTPUT" \
    --n "$N" \
    --q-min "$Q_MIN" \
    --q-max "$Q_MAX" \
    --id-prefix "$ID_PREFIX" \
    --seed "$SEED" \
    --workers "$WORKERS" \
    --chunk-size "$CHUNK_SIZE" \
    --dtype "$DTYPE" \
    "${EXCLUDE_ARGS[@]+"${EXCLUDE_ARGS[@]}"}"

echo "Done. Instances written to: $OUTPUT"
