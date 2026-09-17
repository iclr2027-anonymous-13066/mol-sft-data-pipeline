#!/usr/bin/env bash
# run_scripts/pipeline/2b_functional_group_gen.sh
#
# Generate the functional-group-constraint data for the FG-constrained generation
# task. This is the sibling of 2_substructure_gen.sh (scaffold constraints) and is
# entirely separate from it: different constraint, different output folder, no
# scaffold fields.
#
# For each input record this detects every scorer-visible functional group in the
# molecule, fixes the FG constraint (taken from `answer.fragments` when the record
# has one, otherwise derived from the molecule), attaches the authoritative scoring
# query (`eval_query`), and writes the natural-language description.
#
# WHY THIS NEEDS NO GPU
# ---------------------
# The scaffold pipeline calls an LLM per molecule because every Murcko scaffold is
# unique. A functional group is not: there are exactly 61 scorer-visible patterns,
# fixed. So the descriptions are written ONCE into fg_catalog.json (stage A below,
# the only step that can use a GPU) and the per-row pass (stage B) just looks them
# up. Stage B is pure CPU and runs ~1200 rows/s/8-proc.
#
# Output is fixed-size JSONL shards into a NEW folder: `<prefix>-00000.jsonl`, ...
# Sharded writing is streaming (O(shard) memory, not O(corpus)), atomic per shard,
# and RESUMABLE — re-running skips already-complete shards.
#
# Usage (both inputs, the default):
#   bash run_scripts/pipeline/2b_functional_group_gen.sh
#
# Override with env vars, e.g. a smoke run:
#   BENCH_INPUT= TRAIN_INPUT=/path/in.jsonl TRAIN_OUT_DIR=/tmp/out SHARD_SIZE=1000 \
#     bash run_scripts/pipeline/2b_functional_group_gen.sh
#
#   # rebuild the catalog with LLM-polished descriptions (needs a vLLM pool):
#   POLISH=1 SERVERS="localhost:8080,localhost:8082" \
#     bash run_scripts/pipeline/2b_functional_group_gen.sh
#
#   # skip the catalog step and only re-run the per-row pass:
#   BUILD_CATALOG=0 bash run_scripts/pipeline/2b_functional_group_gen.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
FG_DIR="$PROJECT_DIR/2b_functional_group_gen"
INSTANCES="${INSTANCES:-data/training_data/instances}"

# ---------------------------------------------------------------------------
# Config (override via env vars)
# ---------------------------------------------------------------------------
# Stage A — catalog
BUILD_CATALOG="${BUILD_CATALOG:-1}"
# The catalog is a build artifact, so it lives with the generated data, not in the
# repo. Override with CATALOG=... (or FG_CATALOG=... for the readers).
CATALOG="${CATALOG:-data/training_data/fg_catalog.json}"
CORPUS="${CORPUS:-$INSTANCES/generation_2m.jsonl}"   # measures per-group rarity
CORPUS_LIMIT="${CORPUS_LIMIT:-200000}"
POLISH="${POLISH:-0}"                                # 1 = LLM-polish the 61 descriptions
SERVERS="${SERVERS:-localhost:8080}"
VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen3.6-27B}"

# Stage B — per-row augmentation
TRAIN_INPUT="${TRAIN_INPUT:-$INSTANCES/generation_2m.jsonl}"
TRAIN_OUT_DIR="${TRAIN_OUT_DIR:-$INSTANCES/generation_2m_fg}"
TRAIN_SMILES_KEY="${TRAIN_SMILES_KEY:-ref_smiles}"

BENCH_INPUT="${BENCH_INPUT:-$INSTANCES/benchmark.jsonl}"
BENCH_OUT_DIR="${BENCH_OUT_DIR:-$INSTANCES/benchmark_fg}"
BENCH_SMILES_KEY="${BENCH_SMILES_KEY:-meta_info.ref_smiles}"

FRAGMENTS_KEY="${FRAGMENTS_KEY:-answer.fragments}"
SHARD_SIZE="${SHARD_SIZE:-20000}"
# A derived constraint takes 1 or 2 groups, sampled per row from those the reference
# molecule actually contains; each is required to occur AT LEAST once (REQUIRE_COUNT=1),
# so a reference carrying three copies does not make three a requirement.
SELECT="${SELECT:-random}"        # random | rarest | common | all
N_FG_MIN="${N_FG_MIN:-1}"
N_FG_MAX="${N_FG_MAX:-2}"
REQUIRE_COUNT="${REQUIRE_COUNT:-1}"   # 0 = use the molecule's own count instead
MATCH_MODE="${MATCH_MODE:-min}"       # min (>=) | exact (==) | auto (old grader)
SEED="${SEED:-0}"
# Benchmark side: generation task only, and only rows that have a ref_smiles —
# on benchmark those are exactly the 900 feasible generation instances.
BENCH_KEEP_TASK_TYPE="${BENCH_KEEP_TASK_TYPE:-generation}"
PROCS="${PROCS:-0}"               # 0 = auto (min(32, cpu_count))

# ---------------------------------------------------------------------------
# Python resolution — needs rdkit (+ openai only when POLISH=1).
# ---------------------------------------------------------------------------
if [ -z "${PYTHON:-}" ]; then
    for cand in $CONDA_ROOT/envs/molkit $CONDA_ROOT/envs/benchmark; do
        if [ -x "$cand/bin/python" ] && "$cand/bin/python" -c "import rdkit" 2>/dev/null; then
            PYTHON="$cand/bin/python"; break
        fi
    done
    if [ -z "${PYTHON:-}" ]; then
        if command -v python3 >/dev/null 2>&1; then PYTHON="$(command -v python3)"
        else echo "ERROR: no usable Python found. Set PYTHON=/path/to/python." >&2; exit 1; fi
    fi
fi
echo "Using Python: $PYTHON"

# ---------------------------------------------------------------------------
# Stage A — build fg_catalog.json (61 entries)
# ---------------------------------------------------------------------------
if [ "$BUILD_CATALOG" = "1" ]; then
    echo
    echo "=== Stage A: build FG catalog -> $CATALOG"
    mkdir -p "$(dirname "$CATALOG")"
    CAT_ARGS=(--out "$CATALOG" --corpus "$CORPUS" --corpus-limit "$CORPUS_LIMIT")
    if [ "$POLISH" = "1" ]; then
        # Only the 61x2 catalog descriptions go through the model, so one server is
        # plenty. GPUs 0-3 are the ones this project reserves for it.
        export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
        echo "    LLM polish ON (servers=$SERVERS, CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
        CAT_ARGS+=(--polish --servers "$SERVERS" --model "$VLLM_MODEL")
    else
        echo "    LLM polish OFF -> deterministic template descriptions (faithful by construction)"
    fi
    "$PYTHON" "$FG_DIR/build_fg_catalog.py" "${CAT_ARGS[@]}"
else
    echo "=== Stage A skipped (BUILD_CATALOG=0); using existing $CATALOG"
fi

[ -f "$CATALOG" ] || { echo "ERROR: catalog missing: $CATALOG" >&2; exit 1; }
# Readers that take no --catalog flag (verify_fg_dataset.py, fg_analyzer's rarity
# table) resolve it from here instead of guessing.
export FG_CATALOG="$CATALOG"

# ---------------------------------------------------------------------------
# Stage B — per-row augmentation (CPU only)
# ---------------------------------------------------------------------------
run_augment() {
    local input="$1" out_dir="$2" smiles_key="$3" label="$4"
    shift 4
    [ -n "$input" ] || { echo "--- $label: skipped (input unset)"; return 0; }
    [ -f "$input" ] || { echo "--- $label: skipped (missing $input)"; return 0; }
    echo
    echo "=== Stage B [$label]: $input -> $out_dir"
    "$PYTHON" "$FG_DIR/augment_jsonl_with_fg.py" \
        --input       "$input" \
        --out-dir     "$out_dir" \
        --shard-size  "$SHARD_SIZE" \
        --smiles-key  "$smiles_key" \
        --fragments-key "$FRAGMENTS_KEY" \
        --fg-source   auto \
        --select      "$SELECT" \
        --n-fg-min    "$N_FG_MIN" \
        --n-fg-max    "$N_FG_MAX" \
        --require-count "$REQUIRE_COUNT" \
        --match-mode  "$MATCH_MODE" \
        --seed        "$SEED" \
        --catalog     "$CATALOG" \
        --procs       "$PROCS" \
        "$@"
}

# The benchmark is small and carries authored constraints — run it first so a
# mistake surfaces in seconds rather than after the 2M pass.
run_augment "$BENCH_INPUT" "$BENCH_OUT_DIR" "$BENCH_SMILES_KEY" \
            "benchmark (authored, generation+feasible only)" \
            --keep-task-type "$BENCH_KEEP_TASK_TYPE" --require-smiles
run_augment "$TRAIN_INPUT" "$TRAIN_OUT_DIR" "$TRAIN_SMILES_KEY" "generation_2m (derived)"

echo
echo "Done."
