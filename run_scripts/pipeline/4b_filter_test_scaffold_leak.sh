#!/usr/bin/env bash
# run_scripts/pipeline/4b_filter_test_scaffold_leak.sh
#
# Remove TEST-SET SCAFFOLD LEAKAGE from the stage-4 SFT data.
#
# The eval set (benchmark_exact_subset100.jsonl) scores the model on hitting
# a target Murcko scaffold. Any training chain whose source instance carries the
# SAME scaffold_smiles lets the model memorise the very scaffold it will be asked
# to reproduce. This maps each SFT record back to its instance (the group_id
# prefix), finds the instances sharing a test scaffold, and drops ALL records of
# those chains — the whole chain, since later segments carry earlier context.
#
# Trivial scaffolds (benzene, pyridine, cyclohexane, …) dominate the raw hits and
# are chemistry, not leakage. MIN_HEAVY / MIN_RINGS keep them; the report always
# prints the drop cost at several thresholds first, so run DRY (the default) once
# and pick a threshold before applying.
#
# Safe to run WHILE stage 4 is still generating: --apply rewrites only chunks that
# have a .done marker (set FORCE_INFLIGHT=1 to override), each atomically via
# tmp→move. Re-running later cleans the chunks that finished in the meantime.
# The rewrite is IN PLACE and irreversible — DRY_RUN first.
#
# Usage:
#   bash run_scripts/pipeline/4b_filter_test_scaffold_leak.sh                     # report only
#   MIN_HEAVY=10 APPLY=1 bash run_scripts/pipeline/4b_filter_test_scaffold_leak.sh
#   NAMES="generation_2m_scaffold generation_2m_scaffold_satonly" APPLY=1 bash ...
#   TEST_FILE=/path/other_test.jsonl bash ...   # different eval set (new cache)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_DIR"

OUTPUT_ROOT="${OUTPUT_ROOT:-data/training_data/sftdata}"
# Which SFT subdirs to filter (keep in sync with BASE_NAMES in 4_sftdata_gen.sh).
# NOTE for the 2M run: instances are drawn from pool_split/train_pool_sfttrain
# .parquet, whose split already DROPPED every benchmark scaffold (see
# train_pool_split_meta.json rule_1: dropped_bench = 608,845). So this pass is
# expected to report ~0 leaks — a non-zero count means the pool assumption broke,
# not that the filter is working. Keep running it as the assertion.
NAMES="${NAMES:-generation_2m_scaffold}"

INSTANCES_ROOT="${INSTANCES_ROOT:-data/training_data/instances}"
# The eval set whose scaffolds must NOT appear in training.
TEST_FILE="${TEST_FILE:-$INSTANCES_ROOT/benchmark_exact_subset100.jsonl}"
# The instance pool the SFT data was generated from (group_id prefix → scaffold).
INSTANCES_DIR="${INSTANCES_DIR:-$INSTANCES_ROOT/generation_2m_scaffold}"

# Only count a shared scaffold as leakage above these sizes (0 = drop every hit).
MIN_HEAVY="${MIN_HEAVY:-0}"
MIN_RINGS="${MIN_RINGS:-0}"

PROCS="${PROCS:-48}"
APPLY="${APPLY:-0}"
FORCE_INFLIGHT="${FORCE_INFLIGHT:-0}"
# {instance_id: scaffold_smiles} cache — makes threshold sweeps skip the 13G scan.
CACHE_DIR="${CACHE_DIR:-$OUTPUT_ROOT/.testleak_cache}"
LEAK_IDS="${LEAK_IDS:-$CACHE_DIR/$(basename "$TEST_FILE" .jsonl)__$(basename "$INSTANCES_DIR").json}"

if [ -z "${PYTHON:-}" ]; then
    for c in python python \
             python; do
        [ -x "$c" ] && PYTHON="$c" && break
    done
    PYTHON="${PYTHON:-$(command -v python3)}"
fi
echo "Using Python: $PYTHON"

[ -f "$TEST_FILE" ]   || { echo "Test file not found: $TEST_FILE"; exit 1; }
[ -d "$INSTANCES_DIR" ] || { echo "Instances dir not found: $INSTANCES_DIR"; exit 1; }
mkdir -p "$CACHE_DIR"

echo "Test set:   $TEST_FILE"
echo "Instances:  $INSTANCES_DIR"
echo "Thresholds: min_heavy=$MIN_HEAVY min_rings=$MIN_RINGS"
echo "Cache:      $LEAK_IDS"

flags=()
[ "$APPLY" = "1" ]          && flags+=(--apply)
[ "$FORCE_INFLIGHT" = "1" ] && flags+=(--force-inflight)

found=0
for n in $NAMES; do
    d="$OUTPUT_ROOT/$n"
    if [ ! -d "$d" ]; then
        echo "[skip] not found: $d"
        continue
    fi
    found=1
    echo ""
    echo "=== $n ==="
    "$PYTHON" 4_sftdata_gen/scripts/filter_test_scaffold_leak.py \
        --dir "$d" \
        --test-file "$TEST_FILE" \
        --instances-dir "$INSTANCES_DIR" \
        --leak-ids "$LEAK_IDS" \
        --min-scaffold-heavy "$MIN_HEAVY" \
        --min-scaffold-rings "$MIN_RINGS" \
        --procs "$PROCS" \
        ${flags[@]+"${flags[@]}"}
done

[ "$found" = "0" ] && { echo "No SFT dirs found under $OUTPUT_ROOT"; exit 0; }

echo ""
if [ "$APPLY" = "1" ]; then
    echo "Applied. Re-run after stage 4 finishes to clean the remaining chunks."
else
    echo "Report only. Re-run with APPLY=1 (and a MIN_HEAVY you picked) to rewrite in place."
fi
