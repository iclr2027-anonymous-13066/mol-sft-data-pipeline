#!/usr/bin/env bash
# run_scripts/pipeline/4c_make_satisfied_only.sh
#
# Build the SATISFIED-ONLY copy of the stage-4 SFT data (ablation arm), plus the
# train-set gen-eval instance subset that goes with it.
#
# Stage 4 runs with KEEP_UNSATISFIED=1, so its output mixes chains that reached a
# fully-satisfying molecule (the conversation closes on <ANSWER>) with chains that
# did not (no <ANSWER>). Step 1 writes a sibling dir that keeps a satisfied chain
# whole, and truncates an unsatisfied one to its seed round — the one that derives
# the scaffold SMARTS and runs the 3-way checkpoint, which is correct however the
# chain ended. KEEP_UNSAT_SEED=0 drops those chains
# outright instead. The two training arms can then be compared:
#
#   <name>            satisfied + unsatisfied   → configs/qwen3-{1.7b,8b}-original.yaml
#   <name>_satonly    satisfied only            → the three data-format arms' configs
#
# Step 2 (unless SKIP_TRAIN_SUBSET=1) samples TRAIN_SUBSET_N instances FROM that
# satisfied-only dir and writes their full scoring-format records to
#   <INSTANCES_ROOT>/<name>_satonly-trainsubset<N>.jsonl
# which is what all four ablation configs point gen_eval.train_instances_path at.
# It samples the instances the satisfied-only arm actually trained on — restricted
# to those it saw a FULL solution for, never the seed-only ones — so rerun it
# whenever that dir changes (this script does it automatically).
#
# Safe to run WHILE stage 4 is still generating: only chunks with a .done marker
# are read, and re-running later converts just the newly-finished chunks (a chunk
# whose source is newer than its copy — e.g. rewritten in place by 4b/4c — is
# reconverted). The source dir is never modified. Expect ~88-90% of records to
# survive (~63-67% of chains are satisfied and keep every segment; the rest
# contribute their seed segment alone). Switching KEEP_UNSAT_SEED makes every
# existing destination file stale, so the whole dir is reconverted.
#
# Run this AFTER the cleaning pass (4b_filter_test_scaffold_leak.sh): it rewrites
# the source shards in place, and
# whatever they remove afterwards stays in an already-written copy until the next
# run picks that file up again.
#
# Usage:
#   bash run_scripts/pipeline/4c_make_satisfied_only.sh                    # convert + subset
#   DRY_RUN=1 bash run_scripts/pipeline/4c_make_satisfied_only.sh          # count only
#   NAMES="generation_2m_scaffold" bash run_scripts/pipeline/4c_make_satisfied_only.sh
#   INCLUDE_UNFINISHED=1 OVERWRITE=1 bash run_scripts/pipeline/4c_make_satisfied_only.sh
#   SKIP_TRAIN_SUBSET=1 bash run_scripts/pipeline/4c_make_satisfied_only.sh   # step 1 only
#   TRAIN_SUBSET_N=200 bash run_scripts/pipeline/4c_make_satisfied_only.sh
#   KEEP_UNSAT_SEED=0 bash run_scripts/pipeline/4c_make_satisfied_only.sh   # satisfied chains only
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_DIR"

OUTPUT_ROOT="${OUTPUT_ROOT:-data/training_data/sftdata}"
# Which SFT subdirs to filter (keep in sync with BASE_NAMES in 4_sftdata_gen.sh).
NAMES="${NAMES:-generation_2m_scaffold}"
# Suffix appended to each source dir name to form the destination dir.
DST_SUFFIX="${DST_SUFFIX:-_satonly}"

NUM_PROC="${NUM_PROC:-16}"
ARROW_NUM_PROC="${ARROW_NUM_PROC:-48}"
DRY_RUN="${DRY_RUN:-0}"
OVERWRITE="${OVERWRITE:-0}"
INCLUDE_UNFINISHED="${INCLUDE_UNFINISHED:-0}"
SKIP_ARROW="${SKIP_ARROW:-0}"
# Keep the seed (SMARTS-derivation) segment of chains that never satisfied every
# constraint — it is correct however the chain ended. 0 = drop those chains whole.
KEEP_UNSAT_SEED="${KEEP_UNSAT_SEED:-1}"

# ── Step 2: train-set gen-eval instance subset ───────────────────────────────
# Sampled from the satisfied-only dir (see the header). The instances dir supplies
# the full scoring-format records and is looked up per SOURCE name
# (<INSTANCES_ROOT>/<name>), which is where stage 1 wrote them.
SKIP_TRAIN_SUBSET="${SKIP_TRAIN_SUBSET:-1}"  # default off: step 2 needs the training-stage subset builder, not part of this repository
TRAIN_SUBSET_N="${TRAIN_SUBSET_N:-100}"
SUBSET_SEED="${SUBSET_SEED:-42}"
INSTANCES_ROOT="${INSTANCES_ROOT:-data/training_data/instances}"

if [ -z "${PYTHON:-}" ]; then
    for c in python python \
             python; do
        [ -x "$c" ] && PYTHON="$c" && break
    done
    PYTHON="${PYTHON:-$(command -v python3)}"
fi
echo "Using Python: $PYTHON"

flags=()
[ "$DRY_RUN" = "1" ]            && flags+=(--dry-run)
[ "$OVERWRITE" = "1" ]          && flags+=(--overwrite)
[ "$INCLUDE_UNFINISHED" = "1" ] && flags+=(--include-unfinished)
[ "$SKIP_ARROW" = "1" ]         && flags+=(--no-arrow)
[ "$KEEP_UNSAT_SEED" = "0" ]    && flags+=(--no-unsatisfied-seed)

found=0
for n in $NAMES; do
    src="$OUTPUT_ROOT/$n"
    if [ ! -d "$src" ]; then
        echo "[skip] not found: $src"
        continue
    fi
    found=1
    dst="$OUTPUT_ROOT/${n}${DST_SUFFIX}"
    echo ""
    echo "=== [1/2] $n → ${n}${DST_SUFFIX} ==="
    "$PYTHON" 4_sftdata_gen/scripts/make_satisfied_only.py \
        --src "$src" \
        --dst "$dst" \
        --num-proc "$NUM_PROC" \
        --arrow-num-proc "$ARROW_NUM_PROC" \
        ${flags[@]+"${flags[@]}"}

    # ── Step 2: train-set gen-eval subset from the just-written satonly dir ──
    [ "$DRY_RUN" = "1" ] && continue
    if [ "$SKIP_TRAIN_SUBSET" = "1" ]; then
        echo "[$n] SKIP_TRAIN_SUBSET=1 — not rebuilding the train gen-eval subset"
        continue
    fi
    inst_dir="$INSTANCES_ROOT/$n"
    if [ ! -d "$inst_dir" ]; then
        echo "[$n][warn] instances dir not found, skipping train gen-eval subset:" >&2
        echo "          $inst_dir  (set INSTANCES_ROOT, or SKIP_TRAIN_SUBSET=1)" >&2
        continue
    fi
    subset_out="$INSTANCES_ROOT/${n}${DST_SUFFIX}-trainsubset${TRAIN_SUBSET_N}.jsonl"
    echo ""
    echo "=== [2/2] train gen-eval subset: ${TRAIN_SUBSET_N} instances → $(basename "$subset_out") ==="
    subset_script="5_train_sft/scripts/build_train_gen_subset.py"
    if [ ! -f "$subset_script" ]; then
        echo "[$n][warn] $subset_script not found (the training stage is not part of this repository); skipping" >&2
        continue
    fi
    "$PYTHON" "$subset_script" \
        --sftdata-dir "$dst" \
        --instances-dir "$inst_dir" \
        --out "$subset_out" \
        --n "$TRAIN_SUBSET_N" \
        --seed "$SUBSET_SEED"
done

[ "$found" = "0" ] && { echo "No source dirs found under $OUTPUT_ROOT"; exit 0; }

echo ""
if [ "$DRY_RUN" = "1" ]; then
    echo "Report only. Re-run without DRY_RUN=1 to write the copy."
else
    echo "Done. Satisfied-only data under: $OUTPUT_ROOT/<name>${DST_SUFFIX}"
    [ "$SKIP_TRAIN_SUBSET" = "1" ] || \
        echo "      Train gen-eval subset  : $INSTANCES_ROOT/<name>${DST_SUFFIX}-trainsubset${TRAIN_SUBSET_N}.jsonl"
fi
