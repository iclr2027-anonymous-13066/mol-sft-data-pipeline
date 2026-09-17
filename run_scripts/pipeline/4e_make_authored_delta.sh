#!/usr/bin/env bash
# run_scripts/pipeline/4e_make_authored_delta.sh
#
# Build the ADDITIVE delta for the "+authored" arm of the authored-edit ablation:
# the segments of the authored corpus that actually differ from the base, re-keyed
# so they merge into the base without leaking across the train/val split.
#
#   arm A   train_path: [ <base>_satonly ]
#   arm B   train_path: [ <base>_satonly, <base>_satonly_authoreddelta ]
#
# WHY ONLY THE DELTA. The authored corpus is the SAME tool chains as the base, with
# ~12% of decorate rounds rendered with an empty suggest_edits response and their
# edit reasoning re-derived without a candidate list. Adding the whole thing would
# double the corpus with 88% near-duplicates — that measures paraphrase
# augmentation, not the authored branch. Keeping only the changed segments makes
# arm B exactly "arm A plus the new branch".
#
# WHY THE GROUP-ID REMAP. group_id is "{task_id}__{idx:07d}", and idx comes from an
# UNSEEDED random.shuffle in stage 4 (pipeline.py) — so two runs give the same
# molecule two different group_ids (measured: 17,801/17,801 task_ids shared with the
# base, 0/17,801 group_ids matching). task_id is stable and 1:1 with group_id inside
# a corpus (verified 59,461:59,461, zero collisions), so every delta record is
# re-keyed onto the base's group_id and the two copies of a molecule carry one key.
#
# WHAT THE REMAP DOES *NOT* BUY. It does not make the delta's val slice clean.
# dataset.py splits PER DIRECTORY (one task per train_path entry, each with its own
# _split_indices over its own key list), so the base task and the delta task pick
# unrelated val groups however the keys are named. Base val is ~30 of 59,461 groups,
# so nearly every delta-val molecule is in base-train. The headline metric
# (val/gen/overall_success, from a separate held-out file) and the base task's val
# loss are unaffected; only val/<delta dir>/loss is contaminated, and it should be
# read as a train-fit curve. See the config header of
# the git history of configs/qwen3-8b-satonly-authored.yaml for the full accounting.
#
# PREREQUISITES — both corpora must already be through 4b and 4c, with the SAME 4b
# settings. A record whose task_id is missing from the base is dropped and counted;
# a large drop count means the two went through different filtering.
#
# Usage:
#   bash run_scripts/pipeline/4e_make_authored_delta.sh                 # write the delta
#   DRY_RUN=1 bash run_scripts/pipeline/4e_make_authored_delta.sh       # count only
#
# Env overrides:
#   BASE_DIR      arm A's training dir (supplies the group_id keys)
#   AUTHORED_DIR  the authored corpus, already satonly-filtered
#   OUT_DIR       where the delta goes (default: <BASE_DIR>_authoreddelta)
#   DRY_RUN       1 = count and stop
#   SKIP_ARROW    1 = do not build the _records.arrow sidecar
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_DIR"

SFT_ROOT="${SFT_ROOT:-data/training_data/sftdata}"
BASE_DIR="${BASE_DIR:-$SFT_ROOT/generation_2m_scaffold_satonly}"
# The authored corpus AFTER 4b + 4c. 4a writes the pre-filtered chain set to
# <name>_authoredonly and 4c appends _satonly, so the dir 4e reads is
# <name>_authoredonly_satonly — not <name>_satonly, which is what the plain
# corpus produces under a different root.
AUTHORED_DIR="${AUTHORED_DIR:-${SFT_ROOT}_authored/generation_2m_scaffold_authoredonly_satonly}"
OUT_DIR="${OUT_DIR:-${BASE_DIR}_authoreddelta}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_ARROW="${SKIP_ARROW:-0}"
ARROW_NUM_PROC="${ARROW_NUM_PROC:-48}"

PYTHON_BIN="${PYTHON_BIN:-python}"
if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="python"
fi

for d in "$BASE_DIR" "$AUTHORED_DIR"; do
    if [ ! -d "$d" ]; then
        echo "[error] missing directory: $d" >&2
        echo "        Both corpora must be generated and satonly-filtered first" >&2
        echo "        (4_sftdata_gen.sh → 4b → 4c)." >&2
        exit 1
    fi
done
if [ "$BASE_DIR" = "$AUTHORED_DIR" ]; then
    echo "[error] BASE_DIR and AUTHORED_DIR are the same path." >&2
    exit 1
fi

echo "=============================================================="
echo "  base     : $BASE_DIR"
echo "  authored : $AUTHORED_DIR"
echo "  delta    : $OUT_DIR"
echo "=============================================================="

extra=()
[ "$DRY_RUN" = "1" ] && extra+=(--dry-run)

"$PYTHON_BIN" "$PROJECT_DIR/4_sftdata_gen/scripts/make_authored_delta.py" \
    --base-dir "$BASE_DIR" \
    --authored-dir "$AUTHORED_DIR" \
    --out-dir "$OUT_DIR" \
    ${extra[@]+"${extra[@]}"}

if [ "$DRY_RUN" != "1" ] && [ "$SKIP_ARROW" != "1" ]; then
    echo ""
    echo "Building _records.arrow for $OUT_DIR …"
    "$PYTHON_BIN" "$PROJECT_DIR/4_sftdata_gen/scripts/jsonl_to_arrow.py" \
        "$OUT_DIR" --num-proc "$ARROW_NUM_PROC" \
        || echo "[warn] arrow build failed; training falls back to JSONL parsing."
fi

if [ "$DRY_RUN" != "1" ]; then
    cat <<EOF

Next: point arm B's config at BOTH dirs, arm A's at the base alone.

  # arm A
  data:
    train_path:
      - "$BASE_DIR"

  # arm B
  data:
    train_path:
      - "$BASE_DIR"
      - "$OUT_DIR"
EOF
fi
