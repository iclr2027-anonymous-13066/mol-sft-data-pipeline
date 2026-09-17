#!/usr/bin/env bash
# run_scripts/pipeline/4_sftdata_gen.sh
#
# Convert constructive ground-truth tool chains into SFT training data.
# Runs the substructure-generation passes over the DIRECT-SCAFFOLD-SEED chains
# (seed = the complete scaffold; every checkpoint is the 3-way match_substructure
# ∥ analyze_properties ∥ label_atom_indices, and every edit round is the two-turn
# suggest_edits → edit_fragment pair). Output: minimal system prompt, a seed
# segment whose first reasoning derives the SMARTS AND writes the scaffold SMILES
# (then the 3-way seed checkpoint), one edit round per segment, and a terminal
# ANSWER segment, merged into one conversation per chain. Default input:
#   - $INPUT_ROOT/generation_2m_scaffold   (5-tool chains from stage 3)
# The edit-round reasoning is decided from PRE-COMPUTED blocks, not from the model's own
# arithmetic. Two of them, and the split is the point:
#
#   Landing Safety (fragment_names.landing_safety) — one row per candidate, so it is the
#     only block that can pick one: rank (the tool's own order), safe (constraints that
#     land inside the box with a full predicted std of room), IN after (inside, margin
#     ignored — read against `safe`, a candidate at 5/10 safe but 9/10 IN AFTER lands
#     almost everything on thin margins), gap closed (share of the current box distance
#     removed; the share, because the absolute distance does not compare across molecules
#     of different size), +heavy / ΔMW (edit size), breaks (satisfied constraints the edit
#     pushes back out), overshoot (an axis carried PAST the far edge of its box — a 100%
#     gap share can still be a bad move).
#   This Round (fragment_names.round_context) — identical for every candidate, so nothing
#     in it may pick one; it sets POSTURE. Which ceiling is tightest (how much overshoot
#     the round can afford), whether the LAST edit actually reduced the box distance (if
#     not, this round is a correction and must say so), how far the LAST prediction landed
#     from the measurement (a miss has to be quoted and the hedge sized by it), and whether
#     the edit SITE separates the candidates at all — it usually does not, so "chosen
#     because the site is aromatic" is almost always a claim about nothing.
#
# The columns are the quantities a held-out feature ablation kept, restricted to what a
# model reading the prompt can actually produce: `safe` is the hand-computable stand-in for
# the CDF-based score (it recovers ~78% of its lift by COUNTING), and `overshoot` and the
# gap SHARE replace the z-scaled versions, which need a scale the prompt does not carry.
#
# All of it is arithmetic the generator was measured getting wrong when asked to do it
# itself (over 5 rounds x 11 criteria: a tie was never once reported as a tie, safe counts
# were wrong in 4 of 5 rounds, and one sentence picked a candidate BECAUSE its spread was
# the wider one), so it is computed here and the prose only quotes it. The block also states
# IN WORDS when a criterion does not separate the candidates, lists on one line every
# column that is identical for all of them (so the prose neither argues from nor narrates
# them), and — because the stated decision order lands on the committed candidate only
# about a third of the time, the search does not follow it — runs that order to the end and
# emits a NOTE when it lands elsewhere, naming the criterion the prose has to concede.
#
# `landing_claim_errors` checks the generated span against the block and triggers the same
# regenerate-with-the-error-quoted retry the ± guard uses: counts absent from the block,
# leads claimed on a criterion the block calls a tie, ties claimed in a column that does
# not show them, candidate indices outside 1..N, picking a candidate BECAUSE its value is
# the worse one, and ignoring the NOTE. `ring_locant_errors` does the same for the SEED
# span, where a wrong locant rides into every later round that quotes it.
#
# Cost: Landing Safety is ~1.0k characters and This Round ~350, against ~690 for the
# single block this replaced — stage 4 is prefill-bound (23:1 prompt:output), so budget
# that much extra prefill.
#
# NAIVE_REASONING=1 renders the same chains with all of that withheld (see config.py) —
# the ablation baseline that prices the scaffolding. Not for training corpora.
#
# Chains whose final molecule does NOT satisfy every constraint are KEPT by default
# (KEEP_UNSATISFIED=1 → --keep-unsatisfied): their conversation stops after the last
# tool response with NO terminal <ANSWER>; only fully-satisfied chains get one. Set
# KEEP_UNSATISFIED=0 to drop the unsatisfied chains.
#
# Each pass is silently skipped when its input directory is missing or empty.
#
# After each pass, the memory-mapped raw sidecar <output_dir>/_records.arrow is
# (re)built from the just-written JSONL so training startup memory-maps it
# instead of re-parsing every shard (see data/dataset.py::_ArrowSource). It is
# rebuilt only when missing/stale, skipped entirely with SKIP_ARROW=1, and
# non-fatal (training still works, just slower, without it). NOTE: this is the
# RAW sidecar; the tokenised cache (tok_cache) is a separate layer built lazily
# on the first train.py launch.
#
# Prerequisites:
#   - Stage 3 must have FINISHED, including its part→chunk merge step: the
#     pipeline only reads toolchains_*chunk_*.jsonl. While stage 3 is still
#     running the input dir holds toolchains_generation_part_*.jsonl instead,
#     and this script refuses to start (see the preflight in run_pass).
#   - vLLM servers for Qwen/Qwen3.6-27B must be up (one `vllm serve` per GPU).
#     Defaults to localhost 8080/8082/8084/8086; override with VLLM_URLS. Dead URLs
#     are probed out, so an over-long list only costs a startup warning.
#   - The `molkit` package must be importable (tool schemas are read from
#     molkit.tools.TOOL_REGISTRY). Tool servers are NOT required — the
#     pipeline reuses the pre-computed expected_response fields.
#   - The output dir must not hold JSONL from an EARLIER toolchain generation:
#     --resume skips a chunk whenever a file of the same name already exists, so
#     stale output would be silently kept and mixed into training. The preflight
#     aborts on stale files; move them aside (STALE_OK=1 to override).
#
# Usage:
#   bash run_scripts/pipeline/4_sftdata_gen.sh
#
# Override defaults with env vars, e.g.:
#   VLLM_URLS=http://localhost:8080/v1,http://localhost:8081/v1 \
#   BATCH_SIZE=128 bash run_scripts/pipeline/4_sftdata_gen.sh
#
# Write to a directory named differently from the input chain set (see OUTPUT_NAME):
#   BASE_NAMES=generation_2m_fg OUTPUT_NAME=generation_2m_fg_tmp \
#   bash run_scripts/pipeline/4_sftdata_gen.sh
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# ---------------------------------------------------------------------------
# Config (override via env vars)
# ---------------------------------------------------------------------------
INPUT_ROOT="${INPUT_ROOT:-data/training_data/toolchain}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/training_data/sftdata}"
# Variant subdirs to process, applied ON TOP of each BASE_NAME. Default is 'plain'
# (the base directseed3way chains themselves). Override with a space-separated list;
# a variant dir that does not exist on disk is skipped silently.
SUFFIXES="${SUFFIXES:-plain}"

# Keep chains whose final molecule does NOT satisfy every constraint (default ON):
# their conversation stops after the last tool response with NO terminal <ANSWER>
# (only fully-satisfied chains get one). Set KEEP_UNSATISFIED=0 to drop the
# unsatisfied chains instead.
KEEP_UNSATISFIED="${KEEP_UNSATISFIED:-1}"

BATCH_SIZE="${BATCH_SIZE:-640}"
NUM_GENERATIONS="${NUM_GENERATIONS:-1}"

# Number of worker PROCESSES over the same input/output dir. Each takes the
# chunks where (chunk index % SHARDS) == its own index, so their chunk files —
# and therefore their output files — are disjoint and they cannot collide.
#
# Why more than one: a single worker cannot keep the fleet fed. Its event loop
# needs ~40 ms of CPU per chain, and only 8 ms of that is ours (RDKit,
# segmenting, record writing) — the other ~32 ms is the HTTP + openai-SDK path,
# ~4.8 ms for each of the 6.6 requests a chain makes. That work runs inline in
# the loop, so while it runs nothing is dispatched and the servers drain toward
# idle (measured on the 2M run: in-flight under 100 for 13% of samples, GPU
# below 30% for 15%).
#
# Why 2 and not more (measured, per-shard work held constant at 3k chains):
#   1 shard   14.3 chains/s   156k prefill tok/s
#   2 shards  16.3 chains/s   181k prefill tok/s   <- +14%
# and the fleet tops out near 182-192k prefill tok/s, which a single worker
# already reaches during the windows when its loop is not blocked. So shard 3+
# has nothing left to claim on THIS fleet; it only helps if you add GPUs. Note
# also that pushing past ~1,280 in-flight made the local engine answer "upstream
# connect error" — with SHARDS>2 lower LLM_CONCURRENCY_PER_SERVER to keep
# SHARDS x 40 x #servers under that.
SHARDS="${SHARDS:-2}"

# ── NAIVE_REASONING=1 — the ablation baseline ──────────────────────────────
# Write the EDIT-ROUND reasoning — the span that picks which suggest_edits rule to
# commit — from the prompt ALONE: the user query, the molecule, the raw
# analyze_properties output, the raw suggest_edits JSON and the call about to be
# made, asked for as "the turn that goes between them", 3 sentences / 65 words. No
# Landing Safety table, no round context, no spread verdict, no rendered candidate
# comparison, no verified chemistry names, no decision order, no committed-candidate
# index, and its guards and retries off. Forces the four-call path (the merged
# prompt is built out of computed blocks, so it has no naive form).
#
# The SEED span (derive the SMARTS, transcribe it to the scaffold SMILES) is NOT
# affected — it keeps its verified ring names, aromatic-H note and retry. It is the
# one span the eval harness parses back, and with the naive prompt there 17% of seed
# spans turned into a visible scratchpad (longest 858 words) and one wrote a molecule
# of its own. The ablation is about rule selection, so this span is held fixed.
#
# This prices the scaffolding. Every claim about what the computed blocks buy is a
# claim about the difference between the two prompts, and it is only measurable if
# the same tool chains can be rendered both ways.
#
# The output subdirectory gets NAIVE_SUFFIX appended automatically, so a naive run
# can never land on top of the real corpus — with the guards off nothing stops a
# fabricated count from shipping, so this is a measurement corpus, not a training
# one, and the two must not share a directory. Override the suffix if you want a
# different name; OUTPUT_ROOT still works as usual.
#
#   NAIVE_REASONING=1 bash run_scripts/pipeline/4_sftdata_gen.sh
#     -> $OUTPUT_ROOT/generation_2m_scaffold_naive
NAIVE_REASONING="${NAIVE_REASONING:-0}"
NAIVE_SUFFIX="${NAIVE_SUFFIX:-_naive}"
PYTHON_BIN="${PYTHON_BIN:-python}"
if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="python"
fi

# Raw Arrow sidecar (_records.arrow): after each pass, (re)build the memory-mapped
# raw-record sidecar over the output dir so training startup mmaps it instead of
# re-parsing every JSONL shard. SKIP_ARROW=1 opts out; ARROW_NUM_PROC sets the
# conversion parallelism (see scripts/jsonl_to_arrow.py).
SKIP_ARROW="${SKIP_ARROW:-0}"
ARROW_NUM_PROC="${ARROW_NUM_PROC:-48}"

# vLLM (Qwen/Qwen3.6-27B), one `vllm serve` per GPU. The defaults below assume
# servers on 8080-8087. Every URL is
# reachability-probed below and the dead ones are dropped, so listing a server that
# is not up is harmless (just noisy); append URLs to round-robin over more servers,
# including other hosts: VLLM_URLS="...,http://<host>:8080/v1".
VLLM_URLS="${VLLM_URLS:-\
http://localhost:8080/v1,\
http://localhost:8081/v1,\
http://localhost:8082/v1,\
http://localhost:8083/v1,\
http://localhost:8084/v1,\
http://localhost:8085/v1,\
http://localhost:8086/v1,\
http://localhost:8087/v1,\
http://localhost:8080/v1,\
http://localhost:8081/v1,\
http://localhost:8082/v1,\
http://localhost:8083/v1,\
http://localhost:8084/v1,\
http://localhost:8085/v1,\
http://localhost:8086/v1,\
http://localhost:8087/v1}"
VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen3.6-27B}"

# Probe every URL and KEEP only the servers that are actually up, so VLLM_URLS can
# stay a superset of the fleet (list every host/port you ever bring up; whichever
# are live get used). Aborts only when nothing is reachable. SKIP_VLLM_CHECK=1
# skips the probe and uses VLLM_URLS as given.
SKIP_VLLM_CHECK="${SKIP_VLLM_CHECK:-0}"
check_vllm_fleet() {
    [ "$SKIP_VLLM_CHECK" = "1" ] && return 0
    local live=() down=0 n=0
    local IFS=','
    for url in $VLLM_URLS; do
        n=$((n + 1))
        local served
        # Short timeouts: a live /v1/models answers in milliseconds, and the list
        # may hold many hosts that are simply not up yet.
        # `|| true` matters: under `set -o pipefail` a refused connection would
        # otherwise fail the assignment and kill the script via `set -e`.
        served="$(curl -s --connect-timeout 2 -m 5 "${url%/}/models" 2>/dev/null \
            | "$PYTHON_BIN" -c 'import json,sys
try:
    print(",".join(m["id"] for m in json.load(sys.stdin)["data"]))
except Exception:
    print("")' 2>/dev/null || true)"
        if [ -z "$served" ]; then
            down=$((down + 1))
        elif [ "$served" != "$VLLM_MODEL" ]; then
            echo "[vllm][warn] skipping $url — serves '$served', expected '$VLLM_MODEL'"
        else
            echo "[vllm] ok $url ($served)"
            live+=("$url")
        fi
    done
    if [ ${#live[@]} -eq 0 ]; then
        echo "[vllm][error] none of the $n configured server(s) are up — start the" >&2
        echo "              fleet or set VLLM_URLS (SKIP_VLLM_CHECK=1 to ignore)." >&2
        exit 1
    fi
    local joined
    joined="$(IFS=','; echo "${live[*]}")"
    VLLM_URLS="$joined"
    echo "[vllm] using ${#live[@]}/$n server(s); $down unreachable (skipped)"
}

# Max concurrent requests PER vLLM server (client-side, read by llm_client.py).
# Total in-flight ≈ this × #VLLM_URLS. Raising it does NOT buy much: on the
# 4-server Qwen3.6-27B fleet, 998 chains took 228 s at 40/server vs 208 s at
# 128/server (~9%), and at 128 each engine simply queued ~110 requests while
# running only ~20 (KV cache 5% — the engine, not the client, is the limit).
# Throughput scales with the NUMBER of servers (add URLs to VLLM_URLS), not this
# knob. For reference: ~5 chains/s on 4 servers ⇒ ~4 days for 2M chains.
export LLM_CONCURRENCY_PER_SERVER="${LLM_CONCURRENCY_PER_SERVER:-40}"

# Base chain-set names: the input subdir under INPUT_ROOT and, combined with each
# entry in SUFFIXES, the output subdir under OUTPUT_ROOT. Space-separated and
# overridable like every other stage-3/4 script (3b / 4b / 4c / 4d take BASE_NAMES
# the same way).
BASE_NAMES="${BASE_NAMES:-generation_2m_scaffold}"

# Output subdir name, when it must differ from the input one — the same knob
# 3_toolchain_gen.sh takes. Reading `generation_2m_fg` while writing somewhere
# else is how a regeneration is kept away from the existing corpus instead of
# resuming into it (--resume skips a chunk whose name already exists in the
# output dir, so a rename is the only way to force a full rebuild without
# moving the old directory aside). Empty = output name follows the input name.
# It names ONE directory, so it is rejected when the run expands to more than
# one pass; NAIVE_SUFFIX is still appended on top of it.
OUTPUT_NAME="${OUTPUT_NAME:-}"

# Expand base × suffix into concrete subdir names (skip missing later).
PASSES=()
for _base in $BASE_NAMES; do
    for _suf in $SUFFIXES; do
        _sfx=""; [ "$_suf" != "plain" ] && _sfx="$_suf"
        PASSES+=("${_base}${_sfx}")
    done
done

if [ -n "$OUTPUT_NAME" ] && [ ${#PASSES[@]} -ne 1 ]; then
    echo "[error] OUTPUT_NAME names one output dir, but BASE_NAMES x SUFFIXES expands" >&2
    echo "        to ${#PASSES[@]} pass(es): ${PASSES[*]}" >&2
    exit 1
fi

cd "$PROJECT_DIR"

# Build or refresh <dir>/_records.arrow (the memory-mapped raw sidecar) unless it
# is already newer than every .jsonl in <dir>. Mirrors the freshness check the
# training loader uses, so a pure --resume run with no new data does no work.
# Non-fatal: a failure only means training falls back to slower JSONL parsing.
maybe_build_arrow() {
    local dir="$1"
    local tag; tag="$(basename "$dir")"
    if [ "$SKIP_ARROW" = "1" ]; then
        echo "[$tag] SKIP_ARROW=1 — not building _records.arrow"
        return 0
    fi
    shopt -s nullglob
    local jsonls=("$dir"/*.jsonl)
    shopt -u nullglob
    [ ${#jsonls[@]} -eq 0 ] && return 0

    local arrow="$dir/_records.arrow"
    local newest; newest="$(ls -t "$dir"/*.jsonl | head -1)"
    if [ -f "$arrow" ] && ! [ "$newest" -nt "$arrow" ]; then
        echo "[$tag] _records.arrow already up-to-date."
        return 0
    fi

    echo "[$tag] Building _records.arrow (mmap raw sidecar)…"
    "$PYTHON_BIN" "$PROJECT_DIR/4_sftdata_gen/scripts/jsonl_to_arrow.py" \
        "$dir" --num-proc "$ARROW_NUM_PROC" \
        || echo "[$tag][warn] arrow build failed; training will fall back to JSONL parsing."
}

run_pass() {
    local label="$1"
    local input_dir="$2"
    local output_dir="$3"

    if [ ! -d "$input_dir" ]; then
        echo "[$label] Input dir not found, skipping: $input_dir"
        return 0
    fi

    # The pipeline reads ONLY toolchains_*chunk_*.jsonl (pipeline.run_dir glob).
    # Stage 3 writes toolchains_generation_part_*.jsonl while it runs and merges
    # them into chunk files at the very end, so an unmerged dir would otherwise
    # yield a silent no-op run.
    shopt -s nullglob
    local files=("$input_dir"/toolchains_*chunk_*.jsonl)
    local parts=("$input_dir"/toolchains_generation_part_*.jsonl)
    shopt -u nullglob
    if [ ${#files[@]} -eq 0 ]; then
        if [ ${#parts[@]} -gt 0 ]; then
            echo "[$label][error] $input_dir holds ${#parts[@]} unmerged part file(s) and no" >&2
            echo "              chunk file(s) — stage 3 has not finished its part→chunk merge." >&2
            echo "              Wait for 3_toolchain_gen.sh to complete, then rerun." >&2
            exit 1
        fi
        echo "[$label] No toolchains_*chunk_*.jsonl in $input_dir, skipping."
        return 0
    fi
    if [ ${#parts[@]} -gt 0 ]; then
        echo "[$label][error] $input_dir mixes ${#files[@]} chunk file(s) with ${#parts[@]}" >&2
        echo "              unmerged part file(s) — stage 3 is still running or was" >&2
        echo "              interrupted mid-merge. Let it finish before generating SFT data." >&2
        exit 1
    fi

    # Stale-output guard: --resume skips a chunk whenever a file of the same name
    # exists in the output dir, and chunk names repeat across toolchain
    # regenerations. Output written BEFORE the current input was generated is
    # therefore stale — keeping it would both skip real work and mix
    # previous-format examples into training.
    if [ -d "$output_dir" ] && [ "${STALE_OK:-0}" != "1" ]; then
        shopt -s nullglob
        local outs=("$output_dir"/*.jsonl)
        shopt -u nullglob
        if [ ${#outs[@]} -gt 0 ]; then
            local newest_in oldest_out
            newest_in="$(ls -t "${files[@]}" | head -1)"
            oldest_out="$(ls -t "${outs[@]}" | tail -1)"
            if [ "$newest_in" -nt "$oldest_out" ]; then
                echo "[$label][error] $output_dir holds ${#outs[@]} JSONL file(s) older than the" >&2
                echo "              current toolchains — output from a PREVIOUS generation." >&2
                echo "              oldest output: $(basename "$oldest_out") ($(date -r "$oldest_out" '+%F %H:%M'))" >&2
                echo "              newest input : $(basename "$newest_in") ($(date -r "$newest_in" '+%F %H:%M'))" >&2
                echo "              Move it aside first (the stale _records.arrow goes with it):" >&2
                echo "                  mv '$output_dir' '${output_dir}_v1'" >&2
                echo "              Set STALE_OK=1 to keep the existing files and resume anyway." >&2
                exit 1
            fi
        fi
    fi

    mkdir -p "$output_dir"
    [ "$NAIVE_REASONING" != "0" ] && echo \
        "[$label] NAIVE_REASONING=1 — prompt-only reasoning, guards OFF (ablation)"
    echo "[$label] Input : $input_dir (${#files[@]} chunk(s))"
    echo "[$label] Output: $output_dir"

    # Optional flags (keep-unsatisfied on by default).
    local -a extra=()
    [ "$KEEP_UNSATISFIED" = "1" ] && extra+=(--keep-unsatisfied)
    [ "$NAIVE_REASONING" != "0" ] && extra+=(--naive-reasoning)

    if [ "$SHARDS" -le 1 ]; then
        "$PYTHON_BIN" -m 4_sftdata_gen \
            --input-dir "$input_dir" \
            --output-dir "$output_dir" \
            --generator-urls "$VLLM_URLS" \
            --generator-model "$VLLM_MODEL" \
            --augmentor-urls "$VLLM_URLS" \
            --augmentor-model "$VLLM_MODEL" \
            --verifier-urls "$VLLM_URLS" \
            --verifier-model "$VLLM_MODEL" \
            --batch-size "$BATCH_SIZE" \
            --num-generations "$NUM_GENERATIONS" \
            ${extra[@]+"${extra[@]}"} \
            --resume
    else
        mkdir -p "$PROJECT_DIR/logs"
        local -a pids=()
        local i log fail=0
        for i in $(seq 0 $((SHARDS - 1))); do
            log="$PROJECT_DIR/logs/sft4-${label}-shard${i}.log"
            "$PYTHON_BIN" -m 4_sftdata_gen \
                --input-dir "$input_dir" \
                --output-dir "$output_dir" \
                --generator-urls "$VLLM_URLS" \
                --generator-model "$VLLM_MODEL" \
                --augmentor-urls "$VLLM_URLS" \
                --augmentor-model "$VLLM_MODEL" \
                --verifier-urls "$VLLM_URLS" \
                --verifier-model "$VLLM_MODEL" \
                --batch-size "$BATCH_SIZE" \
                --num-generations "$NUM_GENERATIONS" \
                ${extra[@]+"${extra[@]}"} \
                --resume \
                --shard "$i/$SHARDS" \
                > "$log" 2>&1 &
            pids+=("$!")
            SFT4_PIDS+=("$!")          # what the trap will take down
            echo "[$label] shard $i/$SHARDS started (pid $!) → ${log#$PROJECT_DIR/}"
            sleep 2
        done
        echo "[$label] progress:  tail -f $PROJECT_DIR/logs/sft4-${label}-shard0.log"
        for i in "${!pids[@]}"; do
            if ! wait "${pids[$i]}"; then
                echo "[$label][error] shard $i/$SHARDS exited non-zero — see" >&2
                echo "              logs/sft4-${label}-shard${i}.log" >&2
                fail=1
            fi
        done
        SFT4_PIDS=()               # all done
        [ "$fail" = "1" ] && return 1
    fi

    # Refresh the memory-mapped raw sidecar so training startup skips re-parsing
    # these JSONL shards (skipped when already fresh; SKIP_ARROW=1 to opt out).
    # Built ONCE here, after every shard has finished.
    maybe_build_arrow "$output_dir"
}

# -- Ctrl-C handling --------------------------------------------------------
# Shards are launched with `&`, so ^C reaches this shell only: if the shell dies the
# python processes are reparented to init and keep running, and two generations end up
# writing over the same output directory. So the pids are collected globally and taken
# down explicitly on INT/TERM; anything still alive after TERM gets KILL.
SFT4_PIDS=()
_sft4_cleanup() {
    # `set -e` is on, so every step here is made non-fatal: a `kill` on an
    # already-dead pid returns non-zero and would otherwise abort the handler
    # before it finishes killing the rest.
    set +e
    local sig="${1:-INT}" pid n
    trap - INT TERM
    if [ "${#SFT4_PIDS[@]}" -gt 0 ]; then
        echo >&2
        echo "[stop] $sig — sending TERM to ${#SFT4_PIDS[@]} shard(s)…" >&2
        for pid in "${SFT4_PIDS[@]}"; do kill -TERM "$pid" 2>/dev/null; done
        for n in $(seq 1 30); do
            pgrep -P $$ >/dev/null 2>&1 || break
            local alive=0
            for pid in "${SFT4_PIDS[@]}"; do
                kill -0 "$pid" 2>/dev/null && alive=1
            done
            [ "$alive" = "0" ] && break
            sleep 0.5
        done
        for pid in "${SFT4_PIDS[@]}"; do kill -KILL "$pid" 2>/dev/null; done
    fi
    echo "[stop] stopped. --resume is on, so re-running picks up where it left off" >&2
    echo "       (the last <256 records of an interrupted chunk are regenerated)." >&2
    exit 130
}
trap '_sft4_cleanup INT' INT
trap '_sft4_cleanup TERM' TERM

check_vllm_fleet

i=1
total=${#PASSES[@]}
for sub in "${PASSES[@]}"; do
    echo ""
    echo "[$i/$total] $sub ..."
    out_sub="${OUTPUT_NAME:-$sub}"
    [ "$NAIVE_REASONING" != "0" ] && out_sub="${out_sub}${NAIVE_SUFFIX}"
    run_pass "$sub" "$INPUT_ROOT/$sub" "$OUTPUT_ROOT/$out_sub"
    i=$((i + 1))
done

echo ""
echo "Done. SFT training data under: $OUTPUT_ROOT"
