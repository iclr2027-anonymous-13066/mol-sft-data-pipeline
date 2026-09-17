#!/usr/bin/env bash
# run_scripts/pipeline/34_fg_pipeline.sh
#
# One command, end to end, for the FUNCTIONAL-GROUP instance set:
#
#   phase 3   toolchain search        CPU/ADMET bound, workers over all 8 GPUs
#   phase V   bring the vLLM fleet up Qwen3.6-27B, one engine per GPU
#   phase 4   SFT data generation     LLM-written reasoning around the fixed chains
#   phase V'  bring the fleet down    (also on Ctrl-C / error, via trap)
#
# The two compute phases want the SAME GPUs for different things — phase 3 puts an
# admet_ai copy on each, phase 4 puts a 27B engine on each — so they run strictly
# in sequence and the fleet is started only once phase 3 has finished. Starting
# vLLM early would take the memory admet_ai needs and slow phase 3 down for hours
# to save ten minutes.
#
# This is a driver: it calls 3_toolchain_gen.sh and 4_sftdata_gen.sh with the right
# env, so every knob, preflight and resume rule documented there still applies.
#
# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
#   bash run_scripts/pipeline/34_fg_pipeline.sh                 # the full 2M run
#   LIMIT=2000 bash run_scripts/pipeline/34_fg_pipeline.sh      # smoke test, ~15 min
#   RUN_STAGE3=0 bash run_scripts/pipeline/34_fg_pipeline.sh    # chains exist; do vLLM+4
#   RUN_STAGE4=0 bash run_scripts/pipeline/34_fg_pipeline.sh    # chains only, no fleet
#   KEEP_VLLM=1  bash run_scripts/pipeline/34_fg_pipeline.sh    # leave the fleet up
#
# BOTH phases resume. Ctrl-C and rerun the same command: phase 3 re-deals the
# outstanding parts, phase 4 skips chunks whose output file already exists.
#
# ---------------------------------------------------------------------------
# Runtime for the full 1,999,127-instance set (288 cores + 8x B200)
# ---------------------------------------------------------------------------
# Phase 3 — MEASURED 2026-08-09 on generation_2m_fg itself, beam w4xe4, 48 workers:
#     ~9.1 instances/s steady state  ->  ~61 h  ~= 2.5 days
#   CPU-bound: GPUs sat under 40% and only hold the admet_ai copies. Throughput
#   plateaus at 32-48 workers, so raising NUM_PROCS past that does not help (and
#   past ~96 the simultaneous admet_ai loads thrash startup). SEARCH_MODE=greedy is
#   3-4x faster (~0.8 days) at a lower satisfied rate, if schedule beats yield.
#
# Phase 4 — MEASURED 2026-08-10, 8 local engines: ~5.2 chains/s
#     20k chains ~= 65 min | 500k ~= 27 h | 2M ~= 4.4 days
#   With VLLM_EXTRA_HOSTS adding a second 8-GPU box (16 engines, the default), the
#   work is prefill-bound and scales with engine COUNT, so expect roughly double:
#     ~10 chains/s -> 20k ~= 32 min | 500k ~= 13 h | 2M ~= 2.2 days.  Not yet
#   measured at 16 — verify against the first chunk's rate before trusting it.
#   Do NOT compare this to the scaffold run's "20k in 37 min" (9.0 chains/s) and
#   conclude something is broken. The unit of LLM work is the SEGMENT, not the
#   chain, and an FG chain is simply a bigger job:
#                       edit steps   tool calls   segments/chain
#       scaffold           1.30         4.89          2.85
#       functional group   3.43        11.30          5.00
#   Converted to segments/s the two runs are indistinguishable — scaffold 25.7,
#   FG 26.0 — i.e. the fleet does the same work per second and an FG chain just
#   takes ~1.75x more of it. The remaining gap to 9.0 chains/s is arithmetic, not
#   a regression.
#
# So end to end at LIMIT=2M: ~2.5 days phase 3 + ~4.4 days phase 4 ~= 7 days;
# both scale linearly with LIMIT. Fleet startup is ~5-10 min and rounding error.
# Disk: phase 3 writes ~20 GB, phase 4 ~130 GB (measured on the scaffold 2M run).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_DIR"

# ---------------------------------------------------------------------------
# Config (override via env vars)
# ---------------------------------------------------------------------------
INSTANCES="${INSTANCES:-data/training_data/instances}"
INPUT="${INPUT:-$INSTANCES/generation_2m_fg}"
TOOLCHAIN_ROOT="${TOOLCHAIN_ROOT:-data/training_data/toolchain}"
SFTDATA_ROOT="${SFTDATA_ROOT:-data/training_data/sftdata}"

# Subdir name used by BOTH phases: phase 3 writes $TOOLCHAIN_ROOT/$NAME, phase 4
# reads it and writes $SFTDATA_ROOT/$NAME. Derived from the input like the stage
# scripts do, so overriding INPUT alone is enough.
NAME="${NAME:-}"
if [ -z "$NAME" ]; then
    NAME="$(basename "$INPUT")"; NAME="${NAME%.jsonl}"
fi

# Phase switches — set to 0 to re-enter the pipeline part-way.
RUN_STAGE3="${RUN_STAGE3:-1}"
RUN_STAGE4="${RUN_STAGE4:-1}"
KEEP_VLLM="${KEEP_VLLM:-0}"       # 1 = leave the fleet running at the end

# Phase 3 knobs (see 3_toolchain_gen.sh for the full set and the tuning notes).
SEARCH_MODE="${SEARCH_MODE:-beam}"
NUM_PROCS="${NUM_PROCS:-100}"
PER_PROC="${PER_PROC:-4}"
TOOLCHAIN_GPUS="${TOOLCHAIN_GPUS:-0,1,2,3,4,5,6,7}"
CHUNK_SIZE="${CHUNK_SIZE:-20000}"
PART_SIZE="${PART_SIZE:-1000}"
LIMIT="${LIMIT:-500000}"                # smoke runs; empty = every instance

# Phase 4 knobs (see 4_sftdata_gen.sh).
VLLM_GPUS="${VLLM_GPUS:-0,1,2,3,4,5,6,7}"
VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen3.6-27B}"
VLLM_BASE_PORT="${VLLM_BASE_PORT:-8080}"
VLLM_WAIT_SECS="${VLLM_WAIT_SECS:-1800}"   # how long to wait for the fleet to load
SHARDS="${SHARDS:-2}"

# Concurrency is specified PER ENGINE — one vLLM process on one GPU — because that
# is the thing that actually has a queue. 40 in flight per engine, so 320 per
# 8-GPU node and 40 x N_VLLM_URLS across the whole fleet.
CONCURRENCY_PER_ENGINE="${CONCURRENCY_PER_ENGINE:-80}"

# llm_client.py's knob is per URL *per shard PROCESS*, and the shards dispatch
# independently, so an engine sees SHARDS times whatever it is set to. Divide, or
# raising SHARDS would quietly multiply the load on every engine — at SHARDS=2,
# LLM_CONCURRENCY_PER_SERVER=40 means 80 per engine, not 40.
LLM_CONCURRENCY_PER_SERVER="${LLM_CONCURRENCY_PER_SERVER:-$((CONCURRENCY_PER_ENGINE / SHARDS))}"
[ "$LLM_CONCURRENCY_PER_SERVER" -ge 1 ] || LLM_CONCURRENCY_PER_SERVER=1
if [ $((CONCURRENCY_PER_ENGINE % SHARDS)) -ne 0 ]; then
    echo "[warn] CONCURRENCY_PER_ENGINE=$CONCURRENCY_PER_ENGINE is not divisible by" \
         "SHARDS=$SHARDS — engines will see $((LLM_CONCURRENCY_PER_SERVER * SHARDS)) each." >&2
fi
KEEP_UNSATISFIED="${KEEP_UNSATISFIED:-1}"

# Other HOSTS serving the same model, appended to the locally started fleet. These
# are NOT started or stopped by this script — it only dispatches to them, and
# stop_fleet touches local pidfiles only, so a shared remote fleet is never torn
# down from here. Dead entries cost nothing: 4_sftdata_gen.sh probes every URL and
# drops the unreachable ones.
#
# One address that must NOT go here: this machine's own address (hostname -I),
# so 4_sftdata_gen.sh's stock list names the eight local engines twice. Duplicates
# add no capacity — they only inflate the client-side budget, because
# LLM_CONCURRENCY_PER_SERVER is applied per URL rather than per engine.
# Note ${VAR-default}, not ${VAR:-default}: an explicitly EMPTY VLLM_EXTRA_HOSTS=""
# must mean "localhost only", which the colon form would silently override.
VLLM_EXTRA_HOSTS="${VLLM_EXTRA_HOSTS-localhost}"

# One URL per GPU in VLLM_GPUS (port = base + gpu id), matching the Makefile's
# one-engine-per-GPU layout.
#
# TWO lists, deliberately:
#   VLLM_LOCAL_URLS  the engines THIS script starts and waits for
#   VLLM_URLS        everything phase 4 may dispatch to (local + extra hosts)
# Conflating them would break start_fleet: with 8 remote engines already live, a
# combined probe reports the fleet "already up" and the local engines never start.
_local=()
for g in $(echo "$VLLM_GPUS" | tr ',' ' '); do
    _local+=("http://localhost:$((VLLM_BASE_PORT + g))/v1")
done
VLLM_LOCAL_URLS="$(IFS=','; echo "${_local[*]}")"

if [ -z "${VLLM_URLS:-}" ]; then
    _all=("${_local[@]}")
    for h in $(echo "$VLLM_EXTRA_HOSTS" | tr ',' ' '); do
        [ -n "$h" ] || continue
        for g in $(echo "$VLLM_GPUS" | tr ',' ' '); do
            _all+=("http://$h:$((VLLM_BASE_PORT + g))/v1")
        done
    done
    VLLM_URLS="$(IFS=','; echo "${_all[*]}")"
fi

# How many URLs phase 4 may dispatch to (counted from VLLM_URLS so an explicit
# override is measured too, not just the list built above).
N_VLLM_URLS=0
for _u in $(echo "$VLLM_URLS" | tr ',' ' '); do
    [ -n "$_u" ] && N_VLLM_URLS=$((N_VLLM_URLS + 1))
done

# BATCH_SIZE is NOT a batch — pipeline.py uses it as asyncio.Semaphore(batch_size),
# i.e. the number of chains a shard works on CONCURRENTLY. A chain issues its
# segments one after another, so it holds at most one request in flight, and the
# fleet only stays saturated while
#     BATCH_SIZE  >=  N_VLLM_URLS x LLM_CONCURRENCY_PER_SERVER
# Below that the semaphore, not the engines, is the limit: with 16 URLs x 40 = 640
# request slots, a BATCH_SIZE of 320 would leave half the fleet idle. So derive it
# from the fleet instead of hardcoding a number that silently goes stale whenever a
# host is added. (16 x 40 = 640, which is what the scaffold run effectively used.)
BATCH_SIZE="${BATCH_SIZE:-$((N_VLLM_URLS * LLM_CONCURRENCY_PER_SERVER))}"

PYTHON_BIN="${PYTHON_BIN:-python}"
[ -x "$PYTHON_BIN" ] || PYTHON_BIN="python"

TOOLCHAIN_DIR="$TOOLCHAIN_ROOT/$NAME"
SFTDATA_DIR="$SFTDATA_ROOT/$NAME"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR" "$TOOLCHAIN_ROOT" "$SFTDATA_ROOT"

# ---------------------------------------------------------------------------
# Preflight — fail before spending days, not after
# ---------------------------------------------------------------------------
preflight() {
    local err=0

    if [ ! -e "$INPUT" ]; then
        echo "ERROR: input not found: $INPUT" >&2; err=1
    elif [ -d "$INPUT" ]; then
        shopt -s nullglob; local shards=("$INPUT"/*.jsonl); shopt -u nullglob
        if [ ${#shards[@]} -eq 0 ]; then
            echo "ERROR: no .jsonl shards in $INPUT" >&2; err=1
        else
            echo "  input        : $INPUT (${#shards[@]} shard(s))"
        fi
    else
        echo "  input        : $INPUT (single file)"
    fi

    if [ ! -x "$PYTHON_BIN" ]; then
        echo "ERROR: python not executable: $PYTHON_BIN (set PYTHON_BIN=...)" >&2; err=1
    fi

    # The FG path needs the fields stage 2b writes. A scaffold-era instance set
    # would silently plan scaffold chains instead, so check one record.
    if [ -e "$INPUT" ] && [ -x "$PYTHON_BIN" ]; then
        local probe
        probe="$(shopt -s nullglob
                 if [ -d "$INPUT" ]; then f=("$INPUT"/*.jsonl); echo "${f[0]}"; else echo "$INPUT"; fi)"
        if ! head -1 "$probe" | "$PYTHON_BIN" -c '
import json, sys
r = json.loads(sys.stdin.readline())
missing = [k for k in ("fg_smarts", "eval_query", "seed_smiles", "description") if not r.get(k)]
if missing:
    print("ERROR: %s lacks functional-group fields: %s" % (sys.argv[1], ", ".join(missing)),
          file=sys.stderr)
    print("       Build it with run_scripts/pipeline/2b_functional_group_gen.sh.", file=sys.stderr)
    raise SystemExit(1)
print("  constraint   : functional group (%d SMARTS on the first record)" % len(r["fg_smarts"]))
' "$(basename "$probe")"; then
            err=1
        fi
    fi

    # ~150 GB for a full 2M run (20 phase 3 + 130 phase 4), measured on the
    # scaffold set. Only a warning: LIMIT runs need a small fraction of it.
    local avail_gb
    avail_gb="$(df -BG --output=avail "$TOOLCHAIN_ROOT" 2>/dev/null | tail -1 | tr -dc '0-9')"
    if [ -n "$avail_gb" ]; then
        echo "  free disk    : ${avail_gb} GB on $(df --output=target "$TOOLCHAIN_ROOT" | tail -1)"
        if [ -z "$LIMIT" ] && [ "$avail_gb" -lt 200 ]; then
            echo "  [warn] a full run writes ~150 GB; only ${avail_gb} GB free." >&2
        fi
    fi

    [ "$err" = "0" ] || exit 1
}

# ---------------------------------------------------------------------------
# vLLM fleet
# ---------------------------------------------------------------------------
# Every URL that answers /v1/models with the expected model id.
fleet_live() {   # $1 = comma-separated URLs; defaults to the LOCAL fleet only
    local live=0
    local urls="${1:-$VLLM_LOCAL_URLS}"
    local IFS=','
    for url in $urls; do
        local served
        served="$(curl -s --connect-timeout 2 -m 5 "${url%/}/models" 2>/dev/null \
            | "$PYTHON_BIN" -c 'import json,sys
try:
    print(",".join(m["id"] for m in json.load(sys.stdin)["data"]))
except Exception:
    print("")' 2>/dev/null || true)"
        if [ "$served" = "$VLLM_MODEL" ]; then live=$((live + 1)); fi
    done
    echo "$live"
}

start_fleet() {
    # `grep -c` exits 1 on a zero count, which under `set -e` would kill the script
    # inside a command substitution — count with a loop instead.
    local want=0 g
    for g in $(echo "$VLLM_GPUS" | tr ',' ' '); do want=$((want + 1)); done
    if [ "$want" -eq 0 ]; then
        echo "ERROR: VLLM_GPUS is empty — nothing to serve on." >&2; exit 1
    fi
    local live; live="$(fleet_live)"
    if [ "$live" -ge "$want" ]; then
        echo "  fleet already up ($live/$want) — not starting anything."
        VLLM_STARTED_BY_US=0
        return 0
    fi

    # One engine per GPU, TP=1, bound at port = VLLM_BASE_PORT + GPU id, which is what
    # the VLLM_URLS built above assume. Nothing here wants a wider engine: phase 4 is
    # throughput-bound on many short concurrent requests, so N independent engines beat
    # N/2 tensor-parallel ones on the same GPUs.
    echo "  starting $want engine(s) on GPU(s) $VLLM_GPUS (one per GPU, TP=1) ..."
    mkdir -p "$LOG_DIR"
    for _g in $(echo "$VLLM_GPUS" | tr ',' ' '); do
        _port=$((VLLM_BASE_PORT + _g))
        CUDA_VISIBLE_DEVICES="$_g" nohup vllm serve "$VLLM_MODEL" \
            --served-model-name "$VLLM_MODEL" \
            --trust-remote-code \
            --tensor-parallel-size 1 \
            --gpu-memory-utilization 0.9 \
            --enable-prefix-caching \
            --host 0.0.0.0 --port "$_port" \
            > "$LOG_DIR/vllm-g$_g.log" 2>&1 &
        echo $! > "$LOG_DIR/vllm-g$_g.pid"
    done
    VLLM_STARTED_BY_US=1

    # A 27B engine takes minutes to load, so poll rather than guess — but stop as
    # soon as the count PLATEAUS instead of always burning VLLM_WAIT_SECS. If the
    # launch above is ever changed to one engine per GPU *pair*, only half the
    # expected ports will ever answer and waiting out the full timeout would cost
    # half an hour to learn that. A plateau of PLATEAU_POLLS consecutive unchanged
    # readings, with at least one engine live, is treated as "this is the fleet".
    local waited=0 stable=0 prev=-1
    local PLATEAU_POLLS="${VLLM_PLATEAU_POLLS:-8}"     # 8 x 15s = 2 min unchanged
    while [ "$waited" -lt "$VLLM_WAIT_SECS" ]; do
        live="$(fleet_live)"
        if [ "$live" -ge "$want" ]; then
            echo ""
            echo "  fleet ready: $live/$want local engine(s) after ${waited}s"
            echo "  total dispatch targets live: $(fleet_live "$VLLM_URLS") engine(s)"
            return 0
        fi
        if [ "$live" -eq "$prev" ]; then stable=$((stable + 1)); else stable=0; fi
        prev="$live"
        if [ "$live" -gt 0 ] && [ "$stable" -ge "$PLATEAU_POLLS" ]; then
            echo ""
            echo "  [warn] $live/$want engine(s) up and unchanged for $((PLATEAU_POLLS * 15))s —" >&2
            echo "         treating that as the whole fleet. If you expected $want, check that" >&2
            echo "         every engine really is serving ONE GPU (TP=1) at" >&2
            echo "         port = VLLM_BASE_PORT + gpu id; see $LOG_DIR/vllm-g*.log." >&2
            return 0
        fi
        printf '\r  waiting for engines: %s/%s up (%ss)   ' "$live" "$want" "$waited"
        sleep 15; waited=$((waited + 15))
    done
    echo ""

    live="$(fleet_live)"
    if [ "$live" -eq 0 ]; then
        echo "ERROR: no engine came up within ${VLLM_WAIT_SECS}s — see $LOG_DIR/vllm-qwen-27b-g*.log" >&2
        exit 1
    fi
    # Phase 4 probes the URLs itself and drops the dead ones, so a partial fleet is
    # usable — just slower. Say so instead of failing after a long wait.
    echo "  [warn] only $live/$want engine(s) up after ${VLLM_WAIT_SECS}s; continuing on those." >&2
}

stop_fleet() {
    [ "${VLLM_STARTED_BY_US:-0}" = "1" ] || return 0
    if [ "$KEEP_VLLM" = "1" ]; then
        echo "  KEEP_VLLM=1 — leaving the fleet up (stop it by killing the vllm processes)"
        return 0
    fi
    echo "  stopping the vLLM fleet ..."
    for _g in $(echo "$VLLM_GPUS" | tr ',' ' '); do
        _pf="$LOG_DIR/vllm-g$_g.pid"
        [ -f "$_pf" ] || continue
        kill "$(cat "$_pf")" 2>/dev/null || true
        rm -f "$_pf"
    done
}

# ---------------------------------------------------------------------------
# Stage-4 worker tracking
# ---------------------------------------------------------------------------
# 4_sftdata_gen.sh launches its shards with `&` and waits, so killing THIS script
# leaves the shard pythons running: they are re-parented to init and keep writing
# to the same output files. Two facts make that expensive rather than merely
# untidy — measured on 2026-08-10:
#   * Two writers on one chunk file silently destroy each other's records. One run
#     holds a plain write offset and the other appends, so the first overwrites
#     what the second appended: 13.0k records generated, 7.7k survived on disk.
#   * The EXIT trap used to stop only the fleet, so a Ctrl-C left the orphans
#     hammering vLLM servers that no longer existed, logging APIConnectionError
#     per input and burning through the chunk producing nothing.
# So: refuse to start when workers are already on this output dir, and take our
# own workers down with us — before the fleet, never after.

# PIDs of stage-4 processes writing to $SFTDATA_DIR. Matched against the exact
# argv in /proc rather than a pgrep regex, so a path with dots or a substring of
# a longer directory name cannot produce a false hit.
running_stage4_pids() {
    local p
    for p in $(pgrep -f 'SFT\.4_sftdata_gen' 2>/dev/null || true); do
        if tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null \
                | grep -qF -- "--output-dir $SFTDATA_DIR "; then
            echo "$p"
        fi
    done
}

assert_no_concurrent_stage4() {
    local pids; pids="$(running_stage4_pids | tr '\n' ' ')"
    pids="${pids%% }"
    [ -n "${pids// /}" ] || return 0
    if [ "${STAGE4_FORCE:-0}" = "1" ]; then
        echo "  [warn] STAGE4_FORCE=1 — starting anyway alongside pid(s): $pids" >&2
        return 0
    fi
    echo "ERROR: stage 4 is ALREADY writing to $SFTDATA_DIR" >&2
    echo "       pid(s): $pids" >&2
    echo "       A second writer would silently overwrite its records (~40% loss" >&2
    echo "       measured). These are usually orphans of a killed run — check, then:" >&2
    echo "           kill $pids" >&2
    echo "       STAGE4_FORCE=1 overrides, but only do that if they are really gone." >&2
    exit 1
}

# Every descendant of $1, deepest last. Used instead of a pattern match so cleanup
# can only ever reach processes THIS script fathered.
descendants_of() {
    local queue=("$1") out=() p child
    while [ ${#queue[@]} -gt 0 ]; do
        p="${queue[0]}"; queue=("${queue[@]:1}")
        for child in $(pgrep -P "$p" 2>/dev/null || true); do
            out+=("$child"); queue+=("$child")
        done
    done
    [ ${#out[@]} -eq 0 ] || printf '%s\n' "${out[@]}"
}

stop_stage4() {
    # Two conditions, both required, because getting this wrong is destructive:
    #   * STAGE4_LAUNCHED — we actually started a stage 4 (mirrors
    #     stop_fleet's VLLM_STARTED_BY_US), so the refuse-to-start path does not
    #     clean up a run it just declined to race.
    #   * descendants of STAGE4_PID — only OUR workers. This used to match every
    #     stage-4 process on the output dir, which meant an aborting second
    #     invocation would kill the healthy first one: exactly the damage the
    #     guard exists to prevent. running_stage4_pids stays in use for the
    #     refuse-to-start CHECK, where matching other people's processes is the
    #     entire point, but it must never drive a kill.
    [ "${STAGE4_LAUNCHED:-0}" = "1" ] || return 0
    [ -n "${STAGE4_PID:-}" ] || return 0

    local pids; pids="$(descendants_of "$STAGE4_PID" | tr '\n' ' ')"
    # The 4_sftdata_gen.sh shell itself last in the list but killed first, so it
    # cannot start the next chunk while we take this one down.
    pids="$STAGE4_PID $pids"
    echo "  stopping stage-4 (pid $STAGE4_PID + $(descendants_of "$STAGE4_PID" | wc -l) child(ren))"
    # shellcheck disable=SC2086
    kill $pids 2>/dev/null || true
    local waited=0 alive
    while [ "$waited" -lt 20 ]; do
        alive=""
        for p in $pids; do kill -0 "$p" 2>/dev/null && alive="$alive $p"; done
        [ -n "${alive// /}" ] || return 0
        sleep 2; waited=$((waited + 2))
    done
    echo "  [warn] still alive after 20s — SIGKILL:$alive" >&2
    # shellcheck disable=SC2086
    kill -9 $alive 2>/dev/null || true
}

# Workers FIRST, then the fleet: the reverse order is what left orphans spinning
# on connection errors. Guarded so the INT/TERM and EXIT traps cannot both run it.
_CLEANED=0
cleanup() {
    [ "$_CLEANED" = "1" ] && return 0
    _CLEANED=1
    stop_stage4
    stop_fleet
}
trap cleanup EXIT INT TERM

# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------
hms() {   # seconds -> 1h 02m 03s
    local s=$1
    printf '%dh %02dm %02ds' $((s / 3600)) $(((s % 3600) / 60)) $((s % 60))
}

banner() {
    echo ""
    echo "============================================================================"
    echo "  $*"
    echo "============================================================================"
}

T3=0; TV=0; T4=0

run_stage3() {
    banner "Phase 3 — tool chains   $NAME"
    local t0=$SECONDS
    INPUT="$INPUT" \
    OUTPUT_DIR="$TOOLCHAIN_ROOT" \
    OUTPUT_NAME="$NAME" \
    SEARCH_MODE="$SEARCH_MODE" \
    NUM_PROCS="$NUM_PROCS" \
    PER_PROC="$PER_PROC" \
    GPUS="$TOOLCHAIN_GPUS" \
    CHUNK_SIZE="$CHUNK_SIZE" \
    PART_SIZE="$PART_SIZE" \
    LIMIT="${LIMIT:-}" \
    PYTHON="$PYTHON_BIN" \
        bash "$SCRIPT_DIR/3_toolchain_gen.sh"
    T3=$((SECONDS - t0))
    echo "  phase 3 took $(hms $T3)"
}

run_stage4() {
    # Re-check here as well as in preflight: phase 3 can run for days, and someone
    # may have started a second pipeline in the meantime.
    assert_no_concurrent_stage4

    banner "Phase V — vLLM fleet"
    local t0=$SECONDS
    start_fleet
    TV=$((SECONDS - t0))
    echo "  fleet up in $(hms $TV)"

    banner "Phase 4 — SFT data   $NAME"
    t0=$SECONDS
    STAGE4_LAUNCHED=1          # from here on, the cleanup trap owns these workers
    LLM_CONCURRENCY_PER_SERVER="$LLM_CONCURRENCY_PER_SERVER" \
    INPUT_ROOT="$TOOLCHAIN_ROOT" \
    OUTPUT_ROOT="$SFTDATA_ROOT" \
    BASE_NAMES="$NAME" \
    SUFFIXES="plain" \
    VLLM_URLS="$VLLM_URLS" \
    VLLM_MODEL="$VLLM_MODEL" \
    SHARDS="$SHARDS" \
    BATCH_SIZE="$BATCH_SIZE" \
    KEEP_UNSATISFIED="$KEEP_UNSATISFIED" \
    PYTHON_BIN="$PYTHON_BIN" \
        bash "$SCRIPT_DIR/4_sftdata_gen.sh" &
    # Run it in the BACKGROUND and `wait`, rather than calling it directly. bash
    # defers a trap until the current foreground command returns, so a direct call
    # would sit on the signal for the DAYS phase 4 runs — a TERM aimed at this
    # script alone (dead terminal, kill by pid) would do nothing until the stage
    # finished. `wait` is interruptible, so cleanup runs at once.
    local rc=0
    STAGE4_PID=$!
    wait "$STAGE4_PID" || rc=$?
    STAGE4_PID=""
    T4=$((SECONDS - t0))
    if [ "$rc" != "0" ]; then
        echo "  phase 4 exited non-zero ($rc) after $(hms $T4)" >&2
        return "$rc"
    fi
    echo "  phase 4 took $(hms $T4)"
}

# File count + on-disk size. Deliberately NOT a row count: phase 4 writes ~130 GB
# and `wc -l` over that costs minutes of pure I/O for a number nothing depends on.
describe_output() {
    local dir="$1" pattern="$2"
    [ -d "$dir" ] || { echo "(missing)"; return; }
    shopt -s nullglob; local files=("$dir"/$pattern); shopt -u nullglob
    [ ${#files[@]} -eq 0 ] && { echo "(no $pattern)"; return; }
    echo "${#files[@]} file(s), $(du -sh "$dir" 2>/dev/null | cut -f1)"
}

# ---------------------------------------------------------------------------
banner "FG pipeline: tool chains -> vLLM -> SFT data"
echo "  name         : $NAME"
preflight
echo "  toolchains   : $TOOLCHAIN_DIR"
echo "  sft data     : $SFTDATA_DIR"
echo "  python       : $PYTHON_BIN"
echo "  phase 3      : $([ "$RUN_STAGE3" = 1 ] && echo "run ($SEARCH_MODE, $NUM_PROCS procs, GPUs $TOOLCHAIN_GPUS)" || echo skip)"
echo "  phase 4      : $([ "$RUN_STAGE4" = 1 ] && echo "run ($SHARDS shard(s), local fleet on GPUs $VLLM_GPUS)" || echo skip)"
if [ "$RUN_STAGE4" = "1" ]; then
    _n_gpus=0; for _g in $(echo "$VLLM_GPUS" | tr ',' ' '); do _n_gpus=$((_n_gpus + 1)); done
    echo "  fleet        : $N_VLLM_URLS engine(s)$([ -n "$VLLM_EXTRA_HOSTS" ] && echo " (localhost + $VLLM_EXTRA_HOSTS)")," \
         "$_n_gpus per node"
    echo "  concurrency  : $((LLM_CONCURRENCY_PER_SERVER * SHARDS))/engine" \
         "($LLM_CONCURRENCY_PER_SERVER per shard x $SHARDS shard(s))," \
         "$((LLM_CONCURRENCY_PER_SERVER * SHARDS * _n_gpus))/node," \
         "$((N_VLLM_URLS * LLM_CONCURRENCY_PER_SERVER * SHARDS)) in flight total"
    echo "  BATCH_SIZE   : $BATCH_SIZE concurrent chains/shard" \
         "(needs >= $((N_VLLM_URLS * LLM_CONCURRENCY_PER_SERVER)) to keep the fleet busy)"
fi
if [ -n "$LIMIT" ]; then echo "  LIMIT        : $LIMIT  (smoke run)"; fi
# Fail before phase 3 burns days only to hit this at the phase-4 handoff.
if [ "$RUN_STAGE4" = "1" ]; then assert_no_concurrent_stage4; fi

START=$SECONDS
if [ "$RUN_STAGE3" = "1" ]; then run_stage3; fi
if [ "$RUN_STAGE4" = "1" ]; then run_stage4; fi
TOTAL=$((SECONDS - START))

banner "Done in $(hms $TOTAL)"
if [ "$RUN_STAGE3" = "1" ]; then
    echo "  phase 3 : $(hms $T3)   $(describe_output "$TOOLCHAIN_DIR" 'toolchains_*chunk_*.jsonl')"
    echo "            -> $TOOLCHAIN_DIR"
fi
if [ "$RUN_STAGE4" = "1" ]; then
    echo "  fleet   : $(hms $TV)"
    echo "  phase 4 : $(hms $T4)   $(describe_output "$SFTDATA_DIR" '*.jsonl')"
    echo "            -> $SFTDATA_DIR"
fi
echo ""
echo "Next: the optional post-passes take BASE_NAMES=$NAME the same way —"
echo "  4c_make_satisfied_only.sh   keep only chains that satisfied every constraint"
echo "  4d_qc_authored.sh           QC report over the generated data"
