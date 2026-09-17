#!/usr/bin/env bash
# run_scripts/pipeline/2_substructure_gen.sh
#
# Generate the scaffold-description data for the scaffold-constrained generation task.
#
# For each input record (one molecule per line) this reduces the molecule to its
# Bemis-Murcko scaffold, names/analyzes it deterministically, polishes a faithful
# natural-language description with the vLLM pool (Qwen3), verifies it by RDKit
# substructure check, and attaches the per-dimension / match-any evaluation SMARTS
# (`dimension_smarts`, `eval_query`). The original record keys are kept unchanged and
# the scaffold fields are appended (1:1, input order preserved).
#
# Output is written as fixed-size JSONL shards (default 10000 rows/file) into a NEW
# folder: `<prefix>-00000.jsonl`, `-00001.jsonl`, ... Sharded writing is streaming,
# atomic per shard, and RESUMABLE — re-running skips already-complete shards, so an
# interrupted run continues from the first missing/incomplete shard.
#
# This produces the `*_scaffold` folder that `3_toolchain_gen.sh` consumes as INPUT.
#
# Prerequisites:
#   A vLLM pool serving Qwen/Qwen3.6-27B must be up. Start one server per GPU (or
#   per tensor-parallel group), e.g.
#       CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3.6-27B --port 8080 \
#           --served-model-name Qwen/Qwen3.6-27B --trust-remote-code \
#           --max-model-len 131072 --enable-prefix-caching
#   and list every resulting host:port in SERVERS below.
#   Check: curl -s http://localhost:8080/v1/models
#   NOTE: unlike stage 4, this script does NOT probe the pool — an unreachable
#   entry in SERVERS costs a per-request timeout on every record, so keep the list
#   equal to what is actually running.
#
# Usage (2M run, the default):
#   bash run_scripts/pipeline/2_substructure_gen.sh
#
# Override defaults with env vars, e.g. a small smoke run:
#   INPUT=/path/in.jsonl OUT_DIR=/path/out_scaffold SHARD_SIZE=1000 \
#   SMILES_KEY=ref_smiles CONCURRENCY=24 \
#     bash run_scripts/pipeline/2_substructure_gen.sh
#
#   # a different pool (host:portspec, comma-separated):
#   SERVERS="localhost:8080,localhost:8082" \
#     bash run_scripts/pipeline/2_substructure_gen.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
SUBSTRUCT_DIR="$PROJECT_DIR/2_substructure_gen"

# ---------------------------------------------------------------------------
# Config (override via env vars)
# ---------------------------------------------------------------------------
INPUT="${INPUT:-data/training_data/instances/generation_2m.jsonl}"
OUT_DIR="${OUT_DIR:-data/training_data/instances/generation_2m_scaffold}"
SHARD_SIZE="${SHARD_SIZE:-20000}"
SMILES_KEY="${SMILES_KEY:-ref_smiles}"
CONCURRENCY="${CONCURRENCY:-30}"            # concurrent requests per vLLM server
ANALYZER_PROCS="${ANALYZER_PROCS:-0}"       # worker processes for the CPU analysis (analyze_one). 0 = auto (min(32, cpu))
SHARD_PREFIX="${SHARD_PREFIX:-}"           # default: input filename stem
# One entry per RUNNING vLLM server. The default assumes eight servers on
# 8080-8087; adjust to match however many you started.
SERVERS="${SERVERS:-localhost:8080,localhost:8081,localhost:8082,localhost:8083,localhost:8084,localhost:8085,localhost:8086,localhost:8087}"

# ---------------------------------------------------------------------------
# Python resolution — prefer a caller-provided PYTHON, otherwise the conda env
# that has rdkit + openai for this pipeline (molkit), otherwise python3.
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

# ---------------------------------------------------------------------------
# vLLM reachability preflight (mirrors 4_sftdata_gen.sh)
# ---------------------------------------------------------------------------
# describe_scaffolds does NOT probe: an unreachable entry in SERVERS burns a
# per-request timeout on every record, which over a 2M run is ruinous. So expand
# SERVERS to host:port pairs, keep only the ones actually serving VLLM_MODEL, and
# hand the survivors back. SKIP_VLLM_CHECK=1 skips this and uses SERVERS as given.
SKIP_VLLM_CHECK="${SKIP_VLLM_CHECK:-0}"
VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen3.6-27B}"

check_servers() {
    [ "$SKIP_VLLM_CHECK" = "1" ] && return 0
    local pairs
    pairs="$("$PYTHON" -c '
import sys
sys.path.insert(0, ".")
from describe_scaffolds import build_base_urls
for u in build_base_urls(sys.argv[1], None, "8080-8087"):
    print(u)
' "$SERVERS" 2>/dev/null || true)"
    if [ -z "$pairs" ]; then
        echo "[vllm][error] SERVERS did not expand to any URL: '$SERVERS'" >&2
        exit 1
    fi
    local live=() n=0 down=0 url hostport served
    while IFS= read -r url; do
        [ -n "$url" ] || continue
        n=$((n + 1))
        served="$(curl -s --connect-timeout 2 -m 5 "${url%/}/models" 2>/dev/null \
            | "$PYTHON" -c 'import json,sys
try:
    print(",".join(m["id"] for m in json.load(sys.stdin)["data"]))
except Exception:
    print("")' 2>/dev/null || true)"
        hostport="${url#http://}"; hostport="${hostport%/v1}"
        if [ -z "$served" ]; then
            down=$((down + 1))
        elif [ "$served" != "$VLLM_MODEL" ]; then
            echo "[vllm][warn] skipping $hostport — serves '$served', expected '$VLLM_MODEL'"
            down=$((down + 1))
        else
            echo "[vllm] ok $hostport ($served)"
            live+=("$hostport")
        fi
    done <<< "$pairs"
    if [ ${#live[@]} -eq 0 ]; then
        echo "[vllm][error] none of the $n configured server(s) are up — start the pool" >&2
        echo "              (start a vLLM server, see the header) or set SERVERS." >&2
        echo "              SKIP_VLLM_CHECK=1 to bypass this probe." >&2
        exit 1
    fi
    SERVERS="$(IFS=','; echo "${live[*]}")"
    echo "[vllm] using ${#live[@]}/$n server(s); $down skipped  ->  $SERVERS"
}

cd "$SUBSTRUCT_DIR"
check_servers

echo "Generating scaffold descriptions..."
echo "  input    : $INPUT"
echo "  out-dir  : $OUT_DIR  (shards of $SHARD_SIZE rows)"
echo "  smiles   : $SMILES_KEY"

CMD=("$PYTHON" augment_jsonl_with_scaffold.py
    --input "$INPUT"
    --out-dir "$OUT_DIR"
    --shard-size "$SHARD_SIZE"
    --smiles-key "$SMILES_KEY"
    --concurrency-per-server "$CONCURRENCY"
    --analyzer-procs "$ANALYZER_PROCS")

if [ -n "$SHARD_PREFIX" ]; then
    CMD+=(--shard-prefix "$SHARD_PREFIX")
fi

if [ -n "$SERVERS" ]; then
    CMD+=(--servers "$SERVERS")
fi

"${CMD[@]}"
