"""CLI entry point for the training data generation pipeline.

Usage (run from the project root):
    # Process a single chunk:
    python -m 4_sftdata_gen --input_path input.jsonl -o output.jsonl

    # Process all chunks in frag_2 directory (default):
    python -m 4_sftdata_gen --input-dir /data/.../frag_2 --output-dir /data/.../out/frag_2
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import random
import sys
import os

from .config import PipelineConfig, VLLMConfig
from .pipeline import TrainingDataPipeline

_DEFAULT_INPUT_DIR = (
    "data/training_data/toolchain/generation_benchmark_scaffold"
)
_DEFAULT_OUTPUT_DIR = (
    "data/training_data/sftdata/generation_benchmark_scaffold"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate training data for chemistry tool-calling LLMs",
    )

    # ── Input / output ──────────────────────────────────────────────────────
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--input-dir",
        default=_DEFAULT_INPUT_DIR,
        help="Directory containing toolchains_chunk_*.jsonl files (processed in order)",
    )
    mode.add_argument(
        "--input_path",
        default=None,
        help="Path to a single input JSONL file (one GeneratorInput per line)",
    )
    parser.add_argument(
        "--output-dir",
        default=_DEFAULT_OUTPUT_DIR,
        help="Output directory for per-chunk JSONL files (used with --input-dir)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output JSONL path for single-file mode (default: <output-dir>/training_data.jsonl)",
    )

    # ── Config ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        help="Path to pipeline config JSON file",
    )
    _DEFAULT_VLLM_URLS = ",".join(
        f"http://localhost:{port}/v1" for port in range(8080, 8087)
    )
    _DEFAULT_VLLM_MODEL = "Qwen/Qwen3.6-27B"

    parser.add_argument(
        "--generator-urls",
        default=_DEFAULT_VLLM_URLS,
        help=(
            "Generator vLLM server URL(s).  Pass a comma-separated list or "
            "repeat the flag to use multiple servers with round-robin dispatch.  "
            "Example: --generator-urls http://localhost:8084/v1,http://localhost:8085/v1"
        ),
    )
    parser.add_argument(
        "--generator-model",
        default=_DEFAULT_VLLM_MODEL,
        help="Generator model name",
    )
    parser.add_argument(
        "--augmentor-urls",
        default=_DEFAULT_VLLM_URLS,
        help="Augmentor vLLM server URL(s) — comma-separated for multiple servers.",
    )
    parser.add_argument(
        "--augmentor-model",
        default=_DEFAULT_VLLM_MODEL,
        help="Augmentor model name",
    )
    parser.add_argument(
        "--verifier-urls",
        default=_DEFAULT_VLLM_URLS,
        help="Verifier vLLM server URL(s) — comma-separated for multiple servers.",
    )
    parser.add_argument(
        "--verifier-model",
        default=_DEFAULT_VLLM_MODEL,
        help="Verifier model name",
    )

    # ── Feature flags ───────────────────────────────────────────────────────
    parser.add_argument(
        "--use-augmentor",
        action="store_true",
        default=False,
        help="Enable augmentor step (paraphrase tool names / prompts). Disabled by default.",
    )
    parser.add_argument(
        "--use-verifier",
        action="store_true",
        default=False,
        help="Enable verifier step (quality scoring + retry). Disabled by default.",
    )
    parser.add_argument(
        "--no-segment",
        action="store_true",
        default=False,
        help="Disable edit-boundary segmentation; emit one example per input.",
    )
    parser.add_argument(
        "--include-rejected",
        action="store_true",
        default=False,
        help="Include rejected / failed edit attempts in the trajectory (default: off).",
    )
    parser.add_argument(
        "--keep-unsatisfied",
        action="store_true",
        default=False,
        help=(
            "Keep tool chains whose final molecule does not satisfy every "
            "constraint. By default such chains are dropped (they never reach an "
            "<ANSWER> and only yield truncated intermediate segments)."
        ),
    )
    parser.add_argument(
        "--reflection-prob",
        type=float,
        default=0.3,
        help="Probability of adding a reflection after each tool step (default: 0.3)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=7.0,
        help="Minimum verifier overall score to accept an example (default: 7.0)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max generation retries per example when verifier is enabled (default: 3)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=320,
        help="Max concurrent examples to process (default: 320)",
    )
    parser.add_argument(
        "-n",
        "--num-generations",
        type=int,
        default=1,
        help=(
            "Number of independent generation passes per tool-chain input. "
            "Each pass produces different LLM reasoning while reusing the "
            "same ground-truth tool calls. (default: 1)"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help=(
            "Resume a previously interrupted run.  Already-processed inputs "
            "(matched by user_prompt) are skipped and new results are appended "
            "to existing output files."
        ),
    )
    parser.add_argument(
        "--shard",
        default="",
        metavar="I/N",
        help=(
            "Process only chunk indices where index %% N == I (0-based I). Lets "
            "several processes share one --input-dir/--output-dir without "
            "colliding: chunk files are disjoint, so each worker writes its own "
            "outputs. One process cannot keep the fleet busy — its single event "
            "loop needs ~45 ms of Python/RDKit per chain (~15 min of one core per "
            "20k-chain chunk), and while it runs that, no request is dispatched "
            "and the servers drain to idle. Use with --resume."
        ),
    )
    parser.add_argument(
        "--naive-reasoning",
        action="store_true",
        help=(
            "ABLATION: write the EDIT-ROUND reasoning (which suggest_edits rule to "
            "commit) from the prompt ALONE — the user query, the molecule, the raw "
            "analyze_properties output and the raw suggest_edits JSON, and write the "
            "turn that leads to the call. No Landing Safety table, no round context, "
            "no spread verdict, no rendered candidate comparison, no verified "
            "chemistry names, no decision order, no candidate index, and its guards "
            "and retries off. The SEED span (SMARTS + scaffold SMILES) keeps the "
            "normal prompt. Forces the four-call path. Prices the scaffolding; not "
            "for building a training corpus."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> PipelineConfig:
    """Build pipeline config from CLI args or a JSON config file."""
    if args.config:
        return PipelineConfig.from_json(args.config)

    return PipelineConfig(
        generator=VLLMConfig(
            base_urls=args.generator_urls,
            model=args.generator_model,
            max_tokens=4096,
            temperature=0.7,
        ),
        augmentor=VLLMConfig(
            base_urls=args.augmentor_urls,
            model=args.augmentor_model,
            max_tokens=4096,
            temperature=0.8,
        ),
        verifier=VLLMConfig(
            base_urls=args.verifier_urls,
            model=args.verifier_model,
            max_tokens=1024,
            temperature=0.1,
        ),
        reflection_probability=args.reflection_prob,
        max_retries=args.max_retries,
        verifier_threshold=args.threshold,
        batch_size=args.batch_size,
        seed=args.seed,
        use_augmentor=args.use_augmentor,
        use_verifier=args.use_verifier,
        segment_edit_boundaries=not args.no_segment,
        include_rejected_rounds=args.include_rejected,
        num_generations=args.num_generations,
        drop_unsatisfied_chains=not args.keep_unsatisfied,
        naive_reasoning=args.naive_reasoning,
    )


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    if not args.verbose:
        # Keep only the tqdm progress bar visible. Silence chatty third-party
        # libraries (httpx request logs, openai, urllib3) and the pipeline's
        # own info-level chatter.
        for name in ("httpx", "httpcore", "openai", "urllib3", "asyncio"):
            logging.getLogger(name).setLevel(logging.WARNING)
        for name in (
            "4_sftdata_gen.pipeline",
            "4_sftdata_gen.generator",
            "4_sftdata_gen.llm_client",
            "4_sftdata_gen.verifier",
            "4_sftdata_gen.augmentor",
        ):
            logging.getLogger(name).setLevel(logging.WARNING)

    config = build_config(args)

    if config.seed is not None:
        random.seed(config.seed)

    pipeline = TrainingDataPipeline(config)

    if args.input_path:
        # Single-file mode
        out = args.output or os.path.join(args.output_dir, "training_data.jsonl")
        examples = asyncio.run(pipeline.run(args.input_path, out, resume=args.resume))
        print(f"\nDone. Generated {len(examples)} training examples → {out}")
    else:
        # Directory mode: process all chunks in order
        shard = (0, 1)
        if args.shard:
            i, n = args.shard.split("/")
            shard = (int(i), int(n))
            if not 0 <= shard[0] < shard[1]:
                raise SystemExit(f"--shard {args.shard}: need 0 <= I < N")
        total = asyncio.run(pipeline.run_dir(
            args.input_dir, args.output_dir, resume=args.resume, shard=shard))
        print(f"\nDone. Generated {total} training examples total → {args.output_dir}")


if __name__ == "__main__":
    main()
