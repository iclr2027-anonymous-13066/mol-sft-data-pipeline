"""Command-line interface for the substructure-generation tool-chain builder."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from .builder import ToolChainBuilder, dataset_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build ground-truth tool chains for the scaffold-constrained "
            "generation task. The chain seeds on the described scaffold "
            "(verified against scaffold_smarts), then adds property-tuning side "
            "chains (decorate phase, with analyze_properties read-backs), and "
            "finally verifies the molecule against the substructure and target "
            "properties. The trajectory is planned from the description + "
            "properties alone via the Δ-guided forward search."
        ),
    )
    parser.add_argument(
        "--input",
        default="data/training_data/instances/generation_200k_scaffold",
        help=(
            "A JSONL benchmark file OR a directory of JSONL files (all '*.jsonl' "
            "inside are loaded, sorted by name, and processed together)."
        ),
    )
    parser.add_argument(
        "--output",
        default="data/training_data/toolchain",
        help=(
            "Base output directory. A subfolder named after the input (the "
            "directory name, or the file stem) is created inside it and chunk "
            "files + summary.json are written there — e.g. "
            "<output>/generation_200k_scaffold/."
        ),
    )
    parser.add_argument(
        "--output-name",
        dest="output_name",
        default=None,
        help=(
            "Override the output subfolder name (default: derived from --input). "
            "Use to keep runs of the SAME input separate, e.g. one run per "
            "--search-mode: '<stem>_greedy' vs '<stem>_beam'."
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Number of records per output JSONL chunk (default: 1000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--search-max-steps",
        dest="search_max_steps",
        type=int,
        default=12,
        help=(
            "Decoration budget — max greedy edits, or max beam "
            "rounds, before returning the best molecule (default: 12)."
        ),
    )
    parser.add_argument(
        "--search-mode",
        dest="search_mode",
        choices=["greedy", "beam", "hybrid"],
        default="greedy",
        help=(
            "Decorate strategy: 'greedy' (single path, ~1 measurement "
            "per step), 'beam' (explore beam_width paths, select by real measurement; "
            "higher hit rate, ~beam_width×beam_expand× the measurements), or 'hybrid' "
            "(greedy first, beam-rescue only the instances greedy fails — greedy's "
            "hit rate on the easy ones at greedy's cost, beam's on the hard ones)."
        ),
    )
    parser.add_argument(
        "--beam-width", dest="beam_width", type=int, default=4,
        help="--search-mode beam: number of partial molecules kept per round (default 4).",
    )
    parser.add_argument(
        "--beam-expand", dest="beam_expand", type=int, default=4,
        help="--search-mode beam: candidates expanded per beam node per round (default 4). "
             "In beam mode this IS the top_k suggest_edits is called with.",
    )
    parser.add_argument(
        "--suggest-top-k", dest="suggest_top_k", type=int, default=4,
        help=(
            "Number of candidate edits suggest_edits returns "
            "per step (the top_k it is called with). Greedy commits the top-ranked "
            "guard-passing one; beam explores up to --beam-expand of them (default 4)."
        ),
    )
    parser.add_argument(
        "--analyze-each-decorate",
        dest="analyze_each_decorate",
        action="store_true",
        default=True,
        help=(
            "Emit an analyze_properties read-back after each intermediate "
            "decoration so the chain demonstrates a measure→edit→re-measure loop "
            "(default on)."
        ),
    )
    parser.add_argument(
        "--no-analyze-each-decorate",
        dest="analyze_each_decorate",
        action="store_false",
        help="Only analyze at the scaffold checkpoint and final verification "
             "(legacy assembly only; ignored with --direct-scaffold-seed).",
    )
    ds_group = parser.add_mutually_exclusive_group()
    ds_group.add_argument(
        "--direct-scaffold-seed",
        dest="direct_scaffold_seed",
        action="store_true",
        default=True,
        help=(
            "Default. Start the chain from the COMPLETED scaffold (drop the "
            "scaffold-building edits) and keep only the property-tuning decorate "
            "edits. Every checkpoint is the 3-way match_substructure ∥ "
            "analyze_properties ∥ label_atom_indices (the label grounds "
            "the next edit's atom_index)."
        ),
    )
    ds_group.add_argument(
        "--no-direct-scaffold-seed",
        dest="direct_scaffold_seed",
        action="store_false",
        help=(
            "Legacy incremental assembly: hub-core seed, an explicit scaffold "
            "phase that builds the framework, 2-way checkpoints, and a separate "
            "pre-edit label_atom_indices before every edit."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N instances.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=20,
        help="Number of parallel async coroutines (default: 20).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume an interrupted run: scan existing chunk files for "
            "already-built instance_index values and skip them."
        ),
    )
    parser.add_argument(
        "--local-tools",
        dest="local_tools",
        action="store_true",
        default=(os.environ.get("LOCAL_TOOLS", "0")
                 not in ("0", "", "false", "False")),
        help=(
            "Execute tools in-process instead of POSTing to the tool servers "
            "(no servers need to be running). The stored results are identical; "
            "this removes the HTTP round-trip and the ADMET retry-under-contention "
            "storm. The admet_ai model is loaded once in this process. "
            "Also settable via LOCAL_TOOLS=1."
        ),
    )
    parser.add_argument(
        "--num-procs",
        dest="num_procs",
        type=int,
        default=int(os.environ.get("NUM_PROCS", "1") or 1),
        help=(
            "Local-tools only: number of worker PROCESSES to build across, each "
            "with its own in-process ADMET model (>1 implies --local-tools). The "
            "workload is CPU-bound, so peak throughput is around cores/6 processes "
            "(e.g. ~40 on ~250 cores); more oversubscribes and slows down. "
            "1 (default) = single process. Also settable via NUM_PROCS."
        ),
    )
    parser.add_argument(
        "--per-proc",
        dest="per_proc",
        type=int,
        default=int(os.environ.get("PER_PROC", "4") or 4),
        help=(
            "Local-tools multiprocess only: async concurrency WITHIN each worker "
            "(default 4). Keep low (2-4) — the model is serial per process, so "
            "processes, not intra-process concurrency, provide the parallelism."
        ),
    )
    parser.add_argument(
        "--part-size",
        dest="part_size",
        type=int,
        default=int(os.environ.get("PART_SIZE", "1000") or 1000),
        help=(
            "Local-tools multiprocess only: the parallel WORK unit (instances per "
            "part) and the granularity of incremental crash-safe writes — decoupled "
            "from --chunk-size (the final file size). Keep it well below "
            "N/NUM_PROCS so all workers stay busy and results stream to disk; parts "
            "are merged into --chunk-size files (in order) at the end. Default 1000. "
            "--chunk-size is rounded down to a multiple of this."
        ),
    )
    parser.add_argument(
        "--gpus",
        dest="gpus",
        default=os.environ.get("GPUS", "0"),
        help=(
            "Local-tools multiprocess only: comma-separated GPU ids to round-robin "
            "workers over (default '0'). One GPU handles a whole node's ADMET "
            "bursts; more GPUs give memory/scheduling headroom, not throughput "
            "(the bottleneck is CPU cores). E.g. '0,1' or '0,1,2,3,4,5,6,7'."
        ),
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return parser.parse_args()


def _write_summary(output_dir: str) -> dict:
    """Scan output chunk files, print and persist aggregate stats."""
    sub = Path(output_dir)
    total = success = constructive = 0
    edit_steps: list[int] = []
    tool_calls: list[int] = []
    with_errors = 0

    for chunk_file in sorted(sub.glob("toolchains_generation_chunk_*.jsonl")):
        with open(chunk_file) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    meta = json.loads(line).get("metadata", {})
                except json.JSONDecodeError:
                    continue
                total += 1
                success += bool(meta.get("all_constraints_satisfied"))
                constructive += meta.get("chain_strategy") == "constructive"
                edit_steps.append(int(meta.get("num_edit_steps", 0)))
                tool_calls.append(int(meta.get("num_tool_calls", 0)))
                with_errors += bool(meta.get("tool_errors"))

    stats = {
        "task_type": "generation",
        "total_instances": total,
        "success_count": success,
        "constructive_count": constructive,
        "anchor_count": total - constructive,
        "success_rate": round(success / total, 4) if total else 0.0,
        "avg_tool_calls": round(sum(tool_calls) / total, 2) if total else 0.0,
        "avg_edit_steps": round(sum(edit_steps) / total, 2) if total else 0.0,
        "instances_with_tool_errors": with_errors,
    }

    summary_path = Path(output_dir) / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as fh:
        json.dump(stats, fh, indent=2, ensure_ascii=False)

    sep = "=" * 60
    print(f"\n{sep}\n  generation  (saved: {summary_path})\n{sep}")
    print(f"  Total instances:        {stats['total_instances']}")
    print(f"  Success (strict):       {stats['success_count']} "
          f"({stats['success_rate']:.1%})")
    print(f"  Constructive / anchor:  {stats['constructive_count']} / "
          f"{stats['anchor_count']}")
    print(f"  Avg tool calls:         {stats['avg_tool_calls']}")
    print(f"  Avg edit steps:         {stats['avg_edit_steps']}")
    print(f"  Instances w/ errors:    {stats['instances_with_tool_errors']}")
    print(sep)
    return stats


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Resolve the destination: a folder named after the input (file stem or
    # directory name) under the output base, e.g.
    #   --input .../instances/generation_200k_scaffold
    #   --output .../toolchain
    #   -> .../toolchain/generation_200k_scaffold/
    out_dir = str(Path(args.output) / (args.output_name or dataset_name(args.input)))
    log = logging.getLogger(__name__)
    log.info("writing tool chains to %s", out_dir)

    # Parse --gpus "0,1,.." into a list of ints for the multiprocess runner.
    args.gpus = [int(x) for x in str(getattr(args, "gpus", "0")).split(",") if x != ""]

    # --num-procs > 1 is the multiprocess local path: each worker sets local mode
    # and loads its OWN ADMET model, so the parent must NOT preload (it would grab
    # a GPU it never uses). Single-process --local-tools preloads here up front.
    if int(getattr(args, "num_procs", 1) or 1) > 1:
        args.local_tools = True
        log.info("local-tools multiprocess: %d workers, per_proc=%d, gpus=%s",
                 args.num_procs, getattr(args, "per_proc", 4), args.gpus)
    elif getattr(args, "local_tools", False):
        from . import http_client, local_tools
        http_client.set_local_mode(True)
        log.info("local-tools mode: executing tools in-process (no tool servers)")
        local_tools.preload(with_admet=True)

    builder = ToolChainBuilder(args=args)
    success, total = builder.run(args.input, out_dir, args.limit, args.chunk_size)
    print(f"\nDone. {success} / {total} instances satisfied all constraints.")

    _write_summary(out_dir)
