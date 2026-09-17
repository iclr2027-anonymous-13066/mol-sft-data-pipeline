#!/usr/bin/env python3
"""Detect and remove training samples where the Verification block disagrees
with the ``<ANSWER>`` tag.

Rule
----
* If the **last** ``X/Y satisfied.`` in the last assistant message has X < Y
  (not all satisfied), the message must not carry an ``<ANSWER>``: the chain
  did not finish inside the property box.

Usage
-----
    # Dry-run (report only, no files modified):
    python clean_verification_mismatch.py /path/to/frag_all

    # Delete mismatched lines and write cleaned files in-place:
    python clean_verification_mismatch.py /path/to/frag_all --apply

    # Process multiple directories listed in a config file:
    python 4_sftdata_gen/scripts/clean_verification_mismatch.py --from-config path/to/train_config.yaml --apply

The script writes a JSON report to ``<output_dir>/mismatch_report.json``
regardless of ``--apply``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

SATISFIED_PATTERN = re.compile(r"(\d+)/(\d+)\s+satisfied\.")


def _last_assistant_content(messages: list[dict]) -> Optional[str]:
    last = None
    for msg in messages:
        if msg.get("role") == "assistant" and "content" in msg:
            last = msg
    return last["content"] if last else None


def is_mismatch(content: str) -> Optional[str]:
    """Return "A" for a mismatch, or None if consistent."""
    if not content:
        return None

    matches = list(SATISFIED_PATTERN.finditer(content))
    if not matches:
        return None

    last_match = matches[-1]
    x, y = int(last_match.group(1)), int(last_match.group(2))
    all_satisfied = x == y and y > 0

    after = content[last_match.end():]
    has_answer = "<ANSWER>" in after

    if not all_satisfied and has_answer:
        return "A"
    return None


def process_file(
    jsonl_path: Path,
    apply: bool = False,
) -> dict:
    """Scan a single JSONL file; optionally rewrite it without mismatches.

    Returns a per-file statistics dict.
    """
    stats = {
        "file": str(jsonl_path),
        "total": 0,
        "type_a": 0,
        "kept": 0,
        "removed_lines": [],
    }

    kept_lines: list[str] = []

    with open(jsonl_path, "r") as f:
        for line_num, raw_line in enumerate(f, 1):
            stripped = raw_line.strip()
            if not stripped:
                kept_lines.append(raw_line)
                continue

            stats["total"] += 1

            try:
                data = json.loads(stripped)
            except json.JSONDecodeError:
                kept_lines.append(raw_line)
                stats["kept"] += 1
                continue

            messages = data.get("messages", [])
            content = _last_assistant_content(messages)
            mtype = is_mismatch(content) if content else None

            if mtype == "A":
                stats["type_a"] += 1
                stats["removed_lines"].append(line_num)
                stats["removed_lines"].append(line_num)
            else:
                kept_lines.append(raw_line)
                stats["kept"] += 1

    if apply and stats["removed_lines"]:
        # Atomic rewrite: write to temp file, then rename.
        fd, tmp_path = tempfile.mkstemp(
            dir=jsonl_path.parent, suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w") as tmp_f:
                tmp_f.writelines(kept_lines)
            shutil.move(tmp_path, jsonl_path)
        except Exception:
            os.unlink(tmp_path)
            raise

    return stats


def collect_jsonl_files(directory: str) -> list[Path]:
    return sorted(Path(directory).glob("*.jsonl"))


def load_dirs_from_yaml(yaml_path: str) -> list[str]:
    """Extract train_path list from a YAML config file."""
    import yaml  # optional dep

    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    paths = cfg.get("data", {}).get("train_path", [])
    if isinstance(paths, str):
        paths = [paths]
    return [p for p in paths if p]


def main():
    parser = argparse.ArgumentParser(
        description="Detect and remove Verification ↔ ANSWER mismatches."
    )
    parser.add_argument(
        "directories",
        nargs="*",
        help="One or more frag_all directories to scan.",
    )
    parser.add_argument(
        "--from-config",
        type=str,
        default=None,
        help="Read train_path list from a YAML config instead of positional args.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually rewrite files (remove mismatches). Without this flag, dry-run only.",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Path to write the JSON report (default: ./mismatch_report.json).",
    )
    args = parser.parse_args()

    directories: list[str] = list(args.directories or [])
    if args.from_config:
        directories.extend(load_dirs_from_yaml(args.from_config))
    if not directories:
        parser.error("Provide at least one directory or --from-config.")

    all_stats: list[dict] = []
    total_scanned = 0
    total_removed = 0
    per_dir_summary: dict[str, dict] = {}

    for d in directories:
        jsonl_files = collect_jsonl_files(d)
        if not jsonl_files:
            print(f"  [WARN] No .jsonl files in {d}", file=sys.stderr)
            continue

        dir_a = dir_total = dir_kept = 0
        for jf in jsonl_files:
            stats = process_file(jf, apply=args.apply)
            all_stats.append(stats)
            dir_a += stats["type_a"]
            dir_total += stats["total"]
            dir_kept += stats["kept"]

        removed = dir_a
        total_scanned += dir_total
        total_removed += removed
        per_dir_summary[d] = {
            "total": dir_total,
            "type_a": dir_a,
            "removed": removed,
            "kept": dir_kept,
        }

        tag = "APPLIED" if args.apply else "DRY-RUN"
        print(
            f"  [{tag}] {d}: "
            f"{dir_total} samples, "
            f"{removed} mismatches removed "
            f"(A={dir_a}), "
            f"{dir_kept} kept",
            file=sys.stderr,
        )

    # ── Summary ───────────────────────────────────────────────────────
    print(file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    if args.apply:
        print("CLEANUP COMPLETE", file=sys.stderr)
    else:
        print("DRY-RUN COMPLETE (no files modified, re-run with --apply)", file=sys.stderr)
    print(f"  Total scanned:  {total_scanned}", file=sys.stderr)
    print(f"  Total removed:  {total_removed}", file=sys.stderr)
    print(f"  Total kept:     {total_scanned - total_removed}", file=sys.stderr)
    print("=" * 70, file=sys.stderr)

    # ── JSON report ───────────────────────────────────────────────────
    report = {
        "mode": "apply" if args.apply else "dry-run",
        "summary": {
            "total_scanned": total_scanned,
            "total_removed": total_removed,
            "total_kept": total_scanned - total_removed,
        },
        "per_directory": per_dir_summary,
        "per_file": [
            {
                "file": s["file"],
                "total": s["total"],
                "type_a": s["type_a"],
                "kept": s["kept"],
                "removed_lines": s["removed_lines"],
            }
            for s in all_stats
            if s["type_a"] > 0
        ],
    }
    report_path = args.report or "mismatch_report.json"
    with open(report_path, "w") as rf:
        json.dump(report, rf, indent=2, ensure_ascii=False)
    print(f"\nReport saved to {report_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
