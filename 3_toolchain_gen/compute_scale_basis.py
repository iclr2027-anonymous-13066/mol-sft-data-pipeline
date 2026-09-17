#!/usr/bin/env python
"""Derive the per-property normalisation scale ``_SCALE`` from data.

``search_plan._SCALE`` (and its mirror ``agentic_eval._PROP_SCALE``) normalises the
multi-objective property gap so no single property dominates move selection. Rather
than hand-picking the values, we set each scale to the POPULATION STANDARD DEVIATION
of that property over the dataset's molecules ("1 gap unit = 1 std of natural
variation" — z-score standardisation).

This script recomputes that basis by reading the measured properties already stored
in the toolchain records' ``analyze_properties`` responses (no tool server
needed), and prints the std / MAD / IQR per property next to the current scale so the
derivation is fully reproducible.

Usage:
    python 3_toolchain_gen/compute_scale_basis.py \
        --dir data/training_data/toolchain/generation_2m_scaffold \
        --n 40000
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics

# The current committed scale (population std over ~40k molecules; see SCALE_BASIS.md).
CURRENT_SCALE = {
    "MW": 86.0, "logP": 1.26, "HBD": 0.85, "HBA": 1.62, "TPSA": 26.0, "rotB": 1.93,
    "rings_total": 1.08, "QED": 0.15, "MR": 24.0, "heavy_atoms": 6.24,
    "formal_charge": 1.0, "logD": 1.36, "logS": 1.56, "BBBP": 0.125, "HIA": 0.1,
    "Mutag": 0.19,
}
PROP_ORDER = list(CURRENT_SCALE)


def _iter_measurements(path: str):
    """Yield one measured-property dict per toolchain record (the seed measurement)."""
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            for step in rec.get("tool_chain", []):
                calls = [step.get("tool_call")] + (step.get("parallel_tool_calls") or [])
                resps = [step.get("expected_response")] + (step.get("parallel_expected_responses") or [])
                for call, resp in zip(calls, resps):
                    if call and call.get("name") == "analyze_properties":
                        try:
                            props = json.loads(resp) if isinstance(resp, str) else resp
                        except Exception:
                            props = None
                        if isinstance(props, dict):
                            yield props
                        break
                else:
                    continue
                break


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/training_data/toolchain/generation_2m_scaffold")
    ap.add_argument("--n", type=int, default=40000, help="molecules to sample")
    args = ap.parse_args()

    vals = {p: [] for p in PROP_ORDER}
    seen = 0
    for fp in sorted(glob.glob(os.path.join(args.dir, "toolchains_generation_chunk_*.jsonl"))):
        if seen >= args.n:
            break
        for props in _iter_measurements(fp):
            for p in PROP_ORDER:
                v = props.get(p)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    vals[p].append(float(v))
            seen += 1
            if seen >= args.n:
                break

    print(f"population molecules sampled: {seen}\n")
    print(f"{'prop':<13}{'n':>8}{'std':>10}{'MAD':>10}{'IQR':>10}{'current':>10}")
    print("-" * 61)
    suggested = {}
    for p in PROP_ORDER:
        xs = vals[p]
        if len(xs) < 10:
            print(f"{p:<13}{len(xs):>8}   (no data → keep current {CURRENT_SCALE[p]})")
            suggested[p] = CURRENT_SCALE[p]
            continue
        sd = statistics.pstdev(xs)
        med = statistics.median(xs)
        mad = statistics.median([abs(x - med) for x in xs])
        xs_s = sorted(xs)
        iqr = xs_s[3 * len(xs_s) // 4] - xs_s[len(xs_s) // 4]
        suggested[p] = round(sd, 3)
        print(f"{p:<13}{len(xs):>8}{sd:>10.3f}{mad:>10.3f}{iqr:>10.3f}{CURRENT_SCALE[p]:>10.3f}")

    print("\nsuggested _SCALE (population std; no-data props kept at current):")
    print(json.dumps(suggested, indent=2))


if __name__ == "__main__":
    main()
