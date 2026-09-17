"""Recompute the ``suggest_edits`` responses stored in finished tool-chain files.

Why
---
Stage 3 records each ``suggest_edits`` step's ``expected_response`` by calling the
live tool again, *without* the properties the planner had already measured, so the
tool re-measured the molecule itself. Its ``_measure`` reported ``MW`` as the
AVERAGE molecular weight while everything else in the pipeline
(``analyze_properties``, the mmpdb Δ table, the constraint boxes) uses the
monoisotopic mass — a systematic +0.2…0.5 Da offset. Usually harmless, but near a
box boundary it re-ranks the candidates, and since only ``top_k`` survive, the
committed edit could drop out of the recorded list (~0.3% of edit rounds) or the
list could come back empty (~0.2% of chains).

``suggest_edits._measure`` is fixed now, so a fresh call agrees with the planner.
This script rewrites the stored responses of ALREADY-GENERATED chains with that
fixed ranking, so a dataset built partly before and partly after the fix is
uniform — no stage-3 re-run needed, because the chains themselves (molecules,
edits, checkpoints, constraint verdicts) do not change: the planner passed its own
props and never consulted the buggy path.

What it does, per record
------------------------
For every ``suggest_edits`` step: recompute with the property values measured by
the *preceding* checkpoint (``analyze_properties``) — which, post-fix, is what the
tool computes internally, so no ADMET backend is needed — and overwrite
``expected_response``. Steps are left untouched when the recomputation fails or
still omits the edit the following ``edit_fragment`` commits (counted and
reported, so the miss is never silent).

Usage
-----
    python 3_toolchain_gen/repair_suggest_responses.py \
        --dir data/training_data/toolchain/generation_2m_scaffold \
        --num-proc 64                     # report only (default)

    ... --apply                           # rewrite files in place (atomic)
    ... --limit 5                         # first N files only (smoke test)
    ... --out-dir /some/copy              # write elsewhere instead of in place

Cost / sizing (measured)
------------------------
~0.32 s per suggest step on one core, and work is split per FILE, so a 20k-record
chunk is ~2 h of single-core time: 100 chunks over 64 procs ≈ 4 h wall clock (use
``--num-proc 100`` for one wave, ≈2 h). Each worker holds its own copy of the
mmpdb move index — **~1.1 GB RSS per process** (plus a 2.6 s load from the
``.suggest_cache_v2_c*.pkl`` disk cache) — so keep ``num-proc`` × 1.1 GB inside
available RAM. Run it AFTER stage 3 finishes: it rewrites the same files.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CHUNK_GLOB = "toolchains_*chunk_*.jsonl"


def _edit_key(args: dict) -> tuple:
    return (args.get("from_smiles"), args.get("to_smiles"))


def _lists(response, key: tuple) -> bool:
    if isinstance(response, str):
        try:
            response = json.loads(response)
        except (ValueError, TypeError):
            return False
    if not isinstance(response, list):
        return False
    return any(isinstance(c, dict)
               and (c.get("from_smiles"), c.get("to_smiles")) == key
               for c in response)


def _props_from_step(step: dict) -> dict | None:
    """The ``analyze_properties`` payload of a checkpoint step, if it has one."""
    calls = [step.get("tool_call") or {}] + list(step.get("parallel_tool_calls") or [])
    resps = [step.get("expected_response")] + list(
        step.get("parallel_expected_responses") or [])
    for call, resp in zip(calls, resps):
        if (call or {}).get("name") != "analyze_properties":
            continue
        try:
            parsed = json.loads(resp)
        except (ValueError, TypeError):
            return None
        if isinstance(parsed, dict):
            return {k: v for k, v in parsed.items() if v is not None}
    return None


def _repair_record(rec: dict, stats: dict) -> bool:
    """Rewrite the record's suggest responses in place. True if anything changed."""
    from molkit.utils.suggest_edits import suggest_edits

    steps = rec.get("tool_chain") or []
    props: dict | None = None
    changed = False
    for i, step in enumerate(steps):
        call = step.get("tool_call") or {}
        name = call.get("name")
        if name in ("match_substructure", "analyze_properties"):
            found = _props_from_step(step)
            if found:
                props = found
            continue
        if name != "suggest_edits":
            continue

        stats["suggest_steps"] += 1
        args = call.get("arguments") or {}
        constraints = args.get("constraints") or {}
        # The edit this step justifies: the next edit_fragment in the chain.
        key = None
        for later in steps[i + 1:]:
            lname = ((later.get("tool_call") or {}).get("name"))
            if lname == "edit_fragment":
                key = _edit_key((later.get("tool_call") or {}).get("arguments") or {})
                break
            if lname == "suggest_edits":
                break
        old = step.get("expected_response")
        if key is not None and _lists(old, key):
            stats["already_ok"] += 1
        try:
            need = [k for k in constraints if props is None or props.get(k) is None]
            fresh = suggest_edits(
                args.get("mol_smiles"), constraints, int(args.get("top_k", 10)),
                props=(None if need else {k: props[k] for k in constraints}),
                scaffold_smarts=args.get("scaffold_smarts") or None,
            )
        except Exception as exc:  # noqa: BLE001
            stats["recompute_failed"] += 1
            stats.setdefault("errors", []).append(str(exc)[:120])
            continue

        if key is not None and not _lists(fresh, key):
            # Still missing: keep whatever was there and report it. Stage 4 renders
            # these rounds honestly ("not among the ranked suggestions").
            stats["still_missing"] += 1
            continue
        if not fresh:
            stats["still_empty"] += 1
            continue
        new_text = json.dumps(fresh, ensure_ascii=False)
        if new_text != (old if isinstance(old, str) else json.dumps(old)):
            step["expected_response"] = new_text
            stats["rewritten"] += 1
            changed = True
        else:
            stats["identical"] += 1
    return changed


def _worker(job: tuple) -> dict:
    path_str, out_str, apply_changes = job
    path, out = Path(path_str), Path(out_str)
    stats = {"file": path.name, "records": 0, "suggest_steps": 0, "already_ok": 0,
             "rewritten": 0, "identical": 0, "still_missing": 0, "still_empty": 0,
             "recompute_failed": 0, "changed_records": 0}
    lines_out: list[str] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            stats["records"] += 1
            if _repair_record(rec, stats):
                stats["changed_records"] += 1
            if apply_changes:
                lines_out.append(json.dumps(rec, ensure_ascii=False))
    if apply_changes:
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(out.suffix + ".tmp")
        with open(tmp, "w") as f:
            f.write("\n".join(lines_out) + "\n")
        os.replace(tmp, out)          # atomic: a crash never leaves a partial file
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="tool-chain directory (chunk files)")
    ap.add_argument("--out-dir", default=None,
                    help="write repaired files here instead of in place")
    ap.add_argument("--apply", action="store_true",
                    help="actually write files (default: report only)")
    ap.add_argument("--num-proc", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None, help="first N files only")
    args = ap.parse_args()

    src = Path(args.dir)
    files = sorted(src.glob(CHUNK_GLOB))
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise SystemExit(f"no {CHUNK_GLOB} files in {src}")
    out_dir = Path(args.out_dir) if args.out_dir else src
    jobs = [(str(p), str(out_dir / p.name), args.apply) for p in files]

    print(f"{len(files)} file(s) | {'APPLY (rewriting)' if args.apply else 'report only'}"
          f" | out: {out_dir} | {args.num_proc} procs", flush=True)

    total = {k: 0 for k in ("records", "suggest_steps", "already_ok", "rewritten",
                            "identical", "still_missing", "still_empty",
                            "recompute_failed", "changed_records")}
    errors: list[str] = []
    with mp.Pool(processes=max(1, args.num_proc)) as pool:
        for done, st in enumerate(pool.imap_unordered(_worker, jobs), 1):
            for k in total:
                total[k] += st.get(k, 0)
            errors.extend(st.get("errors", [])[:2])
            if done % 20 == 0 or done == len(files):
                print(f"  [{done}/{len(files)}] records={total['records']} "
                      f"suggest={total['suggest_steps']} rewritten={total['rewritten']} "
                      f"still_missing={total['still_missing']} "
                      f"still_empty={total['still_empty']}", flush=True)

    print("\n=== summary")
    for k in ("records", "suggest_steps", "already_ok", "rewritten", "identical",
              "still_missing", "still_empty", "recompute_failed", "changed_records"):
        print(f"  {k:18s} {total[k]}")
    if total["suggest_steps"]:
        ok = total["suggest_steps"] - total["still_missing"] - total["still_empty"]
        print(f"  committed edit present in the response: "
              f"{ok}/{total['suggest_steps']} ({100 * ok / total['suggest_steps']:.2f}%)")
    if errors:
        print("  sample recompute errors:", errors[:3])
    if not args.apply:
        print("\nreport only — re-run with --apply to rewrite the files.")


if __name__ == "__main__":
    main()
