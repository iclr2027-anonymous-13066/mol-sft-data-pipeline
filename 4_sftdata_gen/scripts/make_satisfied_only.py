#!/usr/bin/env python3
"""Write a SATISFIED-ONLY copy of a stage-4 SFT data dir (ablation arm).

Stage 4 runs with ``--keep-unsatisfied`` by default, so the generated data mixes
two kinds of trajectory:

  * **satisfied** — the chain reaches a molecule that meets every constraint, so
    its conversation closes on ``<ANSWER>``
    (``metadata.ends_with_answer == True``);
  * **unsatisfied** — the chain never gets there and there is no ``<ANSWER>``.

This makes the ablation copy. Per record (= one chain):

  * chain reached an ``<ANSWER>`` → the record is kept whole;
  * chain did not → the record is truncated to its SEED round
    (``messages[:metadata.seed_messages]``).

The seed round is kept because it is right regardless of how the chain ended: it
derives the scaffold SMARTS piece by piece, writes the scaffold SMILES and runs the
3-way match ‖ analyze ‖ label checkpoint against real tool output — none of which
depends on later edit rounds landing in range. Only the edit rounds that chased the
targets and missed are dropped. Pass ``--no-unsatisfied-seed`` to drop those chains
outright instead.

Train one model on the source dir (satisfied + unsatisfied) and one on this copy to
measure what the unsatisfied edit rounds buy. A seed-only record is recognisable
downstream without any extra flag: ``metadata.seed_only == True``.

Safety / restartability:

  * Only chunks stage 4 has FINISHED (a ``<chunk>.jsonl.done`` marker exists) are
    read, so this can run while stage 4 is still generating. ``--include-unfinished``
    reads the in-progress ones too (their trailing partial group is dropped anyway).
  * A destination file already newer than its source is skipped, so re-running
    after more chunks finish only converts the new ones (``--overwrite`` to redo).
  * The filter mode is recorded in ``<dst>/_filter_mode.json``. Changing it (e.g.
    turning ``--no-unsatisfied-seed`` on or off) makes every existing destination
    file stale, so the whole dir is reconverted instead of silently mixing two
    filter policies in one dataset.
  * ``_records.arrow`` (the memory-mapped raw sidecar the training loader prefers)
    is rebuilt at the end over the whole destination dir; ``--no-arrow`` opts out.

Usage::

    # report only — counts what would be kept, writes nothing
    python 4_sftdata_gen/scripts/make_satisfied_only.py \
        --src data/training_data/sftdata/generation_2m_scaffold --dry-run

    # write the copy (default dst = <src>_satonly)
    python 4_sftdata_gen/scripts/make_satisfied_only.py \
        --src data/training_data/sftdata/generation_2m_scaffold \
        --dst data/training_data/sftdata/generation_2m_scaffold_satonly \
        --num-proc 16
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

try:
    import orjson as _json

    def _loads(b):
        return _json.loads(b)
except ImportError:  # stdlib fallback (slower, same result)
    import json as _json

    def _loads(b):
        return _json.loads(b)


# A chain is SATISFIED when its conversation closes on <ANSWER>. schema.py sets
# metadata.ends_with_answer for exactly those, so the flag is the primary signal;
# the "<ANSWER>" text is checked alongside it and any disagreement is reported
# (it would mean the writer changed).
def _classify(rec: dict) -> tuple[bool, bool, int]:
    """Return (has_terminal_meta, has_answer_text, seed_messages)."""
    meta = rec.get("metadata") or {}
    has_meta = bool(meta.get("ends_with_answer"))
    msgs = rec.get("messages") or []
    has_text = bool(msgs) and "<ANSWER>" in (msgs[-1].get("content") or "")
    return has_meta, has_text, int(meta.get("seed_messages") or 0)


def convert_file(task):
    """Stream one source file into its filtered destination.

    Returns a stats dict. With ``dry_run`` nothing is written.
    """
    src, dst, dry_run, keep_seed = task
    st = {
        "file": os.path.basename(src),
        "records_in": 0, "records_out": 0,
        "groups": 0, "groups_kept": 0, "groups_seed_only": 0,
        "groups_dropped": 0,
        "no_seed": 0, "malformed": 0, "answer_mismatch": 0,
    }

    tmp = dst + ".tmp"
    out = None if dry_run else open(tmp, "wb", buffering=1 << 22)

    try:
        with open(src, "rb") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                st["records_in"] += 1
                try:
                    rec = _loads(line)
                except Exception:
                    st["malformed"] += 1
                    continue
                st["groups"] += 1
                has_meta, has_text, seed_n = _classify(rec)
                if has_meta != has_text:
                    st["answer_mismatch"] += 1

                if has_meta:
                    st["groups_kept"] += 1
                    st["records_out"] += 1
                    if out is not None:
                        out.write(line + b"\n")
                    continue

                # Unsatisfied chain: keep just the seed round, which is correct
                # however the chain ended — it derives the SMARTS, writes the
                # scaffold SMILES and runs the 3-way checkpoint on real tool
                # output.
                if not keep_seed:
                    st["groups_dropped"] += 1
                    continue
                msgs = rec.get("messages") or []
                if seed_n <= 0 or seed_n > len(msgs):
                    st["no_seed"] += 1
                    st["groups_dropped"] += 1
                    continue
                rec["messages"] = msgs[:seed_n]
                meta = rec.setdefault("metadata", {})
                meta["seed_only"] = True
                meta["ends_with_answer"] = False
                st["groups_seed_only"] += 1
                st["records_out"] += 1
                if out is not None:
                    out.write(json.dumps(rec, ensure_ascii=False).encode() + b"\n")
    finally:
        if out is not None:
            out.close()

    if not dry_run:
        os.replace(tmp, dst)
    return st


def main():
    ap = argparse.ArgumentParser(
        description="Write a satisfied-only copy of a stage-4 SFT data dir.")
    ap.add_argument("--src", required=True, help="source sftdata dir (mixed)")
    ap.add_argument("--dst", default=None,
                    help="destination dir (default: <src>_satonly)")
    ap.add_argument("--num-proc", type=int, default=min(16, os.cpu_count() or 4),
                    help="source files converted in parallel (each is I/O heavy)")
    ap.add_argument("--dry-run", action="store_true",
                    help="count only; write nothing")
    ap.add_argument("--overwrite", action="store_true",
                    help="reconvert files whose destination is already up to date")
    ap.add_argument("--include-unfinished", action="store_true",
                    help="also read chunks with no .done marker (stage 4 still writing them)")
    ap.add_argument("--no-unsatisfied-seed", dest="keep_seed", action="store_false",
                    help="drop unsatisfied chains entirely instead of keeping their "
                         "seed (SMARTS-derivation) segment")
    ap.add_argument("--no-arrow", action="store_true",
                    help="skip the _records.arrow rebuild at the end")
    ap.add_argument("--arrow-num-proc", type=int, default=48)
    args = ap.parse_args()

    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve() if args.dst else src.parent / f"{src.name}_satonly"
    if not src.is_dir():
        sys.exit(f"[error] --src is not a directory: {src}")
    if dst == src:
        sys.exit("[error] --dst must differ from --src (this never rewrites in place)")

    files = sorted(p for p in src.glob("*.jsonl") if not p.name.startswith("_"))
    if not files:
        sys.exit(f"[error] no *.jsonl in {src}")

    # Finished chunks only, unless asked otherwise: a chunk with no .done marker is
    # one stage 4 is still appending to.
    skipped_unfinished = []
    if not args.include_unfinished:
        keep = []
        for p in files:
            (keep if Path(str(p) + ".done").exists() else skipped_unfinished).append(p)
        files = keep

    mode = {"keep_unsatisfied_seed": bool(args.keep_seed)}
    print(f"src : {src}")
    print(f"dst : {dst}")
    print(f"mode: unsatisfied chains → "
          + ("keep their seed segment" if args.keep_seed else "dropped entirely"))
    print(f"files: {len(files)} finished chunk(s)"
          + (f", {len(skipped_unfinished)} unfinished (skipped)" if skipped_unfinished else ""))
    if not files:
        sys.exit("[error] no finished chunk to convert (--include-unfinished to force)")

    # A dir written under a different filter mode must be rebuilt as a whole —
    # otherwise files converted before and after the switch sit in one dataset
    # under two different policies.
    mode_path = dst / "_filter_mode.json"
    mode_changed = False
    if mode_path.exists():
        try:
            prev = json.loads(mode_path.read_text())
        except Exception:
            prev = None
        if prev != mode:
            mode_changed = True
            print(f"[mode] {mode_path.name} says {prev} — filter mode CHANGED, "
                  f"reconverting every file")
    elif any(dst.glob("*.jsonl")):
        # Written by a version that predates the marker: assume the old policy
        # (unsatisfied chains dropped) and rebuild if that is not what we want now.
        mode_changed = bool(args.keep_seed)
        print("[mode] no _filter_mode.json in an existing dst — assuming the old "
              "'drop unsatisfied' policy"
              + (", reconverting every file" if mode_changed else ""))

    if not args.dry_run:
        dst.mkdir(parents=True, exist_ok=True)

    # Resume: skip a destination that is already newer than its source.
    force = args.overwrite or mode_changed
    tasks, up_to_date = [], 0
    for p in files:
        out_p = dst / p.name
        if (not force and not args.dry_run and out_p.exists()
                and out_p.stat().st_mtime >= p.stat().st_mtime):
            up_to_date += 1
            continue
        tasks.append((str(p), str(out_p), args.dry_run, args.keep_seed))
    if up_to_date:
        print(f"resume: {up_to_date} destination file(s) already up to date (skipped)")
    if not tasks:
        print("nothing to convert.")
    else:
        print(f"converting {len(tasks)} file(s) with {args.num_proc} process(es)"
              + (" [DRY RUN]" if args.dry_run else "") + " …")

    totals = {k: 0 for k in ("records_in", "records_out", "groups", "groups_kept",
                             "groups_seed_only",
                             "groups_dropped", "no_seed",
                             "malformed", "answer_mismatch")}
    done = 0
    if tasks:
        with ProcessPoolExecutor(max_workers=max(1, args.num_proc)) as ex:
            for st in ex.map(convert_file, tasks):
                done += 1
                for k in totals:
                    totals[k] += st[k]
                print(f"  [{done}/{len(tasks)}] {st['file']}: "
                      f"groups {st['groups_kept']} full + {st['groups_seed_only']} "
                      f"seed-only / {st['groups']}, "
                      f"records {st['records_out']}/{st['records_in']}"
                      + (f", malformed {st['malformed']}" if st['malformed'] else ""),
                      flush=True)

    g, gk, gs = totals["groups"], totals["groups_kept"], totals["groups_seed_only"]
    r, rk = totals["records_in"], totals["records_out"]
    print("")
    print(f"groups : {gk}/{g} satisfied → full chain ({100 * gk / max(g, 1):.2f}%)")
    print(f"         {gs}/{g} unsatisfied → seed round only ({100 * gs / max(g, 1):.2f}%)")
    if totals["groups_dropped"]:
        print(f"         {totals['groups_dropped']}/{g} dropped entirely")
    print(f"records: {rk}/{r} kept      ({100 * rk / max(r, 1):.2f}%)")
    if totals["no_seed"]:
        print(f"[warn] {totals['no_seed']} record(s) carried no usable "
              f"seed_messages boundary and were dropped")
    if totals["malformed"]:
        print(f"malformed lines skipped: {totals['malformed']}")
    if totals["answer_mismatch"]:
        print(f"[warn] {totals['answer_mismatch']} record(s) where ends_with_answer "
              f"disagreed with the presence of <ANSWER> — check the writer format.")

    if args.dry_run:
        print("\nDry run — nothing written.")
        return

    # Record the policy this dir was written under (see the mode check above).
    mode_path.write_text(json.dumps(mode) + "\n")

    if args.no_arrow:
        print("\n--no-arrow — not building _records.arrow "
              "(training falls back to slower JSONL parsing).")
        return

    # The loader prefers <dst>/_records.arrow only while it is newer than every
    # .jsonl there, so rebuild it over the WHOLE dst dir after any conversion.
    # Nothing converted and the sidecar already newer than every shard → already
    # correct, and rebuilding it would re-read the whole (tens of GB) dir.
    arrow_path = dst / "_records.arrow"
    dst_jsonls = list(dst.glob("*.jsonl"))
    if not tasks and arrow_path.exists() and dst_jsonls and \
            arrow_path.stat().st_mtime >= max(p.stat().st_mtime for p in dst_jsonls):
        print("\n_records.arrow already up-to-date.")
        return

    arrow_script = Path(__file__).resolve().parent / "jsonl_to_arrow.py"
    print(f"\nBuilding {dst}/_records.arrow …", flush=True)
    rc = subprocess.call([sys.executable, str(arrow_script), str(dst),
                          "--num-proc", str(args.arrow_num_proc)])
    if rc != 0:
        print("[warn] arrow build failed; training will fall back to JSONL parsing.")


if __name__ == "__main__":
    main()
