#!/usr/bin/env python
"""Extract the AUTHORED-only segments of a stage-4 corpus as an additive delta.

Used to build the "+authored" arm of the authored-edit ablation:

    arm A   train_path: [<base>_satonly]
    arm B   train_path: [<base>_satonly, <base>_satonly_authoreddelta]

The authored corpus (generated with AUTHORED_EDIT_FRACTION > 0) is the SAME tool
chains as the base, with ~12% of decorate rounds rendered with an empty
``suggest_edits`` response and their edit reasoning re-derived without a candidate
list. Adding the whole thing would double the corpus with 88% near-duplicates —
a paraphrase-augmentation experiment, not an authored-branch one. This keeps only
the segments that actually differ, so arm B is arm A plus the new branch and
nothing else.

WHY THE GROUP-ID REMAP MATTERS
------------------------------
``group_id`` is ``f"{task_id}__{idx:07d}"`` where ``idx`` is the input's position
in the stage-4 processing order — and that order comes from an UNSEEDED
``random.shuffle`` (pipeline.py). Two stage-4 runs therefore give the same molecule
two different group_ids.

``task_id`` is stable across runs and is a 1:1 key for group_id within a corpus
(verified: 59,461 : 59,461, zero collisions on the 2M corpus), so every delta record
is re-keyed onto the base corpus's group_id and the two copies of a molecule carry
one key instead of two unrelated ones.

This does NOT make the delta's validation slice clean. ``dataset.py`` splits PER
DIRECTORY — each ``train_path`` entry becomes its own task with its own
``_split_indices`` over its own key list — so the base task and the delta task pick
unrelated val groups whatever the keys are called. Base val is ~30 of 59,461 groups,
so nearly every delta-val molecule is also in base-train. What stays clean is the
headline metric (``val/gen/overall_success``, scored on a separate held-out instance
file) and the base task's own val loss, which uses an identical split in both arms.
Read ``val/<delta dir>/loss`` as a train-fit curve, not as generalisation.

Records whose task_id is absent from the base are DROPPED — they cannot be merged
safely, and silently keeping them would reintroduce the leak.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter


def _authored_record(rec: dict) -> bool:
    """True when this segment renders a suggest_edits response as an empty list."""
    ms = rec.get("messages") or []
    for i, t in enumerate(ms):
        if t.get("role") != "tool" or i == 0:
            continue
        if (t.get("content") or "").strip() not in ("[]", "[ ]"):
            continue
        calls = (ms[i - 1].get("tool_calls") or [])
        if any(c.get("function", {}).get("name") == "suggest_edits" for c in calls):
            return True
    return False


def _task_id(rec: dict):
    gid = (rec.get("metadata") or {}).get("group_id")
    return gid.rsplit("__", 1)[0] if isinstance(gid, str) and "__" in gid else None


def build_base_index(base_dir: str) -> dict:
    """task_id -> group_id, over every shard of the base corpus."""
    index: dict = {}
    dupes = 0
    for path in sorted(glob.glob(os.path.join(base_dir, "*.jsonl"))):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                gid = (rec.get("metadata") or {}).get("group_id")
                tid = _task_id(rec)
                if not tid or not gid:
                    continue
                prev = index.setdefault(tid, gid)
                if prev != gid:
                    dupes += 1
    if dupes:
        print(f"[warn] {dupes} task_id(s) map to more than one group_id in the base; "
              f"the first seen wins. The merge is still leak-free, but check the "
              f"base corpus was not built from two different stage-4 runs.",
              file=sys.stderr)
    return index


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--authored-dir", required=True,
                    help="stage-4 corpus generated with AUTHORED_EDIT_FRACTION>0")
    ap.add_argument("--base-dir", required=True,
                    help="the corpus arm A trains on; supplies the group_id keys")
    ap.add_argument("--out-dir", required=True, help="where to write the delta")
    ap.add_argument("--dry-run", action="store_true",
                    help="count what would be written and stop")
    args = ap.parse_args()

    print(f"[1/2] indexing base corpus: {args.base_dir}")
    base = build_base_index(args.base_dir)
    print(f"      {len(base)} task_id -> group_id entries")
    if not base:
        sys.exit("base corpus has no group_ids — wrong directory?")

    shards = sorted(glob.glob(os.path.join(args.authored_dir, "*.jsonl")))
    if not shards:
        sys.exit(f"no *.jsonl under {args.authored_dir}")
    print(f"[2/2] scanning {len(shards)} authored shard(s)")
    if not args.dry_run:
        os.makedirs(args.out_dir, exist_ok=True)

    stat = Counter()
    for path in shards:
        out_lines = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                stat["read"] += 1
                try:
                    rec = json.loads(line)
                except ValueError:
                    stat["unparseable"] += 1
                    continue
                if not _authored_record(rec):
                    stat["not_authored"] += 1
                    continue
                tid = _task_id(rec)
                gid = base.get(tid) if tid else None
                if gid is None:
                    stat["dropped_no_base_match"] += 1
                    continue
                meta = rec.setdefault("metadata", {})
                meta["group_id_authored_src"] = meta.get("group_id")
                meta["group_id"] = gid            # re-key onto the base corpus
                meta["authored_delta"] = True     # so eval can find these later
                out_lines.append(json.dumps(rec, ensure_ascii=False))
                stat["kept"] += 1
        if out_lines and not args.dry_run:
            dst = os.path.join(args.out_dir, os.path.basename(path))
            with open(dst, "w") as f:
                f.write("\n".join(out_lines) + "\n")

    print("\n" + "=" * 68)
    print(f"  records read            : {stat['read']}")
    print(f"  not authored (skipped)  : {stat['not_authored']}")
    print(f"  dropped, no base match  : {stat['dropped_no_base_match']}")
    print(f"  KEPT (the delta)        : {stat['kept']}")
    if stat["read"]:
        print(f"  delta / base ratio      : {stat['kept'] / max(1, stat['read']):.1%} "
              f"of the authored corpus")
    if stat["dropped_no_base_match"]:
        print(f"\n  [warn] {stat['dropped_no_base_match']} authored segment(s) had no "
              f"matching task_id in the base.\n"
              f"         That happens when the two corpora went through different "
              f"filtering (e.g. 4b\n"
              f"         removed different chains). Re-run 4b on both with the same "
              f"settings if the\n"
              f"         count is large.")
    if args.dry_run:
        print("\n  DRY RUN — nothing written.")
    else:
        print(f"\n  wrote {args.out_dir}")


if __name__ == "__main__":
    main()
