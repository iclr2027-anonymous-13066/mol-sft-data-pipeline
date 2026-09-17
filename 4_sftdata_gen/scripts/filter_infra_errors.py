#!/usr/bin/env python3
"""Drop SFT records polluted by tool-server INFRASTRUCTURE errors.

During generation a transient tool-server outage can bake an infrastructure error
("cannot connect to tool server", "Server disconnected", …) into a tool message —
garbage that is NOT a real chemistry-tool result. A model trained on it learns
nothing useful (and the reasoning generator will have rationalised a network error
as chemistry). This finds every CHAIN (group_id) containing such a tool message and,
with --apply, rewrites each file dropping ALL records of the affected chains — the
whole chain is dropped because a polluted tool response can also leak into the next
segment's carried memory.

Report only (default):
    python 4_sftdata_gen/scripts/filter_infra_errors.py --dir <sftdata_subdir>
Apply (rewrite in place, atomic per file):
    python 4_sftdata_gen/scripts/filter_infra_errors.py --dir <sftdata_subdir> --apply
"""
import argparse, glob, json, os, re, shutil

INFRA = re.compile(
    r"cannot connect to tool server|Server disconnected|Connection refused|"
    r"Read timed out|Max retries exceeded|Connection aborted|Remote end closed|"
    r"Failed to establish a new connection|Connection reset",
    re.I,
)


def _polluted_gids(files):
    """First pass: collect the set of group_ids whose any tool message is an infra
    error, plus the total record count."""
    gids, total = set(), 0
    for f in files:
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            total += 1
            r = json.loads(line)
            for m in r["messages"]:
                if m.get("role") == "tool" and INFRA.search(m.get("content") or ""):
                    gids.add(r.get("metadata", {}).get("group_id"))
                    break
    return gids, total


def _drop_count(files, gids):
    n = 0
    for f in files:
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            if json.loads(line).get("metadata", {}).get("group_id") in gids:
                n += 1
    return n


def process_dir(d, apply):
    files = sorted(glob.glob(os.path.join(d, "*.jsonl")))
    if not files:
        print(f"[{os.path.basename(d)}] no *.jsonl, skip")
        return
    gids, total = _polluted_gids(files)
    drop = _drop_count(files, gids) if gids else 0
    print(f"[{os.path.basename(d)}] files={len(files)} records={total} "
          f"polluted_chains={len(gids)} records_to_drop={drop} ({100*drop/max(total,1):.3f}%)")
    if apply and gids:
        for f in files:
            recs = [json.loads(l) for l in open(f) if l.strip()]
            keep = [r for r in recs if r.get("metadata", {}).get("group_id") not in gids]
            if len(keep) != len(recs):
                tmp = f + ".tmp"
                with open(tmp, "w") as fo:
                    for r in keep:
                        fo.write(json.dumps(r, ensure_ascii=False) + "\n")
                shutil.move(tmp, f)
        print(f"[{os.path.basename(d)}] applied — dropped {drop} records from {len(gids)} chains")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, nargs="+", help="one or more sftdata subdirs")
    ap.add_argument("--apply", action="store_true", help="rewrite files in place (default: report only)")
    args = ap.parse_args()
    for d in args.dir:
        process_dir(d, args.apply)


if __name__ == "__main__":
    main()
