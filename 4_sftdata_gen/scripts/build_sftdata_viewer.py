#!/usr/bin/env python
"""Build the standalone HTML viewer for a stage-4 sftdata directory.

Scans every ``*.jsonl`` chunk (safe on a directory still being written — a
half-flushed final line is skipped), aggregates the composition stats, draws a
random sample per ``example_type`` plus one complete trajectory, and injects all
of it into ``sftdata_viewer_template.html`` as one self-contained page.

    python 4_sftdata_gen/scripts/build_sftdata_viewer.py \
        --input-dir data/training_data/sftdata/generation_2m_scaffold \
        --out /tmp/sftdata_viewer.html

Open the result in a browser, or hand the path to Claude to (re)publish it as an
artifact.  RDKit is optional: without it the "seed form" panel is dropped and
everything else still renders.
"""

from __future__ import annotations

import argparse
import collections
import datetime
import glob
import json
import os
import random
import re
import sys

TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "sftdata_viewer_template.html")
DEFAULT_DIR = "data/training_data/sftdata/generation_2m_scaffold"

try:
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
except Exception:  # pragma: no cover
    Chem = None

_STEREO = re.compile(r"@@|@")


def _ring_digits(smi: str) -> list[str]:
    """Ring-closure digits of *smi*, skipping bracket-atom internals."""
    out, i = [], 0
    while i < len(smi):
        c = smi[i]
        if c == "[":
            i = smi.index("]", i) + 1
            continue
        if c == "%":
            out.append(smi[i + 1:i + 3])
            i += 3
            continue
        if c.isdigit():
            out.append(c)
            i += 1
            continue
        i += 1
    return out


def _percentiles(values: list[int]) -> dict:
    s = sorted(values)
    n = len(s) or 1
    return {"mean": round(sum(s) / n), "p50": s[n // 2],
            "p90": s[int(n * 0.9)], "max": s[-1]}


def scan(input_dir: str, per_type: int, seed: int) -> dict:
    random.seed(seed)
    files = sorted(glob.glob(os.path.join(input_dir, "*.jsonl")))
    if not files:
        sys.exit(f"no *.jsonl under {input_dir}")

    counters = {k: collections.Counter() for k in (
        "example_type", "num_tool_calls", "segment_total", "tool_usage",
        "msg_roles", "n_messages", "assistant_turns", "parallel_calls")}
    lengths = {k: [] for k in ("prompt_chars", "asst_chars", "record_chars")}
    by_type: dict[str, list] = collections.defaultdict(list)
    seed_form = collections.Counter()
    seed_examples: list[dict] = []
    groups: set[str] = set()
    tools_available = None
    file_rows, total, has_answer = [], 0, 0
    traj_gid = None
    traj: list[dict] = []

    for path in files:
        n = 0
        with open(path) as fh:
            for line in fh:
                if not line.endswith("\n"):
                    break            # chunk still being written — drop the tail
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                n += 1
                total += 1
                meta = rec.get("metadata") or {}
                msgs = rec.get("messages") or []
                counters["example_type"][meta.get("example_type", "?")] += 1
                counters["num_tool_calls"][meta.get("num_tool_calls")] += 1
                counters["segment_total"][meta.get("segment_total")] += 1
                counters["n_messages"][len(msgs)] += 1
                if meta.get("group_id"):
                    groups.add(meta["group_id"])
                if traj_gid is None and meta.get("segment_total") == 5:
                    traj_gid = meta["group_id"]

                n_asst = 0
                for msg in msgs:
                    role = msg.get("role")
                    counters["msg_roles"][role] += 1
                    if role == "user" and len(lengths["prompt_chars"]) < 20000:
                        lengths["prompt_chars"].append(len(msg.get("content") or ""))
                    if role == "assistant":
                        n_asst += 1
                        calls = msg.get("tool_calls") or []
                        if calls:
                            counters["parallel_calls"][len(calls)] += 1
                        for call in calls:
                            counters["tool_usage"][call["function"]["name"]] += 1
                        if len(lengths["asst_chars"]) < 20000:
                            lengths["asst_chars"].append(len(msg.get("content") or ""))
                counters["assistant_turns"][n_asst] += 1
                if msgs and "<ANSWER>" in (msgs[-1].get("content") or ""):
                    has_answer += 1
                if tools_available is None and rec.get("tools"):
                    tools_available = [t["function"]["name"] for t in rec["tools"]]
                if len(lengths["record_chars"]) < 20000:
                    lengths["record_chars"].append(len(line))

                # seed segment: canonical vs SMARTS-transcribed, and how much of
                # the SMARTS can simply be copied across (see seed_order.py)
                if Chem is not None and meta.get("segment_index") == 0 and len(msgs) == 7:
                    args = {c["function"]["name"]: json.loads(c["function"]["arguments"])
                            for c in msgs[2].get("tool_calls") or []}
                    smi = (args.get("analyze_properties") or {}).get("mol_smiles")
                    query = (args.get("match_substructure") or {}).get("query")
                    mol = Chem.MolFromSmiles(smi) if smi else None
                    if mol is not None and query:
                        canon = Chem.MolToSmiles(mol)
                        seed_form["n"] += 1
                        if canon == smi:
                            seed_form["canonical"] += 1
                        else:
                            seed_form["transcribed"] += 1
                            if len(seed_examples) < 3 and len(smi) < 46:
                                seed_examples.append(
                                    {"smarts": query, "seed": smi, "canonical": canon})
                        digits = _ring_digits(query)
                        if digits:
                            seed_form["ring_n"] += 1
                            seed_form["ring_seed"] += _ring_digits(smi) == digits
                            seed_form["ring_canon"] += _ring_digits(canon) == digits
                        stereo = _STEREO.findall(query)
                        if stereo:
                            seed_form["ste_n"] += 1
                            seed_form["ste_seed"] += _STEREO.findall(smi) == stereo
                            seed_form["ste_canon"] += _STEREO.findall(canon) == stereo

                if traj_gid and meta.get("group_id") == traj_gid:
                    traj.append({"metadata": meta, "messages": msgs})

                bucket = by_type[meta.get("example_type", "?")]
                if len(bucket) < per_type:
                    bucket.append({"metadata": meta, "messages": msgs})
                else:                                    # reservoir sample
                    j = random.randrange(counters["example_type"][meta.get("example_type", "?")])
                    if j < per_type:
                        bucket[j] = {"metadata": meta, "messages": msgs}

        file_rows.append({"name": os.path.basename(path), "records": n,
                          "bytes": os.path.getsize(path),
                          "done": os.path.exists(path + ".done")})
        print(f"scanned {os.path.basename(path)}: {n} records", file=sys.stderr)

    samples = [r for rs in by_type.values() for r in rs]
    samples.sort(key=lambda r: (r["metadata"].get("example_type", ""),
                                r["metadata"].get("segment_total", 0),
                                r["metadata"].get("segment_index", 0)))
    traj.sort(key=lambda r: r["metadata"].get("segment_index", 0))

    payload = {
        "dir": input_dir,
        "scanned_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "total": total,
        "groups": len(groups),
        "has_answer": has_answer,
        "tools_available": tools_available or [],
        "files": file_rows,
        "samples": samples,
        "trajectory": traj,
        "len_stats": {k: _percentiles(v or [0]) for k, v in lengths.items()},
    }
    payload.update({k: dict(v) for k, v in counters.items()})
    if seed_form["n"]:
        payload["seed_form"] = {
            "n": seed_form["n"],
            "canonical": seed_form["canonical"],
            "transcribed": seed_form["transcribed"],
            "examples": seed_examples,
            "copy": [
                {"label": "ring numbering", "n": seed_form["ring_n"],
                 "seed": seed_form["ring_seed"], "canon": seed_form["ring_canon"],
                 "note": "share where the SMARTS ring-closure digits carry over unchanged"},
                {"label": "stereo notation", "n": seed_form["ste_n"],
                 "seed": seed_form["ste_seed"], "canon": seed_form["ste_canon"],
                 "note": "share where the SMARTS @/@@ carries over unchanged"},
            ],
        }
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", default=DEFAULT_DIR)
    ap.add_argument("--out", default="/tmp/sftdata_viewer.html")
    ap.add_argument("--samples-per-type", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if Chem is None:
        print("rdkit unavailable — the seed-form panel will be omitted", file=sys.stderr)
    payload = scan(args.input_dir, args.samples_per_type, args.seed)
    template = open(TEMPLATE).read()
    # `<` is escaped so the JSON can never close the host <script> element.
    blob = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
    with open(args.out, "w") as fh:
        fh.write(template.replace("__PAYLOAD__", blob))
    print(f"{payload['total']} records, {payload['groups']} trajectories "
          f"→ {args.out} ({os.path.getsize(args.out) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
