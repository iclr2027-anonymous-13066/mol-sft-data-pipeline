#!/usr/bin/env python
"""Audit constructive substructure-generation SFT data for logical-flow breaks.

Checks that the reasoning + tool-call sequence the model learns from is
internally consistent.  Run after ``4_sftdata_gen`` on the output dir/file:

    python 4_sftdata_gen/scripts/audit_constructive_sftdata.py \
        /data/.../sftdata/generation_benchmark_scaffold

Checks performed (one record = one chain's whole conversation):

  * mol_smiles_discontinuity   — a tool call's mol_smiles != the molecule the
                                 assistant currently holds (broken SMILES chain)
  * reasoning_leaks_result     — an edit's reasoning already contains the
                                 post-edit SMILES the tool is about to return
  * element_claim_mismatch     — reasoning claims "atom N is a <X>" but the
                                 labelled SMILES shows a different element
  * stray_control_tag          — a tool-call turn contains a stray <ANSWER>
                                 tag (control tokens leaked into reasoning)
  * cot_scratchpad_leak        — reasoning contains a "Thinking Process:" /
                                 numbered scratchpad / meta-instruction echo
  * empty_reasoning            — a tool-call turn has no reasoning text
  * props_without_measurement  — the Verification block reports a measured
                                 property value with no analyze_properties or
                                 match_substructure call to back it

Exit code is non-zero when any non-false-positive issue is found.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

EDIT_TOOLS = {"edit_fragment"}
ANALYZE_TOOLS = {"analyze_properties"}

_ELEM_RE = re.compile(
    r"atom(?:\s*index)?\s*\*?\*?(\d+)\*?\*?\s+is\s+a[n]?\s+\*?\*?([A-Za-z]{1,2})\b", re.I
)
_TAG_RE = re.compile(r"</?ANSWER>", re.I)
_COT_RE = re.compile(r"thinking process|chain of thought", re.I)
_NUM_RE = re.compile(r"^\s*1\.\s*\*\*", re.M)
_META_RE = re.compile(r"\b(the user wants|generate training data|as an ai)\b", re.I)


def _calls(msg: dict):
    out = []
    for tc in msg.get("tool_calls", []) or []:
        fn = tc["function"]
        try:
            args = json.loads(fn["arguments"])
        except json.JSONDecodeError:
            args = {}
        out.append((fn["name"], args))
    return out


def _elem_at(labeled: str, idx: int):
    m = re.search(r"\[([^\[\]]+?):%d\]" % idx, labeled)
    if not m:
        return None
    t = re.match(r"([A-Za-z][a-z]?)", m.group(1))
    if not t:
        return None
    s = t.group(1)
    return s.upper() if len(s) == 1 else s


def _norm_claim(claim: str) -> str:
    """'CH'/'NH'/'OH' (element + hydrogen) → base element letter."""
    if len(claim) == 2 and claim[1] in "Hh" and claim[0].upper() in "CNOSPBF":
        return claim[0].upper()
    return claim.upper() if len(claim) == 1 else claim


def _prose(content: str) -> str:
    p = content or ""
    for tag in ("Verification:", "<ANSWER>"):
        i = p.find(tag)
        if i != -1:
            p = p[:i]
    return p.strip()


def load(path: str):
    p = Path(path)
    files = sorted(p.glob("*.jsonl")) if p.is_dir() else [p]
    rows = []
    for fp in files:
        rows += [json.loads(l) for l in open(fp) if l.strip()]
    return rows


def audit(rows):
    issues = Counter()
    examples = defaultdict(list)

    def flag(kind, gid, detail):
        issues[kind] += 1
        if len(examples[kind]) < 3:
            examples[kind].append(f"{gid}: {detail}")

    for rec in rows:
            gid = rec["metadata"]["group_id"]
            msgs = rec["messages"]
            produced = None
            last_labeled = None
            seq = list(msgs)
            k = 0
            while k < len(seq):
                m = seq[k]
                if m["role"] == "assistant" and m.get("tool_calls"):
                    content = m.get("content", "") or ""
                    cs = _calls(m)
                    if not _prose(content) and m.get("tool_calls"):
                        flag("empty_reasoning", gid, f"{[c[0] for c in cs]}")
                    if _TAG_RE.search(content):
                        flag("stray_control_tag", gid, f"{content[:60]!r}")
                    if _COT_RE.search(content) or _NUM_RE.search(content) or _META_RE.search(content):
                        flag("cot_scratchpad_leak", gid, f"{content[:60]!r}")
                    resp = []
                    j = k + 1
                    while j < len(seq) and seq[j]["role"] == "tool":
                        resp.append(seq[j]["content"]); j += 1
                    for (name, args) in cs:
                        ms = args.get("mol_smiles")
                        if ms is not None:
                            if produced is None:
                                produced = ms
                            elif ms.strip() != produced.strip():
                                flag("mol_smiles_discontinuity", gid,
                                     f"{name}: arg!=current ({ms[:22]} vs {produced[:22]})")
                        if name in EDIT_TOOLS and last_labeled:
                            for cm in _ELEM_RE.finditer(content):
                                ci, claim = int(cm.group(1)), cm.group(2)
                                te = _elem_at(last_labeled, ci)
                                if te and _norm_claim(claim) != te.upper():
                                    flag("element_claim_mismatch", gid,
                                         f"claims atom{ci}={claim} but labeled={te}")
                    for (name, _), rr in zip(cs, resp + [None] * len(cs)):
                        if rr is None:
                            continue
                        rr_s = rr.strip()
                        if name in EDIT_TOOLS and rr_s and not rr_s.startswith("{"):
                            if rr_s in content:
                                flag("reasoning_leaks_result", gid, f"{name}")
                            produced = rr_s
                        if name == "label_atom_indices" and rr_s:
                            last_labeled = rr_s
                    k = j
                else:
                    k += 1

            # A measured property value (or "substructure present") must be
            # backed by an analyze/match call somewhere in the conversation.
            final_txt = msgs[-1]["content"]
            shows_measure = bool(re.search(r"current \d", final_txt)) or "→ present →" in final_txt
            if shows_measure:
                names_here = [c[0] for m in msgs if m["role"] == "assistant" for c in _calls(m)]
                measured_here = (bool(ANALYZE_TOOLS.intersection(names_here))
                                 or "match_substructure" in names_here)
                if not measured_here:
                    flag("props_without_measurement", gid,
                         "shows values but no analyze/match call in the chain")

    return issues, examples


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: audit_constructive_sftdata.py <jsonl-or-dir>")
    rows = load(sys.argv[1])
    groups = {r["metadata"]["group_id"] for r in rows}
    print(f"records: {len(rows)} | groups: {len(groups)}")
    print("=" * 60)
    issues, examples = audit(rows)
    if not issues:
        print("ALL CLEAN — no logical-flow issues found.")
        return
    for k, v in issues.most_common():
        print(f"{v:5d}  {k}")
        for e in examples[k]:
            print(f"        - {e}")
    sys.exit(1)


if __name__ == "__main__":
    main()
