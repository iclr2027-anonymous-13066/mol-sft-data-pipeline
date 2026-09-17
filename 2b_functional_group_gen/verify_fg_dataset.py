#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Verify a produced FG dataset against the grader and against itself.

Run this after ``augment_jsonl_with_fg.py``. It answers four questions that the
generator cannot answer about its own output:

  1. Does each row's ``eval_query`` SMARTS still count exactly what the RDKit
     ``fr_*`` function counts? (the whole scoring premise)
  2. Do the stored phrasings (``description`` / ``description_definition`` /
     ``description_name``) rebuild exactly from the catalog and this row's members,
     and does the requirement clause appear exactly when it carries information?
  3. Does ``fg_verified`` agree with re-evaluating ``eval_query`` from scratch?
  4. Do any descriptions violate their catalog facts?
  5. Is the seed a real molecule that already carries the whole constraint?

Usage:
    PY=python
    $PY 2b_functional_group_gen/verify_fg_dataset.py \
        data/training_data/instances/benchmark_fg \
        --sample 20000
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import re
import sys

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from fg_catalog import (  # noqa: E402
    build_catalog, catalog_json_path, count_matches, fr_catalog,
)
from fg_describer import (  # noqa: E402
    count_sentence, needs_count_sentence, render_constraint_parts,
    render_name_style, validate_text,
)
from fg_seed import satisfies_members  # noqa: E402

# The live RDKit fr_* callables, i.e. exactly what evaluate_benchmark.py counts with.
_FR_FUNC = {k: v["func"] for k, v in fr_catalog(broadest_only=True).items()}

_WORD2NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
             "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}


def iter_rows(path: str):
    files = ([path] if os.path.isfile(path)
             else sorted(glob.glob(os.path.join(path, "*.jsonl"))))
    if not files:
        sys.exit(f"error: no .jsonl found at {path}")
    for f in files:
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="output file or shard folder")
    ap.add_argument("--sample", type=int, default=20000, help="rows to check (0 = all)")
    ap.add_argument("--seed", type=int, default=0,
                    help="the --seed the dataset was built with; needed to reproduce "
                         "which paraphrase each row drew.")
    args = ap.parse_args()

    cat = build_catalog()
    # The stored per-group definitions (LLM-polished when the catalog was built with
    # --polish). Rebuilding from build_catalog() alone would silently compare against
    # the deterministic drafts instead of what actually shipped.
    descr_by_key: dict = {}
    cat_json = catalog_json_path()
    try:
        with open(cat_json) as fh:
            blob = json.load(fh)
        entries = blob.get("entries") or {}
        descr_by_key = {k: (e.get("description_variants")
                            or ([e["description"]] if e.get("description") else []))
                        for k, e in entries.items()}
        for k, e in entries.items():
            if k in cat:
                cat[k] = {**cat[k], **{x: e[x] for x in ("probes", "probe_positive",
                                                         "exclusions", "composition")
                                       if x in e}}
    except OSError:
        print(f"# [warn] no {cat_json}; comparing against deterministic drafts",
              file=sys.stderr)

    rows = []
    for i, r in enumerate(iter_rows(args.path)):
        if args.sample and len(rows) >= args.sample and i % 97:
            continue
        rows.append(r)
        if args.sample and len(rows) > args.sample * 3:
            break
    if args.sample and len(rows) > args.sample:
        random.seed(args.seed)
        rows = random.sample(rows, args.sample)

    fail = {"smarts_vs_fr": 0, "desc_vs_query": 0, "verified_mismatch": 0,
            "facts_violation": 0, "unresolved": 0, "seed_bad": 0}
    checked = {"rows": 0, "members": 0, "with_mol": 0}
    examples: dict[str, list] = {k: [] for k in fail}

    def note(key, detail):
        fail[key] += 1
        if len(examples[key]) < 3:
            examples[key].append(detail)

    for r in rows:
        checked["rows"] += 1
        members = r.get("fg_constraint") or []
        eq = r.get("eval_query") or {}
        desc = r.get("description") or ""
        smi = r.get("smiles") or ""
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is not None:
            checked["with_mol"] += 1

        # Rebuild both phrasings from the catalog and the row's own members. The
        # per-member SEGMENTS matter: a two-group definition legitimately cites one
        # probe molecule for the first group and another for the second, so checking
        # the whole text against either group's probe table reports a false clash.
        # Reproduce the driver's per-row draw so the rebuild is exact even when the
        # catalog holds a paraphrase bank. A --seed mismatch shows up here as a
        # difference, which is the correct outcome: the data was built differently.
        draw = int.from_bytes(hashlib.blake2b(
            f"{args.seed}:{smi}".encode(), digest_size=8).digest(), "big")
        picked = {}
        for m in members:
            fr = m.get("fr_key")
            bank = descr_by_key.get(fr) or []
            if bank:
                picked[fr] = bank[(draw + len(picked)) % len(bank)]
        parts = render_constraint_parts(members, cat, picked)
        rebuilt_def = " ".join(t for _, t in parts).strip()
        rebuilt_name = render_name_style(members, opener=draw % 997)

        # (2a) stored phrasings must be exactly what the catalog + members produce
        if members:
            if (r.get("description_definition") or "") != rebuilt_def:
                note("desc_vs_query", (r.get("id"), "description_definition != rebuild"))
            if (r.get("description_name") or "") != rebuilt_name:
                note("desc_vs_query", (r.get("id"), "description_name != rebuild"))
            style = r.get("description_style")
            expect = rebuilt_name if style == "name" else rebuilt_def
            if desc != expect:
                note("desc_vs_query",
                     (r.get("id"), f"description does not match its style {style!r}"))

        for m, qm in zip(members, eq.get("members") or []):
            checked["members"] += 1
            entry = cat.get(m.get("fr_key") or "")
            if not entry:
                note("unresolved", (r.get("id"), m.get("name")))
                continue
            # (1) the scoring premise: SMARTS count == fr_*() count
            if mol is not None:
                by_smarts = count_matches(mol, qm["smarts"])
                by_fr = int(_FR_FUNC[m["fr_key"]](mol))
                if by_smarts != by_fr:
                    note("smarts_vs_fr", (r.get("id"), m["fr_key"], by_smarts, by_fr))
            # (2b) eval_query must mirror fg_constraint
            want_op = "==" if m["match_mode"] == "exact" else ">="
            if qm["op"] != want_op or qm["count"] != m["count"]:
                note("desc_vs_query", (r.get("id"), "eval_query disagrees with fg_constraint"))
            # (2c) the requirement clause is present exactly when it carries
            #      information, and then states this member's number and operator.
            want_clause = needs_count_sentence(m["count"], m["match_mode"])
            clause = count_sentence(m["count"], m["match_mode"])
            in_def = clause in rebuilt_def
            if want_clause != in_def:
                note("desc_vs_query",
                     (r.get("id"), f"count clause present={in_def}, expected={want_clause}"))
            # (2d) the name phrasing must mention every member by name
            if m["name"].lower() not in rebuilt_name.lower():
                note("desc_vs_query", (r.get("id"), f"name style omits {m['name']!r}"))
            # (4) each member's own definition segment must not contradict its facts
            seg = next((t for mm, t in parts if mm is m), "")
            v = validate_text(seg, entry, m["count"], m["match_mode"])
            if v:
                note("facts_violation", (r.get("id"), v[:2]))

        # (5) the seed must be a real molecule that carries the constraint. A seed
        #     that does not is worse than none: the trajectory would start from a
        #     structure that already fails the very thing it is meant to guarantee.
        if members:
            seed = r.get("seed_smiles")
            if not seed:
                note("seed_bad", (r.get("id"), f"no seed (source={r.get('seed_source')})"))
            elif Chem.MolFromSmiles(seed) is None:
                note("seed_bad", (r.get("id"), f"unparseable seed {seed!r}"))
            elif not satisfies_members(seed, members):
                note("seed_bad", (r.get("id"), f"seed {seed!r} misses its own constraint"))
            elif not r.get("seed_verified"):
                note("seed_bad", (r.get("id"), "seed_verified is false but the seed passes"))

        # (3) fg_verified must match a fresh evaluation
        if mol is not None and members:
            ok = all(count_matches(mol, qm["smarts"]) == qm["count"]
                     if qm["op"] == "=="
                     else count_matches(mol, qm["smarts"]) >= qm["count"]
                     for qm in (eq.get("members") or []) if qm.get("smarts"))
            if bool(r.get("fg_verified")) != bool(ok):
                note("verified_mismatch", (r.get("id"), r.get("fg_verified"), ok))

    print(f"checked rows={checked['rows']} members={checked['members']} "
          f"(with parseable molecule: {checked['with_mol']})")
    print()
    labels = {
        "smarts_vs_fr":      "eval_query SMARTS count != RDKit fr_*() count",
        "desc_vs_query":     "description disagrees with eval_query",
        "verified_mismatch": "fg_verified disagrees with a fresh evaluation",
        "facts_violation":   "description contradicts its catalog facts",
        "unresolved":        "constraint name not in the catalog",
        "seed_bad":          "seed molecule missing / invalid / misses its constraint",
    }
    bad = 0
    for k, label in labels.items():
        n = fail[k]
        bad += n
        print(f"{'FAIL' if n else 'PASS'}  {label}: {n}")
        for ex in examples[k]:
            print(f"        e.g. {ex}")
    print()
    print("ALL GREEN" if not bad else f"{bad} problem(s) found")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
