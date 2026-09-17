#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build ``fg_catalog.json`` — the one-time functional-group description catalog.

Why this is a separate, one-time step
-------------------------------------
The scaffold pipeline calls an LLM per molecule because every Murcko scaffold is
different. A functional-group constraint is not: there are exactly **61** scorer-
visible patterns, fixed. Whatever the constraint is on instance #1 and instance
#1,900,000, it is one of those 61. So the descriptions are generated **once**, here,
and ``augment_jsonl_with_fg.py`` looks them up — which is why the per-instance driver
is pure CPU and needs no GPU at all for a 2M-row run.

Output ``fg_catalog.json``::

    {"meta": {...},
     "entries": {"fr_N_O": {fr_key, name, smarts, rdkit_desc, composition,
                            exclusions, probes, doc_freq,
                            description, template_draft, ...}, ...}}

The stored description is the group DEFINITION only, deliberately count-free. The
requirement clause ("exactly two of them" / "at least two of them") is generated per
record by ``fg_describer.count_sentence`` so it can never drift from ``eval_query``
-- the grader compares counts, and generation scores them exactly while optimization
scores them as a minimum.

Usage
-----
    PY=python

    # deterministic only (no GPU, seconds):
    $PY 2b_functional_group_gen/build_fg_catalog.py \
        --corpus data/training_data/instances/generation_2m.jsonl \
        --corpus-limit 200000

    # with LLM polish (needs a vLLM pool; see 2b_functional_group_gen.sh):
    $PY 2b_functional_group_gen/build_fg_catalog.py \
        --corpus data/training_data/instances/generation_2m.jsonl \
        --corpus-limit 200000 --polish --servers localhost:8080,localhost:8082
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from typing import Optional

from rdkit import Chem, RDLogger
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from fg_catalog import build_catalog, count_matches, DATA_CATALOG_JSON  # noqa: E402
from fg_describer import FGDescriber, normalize_text  # noqa: E402

# Written next to the generated training data, not into the repo — this file is a
# build artifact and a paraphrase bank makes it grow with --n-variants.
DEFAULT_OUT = DATA_CATALOG_JSON


# --------------------------------------------------------------------------- #
#  Corpus document frequency — powers the `rarest` constraint-selection policy
# --------------------------------------------------------------------------- #
def corpus_doc_freq(path: str, smiles_key: str, limit: int,
                    n_procs: int = 0) -> tuple[dict[str, float], int]:
    """Fraction of corpus molecules containing each fr_* group.

    Measured, not assumed: "has a benzene ring" constrains almost nothing (~0.7 of
    the corpus) while "has a beta lactam" constrains a great deal, and the selection
    policy needs to know which is which.
    """
    from concurrent.futures import ProcessPoolExecutor

    smis: list[str] = []
    with open(path) as fh:
        for line in fh:
            if limit and len(smis) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                s = json.loads(line).get(smiles_key)
            except json.JSONDecodeError:
                continue
            if s:
                smis.append(s)
    if not smis:
        return {}, 0

    n_procs = n_procs or min(32, os.cpu_count() or 8)
    chunk = max(1, len(smis) // (n_procs * 4))
    chunks = [smis[i:i + chunk] for i in range(0, len(smis), chunk)]

    totals: dict[str, int] = {}
    with ProcessPoolExecutor(max_workers=n_procs) as ex:
        for partial in tqdm(ex.map(_count_chunk, chunks), total=len(chunks),
                            desc="doc_freq", unit="chunk"):
            for k, v in partial.items():
                totals[k] = totals.get(k, 0) + v
    return {k: v / len(smis) for k, v in totals.items()}, len(smis)


def _count_chunk(smis: list[str]) -> dict[str, int]:
    """Per-worker: how many molecules in *smis* contain each group (presence, not count)."""
    cat = build_catalog()
    patts = {k: Chem.MolFromSmarts(e["smarts"]) for k, e in cat.items()}
    out: dict[str, int] = {}
    for smi in smis:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        for k, p in patts.items():
            if p is not None and mol.HasSubstructMatch(p):
                out[k] = out.get(k, 0) + 1
    return out


# --------------------------------------------------------------------------- #
#  Optional LLM polish
# --------------------------------------------------------------------------- #
def parse_ports(spec: str) -> list[int]:
    ports: list[int] = []
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            ports += list(range(int(a), int(b) + 1))
        else:
            ports.append(int(part))
    return ports


def build_base_urls(servers: Optional[str], host: Optional[str], base_ports: str) -> list[str]:
    """Same host:portspec convention as describe_scaffolds.build_base_urls."""
    urls: list[str] = []
    if host:
        ports = parse_ports(base_ports)
        for h in [x.strip() for x in host.split(",") if x.strip()]:
            urls += [f"http://{h}:{p}/v1" for p in ports]
        return urls
    for entry in (servers or "").split(","):
        entry = entry.strip()
        if not entry:
            continue
        if ":" in entry:
            h, pspec = entry.rsplit(":", 1)
            ports = parse_ports(pspec)
        else:
            h, ports = entry, parse_ports(base_ports)
        urls += [f"http://{h}:{p}/v1" for p in ports]
    return urls


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _extract_text(resp) -> str:
    if not resp.choices:
        return ""
    content = resp.choices[0].message.content or ""
    content = _THINK.sub(" ", content)
    if "<think>" in content.lower():
        content = re.split(r"</think>", content, flags=re.IGNORECASE)[-1].replace("<think>", " ")
    return " ".join(content.split()).strip()


async def polish_all(entries: dict, base_urls: list[str], model: Optional[str],
                     temperature: float, max_tokens: int, timeout: float,
                     validate_passes: int, concurrency: int,
                     n_variants: int = 1) -> dict:
    """Generate *n_variants* wordings of each entry's DEFINITION.

    61 x n_variants calls -- the whole point of doing this once. Per-instance
    generation would cost 2M calls to say the same 61 things, because the constraint
    vocabulary is fixed; a bank of wordings buys the surface diversity that stops a
    model keying on one phrasing, without pretending to add information.

    Variants are sampled at a raised temperature and kept ONLY if they pass
    validation, so a bad wording is dropped rather than shipped. The deterministic
    draft always stays as variant 0, which guarantees every entry has at least one
    valid wording even if every sample fails. The count clause is never sent to the
    model -- it is generated per record downstream.
    """
    from openai import AsyncOpenAI

    clients = [AsyncOpenAI(base_url=u, api_key="EMPTY", timeout=timeout) for u in base_urls]
    models: list[Optional[str]] = [model] * len(clients)
    if not model:
        for i, c in enumerate(clients):
            try:
                listing = await c.models.list()
                models[i] = listing.data[0].id if listing.data else None
            except Exception as exc:  # noqa: BLE001
                print(f"  [warn] {base_urls[i]}: model discovery failed: {exc}", file=sys.stderr)
    live = [i for i, m in enumerate(models) if m]
    if not live:
        raise RuntimeError("no live vLLM server (model discovery failed on every base_url)")
    print(f"# Live servers: {len(live)}/{len(clients)}; model(s): "
          f"{sorted({models[i] for i in live})}")

    d = FGDescriber()
    sem = asyncio.Semaphore(concurrency)
    jobs = [(k, v) for k in entries for v in range(n_variants)]
    pbar = tqdm(total=len(jobs), desc="paraphrase", unit="desc")

    async def _chat(i: int, messages: list[dict], temp: float) -> str:
        resp = await clients[i].chat.completions.create(
            model=models[i], messages=messages, temperature=temp,
            max_tokens=max_tokens, extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        return _extract_text(resp)

    async def _one(n: int, fr_key: str, variant: int) -> None:
        e = entries[fr_key]
        i = live[n % len(live)]
        draft = e["template_draft"]
        # Variant 0 reproduces the single-polish behaviour; later variants are pushed
        # apart with a rising temperature so they do not all collapse onto one wording.
        # CAPPED: an uncapped ramp reached ~3.2 by variant 19 and the samples came back
        # as word salad ("an five-aromatic-bond cyclic substruct composed 4 Cs").
        temp = min(temperature + 0.04 * variant, 1.0)
        try:
            async with sem:
                text = normalize_text(await _chat(i, [
                    {"role": "system", "content": d.system_prompt()},
                    {"role": "user", "content": d.user_prompt(e, draft=draft)}], temp))
                viol = d.validate(text, e)
                for _ in range(max(0, validate_passes)):
                    if not viol:
                        break
                    rev = normalize_text(await _chat(i, [
                        {"role": "system", "content": d.system_prompt()},
                        {"role": "user", "content": d.revise_prompt(e, text, viol)}], temp))
                    rviol = d.validate(rev, e)
                    if len(rviol) < len(viol):
                        text, viol = rev, rviol
                    else:
                        break
            # A wording that still contradicts the facts is worse than the draft, which
            # is faithful by construction. Drop it and record why.
            if viol:
                e.setdefault("variant_violations", []).append(viol)
            elif text and text not in e["description_variants"]:
                e["description_variants"].append(text)
        except Exception as exc:  # noqa: BLE001
            e.setdefault("polish_error", str(exc)[:200])
        pbar.update(1)

    await asyncio.gather(*(_one(n, k, v) for n, (k, v) in enumerate(jobs)))
    for e in entries.values():
        # description stays the first VALID wording, so downstream defaults are unchanged
        # whether or not a model ever ran.
        e["description"] = e["description_variants"][0]
    pbar.close()
    return entries


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=DEFAULT_OUT, help=f"output JSON (default {DEFAULT_OUT}).")
    ap.add_argument("--corpus", default=None,
                    help="JSONL used to measure per-group document frequency (rarity). "
                         "Without it, the `rarest` selection policy has no ordering.")
    ap.add_argument("--smiles-key", default="ref_smiles")
    ap.add_argument("--corpus-limit", type=int, default=200000,
                    help="molecules to sample for doc_freq (0 = all). 200k is ample.")
    ap.add_argument("--procs", type=int, default=0, help="doc_freq workers (0 = auto).")
    # LLM polish
    ap.add_argument("--polish", action="store_true", help="polish descriptions with vLLM.")
    ap.add_argument("--n-variants", type=int, default=1,
                    help="wordings to generate per entry (61 x N calls). >1 builds a "
                         "paraphrase bank the per-row pass samples from, which is what "
                         "gives the 2M set surface diversity. Only valid ones are kept.")
    ap.add_argument("--servers", default="localhost:8080", help="vLLM 'host:portspec', comma-sep.")
    ap.add_argument("--host", default=None)
    ap.add_argument("--base-ports", default="8080-8087")
    ap.add_argument("--model", default=None)
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--validate-passes", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=16)
    args = ap.parse_args()

    d = FGDescriber()
    entries: dict = {}
    for fr_key, base in build_catalog().items():
        e = dict(base)
        e["template_draft"] = d.draft(base)                # count-free definition
        e["description"] = e["template_draft"]             # deterministic default
        # Variant 0 is the deterministic draft: faithful by construction, so every
        # entry has a usable wording even when no model runs or every sample fails.
        e["description_variants"] = [e["template_draft"]]
        e["doc_freq"] = None
        entries[fr_key] = e
    print(f"# Built {len(entries)} catalog entries (deterministic drafts).")

    n_corpus = 0
    if args.corpus:
        freq, n_corpus = corpus_doc_freq(args.corpus, args.smiles_key,
                                         args.corpus_limit, args.procs)
        for fr_key, e in entries.items():
            e["doc_freq"] = round(freq.get(fr_key, 0.0), 6)
        top = sorted(freq.items(), key=lambda kv: -kv[1])[:5]
        rare = sorted(((k, v) for k, v in freq.items() if v > 0), key=lambda kv: kv[1])[:5]
        print(f"# doc_freq over {n_corpus} molecules. "
              f"most common: {[(entries[k]['name'], round(v, 3)) for k, v in top]}")
        print(f"#   rarest (nonzero): {[(entries[k]['name'], round(v, 5)) for k, v in rare]}")

    if args.polish:
        urls = build_base_urls(args.servers, args.host, args.base_ports)
        entries = asyncio.run(polish_all(
            entries, urls, args.model, args.temperature, args.max_tokens,
            args.timeout, args.validate_passes, args.concurrency, args.n_variants))
        nv = [len(e["description_variants"]) for e in entries.values()]
        rejected = sum(len(e.get("variant_violations") or []) for e in entries.values())
        errored = sum(1 for e in entries.values() if e.get("polish_error"))
        print(f"# paraphrase: variants/entry min={min(nv)} mean={sum(nv)/len(nv):.1f} "
              f"max={max(nv)} | rejected_by_validation={rejected} errors={errored}")

    # Final gate: nothing ships that contradicts its own facts.
    bad = 0
    for fr_key, e in entries.items():
        for text in e["description_variants"]:
            v = d.validate(text, e)
            if v:
                bad += 1
                print(f"  [violation] {fr_key}: {v}", file=sys.stderr)
    blob = {
        "meta": {
            "n_entries": len(entries),
            "description_is_count_free": True,
            "n_variants_requested": args.n_variants,
            "corpus": args.corpus,
            "corpus_molecules": n_corpus,
            "polished": bool(args.polish),
            "source": "molkit.utils.fragments.fr_catalog(broadest_only=True)",
        },
        "entries": entries,
    }
    with open(args.out, "w") as fh:
        json.dump(blob, fh, ensure_ascii=False, indent=1)
    print(f"# Wrote {args.out} ({len(entries)} entries, {bad} residual violation(s)).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
