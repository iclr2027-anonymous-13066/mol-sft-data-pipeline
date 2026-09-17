#!/usr/bin/env python3
"""
compare_descriptions.py
=======================
A development harness that regenerates stage-2 scaffold descriptions **for a handful of
instances** and lays the results of several models and prompt variants side by side. It
exists for the loop of eyeballing quality, editing the prompt and running again, before
committing to a full 900-instance batch.

The pipeline logic is reused as is, never reimplemented:
  - scaffold_analyzer.analyze_scaffold   : SMILES -> the structure analysis dict (an)
  - scaffold_describer.format_facts      : an -> the authoritative FACTS block
  - scaffold_describer.render_template   : an -> the deterministic controlled-NL draft
  - scaffold_describer.validate_text     : polished text vs. the facts (hallucination /
                                           style violations)
  - describe_scaffolds.VLLMPool          : model auto-detection, retries, <think> stripping
                                           and the qwen / gpt-oss reasoning-parameter split

An input JSONL record carries only a summary, not the original analysis dict, so an is
recovered by running analyze_scaffold again on the SMILES. That is deterministic, so the
result is identical to the pipeline's.

Examples
-------
  PY=python

  # inspect FACTS / DRAFT / the existing description without calling a model
  $PY compare_descriptions.py --input ../..data/benchmark/generation_benchmark-00000.jsonl \
      --n 3 --dry-run

  # compare two models, with HTML (without --prompts the current prompt is used)
  $PY compare_descriptions.py --sample 8 --seed 0 \
      --html description_results/cmp.html --out description_results/cmp.jsonl

  # the current prompt (scaffold_prompts.py = v8) vs. the one before the revision
  $PY compare_descriptions.py --sample 60 --seed 0 --kind ring --max-sentences 4 \
      --prompts prompt_var/default.py
"""
from __future__ import annotations

import argparse
import hashlib
import asyncio
import html as _html
import importlib.util
import json
import os
import random
import re
import sys
import time
from typing import Any, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rdkit import RDLogger  # noqa: E402

import scaffold_prompts  # noqa: E402
from describe_scaffolds import VLLMPool, analyze_one, _split_sentences  # noqa: E402
from scaffold_describer import (  # noqa: E402
    ScaffoldDescriber, format_facts, normalize_text, render_template, validate_text,
)

RDLogger.DisableLog("rdApp.*")

DEFAULT_INPUT = os.path.join(_ROOT, "data", "benchmark",
                             "generation_benchmark-00000.jsonl")
DEFAULT_ENDPOINTS = ["qwen=http://localhost:20000/v1",
                     "gptoss=http://localhost:20200/v1"]

_PROMPT_NAMES = ("SYSTEM_PROMPT", "USER_TEMPLATE", "REVISE_TEMPLATE", "CONDENSE_TEMPLATE")


def apply_facts_rewrites(facts: str, rewrites) -> str:
    """Delete or rewrite FACTS lines.

    rewrites is a sequence of (pattern, repl). A repl of None deletes the line; a string is
    applied with re.sub. Only the first match is used.

    Why this exists: some phrasing the prompt cannot suppress is triggered by one specific
    FACTS line. A measured case —
      "- Overall: all aromatic; scaffold heteroatoms: 6 nitrogens."
    (a) the heteroatom total on that line is a partial sum of the per-ring lines (the
    analyzer walks ring systems only) yet demands a subject covering the whole, and
    (b) the word "Overall" is itself a cue to "summarise into one". Meanwhile (c) the
    aromatic-mixture flag really does disappear from the output when removed (gpt-oss
    mentions of saturation: 9/30 -> 4/30). So the line needs rewriting, not deleting.
    """
    if not rewrites:
        return facts
    out = []
    for ln in facts.split("\n"):
        for pat, repl in rewrites:
            if re.search(pat, ln):
                if repl is not None:
                    out.append(re.sub(pat, repl, ln))
                break
        else:
            out.append(ln)
    return "\n".join(out)


def strip_facts_lines(facts: str, patterns) -> str:
    """The older interface, for prompt files that use STRIP_FACTS_LINES (v6, v7s)."""
    return apply_facts_rewrites(facts, tuple((p, None) for p in (patterns or ())))


# Opening-strategy candidates. This wording ships inside the prompt, so it is prompt
# CONTENT: it lives in the prompt file as OPENING_CANDIDATES and is versioned with it.
# (Keeping it in the tool would make every version's behaviour change together whenever the
# file is edited, and past measurements would no longer reproduce.) Files that do not define
# it fall back to the list below.
_OPENING_FALLBACK = {
    "always": ["the main ring itself - name it first",
               "the heteroatom pattern - lead with the ring heteroatoms"],
    "linker": ["the linker between the rings - make the linker the subject",
               "the overall arrangement - how the parts sit relative to each other"],
    "fused": ["the fused/spiro/bridged junction - lead with how the rings are joined"],
    "carbonyl": ["the ring carbonyl (lactam / lactone / cyclic urea) - lead with it"],
    "many": ["how many separate rings there are and what they are"],
    "arom": ["the aromatic-versus-saturated contrast across the rings"],
}


class PromptSet:
    """A named bundle of prompts. Any name missing from the override file falls back to
    scaffold_prompts."""

    def __init__(self, name: str, mod: Any = None) -> None:
        self.name = name
        for attr in _PROMPT_NAMES:
            setattr(self, attr, getattr(mod, attr, None) if mod is not None else None)
            if getattr(self, attr) is None:
                setattr(self, attr, getattr(scaffold_prompts, attr))
        self.OPENING_CANDIDATES = (getattr(mod, "OPENING_CANDIDATES", None)
                                   if mod is not None else None) or _OPENING_FALLBACK
        # Use FACTS_REWRITES when present; otherwise take STRIP_FACTS_LINES as (pattern, None).
        rw = (getattr(mod, "FACTS_REWRITES", None) if mod is not None else None)
        if rw is None:
            strip = (getattr(mod, "STRIP_FACTS_LINES", None) if mod is not None else None) or ()
            rw = tuple((p, None) for p in strip)
        self.FACTS_REWRITES = tuple(rw)
        self.STRIP_FACTS_LINES = self.FACTS_REWRITES

    @property
    def overridden(self) -> list[str]:
        return [a for a in _PROMPT_NAMES
                if getattr(self, a) != getattr(scaffold_prompts, a)]


def load_prompt_set(path: str) -> PromptSet:
    """Load a python file as a PromptSet. Defining SYSTEM_PROMPT alone is enough."""
    name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(f"_prompts_{name}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load prompt file: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return PromptSet(name, mod)


def opening_strategy(an: dict, candidates: Optional[dict] = None) -> str:
    """Pick, deterministically, the opening strategy assigned to this scaffold.

    Telling the model to "vary the opening" was measured to be insufficient: 58 of 60
    descriptions still opened with A/An, across 3-4 distinct first words. So one feature that
    actually exists in the structure is chosen by hashing the scaffold SMILES and assigned
    explicitly. Being a hash, it reproduces independently of the model, and the same scaffold
    always receives the same strategy. Only candidates that genuinely apply to that scaffold
    are offered — a scaffold with a single ring system cannot be told to "open on the linker".

    candidates comes from the prompt file's OPENING_CANDIDATES, because the wording is
    prompt content and has to move with the version. Falls back to _OPENING_FALLBACK.
    """
    pool = candidates or _OPENING_FALLBACK
    rss = an.get("ring_systems") or []
    conns = an.get("connections") or []
    keys = ["always"]
    if conns:
        keys.append("linker")
    if any(r.get("n_rings", 1) > 1 for r in rss):
        keys.append("fused")
    if any(r.get("carbonyl_desc") for r in rss):
        keys.append("carbonyl")
    if len(rss) > 2:
        keys.append("many")
    if an.get("aromaticity") in ("mixed aromatic/aliphatic", "all aliphatic (saturated)"):
        keys.append("arom")
    cands = [c for k in keys for c in pool.get(k, [])] or list(_OPENING_FALLBACK["always"])
    key = (an.get("scaffold_smiles") or "").encode()
    h = int(hashlib.md5(key).hexdigest()[:8], 16) if key else 0
    return cands[h % len(cands)]


class FixedExtraBodyPool(VLLMPool):
    """A pool that forces extra_body to a fixed value.

    VLLMPool._extra_body guesses the model family from its name to decide the reasoning
    parameters (describe_scaffolds.py:195-211). It only branches on gpt-oss and qwen, so
    deepseek, nemotron and others fall through to `other` and get **no reasoning control at
    all**. For those, check what the server actually accepts and pass it with --extra-body.
    """

    _fixed: Optional[dict] = None

    def _extra_body(self, model, think: Optional[bool] = None) -> Optional[dict]:
        return self._fixed


class OverrideDescriber(ScaffoldDescriber):
    """Hold the prompts on the instance, avoiding a module-global monkeypatch and the race it
    would create under concurrency.

    The default ScaffoldDescriber reads the module-level SYSTEM_PROMPT and friends from
    scaffold_describer. Running variants concurrently rules out touching those globals, so
    every one of them is overridden here.
    """

    def __init__(self, ps: PromptSet) -> None:
        self._ps = ps

    def system_prompt(self) -> str:
        return self._ps.SYSTEM_PROMPT

    def user_prompt(self, an: dict, draft: Optional[str] = None) -> str:
        # Prompt files that do not use {opening} keep working: str.format ignores unused
        # keyword arguments, so default.py and v1.py are unaffected.
        return self._ps.USER_TEMPLATE.format(
            facts=apply_facts_rewrites(format_facts(an), self._ps.FACTS_REWRITES),
            draft=draft if draft is not None else render_template(an),
            opening=opening_strategy(an, self._ps.OPENING_CANDIDATES))

    def revise_prompt(self, an: dict, draft: str,
                      violations: Optional[list] = None) -> str:
        vlines = "\n".join(f"  - {x}" for x in (violations or [])) or "  - (unspecified)"
        return self._ps.REVISE_TEMPLATE.format(
            facts=apply_facts_rewrites(format_facts(an), self._ps.FACTS_REWRITES),
                                               violations=vlines, draft=draft)

    def condense_prompt(self, text: str, max_sentences: int) -> str:
        return self._ps.CONDENSE_TEMPLATE.format(max_sentences=max_sentences, text=text)


# --------------------------------------------------------------------------- #
#  polish + validate/revise loop
# --------------------------------------------------------------------------- #
async def polish_with_validation(pool: VLLMPool, an: dict, draft: str,
                                 validate_passes: int, server_idx: int = 0,
                                 normalize: bool = True) -> dict:
    """Polish the draft with the LLM, then keep repairing while the violation count falls.

    A straight port of the LLM section of describe_scaffolds.describe_one, which cannot be
    called directly because it is tied to file handles, the pbar and stats. If that logic
    changes, this has to follow.

    server_idx is folded with `% n_servers` inside the pool to choose a server. When one
    endpoint was given several URLs, each instance must pass a different value for the load
    to spread.
    """
    _norm = normalize_text if normalize else (lambda s: s)
    t0 = time.monotonic()
    raw = await pool.describe(server_idx, an, draft=draft)
    desc = _norm(raw)
    # Validate the normalised text, so every downstream metric sees the final wording.
    viol = validate_text(desc, an)
    n_revise = 0
    for _ in range(max(0, validate_passes)):
        if not viol:
            break
        rev = _norm(await pool.revise(server_idx, an, desc, viol))
        n_revise += 1
        rviol = validate_text(rev, an)
        if len(rviol) < len(viol):
            desc, viol = rev, rviol
        else:
            break
    out = {"text": desc, "violations": viol, "n_revise": n_revise,
           "seconds": round(time.monotonic() - t0, 2)}
    if normalize and raw != desc and n_revise == 0:
        out["raw_text"] = raw   # keep the pre-normalisation text for auditing
    return out


# --------------------------------------------------------------------------- #
#  Instance selection
# --------------------------------------------------------------------------- #
def select_records(path: str, args: argparse.Namespace) -> list[tuple[int, dict]]:
    rows: list[tuple[int, dict]] = []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if line:
                rows.append((i, json.loads(line)))
    if args.kind != "all":
        rows = [(i, r) for i, r in rows if r.get("scaffold_kind") == args.kind]
    if args.ids:
        want = [x.strip() for x in args.ids.split(",") if x.strip()]
        by_id = {r.get("id"): (i, r) for i, r in rows}
        missing = [w for w in want if w not in by_id]
        if missing:
            raise SystemExit(f"ids not found: {', '.join(missing)}")
        return [by_id[w] for w in want]
    if args.indices:
        want_idx = [int(x) for x in args.indices.split(",") if x.strip()]
        by_idx = {i: (i, r) for i, r in rows}
        missing_i = [w for w in want_idx if w not in by_idx]
        if missing_i:
            raise SystemExit(f"indices not present (after --kind filter): {missing_i}")
        return [by_idx[w] for w in want_idx]
    if args.sample:
        rnd = random.Random(args.seed)
        return rnd.sample(rows, min(args.sample, len(rows)))
    return rows[: args.n]


# --------------------------------------------------------------------------- #
#  Terminal output
# --------------------------------------------------------------------------- #
_C = {"hdr": "\033[1;36m", "key": "\033[1m", "base": "\033[2m",
      "bad": "\033[1;31m", "ok": "\033[32m", "off": "\033[0m"}


def _c(tag: str, s: str, color: bool) -> str:
    return f"{_C[tag]}{s}{_C['off']}" if color else s


def _wrap(text: str, width: int = 100, indent: str = "    ") -> str:
    import textwrap
    if not text:
        return indent + "(empty)"
    return "\n".join(textwrap.fill(p, width=width, initial_indent=indent,
                                   subsequent_indent=indent)
                     for p in text.split("\n"))


def print_instance(res: dict, color: bool, show_facts: bool) -> None:
    print()
    print(_c("hdr", "=" * 100, color))
    print(_c("hdr", f"[{res['index']}] {res['id']}   kind={res['scaffold_kind']}"
                    f"   verified={res.get('scaffold_verified')}", color))
    print(_c("hdr", "=" * 100, color))
    print(_c("key", "  SMILES        ", color) + res["smiles"])
    print(_c("key", "  scaffold      ", color) + (res.get("scaffold_smiles") or "—"))
    if res.get("analysis_error"):
        print(_c("bad", f"  ANALYSIS ERROR: {res['analysis_error']}", color))
        return
    if show_facts:
        print(_c("key", "\n  FACTS", color))
        print(_wrap(res["facts"], indent="    "))
    print(_c("key", "\n  DRAFT (deterministic)", color))
    print(_wrap(res["draft"]))
    print(_c("key", "\n  BASELINE (description already in the file)", color))
    print(_c("base", _wrap(res.get("baseline_description") or ""), color))
    for gen in res["generations"]:
        label = gen["label"]
        if gen.get("skipped"):
            print(_c("key", f"\n  {label}", color) + _c("base", f"  — {gen['skipped']}", color))
            continue
        if gen.get("error"):
            print(_c("key", f"\n  {label}", color))
            print(_c("bad", f"    ERROR: {gen['error']}", color))
            continue
        meta = (f"  [{gen['seconds']}s, {gen['n_sentences']} sent, "
                f"{gen['n_words']} words, revise x{gen['n_revise']}]")
        print(_c("key", f"\n  {label}", color) + _c("base", meta, color))
        print(_wrap(gen["text"]))
        if gen["violations"]:
            for v in gen["violations"]:
                print(_c("bad", f"    ! {v}", color))
        else:
            print(_c("ok", "    ✓ no violations", color))


def print_summary(results: list[dict], labels: list[str], color: bool) -> None:
    print()
    print(_c("hdr", "=" * 100, color))
    print(_c("hdr", "SUMMARY", color))
    print(_c("hdr", "=" * 100, color))
    hdr = f"  {'variant':<28} {'n':>4} {'viol':>6} {'sent':>7} {'words':>7} {'sec':>7} {'revise':>7}"
    print(_c("key", hdr, color))
    for lb in labels:
        gens = [g for r in results for g in r["generations"]
                if g["label"] == lb and not g.get("skipped") and not g.get("error")]
        if not gens:
            print(f"  {lb:<28} {'—':>4} {'(no successful generations)':>40}")
            continue
        n = len(gens)
        nv = sum(1 for g in gens if g["violations"])
        avg = lambda k: sum(g[k] for g in gens) / n  # noqa: E731
        print(f"  {lb:<28} {n:>4} {nv:>6} {avg('n_sentences'):>7.1f} "
              f"{avg('n_words'):>7.1f} {avg('seconds'):>7.2f} {avg('n_revise'):>7.2f}")
    print("\n  viol = instances with surviving faithfulness violations (lower is better)")


# --------------------------------------------------------------------------- #
#  HTML
# --------------------------------------------------------------------------- #
_HTML_CSS = """
body{font:14px/1.55 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:24px;
     background:#fafafa;color:#1a1a1a}
h1{font-size:20px;margin:0 0 4px}.sub{color:#666;font-size:13px;margin-bottom:20px}
.inst{background:#fff;border:1px solid #e2e2e2;border-radius:8px;padding:16px;margin-bottom:20px}
.ihdr{font-weight:600;font-size:15px;margin-bottom:8px}
.meta{color:#666;font-size:12px;font-family:ui-monospace,Menlo,monospace;
      word-break:break-all;margin-bottom:10px}
.row{display:flex;gap:16px;align-items:flex-start;flex-wrap:wrap}
.mol{flex:0 0 auto;border:1px solid #eee;border-radius:6px;background:#fff;padding:4px}
.cols{flex:1 1 460px;min-width:300px;display:flex;flex-direction:column;gap:10px}
.card{border:1px solid #e6e6e6;border-radius:6px;padding:10px 12px;background:#fff}
.card.base{background:#f6f6f6;color:#555}
.card.draft{background:#f9f9f4}
.lbl{font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.04em;
     color:#555;margin-bottom:5px}
.lbl .m{font-weight:400;text-transform:none;letter-spacing:0;color:#888;margin-left:8px}
.viol{color:#b00020;font-size:12.5px;margin-top:6px}
.okv{color:#0a7a29;font-size:12.5px;margin-top:6px}
.facts{white-space:pre-wrap;font-family:ui-monospace,Menlo,monospace;font-size:11.5px;
       color:#444;background:#f7f7f7;border-radius:6px;padding:8px 10px;margin-top:10px}
table{border-collapse:collapse;margin:10px 0 24px;background:#fff}
th,td{border:1px solid #ddd;padding:6px 12px;text-align:right;font-size:13px}
th:first-child,td:first-child{text-align:left}
th{background:#f0f0f0}
details summary{cursor:pointer;color:#666;font-size:12px;margin-top:8px}
@media(prefers-color-scheme:dark){
  body{background:#141414;color:#e8e8e8}.inst,.card,.mol,table{background:#1e1e1e;border-color:#333}
  .card.base{background:#242424;color:#aaa}.card.draft{background:#232318}
  .facts{background:#232323;color:#bbb}th{background:#272727}
  .lbl{color:#bbb}.meta,.sub{color:#999}.viol{color:#ff7a8a}.okv{color:#5fd07f}
  .mol svg{background:#fff;border-radius:4px}
}
"""


def _esc(s: Any) -> str:
    return _html.escape(str(s if s is not None else ""))


def render_html(results: list[dict], labels: list[str], meta: dict) -> str:
    try:
        from evaluate_baselines.viz.render_valdump_html import mol_svg
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"--html needs evaluate_baselines/viz/render_valdump_html.py: {exc}")

    out = ["<title>Scaffold description comparison</title>",
           f"<style>{_HTML_CSS}</style>",
           "<h1>Scaffold description comparison</h1>",
           f"<div class='sub'>{_esc(meta['input'])} — {len(results)} instances — "
           f"{_esc(', '.join(labels))}</div>"]

    # summary table
    out.append("<table><tr><th>variant</th><th>n</th><th>with violations</th>"
               "<th>avg sent</th><th>avg words</th><th>avg sec</th></tr>")
    for lb in labels:
        gens = [g for r in results for g in r["generations"]
                if g["label"] == lb and not g.get("skipped") and not g.get("error")]
        if not gens:
            out.append(f"<tr><td>{_esc(lb)}</td><td colspan='5'>no successful generations</td></tr>")
            continue
        n = len(gens)
        nv = sum(1 for g in gens if g["violations"])
        f = lambda k: sum(g[k] for g in gens) / n  # noqa: E731
        out.append(f"<tr><td>{_esc(lb)}</td><td>{n}</td><td>{nv}</td>"
                   f"<td>{f('n_sentences'):.1f}</td><td>{f('n_words'):.1f}</td>"
                   f"<td>{f('seconds'):.2f}</td></tr>")
    out.append("</table>")

    for r in results:
        svg, matched = mol_svg(r["smiles"], query=r.get("scaffold_smarts") or "",
                               w=300, h=230)
        out.append("<div class='inst'>")
        out.append(f"<div class='ihdr'>[{r['index']}] {_esc(r['id'])} "
                   f"<span class='m'>kind={_esc(r['scaffold_kind'])}, "
                   f"scaffold match={_esc(matched)}</span></div>")
        out.append(f"<div class='meta'>{_esc(r['smiles'])}<br>"
                   f"scaffold: {_esc(r.get('scaffold_smiles'))}</div>")
        out.append("<div class='row'>")
        out.append(f"<div class='mol'>{svg}</div>")
        out.append("<div class='cols'>")
        if r.get("analysis_error"):
            out.append(f"<div class='card'><div class='viol'>ANALYSIS ERROR: "
                       f"{_esc(r['analysis_error'])}</div></div>")
        else:
            out.append("<div class='card draft'><div class='lbl'>draft (deterministic)</div>"
                       f"{_esc(r['draft'])}</div>")
            out.append("<div class='card base'><div class='lbl'>baseline (in file)</div>"
                       f"{_esc(r.get('baseline_description'))}</div>")
            for g in r["generations"]:
                if g.get("skipped"):
                    out.append(f"<div class='card base'><div class='lbl'>{_esc(g['label'])}</div>"
                               f"{_esc(g['skipped'])}</div>")
                    continue
                if g.get("error"):
                    out.append(f"<div class='card'><div class='lbl'>{_esc(g['label'])}</div>"
                               f"<div class='viol'>ERROR: {_esc(g['error'])}</div></div>")
                    continue
                m = (f"{g['seconds']}s · {g['n_sentences']} sent · {g['n_words']} words "
                     f"· revise ×{g['n_revise']}")
                out.append(f"<div class='card'><div class='lbl'>{_esc(g['label'])}"
                           f"<span class='m'>{_esc(m)}</span></div>{_esc(g['text'])}")
                if g["violations"]:
                    for v in g["violations"]:
                        out.append(f"<div class='viol'>! {_esc(v)}</div>")
                else:
                    out.append("<div class='okv'>✓ no violations</div>")
                out.append("</div>")
        out.append("</div></div>")
        if r.get("facts"):
            out.append("<details><summary>FACTS</summary>"
                       f"<div class='facts'>{_esc(r['facts'])}</div></details>")
        out.append("</div>")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
#  main
# --------------------------------------------------------------------------- #
def parse_endpoints(specs: list[str]) -> list[tuple[str, list[str]]]:
    """'NAME=URL' or 'NAME=URL1,URL2' -> (name, [urls]).

    Several comma-separated URLs round-robin requests across those servers within **one
    comparison column** — for throughput when the same model is served on several GPUs.
    Passing --endpoint several times instead adds that many comparison columns, which is how
    different models are compared.
    """
    eps = []
    for s in specs:
        if "=" in s:
            name, url = s.split("=", 1)
        else:
            name, url = s, s
        urls = [u.strip() for u in url.split(",") if u.strip()]
        if not urls:
            raise SystemExit(f"--endpoint has no URL: {s}")
        eps.append((name.strip(), urls))
    return eps


def parse_extra_bodies(specs: Optional[list[str]]) -> dict[str, dict]:
    """--extra-body 'NAME={"reasoning_effort":"low"}' → {NAME: dict}."""
    out: dict[str, dict] = {}
    for s in specs or []:
        if "=" not in s:
            raise SystemExit(f"--extra-body needs NAME=JSON, got: {s}")
        name, blob = s.split("=", 1)
        try:
            val = json.loads(blob)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"--extra-body {name}: invalid JSON ({exc})")
        if not isinstance(val, dict):
            raise SystemExit(f"--extra-body {name}: must be a JSON object")
        out[name.strip()] = val
    return out


async def run(args: argparse.Namespace) -> int:
    records = select_records(args.input, args)
    if not records:
        raise SystemExit("no records selected")
    print(f"selected {len(records)} instance(s) from {args.input}")

    # --- prompt variants ---
    if args.prompts:
        # The production prompt is always the baseline. Each --prompts file adds a comparison
        # column; it does not replace v8.
        prompt_sets = [PromptSet("v8")] + [load_prompt_set(p) for p in args.prompts]
        for ps in prompt_sets[1:]:
            ov = ", ".join(ps.overridden) or "(nothing overridden — falls back to defaults)"
            print(f"  prompts '{ps.name}': overrides {ov}")
            if ps.FACTS_REWRITES:
                for pat, repl in ps.FACTS_REWRITES:
                    print(f"    FACTS {'drop' if repl is None else 'rewrite'}: {pat}"
                          + (f"  ->  {repl}" if repl is not None else ""))
    else:
        prompt_sets = [PromptSet("v8")]

    # --- deterministic section: always runs, with or without an LLM ---
    results: list[dict] = []
    for idx, rec in records:
        smi = rec.get(args.smiles_key) or ""
        an, item = analyze_one(smi)
        res: dict = {
            "index": idx,
            "id": rec.get("id"),
            "smiles": smi,
            "scaffold_kind": (item or {}).get("scaffold_kind") or rec.get("scaffold_kind"),
            "scaffold_smiles": (item or {}).get("scaffold_smiles") or rec.get("scaffold_smiles"),
            "scaffold_smarts": (item or {}).get("scaffold_smarts") or rec.get("scaffold_smarts"),
            "scaffold_verified": (item or {}).get("scaffold_verified"),
            "baseline_description": rec.get("description"),
            "baseline_draft": rec.get("template_draft"),
            "generations": [],
            "_an": an,
        }
        if an is None or (item or {}).get("analysis_error"):
            res["analysis_error"] = (item or {}).get("analysis_error", "analysis failed")
            res["facts"] = ""
            res["draft"] = ""
        else:
            res["facts"] = format_facts(an)
            res["draft"] = item.get("template_draft") or render_template(an)
        results.append(res)

    labels: list[str] = []
    if args.dry_run:
        print("  [dry-run] no model calls")
    else:
        # --- detect the model per endpoint, once, rather than per combination ---
        live: list[tuple[str, list[str], str]] = []   # (name, [live urls], model)
        for name, urls in parse_endpoints(args.endpoint):
            probe = VLLMPool(urls, timeout=args.timeout, model=args.model)
            await probe.discover_models()
            if args.model:
                urls_live, models = list(urls), [args.model] * len(urls)
            else:
                probe.prune_dead()
                urls_live, models = list(probe.base_urls), probe.detected_models
            if not urls_live:
                print(f"  [warn] endpoint '{name}' ({', '.join(urls)}) not responding — skipping",
                      file=sys.stderr)
                continue
            dead = len(urls) - len(urls_live)
            # Mixing different models into one column makes the comparison meaningless, so
            # this is not passed over silently.
            if len(set(models)) > 1:
                raise SystemExit(
                    f"endpoint '{name}' serves more than one model: {sorted(set(models))}. "
                    "Comma-joined URLs are for load-balancing ONE model across servers; use "
                    "separate --endpoint flags to compare different models.")
            print(f"  endpoint '{name}': {len(urls_live)} server(s) "
                  f"[{', '.join(urls_live)}]  model={models[0]}"
                  + (f"  ({dead} not responding)" if dead else ""))
            live.append((name, urls_live, models[0]))
        if not live:
            raise SystemExit("no live endpoints — start a vLLM server first")

        # --- one dedicated pool per (endpoint x variant) combination ---
        # Sharing a pool and swapping pool._describer just before each call races under
        # concurrency. Building a separate pool per combination and injecting once, at
        # construction, removes that race. The in-flight ceiling becomes
        # (number of combinations x --concurrency).
        extra_bodies = parse_extra_bodies(args.extra_body)
        combos: list[tuple[str, VLLMPool]] = []
        bare = len(prompt_sets) == 1 and prompt_sets[0].name == "v8"
        for name, urls, model in live:
            eb = extra_bodies.get(name)
            cls = FixedExtraBodyPool if eb is not None else VLLMPool
            for ps in prompt_sets:
                pool = cls(urls, per_server_concurrency=args.concurrency,
                           temperature=args.temperature, max_tokens=args.max_tokens,
                           max_sentences=args.max_sentences,
                           reasoning_effort=args.reasoning_effort,
                           timeout=args.timeout, model=model)
                if eb is not None:
                    pool._fixed = eb  # noqa: SLF001
                pool._describer = OverrideDescriber(ps)  # noqa: SLF001
                combos.append((name if bare else f"{name} / {ps.name}", pool))
            if eb is not None:
                print(f"  endpoint '{name}': extra_body forced to {json.dumps(eb)}")
        labels = [c[0] for c in combos]

        # --- run every (instance x combination) concurrently, throttled by a semaphore ---
        # Awaiting them in sequence would waste the server's continuous batching entirely
        # (measured: 2.95x slower).
        for res in results:
            res["generations"] = [None] * len(combos)

        async def one(res: dict, slot: int, label: str, pool: VLLMPool,
                      server_idx: int) -> None:
            if res.get("analysis_error"):
                res["generations"][slot] = {"label": label, "skipped": "analysis failed"}
                return
            if res["scaffold_kind"] != "ring":
                # For a non-ring scaffold the pipeline does not call the LLM: the
                # deterministic draft becomes the description (see describe_one). Mirrored
                # here.
                res["generations"][slot] = {
                    "label": label,
                    "skipped": f"kind={res['scaffold_kind']} — pipeline uses the "
                               "deterministic draft, no LLM"}
                return
            try:
                g = await polish_with_validation(pool, res["_an"], res["draft"],
                                                 args.validate_passes, server_idx,
                                                 normalize=args.normalize)
                g.update({"label": label,
                          "n_sentences": len(_split_sentences(g["text"])),
                          "n_words": len(g["text"].split())})
            except Exception as exc:  # noqa: BLE001
                g = {"label": label, "error": str(exc)}
            res["generations"][slot] = g

        n_calls = sum(1 for r in results if r["scaffold_kind"] == "ring"
                      and not r.get("analysis_error")) * len(combos)
        n_flight = sum(p.n_servers * args.concurrency for _, p in combos)
        print(f"  running {n_calls} generation(s), up to {n_flight} in flight...")
        t0 = time.monotonic()
        # server_idx = the instance's position, which round-robins servers inside the pool.
        await asyncio.gather(*(one(res, slot, label, pool, ridx)
                               for ridx, res in enumerate(results)
                               for slot, (label, pool) in enumerate(combos)))
        print(f"  done in {time.monotonic() - t0:.1f}s")
        for res in results:   # guard against empty slots left by an unexpected cancellation
            res["generations"] = [g if g is not None else
                                  {"label": labels[i], "error": "no result"}
                                  for i, g in enumerate(res["generations"])]
            print_instance(res, args.color, args.facts)

    if args.dry_run:
        for res in results:
            print_instance(res, args.color, args.facts)
    elif labels:
        print_summary(results, labels, args.color)

    # --- write out ---
    for res in results:
        res.pop("_an", None)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            for res in results:
                fh.write(json.dumps(res, ensure_ascii=False) + "\n")
        print(f"\nwrote {args.out}")
    if args.html:
        with open(args.html, "w", encoding="utf-8") as fh:
            fh.write(render_html(results, labels, {"input": args.input}))
        print(f"wrote {args.html}")
    return 0


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Regenerate scaffold descriptions for a few instances and compare "
                    "models / prompt variants side by side.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--input", default=DEFAULT_INPUT, help="instances JSONL")
    p.add_argument("--smiles-key", default="ref_smiles")

    g = p.add_argument_group("instance selection")
    g.add_argument("--n", type=int, default=5, help="first N records")
    g.add_argument("--ids", default="", help="comma-separated record ids")
    g.add_argument("--indices", default="", help="comma-separated 0-based line indices")
    g.add_argument("--sample", type=int, default=0, help="random sample of this size")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--kind", default="ring",
                   choices=["ring", "functional_group", "none", "all"])

    g = p.add_argument_group("models / prompts")
    g.add_argument("--endpoint", action="append", default=None,
                   metavar="NAME=URL",
                   help=f"repeatable; default: {' '.join(DEFAULT_ENDPOINTS)}")
    g.add_argument("--prompts", action="append", default=None, metavar="FILE.py",
                   help="repeatable prompt-override file compared with production v8; "
                        "undefined names fall back to scaffold_prompts.py")
    g.add_argument("--model", default=None, help="force a model name (skip auto-discovery)")
    g.add_argument("--validate-passes", type=int, default=3)
    g.add_argument("--temperature", type=float, default=0.4)
    g.add_argument("--max-tokens", type=int, default=1024)
    g.add_argument("--max-sentences", type=int, default=0, help="0 = no cap")
    g.add_argument("--reasoning-effort", default="low",
                   choices=["none", "low", "medium", "high"],
                   help="gpt-oss -> reasoning_effort; qwen -> enable_thinking (on at "
                        "medium/high). NOTE 'none' is asymmetric: qwen thinking goes off but "
                        "gpt-oss gets no parameter at all and falls back to its own default")
    g.add_argument("--extra-body", action="append", default=None, metavar="NAME=JSON",
                   help="force the extra_body for an endpoint, bypassing the model-family "
                        "guess; needed for families VLLMPool does not know (deepseek, "
                        "nemotron, ...). e.g. --extra-body deepseek='{\"chat_template_kwargs\":"
                        "{\"thinking\":false}}'")
    g.add_argument("--concurrency", type=int, default=8,
                   help="in-flight requests per (endpoint x prompt-variant); vLLM batches "
                        "these server-side, so raising it is the way to go faster")
    g.add_argument("--timeout", type=float, default=180.0)

    g = p.add_argument_group("output")
    g.add_argument("--dry-run", action="store_true",
                   help="no model calls — show FACTS / draft / baseline only")
    g.add_argument("--facts", action="store_true", help="print the FACTS block")
    g.add_argument("--out", default="", help="write results JSONL here")
    g.add_argument("--html", default="", help="write a side-by-side HTML page here")
    g.add_argument("--no-color", dest="color", action="store_false", default=True)
    g.add_argument("--no-normalize", dest="normalize", action="store_false", default=True,
                   help="skip Unicode normalisation of the generated text (fancy dashes and "
                        "subscripts are a per-model fingerprint; folded to ASCII by default)")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    if args.endpoint is None:
        args.endpoint = list(DEFAULT_ENDPOINTS)
    try:
        raise SystemExit(asyncio.run(run(args)))
    except KeyboardInterrupt:
        raise SystemExit(130)


if __name__ == "__main__":
    main()
