#!/usr/bin/env python
"""HTML visualiser for constructive substructure-generation SFT data.

Renders the JSONL produced by ``4_sftdata_gen`` (the direct-scaffold-seed
flow: the seed reasoning derives the substructure SMARTS and writes the complete
scaffold SMILES, then a 3-way ``match_substructure`` ∥ ``analyze_properties``
∥ ``label_atom_indices`` checkpoint runs after the seed and after every edit) as a
self-contained HTML report.

Records are shown one per card, keyed by ``metadata.group_id``: a record is one
chain's whole conversation (seed + each edit + the terminal ANSWER).  For each
record the report renders:

  * each assistant turn — reasoning text + the tool call(s) it issues (a parallel
    checkpoint is labelled and its calls grouped), with the relevant molecule
    drawn as inline SVG (the required substructure is highlighted on every
    molecule for which a ``match_substructure`` query is available),
  * each tool response (SMILES / labelled SMILES drawn; property JSON as a table),
  * the closing ``<ANSWER>``.

Only ``rdkit`` and the standard library are required.

Usage
-----
    python 4_sftdata_gen/view_constructive_sftdata.py \
        --input  data/training_data/sftdata/generation_200k_scaffold_ref/toolchains_generation_chunk_0000.jsonl \
        --out    data/training_data/sftdata/generation_200k_scaffold_ref/view.html \
        --limit  50 --offset 0 \
        --group  generation_9153__0000042   # render only this group_id
"""

from __future__ import annotations

import argparse
import html
import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Optional

try:
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    _RDKIT = True
except Exception:  # pragma: no cover - rdkit should be present
    _RDKIT = False


# ── molecule rendering ──────────────────────────────────────────────────────

_svg_cache: dict[tuple, str] = {}


def _strip_maps(smiles: str) -> str:
    """Drop atom-map numbers (``[c:11]`` → ``c``) so labelled SMILES still draw."""
    return re.sub(r":\d+\]", "]", smiles)


def mol_svg(
    smiles: Optional[str],
    highlight_query: Optional[str] = None,
    size: int = 250,
) -> str:
    """Inline SVG for *smiles*; highlight the *highlight_query* match if given."""
    if not smiles or not _RDKIT:
        return '<div class="noimg">—</div>'
    key = (smiles, highlight_query, size)
    if key in _svg_cache:
        return _svg_cache[key]

    mol = Chem.MolFromSmiles(smiles) or Chem.MolFromSmiles(_strip_maps(smiles))
    if mol is None:
        out = f'<div class="noimg">unparseable<br>{html.escape(smiles[:40])}</div>'
        _svg_cache[key] = out
        return out

    hl_atoms: list[int] = []
    hl_bonds: list[int] = []
    if highlight_query:
        patt = Chem.MolFromSmarts(highlight_query)
        if patt is None:
            patt = Chem.MolFromSmiles(highlight_query)
        if patt is not None:
            match = mol.GetSubstructMatch(patt)
            if match:
                hl_atoms = list(match)
                amap = {patt_i: mol_i for patt_i, mol_i in enumerate(match)}
                for b in patt.GetBonds():
                    a1 = amap.get(b.GetBeginAtomIdx())
                    a2 = amap.get(b.GetEndAtomIdx())
                    if a1 is None or a2 is None:
                        continue
                    bd = mol.GetBondBetweenAtoms(a1, a2)
                    if bd is not None:
                        hl_bonds.append(bd.GetIdx())

    d = rdMolDraw2D.MolDraw2DSVG(size, int(size * 0.8))
    d.drawOptions().addStereoAnnotation = True
    rdMolDraw2D.PrepareAndDrawMolecule(
        d, mol, highlightAtoms=hl_atoms, highlightBonds=hl_bonds
    )
    d.FinishDrawing()
    svg = d.GetDrawingText()
    svg = re.sub(r"<\?xml[^>]*\?>\n?", "", svg)
    _svg_cache[key] = svg
    return svg


# ── small helpers ────────────────────────────────────────────────────────────

_SMILES_CHARS = set("CNOPSFIBrClHcnops()[]=#+-/\\@.1234567890%*")


def _looks_like_smiles(text: str) -> bool:
    t = (text or "").strip()
    if not t or len(t) > 300 or "\n" in t or " " in t.strip():
        return False
    if t.startswith("{") or t.startswith("["):
        # could be labelled SMILES "[O:0]=..." — allow, else JSON-ish reject
        if not re.match(r"\[[A-Za-z]", t):
            return False
    if not _RDKIT:
        return False
    return Chem.MolFromSmiles(t) is not None or Chem.MolFromSmiles(_strip_maps(t)) is not None


def _esc(s: str) -> str:
    return html.escape(s or "")


def _fmt_args(args: dict) -> str:
    """Compact one-line rendering of tool-call arguments, eliding long SMILES."""
    parts = []
    for k, v in args.items():
        sv = json.dumps(v) if not isinstance(v, str) else v
        if isinstance(v, str) and len(sv) > 60:
            sv = sv[:57] + "…"
        parts.append(f"{_esc(k)}=<b>{_esc(sv)}</b>")
    return ", ".join(parts)


def _annotate_verification(text: str) -> str:
    """Colour ✓ / ✗ lines and the N/M satisfied tally inside a Verification block."""
    out_lines = []
    for line in text.splitlines():
        cls = ""
        if line.rstrip().endswith("✓"):
            cls = "ok"
        elif line.rstrip().endswith("✗"):
            cls = "bad"
        elif re.search(r"\b\d+/\d+ satisfied\.", line):
            m = re.search(r"(\d+)/(\d+) satisfied", line)
            cls = "ok" if m and m.group(1) == m.group(2) else "warn"
        out_lines.append(f'<span class="{cls}">{_esc(line)}</span>' if cls else _esc(line))
    return "\n".join(out_lines)


def _prop_table(obj: dict) -> str:
    rows = "".join(
        f"<tr><td>{_esc(str(k))}</td><td>{_esc(str(v))}</td></tr>"
        for k, v in obj.items()
    )
    return f'<table class="props">{rows}</table>'


# ── per-record rendering ─────────────────────────────────────────────────────


def _smarts_for_record(messages: list[dict]) -> Optional[str]:
    """Find a match_substructure query (SMARTS) anywhere in the record."""
    for m in messages:
        for tc in m.get("tool_calls", []) or []:
            fn = tc.get("function", {})
            if fn.get("name") == "match_substructure":
                try:
                    args = json.loads(fn.get("arguments", "{}"))
                except json.JSONDecodeError:
                    continue
                q = args.get("query")
                if q:
                    return q
    return None


def _render_segment(record: dict, smarts: Optional[str]) -> str:
    meta = record.get("metadata", {})
    messages = record.get("messages", [])
    seg_idx = meta.get("segment_index", "?")
    seg_total = meta.get("segment_total", "?")
    etype = meta.get("example_type", "?")
    badge_cls = "badge-final" if etype == "normal" else "badge-mid"

    blocks: list[str] = [
        f'<div class="seg">'
        f'<div class="seg-head"><span class="badge {badge_cls}">{_esc(etype)}</span>'
        f'<span class="seg-pos">segment {seg_idx} / {seg_total}</span>'
        f'<span class="ntools">{meta.get("num_tool_calls", 0)} tool call(s)</span></div>'
    ]

    # Pending tool-call context: map nothing — we render tool responses inline
    # right after their assistant call by walking sequentially.
    last_calls: list[dict] = []
    for msg in messages:
        role = msg.get("role")
        if role == "system":
            continue
        if role == "user":
            continue  # shown once at the group level
        if role == "assistant":
            content = msg.get("content", "") or ""
            calls = msg.get("tool_calls") or []
            last_calls = calls
            body = _render_assistant_content(content, smarts)
            call_html = _render_calls(calls, smarts)
            blocks.append(
                '<div class="turn assistant"><div class="turn-label">assistant</div>'
                f"{body}{call_html}</div>"
            )
            continue
        if role == "tool":
            blocks.append(_render_tool_response(msg.get("content", ""), last_calls, smarts))
            continue

    blocks.append("</div>")
    return "".join(blocks)


def _render_assistant_content(content: str, smarts: Optional[str]) -> str:
    """Render assistant prose; pull out Verification / ANSWER blocks."""
    if not content.strip():
        return ""
    out: list[str] = []

    # <ANSWER>
    ans = re.search(r"<ANSWER>\s*(.*?)\s*</ANSWER>", content, re.DOTALL)
    answer_smi = ans.group(1).strip() if ans else None
    # split off the verification + answer tail from the reasoning prose
    reasoning = content
    for tag in ("Verification:", "<ANSWER>"):
        i = reasoning.find(tag)
        if i != -1:
            reasoning = reasoning[:i]
            break
    reasoning = reasoning.strip()
    if reasoning:
        out.append(f'<div class="reasoning">{_esc(reasoning)}</div>')

    ver = re.search(r"(Verification:.*?)(?:\n\n<|$)", content, re.DOTALL)
    if ver:
        out.append(
            f'<div class="verif"><pre>{_annotate_verification(ver.group(1).strip())}</pre></div>'
        )

    if answer_smi:
        out.append(
            '<div class="answer"><div class="turn-label">&lt;ANSWER&gt;</div>'
            f'<div class="molrow"><div class="mol">{mol_svg(answer_smi, smarts)}</div>'
            f'<code class="smi">{_esc(answer_smi)}</code></div>'
            '<div class="hint">required substructure highlighted</div></div>'
        )
    return "".join(out)


def _render_calls(calls: list[dict], smarts: Optional[str]) -> str:
    if not calls:
        return ""
    items: list[str] = []
    # A multi-call assistant turn IS the parallel verification checkpoint
    # (match ∥ analyze ∥ label). Label it so the three calls read as one unit.
    if len(calls) > 1:
        names = ", ".join(tc.get("function", {}).get("name", "?") for tc in calls)
        items.append(f'<div class="parhead">∥ parallel checkpoint — {_esc(names)}</div>')
    for tc in calls:
        fn = tc.get("function", {})
        name = fn.get("name", "?")
        try:
            args = json.loads(fn.get("arguments", "{}"))
        except json.JSONDecodeError:
            args = {}
        items.append(
            f'<div class="call"><code class="callname">→ {_esc(name)}</code>'
            f'<span class="callargs">({_fmt_args(args)})</span></div>'
        )
        # Draw the molecule a match_substructure / set_stereochemistry acts on.
        mol_smi = args.get("mol_smiles")
        if name == "match_substructure" and mol_smi:
            items.append(
                f'<div class="molrow"><div class="mol">{mol_svg(mol_smi, smarts)}</div>'
                '<div class="hint">substructure query target</div></div>'
            )
    return '<div class="calls">' + "".join(items) + "</div>"


def _render_tool_response(content: str, last_calls: list[dict], smarts: Optional[str]) -> str:
    content = (content or "").strip()
    # JSON response (match result / property dict).
    try:
        obj = json.loads(content)
    except json.JSONDecodeError:
        obj = None

    if isinstance(obj, dict) and "match" in obj:
        ok = obj.get("match") is True
        cls = "ok" if ok else "bad"
        return (
            '<div class="turn tool"><div class="turn-label">tool · match_substructure</div>'
            f'<div class="matchres {cls}">match: {obj.get("match")}</div></div>'
        )
    if isinstance(obj, dict):
        return (
            '<div class="turn tool"><div class="turn-label">tool · properties</div>'
            f"{_prop_table(obj)}</div>"
        )

    # SMILES response (e.g. attach_fragment / set_stereochemistry / label).
    if _looks_like_smiles(content):
        is_labeled = bool(re.search(r"\[[A-Za-z][^\]]*:\d+\]", content))
        label = "labelled atoms" if is_labeled else "result molecule"
        return (
            f'<div class="turn tool"><div class="turn-label">tool · {label}</div>'
            f'<div class="molrow"><div class="mol">{mol_svg(content, None if is_labeled else smarts)}</div>'
            f'<code class="smi">{_esc(content)}</code></div></div>'
        )

    # Fallback: raw text.
    return (
        '<div class="turn tool"><div class="turn-label">tool</div>'
        f'<pre>{_esc(content[:600])}</pre></div>'
    )


# ── grouping + page assembly ─────────────────────────────────────────────────


def _user_prompt(messages: list[dict]) -> str:
    for m in messages:
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


def _render_group(gid: str, records: list[dict]) -> str:
    records = sorted(records, key=lambda r: r.get("metadata", {}).get("segment_index", 0))
    smarts = None
    for r in records:
        smarts = _smarts_for_record(r.get("messages", []))
        if smarts:
            break
    user_prompt = _user_prompt(records[0].get("messages", [])) if records else ""

    # Header: final answer molecule (from the last record that has one).
    final_smi = None
    for r in reversed(records):
        m = re.search(r"<ANSWER>\s*(.*?)\s*</ANSWER>", r["messages"][-1].get("content", ""), re.DOTALL)
        if m:
            final_smi = m.group(1).strip()
            break

    head = [
        f'<div class="group"><h2>{_esc(gid)} '
        f'<span class="seg-count">({len(records)} segment(s))</span></h2>',
        '<div class="grouptop">',
        f'<details class="prompt"><summary>user prompt</summary><pre>{_esc(user_prompt)}</pre></details>',
    ]
    if smarts:
        head.append(f'<div class="smarts">required SMARTS: <code>{_esc(smarts)}</code></div>')
    if final_smi:
        head.append(
            f'<div class="finalmol"><div class="turn-label">final molecule</div>'
            f'<div class="mol">{mol_svg(final_smi, smarts)}</div>'
            f'<code class="smi">{_esc(final_smi)}</code></div>'
        )
    head.append("</div>")

    body = [_render_segment(r, smarts) for r in records]
    return "".join(head) + "".join(body) + "</div>"


_CSS = """
:root{color-scheme:light}
*{box-sizing:border-box}
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#f4f5f7;color:#1d2330}
header{position:sticky;top:0;background:#1864ab;color:#fff;padding:12px 20px;z-index:5}
header h1{margin:0;font-size:17px}
header .meta{font-size:12px;opacity:.85}
.wrap{max-width:1100px;margin:0 auto;padding:18px}
.group{background:#fff;border:1px solid #d8dee4;border-radius:10px;margin:0 0 26px;padding:16px;box-shadow:0 1px 3px rgba(0,0,0,.06)}
.group h2{margin:0 0 10px;font-size:15px;color:#0b3d91;word-break:break-all}
.seg-count{color:#868e96;font-weight:normal;font-size:12px}
.grouptop{display:flex;gap:16px;flex-wrap:wrap;align-items:flex-start;border-bottom:1px dashed #dee2e6;padding-bottom:12px;margin-bottom:12px}
.prompt{flex:1 1 320px;font-size:12px}
.prompt summary,.summary summary{cursor:pointer;color:#1864ab;font-weight:600}
.smarts{flex:1 1 100%;font-size:12px;color:#495057}
.smarts code,.smi{font-family:ui-monospace,Menlo,Consolas,monospace}
.finalmol{flex:0 0 auto;text-align:center}
.mol{display:inline-block;background:#fff;border:1px solid #e9ecef;border-radius:8px;padding:4px}
.mol svg{display:block}
.noimg{width:200px;height:120px;display:flex;align-items:center;justify-content:center;color:#adb5bd;border:1px dashed #ced4da;border-radius:8px;font-size:12px;text-align:center}
.seg{border:1px solid #e9ecef;border-radius:8px;margin:12px 0;overflow:hidden}
.seg-head{display:flex;gap:10px;align-items:center;background:#f1f3f5;padding:7px 12px;font-size:12px}
.badge{color:#fff;border-radius:4px;padding:2px 8px;font-size:11px;font-weight:700}
.badge-final{background:#0a7d2c}.badge-mid{background:#1864ab}
.seg-pos{font-weight:600}.ntools{color:#868e96;margin-left:auto}
.turn{padding:10px 12px;border-top:1px solid #f1f3f5}
.turn-label{font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:#868e96;margin-bottom:5px}
.assistant{background:#fbfdff}
.tool{background:#fcfcf7}
.reasoning{font-size:13.5px;line-height:1.5;white-space:pre-wrap}
.calls{margin-top:8px}
.parhead{font-size:11px;font-weight:700;color:#1864ab;background:#e7f0fb;border-radius:4px;padding:2px 8px;margin:6px 0 4px;display:inline-block}
.call{font-size:12.5px;margin:3px 0}
.callname{color:#a61e4d;font-weight:700}
.callargs{color:#495057;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:11.5px;word-break:break-all}
.molrow{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-top:6px}
.smi{font-size:11.5px;color:#343a40;word-break:break-all;background:#f1f3f5;padding:2px 6px;border-radius:4px}
.hint{font-size:11px;color:#868e96}
pre{margin:0;white-space:pre-wrap;word-break:break-word;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:11.5px;line-height:1.45}
.verif pre{background:#f8f9fa;border:1px solid #e9ecef;border-radius:6px;padding:8px}
.verif .ok{color:#0a7d2c}.verif .bad{color:#c92a2a}.verif .warn{color:#b8860b}
.summary{margin-top:8px;font-size:12px}
.summary pre{background:#f8f5ff;border:1px solid #e7dcff;border-radius:6px;padding:8px;margin-top:6px}
.answer{margin-top:10px;border-top:2px solid #0a7d2c;padding-top:8px}
.matchres{font-family:ui-monospace,monospace;font-size:12.5px;font-weight:700;padding:4px 8px;border-radius:5px;display:inline-block}
.matchres.ok{background:#ebfbee;color:#0a7d2c}.matchres.bad{background:#fff0f0;color:#c92a2a}
table.props{border-collapse:collapse;font-size:11.5px}
table.props td{border:1px solid #e9ecef;padding:2px 8px;font-family:ui-monospace,monospace}
table.props td:first-child{color:#495057;font-weight:600}
"""


def build_html(records: list[dict], title: str) -> str:
    groups: "OrderedDict[str, list[dict]]" = OrderedDict()
    for r in records:
        gid = r.get("metadata", {}).get("group_id", "ungrouped")
        groups.setdefault(gid, []).append(r)

    parts = [_render_group(gid, recs) for gid, recs in groups.items()]
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{_esc(title)}</title><style>{_CSS}</style></head><body>"
        f"<header><h1>{_esc(title)}</h1>"
        f"<div class='meta'>{len(groups)} group(s) · {len(records)} record(s)</div></header>"
        f"<div class='wrap'>{''.join(parts)}</div></body></html>"
    )


# ── IO ───────────────────────────────────────────────────────────────────────


def _iter_jsonl_files(path: Path):
    if path.is_dir():
        yield from sorted(path.glob("*.jsonl"))
    else:
        yield path


def load_records(
    input_path: str,
    limit: Optional[int],
    offset: int,
    group: Optional[str],
) -> list[dict]:
    """Load records, optionally filtered to one group_id.

    ``limit``/``offset`` count GROUPS (so whole trajectories stay together),
    not individual segment records.
    """
    by_group: "OrderedDict[str, list[dict]]" = OrderedDict()
    for fp in _iter_jsonl_files(Path(input_path)):
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                gid = rec.get("metadata", {}).get("group_id", "ungrouped")
                if group and gid != group:
                    continue
                by_group.setdefault(gid, []).append(rec)

    gids = list(by_group.keys())
    if offset:
        gids = gids[offset:]
    if limit is not None:
        gids = gids[:limit]
    out: list[dict] = []
    for gid in gids:
        out.extend(by_group[gid])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="JSONL file or directory")
    ap.add_argument("--out", default=None, help="output HTML path (default: <input-stem>.html)")
    ap.add_argument("--limit", type=int, default=50, help="max number of GROUPS to render")
    ap.add_argument("--offset", type=int, default=0, help="skip this many groups")
    ap.add_argument("--group", default=None, help="render only this group_id")
    args = ap.parse_args()

    if not _RDKIT:
        raise SystemExit("rdkit is required (conda env molkit).")

    records = load_records(args.input, args.limit, args.offset, args.group)
    if not records:
        raise SystemExit(f"No records found in {args.input}")

    in_path = Path(args.input)
    out = args.out or str((in_path if in_path.is_file() else in_path).with_suffix(".html"))
    if in_path.is_dir():
        out = args.out or str(in_path / "view_constructive.html")

    title = f"Constructive SFT — {in_path.name}"
    html_doc = build_html(records, title)
    Path(out).write_text(html_doc, encoding="utf-8")
    n_groups = len({r["metadata"].get("group_id") for r in records})
    print(f"Wrote {out}  ({n_groups} group(s), {len(records)} record(s))")


if __name__ == "__main__":
    main()
