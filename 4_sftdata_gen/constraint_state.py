"""Constraint state read out of a Segment's tool results.

All content is derived **rule-based** from the pre-computed tool results that
live in the ``Round`` objects extracted from the ground-truth tool chain.  No
tool servers are contacted here — every number and the substructure-match flag
originate from the ``expected_response`` fields of the ``ToolChainStep``s.

The generation task is constructive: build a molecule that CONTAINS a required
substructure (described in natural language) while meeting numeric property
targets.  This module answers, for one round, (a) whether the required
substructure currently matches and (b) each target property against its
measured value — which is what decides when a chain is finished and what the
final Verification block prints.
"""

from __future__ import annotations

import re
from typing import Optional

from .segmenter import Round, Segment

# ── value formatting ───────────────────────────────────────────────────────

# Integer-valued props.
_INT_LIKE = {
    "HBD", "HBA", "rotB", "rings_total", "heavy_atoms", "formal_charge",
}


def _fmt_val(name: str, val) -> str:
    """Format a property value, preserving the precision from tool output."""
    if isinstance(val, bool):
        return str(val)
    if not isinstance(val, (int, float)):
        return str(val)
    if name in _INT_LIKE:
        return str(int(round(val)))
    return f"{val:.3f}"


def _fmt_delta(name: str, before, after) -> str:
    if before is None or after is None:
        return ""
    try:
        diff = float(after) - float(before)
    except Exception:
        return ""
    if name in _INT_LIKE:
        d_i = int(round(diff))
        if d_i == 0:
            return f"{name} 0 ({_fmt_val(name, before)}→{_fmt_val(name, after)})"
        sign = "+" if d_i > 0 else ""
        return f"{name} {sign}{d_i} ({_fmt_val(name, before)}→{_fmt_val(name, after)})"
    if abs(diff) < 0.001:
        return f"{name} 0.000 ({_fmt_val(name, before)}→{_fmt_val(name, after)})"
    sign = "+" if diff > 0 else ""
    return f"{name} {sign}{_fmt_val(name, diff)} ({_fmt_val(name, before)}→{_fmt_val(name, after)})"


def _fmt_range(rng, name: Optional[str] = None) -> str:
    def _f(v):
        if name is not None:               # property-aware: ints as int, floats .3f
            return _fmt_val(name, v)
        if isinstance(v, float):
            return f"{v:.3f}"
        return str(v)

    if rng is None:
        return "any"
    if isinstance(rng, (int, float)):
        return f"= {_f(rng)}"
    lo, hi = rng[0], rng[1]
    if lo is None and hi is None:
        return "any"
    if lo is None:
        return f"≤ {_f(hi)}"
    if hi is None:
        return f"≥ {_f(lo)}"
    if lo == hi:
        return _f(lo)          # exact value → just the single number (not "x – x")
    return f"{_f(lo)} – {_f(hi)}"


def _compact_edit_args(tool: str, args: dict) -> str:
    """Short argument rendering for round log lines."""
    if tool == "edit_fragment":
        frm = args.get("from_smiles")
        to = args.get("to_smiles")
        anch = args.get("anchors")
        # attach = lone "[*:1]" from-fragment; remove = "[*:1][H]" to-fragment.
        if frm in ("[*:1]", "*"):
            core = f"attach {to!r}"
        elif to in ("[*:1][H]", "[*:1]", "*"):
            core = f"remove {frm!r}"
        else:
            core = f"{frm!r}→{to!r}"
        return f"{core}, anchors={anch}" if anch else core
    return ", ".join(f"{k}={v}" for k, v in args.items())


# ── satisfaction helpers ─────────────────────────────────────────────────


def _value_in_range(val, rng) -> Optional[bool]:
    """Return True/False if *val* falls in *rng*, or None when undecidable."""
    if val is None:
        return None
    if rng is None:
        return True
    try:
        v = float(val)
    except (TypeError, ValueError):
        return None
    if isinstance(rng, (int, float)):
        return abs(v - float(rng)) < 1e-9
    try:
        lo, hi = rng[0], rng[1]
    except (TypeError, IndexError):
        return None
    if lo is not None and v < float(lo):
        return False
    if hi is not None and v > float(hi):
        return False
    return True


def _prop_ok(current: Round, prop: str, rng) -> Optional[bool]:
    """Decide whether *prop* is satisfied on *current* (strict result preferred)."""
    strict = current.prop_strict.get(prop)
    if strict is not None:
        return bool(strict)
    return _value_in_range(current.properties.get(prop), rng)


def is_fully_satisfied(
    current: Round,
    target_properties: dict,
    require_substructure: bool = False,
) -> bool:
    """Return True iff every property target AND the substructure match hold."""
    if require_substructure and current.substructure_match is not True:
        return False
    for prop, rng in (target_properties or {}).items():
        if not _prop_ok(current, prop, rng):
            return False
    return True


# ── constraint check / target requirements ─────────────────────────────────


def _constraint_lines(
    current: Round,
    target_properties: dict,
    require_substructure: bool,
    substructure_description: str,
) -> tuple[list[str], int, int]:
    """Build the per-constraint check lines + (n_satisfied, n_total)."""
    lines: list[str] = []
    n_total = 0
    n_sat = 0

    if require_substructure:
        n_total += 1
        desc = substructure_description.strip()
        label = "required substructure"
        if desc:
            # Show the substructure requirement VERBATIM (no truncation) so the
            # carried memory states the original structural constraint in full.
            label = f"contains substructure ({desc})"
        if current.substructure_match is True:
            lines.append(f"  - {label} → present → ✓")
            n_sat += 1
        elif current.substructure_match is False:
            lines.append(f"  - {label} → not present → ✗")
        else:
            lines.append(f"  - {label} → not yet checked → ✗")

    for prop, rng in (target_properties or {}).items():
        n_total += 1
        val = current.properties.get(prop)
        ok = _prop_ok(current, prop, rng)
        if val is not None:
            mark = "✓" if ok else "✗"
            lines.append(
                f"  - {prop}: target {_fmt_range(rng, prop)} → current {_fmt_val(prop, val)} → {mark}"
            )
            if ok:
                n_sat += 1
        else:
            lines.append(f"  - {prop}: target {_fmt_range(rng, prop)} → not yet measured → ✗")

    return lines, n_sat, n_total


def build_constraint_check(
    current: Round,
    target_properties: dict,
    require_substructure: bool = False,
    substructure_description: str = "",
) -> str:
    """Build the rule-based ``Verification:`` block emitted before ``<ANSWER>``."""
    lines, n_sat, n_total = _constraint_lines(
        current, target_properties, require_substructure, substructure_description
    )
    if not lines:
        return ""
    out = ["Verification:"]
    out.extend(lines)
    out.append(f"{n_sat}/{n_total} satisfied.")
    return "\n".join(out)


# ── progress / current-candidate blocks ─────────────────────────────────────


# ── public API ─────────────────────────────────────────────────────────────


