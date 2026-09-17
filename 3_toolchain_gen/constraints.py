"""Property-constraint parsing and satisfaction helpers.

All functions here are pure (no I/O, no side effects).  Scoped to the
substructure-generation task: parse the instance ``properties`` list into
``{name: [lo, hi]}`` ranges and evaluate measured values against them with
strict bounds (no tolerance band).
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Benchmark instance parsing
# ---------------------------------------------------------------------------

def extract_properties(instance: dict) -> dict[str, list]:
    """Return ``{property_name: [min_or_None, max_or_None]}`` from an instance.

    Each entry of ``instance["properties"]`` is ``{property, min?, max?}``;
    absent bounds become ``None`` (one-sided constraint).
    """
    props: dict[str, list] = {}
    for p in instance.get("properties", []) or []:
        if not isinstance(p, dict) or "property" not in p:
            continue
        props[p["property"]] = [p.get("min"), p.get("max")]
    return props


# ---------------------------------------------------------------------------
# Strict accept criterion
# ---------------------------------------------------------------------------
# A molecule satisfies a property constraint iff each measured value lies inside
# the closed target interval ``[lo, hi]`` exactly — no tolerance band.


def check_property(
    val: float | None,
    lo: float | None,
    hi: float | None,
) -> bool:
    """Whether *val* lies inside the closed interval ``[lo, hi]``.

    Either bound may be ``None`` (one-sided constraint). A ``None`` or
    unparseable *val* fails.
    """
    if val is None:
        return False
    try:
        val = float(val)
    except (TypeError, ValueError):
        return False
    lo_ok = (lo is None) or (lo <= val)
    hi_ok = (hi is None) or (val <= hi)
    return bool(lo_ok and hi_ok)
