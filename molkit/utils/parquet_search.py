"""molkit/utils/parquet_search.py  —  Partitioned parquet molecule search

Functions
---------
search_partitioned_parquet : query a parquet dataset by property filters
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Supported physicochemical property columns in the dataset
_SUPPORTED_PROPS = frozenset({"HBD", "HBA", "TPSA", "BBBP", "Mutag", "HIA", "QED"})


def _parse_range(value) -> tuple[Optional[float], Optional[float]]:
    """Parse a range spec into (lo, hi).

    Accepts:
      - scalar int/float → exact match [v, v]
      - [lo, hi] or (lo, hi) → range (None = unbounded)
      - {"min": lo, "max": hi} → range
    """
    if isinstance(value, (int, float)):
        return float(value), float(value)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        lo = None if value[0] is None else float(value[0])
        hi = None if value[1] is None else float(value[1])
        return lo, hi
    if isinstance(value, dict):
        lo = value.get("min", value.get("low"))
        hi = value.get("max", value.get("high"))
        return (
            None if lo is None else float(lo),
            None if hi is None else float(hi),
        )
    raise ValueError(f"Cannot parse range spec: {value!r}")


def search_partitioned_parquet(
    db_path: str,
    property_range_dict: Optional[dict] = None,
    max_return: int = 5,
    smiles_col: str = "smiles",
) -> list[str]:
    """Query a partitioned parquet molecule database by property ranges.

    Parameters
    ----------
    db_path:
        Path to the root of the partitioned parquet directory.
        The actual parquet files are expected under ``{db_path}/parts/``.
    property_range_dict:
        Dict of property_name → range spec.  Supported property names:
        HBD, HBA, TPSA, BBBP, Mutag, HIA, QED.
        Example: ``{"TPSA": [60, 100], "QED": [0.7, None]}``.
    max_return:
        Maximum number of SMILES to return.
    smiles_col:
        Name of the SMILES column in the dataset.

    Returns
    -------
    List of SMILES strings (up to *max_return*).

    Raises
    ------
    RuntimeError
        If the database cannot be opened.
    ValueError
        If ``property_range_dict`` is missing or empty.
    """
    if not property_range_dict:
        raise ValueError("'property_range_dict' must be a non-empty dict.")

    try:
        import pyarrow.compute as pc
        import pyarrow.dataset as pads
    except ImportError as exc:
        raise ImportError(
            "'pyarrow' is required for parquet search. "
            "Install with: pip install pyarrow"
        ) from exc

    parts_dir = os.path.join(db_path, "parts")
    try:
        dataset = pads.dataset(parts_dir, format="parquet")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to open parquet dataset at '{parts_dir}': {exc}"
        ) from exc

    cols = set(dataset.schema.names)
    expr = None

    for prop, val in property_range_dict.items():
        if prop not in _SUPPORTED_PROPS or prop not in cols:
            if prop not in _SUPPORTED_PROPS:
                logger.warning("Unsupported property '%s', skipping.", prop)
            else:
                logger.warning("Property '%s' not in dataset schema, skipping.", prop)
            continue
        lo, hi = _parse_range(val)
        if lo is not None:
            e = pc.field(prop) >= lo
            expr = e if expr is None else expr & e
        if hi is not None:
            e = pc.field(prop) <= hi
            expr = e if expr is None else expr & e

    scanner = dataset.scanner(columns=[smiles_col], filter=expr)
    table   = scanner.head(max_return + 1)
    smiles  = table.column(smiles_col).to_pylist()
    return smiles[:max_return]
