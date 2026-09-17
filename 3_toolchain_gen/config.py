"""Tool-server configuration for substructure-generation tool-chain building.

Tool names here must match those registered in :mod:`molkit.tools`
(:data:`molkit.tools.TOOL_SERVER_PORTS`).  The substructure-generation task
uses only this reduced tool set: molecule analysis / verification, index-based
molecule editing, and substructure matching.  The legacy generator / database /
MolMIM-optimisation tools are gone.
"""

from __future__ import annotations

import os

from molkit.tools import TOOL_SERVER_PORTS  # noqa: F401 (re-exported for convenience)

# ---------------------------------------------------------------------------
# Tool server host / timeout
# ---------------------------------------------------------------------------
TOOL_SERVER_HOST: str = os.environ.get("TOOL_SERVER_HOST", "http://localhost")
TOOL_SERVER_TIMEOUT: int = int(os.environ.get("TOOL_SERVER_TIMEOUT", "120"))

# ---------------------------------------------------------------------------
# Tool groups (the reduced, current tool set — see molkit.tools.__init__)
# ---------------------------------------------------------------------------
# Analysis / verification tools (no molecule mutation).
ANALYSIS_TOOLS: list[str] = [
    "analyze_properties",
    "match_substructure",
    "label_atom_indices",
]

# Molecule-editing tools.  ``edit_fragment`` is the single MMP-style edit primitive
# (attach + swap + remove, ``from_smiles -> to_smiles`` pinned with ``anchors``);
# ``suggest_edits`` ranks candidate ``edit_fragment`` arguments toward a property
# box (called immediately before every edit to choose its arguments).
EDIT_TOOLS: list[str] = [
    "suggest_edits",
    "edit_fragment",
]

# Every tool advertised to the model for this task (the ``tool_set`` field of
# each emitted training record).
ALL_TOOL_NAMES: list[str] = sorted(set(ANALYSIS_TOOLS + EDIT_TOOLS))

# ---------------------------------------------------------------------------
# Property-name vocabulary
# ---------------------------------------------------------------------------
# Properties returned by analyze_properties that are predicted by admet_ai
# (slower; only computed when requested).  Used purely for bookkeeping, so the
# set is a superset of what instances actually constrain (HIA is never used).
ADMET_PROPERTY_NAMES: frozenset[str] = frozenset(
    {"logD", "logS", "BBBP", "Mutag", "HIA"}
)
