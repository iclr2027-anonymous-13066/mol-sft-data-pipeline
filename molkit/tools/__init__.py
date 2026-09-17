"""molkit/tools/__init__.py  —  Tool registry

Exports
-------
TOOLS            : list of BaseTool instances (all registered tools)
TOOL_REGISTRY    : dict[str, BaseTool]  name → instance
TOOL_SERVER_PORTS: dict[str, int]       name → port
TOOLS_CLASS      : dict[str, type]      name → class  (for evaluate_benchmark.py compat)
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Port assignments  (one process per tool when running tool_server.py)
# ---------------------------------------------------------------------------
# Canonical layout is anchored at 9000. The whole stack (server bind in
# tool_server.py, the health checker, and the client in
# utils/global_tools.py) reads TOOL_SERVER_PORTS, so shifting the base moves
# servers and clients together — used to dodge a port conflict with a co-tenant
# on a shared node. Default 9000 keeps behavior unchanged for everyone else;
# override per-shell with MOLKIT_TOOL_PORT_BASE (e.g. 10000).

_PORT_BASE = int(os.environ.get("MOLKIT_TOOL_PORT_BASE", "10000"))

_BASE_PORTS: dict[str, int] = {
    "analyze_properties":   9000,  # MolPropAnalyzer
    "match_substructure":   9001,  # SubstructureMatch
    "label_atom_indices":   9002,  # AtomIndexLabeler
    "edit_fragment":        9003,  # EditFragment (attach + swap + remove)
    "suggest_edits":        9004,  # SuggestEdits (rank MMP edits toward a property box)
}

TOOL_SERVER_PORTS: dict[str, int] = {
    name: _PORT_BASE + (port - 9000) for name, port in _BASE_PORTS.items()
}

# ---------------------------------------------------------------------------
# Import all tool classes (graceful: log warning on ImportError)
# ---------------------------------------------------------------------------

TOOL_REGISTRY: dict[str, object] = {}   # name → instance
TOOLS_CLASS:   dict[str, type]   = {}   # name → class  (for eval compat)

def _register(instance) -> None:
    TOOL_REGISTRY[instance.name] = instance
    TOOLS_CLASS[instance.name]   = type(instance)


# -- analysis -----------------------------------------------------------------
try:
    from molkit.tools.analysis import MolPropAnalyzer, SubstructureMatch
    for _cls in (MolPropAnalyzer, SubstructureMatch):
        _register(_cls())
except Exception as e:
    logger.warning("Could not load analysis tools: %s", e)

# -- editing (molecule edit tools) ------------------------------------------------
try:
    from molkit.tools.editing import EditFragment, AtomIndexLabeler, SuggestEdits
    for _cls in (EditFragment, AtomIndexLabeler, SuggestEdits):
        _register(_cls())
except Exception as e:
    logger.warning("Could not load editing tools: %s", e)


# ---------------------------------------------------------------------------
# Flat list convenience accessor
# ---------------------------------------------------------------------------

TOOLS: list = list(TOOL_REGISTRY.values())


def get_tool(name: str):
    """Return the registered tool instance for *name*, or None."""
    return TOOL_REGISTRY.get(name)


def get_schemas() -> list[dict]:
    """Return OpenAI/LiteLLM tool schemas for all registered tools."""
    return [t.to_schema() for t in TOOLS]
