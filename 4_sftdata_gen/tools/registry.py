"""Tool schema registry for chemistry tools.

Tool schemas are sourced from ``molkit.tools.TOOL_REGISTRY`` (each tool
instance exposes ``to_schema()``).  The ``ToolRegistry`` class here manages
schemas and optional runtime implementations for the SFT data pipeline.
"""

from __future__ import annotations

import json
from typing import Callable, Optional

from molkit.tools import TOOL_REGISTRY as _MOLKIT_TOOL_REGISTRY

# Re-export so the rest of the package can ``from .registry import CHEMISTRY_TOOLS``.
CHEMISTRY_TOOLS: dict[str, dict] = {
    name: tool.to_schema() for name, tool in _MOLKIT_TOOL_REGISTRY.items()
}


class ToolRegistry:
    """Registry managing tool schemas and optional runtime implementations."""

    def __init__(self) -> None:
        self._schemas: dict[str, dict] = {}
        self._implementations: dict[str, Callable] = {}

    # -- registration -------------------------------------------------------

    def register(
        self,
        name: str,
        schema: dict,
        implementation: Optional[Callable] = None,
    ) -> None:
        self._schemas[name] = schema
        if implementation is not None:
            self._implementations[name] = implementation

    def register_implementation(self, name: str, fn: Callable) -> None:
        if name not in self._schemas:
            raise KeyError(
                f"Tool '{name}' not in registry. Register its schema first."
            )
        self._implementations[name] = fn

    # -- lookup -------------------------------------------------------------

    def get_schema(self, name: str) -> dict:
        return self._schemas[name]

    def get_schemas(self, names: list[str]) -> list[dict]:
        return [self._schemas[name] for name in names if name in self._schemas]

    def get_implementation(self, name: str) -> Optional[Callable]:
        return self._implementations.get(name)

    def has_implementation(self, name: str) -> bool:
        return name in self._implementations

    @property
    def available_tools(self) -> list[str]:
        return list(self._schemas.keys())

    def get_tool_descriptions(self, names: list[str]) -> str:
        """Return a human-readable description block for the given tool names."""
        parts: list[str] = []
        for name in names:
            schema = self._schemas.get(name)
            if schema is None:
                continue
            func = schema.get("function", schema)
            desc = func.get("description", "No description")
            params = func.get("parameters", {})
            parts.append(
                f"- **{name}**: {desc}\n"
                f"  Parameters: {json.dumps(params, indent=2)}"
            )
        return "\n\n".join(parts)

    # -- factory ------------------------------------------------------------

    @classmethod
    def default(cls) -> ToolRegistry:
        """Create a registry pre-loaded with the molkit chemistry tools."""
        registry = cls()
        for name, schema in CHEMISTRY_TOOLS.items():
            registry.register(name, schema)
        return registry
