"""molkit/tools/base.py

Abstract base class for all molkit tools.

Design goals of this base class, against the earlier tool base it replaces:
  - `execute(**kwargs)`  instead of `_run_base(*args, **kwargs)` — public, keyword-only
  - `parameters: dict`   instead of `properties: dict`           — aligns with JSON Schema
  - `to_schema()`        instead of a separate TOOLS_JSON_SCHEMA  — schema lives with impl
  - Removed: func_name, examples, _init_modules, deprecated run()
"""

from __future__ import annotations

import difflib
import functools
import inspect
import json
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


def _suggest(name: str, accepted: list[str]) -> str:
    """``" (did you mean 'from_smiles'?)"`` for a near-miss argument name, else ``""``.

    Combines fuzzy matching with a prefix rule, because the common LLM slips are
    truncations (``from`` → ``from_smiles``, ``to`` → ``to_smiles``) whose
    similarity ratio falls below any sane fuzzy cutoff.
    """
    close = difflib.get_close_matches(name, accepted, n=1, cutoff=0.6)
    if not close:
        close = [a for a in accepted if a.startswith(name) or name.startswith(a)][:1]
    return f" (did you mean '{close[0]}'?)" if close else ""


class BaseTool(ABC):
    """Abstract base for molkit tools.

    Subclasses must define three class attributes:

        name        : str  — snake_case identifier used in LLM tool calls
        description : str  — natural language description shown to the LLM
        parameters  : dict — JSON Schema
                             {"type": "object", "properties": {...}, "required": [...]}

    And implement one method:

        execute(**kwargs) -> str | dict
            Core execution logic. Return str or dict.

    Usage:
        tool = MyTool()
        result = tool({"mol_smiles": "CCO"})                      # str  (for LLM)
        result = tool({"mol_smiles": "CCO"}, return_text=False)   # dict (for eval code)
        schema = tool.to_schema()                                 # plug into LLM tools
    """

    name: str
    description: str
    parameters: dict  # JSON Schema

    def __init_subclass__(cls, **kwargs) -> None:
        """Enforce the argument-binding guard on every tool, including the many
        that override ``__call__`` for custom text formatting.

        The guard lives in :meth:`_binding_error` and is applied by wrapping a
        subclass's own ``__call__`` here rather than by asking each subclass to
        remember to call it — a tool that forgot would go back to raising
        ``TypeError`` on a mistyped argument name, which the tool server can only
        report as an opaque HTTP 500.
        """
        super().__init_subclass__(**kwargs)
        own_call = cls.__dict__.get("__call__")
        if own_call is None or getattr(own_call, "_binding_guarded", False):
            return

        @functools.wraps(own_call)
        def guarded(self, inputs: dict, *args, **kw):
            err = self._binding_error(inputs)
            return err if err is not None else own_call(self, inputs, *args, **kw)

        guarded._binding_guarded = True
        cls.__call__ = guarded

    @abstractmethod
    def execute(self, **kwargs) -> str | dict:
        """Core execution logic.

        Keyword arguments correspond 1-to-1 with the keys in parameters["properties"].
        Return str  -> passed directly as the LLM tool result.
        Return dict -> serialized to JSON when return_text=True.
        """
        ...

    def _binding_error(self, inputs: dict) -> str | None:
        """Return an ``Input Argument Error: …`` message if *inputs* cannot bind to
        :meth:`execute`, else None.

        Without this, a caller that invents an argument name (an LLM writing
        ``atom_indices`` instead of ``anchors``, or ``from`` instead of
        ``from_smiles``) makes ``execute(**inputs)`` raise ``TypeError`` — which the
        tool server turns into an opaque HTTP 500 and the agent transcript records
        as ``500 Server Error: Internal Server Error``, telling the model nothing.
        Bad arguments are a normal, recoverable tool outcome, so they are reported
        the same way every other validation failure in these tools is: as an error
        STRING the model can read and correct.
        """
        if not isinstance(inputs, dict):
            return (f"Input Argument Error: '{self.name}' expects an object of named "
                    f"arguments, got {type(inputs).__name__}.")
        try:
            params = inspect.signature(self.execute).parameters
        except (TypeError, ValueError):  # pragma: no cover — builtin/C callable
            return None
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return None  # execute(**kwargs) accepts anything — nothing to reject
        accepted = [n for n, p in params.items()
                    if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                  inspect.Parameter.KEYWORD_ONLY)]
        required = [n for n in accepted
                    if params[n].default is inspect.Parameter.empty]
        unknown = [k for k in inputs if k not in accepted]
        missing = [n for n in required if n not in inputs]
        if not unknown and not missing:
            return None
        parts = []
        if unknown:
            parts.append("unexpected argument(s) " + ", ".join(
                f"'{k}'{_suggest(k, accepted)}" for k in unknown))
        if missing:
            parts.append("missing required argument(s) "
                         + ", ".join(f"'{k}'" for k in missing))
        return (f"Input Argument Error: {'; '.join(parts)} for '{self.name}'. "
                f"Accepted arguments: {', '.join(accepted)} "
                f"(required: {', '.join(required) or 'none'}).")

    def __call__(self, inputs: dict, return_text: bool = True) -> str | dict:
        """Unified call interface.

        Args:
            inputs:      Parameter dict matching the parameters schema.
            return_text: True  -> always return str (for LLM tool result).
                         False -> return raw dict/list (for internal eval code).
        """
        binding_error = self._binding_error(inputs)
        if binding_error is not None:
            return binding_error
        result = self.execute(**inputs)
        if return_text and not isinstance(result, str):
            return json.dumps(result, ensure_ascii=False)
        return result

    def to_schema(self) -> dict:
        """Return the OpenAI / LiteLLM function-calling schema for this tool.

        Example:
            tools = [tool.to_schema() for tool in TOOLS]
            client.chat(messages, tools=tools)
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
