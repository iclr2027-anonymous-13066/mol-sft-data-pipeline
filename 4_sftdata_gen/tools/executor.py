"""Tool executor – dispatches tool calls to registered implementations."""

from __future__ import annotations

import json
import logging
from typing import Optional

from ..schema import ToolCall
from .registry import ToolRegistry

logger = logging.getLogger(__name__)


class ToolExecutor:
    """Executes tool calls against registered implementations.

    If an ``expected_response`` is supplied (pre-computed ground-truth), it is
    returned directly without invoking any implementation.
    """

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    def execute(
        self,
        tool_call: ToolCall,
        expected_response: Optional[str] = None,
    ) -> str:
        # Prefer pre-computed response when available
        if expected_response is not None:
            return expected_response

        impl = self.registry.get_implementation(tool_call.name)
        if impl is None:
            raise RuntimeError(
                f"No implementation registered for tool '{tool_call.name}' and "
                f"no expected_response was provided.  Either register an "
                f"implementation or include expected_response in the tool chain."
            )

        try:
            result = impl(**tool_call.arguments)
            if isinstance(result, str):
                return result
            return json.dumps(result, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error("Tool execution failed for %s: %s", tool_call.name, e)
            raise
