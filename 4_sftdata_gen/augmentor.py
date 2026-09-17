"""Augmentor – transforms generated training examples for data diversity.

Sits between Generator and Verifier in the pipeline.  Applies:
    1. Tool name aliasing  (same semantics, different function names)
    2. Tool / parameter description paraphrasing
    3. System prompt paraphrasing
    4. User prompt paraphrasing
    5. Tool-schema order shuffling
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import random
from typing import Optional

from .config import PipelineConfig
from .llm_client import LLMClient
from .prompts import (
    AUGMENT_SYSTEM_PROMPT,
    AUGMENT_TOOLS_PROMPT,
    AUGMENT_USER_PROMPT,
)
from .schema import ToolCall, ToolStep, TrainingExample

logger = logging.getLogger(__name__)


class Augmentor:
    """Augments a *TrainingExample* for textual diversity while preserving semantics."""

    def __init__(self, config: PipelineConfig) -> None:
        self.llm = LLMClient(config.augmentor)

    # -- public API ---------------------------------------------------------

    async def augment(self, example: TrainingExample) -> TrainingExample:
        """Return a new *TrainingExample* with paraphrased text and shuffled tools."""

        # Run the three LLM calls concurrently
        tool_mapping_coro = self._augment_tools(example.tool_schemas)
        system_coro = self._paraphrase_text(
            example.system_prompt, AUGMENT_SYSTEM_PROMPT
        )
        user_coro = self._paraphrase_text(
            example.user_prompt, AUGMENT_USER_PROMPT
        )

        tool_mapping, new_system_prompt, new_user_prompt = await asyncio.gather(
            tool_mapping_coro, system_coro, user_coro
        )

        # Apply tool-name / description remapping to schemas, then shuffle
        new_schemas = self._remap_schemas(example.tool_schemas, tool_mapping)
        random.shuffle(new_schemas)

        # Remap tool names inside tool-step texts and tool_call.name fields
        new_tool_steps = self._remap_tool_steps(example.tool_steps, tool_mapping)

        # Remap tool names that may appear in the final verification text
        new_verification = self._remap_text(
            example.final_verification, tool_mapping
        )

        return TrainingExample(
            system_prompt=new_system_prompt,
            tool_schemas=new_schemas,
            user_prompt=new_user_prompt,
            tool_steps=new_tool_steps,
            final_verification=new_verification,
            molecule_prediction=example.molecule_prediction,
        )

    # -- tool augmentation --------------------------------------------------

    async def _augment_tools(
        self, tool_schemas: list[dict]
    ) -> dict[str, dict]:
        """Generate alternative names and descriptions for every tool.

        Returns a mapping ``original_name → {new_name, new_description,
        param_descriptions}`` .  Falls back to an identity mapping on failure.
        """
        tools_info: list[dict] = []
        for schema in tool_schemas:
            func = schema.get("function", schema)
            name = func["name"]
            desc = func.get("description", "")
            params = func.get("parameters", {}).get("properties", {})
            param_descs = {
                k: v.get("description", "") for k, v in params.items()
            }
            tools_info.append(
                {
                    "name": name,
                    "description": desc,
                    "param_descriptions": param_descs,
                }
            )

        prompt = AUGMENT_TOOLS_PROMPT.format(
            tools_json=json.dumps(tools_info, indent=2)
        )

        try:
            raw = await self.llm.generate(prompt, temperature=0.9)
            mapping = self._parse_json_response(raw)
            if mapping and isinstance(mapping, dict):
                return mapping
        except Exception as e:
            logger.warning("Tool augmentation LLM call failed, using originals: %s", e)

        # Fallback – identity mapping (no change)
        return {
            info["name"]: {
                "new_name": info["name"],
                "new_description": info["description"],
                "param_descriptions": info["param_descriptions"],
            }
            for info in tools_info
        }

    # -- text paraphrasing --------------------------------------------------

    async def _paraphrase_text(
        self, original: str, prompt_template: str
    ) -> str:
        """Paraphrase *original* using the given prompt template."""
        prompt = prompt_template.format(original_text=original)
        try:
            result = await self.llm.generate(prompt, temperature=0.8)
            # Basic sanity – the result should be non-trivial
            if result and len(result) > 10:
                return result
        except Exception as e:
            logger.warning("Paraphrasing failed, using original: %s", e)
        return original

    # -- deterministic remapping helpers ------------------------------------

    @staticmethod
    def _remap_schemas(
        schemas: list[dict], mapping: dict[str, dict]
    ) -> list[dict]:
        """Deep-copy and remap names / descriptions in tool schemas."""
        new_schemas: list[dict] = []
        for schema in schemas:
            schema = copy.deepcopy(schema)
            func = schema.get("function", schema)
            original_name = func.get("name", "")
            aug = mapping.get(original_name)
            if aug:
                func["name"] = aug.get("new_name", original_name)
                if "new_description" in aug:
                    func["description"] = aug["new_description"]
                param_descs = aug.get("param_descriptions", {})
                if param_descs:
                    props = func.get("parameters", {}).get("properties", {})
                    for pname, pdesc in param_descs.items():
                        if pname in props and pdesc:
                            props[pname]["description"] = pdesc
            new_schemas.append(schema)
        return new_schemas

    @staticmethod
    def _remap_tool_steps(
        steps: list[ToolStep], mapping: dict[str, dict]
    ) -> list[ToolStep]:
        """Remap tool names in every *ToolStep*."""
        # Build a simple old→new name lookup (only entries that actually change)
        name_map: dict[str, str] = {
            orig: aug["new_name"]
            for orig, aug in mapping.items()
            if aug.get("new_name") and aug["new_name"] != orig
        }

        new_steps: list[ToolStep] = []
        for step in steps:
            new_name = name_map.get(step.tool_call.name, step.tool_call.name)

            reasoning = step.reasoning
            reflection = step.reflection
            for old, new in name_map.items():
                reasoning = reasoning.replace(old, new)
                if reflection:
                    reflection = reflection.replace(old, new)

            # Remap the name of every parallel companion (analyze / label).
            new_parallel_tool_calls = [
                ToolCall(
                    name=name_map.get(pc.name, pc.name),
                    arguments=pc.arguments,
                )
                for pc in step.parallel_tool_calls
            ]

            new_steps.append(
                ToolStep(
                    reasoning=reasoning,
                    tool_call=ToolCall(
                        name=new_name,
                        arguments=step.tool_call.arguments,
                    ),
                    tool_response=step.tool_response,
                    reflection=reflection,
                    parallel_tool_calls=new_parallel_tool_calls,
                    parallel_tool_responses=list(step.parallel_tool_responses),
                )
            )
        return new_steps

    @staticmethod
    def _remap_text(text: str, mapping: dict[str, dict]) -> str:
        """Replace original tool names with their aliases in free-form text."""
        for orig, aug in mapping.items():
            new_name = aug.get("new_name", orig)
            if new_name != orig:
                text = text.replace(orig, new_name)
        return text

    # -- JSON parsing -------------------------------------------------------

    @staticmethod
    def _parse_json_response(raw: str) -> Optional[dict]:
        """Extract a JSON object from a possibly markdown-fenced LLM response."""
        text = raw.strip()
        if "```" in text:
            start = text.index("```") + 3
            if text[start:].startswith("json"):
                start += 4
            end = text.index("```", start)
            text = text[start:end].strip()
        return json.loads(text)
