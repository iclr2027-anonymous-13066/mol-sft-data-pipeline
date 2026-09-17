from __future__ import annotations

import json
import uuid
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class ToolCall(BaseModel):
    """A single tool call with name and arguments."""

    name: str
    arguments: dict


class ToolChainStep(BaseModel):
    """A step in the ground-truth tool chain.

    If ``expected_response`` is provided it will be used directly instead of
    executing the tool at runtime.

    ``parallel_tool_calls`` holds any tools executed SIMULTANEOUSLY with
    ``tool_call`` — the verification checkpoint dispatches ``match_substructure``
    (the primary ``tool_call``) alongside ``analyze_properties`` and
    ``label_atom_indices`` in parallel, so it is rendered as a single assistant
    message with N tool calls followed by N tool responses. Each entry of
    ``parallel_expected_responses`` is aligned by index with
    ``parallel_tool_calls``.
    """

    tool_call: ToolCall
    expected_response: Optional[str] = None
    # Tools called simultaneously with ``tool_call`` (e.g. the checkpoint's
    # analyze_properties + label_atom_indices companions).
    parallel_tool_calls: list[ToolCall] = Field(default_factory=list)
    parallel_expected_responses: list[Optional[str]] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _migrate_singular_parallel(cls, data):
        """Read legacy chunks that stored a single ``parallel_tool_call``.

        Older tool-chain files used the scalar ``parallel_tool_call`` /
        ``parallel_expected_response`` pair. Fold them into the list form so
        existing on-disk data keeps loading unchanged.
        """
        if isinstance(data, dict) and data.get("parallel_tool_call") is not None \
                and not data.get("parallel_tool_calls"):
            data = dict(data)
            data["parallel_tool_calls"] = [data["parallel_tool_call"]]
            data["parallel_expected_responses"] = [data.get("parallel_expected_response")]
        return data

    @property
    def parallel_tool_call(self) -> Optional[ToolCall]:
        """Back-compat read accessor: the first parallel companion, if any."""
        return self.parallel_tool_calls[0] if self.parallel_tool_calls else None

    @property
    def parallel_expected_response(self) -> Optional[str]:
        return self.parallel_expected_responses[0] if self.parallel_expected_responses else None


class GeneratorInput(BaseModel):
    """Input to the generator pipeline.

    Attributes:
        user_prompt: The user query to answer.
        ground_truth_molecule: The correct molecule (e.g. SMILES).
        tool_chain: Ground-truth sequence of tool calls to follow.
        tool_set: Names of tools available for this example.
        generation_history: Previous generation attempts (used as context).
        metadata: Arbitrary metadata passed through to the output.
    """

    user_prompt: str
    ground_truth_molecule: str
    tool_chain: list[ToolChainStep]
    tool_set: list[str]
    generation_history: list[dict] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)

