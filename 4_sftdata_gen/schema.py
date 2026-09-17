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

    If ``expected_response`` is provided it is used directly instead of
    executing the tool at runtime.

    ``parallel_tool_calls`` holds any tools executed SIMULTANEOUSLY with
    ``tool_call`` — the verification checkpoint runs ``match_substructure`` (the
    primary ``tool_call``) alongside ``analyze_properties`` and
    ``label_atom_indices``, and is rendered as a single assistant message with N
    tool calls followed by N tool responses. ``parallel_expected_responses`` is
    aligned by index with ``parallel_tool_calls``.
    """

    tool_call: ToolCall
    expected_response: Optional[str] = None
    parallel_tool_calls: list[ToolCall] = Field(default_factory=list)
    parallel_expected_responses: list[Optional[str]] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _migrate_singular_parallel(cls, data):
        """Read legacy chunks that stored a single ``parallel_tool_call``."""
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
    """Input to the generator pipeline (one line of a stage-3 toolchain file)."""

    user_prompt: str
    ground_truth_molecule: str
    tool_chain: list[ToolChainStep]
    tool_set: list[str]
    generation_history: list[dict] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class ToolStep(BaseModel):
    """A completed tool interaction step in the generated training data.

    ``parallel_tool_calls`` holds any companions issued in the SAME assistant
    message as ``tool_call`` (the verification checkpoint's
    ``analyze_properties`` + ``label_atom_indices``); the message then
    renders one assistant turn with N tool calls followed by N tool responses.
    ``parallel_tool_responses`` is aligned by index with ``parallel_tool_calls``.
    """

    reasoning: str
    tool_call: ToolCall
    tool_response: str
    reflection: Optional[str] = None
    parallel_tool_calls: list[ToolCall] = Field(default_factory=list)
    parallel_tool_responses: list[Optional[str]] = Field(default_factory=list)

    @property
    def parallel_tool_call(self) -> Optional[ToolCall]:
        """Back-compat read accessor: the first parallel companion, if any."""
        return self.parallel_tool_calls[0] if self.parallel_tool_calls else None

    @property
    def parallel_tool_response(self) -> Optional[str]:
        return self.parallel_tool_responses[0] if self.parallel_tool_responses else None


class TrainingExample(BaseModel):
    """A complete training example ready for fine-tuning.

    Each example is one segment of a constructive build trajectory.  An
    *intermediate* segment stops after its last tool response; a *final* segment
    ends with a ``Verification:`` block followed by ``<ANSWER>``.  The segments
    of one chain are merged into a single conversation by
    :func:`merge_segments`.
    """

    system_prompt: str
    tool_schemas: list[dict]
    user_prompt: str
    tool_steps: list[ToolStep]
    final_verification: str = ""
    molecule_prediction: str = ""
    # Intermediate segment: the trajectory stops after the last tool response
    # instead of emitting ``<ANSWER>``.
    is_intermediate_segment: bool = False
    # Rule-based constraint-satisfaction block emitted right before the
    # ``<ANSWER>`` so the model learns to verify constraints.
    constraint_check: Optional[str] = None
    # Which generation pass produced this example (0-based).
    generation_idx: int = 0

    def to_messages(self) -> list[dict]:
        """Convert to OpenAI chat message format for fine-tuning."""
        messages: list[dict] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_prompt},
        ]

        for step in self.tool_steps:
            content_parts: list[str] = []
            if step.reflection:
                content_parts.append(step.reflection)
            content_parts.append(step.reasoning)
            content = "\n\n".join(p for p in content_parts if p)

            # All tools issued in THIS assistant turn: the primary ``tool_call``
            # followed by any parallel companions (the checkpoint's analyze +
            # label). Emitted as one assistant message with N tool calls, then one
            # ``tool`` response message per call (order-aligned by id).
            calls = [(step.tool_call, step.tool_response)]
            presps = step.parallel_tool_responses
            for j, pc in enumerate(step.parallel_tool_calls):
                calls.append((pc, presps[j] if j < len(presps) else ""))

            call_ids = [f"call_{uuid.uuid4().hex[:8]}" for _ in calls]
            messages.append({
                "role": "assistant",
                "content": content,
                "tool_calls": [
                    {
                        "id": call_ids[k],
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.arguments),
                        },
                    }
                    for k, (call, _) in enumerate(calls)
                ],
            })
            for k, (_, resp) in enumerate(calls):
                messages.append({
                    "role": "tool", "tool_call_id": call_ids[k],
                    "content": resp or "",
                })

        # ── Intermediate segment: stops after its last tool response ────────
        # No Verification block here — it is emitted only in the final <ANSWER>
        # segment, over the state the whole conversation has arrived at.
        if self.is_intermediate_segment:
            return messages

        # ── Final segment: Verification + final reasoning + <ANSWER> ───────
        final_parts: list[str] = []
        if self.constraint_check:
            final_parts.append(self.constraint_check)
        if self.final_verification:
            final_parts.append(self.final_verification)
        final_parts.append(f"<ANSWER>{self.molecule_prediction}</ANSWER>")
        messages.append({"role": "assistant", "content": "\n\n".join(final_parts)})
        return messages

    def to_training_format(self) -> dict:
        """Return the full training record (messages + tool schemas + metadata)."""
        example_type = (
            "segment_intermediate" if self.is_intermediate_segment else "normal"
        )
        return {
            "messages": self.to_messages(),
            "tools": self.tool_schemas,
            "metadata": {
                "molecule": self.molecule_prediction,
                "num_tool_calls": len(self.tool_steps),
                "example_type": example_type,
                "generation_idx": self.generation_idx,
            },
        }


def merge_segments(segments: list["TrainingExample"]) -> Optional[dict]:
    """Merge one chain's segments into a single full-context training record.

    The system and user turns are taken once from the first segment; every
    segment then contributes its assistant/tool turns in order.  In one
    continuous conversation the model can see the earlier rounds directly, so
    nothing has to be carried between them.

    Returns ``None`` for an empty list.
    """
    if not segments:
        return None

    head = segments[0]
    messages = [
        {"role": "system", "content": head.system_prompt},
        {"role": "user", "content": head.user_prompt},
    ]
    for seg in segments:
        # Drop each segment's own system/user pair; keep the rest in order.
        messages.extend(seg.to_messages()[2:])

    last = segments[-1]
    return {
        "messages": messages,
        "tools": head.tool_schemas,
        "metadata": {
            "molecule": last.molecule_prediction,
            "num_tool_calls": sum(len(s.tool_steps) for s in segments),
            "example_type": "normal",
            "generation_idx": head.generation_idx,
            "segments_merged": len(segments),
            "ends_with_answer": not last.is_intermediate_segment,
            # Where the seed round ends: ``messages[:seed_messages]`` is the
            # scaffold derivation on its own, which stays correct however the
            # rest of the chain turned out (see scripts/make_satisfied_only.py).
            "seed_messages": len(head.to_messages()),
        },
    }


class VerificationResult(BaseModel):
    """Result from the verifier model."""

    accepted: bool
    scores: dict[str, float]
    feedback: str
