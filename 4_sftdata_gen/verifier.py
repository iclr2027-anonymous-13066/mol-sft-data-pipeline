"""Verifier – evaluates the quality of generated training examples.

The Verifier LLM scores each training example on multiple dimensions
(reasoning, coherence, reflection, verification) and returns an
accept/reject decision based on the configured threshold.
"""

from __future__ import annotations

import json
import logging

from .config import PipelineConfig
from .llm_client import LLMClient
from .prompts import VERIFIER_PROMPT, VERIFIER_SYSTEM_PROMPT
from .schema import TrainingExample, VerificationResult

logger = logging.getLogger(__name__)


class Verifier:
    """Verifies generated training data quality using a separate LLM."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.llm = LLMClient(config.verifier)
        self.threshold = config.verifier_threshold

    async def verify(self, example: TrainingExample) -> VerificationResult:
        """Score and accept/reject a training example."""
        formatted = self._format_conversation(example)

        prompt = VERIFIER_PROMPT.format(
            user_prompt=example.user_prompt,
            formatted_conversation=formatted,
            molecule_prediction=example.molecule_prediction,
            threshold=self.threshold,
        )

        raw_response = await self.llm.generate(
            prompt=prompt,
            system_prompt=VERIFIER_SYSTEM_PROMPT,
            temperature=0.1,  # low temperature for consistent evaluation
        )

        return self._parse_response(raw_response)

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _format_conversation(example: TrainingExample) -> str:
        parts: list[str] = []
        for i, step in enumerate(example.tool_steps):
            parts.append(f"**Step {i + 1}**")
            parts.append(f"Reasoning: {step.reasoning}")
            parts.append(
                f"Tool Call: {step.tool_call.name}"
                f"({json.dumps(step.tool_call.arguments)})"
            )
            parts.append(f"Response: {step.tool_response}")
            if step.reflection:
                parts.append(f"Reflection: {step.reflection}")
            parts.append("")  # blank line between steps

        parts.append(f"**Final Verification**: {example.final_verification}")
        return "\n".join(parts)

    @staticmethod
    def _parse_response(raw: str) -> VerificationResult:
        """Extract structured evaluation from the verifier's raw output."""
        try:
            text = raw.strip()
            # Handle markdown-fenced JSON blocks
            if "```" in text:
                start = text.index("```") + 3
                # skip optional language tag (e.g. ```json)
                if text[start:].startswith("json"):
                    start += 4
                end = text.index("```", start)
                text = text[start:end].strip()

            data = json.loads(text)
            return VerificationResult(
                accepted=data.get("accepted", False),
                scores=data.get("scores", {}),
                feedback=data.get("feedback", ""),
            )
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning("Failed to parse verifier response: %s\nRaw: %s", e, raw)
            return VerificationResult(
                accepted=False,
                scores={"overall": 0.0},
                feedback=f"Parse error: {e}",
            )
