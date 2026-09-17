from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional


def _parse_urls(raw: str | list[str]) -> list[str]:
    """Accept a single URL string, a comma-separated string, or a list."""
    if isinstance(raw, list):
        return [u.strip() for u in raw if u.strip()]
    return [u.strip() for u in raw.split(",") if u.strip()]


@dataclass
class VLLMConfig:
    """Configuration for one or more vLLM server endpoints.

    ``base_urls`` is a list of OpenAI-compatible base URLs.  The LLMClient
    distributes requests across them in round-robin order.  A single URL may
    be given as a plain string (``base_urls="http://…"``); it is normalised to
    a list automatically via ``__post_init__``.

    The legacy ``base_url`` keyword (single string) is also accepted and
    converted to ``base_urls`` in ``__post_init__`` for backward compatibility.
    """

    base_urls: list[str] | str = field(
        default_factory=lambda: [
            f"http://localhost:{port}/v1" for port in range(8080, 8087)
        ]
    )
    model: str = "Qwen/Qwen3.6-27B"
    api_key: str = "EMPTY"
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.95
    # Disable the model's reasoning/thinking channel. Qwen3.x reasoning models
    # otherwise spend the token budget on a hidden scratchpad — which for short
    # reasoning generations either truncates the visible answer to empty or
    # leaks a "Thinking Process:" block into the content. We only want concise
    # reasoning text here, so thinking is off by default.
    enable_thinking: bool = False

    def __post_init__(self) -> None:
        self.base_urls = _parse_urls(self.base_urls)


def _normalize_vllm_dict(d: dict) -> dict:
    """Normalise a raw config dict: convert legacy ``base_url`` → ``base_urls``."""
    d = dict(d)
    if "base_url" in d and "base_urls" not in d:
        d["base_urls"] = _parse_urls(d.pop("base_url"))
    elif "base_url" in d:
        d.pop("base_url")  # ignore if base_urls already present
    return d


@dataclass
class PipelineConfig:
    """Configuration for the training data generation pipeline."""

    generator: VLLMConfig = field(default_factory=VLLMConfig)
    augmentor: VLLMConfig = field(default_factory=VLLMConfig)
    verifier: VLLMConfig = field(default_factory=VLLMConfig)
    reflection_probability: float = 0.3
    max_retries: int = 3
    verifier_threshold: float = 7.0
    output_path: str = "training_data.jsonl"
    batch_size: int = 10
    seed: Optional[int] = None
    use_augmentor: bool = False
    use_verifier: bool = False
    # Segmented (edit-boundary) training data generation.
    segment_edit_boundaries: bool = True
    include_rejected_rounds: bool = False
    # Whether to include per-round deltas (Δ lines) in ## Progress So Far.
    show_progress_delta: bool = False
    # Write all four reasoning spans of an edit round (suggest / edit / checkpoint
    # / memory note) in ONE LLM call instead of four (MERGE_ROUND_REASONING=1).
    # Falls back to the single-purpose prompt for any span the merged call does not
    # emit, and for rounds that are not the standard suggest_edits → edit_fragment
    # pair.
    #
    # OFF by default: measured on the 998-chain sample it is 1.32x faster
    # (t@90% 54s → 41s; calls/chain 6.3 → 2.9; computed prefill −33%) but one call
    # doing four jobs loses per-span discipline, and the losses land on the axis
    # this dataset was tuned for. Against the four-call path: the edit reasoning's
    # top 6-gram rate 5.0% → 29.7% and the checkpoint's 13.8% → 60.9%; edit spans
    # within the 65-word cap 97.5% → 84.7%; numbers the model computed itself
    # rather than quoting 1 → 116 (1.16%). Everything rule-based stayed identical
    # (whole-instance consistency 13/13, Result-line traceability, 0 inverted ±
    # claims), so this is purely a prose-quality trade — turn it on only if the
    # wall-clock matters more than that.
    merge_round_reasoning: bool = field(
        default_factory=lambda: os.environ.get("MERGE_ROUND_REASONING", "0") != "0")
    # ── LEAN REASONING (LEAN_REASONING=0 to restore the old corpus) ─────────
    # Emit reasoning ONLY where the model actually makes a decision. Three spans
    # in an edit segment do not: the next tool call is already determined by the
    # format, so the prose is a paraphrase of a fixed transition and the model
    # spends tokens learning to recite it.
    #
    # Dropped (edit segments = segment 1 onward, plus the terminal segment):
    #   1. before ``suggest_edits``  — the round always opens with it
    #   2. after ``edit_fragment``, before the verification checkpoint — the
    #      checkpoint always follows an edit
    #   3. before ``<ANSWER>`` in the terminal segment — the conversation
    #      already shows every constraint met, so the confirm sentence
    #      restates it
    #
    # Kept, because each IS a decision:
    #   1. the seed segment's intro (derives the SMARTS + scaffold SMILES from
    #      the free-form prompt — the one span the harness parses back at eval)
    #   2. the ``edit_fragment`` reasoning (picks WHICH suggest_edits rule to
    #      commit — the actual choice in the round)
    #
    # Also saves the LLM calls that produced the dropped spans (the assistant
    # turn keeps its tool_calls and just carries empty content).
    lean_reasoning: bool = field(
        default_factory=lambda: os.environ.get("LEAN_REASONING", "1") != "0")
    # ── NAIVE REASONING (NAIVE_REASONING=1) — the ablation baseline ─────────
    # Write the EDIT-ROUND reasoning — the span that picks which suggest_edits rule
    # to commit — from the prompt alone. The model gets the user query, the current
    # molecule, the raw analyze_properties output, the raw suggest_edits JSON and
    # the call it is about to make, and is asked to write the turn that goes between
    # them. Everything this pipeline normally computes for that span is withheld:
    # the Landing Safety table, the round context, the spread verdict, the rendered
    # candidate comparison, the verified chemistry names, the decision order, the
    # committed candidate's index — and so are its guards and retries.
    #
    # SCOPE: this span only. The seed span (derive the SMARTS, transcribe it to the
    # scaffold SMILES) keeps the normal prompt, because it is the one span the eval
    # harness parses back and because the ablation is about rule selection. Measured
    # with the naive prompt there too: 17% of seed spans became a visible scratchpad,
    # the longest ran 858 words, and one built its own molecule with the double-bond
    # geometry flipped against the tool call.
    #
    # The point is to price the scaffolding. Claims like "ties are reported as ties"
    # or "no inverted ± comparison" are claims about the DIFFERENCE between the two
    # prompts, and that difference is only measurable if the same tool chains can be
    # rendered both ways. Not for training corpora: with the guards off nothing stops
    # a fabricated count from shipping.
    #
    # Forces the four-call path (merge_round_reasoning is meaningless here — the
    # merged prompt is built entirely out of computed blocks).
    naive_reasoning: bool = field(
        default_factory=lambda: os.environ.get("NAIVE_REASONING", "0") != "0")
    # Drop tool chains whose final molecule does NOT satisfy every constraint.
    # Such chains end on an out-of-range checkpoint and never reach an <ANSWER>,
    # so as training data they only contribute truncated "keep editing"
    # trajectories on molecules the planner itself failed to fix. Excluding them
    # keeps the SFT corpus to fully-solved constructive examples.
    drop_unsatisfied_chains: bool = True
    # Number of independent generation passes per tool-chain input.
    # Each pass produces different LLM-generated reasoning text while reusing
    # the same ground-truth tool calls / responses.
    num_generations: int = 1

    # ── AUTHORED-EDIT branch (counterfactual empty suggest_edits) ────────────
    # At inference the agent hits an EMPTY suggest_edits response on 64-80% of
    # calls (measured on the molkit-100 gen-eval dumps), and collapses:
    # every training round pairs a NON-empty candidate list with "Candidate #N is
    # the top-ranked choice", so the trained program has no branch for [] and the
    # model hallucinates a candidate, then emits malformed JSON or runs out of
    # tokens mid tool-call. That accounts for ~19% of val rollouts (`no_action`).
    #
    # This knob rewrites a FRACTION of decorate rounds counterfactually: the
    # suggest_edits response is rendered as `[]` and the edit reasoning is
    # regenerated from the property gap + RDKit fact block alone, with no
    # candidate list. The committed edit is unchanged (it stays the one the
    # planner chose), so the tool calls remain ground truth — only the rendered
    # response and the reasoning span change.
    #
    # CAVEAT (measured, keep in mind when reading results): these rounds are NOT
    # real stuck states. A real [] fires at states with a mean of 1.60 failing
    # properties; an average decorate round has 3.59, and 25.8% of committed edits
    # there break an already-passing constraint. `authored_max_failing` narrows
    # the selection toward the stuck-state profile, but it cannot remove the
    # mismatch — only a planner-side escape (which needs a stage-3 rerun) can.
    # OFF by default: an unparameterised 4_sftdata_gen.sh must keep producing the
    # existing corpus byte-for-byte. To build the authored variant, set this AND
    # point OUTPUT_ROOT at a separate tree so the two corpora stay side by side:
    #   OUTPUT_ROOT=…/sftdata_authored AUTHORED_EDIT_FRACTION=0.5 \
    #       bash run_scripts/pipeline/4_sftdata_gen.sh
    #
    # NOTE this is a fraction of ELIGIBLE rounds, not of all rounds. Measured on
    # chunk 0: 21% of decorate rounds pass the authored_max_failing filter, so
    # 0.5 ≈ 10% of decorate rounds rendered with an empty list. That ratio is the
    # main thing to tune. It deliberately does NOT match the inference rate
    # (64-80%): the point is to install the branch, and flooding the corpus with
    # it risks the opposite failure — the model authoring a rule even when the
    # tool DID return candidates. Check for that at eval before raising this.
    authored_edit_fraction: float = field(
        default_factory=lambda: float(os.environ.get("AUTHORED_EDIT_FRACTION", "0")))
    # Only rounds with at most this many out-of-range properties are eligible.
    # Real [] states: 53% have exactly 1 failing property, 33% have 2, 13% have 3,
    # and NONE have 4+. Capping at 2 covers 86% of the real distribution; 3 covers
    # 99% but widens the gap to the risk regime the branch actually fires in.
    authored_max_failing: int = field(
        default_factory=lambda: int(os.environ.get("AUTHORED_MAX_FAILING", "2")))

    @classmethod
    def from_json(cls, path: str) -> PipelineConfig:
        with open(path) as f:
            data = json.load(f)
        gen_cfg = _normalize_vllm_dict(data.pop("generator", {}))
        aug_cfg = _normalize_vllm_dict(data.pop("augmentor", {}))
        ver_cfg = _normalize_vllm_dict(data.pop("verifier", {}))
        return cls(
            generator=VLLMConfig(**gen_cfg),
            augmentor=VLLMConfig(**aug_cfg),
            verifier=VLLMConfig(**ver_cfg),
            **data,
        )

    def to_json(self, path: str) -> None:
        import dataclasses

        data = dataclasses.asdict(self)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)