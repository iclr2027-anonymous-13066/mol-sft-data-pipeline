"""Split a constructive tool chain into per-edit training segments.

The stage-3 chains this module consumes are *constructive* and start from the
COMPLETE scaffold (direct-scaffold-seed): the molecule is tuned from that seed by
repeated edits, each a ``suggest_edits`` call (which ranks candidate edits toward
the property box) immediately followed by the ``edit_fragment`` that applies the
chosen candidate; after every edit a single 3-way checkpoint runs
``match_substructure`` ∥ ``analyze_properties`` ∥ ``label_atom_indices`` in
parallel — the ``label_atom_indices`` companion grounds the NEXT edit's
``anchors`` (so the label that used to be a separate pre-edit step now rides on
the previous checkpoint).

A ``Segment`` groups one ``edit_fragment`` call — with its ``suggest_edits``
lead-in (``lead_steps``) and the checkpoint that follows it up to the next edit's
lead-in.  Rounds 0..i-1 are packaged as ``prior_rounds`` so the generator can fold
them into the rolling memory block.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from .schema import GeneratorInput, ToolChainStep

logger = logging.getLogger(__name__)

# ── tool-name sets ────────────────────────────────────────────────────────

# Structural edits that produce a new molecule (open a new round). The current
# tool set has ONE edit primitive: edit_fragment (attach + swap + remove).
EDIT_TOOLS: set[str] = {
    "edit_fragment",
}

# suggest_edits ranks candidate edit_fragment arguments toward the property box.
# It is emitted immediately BEFORE the edit it informs, does NOT open a new round
# (produces no molecule), and rides on the segment of the edit that follows it.
SUGGEST_TOOLS: set[str] = {
    "suggest_edits",
}

# Tools that verify / inspect the current candidate.
_VERIFY_SOLO_TOOLS: set[str] = {
    "match_substructure",
    "analyze_properties",
}

# The parallel verification checkpoint pair.
_CHECKPOINT_PAIR: set[str] = {
    "match_substructure",
    "analyze_properties",
}


# ── data model ────────────────────────────────────────────────────────────


@dataclass
class Round:
    """One molecule state in the build trajectory (seed = round 0)."""

    index: int
    tool: str                              # "seed" for round 0, else edit tool name
    args: dict                             # edit args (empty for seed)
    smiles: str
    properties: dict = field(default_factory=dict)
    # Substructure check: None until match_substructure has run on this round.
    substructure_match: Optional[bool] = None
    labeled_atoms: str = ""
    prop_strict: dict = field(default_factory=dict)
    # Brief reason for the edit (populated by generator, shown in memory).
    tool_reason: str = ""
    # The mmpdb Δ this round's edit was COMMITTED on, {property: avg}. Filled from the
    # round's own suggest_edits response. Segmentation is why this has to be stored:
    # each segment is its own training example, so the candidate JSON of round N-1 is
    # not in round N's conversation, and without this the previous prediction is a
    # number the model has no way to obtain at inference — exactly the shape that
    # produces fabricated values.
    pred_delta: dict = field(default_factory=dict)
    # Tool-error message when this round's edit returned an error instead of a
    # molecule (a genuinely failed edit). When set, the molecule is unchanged
    # (smiles == previous round's) and the round is recorded in ## Progress So Far
    # as a failed attempt only. Kept so the eval harness can render a naturally
    # failing edit consistently (see agentic_eval._ingest_tool_result).
    error: str = ""


@dataclass
class Segment:
    """One training example scope: (suggest_edits lead-in +) edit_i + every tail
    step up to the next edit's lead-in."""

    segment_index: int                     # 1..N (matches the round being produced)
    is_final: bool                         # last edit in a satisfied chain
    prior_rounds: list[Round]              # rounds 0..segment_index-1
    edit_step: ToolChainStep               # the edit_fragment tool_call
    tail_steps: list[ToolChainStep]        # all steps after the edit, before next lead-in
    result_round: Round                    # the round produced by this edit
    # suggest_edits step(s) that immediately precede — and inform — this edit.
    lead_steps: list[ToolChainStep] = field(default_factory=list)


@dataclass
class SeedSegment:
    """Initial trajectory before any edit: characterises round 0 (the scaffold).

    Covers the pre-edit portion of the tool chain — in the direct-scaffold-seed
    format this is the single seed checkpoint ``match_substructure`` ∥
    ``analyze_properties`` ∥ ``label_atom_indices`` run on the complete
    scaffold. Has no incoming memory block; ends right after the seed checkpoint's
    tool responses (no <ANSWER>).
    """

    task_type: str
    steps: list[ToolChainStep]             # the pre-edit steps, in chain order
    seed_round: Round                      # round 0


# ── helpers ────────────────────────────────────────────────────────────────


def _safe_json(s: Optional[str]):
    """Return parsed JSON or the raw string. Tool responses may be quote-wrapped."""
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
            try:
                return json.loads(s)
            except Exception:
                return s[1:-1]
        return s


def _step_call_names(step: ToolChainStep) -> set[str]:
    """All tool names issued in *step* (primary ``tool_call`` + parallel ones)."""
    names = {step.tool_call.name}
    names.update(c.name for c in step.parallel_tool_calls)
    return names


def _is_checkpoint_parallel(step: ToolChainStep) -> bool:
    """True for the parallel verification checkpoint.

    Handles the 3-way ``match_substructure`` ∥ ``analyze_properties`` ∥
    ``label_atom_indices`` form (label folded in), the property-only
    ``analyze_properties`` ∥ ``label_atom_indices`` form, and the legacy
    2-way ``match_substructure`` ∥ ``analyze_properties`` form: any
    parallel step that carries a verification tool (match / analyze).
    """
    if not step.parallel_tool_calls:
        return False
    return bool(_CHECKPOINT_PAIR & _step_call_names(step))


def _is_real_edit(step: ToolChainStep) -> bool:
    """True for a molecule-modifying tool call (never a parallel checkpoint)."""
    if step.parallel_tool_calls:
        return False
    return step.tool_call.name in EDIT_TOOLS


def _is_suggest(step: ToolChainStep) -> bool:
    """True for a suggest_edits call (the lead-in that precedes an edit)."""
    if step.parallel_tool_calls:
        return False
    return step.tool_call.name in SUGGEST_TOOLS


# Substrings that mark a tool ERROR response (not a molecule). Covers the tool
# server's error strings: "Execution Error:", "SMILES Syntax Error:", "Input
# Argument Error:", "Query Syntax Error:", "Cannot attach …", "out of range", etc.
_ERROR_MARKERS = (
    "error:", "error ", "cannot ", "out of range", "not a valid", "invalid ",
    "unsupported", "unable", "failed", "no result",
)


def _is_error_response(text: Optional[str]) -> bool:
    """True if a tool response is an error string rather than a molecule."""
    if not text or not isinstance(text, str):
        return False
    low = text.strip().lower()
    return any(m in low for m in _ERROR_MARKERS)


def _looks_like_smiles(text: Optional[str]) -> bool:
    """Heuristic: a non-empty, non-error string we can treat as a molecule."""
    if not text or not isinstance(text, str):
        return False
    t = text.strip()
    if not t or t in ("C", "[H][H]"):
        return False
    if _is_error_response(t):
        return False
    return True


def _extract_checkpoint(step: ToolChainStep) -> tuple[dict, Optional[bool], str]:
    """Return (properties, substructure_match, labeled_atoms) from a checkpoint.

    Handles the 3-way ``match_substructure`` ∥ ``analyze_properties`` ∥
    ``label_atom_indices`` checkpoint (label folded in), the property-only /
    legacy 2-way forms, and the solo verify tools. Every response is keyed by its
    tool name across the primary ``tool_call`` and all parallel companions.
    """
    props: dict = {}
    match: Optional[bool] = None
    labeled: str = ""

    resp_by_name: dict[str, Optional[str]] = {
        step.tool_call.name: step.expected_response,
    }
    for c, r in zip(step.parallel_tool_calls, step.parallel_expected_responses):
        resp_by_name[c.name] = r

    raw_props = _safe_json(resp_by_name.get("analyze_properties"))
    if isinstance(raw_props, dict):
        props = {k: v for k, v in raw_props.items() if isinstance(v, (int, float, bool)) and not isinstance(v, bool)}

    raw_match = _safe_json(resp_by_name.get("match_substructure"))
    if isinstance(raw_match, dict) and "match" in raw_match:
        match = bool(raw_match["match"])

    raw_label = _safe_json(resp_by_name.get("label_atom_indices"))
    if isinstance(raw_label, str):
        labeled = raw_label
    return props, match, labeled


def _verify_mol_smiles(step: ToolChainStep) -> Optional[str]:
    """The ``mol_smiles`` a verify/checkpoint step was run on (None if absent)."""
    for call in [step.tool_call, *step.parallel_tool_calls]:
        ms = call.arguments.get("mol_smiles")
        if isinstance(ms, str) and ms:
            return ms
    return None


def _is_verify_step(step: ToolChainStep) -> bool:
    """True for a match_substructure / analyze checkpoint (parallel or solo)."""
    return _is_checkpoint_parallel(step) or (
        not step.parallel_tool_calls and step.tool_call.name in _VERIFY_SOLO_TOOLS
    )


def _seed_smiles(input_data: GeneratorInput) -> str:
    """Resolve the starting-core SMILES for round 0."""
    meta = input_data.metadata or {}
    seed = meta.get("seed_smiles")
    if isinstance(seed, str) and seed:
        return seed
    if input_data.tool_chain:
        arg = input_data.tool_chain[0].tool_call.arguments.get("mol_smiles")
        if isinstance(arg, str) and arg:
            return arg
    return meta.get("ref_smiles", "") or ""


# ── main entry points ─────────────────────────────────────────────────────


def extract_rounds(input_data: GeneratorInput) -> list[Round]:
    """Walk the tool chain and build a list of Round objects.

    Round 0 is the seed/core; each subsequent round corresponds to one
    molecule-editing tool call.  Verification checkpoints and labels fill in the
    measured properties / substructure-match / labelled atoms on the current
    round.  Returns an empty list when no usable state emerges.
    """
    steps = list(input_data.tool_chain)
    meta = input_data.metadata or {}

    rounds: list[Round] = []
    seed = _seed_smiles(input_data)
    current: Optional[Round] = (
        Round(index=0, tool="seed", args={}, smiles=seed) if seed else None
    )

    def _finalize(round_obj: Round) -> None:
        rounds.append(round_obj)

    # Stamp the metadata-level strict results onto whichever round carries the
    # final predicted molecule (the only step with per-constraint results).
    pred = meta.get("predicted_molecule") or input_data.ground_truth_molecule or ""
    prop_strict_final = meta.get("prop_constraint_results_strict") or {}

    for step in steps:
        name = step.tool_call.name

        if _is_real_edit(step):
            if current is not None:
                _finalize(current)
            new_smiles = _safe_json(step.expected_response)
            if not isinstance(new_smiles, str):
                new_smiles = ""
            edit_error = ""
            if not _looks_like_smiles(new_smiles):
                # A FAILED edit (tool returned an error, not a molecule). The
                # candidate is unchanged: keep the prior SMILES and record the
                # error on the round so it shows in ## Progress So Far as a failed
                # attempt.
                edit_error = new_smiles.strip() if isinstance(new_smiles, str) else str(new_smiles)
                new_smiles = rounds[-1].smiles if rounds else new_smiles
            current = Round(
                index=len(rounds),
                tool=name,
                args=dict(step.tool_call.arguments),
                smiles=new_smiles,
                error=edit_error,
            )

        elif current is None:
            continue

        elif _is_checkpoint_parallel(step) or name in _VERIFY_SOLO_TOOLS:
            # Only absorb the checkpoint when it was actually run on the current
            # molecule.  Some chains begin with a checkpoint that previews the
            # final/target molecule before any edit — that data must NOT be
            # attributed to the seed (or any earlier round).
            vm = _verify_mol_smiles(step)
            if vm is None or vm.strip() == current.smiles.strip():
                props, match, labeled = _extract_checkpoint(step)
                if props:
                    current.properties = props
                if match is not None:
                    current.substructure_match = match
                # The checkpoint's folded-in label_atom_indices grounds the NEXT
                # edit's atom_index (the label that used to be a separate pre-edit
                # step now rides on this checkpoint).
                if labeled:
                    current.labeled_atoms = labeled

        elif name == "label_atom_indices":
            # Legacy standalone label step (pre-3-way-checkpoint chains).
            labeled = _safe_json(step.expected_response)
            if isinstance(labeled, str):
                current.labeled_atoms = labeled

        # else: ignore unrecognised inspection tools

    if current is not None:
        _finalize(current)

    # Stamp the per-constraint strict results onto the round holding the final
    # predicted molecule (used by the constraint check / is_fully_satisfied).
    if pred and prop_strict_final:
        for r in rounds:
            if r.smiles == pred:
                r.prop_strict = dict(prop_strict_final)

    return rounds


def build_seed_segment(input_data: GeneratorInput) -> Optional[SeedSegment]:
    """Return the pre-edit core trajectory as a SeedSegment, or None.

    With the direct-scaffold-seed format the seed IS the complete scaffold, so a
    chain may have NO edits at all (scaffold already satisfies every target). In
    that case the whole chain — label(scaffold) + its match||analyze checkpoint —
    is the seed segment; the answer is emitted by a separate terminal segment.
    """
    steps = list(input_data.tool_chain)
    edit_indices = [i for i, s in enumerate(steps) if _is_real_edit(s)]
    # Boundary of the seed's pre-edit region: the first edit, or (no edits) the
    # whole chain.
    boundary = edit_indices[0] if edit_indices else len(steps)

    # The seed segment introduces + labels the starting core, and may include a
    # baseline checkpoint that measures the SEED itself.  Some chains instead
    # open with a checkpoint that previews the FINAL/target molecule before any
    # edit — that one references a molecule the assistant hasn't built yet, so
    # it must be dropped (and ``extract_rounds`` already refuses to attribute
    # its data to the seed).  Keep only verify steps run on the seed.
    seed = _seed_smiles(input_data)
    kept = [
        s for s in steps[:boundary]
        if not _is_suggest(s)                       # first edit's lead-in → its segment
        and (not _is_verify_step(s)
             or (_verify_mol_smiles(s) or "").strip() == seed.strip())
    ]
    # Order the label(s) before the baseline checkpoint so the seed-introduction
    # reasoning ("…I'll label its atoms") flows into the label call, then the
    # checkpoint verifies/measures the freshly introduced core.
    non_ckpt = [s for s in kept if not _is_verify_step(s)]
    ckpt = [s for s in kept if _is_verify_step(s)]
    pre_edit_steps = non_ckpt + ckpt
    if not pre_edit_steps:
        return None

    rounds = extract_rounds(input_data)
    if not rounds:
        return None

    meta = input_data.metadata or {}
    task_type = (meta.get("task_type") or "").lower() or "generation"

    return SeedSegment(
        task_type=task_type,
        steps=pre_edit_steps,
        seed_round=rounds[0],
    )


def _stamp_pred_delta(result_round, edit_step, lead_steps) -> None:
    """Record, on *result_round*, the Δ the committed edit was predicted to make."""
    if result_round is None or edit_step is None or not lead_steps:
        return
    args = getattr(edit_step.tool_call, "arguments", None) or {}
    frm, to = args.get("from_smiles"), args.get("to_smiles")
    if frm is None and to is None:
        return
    for lead in lead_steps:
        parsed = _safe_json(lead.expected_response)
        if not isinstance(parsed, list):
            continue
        for c in parsed:
            if not isinstance(c, dict):
                continue
            if c.get("from_smiles") == frm and c.get("to_smiles") == to:
                result_round.pred_delta = {
                    k: (v or {}).get("avg")
                    for k, v in (c.get("delta") or {}).items()
                    if isinstance(v, dict) and v.get("avg") is not None
                }
                return


def build_segments(input_data: GeneratorInput) -> list[Segment]:
    """Split a constructive tool chain into edit-centric Segments.

    Returns [] when there are no edit steps (caller should fall back to the
    single-example generator path).  The final segment in a chain whose last
    round satisfies every constraint is flagged ``is_final`` (emits <ANSWER>);
    all others are intermediate and stop after their last tool response.
    """
    steps = list(input_data.tool_chain)
    rounds = extract_rounds(input_data)

    edit_indices = [i for i, s in enumerate(steps) if _is_real_edit(s)]
    if not edit_indices or len(rounds) <= 1:
        return []

    meta = input_data.metadata or {}
    target_properties = meta.get("target_properties") or {}
    require_substructure = bool(meta.get("smarts") or meta.get("substructure_match"))

    from .constraint_state import is_fully_satisfied  # local import to avoid circular

    last_round = rounds[-1]
    strictly_satisfied = is_fully_satisfied(
        last_round, target_properties, require_substructure
    )

    segments: list[Segment] = []
    n_edits = len(edit_indices)
    for seg_pos, edit_idx in enumerate(edit_indices):
        result_idx = seg_pos + 1
        if result_idx >= len(rounds):
            break
        edit_step = steps[edit_idx]
        # Lead-in: the suggest_edits call(s) between the previous edit and this one
        # (they rank the candidates this edit picks from). Ride on THIS segment.
        lower = edit_indices[seg_pos - 1] + 1 if seg_pos > 0 else 0
        lead_steps = [steps[j] for j in range(lower, edit_idx) if _is_suggest(steps[j])]
        next_edit = edit_indices[seg_pos + 1] if seg_pos + 1 < n_edits else len(steps)
        # Tail: steps after the edit up to the next edit, EXCLUDING that next edit's
        # suggest_edits lead-in (which belongs to the next segment).
        tail_steps = [s for s in steps[edit_idx + 1 : next_edit] if not _is_suggest(s)]

        is_last = seg_pos == n_edits - 1
        is_final = is_last and strictly_satisfied
        if is_final:
            tail_steps = _trim_trailing_non_verify(tail_steps)

        # Stamp the committed candidate's predicted Δ onto the round it produced, so
        # the NEXT segment's memory can put it next to the measured change.
        _stamp_pred_delta(rounds[result_idx], edit_step, lead_steps)
        segments.append(
            Segment(
                segment_index=seg_pos + 1,
                is_final=is_final,
                prior_rounds=rounds[:result_idx],
                edit_step=edit_step,
                tail_steps=tail_steps,
                result_round=rounds[result_idx],
                lead_steps=lead_steps,
            )
        )
    return segments


def _trim_trailing_non_verify(tail: list[ToolChainStep]) -> list[ToolChainStep]:
    """Drop trailing label steps after the last verification checkpoint.

    The final assistant message should issue <ANSWER> right after the final
    substructure/property checkpoint, so any ``label_atom_indices`` that would
    follow is elided.
    """
    out = list(tail)
    while out:
        last = out[-1]
        if _is_checkpoint_parallel(last):
            break
        if not last.parallel_tool_calls and last.tool_call.name in _VERIFY_SOLO_TOOLS:
            break
        out.pop()
    return out
