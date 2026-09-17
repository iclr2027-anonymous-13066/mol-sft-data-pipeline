"""Generator – turns a constructive ground-truth tool chain into SFT data.

The chains are in the direct-scaffold-seed format: they start from the COMPLETE
scaffold and tune it. For each input the chain is split (by
``segmenter.build_segments``) into:

  * a **seed segment** — the first reasoning derives the required-substructure
    SMARTS from the prompt AND writes a concrete scaffold SMILES containing it,
    then issues the 3-way seed checkpoint ``match_substructure`` ∥
    ``analyze_properties`` ∥ ``label_atom_indices`` on that scaffold, and
  * one **edit segment per structural edit** — apply the edit (attach / replace /
    set stereochemistry), then run the SAME 3-way checkpoint on the result (its
    ``label_atom_indices`` companion grounds the NEXT edit's atom_index).

Each segment becomes one ``TrainingExample``, and the segments of one chain are
merged into a single conversation: the system and user turns are taken once and
every segment contributes its assistant/tool turns in order.  A terminal segment
emits ``<ANSWER>`` (no tool call) once the final round's own measurements satisfy
every constraint.
"""

from __future__ import annotations

import json
import logging
import random
import re as _re
from typing import Optional

try:
    from rdkit import Chem as _Chem
    from rdkit import RDLogger as _RDLogger

    _RDLogger.DisableLog("rdApp.*")
except Exception:  # pragma: no cover - rdkit expected in the pipeline env
    _Chem = None

from .config import PipelineConfig
from .llm_client import LLMClient
from .constraint_state import (
    _compact_edit_args,
    build_constraint_check,
    is_fully_satisfied,
)
from .prompts import (
    CHECKPOINT_REASONING_PROMPT,
    DECORATE_EDIT_REASONING_PROMPT,
    EDIT_FRAGMENT_AUTHORED_PROMPT,
    EDIT_FRAGMENT_REASONING_PROMPT,
    SUGGEST_REASONING_PROMPT,
    REASONING_PROMPT,
    ROUND_REASONING_PROMPT,
    REFLECTION_PROMPT,
    FG_SEED_INTRO_PROMPT,
    SEED_INTRO_PROMPT,
    NAIVE_EDIT_REASONING_PROMPT,
    SYSTEM_PROMPT_TEMPLATE,
    TOOL_REASON_PROMPT,
)
from .fragment_names import (
    aromatic_h_note,
    ring_locant_note,
    ring_locant_errors,
    format_candidate_options,
    landing_safety,
    round_context,
    candidate_rank_note,
    picked_number,
    describe_anchor,
    describe_fragment,
    invented_chemical_names,
    inverted_spread_claims,
    landing_claim_errors,
    unknowable_numbers,
    spread_verdict,
)
from .schema import GeneratorInput, ToolCall, ToolStep, TrainingExample
from .seed_order import retranscribe_seed
from .segmenter import (
    EDIT_TOOLS as _EDIT_TOOLS,
    Segment,
    SeedSegment,
    _is_checkpoint_parallel,
    build_seed_segment,
    build_segments,
    extract_rounds,
)
from .tools.executor import ToolExecutor
from .tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


# ── helpers ─────────────────────────────────────────────────────────────────


def _looks_like_bare_smiles(line: str) -> bool:
    """True when *line* is nothing but a SMILES string.

    Used to drop a trailing SMILES the model sometimes appends on its own line:
    the seed reasoning re-states the scaffold it already gave inline (~9% of seed
    segments), and an edit reasoning occasionally leaks the resulting molecule
    even though the prompt forbids predicting it. Prose never legitimately ends
    with a bare SMILES, and the renderer emits every SMILES the turn needs.
    """
    s = line.strip().strip("`").rstrip(".")
    if not s or len(s) < 6 or _re.search(r"\s", s):
        return False
    if not _re.fullmatch(r"[A-Za-z0-9@+\-\[\]\(\)=#$%/\\\.:*]+", s):
        return False
    # Prose-like single words (no ring closure, branch, or second element) are
    # not treated as SMILES even if they happen to parse.
    if not _re.search(r"[0-9\(\)=#\[]", s):
        return False
    if _Chem is None:
        return False
    return _Chem.MolFromSmiles(s) is not None


def _strip_trailing_bare_smiles(text: str, seed_smiles: Optional[str] = None) -> str:
    """Drop trailing lines that are just a SMILES string (see above)."""
    if not text:
        return text
    lines = text.rstrip().split("\n")
    while len(lines) > 1:
        last = lines[-1].strip().strip("`")
        if not last:
            lines.pop()
            continue
        if (seed_smiles and last == seed_smiles.strip()) or _looks_like_bare_smiles(last):
            lines.pop()
            continue
        break
    return "\n".join(lines).rstrip()


def _suggest_candidates(response) -> list:
    """The candidate list a ``suggest_edits`` response carries (``[]`` if none).

    The response is a JSON array of candidate dicts; anything else (an error
    string, an empty array) means the round has no candidates to commit from.
    """
    if isinstance(response, list):
        return [c for c in response if isinstance(c, dict)]
    if not isinstance(response, str) or not response.strip():
        return []
    try:
        parsed = json.loads(response)
    except (ValueError, TypeError):
        return []
    if isinstance(parsed, list):
        return [c for c in parsed if isinstance(c, dict)]
    return []


def _committed_candidate(candidates: list, frm: str, to: str) -> Optional[dict]:
    """The candidate whose rule is the one the edit actually commits."""
    for c in candidates or []:
        if c.get("from_smiles") == frm and c.get("to_smiles") == to:
            return c
    return None


# Endpoints that come out of a trained model or a contribution table, never out of
# counting atoms: a Δ for these is NOT something the assistant can reproduce when
# suggest_edits gave it nothing, so the authored reasoning must never state one.
_NON_DERIVABLE = frozenset({
    "logP", "logD", "logS", "TPSA", "MR", "QED", "BBBP", "HIA", "Mutag",
})


def _authored_evidence(candidates: list, frm: str, to: str, props: Optional[dict],
                       targets: Optional[dict]) -> tuple:
    """Evidence blocks for the authored-edit prompt, split by REPRODUCIBILITY.

    The authored branch exists because at inference ``suggest_edits`` returns an
    empty list — so the assistant has no candidate table, and therefore no mmpdb Δ.
    Handing it every Δ during data generation taught it to quote numbers it cannot
    obtain at inference (measured: 5 of 6 sample spans cited a Δ for an endpoint
    like BBBP or Mutag). Withholding every Δ is worse — that is what made it invent
    mechanisms instead ("adding lipophilic bulk raises Mutag").

    So the Δ is split on whether the assistant could derive it unaided:

    * ``std == 0`` means the change was identical across every matched pair, i.e.
      it is fixed by the fragment alone — ``heavy_atoms`` for a known substituent,
      ``MW`` for a known atom swap. Countable at inference, so it may be quoted.
      This is self-validating: a property whose Δ genuinely depends on the
      attachment site carries a non-zero spread and drops out on its own.
    * everything else — and every endpoint in ``_NON_DERIVABLE`` regardless of its
      spread, since low support can zero a std spuriously — is reported as a
      DIRECTION only. The direction is available at inference: it is in the memory
      block the assistant is already reading.

    Returns ``(countable, direction_only, protect)``.
    """
    from .constraint_state import _fmt_val, _fmt_range, _value_in_range

    cand = _committed_candidate(candidates, frm, to)
    delta = (cand or {}).get("delta") or {}

    countable, direction, protect = [], [], []
    for name, d in delta.items():
        if not isinstance(d, dict):
            continue
        avg, std = d.get("avg"), d.get("std")
        cur = (props or {}).get(name)
        rng = (targets or {}).get(name)
        failing = rng is not None and cur is not None and \
            _value_in_range(cur, rng) is False
        derivable = (name not in _NON_DERIVABLE
                     and std is not None and abs(float(std)) < 1e-9)
        if derivable:
            after = None
            try:
                after = float(cur) + float(avg) if cur is not None else None
            except (TypeError, ValueError):
                pass
            land = f" → {_fmt_val(name, after)}" if after is not None else ""
            cur_s = f"{_fmt_val(name, cur)}" if cur is not None else "?"
            countable.append(
                f"  - {name}: {cur_s}{land}  (the fragment fixes this change at "
                f"{avg:+}, identical across every matched pair)")
        elif failing:
            direction.append(
                f"  - {name}: {_fmt_val(name, cur)}, must reach "
                f"{_fmt_range(rng, name)} — NOT predictable from the structure")
        elif rng is not None and cur is not None:
            protect.append(f"  - {name} = {_fmt_val(name, cur)} "
                           f"(target {_fmt_range(rng, name)})")

    return (
        "\n".join(countable) or "  (none — nothing about this edit is countable)",
        "\n".join(direction) or "  (none)",
        "\n".join(protect) or "  (none)",
    )


# Per-round nudges that break the authored branch out of a single template. A
# worked example in the prompt fixes the register but gets copied clause for clause
# — measured, the authored spans hit a 28.6% repeated-6-gram rate against 2.4% on
# the normal edit path in the same run. One generation cannot see what the other
# rounds wrote, so instead each round is deterministically handed a DIFFERENT way
# in and a different way out, and the template has nothing to converge on.
# (opening, closing, may_open_on_the_tool). Only ONE variant is allowed to lead with
# "suggest_edits returned …"; the rest must start on the chemistry, and that is
# ENFORCED below — left as a hint alone, 12 of 26 spans opened with the tool name
# regardless and the corpus kept one dominant formula.
_AUTHORED_VOICE = (
    ("Open on the property and its gap; mention the empty result second, in a "
     "subordinate clause.",
     "Close by saying you will measure it.", False),
    ("Open on the empty result, but phrase it your own way — not a stock formula.",
     "Close by naming what the checkpoint has to confirm.", True),
    ("Open on what the molecule structurally needs, then the number that says so.",
     "Close on the edit itself, with re-measuring implied by 'then check'.", False),
    ("Open on the decision now facing you with no ranked list to lean on.",
     "Close by saying where you expect it to land and that you will verify.", False),
    ("Open on the single constraint still standing between you and the answer.",
     "Close on the transformation and the measurement that follows it.", False),
)


def _authored_voice(key: str, segment_index: int) -> tuple:
    """(hint text, may_open_on_the_tool) for this round, deterministically."""
    import hashlib

    h = hashlib.md5(f"{key}|{segment_index}|voice".encode()).digest()
    opening, closing, may_lead = _AUTHORED_VOICE[h[0] % len(_AUTHORED_VOICE)]
    hint = f"- {opening}\n- {closing}"
    if not may_lead:
        hint += ("\n- Do NOT begin with the words \"suggest_edits\". The empty result "
                 "belongs later in the sentence, not at the front.")
    return hint, may_lead


def _should_author(cfg, task_id: str, segment_index: int, n_failing: int) -> bool:
    """Whether this decorate round is rendered as an AUTHORED edit (empty list).

    Deterministic in (task_id, segment_index) so `--resume` and the per-shard
    split reproduce the same selection — a random draw would re-roll every rerun
    and silently change the corpus. Rounds with more out-of-range properties than
    ``authored_max_failing`` are never selected: a real empty response fires at
    states with 1-2 failing properties, and training the branch on 5-failing
    rounds teaches a policy calibrated for the wrong risk regime.
    """
    frac = float(getattr(cfg, "authored_edit_fraction", 0.0) or 0.0)
    if frac <= 0.0:
        return False
    cap = int(getattr(cfg, "authored_max_failing", 2) or 2)
    if n_failing <= 0 or n_failing > cap:
        return False
    import hashlib

    h = hashlib.md5(f"{task_id}|{segment_index}|authored".encode()).digest()
    return (int.from_bytes(h[:4], "big") / 2 ** 32) < frac


def _smarts_ordered_smiles(scaffold_smiles: str, smarts: str) -> Optional[str]:
    """A SMILES of *scaffold_smiles* written in the SAME atom order as *smarts*.

    The canonical SMILES traverses atoms in a different order than the SMARTS
    lists them, so ``SMARTS → SMILES`` reads like a jump. Rewriting the scaffold
    rooted/ordered to follow the SMARTS makes it a near-direct transcription (e.g.
    SMARTS ``[#6]1:[#6]:...:[#6](-[#16](=[#8])(=[#8])-...)...`` → ordered SMILES
    ``c1ccc(S(=O)(=O)N...)cc1`` — benzene then sulfonyl, matching the SMARTS).

    Returns ``None`` when the SMARTS does not cover the whole scaffold or the
    reorder fails; the caller then falls back to the canonical SMILES.
    """
    if _Chem is None or not scaffold_smiles or not smarts:
        return None
    mol = _Chem.MolFromSmiles(scaffold_smiles)
    patt = _Chem.MolFromSmarts(smarts)
    if mol is None or patt is None:
        return None
    match = mol.GetSubstructMatch(patt)
    if not match or len(match) != mol.GetNumAtoms():
        return None
    try:
        ordered = _Chem.MolToSmiles(_Chem.RenumberAtoms(mol, list(match)), canonical=False)
    except Exception:
        return None
    chk = _Chem.MolFromSmiles(ordered)
    if chk is not None and _Chem.MolToSmiles(chk) == _Chem.MolToSmiles(mol):
        return ordered
    return None


def _element_at_atom_index(labeled_smiles: str, atom_index) -> Optional[str]:
    """Return the element symbol at *atom_index* parsed from a labeled SMILES."""
    if not labeled_smiles or atom_index is None:
        return None
    try:
        idx = int(atom_index)
    except (TypeError, ValueError):
        return None
    m = _re.search(rf"\[([^\[\]]+?):{idx}\]", labeled_smiles)
    if not m:
        return None
    token = m.group(1)
    mm = _re.match(r"([A-Za-z][a-z]?)", token)
    if not mm:
        return None
    sym = mm.group(1)
    if len(sym) == 1:
        sym = sym.upper()
    return sym


def _phase_note(phase: str) -> str:
    p = (phase or "").lower()
    if p == "scaffold":
        return "Building the core scaffold of the required substructure.\n"
    if p == "decorate":
        return "Decorating the completed scaffold to meet the property targets.\n"
    return ""


def _hub_context(meta: dict) -> str:
    """Describe the hub ring system so the seed choice reads as a derivation.

    The starting core is the *hub* of the described substructure — the ring
    system the other rings hang off through linkers. We surface its identity
    (from ``ring_systems``, ranked by connection degree, then ring count) plus
    the overall topology, so the seed-intro reasoning can point to the phrase in
    the description that names this central piece instead of asserting the core
    out of nowhere.
    """
    ring_systems = meta.get("ring_systems") or []
    connections = meta.get("connections") or []
    topology = (meta.get("topology_summary") or "").strip()

    hub_idx: Optional[int] = None
    if connections:
        degree: dict[int, int] = {}
        for c in connections:
            for endpoint in (c.get("a"), c.get("b")):
                if isinstance(endpoint, int):
                    degree[endpoint] = degree.get(endpoint, 0) + 1
        if degree:
            hub_idx = max(degree, key=lambda k: degree[k])
    if hub_idx is None and ring_systems:
        # Fallback: the most fused ring system is the most likely hub.
        hub_idx = max(
            range(len(ring_systems)),
            key=lambda i: ring_systems[i].get("n_rings", 1) or 1,
        )

    lines: list[str] = []
    if hub_idx is not None and hub_idx < len(ring_systems):
        hub = ring_systems[hub_idx]
        name = hub.get("name") or "the central ring system"
        carbonyl = (hub.get("carbonyl_desc") or "").strip()
        detail = f" — {carbonyl}" if carbonyl else ""
        lines.append(
            f"Central/hub ring system (the piece the other rings attach to): "
            f"{name}{detail}."
        )
        others = [
            rs.get("name") for i, rs in enumerate(ring_systems)
            if i != hub_idx and rs.get("name")
        ]
        if others:
            lines.append("Peripheral ring systems that hang off the hub: " + ", ".join(others) + ".")
    if topology:
        lines.append(f"Overall topology: {topology}")
    return "\n".join(lines) if lines else "(no structural breakdown available)"


def _last_prediction(prev_seg) -> dict:
    """The mmpdb Δ the PREVIOUS edit was committed on, ``{property: avg}``.

    `h_pred_error_last` needs the predicted change of the last step to compare against
    the measured one. The Δ lives in that step's ``suggest_edits`` response, which the
    round being generated no longer carries — but the previous segment does, and the
    generator walks the segments in order, so it is one lookback away.

    Empty when the previous segment had no candidate list (an authored edit, or a seed).
    """
    if prev_seg is None or not getattr(prev_seg, "lead_steps", None):
        return {}
    step = prev_seg.edit_step
    args = (step.tool_call.arguments or {}) if step is not None else {}
    frm, to = args.get("from_smiles"), args.get("to_smiles")
    for lead in prev_seg.lead_steps:
        cands = _suggest_candidates(lead.expected_response) or []
        for c in cands:
            if not isinstance(c, dict):
                continue
            if c.get("from_smiles") == frm and c.get("to_smiles") == to:
                return {k: (v or {}).get("avg")
                        for k, v in (c.get("delta") or {}).items()
                        if isinstance(v, dict) and v.get("avg") is not None}
    return {}


def _out_of_range(prior_round, target_properties: dict) -> list[str]:
    """Names of the constrained properties *prior_round* failed — nothing else.

    Fed to the checkpoint reasoning so it can anchor on what the edit it follows
    was aiming at. Names only, no values: at that point in the trajectory nothing
    has been re-measured yet, so a value in that sentence would be a claim about
    a measurement the assistant has not made.
    """
    from .constraint_state import _value_in_range

    if prior_round is None or not getattr(prior_round, "properties", None):
        return []
    props = prior_round.properties
    return [p for p, rng in (target_properties or {}).items()
            if p in props and _value_in_range(props.get(p), rng) is False]


def _property_gap(prior_round, target_properties: dict) -> Optional[str]:
    """Render the property state of *prior_round* vs targets for decorate steps.

    Returns a block that names every out-of-range property with its current
    value, target range and the direction it must move, plus a short note on
    which targets already hold. Returns None when there is nothing measured yet
    (so the caller can fall back to the substructure-framed prompt).
    """
    from .constraint_state import _fmt_range, _fmt_val, _value_in_range

    if prior_round is None or not getattr(prior_round, "properties", None):
        return None
    props = prior_round.properties

    failing: list[str] = []
    passing: list[str] = []
    for name, rng in (target_properties or {}).items():
        val = props.get(name)
        if val is None:
            continue
        ok = _value_in_range(val, rng)
        if ok is False:
            direction = ""
            try:
                lo = rng[0] if isinstance(rng, (list, tuple)) else rng
                hi = rng[1] if isinstance(rng, (list, tuple)) else rng
                if lo is not None and float(val) < float(lo):
                    direction = " → too low, needs to INCREASE"
                elif hi is not None and float(val) > float(hi):
                    direction = " → too high, needs to DECREASE"
            except (TypeError, ValueError, IndexError):
                direction = " → out of range"
            # pass `name`: without it _fmt_range prints an integer target as
            # "28.000", and the reasoning dutifully quotes "exactly 28.000 heavy atoms"
            failing.append(
                f"  - {name} = {_fmt_val(name, val)} (target {_fmt_range(rng, name)})"
                f"{direction}"
            )
        elif ok is True:
            passing.append(name)

    if not failing:
        return None

    out = ["The required substructure is already present. Out-of-range properties:"]
    out.extend(failing)
    if passing:
        out.append("Already within range: " + ", ".join(passing) + ".")
    return "\n".join(out)


class Generator:
    """Generates ``TrainingExample`` segments from a constructive tool chain."""

    def __init__(
        self,
        config: PipelineConfig,
        tool_registry: ToolRegistry,
        tool_executor: ToolExecutor,
    ) -> None:
        self.config = config
        self.llm = LLMClient(config.generator)
        self.registry = tool_registry
        self.executor = tool_executor
        self.reflection_prob = config.reflection_probability

    # -- public API ---------------------------------------------------------

    async def generate(self, input_data: GeneratorInput) -> Optional[TrainingExample]:
        """Single-example fallback for chains with no usable edit segments.

        Runs the chain in order, emitting <ANSWER> when the metadata says every
        constraint is satisfied, otherwise an intermediate segment.  The default
        path is :meth:`generate_segments`; this is only reached when the chain
        cannot be segmented.
        """
        retranscribe_seed(input_data)
        meta = input_data.metadata or {}
        tool_schemas = self.registry.get_schemas(input_data.tool_set)
        tool_descriptions = self.registry.get_tool_descriptions(input_data.tool_set)
        satisfied = bool(
            meta.get("all_constraints_strictly_satisfied")
            or meta.get("all_constraints_satisfied")
        )
        predicted = meta.get("predicted_molecule") or input_data.ground_truth_molecule

        ctx: list[str] = ["(start of conversation)"]
        tool_steps: list[ToolStep] = []
        for step in input_data.tool_chain:
            tc = step.tool_call
            reasoning = await self._reason_for_step(
                step, input_data.user_prompt, tool_descriptions,
                "\n".join(ctx), input_data,
            )
            resp, pcalls, presps = self._exec_step_calls(step)
            tool_steps.append(ToolStep(
                reasoning=reasoning, tool_call=tc, tool_response=resp,
                parallel_tool_calls=pcalls, parallel_tool_responses=presps,
            ))
            self._append_ctx(ctx, reasoning, tc, resp, pcalls, presps)

        if not tool_steps:
            return None
        # Rare degenerate fallback (segmentation unavailable). No Verification
        # block — the ANSWER decision lives in the terminal segment of the main
        # segmented path; this just emits the chain in one scope.
        return TrainingExample(
            system_prompt=SYSTEM_PROMPT_TEMPLATE,
            tool_schemas=tool_schemas,
            user_prompt=input_data.user_prompt,
            tool_steps=tool_steps,
            molecule_prediction=predicted,
            is_intermediate_segment=not satisfied,
        )

    async def generate_segments(
        self,
        input_data: GeneratorInput,
        include_rejected: bool = False,  # unused; kept for pipeline compat
    ) -> list[TrainingExample]:
        """Produce one TrainingExample per build segment.

        Layout: seed segment (derive the SMARTS + write the scaffold SMILES, then
        the 3-way seed checkpoint) → one intermediate segment per edit (edit +
        3-way checkpoint) → a terminal ANSWER segment that emits <ANSWER> once the
        final round's own measurements satisfy every constraint (no tool call, no
        Verification block).
        Chains with no edits (scaffold already satisfies) produce just seed +
        terminal.
        """
        # The seed the model must WRITE: stage 3 stores it canonicalised, which
        # turns the SMARTS read-off into a re-ordering puzzle. Rewrite the chain
        # so the seed follows the SMARTS' own atom order (no-op when it already
        # does, or when the chain cannot be rewritten safely).
        retranscribe_seed(input_data)
        meta = input_data.metadata or {}
        segments = build_segments(input_data)

        # A suggest_edits round that returned NO candidates cannot support the
        # trajectory it leads: the edit that follows has nothing to have been
        # chosen from, and any reasoning for it would be invented. Drop the whole
        # chain (measured at ~0.2% of chains).
        for _seg in segments:
            for _lead in _seg.lead_steps:
                if not _suggest_candidates(_lead.expected_response):
                    logger.info(
                        "Dropping chain — suggest_edits returned no candidates: %s",
                        (input_data.user_prompt or "")[:80],
                    )
                    return []

        tool_schemas = self.registry.get_schemas(input_data.tool_set)
        tool_descriptions = self.registry.get_tool_descriptions(input_data.tool_set)
        target_properties = meta.get("target_properties") or {}
        require_substructure = bool(meta.get("smarts") or meta.get("substructure_match"))
        substructure_description = (meta.get("description") or "").strip()
        # The single canonical machine-checkable target — same string every
        # ``match_substructure`` uses (builder prefers scaffold_smarts → smarts).
        # Pinned once in the seed segment's Target Requirements and carried forward.
        substructure_smarts = (
            meta.get("scaffold_smarts") or meta.get("smarts") or ""
        ).strip()
        # Functional-group instances (2b) carry one SMARTS per required group and
        # no scaffold. Their presence is what routes the seed segment to the FG prompt.
        fg_smarts = [s for s in (meta.get("fg_smarts") or []) if s]
        if not fg_smarts:
            fg_smarts = [m["smarts"] for m in (meta.get("fg_constraint") or [])
                         if isinstance(m, dict) and m.get("smarts")]
        if fg_smarts and not substructure_smarts:
            # Keep the memory block's SMARTS pin populated: it shows one canonical
            # target, so a multi-group constraint pins the first and the rest are
            # checked by their own match_substructure calls.
            substructure_smarts = fg_smarts[0]
        rounds = extract_rounds(input_data)
        if not rounds:
            return []
        final_round = rounds[-1]
        strictly_satisfied = is_fully_satisfied(
            final_round, target_properties, require_substructure
        )
        predicted = (
            meta.get("predicted_molecule")
            or input_data.ground_truth_molecule
            or final_round.smiles
        )
        seed_smiles = meta.get("seed_smiles") or rounds[0].smiles
        edit_phases = meta.get("edit_step_phases") or []
        task_type = (meta.get("task_type") or "generation").lower()
        # Stable per-chain key for the authored-edit draw (see _should_author).
        authored_key = str(meta.get("task_id") or input_data.user_prompt[:96])

        examples: list[TrainingExample] = []

        # ── Seed segment: introduce the core + label its atoms ─────────────
        seed_seg = build_seed_segment(input_data)
        seed_example: Optional[TrainingExample] = None
        if seed_seg is not None:
            seed_example = await self._build_seed_example(
                input_data, seed_seg, tool_schemas, tool_descriptions,
                substructure_description, seed_smiles, substructure_smarts,
                fg_smarts,
            )
            if seed_example is not None:
                examples.append(seed_example)

        # ── Edit segments ──────────────────────────────────────────────────
        # One continuous conversation: the history accumulates across rounds
        # instead of restarting per segment against a carried summary, so a span
        # prompt sees the earlier rounds themselves.
        ctx: list[str] = ["(the starting scaffold has been proposed and measured)"]
        for seg_i, seg in enumerate(segments):
            # Round context (`round_context`) needs the state one step FURTHER back
            # than this round: what the molecule measured before the previous edit,
            # and what that edit's Δ had predicted. Both are one lookback away.
            _prev_seg = segments[seg_i - 1] if seg_i > 0 else None
            round_history = {
                "prev_props": (seg.prior_rounds[-2].properties
                               if len(seg.prior_rounds) >= 2 else None),
                "last_pred": _last_prediction(_prev_seg),
            }
            # All four reasoning spans of this round come from ONE call when the
            # round is the standard suggest_edits → edit_fragment pair (the whole
            # 5-tool format). Anything unusual falls through to the four
            # single-purpose prompts below, and so does any span the merged call
            # failed to emit.
            spans: dict = {}
            if self.config.merge_round_reasoning \
                    and not self.config.naive_reasoning \
                    and seg.edit_step is not None \
                    and seg.edit_step.tool_call.name == "edit_fragment" \
                    and seg.lead_steps:
                lead0 = seg.lead_steps[0]
                cands = _suggest_candidates(
                    self.executor.execute(
                        lead0.tool_call, expected_response=lead0.expected_response)
                ) or []
                ck = " ∥ ".join(
                    c.name for s in seg.tail_steps
                    for c in ([s.tool_call] + list(s.parallel_tool_calls or []))
                    if c is not None
                )
                spans = await self._reason_round(
                    round_history=round_history,
                    user_prompt=input_data.user_prompt,
                    edit_tool_call=seg.edit_step.tool_call,
                    prior_round=(seg.prior_rounds[-1] if seg.prior_rounds else None),
                    result_round=seg.result_round,
                    target_properties=target_properties,
                    candidates=cands,
                    checkpoint_tools=ck,
                )
                # The round's intent line is consumed downstream (the edit-argument
                # rendering), so it has to be in place before it.
                if spans.get("note") and not seg.result_round.tool_reason \
                        and seg.result_round.index != 0:
                    seg.result_round.tool_reason = spans["note"]

            await self._populate_tool_reasons(
                seg.prior_rounds + [seg.result_round],
                user_prompt=input_data.user_prompt,
                target_properties=target_properties,
            )
            prior_round = seg.prior_rounds[-1] if seg.prior_rounds else None
            tool_steps: list[ToolStep] = []

            # 0) suggest_edits lead-in(s): rank candidate edits toward the box, so
            #    the edit_fragment that follows commits one of these candidates.
            #
            #    AUTHORED branch: for a deterministic fraction of rounds the
            #    response is rewritten to `[]` and the edit reasoning is derived
            #    without a list, so the model learns what to do when the tool
            #    returns nothing (it does, on 64-80% of calls at inference).
            #    The tool CALLS are untouched — only the rendered response and the
            #    edit reasoning change.
            n_failing = len(_out_of_range(prior_round, target_properties or {}))
            authored = bool(seg.lead_steps) and _should_author(
                self.config, authored_key, seg.segment_index, n_failing)
            candidates: list = []
            true_candidates: list = []
            lean = self.config.lean_reasoning
            for lead_i, lead in enumerate(seg.lead_steps):
                lead_tc = lead.tool_call
                # LEAN: an edit round always opens with suggest_edits, so a span
                # explaining that it is about to is a paraphrase of the format.
                lead_reasoning = "" if lean else (
                    spans.get("suggest") if lead_i == 0 else ""
                ) or await self._reason_suggest(
                    user_prompt=input_data.user_prompt,
                    prior_round=prior_round,
                    target_properties=target_properties,
                )
                lead_resp = self.executor.execute(
                    lead_tc, expected_response=lead.expected_response
                )
                true_candidates = _suggest_candidates(lead_resp) or true_candidates
                if authored:
                    lead_resp = "[]"          # what the model will see at inference
                else:
                    candidates = _suggest_candidates(lead_resp) or candidates
                tool_steps.append(ToolStep(
                    reasoning=lead_reasoning, tool_call=lead_tc, tool_response=lead_resp,
                ))
                self._append_ctx(ctx, lead_reasoning, lead_tc, lead_resp)

            # 1) The structural edit. The candidate list is passed on so the
            #    reasoning can state the committed candidate's TRUE rank.
            edit_tc = seg.edit_step.tool_call
            phase = edit_phases[seg_i] if seg_i < len(edit_phases) else ""
            # An authored round must NOT reuse the merged-call span: that span was
            # written against the candidate table and opens with "Candidate #N".
            edit_reasoning = (None if authored else spans.get("edit")) or \
                await self._reason_edit(
                    round_history=round_history,
                    user_prompt=input_data.user_prompt,
                    substructure_description=substructure_description,
                    tool_call=edit_tc,
                    prior_round=prior_round,
                    phase=phase,
                    tool_descriptions=tool_descriptions,
                    history="\n".join(ctx),
                    target_properties=target_properties,
                    candidates=candidates,
                    authored=authored,
                    true_candidates=true_candidates,
                    voice=_authored_voice(authored_key, seg.segment_index),
                )
            edit_resp = self.executor.execute(
                edit_tc, expected_response=seg.edit_step.expected_response
            )
            tool_steps.append(ToolStep(
                reasoning=edit_reasoning, tool_call=edit_tc, tool_response=edit_resp,
            ))
            self._append_ctx(ctx, edit_reasoning, edit_tc, edit_resp)

            # 2) Tail steps (checkpoint / label) in chain order. The checkpoint
            #    reasoning gets a COMPACT context (the edit just applied + which
            #    properties it was aiming at) instead of the replayed conversation.
            ck_context = "\n".join([
                f"{edit_tc.name}({_compact_edit_args(edit_tc.name, edit_tc.arguments)})",
                f"Stated intent: {edit_reasoning}",
            ])
            ck_gaps = ", ".join(_out_of_range(prior_round, target_properties)) or ""
            for step_i, step in enumerate(seg.tail_steps):
                # LEAN: the verification checkpoint always follows the edit, so
                # announcing it decides nothing.
                await self._emit_tail_step(
                    step, input_data.user_prompt, tool_descriptions, ctx, tool_steps,
                    edit_context=ck_context, open_gaps=ck_gaps,
                    reasoning=("" if lean else (spans.get("check") if step_i == 0 else "")),
                    no_reasoning=lean,
                )

            # Every edit segment is intermediate: it contributes its tool turns
            # and nothing else. The ANSWER is emitted by the terminal segment
            # below, once the measured state shows every constraint satisfied.
            examples.append(TrainingExample(
                system_prompt=SYSTEM_PROMPT_TEMPLATE,
                tool_schemas=tool_schemas,
                user_prompt=input_data.user_prompt,
                tool_steps=tool_steps,
                molecule_prediction=seg.result_round.smiles,
                is_intermediate_segment=True,
            ))

        # ── Terminal ANSWER segment ────────────────────────────────────────
        # `strictly_satisfied` is read off the final round's own tool results, so
        # the decision to answer needs nothing carried: emits <ANSWER>, no tool
        # call, no Verification block. Also covers scaffold-only chains (no edit
        # segments): seed segment → terminal answer.
        if strictly_satisfied and examples:
            # LEAN: every constraint is already met and the conversation shows
            # the measurements that say so — a confirm sentence only restates
            # them, so the segment is just the <ANSWER>.
            confirm = "" if self.config.lean_reasoning else \
                await self._generate_answer_confirm(
                    user_prompt=input_data.user_prompt,
                    constraint_check=build_constraint_check(
                        final_round, target_properties,
                        require_substructure, substructure_description,
                    ),
                    molecule=predicted,
                )
            examples.append(TrainingExample(
                system_prompt=SYSTEM_PROMPT_TEMPLATE,
                tool_schemas=tool_schemas,
                user_prompt=input_data.user_prompt,
                tool_steps=[],
                final_verification=confirm,
                molecule_prediction=predicted,
                is_intermediate_segment=False,
            ))

        return examples

    # -- seed segment -------------------------------------------------------

    async def _build_seed_example(
        self,
        input_data: GeneratorInput,
        seed: SeedSegment,
        tool_schemas: list[dict],
        tool_descriptions: str,
        substructure_description: str,
        seed_smiles: str,
        substructure_smarts: str = "",
        fg_smarts: Optional[list] = None,
    ) -> Optional[TrainingExample]:
        ctx: list[str] = ["(start of conversation — proposing a starting scaffold)"]
        tool_steps: list[ToolStep] = []
        for i, step in enumerate(seed.steps):
            tc = step.tool_call
            # The FIRST seed step (the seed checkpoint in the direct-scaffold-seed
            # format — match ∥ analyze ∥ label on the complete scaffold) carries the
            # seed-introduction reasoning: it derives the SMARTS from the prompt and
            # WRITES the scaffold SMILES it will verify. (Legacy chains open with a
            # solo label; it takes the same seed-intro reasoning.)
            if i == 0:
                reasoning = await self._reason_seed_intro(
                    input_data.user_prompt, substructure_description, seed_smiles,
                    _hub_context(input_data.metadata or {}), substructure_smarts,
                    fg_smarts,
                )
            else:
                reasoning = await self._reason_for_step(
                    step, input_data.user_prompt, tool_descriptions,
                    "\n".join(ctx), input_data,
                )
            resp, pcalls, presps = self._exec_step_calls(step)
            tool_steps.append(ToolStep(
                reasoning=reasoning, tool_call=tc, tool_response=resp,
                parallel_tool_calls=pcalls, parallel_tool_responses=presps,
            ))
            self._append_ctx(ctx, reasoning, tc, resp, pcalls, presps)

        if not tool_steps:
            return None
        return TrainingExample(
            system_prompt=SYSTEM_PROMPT_TEMPLATE,
            tool_schemas=tool_schemas,
            user_prompt=input_data.user_prompt,
            tool_steps=tool_steps,
            molecule_prediction=seed.seed_round.smiles,
            is_intermediate_segment=True,
        )

    # -- tail-step dispatcher ----------------------------------------------

    async def _emit_tail_step(
        self, step, user_prompt, tool_descriptions, ctx, tool_steps,
        edit_context: str = "", open_gaps: str = "", reasoning: str = "",
        no_reasoning: bool = False,
    ) -> None:
        """``no_reasoning`` emits the step with EMPTY content — distinct from an
        empty ``reasoning`` argument, which means "none supplied, generate one"."""
        tc = step.tool_call
        if not no_reasoning:
            reasoning = reasoning or await self._reason_for_step(
                step, user_prompt, tool_descriptions, "\n".join(ctx), None,
                edit_context=edit_context, open_gaps=open_gaps,
            )
        resp, pcalls, presps = self._exec_step_calls(step)
        # Optional reflection on a SOLO inspection result (never on a checkpoint).
        # Suppressed with the reasoning: a reflection with no reasoning around it
        # would be the only prose in a turn that is supposed to carry none.
        reflection = None
        if not no_reasoning and not pcalls and random.random() < self.reflection_prob:
            reflection = await self._generate_reflection(
                user_prompt, "\n".join(ctx), tc, resp,
            )
        tool_steps.append(ToolStep(
            reasoning=reasoning, tool_call=tc, tool_response=resp,
            parallel_tool_calls=pcalls, parallel_tool_responses=presps,
            reflection=reflection,
        ))
        self._append_ctx(ctx, reasoning, tc, resp, pcalls, presps)

    async def _reason_for_step(
        self, step, user_prompt, tool_descriptions, history, input_data,
        edit_context: str = "", open_gaps: str = "",
    ) -> str:
        """Reasoning for a non-edit step (checkpoint / label / solo verify)."""
        tc = step.tool_call

        if _is_checkpoint_parallel(step):
            # Frame the "verify + measure" reasoning around the analyze companion
            # (the label_atom_indices companion is incidental grounding).
            analyze = next(
                (c for c in step.parallel_tool_calls
                 if c.name == "analyze_properties"),
                step.parallel_tool_calls[0],
            )
            # The checkpoint sentence only needs to know what the edit it follows
            # was aiming at. Replaying the whole conversation here cost ~1,750
            # prompt tokens per round (a raw suggest_edits JSON dump) to produce a
            # ~20-token sentence; the compact edit context says the same thing.
            return await self._reason_checkpoint(
                user_prompt,
                edit_context or "\n".join(history.splitlines()[-2:]),
                open_gaps, tc, analyze,
            )

        name = tc.name
        if name == "label_atom_indices":
            return ("Labeling the atom indices so I can target a precise "
                    "position for the next attachment.")
        if name == "match_substructure":
            return "Confirming the required substructure is present in the current molecule."
        if name == "analyze_properties":
            return "Measuring the molecule's properties against the targets."

        # Generic fallback.
        prompt = REASONING_PROMPT.format(
            user_prompt=user_prompt,
            requirements_context="",
            conversation_history=history,
            tool_name=tc.name,
            tool_arguments=json.dumps(tc.arguments, indent=2),
            first_step_note="",
        )
        return await self._gen(prompt)

    # -- reasoning generators ----------------------------------------------

    async def _reason_seed_intro(
        self, user_prompt, substructure_description, seed_smiles, hub_context="",
        substructure_smarts="", fg_smarts=None,
    ) -> str:
        # #2/#3: describe the substructure high-level, write the FULL SMARTS in one
        # step, then transcribe it straight to the scaffold SMILES used by the tools
        # (no separate canonicalization step). ``hub_context`` is no longer used.
        #
        # Functional-group instances take a different prompt: their seed is CHOSEN,
        # not transcribed (see FG_SEED_INTRO_PROMPT), and there is one SMARTS per
        # required group instead of one for the whole scaffold.
        # NOTE naive_reasoning does NOT touch this span. The seed derives the SMARTS
        # and transcribes it to the scaffold SMILES, and it is the one span the eval
        # harness parses back — so it keeps its verified ring names, its aromatic-H
        # note and its retry. Measured with the naive prompt here instead: 17% of
        # seed spans turned into a visible scratchpad ("No, `c2nc...n2` is a
        # 6-membered ring... Or is it `N(O)`?"), the longest ran to 858 words, and
        # one assembled a molecule of its own with the double-bond geometry flipped
        # against the SMILES the tool call uses. The ablation is about which EDIT
        # RULE gets picked, so holding this span fixed is also the cleaner contrast.
        if fg_smarts:
            prompt = FG_SEED_INTRO_PROMPT.format(
                user_prompt=user_prompt,
                substructure_description=substructure_description or "(see the user query)",
                fg_smarts_block="\n".join(f"{i + 1}. {s}" for i, s in enumerate(fg_smarts)),
                seed_smiles=seed_smiles or "?",
                aromatic_h_note=aromatic_h_note(seed_smiles) or "(none)",
                ring_locant_note=ring_locant_note(seed_smiles) or "(no ring needs one)",
            )
            fallback = (
                f"The query calls for {'a specific functional group' if len(fg_smarts) == 1 else 'two functional groups'}, "
                f"which I write as {' and '.join(fg_smarts)} so each can be checked on its own. "
                f"{seed_smiles or 'The starting molecule'} already carries "
                f"{'it' if len(fg_smarts) == 1 else 'both'} and is small enough to build on, "
                f"so I take it as my starting point. Let me confirm the "
                f"{'match' if len(fg_smarts) == 1 else 'matches'}, measure its properties, "
                f"and label its atom indices."
            )
        else:
            prompt = SEED_INTRO_PROMPT.format(
                user_prompt=user_prompt,
                substructure_description=substructure_description or "(see the user query)",
                substructure_smarts=substructure_smarts or "(derive it from the description)",
                seed_smiles=seed_smiles or "?",
                aromatic_h_note=aromatic_h_note(seed_smiles) or "(none)",
                ring_locant_note=ring_locant_note(seed_smiles) or "(no ring needs one)",
            )
            fallback = (
                f"The requirement is a specific substructure, so I write it as the SMARTS "
                f"{substructure_smarts or 'shown'} to make it machine-checkable, then "
                f"transcribe that pattern directly to the SMILES {seed_smiles or 'shown'}, "
                f"which I'll use as my starting scaffold. Let me confirm it matches the "
                f"SMARTS, measure its properties, and label its atom indices."
            )
        # Locants are the one chemistry name the seed gets wrong often enough to matter
        # (35 of 559 measured, and a seed error rides into every later segment's memory
        # into every later round that quotes it), so the same quote-the-error-back retry the edit spans
        # use is applied here.
        text = (await self._gen(prompt)).strip()
        for _ in range(2):
            wrong = ring_locant_errors(text, seed_smiles)
            if not wrong:
                break
            note = ("\n\n## Your previous attempt was WRONG\n"
                    + "\n".join("- it " + w for w in wrong)
                    + "\nUse the ring names in the Verified ring names block exactly as "
                      "written, and do not guess a locant.")
            text = (await self._gen(prompt + note)).strip()
        return _strip_trailing_bare_smiles(text, seed_smiles) or fallback

    async def _reason_suggest(
        self, user_prompt, prior_round, target_properties=None,
    ) -> str:
        """Reasoning for the ``suggest_edits`` lead-in: name the out-of-range
        properties and say we'll ask the tool to rank candidate edits."""
        gap = _property_gap(prior_round, target_properties or {})
        prompt = SUGGEST_REASONING_PROMPT.format(
            user_prompt=user_prompt,
            property_gap=gap or "(some target properties are still out of range)",
        )
        return (await self._gen(prompt)).strip() or (
            "Some target properties are still out of range; rather than guess a "
            "substituent, I'll call suggest_edits with the property constraints to "
            "get candidate edits ranked by predicted gap-reduction."
        )

    # -- merged round reasoning ---------------------------------------------

    _SPAN_RE = _re.compile(
        r"^\[(SUGGEST|EDIT|CHECK|NOTE)\]\s*$(.*?)(?=^\[(?:SUGGEST|EDIT|CHECK|NOTE)\]\s*$|\Z)",
        _re.M | _re.S,
    )

    async def _reason_round(
        self, user_prompt, edit_tool_call, prior_round, result_round,
        target_properties, candidates, checkpoint_tools, round_history=None,
    ) -> dict:
        """All four reasoning spans of one edit round in ONE LLM call.

        The four prompts this replaces re-sent the same round from four angles —
        the user query four times, the RDKit fact block twice, the candidate table
        once in formatted form and once as raw JSON. Returns a dict with the keys
        ``suggest`` / ``edit`` / ``check`` / ``note``; a key is absent when the
        model did not emit that block, and the caller falls back to the
        single-purpose prompt for it.
        """
        from .constraint_state import _fmt_val

        args = edit_tool_call.arguments or {}
        frm = args.get("from_smiles") or "[*:1]"
        to = args.get("to_smiles") or "[*:1]"
        anch = args.get("anchors")
        if frm in ("[*:1]", "*"):
            action = f"attach {to}"
        elif to in ("[*:1][H]",):
            action = f"remove {frm}"
        else:
            action = f"swap {frm} → {to}"
        props = prior_round.properties if prior_round is not None else None
        labeled = prior_round.labeled_atoms if prior_round is not None else ""

        def _state(rnd) -> str:
            if rnd is None or not getattr(rnd, "properties", None):
                return "(not measured)"
            return ", ".join(f"{k}={_fmt_val(k, v)}" for k, v in rnd.properties.items())

        # the molecule being edited — `round_context` needs it to say whether the
        # edit SITE separates the candidates at all, and `landing_safety` to name it
        cur_smiles = (args.get("mol_smiles")
                      or (prior_round.smiles if prior_round is not None else None))
        prompt = ROUND_REASONING_PROMPT.format(
            user_prompt=user_prompt,
            property_gap=(_property_gap(prior_round, target_properties or {})
                          or "(bring the remaining out-of-range properties into range)"),
            candidate_options=format_candidate_options(
                candidates or [], props, target_properties or {}, frm, to,
                anchors=anch),
            round_context=round_context(
                props, target_properties or {}, candidates or [],
                mol_smiles=cur_smiles,
                prev_props=(round_history or {}).get("prev_props"),
                last_pred=(round_history or {}).get("last_pred")),
            landing_safety=landing_safety(
                candidates or [], props, target_properties or {}, frm, to,
                mol_smiles=cur_smiles, anchors=anch),
            spread_verdict=spread_verdict(
                candidates or [], props, target_properties or {}, frm, to,
                anchors=anch),
            action=action,
            from_smiles=frm,
            to_smiles=to,
            anchors=anch,
            candidate_rank=candidate_rank_note(candidates or [], frm, to, anch),
            picked_number=picked_number(candidates or [], frm, to, anch),
            labeled_atoms=labeled or "(not labeled)",
            from_desc=describe_fragment(frm),
            to_desc=describe_fragment(to),
            anchor_facts=describe_anchor(
                args.get("mol_smiles")
                or (prior_round.smiles if prior_round is not None else None),
                anch,
            ),
            prev_state=_state(prior_round),
            curr_state=_state(result_round),
            open_gaps=", ".join(_out_of_range(prior_round, target_properties or {}))
            or "(none were out of range — describe the structural change only)",
            checkpoint_tools=checkpoint_tools or "match_substructure ∥ analyze_properties",
        )

        text = await self.llm.generate(prompt)
        spans = self._split_spans(text)

        # Same two guards the single-purpose prompts carry, applied to the merged
        # output: a numerically inverted ± comparison in [EDIT], and a raw fragment
        # SMILES in [NOTE] (that line is replayed in the next segment's memory).
        block = landing_safety(candidates or [], props, target_properties or {},
                               frm, to, mol_smiles=cur_smiles, anchors=anch)
        for _ in range(2):
            wrong = inverted_spread_claims(spans.get("edit", ""))
            drift = landing_claim_errors(spans.get("edit", ""), block)
            leaks = "[*:" in spans.get("note", "")
            if not wrong and not drift and not leaks:
                break
            notes = ["\n\n## Your previous attempt was WRONG"]
            if wrong:
                notes.append(
                    "[EDIT] claimed: " + "; ".join(f'"{w}"' for w in wrong) + "\n"
                    "That comparison is numerically FALSE — a SMALLER ± is the tighter "
                    "one. Do not compare ± numbers yourself: state only what the Spread "
                    "Comparison block and the [..-tightest of N] tags say, or drop the "
                    "comparison and give the chemical reason instead."
                )
            if drift:
                notes.append(
                    "[EDIT] contradicts the Landing Safety block: "
                    + "; ".join(drift) + ".\nEvery count, tie verdict and candidate "
                    "number must be copied from that block, never re-derived."
                )
            if leaks:
                notes.append(
                    "[NOTE] wrote a raw fragment SMILES. Name the group in words "
                    "instead (use the Verified Chemistry Facts names), no SMILES at all."
                )
            notes.append("Rewrite all four blocks.\n\n"
                         "[SUGGEST]")
            retry = await self.llm.generate(prompt + "\n".join(notes))
            fixed = self._split_spans(
                retry if "[SUGGEST]" in retry else "[SUGGEST]\n" + retry)
            if fixed:
                spans = fixed
        return spans

    @classmethod
    def _split_spans(cls, text: str) -> dict:
        """Parse the four tagged blocks out of a merged-round generation."""
        out: dict[str, str] = {}
        for m in cls._SPAN_RE.finditer(text or ""):
            body = cls._clean_reasoning(m.group(2))
            if body:
                out[m.group(1).lower()] = body
        return out

    async def _reason_edit(
        self, user_prompt, substructure_description, tool_call,
        prior_round, phase, tool_descriptions, history, target_properties=None,
        candidates=None, authored=False, true_candidates=None, voice=None,
        round_history=None,
    ) -> str:
        labeled = prior_round.labeled_atoms if prior_round is not None else ""
        name = tool_call.name
        args = tool_call.arguments

        if authored and name == "edit_fragment":
            return await self._reason_edit_authored(
                user_prompt=user_prompt, args=args, prior_round=prior_round,
                target_properties=target_properties,
                true_candidates=true_candidates or [], labeled=labeled,
                voice=voice,
            )

        if name == "edit_fragment":
            frm = args.get("from_smiles") or "[*:1]"
            to = args.get("to_smiles") or "[*:1]"
            anch = args.get("anchors")
            if frm in ("[*:1]", "*"):
                action = f"attach {to}"
            elif to in ("[*:1][H]",):
                action = f"remove {frm}"
            else:
                action = f"swap {frm} → {to}"
            gap = _property_gap(prior_round, target_properties or {})
            # Hand the model the RDKit-verified vocabulary (group names, edit-site
            # environment), the committed candidate's true rank, and the FULL option
            # set with each candidate's predicted Δ ± std per constrained property —
            if self.config.naive_reasoning:
                # The raw tool JSON, nothing else, no guard. `_gen` already retries
                # on transport errors; there is no content check to retry on here.
                return (await self._gen(NAIVE_EDIT_REASONING_PROMPT.format(
                    user_prompt=user_prompt,
                    mol_smiles=(args.get("mol_smiles")
                                or (prior_round.smiles if prior_round is not None
                                    else "?")),
                    raw_props=json.dumps(
                        (prior_round.properties if prior_round is not None else {}),
                        ensure_ascii=False),
                    raw_candidates=json.dumps(candidates or [], ensure_ascii=False),
                    from_smiles=frm, to_smiles=to, anchors=anch,
                ))).strip()
            # so the reasoning can weigh the alternatives and arrive at the choice
            # instead of rationalising a decision it was simply handed.
            rank_note = candidate_rank_note(candidates or [], frm, to, anch)
            options = format_candidate_options(
                candidates or [],
                (prior_round.properties if prior_round is not None else None),
                target_properties or {}, frm, to, anchors=anch,
            )
            cur_smiles = (args.get("mol_smiles")
                          or (prior_round.smiles if prior_round is not None
                              else None))
            prompt = EDIT_FRAGMENT_REASONING_PROMPT.format(
                candidate_options=options,
                round_context=round_context(
                    (prior_round.properties if prior_round is not None else None),
                    target_properties or {}, candidates or [],
                    mol_smiles=cur_smiles,
                    prev_props=(round_history or {}).get("prev_props"),
                    last_pred=(round_history or {}).get("last_pred")),
                landing_safety=landing_safety(
                    candidates or [],
                    (prior_round.properties if prior_round is not None else None),
                    target_properties or {}, frm, to, mol_smiles=cur_smiles,
                    anchors=anch),
                spread_verdict=spread_verdict(
                    candidates or [],
                    (prior_round.properties if prior_round is not None else None),
                    target_properties or {}, frm, to, anchors=anch,
                ),
                user_prompt=user_prompt,
                property_gap=gap or "(bring the remaining out-of-range properties into range)",
                action=action,
                from_smiles=frm,
                to_smiles=to,
                anchors=anch,
                candidate_rank=rank_note,
                picked_number=picked_number(candidates or [], frm, to, anch),
                from_desc=describe_fragment(frm),
                to_desc=describe_fragment(to),
                anchor_facts=describe_anchor(
                    args.get("mol_smiles")
                    or (prior_round.smiles if prior_round is not None else None),
                    anch,
                ),
                labeled_atoms=labeled or "(not labeled)",
            )
            text = (await self._gen(prompt)).strip()
            # The model sometimes ignores the pre-computed tightness tags and
            # compares two ± numbers itself, backwards ("±0.14, tighter than
            # ±0.05"). Such a sentence would teach wrong arithmetic, so quote the
            # false claim back and regenerate rather than ship it.
            block = landing_safety(
                candidates or [],
                (prior_round.properties if prior_round is not None else None),
                target_properties or {}, frm, to, mol_smiles=cur_smiles,
                anchors=anch)
            for _ in range(2):
                wrong = inverted_spread_claims(text)
                drift = landing_claim_errors(text, block)
                if not wrong and not drift:
                    break
                note = "\n\n## Your previous attempt was WRONG\n"
                if wrong:
                    note += (
                        "It claimed: " + "; ".join(f'"{w}"' for w in wrong) + "\n"
                        "That comparison is numerically FALSE — a SMALLER ± is the "
                        "tighter one. Do not compare ± numbers yourself: state only "
                        "what the Spread Comparison block and the [..-tightest of N] "
                        "tags already say, or drop the comparison and give the "
                        "chemical reason instead.\n")
                if drift:
                    note += (
                        "It contradicts the Landing Safety block: " + "; ".join(drift)
                        + ".\nEvery count, tie verdict and candidate number must be "
                        "copied from that block, never re-derived. When the block calls "
                        "a criterion a tie, say it does not separate the candidates.\n")
                text = (await self._gen(
                    prompt + note + "Rewrite the 3 sentences.\n\n"
                    "Reasoning (3 sentences, ≤65 words):"
                )).strip() or text
            return text or (
                f"The properties still outside their targets need the shifts above, "
                f"and among the returned candidates this one moves them the right way "
                f"with the tightest spread, so I take it ({rank_note}) and {action} "
                f"at anchors {anch}."
            )

        # Edit path (`edit_fragment` — attach / swap / remove). In the
        # decorate/finalize phase edits are property-driven, so ground the
        # reasoning in the measured property gap when one is available.
        from .constraint_state import _compact_edit_args

        if (phase or "").lower() in ("decorate", "finalize"):
            gap = _property_gap(prior_round, target_properties or {})
            if gap is not None:
                prompt = DECORATE_EDIT_REASONING_PROMPT.format(
                    user_prompt=user_prompt,
                    property_gap=gap,
                    tool_name=name,
                    tool_arguments=_compact_edit_args(name, args),
                    labeled_atoms=labeled or "(not labeled)",
                )
                return (await self._gen(prompt)).strip() or (
                    f"The substructure is already present; I'll apply {name} to "
                    f"bring the out-of-range properties into their target ranges."
                )

        prompt = REASONING_PROMPT.format(
            user_prompt=user_prompt,
            requirements_context="",
            conversation_history=history,
            tool_name=name,
            tool_arguments=json.dumps(args, indent=2),
            first_step_note="",
        )
        return (await self._gen(prompt)).strip() or (
            f"Applying {name} to advance the build toward the required substructure."
        )

    async def _reason_checkpoint(
        self, user_prompt, edit_context, open_gaps, tc, ptc,
    ) -> str:
        prompt = CHECKPOINT_REASONING_PROMPT.format(
            user_prompt=user_prompt,
            edit_context=edit_context or "(the molecule was just modified)",
            open_gaps=open_gaps or "(none were recorded as out of range)",
            tool_name_1=tc.name,
            tool_name_2=ptc.name,
        )
        return (await self._gen(prompt)).strip() or (
            "Let me verify whether the required substructure is present and "
            "measure the current molecule's properties against the targets."
        )

    async def _reason_edit_authored(
        self, user_prompt, args, prior_round, target_properties,
        true_candidates, labeled, voice=None,
    ) -> str:
        """Edit reasoning for a round whose suggest_edits was rendered as ``[]``.

        Three defects were measured on this template and each is checked and
        regenerated rather than shipped: a reference to the candidate list that
        no longer exists (the whole point of the branch), a chemical name the
        round's own fact block never licensed (16% of generations — and wrong,
        not merely unsupported), and the 3-sentence / 70-word budget (16%).
        """
        frm = args.get("from_smiles") or "[*:1]"
        to = args.get("to_smiles") or "[*:1]"
        anch = args.get("anchors")
        if frm in ("[*:1]", "*"):
            action = f"attach {to}"
        elif to in ("[*:1][H]",):
            action = f"remove {frm}"
        else:
            action = f"swap {frm} → {to}"
        props = prior_round.properties if prior_round is not None else None
        mol = args.get("mol_smiles") or (
            prior_round.smiles if prior_round is not None else None)
        from_desc = describe_fragment(frm)
        to_desc = describe_fragment(to)
        anchor_facts = describe_anchor(mol, anch)
        facts = " ".join([from_desc, to_desc, anchor_facts, user_prompt or ""])

        voice_hint, may_lead_with_tool = voice or ("", True)
        countable, direction, protect = _authored_evidence(
            true_candidates, frm, to, props, target_properties or {})
        prompt = EDIT_FRAGMENT_AUTHORED_PROMPT.format(
            user_prompt=user_prompt,
            property_gap=(_property_gap(prior_round, target_properties or {})
                          or "(bring the remaining out-of-range properties into range)"),
            labeled_atoms=labeled or "(not labeled)",
            action=action,
            from_smiles=frm, to_smiles=to, anchors=anch,
            countable_block=countable, direction_block=direction,
            protect_block=protect, voice_hint=voice_hint,
            from_desc=from_desc, to_desc=to_desc, anchor_facts=anchor_facts,
        )
        # Values the span is allowed to quote: what the assistant has measured, plus
        # anything the fragment fixes arithmetically. A number outside this set is one
        # it could not have known when the tool returned nothing.
        allowed = (_property_gap(prior_round, target_properties or {}) or "") + \
            "\n" + countable + "\n" + direction + "\n" + protect
        text = (await self._gen(prompt)).strip()
        clean = False
        for _ in range(5):
            leak = _re.search(r"candidate|top-ranked|highest-ranked|predicted_gap|#\d",
                              text, _re.I)
            invented = invented_chemical_names(text, facts)
            unknowable = unknowable_numbers(text, allowed, _NON_DERIVABLE)
            # Register defects: the prompt's own scaffolding coming back out. These
            # read as the model reciting instructions rather than reasoning, and the
            # trained model would reproduce the tell verbatim.
            meta = _re.search(
                r"\bnon-?derivabl\w*|\bderivabl\w*|countable block|"
                r"predictable from the structure|\bmanually\b|\bunknowable\b",
                text, _re.I)
            # Strip attachment dummies first — "[*:1]OC" is a SMILES the span is
            # supposed to quote, not markdown emphasis.
            markup = _re.search(r"[`*]", _re.sub(r"\[\*:?\d*\]", "", text))
            # One opening formula dominated the corpus when this was only a hint.
            stock_open = (not may_lead_with_tool
                          and _re.match(r"""\s*['"`]?suggest_edits""", text, _re.I))
            n_words = len(text.split())
            n_sent = len([x for x in _re.split(r"(?<=[.!?])\s+", text.strip()) if x])
            fails = {
                "candidate-reference": bool(leak),
                "invented-name": bool(invented),
                "unknowable-number": bool(unknowable),
                "prompt-vocabulary": bool(meta),
                "markdown": bool(markup),
                "stock-opening": bool(stock_open),
                "over-80-words": n_words > 80,
                "over-3-sentences": n_sent > 3,
            }
            clean = not any(fails.values())
            if clean:
                break
            notes = ["\n\n## Your previous attempt was WRONG"]
            if meta:
                notes.append(
                    f'It wrote "{meta.group(0)}". That is this prompt\'s vocabulary for '
                    "how knowledge is organised, not something a chemist says. Put the "
                    "open question in the hedge instead — \"whether it carries X that "
                    "far is what the next checkpoint is for\".")
            if markup:
                notes.append(
                    "It used markdown (backticks or asterisks). Write plain prose; tool "
                    "and property names appear bare.")
            if stock_open:
                notes.append(
                    "It opened with \"suggest_edits\", which this round was told not to "
                    "do. Start on the chemistry — the property and its gap — and put the "
                    "empty result in a later clause.")
            if leak:
                notes.append(
                    f'It wrote "{leak.group(0)}". suggest_edits returned an EMPTY '
                    "list — there is no candidate, no rank and no predicted_gap to "
                    "refer to. Decide the edit from the constraint status you already "
                    "track and the arithmetic on the fragment.")
            if unknowable:
                notes.append(
                    "It stated " + ", ".join(f'"{u}"' for u in unknowable)
                    + ". The assistant CANNOT know that when suggest_edits returned "
                      "nothing — there is no Δ table to read it from, and it is not "
                      "countable from the fragment. Name the direction the property "
                      "must move and say the edit is a probe to be measured.")
            if invented:
                notes.append(
                    "It used chemical name(s) " + ", ".join(f'"{n}"' for n in invented)
                    + " that the Verified Chemistry Facts block does not state. Use "
                      "only the names given there, or the fragment SMILES.")
            if n_words > 80:
                notes.append(f"It ran to {n_words} words. Hard cap is 3 sentences, "
                             "70 words total.")
            if n_sent > 3:
                notes.append(f"It ran to {n_sent} sentences. Hard cap is 3.")
            notes.append("Rewrite the 3 sentences.\n\nReasoning (3 sentences, <=70 words):")
            retry = (await self._gen(prompt + "\n".join(notes))).strip()
            if retry:
                text = retry
        if clean and text:
            return text
        # Every retry still failed a check. Shipping the last attempt would put a
        # defect into training — measured, the spans that exhaust the retries carry
        # several at once (an invented acyl name AND this prompt's own vocabulary).
        # A plain sentence built from fields we know are true is worth more than a
        # fluent wrong one, and it is rare enough not to shape the corpus.
        logger.warning(
            "authored-edit span failed validation after retries (%s); falling back to "
            "the rule-based sentence for %s",
            ",".join(k for k, v in fails.items() if v) or "unknown", action)
        gap_names = ", ".join(_out_of_range(prior_round, target_properties or {})) \
            or "the remaining targets"
        # Respect this round's voice slot even here, so the fallback does not become
        # its own repeated opening formula in the corpus.
        if may_lead_with_tool:
            return (f"suggest_edits came back empty, so I choose the edit myself: "
                    f"{gap_names} still sits outside the target box. I {action} at "
                    f"anchors {anch} and re-measure.")
        return (f"{gap_names} still sits outside the target box and suggest_edits "
                f"offered nothing, so I choose the edit myself. I {action} at "
                f"anchors {anch} and re-measure.")

    async def _generate_reflection(self, user_prompt, history, tc, resp) -> str:
        prompt = REFLECTION_PROMPT.format(
            user_prompt=user_prompt,
            conversation_history=history,
            tool_name=tc.name,
            tool_arguments=json.dumps(tc.arguments, indent=2),
            tool_response=resp,
        )
        return await self._gen(prompt)

    async def _generate_answer_confirm(
        self, user_prompt, constraint_check, molecule,
    ) -> str:
        """Closing reasoning for the terminal ANSWER segment: confirm, by reading
        the rule-based constraint check, that the generation is complete."""
        prompt = ANSWER_CONFIRM_PROMPT.format(
            user_prompt=user_prompt,
            constraint_check=constraint_check,
            molecule=molecule,
        )
        return await self._gen(prompt)

    async def _populate_tool_reasons(self, rounds, user_prompt,
                                     target_properties=None) -> None:
        """Fill ``tool_reason`` on each non-seed Round that lacks one."""
        for i, r in enumerate(rounds):
            if r.index == 0 or r.tool_reason:
                continue
            prev = rounds[i - 1] if i > 0 else None
            r.tool_reason = await self._generate_tool_reason(
                user_prompt, r, prev, target_properties)

    async def _generate_tool_reason(self, user_prompt, rnd, prev,
                                    target_properties=None) -> str:
        from .constraint_state import _compact_edit_args, _fmt_val

        tool_args = _compact_edit_args(rnd.tool, rnd.args)
        # This one-liner is the round's intent line, quoted by later rounds, so
        # a guessed group name propagates.
        # Give it the same RDKit-verified vocabulary the edit reasoning gets.
        edit_facts = "(no structural edit)"
        if rnd.tool == "edit_fragment":
            args = rnd.args or {}
            edit_facts = "\n".join([
                f"Group removed  (from_smiles): "
                f"{describe_fragment(args.get('from_smiles') or '[*:1]')}",
                f"Group attached (to_smiles):   "
                f"{describe_fragment(args.get('to_smiles') or '[*:1]')}",
                "Edit site:",
                describe_anchor(
                    args.get("mol_smiles") or (prev.smiles if prev is not None else None),
                    args.get("anchors"),
                ),
            ])
        prev_parts, curr_parts = [], []
        if prev and prev.properties:
            prev_parts = [f"{k}={_fmt_val(k, v)}" for k, v in prev.properties.items()]
        if rnd.properties:
            curr_parts = [f"{k}={_fmt_val(k, v)}" for k, v in rnd.properties.items()]
        prev_state = ", ".join(prev_parts) if prev_parts else "(core / not yet measured)"
        curr_state = ", ".join(curr_parts) if curr_parts else "(not measured this round)"
        # Which properties this edit could actually have been FOR. Without this the
        # one-liner claims a goal that was already met ("… to increase MR and logP"
        # when logP was in range before the edit) — measured 297 of 1225 rounds.
        gaps: list[str] = []
        if prev is not None and (target_properties or {}):
            from .constraint_state import _value_in_range
            gaps = [p for p, rng in target_properties.items()
                    if p in (prev.properties or {})
                    and _value_in_range(prev.properties.get(p), rng) is False]
        prompt = TOOL_REASON_PROMPT.format(
            user_prompt=user_prompt,
            tool_name=rnd.tool,
            tool_args=tool_args,
            edit_facts=edit_facts,
            prev_state=prev_state,
            curr_state=curr_state,
            open_gaps=", ".join(gaps) if gaps
            else "(none were out of range — describe the structural change only)",
        )
        text = (await self._gen(prompt)).strip()
        # This line is the round's intent line, quoted by later rounds, where a
        # raw fragment SMILES ("Attach [*:1]C(=C)C to …")
        # reads as a tool argument rather than a note; the prompt forbids SMILES but
        # it still leaks, so ask again with the offending string quoted back.
        for _ in range(2):
            if "[*:" not in text:
                break
            text = (await self._gen(
                prompt + "\n\n## Your previous attempt was WRONG\n"
                f"It wrote a raw fragment SMILES: {text!r}\n"
                "Name the group in words instead (use the Verified Chemistry Facts "
                "names), and include NO SMILES at all. Rewrite the one sentence.\n\n"
                "Reason:"
            )).strip() or text
        return text

    # -- reasoning sanitiser ------------------------------------------------

    _TAG_RE = _re.compile(r"</?ANSWER>", _re.IGNORECASE)
    _LABEL_RE = _re.compile(
        r"^\s*(?:Reasoning|Reflection|Final Verification|Summary|Post-analysis Reasoning)\s*:\s*",
        _re.IGNORECASE,
    )
    # Field labels from the injected fact blocks that the model sometimes quotes
    # verbatim into its prose ("I commit the Rank of this candidate: …"). The
    # value that follows is correct, so drop just the label.
    _PROMPT_LABEL_RE = _re.compile(
        r"\b(?:the\s+)?(?:Rank (?:of this candidate|in the returned list)|"
        r"Candidate rank|Group being (?:removed|attached)(?:\s*\([a-z_]+\))?|"
        r"Edit site|Verified Chemistry Facts)\s*:\s*",
        _re.IGNORECASE,
    )
    # Fact-block wording that would be nonsense inside prose. Kept deliberately
    # narrow: "adds 3 heavy atoms" or "connects through its aliphatic C" are
    # legitimate sentences, so only the meta-note about naming is dropped.
    _FACT_FILLER_RE = _re.compile(
        r"\s*;?\s*no curated common name[^.;]*(?=[;.]|$)", _re.IGNORECASE)
    # Fourth-wall / meta-commentary the model occasionally slips into (it starts
    # reasoning ABOUT the data-generation task instead of staying in character).
    # Reasoning is truncated at the first such phrase.
    _META_RE = _re.compile(
        r"(?:\bas an ai\b|as a language model|generating reasoning|"
        r"\bthe user wants\b|\bthe user is asking\b|\bthe user asked\b|"
        r"training data|for a \*?given\*? action|let'?s look closer|"
        r"is it possible the user|"
        # Backtracking / think-out-loud rambling (esp. the seed-intro derivation):
        r"\bwait,|\bwait\.|\bhold on\b|let me re-?(?:read|trace|evaluate|examine|"
        r"consider|check|verify|look)|let'?s re-?(?:read|trace|evaluate|examine|"
        r"look)|let'?s trace|on second thought|\bhmm\b|\bactually,)",
        _re.IGNORECASE,
    )

    async def _gen(self, prompt: str) -> str:
        """LLM reasoning call with sanitisation of stray control tags/labels.

        The model occasionally wraps its reasoning in ``<ANSWER>…</ANSWER>`` or
        ``<ANSWER>…</ANSWER>`` tags or prefixes a ``Reasoning:`` label (echoed
        from the prompt).  These tags are control tokens this pipeline emits
        itself in the right places, so they must never appear inside generated
        reasoning prose.
        """
        return self._clean_reasoning(await self.llm.generate(prompt))

    @classmethod
    def _clean_reasoning(cls, text: str) -> str:
        t = (text or "").strip()
        t = cls._TAG_RE.sub("", t)          # drop stray <ANSWER> tags
        t = cls._LABEL_RE.sub("", t)        # drop a leading echoed label
        t = cls._PROMPT_LABEL_RE.sub("", t)  # drop echoed fact-block field labels
        t = cls._FACT_FILLER_RE.sub("", t)   # drop echoed fact-block fillers
        t = _re.sub(r"[ \t]{2,}", " ", t)
        t = cls._truncate_meta(t)           # cut fourth-wall / meta rambling
        t = _strip_trailing_bare_smiles(t)  # drop a trailing bare-SMILES line
        t = t.strip()
        # Occasionally the model answers with nothing but a SMILES string. That
        # is not reasoning, so report it as empty and let the caller fall back to
        # its deterministic sentence.
        if _looks_like_bare_smiles(t):
            return ""
        return t

    @classmethod
    def _truncate_meta(cls, t: str) -> str:
        """Truncate reasoning at the first meta-commentary phrase.

        Keeps the clean leading reasoning up to the last sentence boundary
        before the meta phrase; returns "" when nothing clean precedes it (the
        caller then substitutes a deterministic fallback sentence).
        """
        m = cls._META_RE.search(t)
        if not m:
            return t
        head = t[: m.start()]
        cut = max(head.rfind(". "), head.rfind("! "), head.rfind("? "))
        if cut != -1:
            return head[: cut + 1].strip()
        head = head.strip()
        return head if head.endswith((".", "!", "?")) else ""

    # -- tool execution / context helpers -----------------------------------

    def _exec_step_calls(self, step):
        """Execute a chain step's primary + parallel calls (from expected_response).

        Returns ``(resp, parallel_calls, parallel_responses)`` — the parallel lists
        are aligned by index (the checkpoint's ``analyze_properties`` +
        ``label_atom_indices`` companions), empty for a solo step.
        """
        resp = self.executor.execute(
            step.tool_call, expected_response=step.expected_response)
        pcalls = list(step.parallel_tool_calls)
        presps: list = []
        for j, pc in enumerate(pcalls):
            exp = (step.parallel_expected_responses[j]
                   if j < len(step.parallel_expected_responses) else None)
            presps.append(self.executor.execute(pc, expected_response=exp))
        return resp, pcalls, presps

    @staticmethod
    def _append_ctx(
        ctx: list[str], reasoning: str, tc: ToolCall, resp: str,
        pcalls: Optional[list] = None, presps: Optional[list] = None,
    ) -> None:
        ctx.extend([
            f"[Reasoning] {reasoning}",
            f"[Tool Call] {tc.name}({json.dumps(tc.arguments)})",
            f"[Tool Response] {resp}",
        ])
        for j, pc in enumerate(pcalls or []):
            pr = presps[j] if presps and j < len(presps) else ""
            ctx.extend([
                f"[Tool Call] {pc.name}({json.dumps(pc.arguments)})",
                f"[Tool Response] {pr or ''}",
            ])
