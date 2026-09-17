"""Ground-truth tool-chain builder for the scaffold-constrained generation task.

Task
----
At inference the model is given **property ranges + a natural-language scaffold
description** and must generate ONE molecule satisfying every constraint — it does
NOT see ``ref_smiles``.  The training trajectory must therefore read as a
*derivation* from the description and properties, not a transcription of a known
answer, even though (in ref mode) it happens to terminate at the gold molecule.

Input
-----
A JSONL benchmark (e.g. ``generation_200k_scaffold``).  Each instance carries:

* ``ref_smiles``      – the gold reference molecule (a hit for the constraints).
* ``scaffold_smarts`` – the strict scaffold substructure SMARTS the molecule
  must embed (the framework the ``description`` is written around).
* ``properties``      – a list of ``{property, min?, max?}`` target ranges.
* ``description`` / ``topology_summary`` / ``connections`` / ``ring_systems`` –
  the natural-language + structured scaffold spec.

Strategy (direct complete-scaffold seed, default)
-------------------------------------------------
A trajectory is planned by the Δ-guided forward search over the described scaffold
(see :mod:`3_toolchain_gen.search_plan`), with every edit tagged ``decorate``
(side chains that tune the properties) or ``finalize`` (stereochemistry).  With
``--direct-scaffold-seed`` (default on) the chain **starts from the completed
scaffold**, so it teaches only property tuning:

* **seed** – the completed scaffold, immediately followed by a 3-way verification
  *checkpoint* ``match_substructure`` ∥ ``analyze_properties`` ∥
  ``label_atom_indices`` (match confirms the framework, analyze reports the
  baseline, and the label grounds the atom indices the FIRST decorate edit targets).
* **decorate phase** – each side-chain edit (``attach_fragment`` / ``form_bond`` /
  ``replace_fragment``), each followed by the SAME 3-way checkpoint on the edited
  molecule — teaching a measure→edit→re-measure loop where every checkpoint's
  ``label_atom_indices`` companion grounds the NEXT edit.
* **finalize** – an optional ``set_stereochemistry`` step (kept as a decorate-like
  edit), followed by its 3-way checkpoint; the last checkpoint's results drive the
  constraint evaluation and the molecule is emitted as the answer.

``--no-direct-scaffold-seed`` restores the legacy incremental assembly: a lone
``label_atom_indices`` on a hub-core seed, the scaffold phase that builds the
framework, and 2-way ``match_substructure`` ∥ ``analyze_properties``
checkpoints with a separate pre-edit label before every edit.

ref mode
--------
``--use-ref`` (default on) plans from ``ref_smiles``.  ``--no-use-ref`` plans
purely from the description + properties via :meth:`ToolChainBuilder._plan_search`
(a forward-looking, exploratory planner — stubbed for now); when it yields no
plan the instance is skipped.  Both modes share the same phase-aware assembly.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Optional

from rdkit import Chem, RDLogger

from .config import ADMET_PROPERTY_NAMES, ALL_TOOL_NAMES  # noqa: F401
from .constraints import check_property, extract_properties
from .http_client import (
    api_call_async, close_async_session, local_mode_enabled)
from .schema import GeneratorInput, ToolCall, ToolChainStep

# Forward planner: suggest_edits-guided decoration of the scaffold seed
# (greedy / beam / hybrid), measured in the loop, scaffold-guarded.
from .search_plan import (
    beam_search, hybrid_search, measure_with_retry, plan_search)

RDLogger.DisableLog("rdApp.*")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RDKit helpers
# ---------------------------------------------------------------------------

def canonical(smiles: Optional[str]) -> Optional[str]:
    """Canonical SMILES, or ``None`` if *smiles* is empty / unparseable."""
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


def smarts_to_smiles(smarts: str, fallback: Optional[str] = None) -> Optional[str]:
    """Convert a SMARTS query into a concrete, sanitizable SMILES.

    The substructure SMARTS encodes the seed scaffold.  Most strict SMARTS carry
    query-only primitives (degree / H-count / ring-size) and aromatic patterns
    that RDKit can't kekulise back into a valid molecule, so this falls back to
    *fallback* (the substructure's canonical ``smiles_key``) whenever the direct
    conversion fails to yield a parseable molecule.
    """
    if smarts:
        qmol = Chem.MolFromSmarts(smarts)
        if qmol is not None:
            try:
                smi = Chem.MolToSmiles(qmol)
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    return Chem.MolToSmiles(mol)
            except Exception:
                pass
    return canonical(fallback)


def _lists_edit(response, edit_args: dict) -> bool:
    """Whether a ``suggest_edits`` response offers the edit *edit_args* commits.

    *response* is the tool's candidate list (or its JSON text). Matching is on the
    ``(from_smiles, to_smiles)`` rule — the same identity ``edit_fragment`` takes —
    so an anchor chosen at a different equivalent site still counts.
    """
    if isinstance(response, str):
        try:
            response = json.loads(response)
        except (ValueError, TypeError):
            return False
    if not isinstance(response, list):
        return False
    key = (edit_args.get("from_smiles"), edit_args.get("to_smiles"))
    return any(isinstance(c, dict)
               and (c.get("from_smiles"), c.get("to_smiles")) == key
               for c in response)


def is_failed_instance(instance: dict) -> bool:
    """Whether *instance* is a generation failure that should be skipped.

    The instance generator marks failures two ways: ``has_scaffold == False``
    (no scaffold could be extracted) and a ``description_violations`` key (the
    natural-language description disagrees with the actual scaffold). Either
    means the ref / description pair is unreliable, so no tool chain is built.
    """
    if instance.get("has_scaffold") is False:
        return True
    # Functional-group instances carry has_fg instead; an empty constraint cannot be
    # planned for and cannot be scored, so it is the same kind of failure.
    if instance.get("has_fg") is False:
        return True
    if instance.get("description_violations"):
        return True
    return False


def substructure_smarts(instance: dict) -> str:
    """The strict scaffold SMARTS the molecule must embed.

    Prefers the new ``scaffold_smarts`` field; falls back to the legacy
    ``smarts`` / ``smarts_by_aspect.strict`` keys, then to the element-level
    dimension SMARTS, so both the new scaffold dataset and the old
    ``generation_sub`` benchmark are supported.
    """
    if instance.get("scaffold_smarts"):
        return instance["scaffold_smarts"]
    if instance.get("smarts"):
        return instance["smarts"]
    by_aspect = instance.get("smarts_by_aspect") or {}
    if by_aspect.get("strict"):
        return by_aspect["strict"]
    fg = [s for s in (instance.get("fg_smarts") or []) if s]
    if fg:
        # Functional-group instance (2b): no scaffold exists. The memory block
        # pins ONE canonical SMARTS, so the first group takes that slot; the rest are
        # still enforced, by the guard below and by their own match_substructure calls.
        return fg[0]
    dims = instance.get("dimension_smarts") or {}
    return dims.get("element") or dims.get("skeleton") or ""


def _match_query(queries: list):
    """The ``query`` argument for one ``match_substructure`` call.

    A single pattern stays a bare STRING so scaffold chains are byte-identical to
    every trajectory generated before; several patterns become a LIST, which the tool
    matches one by one and ANDs. They are never joined with '.' — that would demand
    atom-disjoint matches and reject a molecule whose two required groups share an
    atom (an amide N that is also a piperazine N).
    """
    return queries[0] if len(queries) == 1 else list(queries)


def guard_smarts(instance: dict) -> list:
    """Every SMARTS an edit must preserve — one entry, or one per required group.

    A scaffold instance has a single strict pattern. A functional-group instance has
    one per member and they must be checked SEPARATELY: joining them with '.' would
    demand atom-disjoint matches and so reject a molecule whose amide N is also its
    piperazine N (measured: 13.1% of two-group constraints).
    """
    fg = [s for s in (instance.get("fg_smarts") or []) if s]
    if fg:
        return fg
    fg = [m["smarts"] for m in (instance.get("fg_constraint") or [])
          if isinstance(m, dict) and m.get("smarts")]
    if fg:
        return fg
    one = substructure_smarts(instance)
    return [one] if one else []


def seed_for(instance: dict) -> str:
    """The molecule the chain starts from.

    ``scaffold_smiles`` for a scaffold instance — it IS the substructure. A
    functional-group instance instead carries ``seed_smiles``, a molecule CONSTRUCTED
    to hold every required group (see 2b_functional_group_gen/fg_seed.py); its
    SMARTS cannot be turned into a usable seed by transcription, and for two groups
    would not even be a connected molecule.
    """
    scaffold = instance.get("scaffold_smiles") or ""
    if scaffold:
        return scaffold
    if instance.get("seed_smiles"):
        return instance["seed_smiles"]
    return smarts_to_smiles(substructure_smarts(instance)) or ""


# ---------------------------------------------------------------------------
# User-prompt construction (mirrors the evaluation prompt when available)
# ---------------------------------------------------------------------------

# Integer-valued properties (shown as integers; every other property is a float
# rendered to at most 3 decimals). MIRROR of memory._INT_LIKE.
_INT_PROPS = frozenset({"HBD", "HBA", "rotB", "rings_total", "heavy_atoms", "formal_charge"})


def _fmt_prop(name: str, v) -> Optional[str]:
    """Format a property value for the prompt: integer props as ints, floats to at
    most 3 decimals (trailing zeros dropped). ``None`` passes through."""
    if v is None:
        return None
    if name in _INT_PROPS:
        return str(int(round(float(v))))
    return f"{float(v):.3f}"


def _build_user_prompt(instance: dict, properties: dict[str, list]) -> str:
    """Construct the user message for one instance.

    Self-contained: builds a substructure-description + property-range prompt
    ending in an ``<ANSWER>`` directive. Kept independent of the benchmark
    evaluation code (``evaluate_baselines/``) so the SFT pipeline carries no
    dependency on it.
    """
    desc = (instance.get("description") or "").strip()
    lines = [
        "Generate ONE chemically valid, drug-like molecule that contains the "
        "substructure described below."
    ]
    if desc:
        lines.append(f"\nSubstructure description:\n{desc}")
    if properties:
        lines.append(
            "\nThe molecule must also satisfy ALL of the following target "
            "properties — each computed value must fall within the stated range:"
        )
        for name, (lo, hi) in properties.items():
            lo_s, hi_s = _fmt_prop(name, lo), _fmt_prop(name, hi)
            if lo is not None and hi is not None:
                if lo == hi:
                    lines.append(f"  - {name}: exactly {lo_s}")
                else:
                    lines.append(f"  - {name}: between {lo_s} and {hi_s}")
            elif lo is not None:
                lines.append(f"  - {name}: at least {lo_s}")
            elif hi is not None:
                lines.append(f"  - {name}: at most {hi_s}")
    lines.append(
        "\nReturn the final molecule as a single SMILES string enclosed in "
        "<ANSWER> and </ANSWER> tags."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class ToolChainBuilder:
    """Build ground-truth tool chains for substructure-generation instances."""

    def __init__(self, args) -> None:
        self.args = args
        self.rng = random.Random(getattr(args, "seed", 42))
        self.num_workers = max(1, int(getattr(args, "num_workers", 20)))

    # -- tool calls ---------------------------------------------------------

    async def _call(self, name: str, args: dict, errors: list) -> Any:
        """Call a tool over HTTP and return its structured result.

        Uses ``return_text=False`` so the structured payload (dict / list / SMILES
        string) comes back in a single round-trip — used directly for constraint
        evaluation and serialised for the stored ``expected_response`` (see
        :meth:`_resp_text`). Failures are recorded in *errors* and return ``None``.
        """
        try:
            return await api_call_async(name, args, return_text=False)
        except Exception as exc:  # pragma: no cover - network/runtime guard
            errors.append({"tool": name, "args": args, "error": str(exc)})
            return None

    @staticmethod
    def _resp_text(result: Any) -> Optional[str]:
        """Serialise a tool result for storage as ``expected_response`` — dicts /
        lists become JSON, strings (SMILES / prose) pass through, ``None`` stays."""
        if result is None or isinstance(result, str):
            return result
        return json.dumps(result, ensure_ascii=False)

    async def _analyze(self, mol_smiles: str, prop_names: list, errors: list) -> Any:
        """``analyze_properties`` with retry-on-None-ADMET.

        The ADMET backend intermittently fails its five model outputs as a unit
        (returning ``None`` for all of them at once) under GPU contention, while
        the RDKit physchem values are unaffected; the values are deterministic
        once they return.  A bounded retry (see :func:`search_plan.measure_with_retry`)
        recovers a complete measurement, so a validly-built molecule is not marked
        failed — and the stored ``expected_response`` never carries a null ADMET.
        """
        async def one(smi: str, props: list) -> Any:
            args: dict = {"mol_smiles": smi}
            if props:
                args["property_names"] = props
            return await self._call("analyze_properties", args, errors)

        return await measure_with_retry(one, mol_smiles, prop_names or [])

    # Every step helper returns a uniform ``(step, match_struct, analyze_struct)``
    # tuple (structs are ``None`` unless a verification fetched them) so the steps
    # can all be dispatched concurrently with ``asyncio.gather`` and reassembled
    # in order — the plan is precomputed offline, so no call depends on another's
    # response and the whole instance fans out in a single round-trip wave.

    async def _label_step(self, mol_smiles: str, errors: list):
        """``label_atom_indices`` on *mol_smiles* — grounds the 0-based atom_index
        the following edit uses (its indices match the edit's, both parsing the
        same SMILES string)."""
        args = {"mol_smiles": mol_smiles}
        res = await self._call("label_atom_indices", args, errors)
        return (ToolChainStep(
            tool_call=ToolCall(name="label_atom_indices", arguments=args),
            expected_response=self._resp_text(res)), None, None)

    async def _suggest_step(self, suggest: dict, errors: list,
                            committed: Optional[dict] = None):
        """The ``suggest_edits`` call that precedes — and justifies — an
        ``edit_fragment`` edit: ranked candidate edits toward the target property
        box, each a ready-to-use ``{from_smiles, to_smiles, anchors}`` triple.
        Emitted as its own assistant turn right before the edit it informs; the
        committed edit is one of these candidates.

        Calls the live tool for the stored ``expected_response`` (byte-identical to
        what the server returns); falls back to the planner's own candidate ranking
        when the tool is unavailable OR when its answer does not contain the edit
        this step commits (*committed*).

        That last case must not silently ship: the planner ranked with the props it
        had already measured, while the live call re-measures internally, so any
        disagreement between the two measurements can push the committed edit out of
        the ``top_k`` slice — leaving a trajectory that commits an edit the tool
        never offered (or an empty candidate list). The measurement itself is now
        aligned (see ``suggest_edits._measure``); this guard keeps the invariant
        true by construction."""
        args = dict(suggest["arguments"])
        res = await self._call("suggest_edits", args, errors)
        # Fall back to the planner's own (props-based, guarded) candidate ranking if
        # the live tool is unavailable or errors — so the emitted trajectory never
        # carries a tool-error string for a step the search actually succeeded on.
        if res is None or (isinstance(res, str) and res.lstrip().startswith("Error")):
            res = suggest.get("candidates")
        elif committed is not None and not _lists_edit(res, committed):
            logger.warning(
                "suggest_edits response omits the committed edit (%s -> %s); "
                "falling back to the planner's candidate ranking",
                committed.get("from_smiles"), committed.get("to_smiles"),
            )
            res = suggest.get("candidates") or res
        return (ToolChainStep(
            tool_call=ToolCall(name="suggest_edits", arguments=args),
            expected_response=self._resp_text(res)), None, None)

    async def _edit_step(self, op: dict, errors: list):
        """A single planned edit — ``edit_fragment`` (attach / swap / remove)."""
        args = dict(op["arguments"])
        res = await self._call(op["tool"], args, errors)
        return (ToolChainStep(
            tool_call=ToolCall(name=op["tool"], arguments=args),
            expected_response=self._resp_text(res)), None, None)

    async def _verify_step(
        self, smarts, mol_smiles: str, prop_names: list, errors: list,
    ) -> tuple[ToolChainStep, Any, Any]:
        """A ``match_substructure`` ∥ ``analyze_properties`` check on
        *mol_smiles* — two read-only calls on the same molecule with no mutual
        dependency, so they run concurrently and are emitted as one parallel step.

        For property-only instances (no substructure constraint, *smarts* empty)
        the match call is dropped and a lone ``analyze_properties`` step
        is emitted instead.

        Returns the structured match / analyze results too (used for constraint
        evaluation at the final verification; discarded at intermediate checks).
        """
        queries = ([smarts] if isinstance(smarts, str) else list(smarts or []))
        queries = [q for q in queries if q]
        analyze_args: dict = {"mol_smiles": mol_smiles}
        if prop_names:
            analyze_args["property_names"] = prop_names

        if not queries:
            analyze_res = await self._analyze(mol_smiles, prop_names, errors)
            step = ToolChainStep(
                tool_call=ToolCall(
                    name="analyze_properties", arguments=analyze_args),
                expected_response=self._resp_text(analyze_res),
            )
            return step, None, analyze_res

        # One call; the tool ANDs over a list (see _checkpoint_step / _match_query).
        match_args = {"query": _match_query(queries), "mol_smiles": mol_smiles,
                      "query_type": "smarts"}
        match_res, analyze_res = await asyncio.gather(
            self._call("match_substructure", match_args, errors),
            self._analyze(mol_smiles, prop_names, errors),
        )
        step = ToolChainStep(
            tool_call=ToolCall(name="match_substructure", arguments=match_args),
            expected_response=self._resp_text(match_res),
            parallel_tool_calls=[ToolCall(
                name="analyze_properties", arguments=analyze_args
            )],
            parallel_expected_responses=[self._resp_text(analyze_res)],
        )
        return step, match_res, analyze_res

    async def _checkpoint_step(
        self, smarts, mol_smiles: str, prop_names: list, errors: list,
    ) -> tuple[ToolChainStep, Any, Any]:
        """The 3-way verification checkpoint on *mol_smiles*, run as ONE parallel step:

            match_substructure ∥ analyze_properties ∥ label_atom_indices

        The three calls are read-only/independent on the SAME molecule — ``match``
        confirms the framework is intact, ``analyze`` reports where the properties
        stand, and ``label_atom_indices`` grounds the atom indices the NEXT edit
        will target (so the label that used to precede each edit is folded into the
        checkpoint that follows the previous one). ``match_substructure`` is the
        primary ``tool_call``; ``analyze`` + ``label`` are its parallel companions.

        For property-only instances (no substructure constraint, *smarts* empty)
        the match call is dropped and ``analyze`` becomes the primary, with
        ``label`` as its lone companion.

        *smarts* may be a LIST — a functional-group instance requires one pattern per
        group, and each needs its OWN match call. They cannot be folded into a single
        dot-joined query: that would demand atom-disjoint matches and reject a
        molecule whose amide N is also its piperazine N. The first match is the
        primary call and the rest join the parallel companions, so the checkpoint
        still reads as one step.

        Returns the structured match / analyze results (the label response is only
        stored in the step) for the final constraint evaluation.
        """
        queries = ([smarts] if isinstance(smarts, str) else list(smarts or []))
        queries = [q for q in queries if q]
        analyze_args: dict = {"mol_smiles": mol_smiles}
        if prop_names:
            analyze_args["property_names"] = prop_names
        label_args = {"mol_smiles": mol_smiles}

        if not queries:
            analyze_res, label_res = await asyncio.gather(
                self._analyze(mol_smiles, prop_names, errors),
                self._call("label_atom_indices", label_args, errors),
            )
            step = ToolChainStep(
                tool_call=ToolCall(
                    name="analyze_properties", arguments=analyze_args),
                expected_response=self._resp_text(analyze_res),
                parallel_tool_calls=[
                    ToolCall(name="label_atom_indices", arguments=label_args)],
                parallel_expected_responses=[self._resp_text(label_res)],
            )
            return step, None, analyze_res

        # ONE call, whatever the number of patterns: match_substructure takes a list
        # and reports match=AND over the entries. Each is still matched separately
        # inside the tool — they are never concatenated into one dot-joined query.
        match_args = {"query": _match_query(queries), "mol_smiles": mol_smiles,
                      "query_type": "smarts"}
        match_res, analyze_res, label_res = await asyncio.gather(
            self._call("match_substructure", match_args, errors),
            self._analyze(mol_smiles, prop_names, errors),
            self._call("label_atom_indices", label_args, errors),
        )
        step = ToolChainStep(
            tool_call=ToolCall(name="match_substructure", arguments=match_args),
            expected_response=self._resp_text(match_res),
            parallel_tool_calls=[
                ToolCall(name="analyze_properties", arguments=analyze_args),
                ToolCall(name="label_atom_indices", arguments=label_args),
            ],
            parallel_expected_responses=[
                self._resp_text(analyze_res),
                self._resp_text(label_res),
            ],
        )
        return step, match_res, analyze_res

    # -- planning (forward-looking search over the scaffold seed) -----------

    async def _plan_search(self, instance: dict) -> Optional[dict]:
        """Forward-looking planner (scaffold + properties → trajectory), no ref.

        The described scaffold ``scaffold_smiles`` is the seed, and the
        **decorate phase** greedily applies mmpdb moves
        (:func:`search_plan.plan_search`) that reduce the measured distance to
        the target property box, guarding every edit against the scaffold SMARTS.

        Returns a plan with ``seed``, phase-tagged ``steps``, and ``ref_iso`` =
        the constructed molecule; ``None`` only if unusable.
        """
        scaffold = seed_for(instance)
        # The guard is a LIST: one strict scaffold pattern, or one per required
        # functional group. search_plan checks each on its own.
        smarts = guard_smarts(instance)
        if not scaffold or not smarts:
            return None
        targets = extract_properties(instance)
        errors: list[dict] = []

        async def measure(smi: str, props: list) -> dict:
            res = await self._call(
                "analyze_properties",
                {"mol_smiles": smi, "property_names": props}, errors)
            return res if isinstance(res, dict) else {}

        # Batched measure for beam rounds: featurise + predict a whole round's
        # candidates in ONE pass (~3x cheaper/molecule — amortises the ADMET
        # per-prediction overhead). Only wired in local (in-process) mode, where a
        # true single-pass ``_batch`` is available; the HTTP path keeps the
        # per-candidate measure (with its ADMET-None retry). These search-internal
        # measurements are throwaway, so batching them never affects stored output.
        measure_batch = None
        if local_mode_enabled():
            async def measure_batch(smi_list: list, props: list) -> dict:
                if not smi_list:
                    return {}
                from .local_tools import local_call_batch
                return await asyncio.to_thread(
                    local_call_batch, "analyze_properties",
                    smi_list, property_names=props)

        # Decorate phase: property tuning from the completed scaffold, over the
        # suggest_edits candidates — greedy (commit the top-ranked guard-passing
        # candidate) or beam (explore beam_width paths, select by real measurement).
        # --search-mode picks; all share the same suggest_edits oracle + guard.
        budget = int(getattr(self.args, "search_max_steps", 12))
        mode = getattr(self.args, "search_mode", "greedy")
        bw = int(getattr(self.args, "beam_width", 4))
        bk = int(getattr(self.args, "beam_expand", 6))
        top_k = int(getattr(self.args, "suggest_top_k", 10))
        if mode == "beam":
            dec = await beam_search(scaffold, smarts, targets, measure,
                                    beam_width=bw, expand_k=bk, max_rounds=budget,
                                    top_k=top_k, measure_batch_fn=measure_batch)
        elif mode == "hybrid":
            dec = await hybrid_search(scaffold, smarts, targets, measure,
                                      max_steps=budget, beam_width=bw, expand_k=bk,
                                      top_k=top_k, measure_batch_fn=measure_batch)
        else:
            dec = await plan_search(scaffold, smarts, targets, measure,
                                    max_steps=budget, top_k=top_k)
        # The described scaffold is the seed: the chain commits it whole and then
        # decorates it. (The old incremental scaffold-build phase grew it from a
        # hub seed via the 13-tool attach_fragment/form_bond planner; that planner
        # is gone, and its steps were dropped under the direct-scaffold seed
        # anyway, so `dec` is the plan.)
        return dec

    # -- per-instance chain -------------------------------------------------

    async def build(self, instance: dict, instance_index: int) -> Optional[GeneratorInput]:
        # Skip generation failures (no scaffold, or description ↔ scaffold mismatch).
        if is_failed_instance(instance):
            logger.info("instance %d (%s): generation failure, skipping",
                        instance_index, instance.get("id"))
            return None

        # Every pattern the chain must keep alive. A scaffold instance yields one; a
        # functional-group instance yields one per required group, and the checkpoint
        # emits a match_substructure for each.
        smarts = guard_smarts(instance)
        ref_raw = instance.get("ref_smiles", "")
        ref = canonical(ref_raw)

        # Plan a phase-tagged trajectory (scaffold seed → property-tuning
        # decorations → stereochemistry). None when no constructive path exists.
        plan_obj = await self._plan_search(instance)
        constructive = plan_obj is not None
        if not constructive:
            logger.warning(
                "instance %d: search planner found no path, skipping",
                instance_index,
            )
            return None
        seed = plan_obj["seed"]
        final_mol = plan_obj.get("ref_iso") or ref or seed
        scaffold_smiles = plan_obj.get("scaffold_smiles") or seed
        plan_steps = plan_obj["steps"]
        if not seed:
            logger.warning("instance %d: no usable seed/ref, skipping", instance_index)
            return None

        # ── Direct complete-scaffold seed (default) ─────────────────────────
        # The seed IS the finished scaffold: the scaffold-building edits are
        # dropped and only the property-tuning (decorate) + stereochemistry
        # (finalize) edits are kept. Each checkpoint folds in ``label_atom_indices``
        # so the label that used to precede every edit now rides on the PREVIOUS
        # checkpoint — the plan chains ``step[i]["result"]`` byte-identically to
        # ``step[i+1]["arguments"]["mol_smiles"]``, so the folded label grounds the
        # next edit's atom_index exactly as a dedicated pre-edit label would.
        direct_seed = bool(getattr(self.args, "direct_scaffold_seed", True))
        if direct_seed:
            if constructive:
                seed = scaffold_smiles or seed
                body_steps = [op for op in plan_steps if op.get("phase") != "scaffold"]
                final_mol = (body_steps[-1].get("result") if body_steps else seed) or seed
            else:
                # Anchor (no constructive path): the seed IS the proposed molecule.
                seed = final_mol or seed
                body_steps = []
        else:
            body_steps = plan_steps

        properties = extract_properties(instance)
        prop_names = list(properties.keys())
        # The guard the planner enforced, carried into the metadata so stage 4 can
        # emit one match_substructure per required pattern.
        fg_guard = guard_smarts(instance)
        errors: list[dict] = []

        # Assemble the ordered list of step coroutines, then dispatch them all
        # concurrently: every call carries precomputed SMILES/indices, so none
        # depends on another's response and the instance resolves in one fan-out
        # wave rather than many sequential round-trips. (The edit→checkpoint ORDER
        # is what the trace teaches; the calls themselves are independent.)
        if direct_seed:
            # Chain: [seed 3-way checkpoint] → for each edit:
            #   [suggest_edits] → [edit_fragment] → [3-way checkpoint].
            # suggest_edits is emitted immediately before the edit it informs.
            coros = [self._checkpoint_step(smarts, seed, prop_names, errors)]
            for op in body_steps:
                if op.get("suggest"):
                    coros.append(self._suggest_step(op["suggest"], errors, op.get("arguments")))
                coros.append(self._edit_step(op, errors))
                result_mol = op.get("result") or op["arguments"].get("mol_smiles") or seed
                coros.append(self._checkpoint_step(smarts, result_mol, prop_names, errors))
        else:
            # ── Legacy incremental assembly (2-way checkpoints, explicit pre-edit
            # steps). With the current tool set the only edit primitive is
            # edit_fragment (preceded by suggest_edits); step-by-step scaffold
            # BUILDING (the old attach_fragment/form_bond path) is no longer
            # expressible, so any such step is a stale planner artifact — skip the
            # instance loudly rather than emit a call to a removed tool. ──────────
            stale = [op["tool"] for op in plan_steps
                     if op.get("tool") not in ("edit_fragment", None)]
            if stale:
                logger.warning("instance %d: legacy assembly saw removed edit tool(s) "
                               "%s — skipping (use --direct-scaffold-seed)",
                               instance_index, sorted(set(stale)))
                return None
            analyze_each = bool(getattr(self.args, "analyze_each_decorate", True))
            decorate_idxs = [i for i, op in enumerate(plan_steps)
                             if op.get("phase") == "decorate"]
            last_decorate = decorate_idxs[-1] if decorate_idxs else None
            coros = []
            if not plan_steps:                   # anchor / no-edit: just inspect the seed
                coros.append(self._label_step(seed, errors))
            scaffold_checkpoint_done = False
            for i, op in enumerate(plan_steps):
                phase = op.get("phase")
                if (phase == "decorate" and not scaffold_checkpoint_done
                        and decorate_idxs):
                    coros.append(self._verify_step(
                        smarts, scaffold_smiles, prop_names, errors))
                    scaffold_checkpoint_done = True
                coros.append(self._label_step(op["arguments"]["mol_smiles"], errors))
                if op.get("suggest"):
                    coros.append(self._suggest_step(op["suggest"], errors, op.get("arguments")))
                coros.append(self._edit_step(op, errors))
                if (analyze_each and phase == "decorate" and i != last_decorate
                        and op.get("result")):
                    coros.append(self._verify_step(
                        smarts, op["result"], prop_names, errors))
            coros.append(self._verify_step(smarts, final_mol, prop_names, errors))

        results = await asyncio.gather(*coros)
        steps: list[ToolChainStep] = [r[0] for r in results]
        _, match_struct, analyze_struct = results[-1]

        # Evaluate constraint satisfaction. Property-only instances (no
        # substructure constraint) count as matched.
        if not smarts:
            match_ok = True
        else:
            match_ok = bool(match_struct.get("match")) if isinstance(match_struct, dict) else False

        measured = analyze_struct if isinstance(analyze_struct, dict) else {}
        prop_results: dict[str, bool] = {}
        all_ok = True
        for name, (lo, hi) in properties.items():
            ok = check_property(measured.get(name), lo, hi)
            prop_results[name] = ok
            all_ok = all_ok and ok

        all_satisfied = bool(match_ok and all_ok)

        # Phase bookkeeping reflects the steps actually EMITTED: in direct-seed mode
        # only the non-scaffold (decorate + finalize) edits survive, so
        # num_scaffold_steps == 0 and every emitted edit is decorate/finalize.
        emitted_steps = body_steps if direct_seed else plan_steps
        if direct_seed:
            num_scaffold_out = 0
            num_decorate_out = sum(1 for op in emitted_steps
                                   if op.get("phase") == "decorate")
        else:
            num_scaffold_out = plan_obj.get("n_scaffold_steps", 0) if constructive else 0
            num_decorate_out = plan_obj.get("n_decorate_steps", 0) if constructive else 0

        metadata = {
            "task_id": instance.get("id"),
            "task_type": "generation",
            "instance_index": instance_index,
            # ONE canonical pattern, as a string: the memory block pins a single
            # SMARTS and stage 4 reads this as such. The full set is in fg_smarts.
            "smarts": smarts[0] if smarts else "",
            # Every SMARTS an edit had to preserve. One entry for a scaffold task,
            # one per required group for a functional-group task — stage 4 routes on
            # its presence, and the generator writes one match_substructure per entry.
            "fg_smarts": fg_guard if len(fg_guard) > 1 or instance.get("fg_smarts") else [],
            "fg_constraint": instance.get("fg_constraint") or [],
            "ref_smiles": ref or seed,
            "seed_smiles": seed,
            "scaffold_smiles": scaffold_smiles,
            "predicted_molecule": final_mol,
            "target_properties": properties,
            "target_fragments": {},
            "prop_constraint_results": prop_results,
            "prop_constraint_results_strict": prop_results,
            "frag_constraint_results": {},
            "substructure_match": match_ok,
            "all_constraints_satisfied": all_satisfied,
            "all_constraints_strictly_satisfied": all_satisfied,
            "chain_strategy": "constructive" if constructive else "anchor",
            "seed_mode": "scaffold_direct" if direct_seed else "scaffold_whole",
            "direct_scaffold_seed": direct_seed,
            "num_edit_steps": len(emitted_steps),
            "num_scaffold_steps": num_scaffold_out,
            "num_decorate_steps": num_decorate_out,
            "edit_step_phases": [op.get("phase") for op in emitted_steps],
            "num_tool_calls": len(steps),
            # Natural-language / structured scaffold anchors for the narrator stage.
            "description": instance.get("description"),
            "topology_summary": instance.get("topology_summary"),
            "connections": instance.get("connections"),
            "ring_systems": instance.get("ring_systems"),
            "tool_errors": errors,
            "success": all_satisfied,
        }

        return GeneratorInput(
            user_prompt=_build_user_prompt(instance, properties),
            ground_truth_molecule=final_mol,
            tool_chain=steps,
            tool_set=ALL_TOOL_NAMES,
            metadata=metadata,
        )

    # -- batched runner -----------------------------------------------------

    def run(
        self,
        input_path: str,
        output_dir: str,
        limit: Optional[int],
        chunk_size: int,
    ) -> tuple[int, int]:
        # Multiprocess (server-free) path: fan the build across worker processes,
        # each with its own in-process ADMET model. Used when --num-procs > 1
        # (which implies --local-tools). Single-process falls through below.
        num_procs = int(getattr(self.args, "num_procs", 1) or 1)
        if num_procs > 1:
            from .mp_runner import run_multiprocess
            gpus = list(getattr(self.args, "gpus", [0]) or [0])
            per_proc = int(getattr(self.args, "per_proc", 4) or 4)
            part_size = int(getattr(self.args, "part_size", 0) or 0) or None
            return run_multiprocess(
                self.args, input_path, output_dir, limit, chunk_size,
                num_procs, per_proc, gpus, part_size=part_size)

        records = _load_jsonl(input_path, limit=limit)
        if limit is not None:
            records = records[:limit]

        # output_dir is the already-resolved destination (e.g.
        # .../toolchain/<input-folder-name>); chunks land directly inside it.
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        done: set[int] = set()
        start_chunk = 0
        if getattr(self.args, "resume", False):
            done, start_chunk = _scan_existing(out_dir)
            if done:
                logger.info("resume: skipping %d already-built instances", len(done))

        return asyncio.run(
            self._run_async(records, out_dir, chunk_size, done, start_chunk)
        )

    async def _run_async(
        self,
        records: list[dict],
        out_dir: Path,
        chunk_size: int,
        done: set[int],
        start_chunk: int,
    ) -> tuple[int, int]:
        # Cool-downs between batches (see batch loop below). RECYCLE_SESSION drops
        # stale keep-alive sockets after each chunk (on by default — cheap and the
        # usual cure for warnings that build up under sustained load); set
        # BATCH_PAUSE_SEC>0 to additionally idle between batches so the tool servers
        # can drain their backlog.
        pause = float(os.environ.get("BATCH_PAUSE_SEC", "0") or 0)
        recycle = os.environ.get("RECYCLE_SESSION", "1") != "0"

        sem = asyncio.Semaphore(self.num_workers)

        async def worker(idx: int, rec: dict) -> Optional[GeneratorInput]:
            if idx in done:
                return None
            async with sem:
                try:
                    return await self.build(rec, idx)
                except Exception as exc:  # pragma: no cover - per-instance guard
                    logger.exception("instance %d failed: %s", idx, exc)
                    return None

        pending = [i for i in range(len(records)) if i not in done]
        success = total = 0
        chunk_idx = start_chunk

        # Dispatch ONE batch (= chunk_size instances) at a time and wait for the
        # whole batch before starting the next. The barrier means NO requests are
        # in flight between batches, so we can safely (a) drop stale keep-alive
        # sockets (RECYCLE_SESSION) and (b) let the tool servers idle and drain
        # (BATCH_PAUSE_SEC) before the next wave — this is what stops the
        # "Can not write request body" / connection-reset warnings that otherwise
        # accumulate under continuous fan-out. (Pausing the consumer alone would
        # not help: with a single pre-scheduled task pool the workers keep firing
        # regardless, and closing the session mid-flight would break live
        # requests.) Each batch is flushed immediately, so the run stays crash-safe
        # and --resume picks up from the last written chunk; instance_index is
        # recorded per record, so batch ordering is irrelevant.
        try:
            for b in range(0, len(pending), chunk_size):
                batch = pending[b:b + chunk_size]
                results = await asyncio.gather(
                    *(worker(i, records[i]) for i in batch))
                chunk = [r for r in results if r is not None]
                total += len(chunk)
                success += sum(1 for r in chunk if r.metadata.get("success"))
                if chunk:
                    _write_chunk(out_dir, chunk_idx, chunk)
                    chunk_idx += 1
                    logger.info(
                        "progress: %d built (%d ok), %d chunk(s) flushed → %s",
                        total, success, chunk_idx - start_chunk, out_dir,
                    )
                # No requests in flight here — reset the connection pool and/or
                # pause so the next batch hits fresh, idle servers.
                if recycle:
                    await close_async_session()
                if pause:
                    await asyncio.sleep(pause)
        finally:
            await close_async_session()

        return success, total


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: str, limit: Optional[int] = None) -> list[dict]:
    """Load records from a single ``.jsonl`` file or every ``.jsonl`` in a
    directory (sorted by name, concatenated — so instance indices are stable
    across runs and resume works over the combined record list).

    When *limit* is given, loading stops as soon as *limit* records have been
    read (the first records always come from the lowest-numbered file, so this
    is the same prefix that would be processed anyway) — avoiding a full
    multi-GB read just to take a small head for a test run.
    """
    p = Path(path)
    files = sorted(p.glob("*.jsonl")) if p.is_dir() else [p]
    records: list[dict] = []
    used = 0
    for fp in files:
        used += 1
        with open(fp) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
                    if limit is not None and len(records) >= limit:
                        logger.info("loaded %d records (limit) from %d file(s) under %s",
                                    len(records), used, path)
                        return records
    logger.info("loaded %d records from %d file(s) under %s",
                len(records), len(files), path)
    return records


def dataset_name(input_path: str) -> str:
    """Output subfolder name derived from the input: the directory name when
    *input_path* is a directory, else the file stem."""
    p = Path(input_path)
    return p.name if p.is_dir() else p.stem


def _chunk_name(chunk_idx: int) -> str:
    return f"toolchains_generation_chunk_{chunk_idx:04d}.jsonl"


def _write_chunk(out_dir: Path, chunk_idx: int, chunk: list[GeneratorInput]) -> None:
    path = out_dir / _chunk_name(chunk_idx)
    with open(path, "w") as fh:
        for gi in chunk:
            fh.write(json.dumps(gi.model_dump(), ensure_ascii=False) + "\n")
    logger.info("wrote %d records → %s", len(chunk), path)


def _scan_existing(out_dir: Path) -> tuple[set[int], int]:
    """Return ``(processed instance_index set, next chunk index)``."""
    done: set[int] = set()
    max_chunk = -1
    for path in out_dir.glob("toolchains_generation_chunk_*.jsonl"):
        try:
            max_chunk = max(max_chunk, int(path.stem.rsplit("_", 1)[1]))
        except (ValueError, IndexError):
            pass
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    idx = json.loads(line).get("metadata", {}).get("instance_index")
                except json.JSONDecodeError:
                    continue
                if idx is not None:
                    done.add(int(idx))
    return done, max_chunk + 1
