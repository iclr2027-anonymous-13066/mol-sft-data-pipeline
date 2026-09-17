"""Rewrite a chain's seed so the scaffold SMILES *is* the SMARTS transcription.

Stage 3 hands the seed over as RDKit's canonical SMILES.  Canonicalisation
re-roots and re-orders the atoms, so ``SMARTS -> SMILES`` stops being a
read-off: measured over 3k seed segments the canonical string coincided with a
straight transcription only 16% of the time, yet the seed segment asks the model
to produce it in one shot, with no tool feedback in between.  Since the required
substructure covers *every* seed atom (the seed IS the scaffold), that step is
pure notation — the model should not also have to run Morgan canonicalisation in
its head.

``retranscribe_seed`` renumbers the seed to the SMARTS' own atom order and
rewrites the chain around it:

* every tool call taking the seed gets the transcribed string;
* atom indices (``anchors``) — recorded by stage 3 against the canonical order —
  are remapped to the new order, both in ``edit_fragment`` arguments and in the
  ``suggest_edits`` candidates the model picks from;
* the seed's ``label_atom_indices`` response is recomputed.

Everything from round 1 on is untouched: ``edit_fragment`` canonicalises its
product, so the chain re-joins the canonical world as soon as the first edit
lands.  The rewrite is applied only when the round trip verifies — the seed must
re-canonicalise to the original string, and the first edit must still yield the
recorded product — otherwise the chain keeps its canonical seed.

Functional-group instances (2b) are skipped: their seed is a *constructed*
carrier, not a transcription of the constraint.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:  # RDKit is present wherever stage 4 runs; stay importable if it is not.
    from rdkit import Chem as _Chem
except Exception:  # pragma: no cover
    _Chem = None  # type: ignore

try:
    from molkit.utils.molecule_edit_utils import (
        label_atom_indices as _label_atom_indices,
        replace_fragment as _replace_fragment,
    )
except Exception:  # pragma: no cover
    _label_atom_indices = None  # type: ignore
    _replace_fragment = None  # type: ignore


def _index_map(seed: str, smarts: str) -> Optional[tuple[str, dict[int, int]]]:
    """``(transcribed SMILES, {canonical atom index: transcribed atom index})``.

    ``None`` when the SMARTS does not cover the whole seed, the reorder fails, or
    the transcription does not canonicalise back to *seed*.
    """
    if _Chem is None or not seed or not smarts:
        return None
    mol = _Chem.MolFromSmiles(seed)
    patt = _Chem.MolFromSmarts(smarts)
    if mol is None or patt is None:
        return None
    match = mol.GetSubstructMatch(patt)
    if not match or len(match) != mol.GetNumAtoms():
        return None
    try:
        renum = _Chem.RenumberAtoms(mol, list(match))
        ordered = _Chem.MolToSmiles(renum, canonical=False)
        # Written order != index order in general (branches, ring closures), so
        # take the map RDKit itself recorded while writing the string.
        out_order = json.loads(renum.GetProp("_smilesAtomOutputOrder"))
    except Exception:
        return None
    chk = _Chem.MolFromSmiles(ordered)
    if chk is None or _Chem.MolToSmiles(chk) != _Chem.MolToSmiles(mol):
        return None
    # written position j  <-  renum atom out_order[j]  <-  canonical atom match[...]
    canon_to_new = {match[k]: j for j, k in enumerate(out_order)}
    if len(canon_to_new) != mol.GetNumAtoms():
        return None
    return ordered, canon_to_new


def _remap_anchors(anchors, canon_to_new: dict[int, int]):
    """Rewrite ``{label: atom_index}`` into the transcribed atom order."""
    if not isinstance(anchors, dict):
        return anchors, True
    out = {}
    for label, idx in anchors.items():
        try:
            new = canon_to_new[int(idx)]
        except (KeyError, TypeError, ValueError):
            return anchors, False
        out[label] = new
    return out, True


def _remap_candidates(response: Optional[str], canon_to_new: dict[int, int]) -> Optional[str]:
    """Remap the ``anchors`` of every recorded ``suggest_edits`` candidate."""
    if not response:
        return response
    try:
        cands = json.loads(response)
    except Exception:
        return None
    if not isinstance(cands, list):
        return None
    for cand in cands:
        if not isinstance(cand, dict) or "anchors" not in cand:
            continue
        remapped, ok = _remap_anchors(cand.get("anchors"), canon_to_new)
        if not ok:
            return None
        cand["anchors"] = remapped
    return json.dumps(cands, ensure_ascii=False)


def _verify_first_edit(new_seed: str, args: dict, expected: Optional[str]) -> bool:
    """Re-run the first edit on the transcribed seed and check the product."""
    if _replace_fragment is None or not expected:
        return True  # nothing to check against
    try:
        products = _replace_fragment(
            mol_smiles=new_seed,
            from_smiles=args.get("from_smiles"),
            to_smiles=args.get("to_smiles"),
            anchors=args.get("anchors"),
        )
    except Exception:
        return False
    return expected.strip() in {p.strip() for p in products}


def retranscribe_seed(input_data) -> Optional[str]:
    """Rewrite *input_data* in place so its seed follows the SMARTS atom order.

    Returns the new seed SMILES, or ``None`` when the chain was left as-is.
    """
    meta = input_data.metadata or {}
    if meta.get("fg_smarts") or meta.get("fg_constraint"):
        return None  # constructed FG carrier — not a transcription
    smarts = (meta.get("scaffold_smarts") or meta.get("smarts") or "").strip()
    seed = (meta.get("seed_smiles") or "").strip()
    if not seed and input_data.tool_chain:
        arg = input_data.tool_chain[0].tool_call.arguments.get("mol_smiles")
        seed = arg.strip() if isinstance(arg, str) else ""
    if not smarts or not seed:
        return None

    mapped = _index_map(seed, smarts)
    if mapped is None:
        return None
    new_seed, canon_to_new = mapped
    if new_seed == seed:
        return None  # already a transcription (~16% of chains) — nothing to do

    # A later step that lands back ON the seed would receive the canonical string
    # from the real tool, splitting the chain between two spellings. Rare; skip.
    for step in input_data.tool_chain:
        if (step.expected_response or "").strip() == seed:
            return None

    # Stage the rewrite; commit only once the first edit verifies.
    pending: list[tuple[object, dict]] = []          # (ToolCall, new arguments)
    pending_resp: list[tuple[object, int, str]] = []  # (step, slot, new response)
    first_edit: Optional[dict] = None
    first_edit_expected: Optional[str] = None

    for step in input_data.tool_chain:
        calls = [(step.tool_call, -1)] + [
            (c, i) for i, c in enumerate(step.parallel_tool_calls or [])
        ]
        for call, slot in calls:
            args = call.arguments or {}
            if args.get("mol_smiles") != seed:
                continue
            new_args = dict(args)
            new_args["mol_smiles"] = new_seed
            if "anchors" in new_args:
                remapped, ok = _remap_anchors(new_args["anchors"], canon_to_new)
                if not ok:
                    return None
                new_args["anchors"] = remapped
            for key in ("atom_index", "anchor_atom_index"):
                if new_args.get(key) is not None:
                    try:
                        new_args[key] = canon_to_new[int(new_args[key])]
                    except (KeyError, TypeError, ValueError):
                        return None
            pending.append((call, new_args))

            resp = (step.expected_response if slot < 0
                    else (step.parallel_expected_responses or [None] * (slot + 1))[slot])
            if call.name == "label_atom_indices":
                if _label_atom_indices is None:
                    return None
                try:
                    pending_resp.append((step, slot, _label_atom_indices(new_seed)))
                except Exception:
                    return None
            elif call.name == "suggest_edits":
                fixed = _remap_candidates(resp, canon_to_new)
                if fixed is None:
                    return None
                pending_resp.append((step, slot, fixed))
            elif call.name == "edit_fragment" and first_edit is None:
                first_edit, first_edit_expected = new_args, resp

    if not pending:
        return None
    if first_edit is not None and not _verify_first_edit(
            new_seed, first_edit, first_edit_expected):
        logger.debug("seed retranscription rejected — first edit did not reproduce")
        return None

    for call, new_args in pending:
        call.arguments = new_args
    for step, slot, resp in pending_resp:
        if slot < 0:
            step.expected_response = resp
        else:
            responses = list(step.parallel_expected_responses or [])
            while len(responses) <= slot:
                responses.append(None)
            responses[slot] = resp
            step.parallel_expected_responses = responses

    meta["seed_smiles_canonical"] = seed
    meta["seed_smiles"] = new_seed
    input_data.metadata = meta
    return new_seed
