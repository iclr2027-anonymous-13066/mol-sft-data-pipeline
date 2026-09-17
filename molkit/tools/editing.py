"""molkit/tools/editing.py — Molecule editing tools

Tools
-----
EditFragment      edit_fragment       attach / swap / remove a substituent (MMP-style)
AtomIndexLabeler  label_atom_indices  SMILES with each atom's map number = its 0-based index
SuggestEdits      suggest_edits       rank MMP edits by predicted gap-reduction toward a
                                      target property box (returns edit_fragment args)
"""

from __future__ import annotations

import logging

from rdkit import Chem

from molkit.tools.base import BaseTool

logger = logging.getLogger(__name__)


class EditFragment(BaseTool):
    """Unified fragment edit — attach, swap, or remove a substituent in one tool.

    A single matched-molecular-pair primitive ``from_smiles -> to_smiles``, where BOTH
    sides mark their attachment point(s) with a MAPPED dummy ``[*:1]`` (``[*:2]``/``[*:3]``
    for double/triple cuts); bare ``*`` is not accepted.  Attaching is the case where
    ``from_smiles`` is a lone attachment point ``[*:1]``; removing is
    ``to_smiles = "[*:1][H]"``.  The site — and, for a multi-cut, the orientation — is
    pinned with ``anchors``.  Everything runs through the ``replace_fragment`` engine.
    """

    name = "edit_fragment"
    description = (
        "Edit a substituent via a matched-molecular-pair transform from_smiles -> to_smiles, "
        "and return the resulting SMILES — one tool for attach, swap, and remove. BOTH "
        "from_smiles and to_smiles must mark their attachment point(s) with a MAPPED dummy "
        "atom '[*:1]' (and '[*:2]'/'[*:3]' for double/triple cuts); a bare '*' is not "
        "accepted, and the label set must match on both sides. "
        "ATTACH (grow): from_smiles='[*:1]' (a lone attachment point), to_smiles the fragment "
        "to add (e.g. '[*:1]C(=O)O'; use '[*:1]=O' for a double bond). "
        "SWAP: from_smiles='[*:1]Cl', to_smiles='[*:1]OC'. "
        "REMOVE (shrink): to_smiles='[*:1][H]'. "
        "If the fragment matches more than one site each distinct product is returned; pin the "
        "site — and, for a double/triple cut, the orientation (which core atom each labelled "
        "point maps to) — with anchors = {attachment-label: atom-index} using indices from "
        "label_atom_indices, e.g. {\"1\": 3, \"2\": 1}."
    )
    parameters = {
        "type": "object",
        "properties": {
            "mol_smiles": {"type": "string", "description": "SMILES of the molecule to edit."},
            "from_smiles": {
                "type": "string",
                "description": ("SMILES of the fragment to remove, marking its attachment "
                                "point(s) with a mapped dummy '[*:1]' (e.g. '[*:1]Cl'). Use "
                                "'[*:1]' — a lone attachment point — to ATTACH a new fragment."),
            },
            "to_smiles": {
                "type": "string",
                "description": ("SMILES of the replacement/new fragment, with a mapped dummy "
                                "'[*:1]' marking its attachment point (e.g. '[*:1]OC'; "
                                "'[*:1]=O' for a double bond). Use '[*:1][H]' to REMOVE."),
            },
            "anchors": {
                "type": "object",
                "additionalProperties": {"type": "integer"},
                "description": (
                    "Optional. {attachment-label: atom-index} (indices from "
                    "label_atom_indices) pinning every attachment point — fixes the "
                    "site and, for a double/triple cut, the orientation. Each index is "
                    "the CORE atom that STAYS, i.e. the atom just OUTSIDE from_smiles "
                    "that the dummy is bonded to — NOT an atom inside the fragment "
                    "being removed. E.g. to swap the benzyl phenyl of "
                    "'O=C1N(Cc2ccccc2)CSN1c1ccccc1' with from_smiles='[*:1]c1ccccc1', "
                    "the anchor is the CH2 (index 3), not the ring carbon (index 4). "
                    "When from_smiles is a lone '[*:1]' (an ATTACH), the anchor is "
                    "simply the atom you are attaching to. Omit to apply at all "
                    "matching sites."),
            },
        },
        "required": ["mol_smiles", "from_smiles", "to_smiles"],
    }

    examples = [
        {
            "input": {"mol_smiles": "c1ccccc1", "from_smiles": "[*:1]",
                      "to_smiles": "[*:1]C(=O)O", "anchors": {"1": 0}},
            "output": "Attached '[*:1]C(=O)O' (atom 0). Result: O=C(O)c1ccccc1",
        },
        {
            "input": {"mol_smiles": "Clc1ccccc1", "from_smiles": "[*:1]Cl", "to_smiles": "[*:1]OC"},
            "output": "Replaced '[*:1]Cl' with '[*:1]OC'. Result: COc1ccccc1",
        },
        {
            "input": {"mol_smiles": "Clc1ccccc1", "from_smiles": "[*:1]Cl", "to_smiles": "[*:1][H]"},
            "output": "Removed '[*:1]Cl'. Result: c1ccccc1",
        },
        # The anchor convention, which is where site-targeted edits go wrong:
        # index 3 is the CH2 the phenyl hangs off, not the phenyl's own ring
        # carbon (index 4). Pointing at atom 4 matches nothing.
        {
            "input": {"mol_smiles": "O=C1N(Cc2ccccc2)CSN1c1ccccc1",
                      "from_smiles": "[*:1]c1ccccc1",
                      "to_smiles": "[*:1]c1ccc(O)cc1", "anchors": {"1": 3}},
            "output": ("Replaced '[*:1]c1ccccc1' with '[*:1]c1ccc(O)cc1'. "
                       "Result: O=C1N(Cc2ccc(O)cc2)CSN1c1ccccc1"),
        },
    ]

    # Sentinel telling the two failure modes apart. Conflating them cost real
    # rollouts: a model that had already written '[*:1]' was told to "mark its
    # attachment point(s) with a mapped dummy", could not act on that, and
    # retried the same unparseable fragment for turn after turn. Measured on the
    # 1500-instance Qwen3-235B `wtool-ordered` run: 1603 of 3514 failed
    # edit_fragment calls (18.8% of ALL calls) got this message, and hand-written
    # aromatic-H substitutions ('[*:1][cH]', '[*:1]cO', '[*:1][cH:5]' — every one
    # an RDKit PARSE failure, not a mapping problem) succeeded 39 times out of 480.
    _UNPARSEABLE = object()

    @classmethod
    def _mapped_labels(cls, frag: str):
        """Sorted list of mapped [*:N] labels on *frag*'s dummy atoms.

        Returns ``_UNPARSEABLE`` when *frag* is not a parseable SMILES, and ``[]``
        when it parses but carries no mapped dummy (bare '*', or no dummy at all).
        Both are errors — this tool requires every attachment point to be mapped
        ('[*:1]', not '*') — but they need different messages, so the caller must
        distinguish them.
        """
        m = Chem.MolFromSmiles(frag)
        if m is None:
            return cls._UNPARSEABLE
        labels = []
        for a in m.GetAtoms():
            if a.GetAtomicNum() == 0:                 # attachment dummy
                mn = a.GetAtomMapNum()
                if mn == 0:
                    return []                         # bare/unmapped dummy not allowed
                labels.append(mn)
        return sorted(labels)

    @classmethod
    def _fragment_error(cls, side: str, frag: str, labels) -> str | None:
        """The error string for a rejected *side* fragment, or None if it is fine."""
        if labels is cls._UNPARSEABLE:
            return (f"SMILES Syntax Error: the '{side}' fragment '{frag}' is not a parseable "
                    "SMILES. Common causes: an aromatic atom written outside a ring (use "
                    "'[*:1]c1ccccc1', not '[*:1][cH]' or '[*:1]cO'), a ring that cannot be "
                    "kekulised (check the ring size and the atom count), or an atom map on a "
                    "non-dummy atom (write '[*:1]', not '[cH:5]'). To substitute a hydrogen on "
                    "an existing atom, attach with from_smiles='[*:1]' and give that atom's "
                    "index in 'anchors'.")
        if not labels:
            hint = ("To attach, use '[*:1]'." if side == "from_smiles"
                    else "To remove, use '[*:1][H]'.")
            return (f"Input Argument Error: '{side}' must mark its attachment point(s) with a "
                    f"mapped dummy like '[*:1]' (bare '*' not accepted). {hint}")
        return None

    @staticmethod
    def _is_open(frag: str) -> bool:
        """True when *frag* is only dummies/hydrogens (a lone attachment point)."""
        m = Chem.MolFromSmiles(frag)
        return m is not None and all(a.GetAtomicNum() in (0, 1) for a in m.GetAtoms())

    def execute(self, mol_smiles: str, from_smiles: str, to_smiles: str,
                anchors: dict | None = None) -> str | dict:
        from molkit.utils import is_smiles
        from molkit.utils.molecule_edit_utils import replace_fragment

        if not mol_smiles or not isinstance(mol_smiles, str):
            return "SMILES Syntax Error: 'mol_smiles' must be a non-empty string."
        if not is_smiles(mol_smiles):
            return f"SMILES Syntax Error: '{mol_smiles}' is not a valid SMILES string."
        if not from_smiles or not isinstance(from_smiles, str):
            return "Input Argument Error: 'from_smiles' must be a non-empty string."
        if not to_smiles or not isinstance(to_smiles, str):
            return "Input Argument Error: 'to_smiles' must be a non-empty string."

        # Require mapped attachment points ('[*:1]', not bare '*') on both sides.
        # An unparseable fragment gets its own message — see _mapped_labels.
        from_labels = self._mapped_labels(from_smiles)
        to_labels = self._mapped_labels(to_smiles)
        err = (self._fragment_error("from_smiles", from_smiles, from_labels)
               or self._fragment_error("to_smiles", to_smiles, to_labels))
        if err:
            return err
        if set(from_labels) != set(to_labels):
            return (f"Input Argument Error: from_smiles labels {from_labels} must match to_smiles "
                    f"labels {to_labels} — the same attachment points on both sides.")

        # Validate anchors reference real attachment labels (friendly upfront error).
        if anchors is not None:
            try:
                anchors_i = {int(k): int(v) for k, v in anchors.items()}
            except (AttributeError, TypeError, ValueError):
                return "Input Argument Error: 'anchors' must be {attachment-label: atom-index}, e.g. {\"1\": 3}."
            bad = sorted(k for k in anchors_i if k not in from_labels)
            if bad:
                return (f"Input Argument Error: anchors label(s) {bad} are not attachment points of "
                        f"the fragment (its labels are {from_labels}).")

        products = replace_fragment(mol_smiles, from_smiles, to_smiles, anchors=anchors)
        if not products:
            # Re-run without the anchors: if it applies then, the fragment WAS
            # found and only the requested site was wrong — nearly always an
            # anchor pointing at an atom inside `from_smiles` instead of the core
            # atom it hangs off. Saying so is the difference between a fixable
            # error and a retry loop (measured: 20.1% of all edit_fragment calls
            # in the 1500-instance ordered run returned this message).
            hint = ""
            if anchors:
                try:
                    if replace_fragment(mol_smiles, from_smiles, to_smiles):
                        hint = (f" The fragment '{from_smiles}' IS present — it is the "
                                f"'anchors' {anchors} that match no site. Each anchor "
                                "index must be the CORE atom that stays (the atom just "
                                "outside the fragment that the dummy bonds to), not an "
                                "atom inside the fragment. Omit 'anchors' to apply at "
                                "every matching site.")
                except Exception:                                    # noqa: BLE001
                    pass
            return (f"Execution Error: could not apply '{from_smiles}' -> '{to_smiles}' on "
                    f"'{mol_smiles}' (fragment not found, invalid product, or requested "
                    f"site/orientation not present).{hint}")

        mode = ("attach" if self._is_open(from_smiles)
                else "remove" if self._is_open(to_smiles) else "swap")
        return {"smiles": products[0], "products": products, "mode": mode,
                "from_smiles": from_smiles, "to_smiles": to_smiles}

    def __call__(self, inputs: dict, return_text: bool = True) -> str:
        result = self.execute(**inputs)
        if isinstance(result, str):
            return result
        if not return_text:
            return result["smiles"]
        n = len(result["products"])
        extra = "" if n == 1 else f" ({n} products; showing the first)"
        if result["mode"] == "attach":
            return f"Attached '{result['to_smiles']}'{extra}. Result: {result['smiles']}"
        if result["mode"] == "remove":
            return f"Removed '{result['from_smiles']}'{extra}. Result: {result['smiles']}"
        return (f"Replaced '{result['from_smiles']}' with '{result['to_smiles']}'{extra}. "
                f"Result: {result['smiles']}")


class AtomIndexLabeler(BaseTool):
    """Return a mapped SMILES where each atom is annotated with its 0-based index."""

    name = "label_atom_indices"
    description = (
        "Return a mapped SMILES where each atom is annotated with its 0-based index "
        "as the atom map number (e.g. [C:0], [O:1]). "
        "Use this tool to identify which atom index to pass in edit_fragment's 'anchors'."
    )
    parameters = {
        "type": "object",
        "properties": {
            "mol_smiles": {
                "type": "string",
                "description": "SMILES string of the molecule to label.",
            },
        },
        "required": ["mol_smiles"],
    }

    examples = [
        {
            "input": {"mol_smiles": "CCO"},
            "output": "Atom-indexed SMILES: [CH3:0][CH2:1][OH:2]",
        },
        {
            "input": {"mol_smiles": "c1ccccc1"},
            "output": "Atom-indexed SMILES: [cH:0]1[cH:1][cH:2][cH:3][cH:4][cH:5]1",
        },
    ]

    def execute(self, mol_smiles: str) -> str:
        from molkit.utils import is_smiles
        from molkit.utils.molecule_edit_utils import label_atom_indices

        if not mol_smiles or not isinstance(mol_smiles, str):
            return "SMILES Syntax Error: 'mol_smiles' must be a non-empty string."
        if not is_smiles(mol_smiles):
            return f"SMILES Syntax Error: '{mol_smiles}' is not a valid SMILES string."

        return label_atom_indices(mol_smiles)

    def __call__(self, inputs: dict, return_text: bool = True) -> str:
        result = self.execute(**inputs)
        if return_text and not result.startswith(
            ("SMILES Syntax Error", "Input Argument Error", "Execution Error")
        ):
            return f"Atom-indexed SMILES: {result}"
        return result


# Candidate count served to every caller of the TOOL, whatever `top_k` they send.
# The ordered eval commits exactly one candidate per turn, so the length of the menu
# is a property of the experiment rather than a knob the model may turn: a model that
# asks for 10 chooses from a wider list than one that asks for 3, and the two runs are
# then not comparable. 4 is what every in-repo caller already uses (the stage-3 CLI's
# --suggest-top-k default and search_plan's top_k both being 4), so pinning it here
# changes no existing behaviour -- it only removes the model's ability to widen it.
# Callers that legitimately want another count use the FUNCTION,
# molkit.utils.suggest_edits.suggest_edits, which is untouched and is what stage-3
# actually calls; this pin applies to the tool-server path only.
_FORCED_TOP_K = 4


class SuggestEdits(BaseTool):
    """Rank mmpdb-derived edits by how much they close the gap to a target property
    box, returned as ready-to-use ``edit_fragment`` arguments.

    Given a molecule and per-property ``[lo, hi]`` constraints, this measures the
    molecule, scores every mmpdb move (both directions) by the predicted reduction
    in the normalised (z-score) distance to the box, verifies applicability, and
    returns the top-K as ``{from_smiles, to_smiles, anchors, current_gap,
    predicted_gap}`` — each directly consumable by ``edit_fragment``.
    """

    name = "suggest_edits"
    description = (
        "Suggest the substituent edits that most move a molecule toward a target "
        "property box, ranked by predicted gap-reduction. Give 'constraints' as "
        "{property: [lo, hi]} (use null for an open bound); allowed properties: MW, "
        "logP, HBD, HBA, TPSA, rotB, rings_total, QED, MR, heavy_atoms, "
        "logD, logS, BBBP, Mutag. Returns up to 4 candidates, each an "
        "{from_smiles, to_smiles, anchors} triple usable verbatim as edit_fragment "
        "arguments, plus predicted_gap (normalised box distance after the edit) and "
        "delta (predicted change as {avg, std} for every property you constrained, in "
        "that order — a Δ of 0 means the edit leaves that property alone). Pass "
        "'scaffold_smarts' to keep only edits whose product still matches it "
        "(preserve the core). It may be a list of SMARTS, and then every one must "
        "survive — required when more than one substructure has to be kept."
    )
    parameters = {
        "type": "object",
        "properties": {
            "mol_smiles": {"type": "string",
                           "description": "SMILES of the molecule to improve."},
            "constraints": {
                "type": "object",
                "description": ("Target property box {property: [lo, hi]}; use null for "
                                "an open bound (e.g. {\"logP\": [1.0, 3.0], \"QED\": "
                                "[0.6, null]})."),
                "additionalProperties": {
                    "type": "array",
                    "items": {"type": ["number", "null"]},
                    "minItems": 2, "maxItems": 2,
                },
            },
            "top_k": {"type": "integer", "default": 4,
                      "description": ("Number of candidate edits to return. Fixed at 4 "
                                      "for this tool — any other value is accepted and "
                                      "then ignored, so you always get the same 4-item "
                                      "menu to choose from.")},
            "max_cut": {"type": "integer", "enum": [1, 2, 3], "default": 3,
                        "description": ("Max attachment points per edit: 1=single, "
                                        "2=+double, 3=+triple cut (default 3).")},
            "scaffold_smarts": {
                "type": ["string", "array"],
                "items": {"type": "string"},
                "description": ("Optional. Only return edits whose product still matches "
                                "this SMARTS, i.e. edits that preserve the substructure "
                                "you want to keep. Give a LIST to require several at "
                                "once; each is checked separately, so do NOT join them "
                                "with '.' (a dot-joined query demands atom-disjoint "
                                "matches and would reject groups that share an atom)."),
            },
        },
        "required": ["mol_smiles", "constraints"],
    }

    examples = [
        {
            "input": {"mol_smiles": "CCOc1ccccc1",
                      "constraints": {"logP": [3.0, 5.0], "MW": [180, 260]}, "top_k": 4},
            "output": ("Top 4 edits (each usable as edit_fragment args):\n"
                       "1. from_smiles='[*:1]' to_smiles='[*:1]c1ccccc1' anchors={'1': 0}  "
                       "predicted_gap 0.0  Δ{MW +76.1±2.3, logP +1.7±0.4}"),
        },
    ]

    def execute(self, mol_smiles: str, constraints: dict, top_k: int = 4,
                max_cut: int = 3,
                scaffold_smarts: str | list | None = None) -> str | list:
        from molkit.utils import is_smiles
        from molkit.utils.suggest_edits import suggest_edits

        if not mol_smiles or not isinstance(mol_smiles, str):
            return "SMILES Syntax Error: 'mol_smiles' must be a non-empty string."
        if not is_smiles(mol_smiles):
            return f"SMILES Syntax Error: '{mol_smiles}' is not a valid SMILES string."
        if not isinstance(constraints, dict) or not constraints:
            return ("Input Argument Error: 'constraints' must be a non-empty object like "
                    "{\"logP\": [1.0, 3.0]}.")
        for k, rng in constraints.items():
            if not isinstance(rng, (list, tuple)) or len(rng) != 2:
                return (f"Input Argument Error: constraint '{k}' must be a [lo, hi] pair "
                        "(use null for an open bound).")
        try:
            top_k, max_cut = int(top_k), int(max_cut)
        except (TypeError, ValueError):
            return "Input Argument Error: 'top_k' and 'max_cut' must be integers."
        # Whatever was asked for, serve _FORCED_TOP_K. A wrong-typed top_k is still an
        # argument error above rather than being silently swallowed, and `top_k` stays
        # in the signature and in the schema on purpose: stage-3 records it in every
        # stored suggest_edits call (search_plan._make_step) and the trained model
        # reproduces that argument, so removing it would put inference out of step with
        # the training format.
        top_k = _FORCED_TOP_K
        if scaffold_smarts is not None and not isinstance(scaffold_smarts, (str, list)):
            return ("Input Argument Error: 'scaffold_smarts' must be a SMARTS string "
                    "or a list of SMARTS strings.")
        if isinstance(scaffold_smarts, list) and not all(
                isinstance(x, str) for x in scaffold_smarts):
            return "Input Argument Error: 'scaffold_smarts' list must hold only strings."
        try:
            return suggest_edits(mol_smiles, constraints, top_k=top_k, max_cut=max_cut,
                                 scaffold_smarts=scaffold_smarts or None)
        except ValueError as e:
            return f"Input Argument Error: {e}"
        except Exception as e:  # noqa: BLE001
            return f"Execution Error: {e}"

    @classmethod
    def _format_text(cls, result: list) -> str:
        """Render the candidate list as the model reads it.

        Split out of ``__call__`` (mirroring ``MolPropAnalyzer._format_text``) so a
        caller that wants BOTH forms can execute once at ``return_text=False`` and
        render the prose locally. Ranking these candidates runs mmpdb moves and a
        property evaluation, so calling the tool twice to see both forms is not
        the microsecond it is for the RDKit tools.
        """
        if not result:
            return ("No gap-reducing edits found (the molecule may already satisfy the "
                    "constraints, or no mmpdb move applies).")
        lines = [f"Top {len(result)} edits (each usable as edit_fragment args):"]
        for i, c in enumerate(result, 1):
            d = c.get("delta", {})
            # No truncation: delta now carries exactly the properties the caller
            # constrained, so every entry is one they asked about.
            dstr = ", ".join(f"{p} {v['avg']:+g}±{v['std']:g}" for p, v in d.items())
            lines.append(f"{i}. from_smiles={c['from_smiles']!r} "
                         f"to_smiles={c['to_smiles']!r} anchors={c['anchors']}  "
                         f"predicted_gap {c['predicted_gap']}  Δ{{{dstr}}}")
        return "\n".join(lines)

    def __call__(self, inputs: dict, return_text: bool = True):
        result = self.execute(**inputs)
        if isinstance(result, str):          # error string
            return result
        if not return_text:
            return result
        return self._format_text(result)
