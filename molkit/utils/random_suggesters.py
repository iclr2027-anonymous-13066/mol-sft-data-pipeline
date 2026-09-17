"""Vocabulary-only edit suggesters — the ablation baselines for ``suggest_edits``.

:mod:`molkit.utils.suggest_edits` does two separable things: it **enumerates** the
edits that apply to a molecule (mmpdb from-fragments matched at anchor sites) and then
**ranks** them by the predicted normalised reduction in the distance to the target
property box (the mmpdb Δ table, refined by the matched environment). This module keeps
the enumeration and throws the ranking away: each suggester enumerates a *fixed,
property-blind* edit vocabulary, drops every product that breaks the guard, and returns
``top_k`` of the survivors **uniformly at random**.

Three vocabularies, one union:

=================  ==========================================================
``atom``           attach one heavy atom (:data:`ATOM_ATTACH`) at any position
                   with a free hydrogen
``fg``             attach one substituent from :data:`FG_CATALOG` (a fixed
                   catalog of common medicinal-chemistry groups) likewise
``delete``         delete one terminal heavy atom
``mix``            the union of the three pools, sampled as one
=================  ==========================================================

Everything downstream of the vocabulary is shared with the real tool ON PURPOSE, so a
comparison isolates the ranking rather than a pile of incidental differences:

* products are built by ``suggest_edits._products_by_anchor`` — the same
  ``RunReactants`` path, the same clean-swap heavy-atom check, the same rejection of
  unsanitisable / unchanged / fragmented products;
* the guard is the same test on the same object (``HasSubstructMatch`` on the product,
  every pattern required separately);
* candidates come back in the same ``edit_fragment``-ready shape
  (``from_smiles`` / ``to_smiles`` / ``anchors``), pinned to one site, and re-applying
  one with ``replace_fragment(anchors=...)`` reproduces the product byte-for-byte.

What is deliberately ABSENT is ``delta``: these suggesters never look at
``constraints``, so ``predicted_gap`` is ``None``. A caller that wants to know whether a
candidate helps has to measure it.

This module does not import from, patch, or otherwise alter the behaviour of
``suggest_edits`` — it only calls two of its pure helpers.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple, Union

from rdkit import Chem, RDLogger

from molkit.utils.molecule_edit_utils import _map_attachments

RDLogger.DisableLog("rdApp.*")

MODES = ("atom", "fg", "delete", "mix")

# Single heavy atoms attached by a SINGLE bond — the elements that make up
# essentially all of drug-like chemical space. Bond order is fixed to single, so
# "attach an element" stays one unambiguous operation.
ATOM_ATTACH: Tuple[Tuple[str, str], ...] = (
    ("methyl (C)", "[*:1]C"),
    ("amino (N)", "[*:1]N"),
    ("hydroxy (O)", "[*:1]O"),
    ("fluoro (F)", "[*:1]F"),
    ("thio (S)", "[*:1]S"),
    ("chloro (Cl)", "[*:1]Cl"),
    ("bromo (Br)", "[*:1]Br"),
    ("iodo (I)", "[*:1]I"),
)

# A fixed catalog of common medicinal-chemistry substituents. Chosen to span the
# axes the property box actually constrains — size, lipophilicity, H-bonding,
# aromatic/aliphatic ring count, charge — rather than to be exhaustive. Every entry
# carries exactly one ``[*:1]``.
FG_CATALOG: Tuple[Tuple[str, str], ...] = (
    # small alkyl / cycloalkyl
    ("methyl", "[*:1]C"),
    ("ethyl", "[*:1]CC"),
    ("isopropyl", "[*:1]C(C)C"),
    ("tert-butyl", "[*:1]C(C)(C)C"),
    ("cyclopropyl", "[*:1]C1CC1"),
    ("vinyl", "[*:1]C=C"),
    ("ethynyl", "[*:1]C#C"),
    # halogen / halogenated
    ("fluoro", "[*:1]F"),
    ("chloro", "[*:1]Cl"),
    ("bromo", "[*:1]Br"),
    ("trifluoromethyl", "[*:1]C(F)(F)F"),
    ("trifluoromethoxy", "[*:1]OC(F)(F)F"),
    # oxygen
    ("hydroxy", "[*:1]O"),
    ("methoxy", "[*:1]OC"),
    ("ethoxy", "[*:1]OCC"),
    ("hydroxymethyl", "[*:1]CO"),
    ("acetyl", "[*:1]C(C)=O"),
    ("carboxylic acid", "[*:1]C(=O)O"),
    ("methyl ester", "[*:1]C(=O)OC"),
    ("acetoxy", "[*:1]OC(C)=O"),
    # nitrogen
    ("amino", "[*:1]N"),
    ("methylamino", "[*:1]NC"),
    ("dimethylamino", "[*:1]N(C)C"),
    ("primary carboxamide", "[*:1]C(N)=O"),
    ("N-methylcarboxamide", "[*:1]C(=O)NC"),
    ("acetamido", "[*:1]NC(C)=O"),
    ("nitrile", "[*:1]C#N"),
    ("nitro", "[*:1][N+](=O)[O-]"),
    ("aminomethyl", "[*:1]CN"),
    ("urea", "[*:1]NC(N)=O"),
    # sulfur
    ("methylthio", "[*:1]SC"),
    ("methylsulfonyl", "[*:1]S(C)(=O)=O"),
    ("sulfonamide", "[*:1]S(N)(=O)=O"),
    # saturated N/O heterocycles
    ("morpholin-4-yl", "[*:1]N1CCOCC1"),
    ("piperidin-1-yl", "[*:1]N1CCCCC1"),
    ("piperazin-1-yl", "[*:1]N1CCNCC1"),
    ("pyrrolidin-1-yl", "[*:1]N1CCCC1"),
    ("azetidin-1-yl", "[*:1]N1CCC1"),
    ("tetrahydropyran-4-yl", "[*:1]C1CCOCC1"),
    # aromatics / heteroaromatics
    ("phenyl", "[*:1]c1ccccc1"),
    ("4-fluorophenyl", "[*:1]c1ccc(F)cc1"),
    ("pyridin-3-yl", "[*:1]c1cccnc1"),
    ("pyridin-4-yl", "[*:1]c1ccncc1"),
    ("pyrimidin-2-yl", "[*:1]c1ncccn1"),
    ("imidazol-1-yl", "[*:1]n1ccnc1"),
    ("pyrazol-1-yl", "[*:1]n1cccn1"),
    ("1,2,4-triazol-1-yl", "[*:1]n1cncn1"),
    ("tetrazol-5-yl", "[*:1]c1nnn[nH]1"),
    ("thiophen-2-yl", "[*:1]c1cccs1"),
    ("furan-2-yl", "[*:1]c1ccco1"),
    ("1,3,4-oxadiazol-2-yl", "[*:1]c1nnco1"),
)

# The bare attachment point: a ``from`` fragment that removes nothing and hangs a new
# bond off the anchor (needs a free hydrogen there — the product simply fails to
# sanitise otherwise, which the product builder already rejects).
_OPEN = "[*:1]"
_REMOVE = "[*:1][H]"

# Blow-up bound on the enumerated pool. Reached only by ``fg``/``mix`` on large
# molecules (sites x |catalog|); the rule order is shuffled BEFORE enumeration, so
# truncating at the cap keeps the sample uniform over vocabulary rather than biased
# towards whichever fragment happens to sort first.
POOL_CAP = 6000

Guard = Union[str, Sequence[str], None]


# ---------------------------------------------------------------------------
# vocabularies
# ---------------------------------------------------------------------------
def _guard_mols(scaffold_smarts: Guard) -> List[Chem.Mol]:
    """Parse the guard spec into query mols — same contract as the real tool: a bare
    string is one pattern, a sequence is several and EVERY one must survive."""
    if not scaffold_smarts:
        return []
    items = [scaffold_smarts] if isinstance(scaffold_smarts, str) else list(scaffold_smarts)
    out = []
    for s in items:
        if not s:
            continue
        g = Chem.MolFromSmarts(s)
        if g is None:
            raise ValueError(f"Invalid scaffold_smarts: {s}")
        out.append(g)
    return out


def _terminal_fragment(base: Chem.Mol, atom: Chem.Atom) -> Optional[str]:
    """``from_smiles`` that matches *atom* as a terminal substituent, or None.

    Built as a real 2-atom molecule (dummy + a copy of the atom, joined by the bond
    actually present) rather than string-formatted, so charge, isotope and bond order
    come out right without a per-element special case: a nitro oxygen becomes
    ``[*:1][O-]`` and a carbonyl oxygen ``O=[*:1]``, both of which the product builder
    then applies as the reaction ``<frag> >> [*:1][H]``.

    The fragment is a PATTERN, not a unique site: ``[*:1]C`` matches every terminal
    methyl in the molecule. That is fine and cheap — the product builder enumerates
    every match in one reaction and keys the products by anchor, and the caller keeps
    the anchors that pinned each one.
    """
    nbrs = atom.GetNeighbors()
    if len(nbrs) != 1:
        return None
    bond = base.GetBondBetweenAtoms(atom.GetIdx(), nbrs[0].GetIdx())
    if bond is None or bond.GetBondType() == Chem.BondType.AROMATIC:
        return None                       # a degree-1 atom is never in a ring
    rw = Chem.RWMol()
    d = rw.AddAtom(Chem.Atom(0))
    rw.GetAtomWithIdx(d).SetAtomMapNum(1)
    a = Chem.Atom(atom.GetAtomicNum())
    a.SetFormalCharge(atom.GetFormalCharge())
    a.SetIsotope(atom.GetIsotope())
    i = rw.AddAtom(a)
    rw.AddBond(d, i, bond.GetBondType())
    frag = rw.GetMol()
    try:
        Chem.SanitizeMol(frag)
    except Exception:                      # noqa: BLE001 - unrepresentable fragment
        return None
    return _map_attachments(Chem.MolToSmiles(frag))


def rules_for(base: Chem.Mol, mode: str) -> List[Tuple[str, str, str]]:
    """The ``(label, from_smiles, to_smiles)`` rules this *mode* offers on *base*.

    ``atom`` / ``fg`` are molecule-independent (attach at the open attachment point);
    ``delete`` depends on which terminal atoms the molecule actually has.
    """
    if mode == "atom":
        return [(n, _OPEN, t) for n, t in ATOM_ATTACH]
    if mode == "fg":
        return [(n, _OPEN, t) for n, t in FG_CATALOG]
    if mode == "delete":
        seen: Dict[str, None] = {}
        for a in base.GetAtoms():
            if a.GetAtomicNum() <= 1 or a.GetDegree() != 1:
                continue
            frag = _terminal_fragment(base, a)
            if frag is not None:
                seen.setdefault(frag, None)
        return [(f"delete {f}", f, _REMOVE) for f in seen]
    if mode == "mix":
        out: List[Tuple[str, str, str]] = []
        for m in ("atom", "fg", "delete"):
            out.extend(rules_for(base, m))
        return out
    raise ValueError(f"unknown mode '{mode}'; choose from {MODES}")


# ---------------------------------------------------------------------------
# enumeration
# ---------------------------------------------------------------------------
def enumerate_candidates(mol_smiles: str, mode: str, *,
                         scaffold_smarts: Guard = None,
                         rng: Optional[random.Random] = None,
                         pool_cap: int = POOL_CAP,
                         with_products: bool = True) -> List[dict]:
    """Every guard-passing edit of this *mode* on *mol_smiles*, in random rule order.

    Each element is ``{label, from_smiles, to_smiles, anchors, mol}`` with ``anchors``
    pinning the site (string keys, as ``edit_fragment`` takes them) and ``mol`` the
    built product. ``with_products`` additionally canonicalises each product to a
    ``product`` SMILES; turning it off is worth it when only a few of the pool will be
    used, since canonicalisation is the single most expensive step per candidate.

    Products come from ``suggest_edits._products_by_anchor``: one ``RunReactants`` per
    rule covering every site at once, with its clean-swap heavy-atom check, so a rule
    that would delete collateral atoms at a site yields nothing there. (That helper
    shares its compiled-reaction cache with the real tool. The cache is keyed by
    ``(from, to)`` and its entries are deterministic, so the two cannot affect each
    other's results — at worst they evict each other's compiled reactions.)
    """
    from molkit.utils.suggest_edits import _products_by_anchor

    if not mol_smiles or not isinstance(mol_smiles, str):
        raise ValueError("mol_smiles must be a non-empty SMILES string")
    base = Chem.MolFromSmiles(mol_smiles)
    if base is None or base.GetNumAtoms() == 0:
        raise ValueError(f"Invalid SMILES: {mol_smiles}")
    guards = _guard_mols(scaffold_smarts)
    in_canon = Chem.MolToSmiles(base)

    rules = rules_for(base, mode)
    if rng is not None:
        rng.shuffle(rules)

    out: List[dict] = []
    for label, mv_from, mv_to in rules:
        if len(out) >= pool_cap:
            break
        for key, prod in _products_by_anchor(base, in_canon, mv_from, mv_to).items():
            if guards and not all(prod.HasSubstructMatch(g) for g in guards):
                continue
            cand = {
                "label": label,
                "from_smiles": mv_from,
                "to_smiles": mv_to,
                "anchors": {str(k): int(v) for k, v in key},
                "mol": prod,
            }
            if with_products:
                cand["product"] = Chem.MolToSmiles(prod)
            out.append(cand)
            if len(out) >= pool_cap:
                break
    return out


def suggest_edits_random(mol_smiles: str, constraints: Optional[dict] = None,
                         top_k: int = 4, *, mode: str = "mix",
                         scaffold_smarts: Guard = None,
                         rng: Optional[random.Random] = None,
                         pool_cap: int = POOL_CAP,
                         keep_product: bool = True,
                         **_ignored) -> List[dict]:
    """``top_k`` guard-passing edits drawn UNIFORMLY AT RANDOM from *mode*'s vocabulary.

    Signature-compatible with :func:`molkit.utils.suggest_edits.suggest_edits` so a
    harness can swap one for the other, but *constraints* (and ``props`` / ``max_cut`` /
    ``context_aware`` / ``return_delta``, absorbed by ``**_ignored``) are NOT USED: that
    is the whole point of the ablation. ``predicted_gap`` is therefore ``None`` and no
    ``delta`` key is emitted.

    Returns fewer than *top_k* elements when the vocabulary offers fewer guard-passing
    edits, and ``[]`` when it offers none. Every returned candidate is known to apply
    (its product was built to produce it).
    """
    pool = enumerate_candidates(mol_smiles, mode, scaffold_smarts=scaffold_smarts,
                                rng=rng, pool_cap=pool_cap, with_products=False)
    if not pool:
        return []
    k = max(0, min(int(top_k), len(pool)))
    picked = (rng.sample(pool, k) if rng is not None else pool[:k])
    out = []
    for c in picked:
        cand = {"from_smiles": c["from_smiles"], "to_smiles": c["to_smiles"],
                "anchors": dict(c["anchors"]), "predicted_gap": None,
                "label": c["label"]}
        if keep_product:
            cand["product"] = Chem.MolToSmiles(c["mol"])
        out.append(cand)
    return out
