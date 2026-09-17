"""Utility functions backing the molecule-edit tools.

Exposes:
  - replace_fragment:    apply a matched-molecular-pair edit from_smiles -> to_smiles
                         (swap A->B; attach = lone "[*:1]" from-fragment; remove =
                         "[*:1][H]" to-fragment), optionally pinned with ``anchors``.
  - label_atom_indices:  return a SMILES with each atom's map number = its 0-based index.
"""

import re
from typing import Dict, List, Optional, Tuple

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog('rdApp.*')


def _mol_from_smiles(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return mol




def _map_attachments(frag: str) -> str:
    """Ensure each attachment dummy is map-numbered ([*:1], [*:2], ...).

    Accepts fragments written with bare ``*``/``[*]`` or already-mapped
    ``[*:1]``.  Bare dummies are numbered left-to-right; already-mapped dummies
    are left untouched (so callers can control multi-attachment pairing).
    """
    if "[*:" in frag:
        return frag  # already mapped — trust the caller's pairing
    counter = {"n": 0}
    def repl(_m):
        counter["n"] += 1
        return f"[*:{counter['n']}]"
    # match [*] or a bare * not already in brackets
    return re.sub(r"\[\*\]|\*", repl, frag)



def _nondummy_heavy(frag: str) -> int:
    """Heavy-atom count of a fragment SMILES/SMARTS, excluding attachment dummies.
    Returns -1 if the fragment cannot be parsed (caller then skips the guard)."""
    m = Chem.MolFromSmiles(frag)
    if m is None:
        m = Chem.MolFromSmarts(frag)
    if m is None:
        return -1
    return sum(1 for a in m.GetAtoms() if a.GetAtomicNum() > 1)



def _target_removed_sets(base, patt, site_map, anchor_atom_index):
    """Molecule-atom sets removed by each ``from`` match that satisfies the
    requested site targeting.

    ``patt`` is the attachment-dummy-preserving, molecule-map-stripped any-atom
    query; ``site_map`` maps a pattern-atom index to the molecule atom index it
    must hit (from non-dummy atom maps in ``from_smiles``); ``anchor_atom_index``
    requires the core atom the fragment stays attached to (matched by a dummy) to
    be that index.  Each returned frozenset is the non-dummy matched atoms of one
    valid site — used to pick the matching RunReactants product via its
    ``react_atom_idx`` provenance.
    """
    if patt is None:
        return set()
    dummy_idx = {a.GetIdx() for a in patt.GetAtoms() if a.GetAtomicNum() == 0}
    out = set()
    for m in base.GetSubstructMatches(patt):
        if any(m[p] != mi for p, mi in site_map.items()):
            continue
        if anchor_atom_index is not None and not any(m[d] == anchor_atom_index for d in dummy_idx):
            continue
        out.add(frozenset(m[i] for i in range(len(m)) if i not in dummy_idx))
    return out



def replace_fragment(
    mol_smiles: str,
    from_smiles: str,
    to_smiles: str,
    max_products: int = 20,
    anchor_atom_index: Optional[int] = None,
    anchors: Optional[Dict[int, int]] = None,
) -> List[str]:
    """Apply an MMP-style substituent swap ``from_smiles`` → ``to_smiles``.

    Both fragments mark their attachment point(s) with a dummy atom: ``*`` /
    ``[*]`` (numbered left-to-right) or explicit ``[*:1]`` / ``[*:2]`` (e.g. the
    raw rules from an mmpdb database).  The swap is applied as an RDKit reaction,
    so the attachment atom(s) of the core are preserved and only the variable
    part changes — exactly how matched-molecular-pair transforms apply.

    **Collateral-deletion guard (always on):** an RDKit reaction deletes every
    matched ``from`` atom and anything hanging off it that is not the declared
    attachment, so a ``from`` fragment that under-specifies the real substituent
    (e.g. ``[*:1]c1ccccc1OC`` matching a phenyl whose ring carbon actually bears a
    longer ``O-CH2-C(F)(F)C(F)F`` ether) would silently drop the extra atoms.
    Such products are rejected: only products whose heavy-atom count equals
    ``heavy(mol) - heavy(from) + heavy(to)`` — a clean single-attachment swap that
    changed exactly the named fragment — are returned.

    **Optional site targeting (off by default):** the swap normally applies at
    every match site.  To pin it to one site, either
      * pass ``anchor_atom_index`` — the core atom the fragment stays attached to
        (an index from ``label_atom_indices``); or
      * give ``from_smiles`` non-dummy atoms atom-map numbers equal to their
        molecule indices, transcribed faithfully from ``label_atom_indices``
        output **including H counts** (e.g.
        ``[*:1][c:3]1[cH:4][cH:5][cH:6][cH:7][c:8]1[O:9][CH3:10]``).  Map ``0``
        cannot be targeted this way (RDKit treats ``:0`` as unmapped) — use
        ``anchor_atom_index`` for atom 0.
      * pass ``anchors`` — a ``{attachment-label: core-atom-index}`` map, where the
        labels are the ``[*:1]``/``[*:2]``/``[*:3]`` numbers of ``from_smiles`` (bare
        ``*`` are numbered left-to-right) and the indices come from
        ``label_atom_indices``.  Unlike ``anchor_atom_index`` (which pins only ONE
        attachment point), ``anchors`` pins EVERY listed point, so for a double/triple
        cut it fixes both the site AND the orientation — i.e. which core atom each
        labelled attachment maps to.  This matters when the two directions give
        different products (e.g. ``[*:1]O[*:2]`` -> ``[*:1]OC[*:2]`` on ``Et-O-Ph``:
        the CH2 lands on the Et side or the Ph side depending on the pairing).
    When targeting is requested and no clean matching site exists, ``[]`` is
    returned rather than a swap at some other site.

    Returns the list of unique, sanitisable, single-fragment product SMILES that
    differ from the input (one per distinct match site), capped at *max_products*.
    """
    base = Chem.MolFromSmiles(mol_smiles)
    if base is None:
        return []
    in_canon = Chem.MolToSmiles(base)

    # Normalise `anchors` up front: coerce keys/values to int (tool calls arrive as
    # JSON with string keys) and reject out-of-range atom indices. An invalid map
    # yields [] rather than silently applying at some other site.
    if anchors is not None:
        try:
            anchors = {int(k): int(v) for k, v in anchors.items()}
        except (AttributeError, TypeError, ValueError):
            return []
        if any(not (0 <= v < base.GetNumAtoms()) for v in anchors.values()):
            return []

    # Optional site targeting: lift molecule-index maps off the non-dummy atoms of
    # from_smiles (kept out of the reaction pairing), and build an any-atom-dummy
    # query for site matching that preserves atom order (so site_map keys stay valid).
    patt0 = Chem.MolFromSmiles(from_smiles)
    site_map: Dict[int, int] = {}
    patt = None
    if patt0 is not None:
        for a in patt0.GetAtoms():
            mn = a.GetAtomMapNum()
            if mn and a.GetAtomicNum() > 0:
                site_map[a.GetIdx()] = mn
                a.SetAtomMapNum(0)
        _qp = Chem.AdjustQueryParameters.NoAdjustments()
        _qp.makeDummiesQueries = True
        patt = Chem.AdjustQueryProperties(patt0, _qp)
    targeting = bool(site_map) or anchor_atom_index is not None

    # Build the reaction from the molecule-map-stripped `from` (attachment dummy
    # maps preserved); with no molecule maps this is the input verbatim.
    from_for_rxn = Chem.MolToSmiles(patt0) if (patt0 is not None and site_map) else from_smiles
    frm = _map_attachments(from_for_rxn)
    to = _map_attachments(to_smiles)
    try:
        rxn = AllChem.ReactionFromSmarts(f"{frm}>>{to}")
    except Exception:
        return []
    if rxn is None or rxn.GetNumReactantTemplates() != 1:
        return []

    # Clean-swap heavy-atom target (guard); skipped only if a fragment won't parse.
    nd_from, nd_to = _nondummy_heavy(from_smiles), _nondummy_heavy(to_smiles)
    guard = nd_from >= 0 and nd_to >= 0
    exp_heavy = base.GetNumHeavyAtoms() - nd_from + nd_to

    target_removed = None
    if targeting:
        target_removed = _target_removed_sets(base, patt, site_map, anchor_atom_index)
        if not target_removed:
            return []  # requested site not present / not a clean single-cut

    products: List[str] = []
    seen = set()
    try:
        outcomes = rxn.RunReactants((base,))
    except Exception:
        return []
    n_base = base.GetNumAtoms()
    for outcome in outcomes:
        mol = outcome[0]
        # Ordered multi-anchor targeting: RDKit tags each product atom that came from
        # a mapped reactant template atom with `old_mapno` (the [*:N] label) and
        # `react_atom_idx` (its index in `base`). Rebuild {label: base_atom} for this
        # outcome and require it to match every requested (label, atom) pair — this
        # pins the site AND the :1/:2/:3 orientation, unlike anchor_atom_index.
        if anchors is not None:
            lbl2idx = {int(a.GetProp("old_mapno")): a.GetIntProp("react_atom_idx")
                       for a in mol.GetAtoms()
                       if a.HasProp("old_mapno") and a.HasProp("react_atom_idx")}
            if any(lbl2idx.get(lbl) != idx for lbl, idx in anchors.items()):
                continue
        try:
            prod = Chem.RemoveHs(mol)
            Chem.SanitizeMol(prod)
        except Exception:
            continue
        smi = Chem.MolToSmiles(prod)
        if "." in smi or smi == in_canon or smi in seen:
            continue  # drop disconnected, unchanged, or duplicate products
        rp = Chem.MolFromSmiles(smi)
        if rp is None:
            continue
        if guard and rp.GetNumHeavyAtoms() != exp_heavy:
            continue  # collateral atoms were deleted — not a clean swap
        if targeting:
            surviving = {a.GetIntProp("react_atom_idx") for a in mol.GetAtoms()
                         if a.HasProp("react_atom_idx")}
            if (frozenset(range(n_base)) - surviving) not in target_removed:
                continue  # product is from a different site than requested
        seen.add(smi)
        products.append(smi)
        if len(products) >= max_products:
            break
    return products




def label_atom_indices(mol_smiles: str) -> str:
    """
    Return a mapped SMILES where each atom's map number equals its 0-based index.
    Useful for identifying which atom_index to pass to attach/remove tools.
    Example: 'CCO' -> '[CH3:0][CH2:1][OH:2]'  (exact form depends on RDKit canonicalization)

    Implementation note: RDKit treats atom map number 0 as "unmapped" and
    omits ``:0`` from the emitted SMILES, so we shift every map number by +1
    before serialisation and decrement it back in the output string.
    """
    mol = _mol_from_smiles(mol_smiles)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
    smi = Chem.MolToSmiles(mol, canonical=False)
    return re.sub(r":(\d+)\]", lambda m: f":{int(m.group(1)) - 1}]", smi)
