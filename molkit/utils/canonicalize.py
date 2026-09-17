"""molkit/utils/canonicalize.py  —  SMILES canonicalization helpers

Functions
---------
canonicalize_molecule_smiles  : canonicalize a molecule SMILES (handles salts, maps)
get_molecule_id               : stable hash-based ID from canonical SMILES
"""

from __future__ import annotations

import hashlib
import re
from typing import Optional

from rdkit import Chem
from rdkit.Chem import AllChem


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _remove_atom_map(mol: Chem.Mol) -> Chem.Mol:
    """Strip all atom-map numbers from *mol* in place and return it."""
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return mol


def _strip_stereo(mol: Chem.Mol) -> Chem.Mol:
    """Remove all stereo information from *mol* in place and return it."""
    Chem.RemoveStereochemistry(mol)
    return mol


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def canonicalize_molecule_smiles(
    smiles: str,
    remove_atom_map: bool = True,
    sanitize: bool = True,
    largest_fragment_only: bool = False,
) -> Optional[str]:
    """Return the RDKit canonical SMILES for *smiles*.

    Parameters
    ----------
    smiles:
        Input SMILES string (may contain atom maps, salts, etc.).
    remove_atom_map:
        Strip atom-map numbers before canonicalizing.
    sanitize:
        Run ``Chem.SanitizeMol``; set False only if you know the input is
        already sanitized.
    largest_fragment_only:
        If True and the molecule has multiple fragments (e.g. salts), keep
        only the largest fragment by heavy-atom count.

    Returns
    -------
    Canonical SMILES string, or None if the input cannot be parsed.
    """
    if not smiles or not isinstance(smiles, str):
        return None

    mol = Chem.MolFromSmiles(smiles, sanitize=sanitize)
    if mol is None:
        return None

    if remove_atom_map:
        _remove_atom_map(mol)

    if largest_fragment_only:
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=sanitize)
        if frags:
            mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())

    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def get_molecule_id(smiles: str) -> Optional[str]:
    """Return a stable 16-character hex ID derived from the canonical SMILES.

    Useful for deduplication and caching.  Returns None for invalid SMILES.
    """
    canonical = canonicalize_molecule_smiles(smiles)
    if canonical is None:
        return None
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    return digest[:16]
