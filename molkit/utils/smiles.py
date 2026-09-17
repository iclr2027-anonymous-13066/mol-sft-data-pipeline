"""molkit/utils/smiles.py  —  SMILES validation and molecular property helpers

Functions
---------
is_smiles                  : check whether a string is a valid SMILES
is_multiple_smiles         : check whether a string contains multiple SMILES (dot notation)
split_smiles               : split a multi-fragment SMILES into individual SMILES
largest_mol                : return the SMILES of the largest fragment
tanimoto                   : Tanimoto similarity between two SMILES (Morgan FP)
calc_rotatable_bonds       : number of rotatable bonds
calc_ring_counts           : {ring_size: count} for all rings
calc_heavy_atoms_and_charge: (n_heavy_atoms, formal_charge)
calc_fsp3                  : fraction of sp3 carbons
calc_hetero_atom_count     : number of heteroatoms (non-C, non-H heavy atoms)
calc_topological_diameter  : topological diameter (longest shortest path)
calc_molar_refractivity    : molar refractivity (Crippen)
"""

from __future__ import annotations

from typing import Optional

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors, rdMolDescriptors


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def is_smiles(s: str) -> bool:
    """Return True if *s* is a non-empty string parseable by RDKit as SMILES."""
    if not s or not isinstance(s, str):
        return False
    mol = Chem.MolFromSmiles(s)
    return mol is not None


def is_multiple_smiles(s: str) -> bool:
    """Return True if *s* contains multiple disconnected fragments (dot notation)."""
    if not is_smiles(s):
        return False
    mol = Chem.MolFromSmiles(s)
    return len(Chem.GetMolFrags(mol)) > 1


def split_smiles(s: str) -> list[str]:
    """Split a multi-fragment SMILES into individual canonical SMILES strings.

    Returns a list with a single entry for single-fragment inputs.
    Returns an empty list if *s* is not a valid SMILES.
    """
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return []
    frags = Chem.GetMolFrags(mol, asMols=True)
    return [Chem.MolToSmiles(f, canonical=True) for f in frags]


def largest_mol(s: str) -> Optional[str]:
    """Return the canonical SMILES of the largest fragment in *s* by heavy-atom count.

    Returns None if *s* is not a valid SMILES.
    """
    frags = split_smiles(s)
    if not frags:
        return None
    return max(frags, key=lambda smi: Chem.MolFromSmiles(smi).GetNumHeavyAtoms())


# ---------------------------------------------------------------------------
# Fingerprint / similarity
# ---------------------------------------------------------------------------

def tanimoto(smiles_1: str, smiles_2: str, radius: int = 2) -> Optional[float]:
    """Morgan fingerprint Tanimoto similarity between two SMILES strings.

    Returns None if either SMILES is invalid.
    """
    mol1 = Chem.MolFromSmiles(smiles_1)
    mol2 = Chem.MolFromSmiles(smiles_2)
    if mol1 is None or mol2 is None:
        return None
    fp1 = AllChem.GetMorganFingerprint(mol1, radius)
    fp2 = AllChem.GetMorganFingerprint(mol2, radius)
    return DataStructs.TanimotoSimilarity(fp1, fp2)


# ---------------------------------------------------------------------------
# Molecular property helpers
# ---------------------------------------------------------------------------

def calc_rotatable_bonds(mol_smiles: str) -> Optional[int]:
    """Return the number of rotatable bonds, or None for invalid SMILES."""
    mol = Chem.MolFromSmiles(mol_smiles)
    if mol is None:
        return None
    return rdMolDescriptors.CalcNumRotatableBonds(mol)


def calc_ring_counts(mol_smiles: str) -> Optional[dict[int, int]]:
    """Return a dict mapping ring size → count, or None for invalid SMILES."""
    mol = Chem.MolFromSmiles(mol_smiles)
    if mol is None:
        return None
    ri = mol.GetRingInfo()
    counts: dict[int, int] = {}
    for ring in ri.AtomRings():
        size = len(ring)
        counts[size] = counts.get(size, 0) + 1
    return counts


def calc_heavy_atoms_and_charge(mol_smiles: str) -> Optional[tuple[int, int]]:
    """Return (num_heavy_atoms, formal_charge), or None for invalid SMILES."""
    mol = Chem.MolFromSmiles(mol_smiles)
    if mol is None:
        return None
    n_heavy = mol.GetNumHeavyAtoms()
    charge  = sum(a.GetFormalCharge() for a in mol.GetAtoms())
    return n_heavy, charge


def calc_fsp3(mol_smiles: str) -> Optional[float]:
    """Return the fraction of sp3 carbons (Fsp3), or None for invalid SMILES."""
    mol = Chem.MolFromSmiles(mol_smiles)
    if mol is None:
        return None
    return rdMolDescriptors.CalcFractionCSP3(mol)


def calc_hetero_atom_count(mol_smiles: str) -> Optional[int]:
    """Return the number of heteroatoms (non-C, non-H heavy atoms), or None."""
    mol = Chem.MolFromSmiles(mol_smiles)
    if mol is None:
        return None
    return rdMolDescriptors.CalcNumHeteroatoms(mol)


def calc_topological_diameter(mol_smiles: str) -> Optional[int]:
    """Return the topological diameter (longest shortest-path distance), or None."""
    mol = Chem.MolFromSmiles(mol_smiles)
    if mol is None:
        return None
    try:
        from rdkit.Chem import rdmolops
        dm = rdmolops.GetDistanceMatrix(mol)
        return int(dm.max())
    except Exception:
        return None


def calc_molar_refractivity(mol_smiles: str) -> Optional[float]:
    """Return Crippen molar refractivity, or None for invalid SMILES."""
    mol = Chem.MolFromSmiles(mol_smiles)
    if mol is None:
        return None
    return Descriptors.MolMR(mol)
