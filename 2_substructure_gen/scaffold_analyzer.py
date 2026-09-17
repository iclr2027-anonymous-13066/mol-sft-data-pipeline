#!/usr/bin/env python3
"""
scaffold_analyzer.py
====================
A pure-RDKit library that extracts one molecule's **Murcko scaffold** from its SMILES and
decomposes and names that skeleton as a medicinal-chemistry structural constraint,

The pipeline (the part this file covers):
    SMILES
     -> extract the Murcko scaffold      (get_murcko_scaffold)
     → ring system decomposition       (decompose_ring_systems)
     -> map ring / heterocycle names     (name_ring_system: curated dict -> systematic monocycle -> composite)
     -> linker & fusion / spiro / bridge (classify_linkers, plus intra-ring-system topology)
     -> strip substituents, record attachment points (find_attachment_points)
     -> a structured analysis dict       (analyze_scaffold)

scaffold_describer.py then turns that dict into (1) a controlled natural-language template and
(2) a validation of the generated text against these same facts, while describe_scaffolds.py has
an LLM polish the prose and verifies it with an RDKit substructure check.

Naming principle (faithfulness): describe only what the scaffold actually guarantees. Ring
composition (element / aromaticity / size), inter-ring connections (linker / direct bond / fusion /
spiro / bridge) and substitution positions are all computed deterministically with RDKit; IUPAC
locants that cannot be asserted (e.g. 'C4') are never invented, and only trustworthy positional

Dependencies: rdkit
"""
from __future__ import annotations

import itertools
import re
from typing import Optional

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")
_PT = Chem.GetPeriodicTable()


# --------------------------------------------------------------------------- #
#  Curated ring-system dictionary: canonical SMILES -> common name.
#  Covers both monocycles (benzene, pyridine, morpholine, ...) and fused / spiro / bridged
#  polycycles (indole, quinazoline, purine, ...). Each reference SMILES is canonicalised with
#  RDKit at import time and used as the key, so a hand-written SMILES still matches the same ring
#  system. A miss falls through: systematic monocycle name (name_monocycle) -> composite name
# --------------------------------------------------------------------------- #
_RAW_RING_NAMES: list[tuple[str, str]] = [
    # ---- carbocycles ----
    ("C1CC1", "cyclopropane"), ("C1CCC1", "cyclobutane"),
    ("C1CCCC1", "cyclopentane"), ("C1CCCCC1", "cyclohexane"),
    ("C1CCCCCC1", "cycloheptane"), ("C1CCCCCCC1", "cyclooctane"),
    ("c1ccccc1", "benzene"),
    ("C1=CCCCC1", "cyclohexene"), ("C1=CCCC1", "cyclopentene"),
    # ---- six-membered aromatic N-heterocycles ----
    ("c1ccncc1", "pyridine"),
    ("c1ccnnc1", "pyridazine"), ("c1cnccn1", "pyrazine"), ("c1cncnc1", "pyrimidine"),
    ("c1cnnnc1", "1,2,4-triazine"), ("c1ncncn1", "1,3,5-triazine"),
    ("c1nnncn1", "1,2,4,5-tetrazine"),
    # ---- five-membered aromatic (N/O/S) ----
    ("c1cc[nH]c1", "pyrrole"), ("c1ccoc1", "furan"), ("c1ccsc1", "thiophene"),
    ("c1c[nH]cn1", "imidazole"), ("c1cc[nH]n1", "pyrazole"),
    ("c1ocnc1", "oxazole"), ("c1occn1", "isoxazole"),
    ("c1scnc1", "thiazole"), ("c1sccn1", "isothiazole"),
    ("c1nc[nH]n1", "1,2,4-triazole"), ("c1cn[nH]n1", "1,2,3-triazole"),
    ("c1n[nH]nn1", "tetrazole"), ("c1nocn1", "1,2,4-oxadiazole"),
    ("c1ncon1", "1,2,4-oxadiazole"), ("c1onnc1", "1,2,5-oxadiazole (furazan)"),
    # ---- six-membered saturated / partly saturated N-heterocycles ----
    ("C1CCNCC1", "piperidine"), ("C1CNCCN1", "piperazine"),
    ("C1CCOCC1", "tetrahydropyran"), ("C1CCSCC1", "thiane"),
    ("C1COCCN1", "morpholine"), ("C1CSCCN1", "thiomorpholine"),
    ("C1CCOCO1", "1,3-dioxane"), ("C1COCCO1", "1,4-dioxane"),
    ("C1CCNCN1", "1,3-diazinane"),
    ("O=C1CCCCN1", "piperidinone"),
    # ---- five-membered saturated N/O/S heterocycles ----
    ("C1CCNC1", "pyrrolidine"), ("C1CCOC1", "tetrahydrofuran"),
    ("C1CCSC1", "tetrahydrothiophene"),
    ("C1CNCN1", "imidazolidine"), ("C1CCNN1", "pyrazolidine"),
    ("C1COCN1", "oxazolidine"), ("C1CONC1", "isoxazolidine"),
    ("C1CSCN1", "thiazolidine"),
    ("C1COCO1", "1,3-dioxolane"), ("C1CCOO1", "1,2-dioxolane"),
    # ---- small rings ----
    ("C1CN1", "aziridine"), ("C1CO1", "oxirane"), ("C1CS1", "thiirane"),
    ("C1CCN1", "azetidine"), ("C1CCO1", "oxetane"), ("C1CCS1", "thietane"),
    # ---- large rings ----
    ("C1CCNCCC1", "azepane"), ("C1CCOCCC1", "oxepane"),
    ("C1CNCCNC1", "1,4-diazepane (homopiperazine)"),
    ("C1CCNCCCC1", "azocane"),
    # ---- benzo-fused five-membered ----
    ("c1ccc2[nH]ccc2c1", "indole"), ("c1ccc2cc[nH]c2c1", "isoindole"),
    ("C1Cc2ccccc2N1", "indoline"), ("C1Cc2ccccc2C1", "2,3-dihydro-1H-indene (indane)"),
    ("c1ccc2[nH]cnc2c1", "benzimidazole"),
    ("c1ccc2nc[nH]c2c1", "benzimidazole"),
    ("c1ccc2[nH]ncc2c1", "indazole"), ("c1ccc2cn[nH]c2c1", "indazole"),
    ("c1ccc2occc2c1", "benzofuran"), ("c1ccc2sccc2c1", "benzothiophene"),
    ("c1ccc2ocnc2c1", "benzoxazole"), ("c1ccc2scnc2c1", "benzothiazole"),
    ("c1ccc2[nH]nnc2c1", "benzotriazole"),
    ("c1ccc2c(c1)oc1ccccc12", "dibenzofuran"),
    ("C1CCc2ccccc2C1", "tetralin (1,2,3,4-tetrahydronaphthalene)"),
    ("c1ccc2c(c1)CCO2", "2,3-dihydrobenzofuran"),
    ("O=c1ccc2ccccc2o1", "coumarin (2H-chromen-2-one)"),
    ("c1ccc2c(c1)OCCO2", "1,4-benzodioxane"),
    ("c1ccc2c(c1)cco2", "benzofuran"),
    # ---- benzo-fused six-membered ----
    ("c1ccc2ccccc2c1", "naphthalene"),
    ("c1ccc2ncccc2c1", "quinoline"), ("c1ccc2cccnc2c1", "isoquinoline"),
    ("c1ccc2ncncc2c1", "quinazoline"), ("c1ccc2nccnc2c1", "quinoxaline"),
    ("c1ccc2nnccc2c1", "cinnoline"), ("c1ccc2ccnnc2c1", "phthalazine"),
    ("C1CCc2ncccc2C1", "5,6,7,8-tetrahydroquinoline"),
    ("c1ccc2c(c1)cccc2", "naphthalene"),
    ("c1cc2cccnc2cn1", "1,5-naphthyridine"),
    ("c1ccc2c(c1)CCCC2", "tetralin (1,2,3,4-tetrahydronaphthalene)"),
    ("O=c1ccc2ccccc2[nH]1", "2-quinolinone (carbostyril)"),
    ("c1ccc2[nH]ccc2c1", "indole"),
    ("C1CCc2ccccc2CC1", "benzosuberane"),
    ("C1=Cc2ccccc2CC1", "dihydronaphthalene"),
    ("c1ccc2c(c1)CCNC2", "1,2,3,4-tetrahydroisoquinoline"),
    ("c1ccc2c(c1)CNCC2", "1,2,3,4-tetrahydroisoquinoline"),
    ("C1Cc2ccccc2CN1", "tetrahydroisoquinoline"),
    ("c1ccc2c(c1)[nH]c1ccccc12", "carbazole"),
    ("c1ccc2c(c1)Cc1ccccc1C2", "9H-fluorene"),
    ("c1ccc2c(c1)Cc1ccccc1-2", "fluorene"),
    ("c1ccc-2c(c1)Cc1ccccc1-2", "fluorene"),
    ("c1ccc2cc3ccccc3cc2c1", "anthracene"),
    ("c1ccc2c(c1)ccc1ccccc12", "phenanthrene"),
    ("c1ccc2nc3ccccc3nc2c1", "phenazine"),
    ("c1ccc2c(c1)Nc1ccccc1S2", "phenothiazine"),
    ("c1ccc2c(c1)Nc1ccccc1O2", "phenoxazine"),
    ("c1ccc2c(c1)Oc1ccccc1C2", "xanthene"),
    ("c1ccc2c(c1)Sc1ccccc1C2", "thioxanthene"),
    ("c1ccc2c(c1)nc1ccccc1c2", "acridine"),
    ("O=c1c2ccccc2oc2ccccc12", "xanthone"),
    # ---- chromene / chroman ----
    ("C1CCc2ccccc2O1", "chromane (3,4-dihydro-2H-1-benzopyran)"),
    ("C1=Cc2ccccc2OC1", "2H-chromene"), ("C1=COc2ccccc2C1", "2H-chromene"),
    ("O=c1ccoc2ccccc12", "chromone (4H-chromen-4-one)"),
    ("C1CCc2ccccc2N1", "1,2,3,4-tetrahydroquinoline"),
    # ---- purine / pteridine / azaindole ----
    ("c1ncc2[nH]cnc2n1", "purine"), ("c1nc2[nH]cnc2cn1", "purine"),
    ("c1ncc2nc[nH]c2n1", "purine"), ("c1cnc2[nH]cnc2n1", "purine"),
    ("c1nc2cnc[nH]c2n1", "purine"),
    ("c1ncc2[nH]ccc2n1", "pyrrolo[2,3-b]pyrazine"),
    ("c1cnc2[nH]ccc2c1", "7-azaindole"), ("c1cc2cc[nH]c2nc1", "azaindole"),
    ("c1cnc2nccnc2c1", "pteridine"), ("c1cnc2nccnc2n1", "pteridine"),
    ("c1ccc2ccc3ccccc3c2c1", "phenanthrene"),
    # ---- 5-6 fused heteroaromatics (scaffolds common in drugs) ----
    ("c1ccn2ccnc2c1", "imidazo[1,2-a]pyridine"),
    ("c1cnc2[nH]cnc2c1", "imidazo[4,5-b]pyridine (3-deazapurine)"),
    ("c1cc2cc[nH]c2nc1", "pyrrolo[2,3-b]pyridine (7-azaindole)"),
    ("c1cc2[nH]ccc2cn1", "pyrrolo[3,2-c]pyridine (azaindole)"),
    ("c1cc2cc[nH]c2cn1", "pyrrolo[3,2-b]pyridine (azaindole)"),
    ("c1cc2cc[nH]c2cc1", "indole"),
    ("c1ccn2nccc2c1", "pyrazolo[1,5-a]pyridine"),
    ("c1cnn2ccccc12", "pyrazolo[1,5-a]pyridine"),
    ("c1cnc2[nH]ncc2c1", "1H-pyrazolo[3,4-b]pyridine"),
    ("c1cnc2[nH]ccc2n1", "pyrrolo[2,3-b]pyrazine"),
    ("c1csc2ncncc12", "thieno[2,3-d]pyrimidine"),
    ("c1csc2ncccc12", "thieno[2,3-b]pyridine"),
    ("c1coc2ncncc12", "furo[3,2-d]pyrimidine"),
    ("c1ccc2ncncc2c1", "quinazoline"),
    ("c1ccc2c(c1)OCO2", "1,3-benzodioxole"),
    ("c1ccc2c(c1)OCCO2", "1,4-benzodioxane"),
    ("C1NCc2ccccc21", "isoindoline"),
    ("O=C1CCc2ccccc21", "1-indanone"),
    ("O=C1c2ccccc2C(=O)c2ccccc21", "anthraquinone"),
    ("O=C1c2ccccc2-c2ccccc21", "fluorenone"),
    ("c1ccc2c(c1)ncc1ccccc12", "acridine"),
    ("c1ccc2ncc3ccccc3c2c1", "acridine"),
    # ---- indolizine / quinolizine and other bridgehead-N systems ----
    ("c1ccn2ccccc12", "indolizine"), ("C1CCN2CCCCC12", "quinolizidine"),
    ("C1CC2CCC1N2", "azanorbornane"),
    # ---- bridged / cage ----
    ("C1CC2CCC1C2", "norbornane (bicyclo[2.2.1]heptane)"),
    ("C1CC2CCC1CC2", "bicyclo[2.2.2]octane"),
    ("C1CC2CCC(C1)N2", "tropane skeleton"),
    ("C1CN2CCC1CC2", "quinuclidine"),
    ("C1C2CC3CC1CC(C2)C3", "adamantane"),
    # ---- spiro ----
    ("C1CCC2(CC1)CCCCC2", "spiro[5.5]undecane"),
    ("C1CCC2(CC1)CCOCC2", "1-oxaspiro[5.5]undecane"),
    # ---- exocyclic carbonyl on a ring (lactam / lactone / -one / imide). Matched on the
    #      canonical SMILES that keeps the ring atom's =O (_submol_smiles_keep_exo), so names
    #      distorted by dropping the =O (benzimidazol-2-one and friends) come out right. ----
    ("O=C1CCCN1", "pyrrolidin-2-one (2-pyrrolidinone)"),
    ("O=C1CCCCN1", "piperidin-2-one"),
    ("O=C1CCCCC1", "cyclohexanone"), ("O=C1CCCC1", "cyclopentanone"),
    ("O=C1CCC(=O)N1", "succinimide (pyrrolidine-2,5-dione)"),
    ("O=C1CNC(=O)N1", "hydantoin (imidazolidine-2,4-dione)"),
    ("O=C1CC(=O)NC(=O)N1", "barbituric acid"),
    ("O=c1cccc[nH]1", "pyridin-2(1H)-one (2-pyridone)"),
    ("O=c1cc[nH]cc1", "pyridin-4(1H)-one (4-pyridone)"),
    ("O=c1cc[nH]c(=O)[nH]1", "uracil (pyrimidine-2,4-dione)"),
    ("O=c1[nH]cnc(=O)[nH]1", "1,3,5-triazinane-2,4-dione"),
    ("O=c1[nH]c2ccccc2[nH]1", "benzimidazol-2(3H)-one"),
    ("O=c1[nH]c2ccccc2o1", "benzoxazol-2(3H)-one"),
    ("O=c1[nH]c2ccccc2s1", "benzothiazol-2(3H)-one"),
    ("O=c1[nH]cnc2ccccc12", "quinazolin-4(3H)-one"),
    ("O=c1cc[nH]c2ccccc12", "quinolin-4(1H)-one (4-quinolone)"),
    ("O=c1ccc2ccccc2[nH]1", "quinolin-2(1H)-one (carbostyril)"),
    ("O=c1[nH]ccc2ccccc12", "isoquinolin-1(2H)-one"),
    ("O=C1Cc2ccccc2N1", "indolin-2-one (oxindole)"),
    ("O=C1NCc2ccccc21", "isoindolin-1-one"),
    ("O=C1c2ccccc2C(=O)N1", "phthalimide (isoindoline-1,3-dione)"),
    ("O=C1Nc2ccccc2C1=O", "isatin (indoline-2,3-dione)"),
    ("O=c1ccc2ccccc2o1", "coumarin (2H-chromen-2-one)"),
    ("O=c1ccoc2ccccc12", "chromone (4H-chromen-4-one)"),
    ("O=c1[nH]c(=O)c2ccccc2[nH]1", "quinazoline-2,4(1H,3H)-dione"),
]


def _canon(smi: str) -> Optional[str]:
    """SMILES -> RDKit canonical SMILES, aromaticity perception included. None on failure."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return Chem.MolToSmiles(m)


# canonical SMILES -> name (built once at import; first registration wins)
RING_SMILES_NAMES: dict[str, str] = {}
for _smi, _name in _RAW_RING_NAMES:
    _c = _canon(_smi)
    if _c and _c not in RING_SMILES_NAMES:
        RING_SMILES_NAMES[_c] = _name


# --------------------------------------------------------------------------- #
#  Murcko scaffold extraction + molecule <-> scaffold mapping (for attachment points)
# --------------------------------------------------------------------------- #
def get_murcko_scaffold(smiles: str) -> dict:
    """SMILES -> the Murcko scaffold (rings plus the linkers between them) and related information.

    Returns a dict:
      mol            : the original molecule (canonical)
      scaffold       : the Bemis-Murcko scaffold mol (substituents stripped, rings/linkers kept)
      scaffold_smiles: its canonical SMILES ("" means no rings)
      has_scaffold   : True when at least one ring is present
      match          : the scaffold_atom_idx -> mol_atom_idx mapping (tuple); () when empty.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"mol": None, "scaffold": None, "scaffold_smiles": "",
                "has_scaffold": False, "match": ()}
    scaf = MurckoScaffold.GetScaffoldForMol(mol)
    has = scaf is not None and scaf.GetNumAtoms() > 0
    scaf_smiles = Chem.MolToSmiles(scaf) if has else ""
    # Substructure-match the scaffold against the original to recover the position mapping,
    match = mol.GetSubstructMatch(scaf) if has else ()
    return {"mol": mol, "scaffold": scaf, "scaffold_smiles": scaf_smiles,
            "has_scaffold": has, "match": match}


def scaffold_to_smarts(scaf) -> str:
    """scaffold mol -> the SMARTS used for verification, passed straight to an RDKit substructure check.

    Occasionally the Murcko reduction leaves a double bond in a 'bad bond stereo' state as it strips
    substituents, and MolToSmarts raises RuntimeError (Pre-condition Violation). Stereo is cleared
    and it is retried, then a canonical-SMILES route is tried; a final failure returns "", which
    """
    if scaf is None or scaf.GetNumAtoms() == 0:
        return ""
    try:
        return Chem.MolToSmarts(scaf)
    except Exception:  # noqa: BLE001
        pass
    try:
        m = Chem.Mol(scaf)
        Chem.RemoveStereochemistry(m)
        for b in m.GetBonds():
            b.SetStereo(Chem.BondStereo.STEREONONE)
            b.SetBondDir(Chem.BondDir.NONE)
        return Chem.MolToSmarts(m)
    except Exception:  # noqa: BLE001
        pass
    try:
        mm = Chem.MolFromSmiles(Chem.MolToSmiles(scaf))
        if mm is not None:
            return Chem.MolToSmarts(mm)
    except Exception:  # noqa: BLE001
        pass
    return ""


# --------------------------------------------------------------------------- #
#  Per-dimension projected SMARTS: a single SMARTS that makes one dimension concrete and leaves
#  the rest generic. All projections share the same skeleton (connectivity) and differ only in
#  their atom/bond tokens. (skeleton / element / aromaticity / ring keep bonds generic as '~';
#  the bond dimension keeps atoms as '*'.) When a per-dimension match-any set is built later,
# --------------------------------------------------------------------------- #
_DIMENSIONS = ("skeleton", "element", "aromaticity", "bond", "ring")


def _dim_atom_token(scaf, atom, dim: str, ri) -> str:
    """One atom as the SMARTS atom token for a given dimension."""
    if dim == "skeleton" or dim == "bond":
        return "*"
    if dim == "element":
        return f"[#{atom.GetAtomicNum()}]"
    if dim == "aromaticity":
        return "a" if atom.GetIsAromatic() else "A"
    if dim == "ring":
        if atom.IsInRing():
            sizes = [k for k in range(3, 20) if ri.IsAtomInRingOfSize(atom.GetIdx(), k)]
            return f"[r{min(sizes)}]" if sizes else "[R]"
        return "[R0]"
    return "*"


def _dim_bond_token(bond, dim: str) -> str:
    """One bond as the SMARTS bond token for a dimension: only the bond dimension spells out the order."""
    if dim != "bond":
        return "~"
    if bond.GetIsAromatic() or bond.GetBondType() == Chem.BondType.AROMATIC:
        return ":"
    return {Chem.BondType.SINGLE: "-", Chem.BondType.DOUBLE: "=",
            Chem.BondType.TRIPLE: "#"}.get(bond.GetBondType(), "~")


def _project_dimension_smarts(scaf, dim: str) -> str:
    """The scaffold projected onto one dimension, the rest generic. "" on failure."""
    if scaf is None or scaf.GetNumAtoms() == 0:
        return ""
    try:
        ri = scaf.GetRingInfo()
        rw = Chem.RWMol(scaf)
        for a in scaf.GetAtoms():
            rw.ReplaceAtom(a.GetIdx(), Chem.AtomFromSmarts(_dim_atom_token(scaf, a, dim, ri)))
        for b in scaf.GetBonds():
            rw.ReplaceBond(b.GetIdx(), Chem.BondFromSmarts(_dim_bond_token(b, dim)))
        return Chem.MolToSmarts(rw.GetMol())
    except Exception:  # noqa: BLE001
        return ""


def scaffold_dimension_smarts(scaf) -> dict:
    """scaffold → {dimension: SMARTS} (skeleton/element/aromaticity/bond/ring).

    Each SMARTS makes only its own dimension concrete, so one aspect of the scaffold's identity can
    be checked independently. No degree or H tokens are emitted, so it still matches substituted molecules."""
    return {dim: _project_dimension_smarts(scaf, dim) for dim in _DIMENSIONS}


def _atoms_to_smarts(mol, atoms: set) -> str:
    """A substructure SMARTS keeping only the given atom set, normally one ring. Degree and H are
    not encoded, so it matches substituted and fused molecules too. Used for the constituent-ring
    patterns of a composite core decomposition."""
    keep = set(atoms)
    rw = Chem.RWMol(mol)
    for idx in sorted((a.GetIdx() for a in mol.GetAtoms() if a.GetIdx() not in keep),
                      reverse=True):
        rw.RemoveAtom(idx)
    sub = rw.GetMol()
    try:
        Chem.RemoveStereochemistry(sub)
        for b in sub.GetBonds():
            b.SetStereo(Chem.BondStereo.STEREONONE)
            b.SetBondDir(Chem.BondDir.NONE)
    except Exception:  # noqa: BLE001
        pass
    try:
        return Chem.MolToSmarts(sub)
    except Exception:  # noqa: BLE001
        try:
            mm = Chem.MolFromSmiles(Chem.MolToSmiles(sub))
            return Chem.MolToSmarts(mm) if mm is not None else ""
        except Exception:  # noqa: BLE001
            return ""


# --------------------------------------------------------------------------- #
#  Ring order / sequence / locant utilities
# --------------------------------------------------------------------------- #
def _z(atom) -> int:
    return atom.GetAtomicNum()


def _ring_order(mol, ring_atoms: tuple) -> list[int]:
    """Return the ring's atoms sorted into connection order, walking once around the ring.

    RingInfo.AtomRings() usually returns them in order already, but they are conservatively re-linked
    """
    ring = list(ring_atoms)
    rs = set(ring)
    adj = {i: [] for i in ring}
    for i in ring:
        for nb in mol.GetAtomWithIdx(i).GetNeighbors():
            j = nb.GetIdx()
            if j in rs:
                adj[i].append(j)
    start = ring[0]
    order = [start]
    prev, cur = None, start
    while len(order) < len(ring):
        nxts = [j for j in adj[cur] if j != prev]
        if not nxts:
            break
        nxt = nxts[0]
        if nxt in order:
            break
        order.append(nxt)
        prev, cur = cur, nxt
    return order if len(order) == len(ring) else ring


def _ring_element_seq(mol, order: list[int]) -> list[str]:
    return [_PT.GetElementSymbol(_z(mol.GetAtomWithIdx(i))) for i in order]


def _hetero_signature(seq: list[str]) -> tuple:
    """Fold an element sequence into a canonical tuple invariant under rotation and reflection

    Sorted so the heteroatoms take the lowest locants, pushing carbons to the back, which
    distinguishes 1,2- from 1,3- from 1,4-diazine (pyridazine / pyrimidine / pyrazine) deterministically."""
    n = len(seq)
    rank = {"C": 1}  # heteroatoms (0) sort before carbon (1), so they take the lowest locants

    def key(t):
        return tuple((rank.get(e, 0), e) for e in t)

    cands = []
    for s in (seq, list(reversed(seq))):
        for r in range(n):
            cands.append(tuple(s[r:] + s[:r]))
    return min(cands, key=key)


def _locants(sig: tuple) -> list[tuple[int, str]]:
    """The (1-based position, element) heteroatom list read off a canonical signature."""
    return [(i + 1, e) for i, e in enumerate(sig) if e != "C"]


# --------------------------------------------------------------------------- #
#  Systematic monocycle naming: size + aromaticity + heteroatom arrangement -> a name
# --------------------------------------------------------------------------- #
_CARBOCYCLE = {3: "cyclopropane", 4: "cyclobutane", 5: "cyclopentane",
               6: "cyclohexane", 7: "cycloheptane", 8: "cyclooctane",
               9: "cyclononane", 10: "cyclodecane"}
# One heteroatom of a single kind, the rest carbon: (element, size, aromatic) -> trivial name
_MONO_HETERO_1 = {
    ("N", 3, False): "aziridine", ("N", 4, False): "azetidine",
    ("N", 5, False): "pyrrolidine", ("N", 6, False): "piperidine",
    ("N", 7, False): "azepane", ("N", 8, False): "azocane",
    ("N", 5, True): "pyrrole", ("N", 6, True): "pyridine", ("N", 7, True): "azepine",
    ("O", 3, False): "oxirane", ("O", 4, False): "oxetane",
    ("O", 5, False): "tetrahydrofuran", ("O", 6, False): "tetrahydropyran",
    ("O", 7, False): "oxepane",
    ("O", 5, True): "furan", ("O", 6, True): "pyran",
    ("S", 3, False): "thiirane", ("S", 4, False): "thietane",
    ("S", 5, False): "tetrahydrothiophene", ("S", 6, False): "thiane",
    ("S", 5, True): "thiophene", ("S", 6, True): "thiopyran",
}
_NUM2LOC = {2: "1,2", 3: "1,3", 4: "1,4", 23: "1,2,3", 24: "1,2,4", 35: "1,3,5"}


def _hetero_counts(sig: tuple) -> dict[str, int]:
    c: dict[str, int] = {}
    for e in sig:
        if e != "C":
            c[e] = c.get(e, 0) + 1
    return c


def name_monocycle(mol, ring_atoms: tuple) -> dict:
    """Name a single ring deterministically. Returns {name, name_source, ...}."""
    order = _ring_order(mol, ring_atoms)
    seq = _ring_element_seq(mol, order)
    sig = _hetero_signature(seq)
    size = len(sig)
    aromatic = all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring_atoms)
    # whether the ring carries a non-aromatic double bond (partial unsaturation)
    rs = set(ring_atoms)
    endo_db = any(b.GetBondType() == Chem.BondType.DOUBLE
                  and b.GetBeginAtomIdx() in rs and b.GetEndAtomIdx() in rs
                  and not (b.GetBeginAtom().GetIsAromatic() or b.GetEndAtom().GetIsAromatic())
                  for b in mol.GetBonds())
    counts = _hetero_counts(sig)
    locs = _locants(sig)
    hetero_str = _heteroatom_phrase(counts)

    name = None
    # (1) carbon only
    if not counts:
        if aromatic and size == 6:
            name = "benzene"
        elif aromatic:
            name = f"{size}-membered aromatic carbocycle"
        elif endo_db:
            name = f"{_CARBOCYCLE.get(size, f'{size}-membered carbocycle')[:-3]}ene" \
                if size in _CARBOCYCLE else f"{size}-membered unsaturated carbocycle"
        else:
            name = _CARBOCYCLE.get(size, f"{size}-membered carbocycle")
    # (2) one heteroatom of a single kind
    elif len(counts) == 1 and sum(counts.values()) == 1:
        elem = next(iter(counts))
        name = _MONO_HETERO_1.get((elem, size, aromatic))
    # (3) two N (diazine / diazole / diazinane ...) and the N,O / N,S combinations
    if name is None:
        name = _name_multi_hetero(size, aromatic, counts, locs)

    if name is None:  # descriptive fallback: always returns something
        kind = "aromatic" if aromatic else ("unsaturated" if endo_db else "saturated")
        name = f"{size}-membered {kind} ring with {hetero_str}" if counts \
            else f"{size}-membered {kind} ring"

    return {"name": name, "name_source": "monocycle", "size": size,
            "aromatic": aromatic, "saturated": (not aromatic and not endo_db),
            "hetero_counts": counts, "locants": locs, "hetero_phrase": hetero_str}


def _name_multi_hetero(size: int, aromatic: bool, counts: dict, locs: list) -> Optional[str]:
    """The common name of a monocycle with two or more heteroatoms, where one exists; None otherwise."""
    loc_nums = sorted(p for p, _ in locs)
    key = int("".join(str(n) for n in loc_nums)) if loc_nums else 0
    loc_str = _NUM2LOC.get(key, ",".join(map(str, loc_nums)))
    elems = sorted(counts)
    # --- six-membered ---
    if size == 6 and aromatic:
        if counts == {"N": 2}:
            return {2: "pyridazine", 3: "pyrimidine", 4: "pyrazine"}.get(loc_nums[1], "diazine")
        if counts == {"N": 3}:
            return {24: "1,2,4-triazine", 35: "1,3,5-triazine"}.get(key, "triazine")
    if size == 6 and not aromatic:
        if counts == {"N": 2}:
            return {4: "piperazine", 3: "hexahydropyrimidine", 2: "hexahydropyridazine"}.get(
                loc_nums[1], f"{loc_str}-diazinane")
        if counts == {"O": 1, "N": 1}:
            return "morpholine"
        if counts == {"S": 1, "N": 1}:
            return "thiomorpholine"
        if counts == {"O": 2}:
            return {2: "1,2-dioxane", 3: "1,3-dioxane", 4: "1,4-dioxane"}.get(loc_nums[1], "dioxane")
    # --- five-membered ---
    if size == 5 and aromatic:
        if counts == {"N": 2}:
            return {2: "pyrazole", 3: "imidazole"}.get(loc_nums[1], "diazole")
        if counts == {"O": 1, "N": 1}:
            return "isoxazole" if loc_nums[1] == 2 else "oxazole"
        if counts == {"S": 1, "N": 1}:
            return "isothiazole" if loc_nums[1] == 2 else "thiazole"
        if counts == {"N": 3}:
            return {23: "1,2,3-triazole", 24: "1,2,4-triazole"}.get(key, "triazole")
        if counts == {"O": 1, "N": 2}:
            return f"{loc_str}-oxadiazole"
        if counts == {"S": 1, "N": 2}:
            return f"{loc_str}-thiadiazole"
        if counts == {"N": 4}:
            return "tetrazole"
    if size == 5 and not aromatic:
        if counts == {"N": 2}:
            return {2: "pyrazolidine", 3: "imidazolidine"}.get(loc_nums[1], "diazolidine")
        if counts == {"O": 1, "N": 1}:
            return "isoxazolidine" if loc_nums[1] == 2 else "oxazolidine"
        if counts == {"S": 1, "N": 1}:
            return "thiazolidine"
        if counts == {"O": 2}:
            return {2: "1,2-dioxolane", 3: "1,3-dioxolane"}.get(loc_nums[1], "dioxolane")
    return None


def _heteroatom_phrase(counts: dict) -> str:
    """{'N':2} → 'two nitrogens'; {'N':1,'O':1} → 'one nitrogen and one oxygen'."""
    if not counts:
        return "no heteroatoms"
    names = {"N": "nitrogen", "O": "oxygen", "S": "sulfur", "P": "phosphorus",
             "B": "boron", "Se": "selenium", "Si": "silicon",
             "F": "fluorine", "Cl": "chlorine", "Br": "bromine", "I": "iodine"}
    num = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
    parts = []
    for e in sorted(counts, key=lambda x: (-counts[x], x)):
        nm = names.get(e, e)
        cnt = counts[e]
        parts.append(f"{num.get(cnt, str(cnt))} {nm}{'' if cnt == 1 else 's'}")
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


# --------------------------------------------------------------------------- #
#  Ring-system decomposition + internal topology (fused / spiro / bridged)
# --------------------------------------------------------------------------- #
def _pair_relation(mol, ring_a: set, ring_b: set) -> Optional[str]:
    """Classify the relation between two rings by how many atoms they share; None when they share none.
       1 atom -> spiro; 2 adjacent (bonded) -> fused (ortho); 2 non-adjacent -> bridged; 3+ -> bridged."""
    shared = ring_a & ring_b
    if not shared:
        return None
    if len(shared) == 1:
        return "spiro"
    if len(shared) == 2:
        a, b = tuple(shared)
        return "fused" if mol.GetBondBetweenAtoms(a, b) is not None else "bridged"
    return "bridged"


def decompose_ring_systems(mol) -> list[dict]:
    """Decompose a scaffold (or a molecule) into connected ring systems.

    SSSR rings that share any atom are union-found into one system, and each system's internal
    topology (single / ortho-fused / spiro / bridged / mixed) is classified.
    Returns a list of system dicts: {id, ring_idxs, atoms(set), rings(list[set]),
          n_rings, internal_topology, pair_relations}.
    """
    ri = mol.GetRingInfo()
    rings = [set(r) for r in ri.AtomRings()]
    n = len(rings)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if rings[i] & rings[j]:
                parent[find(i)] = find(j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    systems = []
    for sid, (_, idxs) in enumerate(sorted(groups.items())):
        atoms: set = set().union(*(rings[k] for k in idxs))
        rels = []
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                rel = _pair_relation(mol, rings[idxs[a]], rings[idxs[b]])
                if rel:
                    rels.append(rel)
        if len(idxs) == 1:
            topo = "single"
        elif all(r == "spiro" for r in rels):
            topo = "spiro"
        elif all(r == "fused" for r in rels):
            topo = "ortho-fused"
        elif any(r == "bridged" for r in rels):
            topo = "bridged"
        else:
            topo = "fused/spiro"  # mixed, e.g. fusion together with spiro
        systems.append({"id": sid, "ring_idxs": idxs, "atoms": atoms,
                        "rings": [rings[k] for k in idxs], "n_rings": len(idxs),
                        "internal_topology": topo, "pair_relations": rels})
    return systems


def _submol_smiles(mol, atoms: set) -> str:
    """The canonical SMILES of the 'parent' heterocycle, keeping only the ring-system atoms.

    Non-system atoms are deleted and the fragment is **re-sanitised**, so a ring atom left bare by a
    stripped substituent (or by a bond to another ring system) regains its implicit H. Cutting an
    N-aryl benzimidazole, for instance, gives the nitrogen its H back and normalises to the real
    benzimidazole `c1ccc2[nH]cnc2c1`. PathToSubmol froze the valences instead and produced
    non-kekulisable fragments like 'c1ccc2ncnc2c1' that missed the dictionary. Falls back to
    """
    atoms = set(atoms)
    rw = Chem.RWMol(mol)
    # Kekulise first (dropping aromatic flags) to pin the bond orders, then delete the non-system
    # atoms, unfreeze the implicit H of what remains and re-sanitise. That is what lets a ring atom
    # left by a stripped substituent — a pyrrole-type N-substituted nitrogen above all — regain its H
    # and normalise to the parent heterocycle (N-aryl benzimidazole -> c1ccc2[nH]cnc2c1).
    try:
        Chem.Kekulize(rw, clearAromaticFlags=True)
    except Exception:  # noqa: BLE001
        pass
    for idx in sorted((a.GetIdx() for a in mol.GetAtoms() if a.GetIdx() not in atoms),
                      reverse=True):
        rw.RemoveAtom(idx)
    sub = rw.GetMol()
    for a in sub.GetAtoms():
        a.SetNoImplicit(False)
        a.SetNumExplicitHs(0)
    try:
        Chem.SanitizeMol(sub)
        smi = Chem.MolToSmiles(sub)
        r = Chem.MolFromSmiles(smi)
        return Chem.MolToSmiles(r) if r is not None else smi
    except Exception:  # noqa: BLE001
        pass
    # fallback: PathToSubmol, which freezes the valences
    bonds = [b.GetIdx() for b in mol.GetBonds()
             if b.GetBeginAtomIdx() in atoms and b.GetEndAtomIdx() in atoms]
    if not bonds:
        return ""
    try:
        smi = Chem.MolToSmiles(Chem.PathToSubmol(mol, bonds))
        r = Chem.MolFromSmiles(smi)
        return Chem.MolToSmiles(r) if r is not None else smi
    except Exception:  # noqa: BLE001
        return ""


def _submol_smiles_keep_exo(mol, atoms: set) -> str:
    """The canonical SMILES keeping the ring-system atoms **plus any exocyclic heteroatom double-bonded
    to a ring atom (=O/=S/=N)**. This preserves the carbonyl of a lactam, lactone or -one that
    _submol_smiles drops, so the exact name matches (benzimidazol-2-one, quinazolin-4(3H)-one, ...)."""
    keep = set(atoms)
    for i in atoms:
        a = mol.GetAtomWithIdx(i)
        for b in a.GetBonds():
            o = b.GetOtherAtom(a)
            if (o.GetIdx() not in atoms and b.GetBondType() == Chem.BondType.DOUBLE
                    and o.GetAtomicNum() in (7, 8, 16)):
                keep.add(o.GetIdx())
    rw = Chem.RWMol(mol)
    try:
        Chem.Kekulize(rw, clearAromaticFlags=True)
    except Exception:  # noqa: BLE001
        pass
    for idx in sorted((a.GetIdx() for a in mol.GetAtoms() if a.GetIdx() not in keep),
                      reverse=True):
        rw.RemoveAtom(idx)
    sub = rw.GetMol()
    for a in sub.GetAtoms():
        a.SetNoImplicit(False)
        a.SetNumExplicitHs(0)
    try:
        Chem.SanitizeMol(sub)
        smi = Chem.MolToSmiles(sub)
        r = Chem.MolFromSmiles(smi)
        return Chem.MolToSmiles(r) if r is not None else smi
    except Exception:  # noqa: BLE001
        return ""


def _ring_carbonyl_info(mol, atoms: set) -> tuple:
    """Find the exocyclic =O/=S/=N double-bonded to a ring atom and classify what it is.

    Returns (groups, details), where groups is a list of strings such as 'lactam (cyclic amide)',
    'cyclic urea', 'lactone (cyclic ester)', 'ring ketone' or 'cyclic sulfone'. The character is
    decided from the ring neighbours of the atom bearing the =O: between two nitrogens gives a cyclic
    urea, next to one nitrogen a lactam, next to an oxygen a lactone, and so on."""
    groups: list[str] = []
    details: list[dict] = []
    for i in sorted(atoms):
        a = mol.GetAtomWithIdx(i)
        exo = [o.GetAtomicNum() for b in a.GetBonds() for o in (b.GetOtherAtom(a),)
               if o.GetIdx() not in atoms and b.GetBondType() == Chem.BondType.DOUBLE
               and o.GetAtomicNum() in (7, 8, 16)]
        if not exo:
            continue
        sym = _PT.GetElementSymbol(a.GetAtomicNum())
        ring_nbr = [nb.GetAtomicNum() for nb in a.GetNeighbors() if nb.GetIdx() in atoms]
        nN = ring_nbr.count(7)
        nOx, nSx = exo.count(8), exo.count(16)
        if sym == "S":
            char = ("cyclic sulfone" if nOx >= 2 else
                    "cyclic sulfoxide" if nOx == 1 else "ring thiocarbonyl")
        elif sym == "N":
            char = "N-oxide"
        elif sym == "C" and nOx >= 1:
            char = ("cyclic urea" if nN >= 2 else "lactam (cyclic amide)" if nN == 1 else
                    "lactone (cyclic ester)" if 8 in ring_nbr else
                    "cyclic thioester" if 16 in ring_nbr else "ring ketone")
        elif sym == "C" and nSx >= 1:
            char = ("cyclic thiourea" if nN >= 2 else
                    "thiolactam (cyclic thioamide)" if nN == 1 else "ring thioketone")
        elif sym == "C":
            char = "cyclic amidine (C=N)" if nN >= 1 else "exocyclic imine (C=N)"
        else:
            char = f"ring {sym} with an exocyclic double bond"
        groups.append(char)
        details.append({"atom": i, "character": char,
                        "double_to": {8: "O", 16: "S", 7: "N"}.get(exo[0])})
    return groups, details


_CARBONYL_PHRASE = {
    "cyclic urea": "a ring carbonyl flanked by two ring nitrogens (a cyclic urea)",
    "lactam (cyclic amide)": "a lactam carbonyl (a cyclic amide)",
    "lactone (cyclic ester)": "a lactone carbonyl (a cyclic ester)",
    "cyclic thioester": "a cyclic thioester carbonyl",
    "ring ketone": "an exocyclic ketone carbonyl",
    "cyclic thiourea": "a ring C=S flanked by two ring nitrogens (a cyclic thiourea)",
    "thiolactam (cyclic thioamide)": "a thiolactam C=S (a cyclic thioamide)",
    "ring thioketone": "an exocyclic C=S (thioketone)",
    "cyclic sulfone": "a ring sulfone (the ring sulfur bears two =O)",
    "cyclic sulfoxide": "a ring sulfoxide (the ring sulfur bears one =O)",
    "N-oxide": "a ring N-oxide",
    "cyclic amidine (C=N)": "an exocyclic C=N (a cyclic amidine)",
    "exocyclic imine (C=N)": "an exocyclic C=N (imine)",
}


def _carbonyl_desc(groups: list[str]) -> str:
    """A carbonyl-group list -> one FACTS phrase ('bears ...'). An empty list gives ""."""
    if not groups:
        return ""
    lact = sum(1 for g in groups if "lactam" in g or g == "cyclic urea")
    if len(groups) >= 2 and lact >= 2:
        return "bears two exocyclic ring carbonyls (a cyclic imide)"
    uniq: list[str] = []
    for g in groups:
        if g not in uniq:
            uniq.append(g)
    parts = [_CARBONYL_PHRASE.get(g, g) for g in uniq]
    if len(parts) == 1:
        return "bears " + parts[0]
    return "bears " + ", ".join(parts[:-1]) + " and " + parts[-1]


def name_ring_system(mol, system: dict) -> dict:
    """Name one ring system.
       (1) the carbonyl-preserving dict -> (2) the curated dict -> (3) a systematic monocycle name -> (4) a composite name."""
    atoms = system["atoms"]
    smi = _submol_smiles(mol, atoms)
    # Aromaticity per ring: a fused system can mix aromatic and saturated rings, so never assert it system-wide.
    ring_arom = [all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)
                 for ring in system["rings"]]
    n_arom = sum(ring_arom)
    all_arom = (n_arom == len(ring_arom))
    if all_arom:
        arom_desc = "aromatic"
    elif n_arom == 0:
        arom_desc = "saturated / non-aromatic"
    else:
        arom_desc = "partially aromatic (some rings aromatic, some saturated)"
    # The heteroatom composition of the whole system
    counts: dict[str, int] = {}
    for i in atoms:
        e = _PT.GetElementSymbol(_z(mol.GetAtomWithIdx(i)))
        if e != "C":
            counts[e] = counts.get(e, 0) + 1

    # Detect an exocyclic carbonyl on a ring atom (lactam / lactone / -one / imide, ...).
    co_groups, co_details = _ring_carbonyl_info(mol, atoms)
    info = {"smiles": smi, "n_rings": system["n_rings"],
            "internal_topology": system["internal_topology"],
            "aromatic": all_arom, "any_aromatic": n_arom > 0,
            "n_aromatic_rings": n_arom, "aromaticity_desc": arom_desc,
            "hetero_counts": counts,
            "hetero_phrase": _heteroatom_phrase(counts),
            "ring_sizes": sorted(len(r) for r in system["rings"]),
            "n_ring_carbonyls": len(co_groups), "carbonyl_groups": co_groups,
            "carbonyl_details": co_details, "carbonyl_desc": _carbonyl_desc(co_groups)}

    # (0) Carbonyl-preserving dict: when a ring carries =O/=S, match first on the SMILES that keeps it.
    #     That rescues the names distorted by dropping the =O (benzimidazol-2-one and friends).
    if co_groups:
        smi_exo = _submol_smiles_keep_exo(mol, atoms)
        if smi_exo and smi_exo in RING_SMILES_NAMES:
            # The name already carries '-one'/'dione'/'lactam', so no separate carbonyl phrase.
            info.update(name=RING_SMILES_NAMES[smi_exo], name_source="curated-carbonyl",
                        smiles_exo=smi_exo, carbonyl_desc="")
            return info

    # (1) exact match against the curated dict
    if smi and smi in RING_SMILES_NAMES:
        info.update(name=RING_SMILES_NAMES[smi], name_source="curated")
        return info

    # (2) a monocycle gets the systematic name
    if system["n_rings"] == 1:
        mono = name_monocycle(mol, tuple(system["rings"][0]))
        info.update(name=mono["name"], name_source=mono["name_source"],
                    locants=mono.get("locants"), saturated=mono.get("saturated"))
        return info

    # (3) polycyclic composite name: name each constituent ring and describe how they are fused
    info.update(name=_composite_name(mol, system), name_source="composite")
    return info


def _composite_name(mol, system: dict) -> str:
    """A polycycle absent from the curated dict: build a descriptive name from its constituent ring names and internal topology."""
    sub_names = []
    for ring in system["rings"]:
        sub_names.append(name_monocycle(mol, tuple(ring))["name"])
    topo = system["internal_topology"]
    join = {"ortho-fused": "fused", "spiro": "spiro-linked",
            "bridged": "bridged", "fused/spiro": "fused/spiro-linked"}.get(topo, "fused")
    # Repeated names collapse into 'two fused benzene rings' and the like. (A bare name is returned;
    # the 'ring system' suffix is added by the layer above, only where it is needed.)
    if len(set(sub_names)) == 1 and len(sub_names) == 2:
        return f"two {join} {sub_names[0]} rings"
    uniq = []
    for nm in sub_names:
        if nm not in uniq:
            uniq.append(nm)
    # Two rings: 'X fused/spiro-linked/bridged to Y', naming the connection explicitly.
    if len(sub_names) == 2:
        return (" " + join + " to ").join(uniq)
    # Three or more: avoid repeating the connective and summarise as 'N-ring system of A, B, and C'.
    # The connection kinds are kept separately in internal_topology (FACTS); every constituent ring name is listed faithfully.
    poly = {3: "tricyclic", 4: "tetracyclic", 5: "pentacyclic", 6: "hexacyclic",
            7: "heptacyclic", 8: "octacyclic"}.get(len(sub_names))
    base = f"{poly} ring system" if poly else f"{len(sub_names)}-ring system"
    if len(uniq) == 1:
        listing = uniq[0]
    elif len(uniq) == 2:
        listing = f"{uniq[0]} and {uniq[1]}"
    else:
        listing = ", ".join(uniq[:-1]) + ", and " + uniq[-1]
    return f"{base} of {listing}"


# --------------------------------------------------------------------------- #
#  Connections between ring systems: a direct bond (biaryl, ...) vs. a chain linker
# --------------------------------------------------------------------------- #
# Linker pattern: chain atoms + pendant (=O/=S/=N) + bond orders -> a readable pattern string and a
# functional-group name. Pendant double-bonded atoms (the =O of a carbonyl or sulfonyl, the =S of a
# thiocarbonyl) are excluded from the backbone length but always shown in the pattern and the name,
def _bond_char(bond) -> str:
    """Bond order as a pattern connector: single '-', double '=', triple '#'."""
    if bond is None:
        return "-"
    bt = bond.GetBondType()
    if bt == Chem.BondType.DOUBLE:
        return "="
    if bt == Chem.BondType.TRIPLE:
        return "#"
    return "-"


def _pendants(mol, idx: int, exclude: set) -> list[tuple]:
    """The double/triple-bonded pendant atoms of backbone atom idx, excluding the backbone and ring attachments. [(element, '='|'#'), ...].

    The Murcko reduction removes single-bonded terminal substituents, so what is left as a pendant is
    a double-bonded atom: a carbonyl =O, a thiocarbonyl =S, an imine =N. These feed the pattern and the functional-group classification."""
    out = []
    a = mol.GetAtomWithIdx(idx)
    for b in a.GetBonds():
        o = b.GetOtherAtom(a)
        if o.GetIdx() in exclude:
            continue
        bt = b.GetBondType()
        if bt == Chem.BondType.DOUBLE:
            out.append((_PT.GetElementSymbol(o.GetAtomicNum()), "="))
        elif bt == Chem.BondType.TRIPLE:
            out.append((_PT.GetElementSymbol(o.GetAtomicNum()), "#"))
    return out


def _linker_token(mol, idx: int, sym: str, pend: list[tuple]) -> str:
    """One backbone atom as a readable token (CH2 / C(=O) / NH / S(=O)2 / O ...)."""
    if sym == "S":
        no = sum(1 for e, o in pend if e == "O" and o == "=")
        return "S(=O)2" if no >= 2 else ("S(=O)" if no == 1 else "S")
    suf = ""
    for e, o in pend:
        if o == "=":
            suf += {"O": "(=O)", "S": "(=S)", "N": "(=NH)"}.get(e, f"(={e})")
        else:
            suf += f"(#{e})"
    if sym == "C":
        if suf:
            return "C" + suf
        h = mol.GetAtomWithIdx(idx).GetTotalNumHs()
        return {0: "C", 1: "CH", 2: "CH2", 3: "CH3"}.get(h, "C")
    if sym == "N":
        if suf:
            return "N" + suf
        return "NH" if mol.GetAtomWithIdx(idx).GetTotalNumHs() >= 1 else "N"
    return sym + suf


def _scan_motifs(mol, chain: list[int], elems: list[str], pend: list[list],
                 conns: list[str], end_a: int, end_b: int) -> list[str]:
    """Collect the functional groups a linker unambiguously contains (amide, ester, urea, sulfonamide,
    ether, imine, ...) from left to right. Read straight off the graph rather than heuristically, so it is faithful at any length."""
    L = len(chain)
    groups: list[str] = []

    def add(g):
        if g not in groups:
            groups.append(g)

    el_a = _PT.GetElementSymbol(_z(mol.GetAtomWithIdx(end_a)))
    el_b = _PT.GetElementSymbol(_z(mol.GetAtomWithIdx(end_b)))

    def lel(p):
        return elems[p - 1] if p > 0 else el_a

    def rel(p):
        return elems[p + 1] if p < L - 1 else el_b

    def has(p, e):
        return (e, "=") in pend[p]

    for p in range(L):
        e = elems[p]
        if e == "C":
            if has(p, "O") or has(p, "S"):
                o = has(p, "O")
                fset = {lel(p), rel(p)}
                if has(p, "N"):                       # =O/=S together with =N -> an imidate-type group
                    add("amidine" if "N" in fset else "imino-acyl")
                elif fset == {"N"}:
                    add("urea" if o else "thiourea")
                elif fset == {"N", "O"}:
                    add("carbamate" if o else "thiocarbamate")
                elif fset == {"O"}:
                    add("carbonate" if o else "thiocarbonate")
                elif fset == {"N", "S"} or fset == {"S"} and "N" in (lel(p), rel(p)):
                    add("thioamide" if not o else "amide")
                elif "N" in fset:
                    add("amide" if o else "thioamide")
                elif "O" in fset:
                    add("ester" if o else "thionoester")
                elif "S" in fset:
                    add("thioester" if o else "dithioester")
                else:
                    add("ketone" if o else "thioketone")
            elif has(p, "N"):                          # a pendant =N (double bond)
                fset = {lel(p), rel(p)}
                if "N" in fset:
                    add("guanidine" if list(fset).count("N") + (lel(p) == "N") + (rel(p) == "N") >= 2
                        else "amidine")
                else:
                    add("amidine")
        elif e == "S":
            no = sum(1 for x, o in pend[p] if x == "O" and o == "=")
            fset = {lel(p), rel(p)}
            if no >= 2:
                add("sulfonamide" if "N" in fset else "sulfone")
            elif no == 1:
                add("sulfinamide" if "N" in fset else "sulfoxide")
            elif "C" in fset:
                add("thioether")
        elif e == "O":
            up_co = p > 0 and elems[p - 1] == "C" and has(p - 1, "O")
            dn_co = p < L - 1 and elems[p + 1] == "C" and has(p + 1, "O")
            # ether means C-O-C only: the O of an ester, a carbamate or an oxime ether is excluded.
            if not (up_co or dn_co) and lel(p) == "C" and rel(p) == "C":
                add("ether")
        elif e == "N":
            adj_unsat_c = (
                (p > 0 and elems[p - 1] == "C" and (has(p - 1, "O") or has(p - 1, "S") or has(p - 1, "N")))
                or (p < L - 1 and elems[p + 1] == "C"
                    and (has(p + 1, "O") or has(p + 1, "S") or has(p + 1, "N"))))
            adj_s = (p > 0 and elems[p - 1] == "S") or (p < L - 1 and elems[p + 1] == "S")
            if not (adj_unsat_c or adj_s) and "C" in {lel(p), rel(p)} \
                    and all(c == "-" for c in (conns[p - 1:p] + conns[p:p + 1])):
                add("amine")

    # backbone bond-order motifs
    for p in range(L - 1):
        a, b, c = elems[p], elems[p + 1], conns[p]
        if c == "#":
            add("alkyne")
        elif c == "=":
            if {a, b} == {"C"}:
                add("alkene")
            elif {a, b} == {"N"}:
                add("azo")
            elif {a, b} == {"C", "N"}:
                npos = p if a == "N" else p + 1
                near_o = any(elems[q] == "O" and conns[min(q, npos)] == "-"
                             for q in (npos - 1, npos + 1) if 0 <= q < L)
                add("oxime ether" if near_o else "imine")
    # N-N single bond: a hydrazide (adjacent C=O) vs. a hydrazo group
    for p in range(L - 1):
        if elems[p] == "N" and elems[p + 1] == "N" and conns[p] == "-":
            adj_co = (p > 0 and elems[p - 1] == "C" and has(p - 1, "O")) or \
                     (p + 2 < L and elems[p + 2] == "C" and has(p + 2, "O"))
            add("hydrazide" if adj_co else "hydrazo")
    return _dedupe_groups(groups)


# When a larger group subsumes its parts (amide / amine / ester / ether, ...), drop the duplicate labels.
_GROUP_SUBSUMES = {
    "urea": {"amide", "amine"}, "thiourea": {"thioamide", "amine"},
    "guanidine": {"amidine", "amine", "imine"}, "amidine": {"amine", "imine"},
    "carbamate": {"amide", "ester", "amine", "ether"},
    "thiocarbamate": {"thioamide", "amine", "ester", "ether"},
    "carbonate": {"ester", "ether"}, "hydrazide": {"amide", "amine", "hydrazo"},
    "sulfonamide": {"amine", "sulfone"}, "sulfinamide": {"amine", "sulfoxide"},
    "oxime ether": {"imine", "ether"}, "hydrazo": {"amine"}, "azo": {"amine"},
}


def _dedupe_groups(groups: list[str]) -> list[str]:
    """Remove the subsumed sub-groups, preserving order. e.g. ['amine','guanidine'] -> ['guanidine']."""
    drop: set = set()
    for g in groups:
        drop |= _GROUP_SUBSUMES.get(g, set())
    return [g for g in groups if g not in drop]


def _name_linker(groups: list[str], elems: list[str], conns: list[str], L: int) -> str:
    """Name the common two-ring bridges of length <= 3 exactly; anything else exposes its pattern and groups as they are."""
    single = {
        "urea": "urea linker", "thiourea": "thiourea linker",
        "guanidine": "guanidine linker", "carbamate": "carbamate linkage",
        "thiocarbamate": "thiocarbamate linkage", "carbonate": "carbonate linkage",
        "amide": "amide linker", "thioamide": "thioamide linker",
        "ester": "ester linkage", "thioester": "thioester linkage",
        "ketone": "ketone (carbonyl) linker", "thioketone": "thiocarbonyl linker",
        "ether": "ether linker", "thioether": "thioether linker",
        "amine": "amino linker", "sulfonamide": "sulfonamide linker",
        "sulfone": "sulfonyl linker", "sulfoxide": "sulfinyl linker",
        "sulfinamide": "sulfinamide linker", "imine": "imine linker",
        "oxime ether": "oxime-ether linker", "amidine": "amidine linker",
        "azo": "azo linker", "hydrazo": "hydrazo linker", "hydrazide": "hydrazide linker",
        "alkyne": "alkyne linker", "alkene": "alkene linker",
    }
    if groups == ["amine"]:                            # an amine-only chain is pinned down exactly by its composition
        if L == 1:
            return "amino linker (-NH-)"
        if sorted(elems) == ["C", "N"]:
            return "aminomethylene linker (-CH2-NH-)"
        if elems == ["N", "C", "N"]:
            return "methylenediamine (aminal) linker (-NH-CH2-NH-)"
        return f"{L}-atom amino linker"
    if L <= 3 and len(groups) == 1 and groups[0] in single:
        return single[groups[0]]
    if L == 1 and not groups and elems == ["C"]:
        return "methylene linker (-CH2-)"
    if L == 2 and not groups and elems == ["C", "C"]:
        return "ethylene linker (-CH2CH2-)"
    if L == 3 and not groups and elems == ["C", "C", "C"]:
        return "propylene linker (-CH2CH2CH2-)"
    if groups:
        return f"{L}-atom linker ({' + '.join(groups)})"
    comp = _heteroatom_phrase({e: elems.count(e) for e in set(elems) if e != "C"})
    cc = elems.count("C")
    bits = ", ".join(b for b in (f"{cc} carbon{'s' if cc != 1 else ''}" if cc else "",
                                 comp if any(e != "C" for e in elems) else "") if b)
    return f"{L}-atom linker ({bits})" if bits else f"{L}-atom linker"


def classify_linker(mol, chain_atoms: list[int], end_a: int, end_b: int) -> dict:
    """Classify the acyclic chain joining two ring systems (chain_atoms, with ring atoms end_a/end_b at each end).

    An empty chain_atoms means the two rings are bonded **directly** (biaryl or another direct link). Besides
    the kind (type) and the functional groups, the returned dict carries the **total length, the number of
    bonds (bond_span), a readable pattern (including pendant =O/=S and the bond orders), the atom sequence
    (atom_sequence: [{index, element, nH, pendant}] ordered end_a -> end_b) and the ring atoms at each
    end (attach_a / attach_b)**."""
    elems = [_PT.GetElementSymbol(_z(mol.GetAtomWithIdx(i))) for i in chain_atoms]
    base = {"kind": "linker" if chain_atoms else "direct",
            "length": len(chain_atoms), "bond_span": len(chain_atoms) + 1,
            "atoms": list(chain_atoms), "elements": elems,
            "attach_a": end_a, "attach_b": end_b, "functional_groups": []}

    if not chain_atoms:
        bond = mol.GetBondBetweenAtoms(end_a, end_b)
        bt = bond.GetBondType() if bond else None
        both_arom = mol.GetAtomWithIdx(end_a).GetIsAromatic() and \
            mol.GetAtomWithIdx(end_b).GetIsAromatic()
        if bt == Chem.BondType.TRIPLE:
            base.update(type="alkyne (direct -C#C-)", pattern="-#-", smarts="#")
        elif bt == Chem.BondType.DOUBLE:
            base.update(type="direct double bond", pattern="-=-", smarts="=")
        elif both_arom:
            base.update(type="biaryl linkage", pattern="single bond", smarts="single bond")
        else:
            base.update(type="directly linked (single bond)", pattern="single bond",
                        smarts="single bond")
        base["atom_sequence"] = []
        return base

    # pendants (=O/=S/=N) and the backbone bond orders
    bset = set(chain_atoms)
    pend = [_pendants(mol, i, bset | {end_a, end_b}) for i in chain_atoms]
    conns = [_bond_char(mol.GetBondBetweenAtoms(chain_atoms[p], chain_atoms[p + 1]))
             for p in range(len(chain_atoms) - 1)]
    toks = [_linker_token(mol, chain_atoms[p], elems[p], pend[p])
            for p in range(len(chain_atoms))]
    pattern = "-" + toks[0] + "".join(conns[p - 1] + toks[p]
                                      for p in range(1, len(toks))) + "-"

    base["atom_sequence"] = [
        {"index": chain_atoms[p], "element": elems[p],
         "nH": mol.GetAtomWithIdx(chain_atoms[p]).GetTotalNumHs(),
         "pendant": [{"element": e, "order": o} for e, o in pend[p]]}
        for p in range(len(chain_atoms))]
    groups = _scan_motifs(mol, chain_atoms, elems, pend, conns, end_a, end_b)
    name = _name_linker(groups, elems, conns, len(chain_atoms))
    base.update(type=name, pattern=pattern, functional_groups=groups,
                smarts="".join(elems))
    return base


def _shortest_path(mol, src: int, dst: int, allowed: set) -> list[int]:
    """Shortest path src -> dst as atom indices, travelling only through atoms in `allowed`. [] when unreachable."""
    if src == dst:
        return [src]
    prev = {src: None}
    q = [src]
    while q:
        nq = []
        for u in q:
            for nb in mol.GetAtomWithIdx(u).GetNeighbors():
                k = nb.GetIdx()
                if k in prev or (k != dst and k not in allowed):
                    continue
                prev[k] = u
                if k == dst:
                    path = [dst]
                    while prev[path[-1]] is not None:
                        path.append(prev[path[-1]])
                    return list(reversed(path))
                nq.append(k)
        q = nq
    return []


def find_connections(mol, systems: list[dict]) -> list[dict]:
    """Find every connection between ring systems: direct bonds and chain linkers alike.

    For each connected component of acyclic (linker) atoms, the ring systems it touches are found and a
    linker is registered for that pair. Ring atoms of different systems bonded directly are caught too.
    Returns [{a, b, relation('directly linked'|'linker-connected'|...), type, ...}].
    """
    atom2sys: dict[int, int] = {}
    for s in systems:
        for a in s["atoms"]:
            atom2sys[a] = s["id"]
    ring_atoms = set(atom2sys)
    conns = []
    seen_pairs: set = set()

    # (a) direct bonds: a bond between ring atoms of two different systems
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in atom2sys and j in atom2sys and atom2sys[i] != atom2sys[j]:
            sa, sb = atom2sys[i], atom2sys[j]
            a_atom, b_atom = (i, j) if sa <= sb else (j, i)
            if sa > sb:
                sa, sb = sb, sa
            lk = classify_linker(mol, [], a_atom, b_atom)
            conns.append({"a": sa, "b": sb, "relation": "directly linked",
                          "type": lk["type"], "linker": lk,
                          "a_atom": a_atom, "b_atom": b_atom})
            seen_pairs.add((sa, sb))

    # (b) chain linkers: connected components of acyclic atoms
    linker_atoms = [a.GetIdx() for a in mol.GetAtoms() if a.GetIdx() not in ring_atoms]
    la_set = set(linker_atoms)
    visited: set = set()
    for start in linker_atoms:
        if start in visited:
            continue
        comp = []
        stack = [start]
        touch: dict[int, int] = {}   # system_id -> the ring atom through which it is entered
        while stack:
            x = stack.pop()
            if x in visited:
                continue
            visited.add(x)
            comp.append(x)
            for nb in mol.GetAtomWithIdx(x).GetNeighbors():
                k = nb.GetIdx()
                if k in la_set and k not in visited:
                    stack.append(k)
                elif k in atom2sys:
                    touch[atom2sys[k]] = k
        if len(touch) >= 2:
            sids = sorted(touch)
            comp_set = set(comp)
            # Register a linker for each pair of systems touched, normally two. The linker's 'backbone'
            # is the intermediate atoms on the shortest path between the two ring attachment atoms;
            # pendants (a sulfonyl or carbonyl =O) stay out of the length, but a backbone =O is detected via has_dbl_o to tell amide from sulfonamide.
            for a in range(len(sids)):
                for b in range(a + 1, len(sids)):
                    sa, sb = sids[a], sids[b]
                    path = _shortest_path(mol, touch[sa], touch[sb], comp_set)
                    backbone = path[1:-1] if len(path) >= 2 else comp
                    lk = classify_linker(mol, backbone, touch[sa], touch[sb])
                    conns.append({"a": sa, "b": sb, "relation": "linker-connected",
                                  "type": lk["type"], "linker": lk,
                                  "a_atom": touch[sa], "b_atom": touch[sb]})
    return conns


def _atom_ring_position(mol, system: dict, atom_idx: int, ring_name: str) -> dict:
    """The exact position (locant / relation / label) of one atom in a ring system, normally a linker attachment point."""
    containing = next((r for r in system["rings"] if atom_idx in r), None)
    locant = (_ring_locant(mol, tuple(containing), atom_idx)
              if system["n_rings"] == 1 and containing else None)
    rel = _hetero_relation(mol, system["atoms"], atom_idx)
    is_fusion = sum(1 for r in system["rings"] if atom_idx in r) >= 2
    sym = _PT.GetElementSymbol(_z(mol.GetAtomWithIdx(atom_idx)))
    return {"atom": atom_idx, "element": sym, "locant": locant,
            "hetero_relation": rel,
            "position": _position_label(sym, locant, rel, is_fusion, ring_name)}


# --------------------------------------------------------------------------- #
#  Substituent removal + attachment-point records
# --------------------------------------------------------------------------- #
def _benzene_substitution_relations(mol, ring_atoms: tuple, subbed: set) -> list[str]:
    """Compute the relation between substitution positions on a six-membered aromatic ring (ortho/meta/para) reliably."""
    order = _ring_order(mol, ring_atoms)
    if len(order) != 6:
        return []
    rel = {1: "ortho", 2: "meta", 3: "para"}
    pos = [order.index(a) for a in subbed if a in order]
    out = set()
    for x in range(len(pos)):
        for y in range(x + 1, len(pos)):
            d = min((pos[x] - pos[y]) % 6, (pos[y] - pos[x]) % 6)
            if d in rel:
                out.add(rel[d])
    # sort into the natural order (ortho < meta < para)
    rank = {"ortho": 0, "meta": 1, "para": 2}
    return sorted(out, key=lambda r: rank[r])


def _ring_locant(mol, ring_atoms: tuple, atom_idx: int) -> Optional[int]:
    """The IUPAC-style lowest locant of atom_idx on a monocycle (heteroatoms take the lowest numbers; ties
    break toward the target atom). None for an all-carbon ring, where symmetry makes an absolute locant meaningless, or when it cannot be derived."""
    order = _ring_order(mol, list(ring_atoms))
    if atom_idx not in order:
        return None
    if all(_z(mol.GetAtomWithIdx(i)) == 6 for i in order):
        return None
    rank = {6: 1}  # heteroatom (0) < carbon (1), so heteroatoms take the lowest locant

    def key(rot):
        return tuple((rank.get(_z(mol.GetAtomWithIdx(i)), 0),
                      _PT.GetElementSymbol(_z(mol.GetAtomWithIdx(i)))) for i in rot)

    n = len(order)
    cands = []
    for direction in (order, list(reversed(order))):
        for r in range(n):
            cands.append(direction[r:] + direction[:r])
    best_key = min(key(rot) for rot in cands)
    return min(rot.index(atom_idx) + 1 for rot in cands if key(rot) == best_key)


def _hetero_relation(mol, sys_atoms: set, atom_idx: int) -> Optional[str]:
    """The distance from atom_idx to the nearest ring heteroatom, in words (adjacent / two bonds from ...).
    None when the system has no heteroatom. BFS over the system subgraph, so fused systems work too."""
    dist = {atom_idx: 0}
    q = [atom_idx]
    while q:
        nq = []
        for u in q:
            for nb in mol.GetAtomWithIdx(u).GetNeighbors():
                k = nb.GetIdx()
                if k in sys_atoms and k not in dist:
                    dist[k] = dist[u] + 1
                    nq.append(k)
        q = nq
    het = [(d, _PT.GetElementSymbol(_z(mol.GetAtomWithIdx(i))))
           for i, d in dist.items() if i != atom_idx and _z(mol.GetAtomWithIdx(i)) != 6]
    if not het:
        return None
    d, e = min(het)
    name = {"N": "nitrogen", "O": "oxygen", "S": "sulfur"}.get(e, e)
    word = {1: "adjacent to", 2: "two bonds from", 3: "three bonds from"}.get(
        d, f"{d} bonds from")
    return f"{word} a ring {name}"


def _position_label(sym: str, locant: Optional[int], rel: Optional[str],
                    is_fusion: bool, ring_name: str) -> str:
    """A human-readable, faithful label for the exact position of an attachment point."""
    name = {"N": "nitrogen", "O": "oxygen", "S": "sulfur", "C": "carbon"}.get(sym, sym)
    if locant is not None:
        return f"position {locant} of the {ring_name}"
    if sym in ("N", "O", "S"):
        # The heteroatom identifies the position by itself, so no 'distance to another heteroatom' is
        # appended (which would read awkwardly, as in "a ring nitrogen 4 bonds from a ring nitrogen").
        return f"a ring {name}"
    if rel:
        return f"a ring carbon {rel}"
    if is_fusion:
        return "a ring-fusion position"
    return "a ring carbon"


def find_attachment_points(mol, scaf, match: tuple, systems: list[dict],
                           names: Optional[dict] = None) -> dict:
    """Record the sites left by removed substituents (attachment points) **together with their exact position**.

    Each point: scaffold_atom (index), element, n_substituents, plus the position information —
      locant         : the IUPAC-style ring number on a monocycle (heteroatom lowest), otherwise None
      hetero_relation: the relation to the nearest ring heteroatom ('adjacent to a ring nitrogen', ...)
      is_fusion_atom : whether the atom is a ring-fusion position
      position       : a human-readable label combining the above
    For benzene (a single six-membered aromatic carbocycle) the ortho/meta/para relations between substitution positions are given as well.
    Attachments on a linker are collected separately under out["linker"], indexable against the connections.
    Returns {"per_system": {sid: {...}}, "linker": [...], "n_total": int}.
    """
    out = {"per_system": {}, "linker": [], "n_total": 0}
    if not match:
        return out
    names = names or {}
    matchset = set(match)
    sysid_of: dict[int, int] = {}
    sysobj: dict[int, dict] = {}
    for s in systems:
        sysobj[s["id"]] = s
        for a in s["atoms"]:
            sysid_of[a] = s["id"]

    sys_attach: dict[int, list[dict]] = {s["id"]: [] for s in systems}
    for s_idx, mol_idx in enumerate(match):
        a = mol.GetAtomWithIdx(mol_idx)
        removed = [nb for nb in a.GetNeighbors() if nb.GetIdx() not in matchset]
        if not removed:
            continue
        sym = _PT.GetElementSymbol(_z(scaf.GetAtomWithIdx(s_idx)))
        rec = {"scaffold_atom": s_idx, "element": sym,
               "in_ring": scaf.GetAtomWithIdx(s_idx).IsInRing(),
               "n_substituents": len(removed)}
        if s_idx in sysid_of:
            s = sysobj[sysid_of[s_idx]]
            containing = next((r for r in s["rings"] if s_idx in r), None)
            locant = (_ring_locant(scaf, tuple(containing), s_idx)
                      if s["n_rings"] == 1 and containing else None)
            rel = _hetero_relation(scaf, s["atoms"], s_idx)
            is_fusion = sum(1 for r in s["rings"] if s_idx in r) >= 2
            rec.update(ring_system=s["id"], locant=locant, hetero_relation=rel,
                       is_fusion_atom=is_fusion,
                       position=_position_label(sym, locant, rel, is_fusion,
                                                names.get(s["id"], "ring")))
            sys_attach[s["id"]].append(rec)
        else:
            rec.update(on="linker", position="on the linker chain")
            out["linker"].append(rec)

    n_total = 0
    for s in systems:
        recs = sys_attach[s["id"]]
        n_total += sum(r["n_substituents"] for r in recs)
        entry = {"points": recs, "n_points": len(recs)}
        if s["n_rings"] == 1:
            ring = tuple(s["rings"][0])
            if len(ring) == 6 and all(scaf.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
                # 'substituted positions' = the sites of removed substituents, union the sites where a
                # bond leaves the ring (to a linker or another ring system). Both are needed for the
                exo = {i for i in ring
                       if any(nb.GetIdx() not in s["atoms"]
                              for nb in scaf.GetAtomWithIdx(i).GetNeighbors())}
                subbed = {r["scaffold_atom"] for r in recs} | exo
                rels = _benzene_substitution_relations(scaf, ring, subbed)
                if rels:
                    entry["benzene_relations"] = rels
        out["per_system"][s["id"]] = entry
    out["n_total"] = n_total
    return out


# --------------------------------------------------------------------------- #
#  Top level: analyze_scaffold
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
#  Evaluation query builder: what is 'asserted' becomes a single SMARTS, what is 'unasserted' becomes a match-any set J.
#  Only the free attachment points are re-attached at symmetry-distinct candidates to form the family J, whose members
#  are emitted as (a) a whole-SMARTS match-any set (the authoritative verdict) and (b) a five-dimension projected dedup
#  (the diagnostic set). Everything asserted or symmetric gives J=1, a single SMARTS; any free point gives J>1, a set.
# --------------------------------------------------------------------------- #
def _bfs_dist_within(mol, src: int, allowed: set) -> dict:
    """Graph distance from src within `allowed`, normally one ring system."""
    dist = {src: 0}
    q = [src]
    while q:
        nq = []
        for u in q:
            for nb in mol.GetAtomWithIdx(u).GetNeighbors():
                k = nb.GetIdx()
                if k in allowed and k not in dist:
                    dist[k] = dist[u] + 1
                    nq.append(k)
        q = nq
    return dist


def _system_sym_classes(mol, atoms: set) -> dict:
    """Atom -> symmetry class under the ring system's **intrinsic** symmetry, with attached linkers removed.
    Computed on the bare submol, so a symmetric ring such as monosubstituted benzene collapses to a single candidate even when 'free'."""
    order = sorted(atoms)
    rw = Chem.RWMol()
    idxmap = {}
    for a in order:
        idxmap[a] = rw.AddAtom(Chem.Atom(mol.GetAtomWithIdx(a).GetAtomicNum()))
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in idxmap and j in idxmap:
            rw.AddBond(idxmap[i], idxmap[j], b.GetBondType())
    sub = rw.GetMol()
    try:
        Chem.SanitizeMol(sub)
    except Exception:  # noqa: BLE001
        pass
    try:
        ranks = list(Chem.CanonicalRankAtoms(sub, breakTies=False))
        return {a: ranks[idxmap[a]] for a in order}
    except Exception:  # noqa: BLE001
        return {a: a for a in order}


def _eligible_free_atoms(mol, system: dict, pos: Optional[dict]):
    """The candidate ring atoms matching the description when an attachment end is 'free'; None when it is asserted.

    - A locant (e.g. 'position 4') means asserted -> None.
    - 'a ring carbon' -> every ring carbon of that system.
    - 'k bonds from a ring nitrogen/oxygen/sulfur' -> the carbons satisfying that distance.
    - 'a ring nitrogen/oxygen/sulfur' -> those heteroatoms."""
    if not pos or pos.get("locant") is not None:
        return None
    label = pos.get("position", "") or ""
    atoms = system["atoms"]
    if label == "a ring carbon":
        return [a for a in atoms if _z(mol.GetAtomWithIdx(a)) == 6]
    m = re.search(r"(adjacent to|two bonds from|three bonds from|(\d+) bonds from) "
                  r"a ring (nitrogen|oxygen|sulfur)", label)
    if m:
        dist = {"adjacent to": 1, "two bonds from": 2, "three bonds from": 3}.get(m.group(1))
        if dist is None and m.group(2):
            dist = int(m.group(2))
        hetZ = {"nitrogen": 7, "oxygen": 8, "sulfur": 16}[m.group(3)]
        hets = [a for a in atoms if _z(mol.GetAtomWithIdx(a)) == hetZ]
        out = []
        for a in atoms:
            if _z(mol.GetAtomWithIdx(a)) != 6:
                continue
            d = _bfs_dist_within(mol, a, set(atoms))
            if any(d.get(h) == dist for h in hets):
                out.append(a)
        return out or None
    if label in ("a ring nitrogen", "a ring oxygen", "a ring sulfur"):
        Z = {"a ring nitrogen": 7, "a ring oxygen": 8, "a ring sulfur": 16}[label]
        return [a for a in atoms if _z(mol.GetAtomWithIdx(a)) == Z]
    return None


def _linker_neighbor(mol, ring_atom: int, system_atoms: set):
    """The non-ring (linker-side) neighbour of ring_atom: the linker end of the bond moved during re-attachment."""
    for nb in mol.GetAtomWithIdx(ring_atom).GetNeighbors():
        if nb.GetIdx() not in system_atoms:
            return nb.GetIdx()
    return None


def _distinct_label_assignments(ring_atoms: list, elems: list, cap: int = 400) -> list:
    """The distinct ways of placing the element multiset (elems) onto ring_atoms, as dicts atom_idx -> Z.

    Swaps between identical elements (many carbons above all) are removed naturally by itertools.combinations,
    so there is no factorial blow-up. Placements made equal by ring symmetry are folded once more by the
    caller's canonical-SMARTS dedup (thiazole's 20 placements converge to 1,2- and 1,3-)."""
    n = len(ring_atoms)
    hetero_counts: dict = {}
    for z in elems:
        if z != 6:
            hetero_counts[z] = hetero_counts.get(z, 0) + 1
    if not hetero_counts:
        return []
    partials = [{}]
    for z, cnt in hetero_counts.items():
        nxt = []
        for partial in partials:
            used = set(partial)
            free = [i for i in range(n) if i not in used]
            for combo in itertools.combinations(free, cnt):
                d = dict(partial)
                for i in combo:
                    d[i] = z
                nxt.append(d)
                if len(nxt) >= cap:
                    break
            if len(nxt) >= cap:
                break
        partials = nxt
    return [{ring_atoms[i]: partial.get(i, 6) for i in range(n)} for partial in partials]


def _arrangement_canon_key(n: int, edges: list, atomicnums: list):
    """An isomorphism-invariant certificate of (connectivity graph + atomic-number labels). Placements that are
    rotations or reflections of the same graph fold to the same key (thiazole's 20 labellings -> 1,2- and 1,3-).

    Every atom is set to carbon and the element identity is encoded as an isotope label, giving a single-bonded
    graph whose canonical SMILES is the key. Being all carbon, sanitisation never fails on valence, and the
    canonical SMILES reflects the isotopes, so it is isomorphism-invariant and symmetric placements fold to
    exactly one key. (MolToSmarts does not canonicalise query atoms and cannot fold the symmetry.)"""
    rw = Chem.RWMol()
    for z in atomicnums:
        a = Chem.Atom(6)
        a.SetIsotope(int(z))                        # encode the element as an isotope label, keeping all-carbon
        rw.AddAtom(a)
    for i, j in edges:
        rw.AddBond(int(i), int(j), Chem.BondType.SINGLE)
    try:
        m = rw.GetMol()
        Chem.SanitizeMol(m)
        return Chem.MolToSmiles(m)
    except Exception:  # noqa: BLE001
        return tuple(atomicnums)                     # fallback: no folding, which is safe


def _element_arrangement_set(mol, systems: list, base: str, cap: int = 64) -> list:
    """The element-dimension match-any set: the element projections of the placements that move the heteroatom
    *positions* around within each ring system.

    A description normally states the kind and count of the elements without pinning their exact arrangement
    ("a five-membered ring with one nitrogen and one sulfur" leaves thiazole 1,3 vs. isothiazole 1,2 open). So
    the element dimension becomes a diagnostic set that fixes composition and skeleton connectivity while
    leaving the heteroatom arrangement free. The skeleton/aromaticity/bond/ring dimensions do not depend on
    element identity and are unaffected, and verdict_match_any — the authoritative check — stays strict on the real arrangement.

    Placements made equal by symmetry are folded through graph canonicalisation (_arrangement_canon_key)
    so only a minimal, duplicate-free set remains. (Symmetry broken by an attachment is reflected automatically, since the key is computed on the whole-scaffold graph.)"""
    n = mol.GetNumAtoms()
    edges = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    base_Z = [mol.GetAtomWithIdx(i).GetAtomicNum() for i in range(n)]
    per_sys = []   # per element-mixed ring system: [list of placements, the real (base) placement dict]
    for s in systems:
        atoms = [a for a in s.get("atoms", [])
                 if 0 <= a < n and mol.GetAtomWithIdx(a).IsInRing()]
        if not atoms:
            continue
        elems = [base_Z[a] for a in atoms]
        if all(z == 6 for z in elems):
            continue                                # all-carbon ring: no heteroatom to move
        opts = _distinct_label_assignments(sorted(atoms), elems)
        if len(opts) > 1:
            per_sys.append([opts, {a: base_Z[a] for a in atoms}])
    if not per_sys:
        return [base] if base else []
    # Avoid a cross-product blow-up: pin the systems with the most placements to their real arrangement until product <= cap.
    # Pinning is used instead of truncating midway (which would drop members arbitrarily), so the sets of the
    # smaller systems left free stay complete. A complex fused system with many heteroatoms usually has a proper name anyway, so pinning its arrangement is natural.
    def _product():
        p = 1
        for opts, _ in per_sys:
            p *= len(opts)
        return p
    per_sys.sort(key=lambda x: len(x[0]), reverse=True)
    while _product() > cap and len(per_sys[0][0]) > 1:
        per_sys[0][0] = [per_sys[0][1]]             # pin the largest ring system to its real arrangement
        per_sys.sort(key=lambda x: len(x[0]), reverse=True)
    out, seen_key, seen_sm = [], set(), set()
    if base:
        out.append(base)
        seen_sm.add(base)
    seen_key.add(_arrangement_canon_key(n, edges, base_Z))
    for combo in itertools.product(*[opts for opts, _ in per_sys]):
        Z = list(base_Z)
        for assign in combo:
            for a, z in assign.items():
                Z[a] = z
        key = _arrangement_canon_key(n, edges, Z)
        if key in seen_key:                         # a placement already seen up to symmetry
            continue
        seen_key.add(key)
        rw = Chem.RWMol(mol)
        for a in range(n):
            if Z[a] != base_Z[a]:
                rw.GetAtomWithIdx(a).SetAtomicNum(Z[a])
        sm = _project_dimension_smarts(rw.GetMol(), "element")
        if sm and sm not in seen_sm:
            seen_sm.add(sm)
            out.append(sm)
    return out


def _build_decompositions(mol, named: list, systems: list) -> list:
    """Decompose an unnamed composite (polycyclic) core into a 'constituent rings + fusions' spec and store it.

    Rather than pinning the exact fusion graph as a whole (which combinatorially explodes), this keeps only the
    SMARTS of each constituent ring and how each pair of rings meets (number of shared atoms / adjacency /
    spiro / fused / bridged). At evaluation time match_decomposition() checks only that every constituent ring
    is present and that they overlap as described, which gives a match-any that leaves the exact fusion position free."""
    nm = {ns["id"]: ns for ns in named}
    out = []
    for s in systems:
        info = nm.get(s["id"], {})
        if info.get("name_source") != "composite" or s["n_rings"] < 2:
            continue
        rings = s["rings"]
        comps = [_atoms_to_smarts(mol, r) for r in rings]
        if any(not c for c in comps):
            continue
        pairs = []
        for i in range(len(rings)):
            for j in range(i + 1, len(rings)):
                sh = rings[i] & rings[j]
                if not sh:
                    continue
                rel = _pair_relation(mol, rings[i], rings[j]) or "fused"
                adj = any(mol.GetBondBetweenAtoms(a, b) is not None
                          for a in sh for b in sh if a < b)
                pairs.append({"i": i, "j": j, "relation": rel,
                              "shared": len(sh), "shared_adjacent": bool(adj)})
        out.append({"system_id": s["id"], "name": info.get("name", ""),
                    "topology": s["internal_topology"], "n_components": len(rings),
                    "components": comps, "pair_fusions": pairs})
    return out


def match_decomposition(mol, decomp: dict, cap: int = 5000) -> bool:
    """Does the molecule satisfy one composite decomposition spec (constituent rings + fusions)?

    Each constituent-ring SMARTS is located in the molecule, and the answer is True if at least one assignment of
    distinct rings satisfies the shared-atom conditions in pair_fusions (count / adjacency / spiro)."""
    comps = [Chem.MolFromSmarts(s) for s in decomp.get("components", [])]
    if not comps or any(c is None for c in comps):
        return False
    matches = [mol.GetSubstructMatches(c, uniquify=True) for c in comps]
    if any(len(m) == 0 for m in matches):
        return False
    cnt = 0
    for combo in itertools.product(*matches):
        cnt += 1
        if cnt > cap:
            break
        sets = [set(m) for m in combo]
        if len({frozenset(x) for x in sets}) != len(sets):   # the constituent rings must be distinct
            continue
        ok = True
        for pf in decomp.get("pair_fusions", []):
            inter = sets[pf["i"]] & sets[pf["j"]]
            if len(inter) < pf["shared"]:
                ok = False
                break
            if pf["relation"] == "spiro" and len(inter) != 1:
                ok = False
                break
            if pf["relation"] == "fused" and pf.get("shared_adjacent") and not any(
                    mol.GetBondBetweenAtoms(a, b) is not None
                    for a in inter for b in inter if a != b):
                ok = False
                break
        if ok:
            return True
    return False


def build_eval_query(mol, named: list, connections: list, systems: list,
                     cap: int = 16) -> dict:
    """The evaluation spec: what is asserted stays as is, and only the free attachment points are re-attached at
    symmetry-distinct candidates to form the family J -> verdict_match_any (the whole SMARTS per member; matching
    any one of them passes) + dimension_sets (deduplicated per dimension).
    Everything asserted or symmetric gives J=1 (a single SMARTS); any free point gives J>1 (a set)."""
    sysmap = {s["id"]: s for s in systems}
    free_slots = []   # (linker_nbr, ring_atom, [candidate atoms])
    for c in connections:
        if c.get("relation") != "linker-connected":
            continue
        for poskey, atomkey, sid in (("a_position", "a_atom", c["a"]),
                                     ("b_position", "b_atom", c["b"])):
            if sid not in sysmap:
                continue
            ratom = c.get(atomkey)
            elig = _eligible_free_atoms(mol, sysmap[sid], c.get(poskey))
            if not elig or ratom is None:
                continue
            classes = _system_sym_classes(mol, sysmap[sid]["atoms"])
            reps = {}
            for a in elig:
                reps.setdefault(classes.get(a, a), a)
            cands = list(reps.values())
            if len(cands) <= 1:
                continue                       # only one possibility up to symmetry -> effectively asserted
            nbr = _linker_neighbor(mol, ratom, sysmap[sid]["atoms"])
            if nbr is None:
                continue
            free_slots.append((nbr, ratom, cands))

    members = [mol]
    if free_slots:
        combos = list(itertools.product(*[c[2] for c in free_slots]))[:cap]
        # Always keep the original scaffold as a member; everything below is an *additional* hypothesis. Because
        # the candidates for a free attachment point are chosen from the symmetry of the BARE ring system, a
        # substituent that breaks that symmetry can leave every variant mismatching the real molecule — dropping the original would open a hole where the answer molecule fails its own eval_query.
        members = [mol]
        for combo in combos:
            rw = Chem.RWMol(mol)
            ok = True
            for (nbr, ratom, _), newatom in zip(free_slots, combo):
                b = rw.GetBondBetweenAtoms(nbr, ratom)
                if b is None:
                    ok = False
                    break
                bt = b.GetBondType()
                rw.RemoveBond(nbr, ratom)
                if rw.GetBondBetweenAtoms(nbr, newatom) is None:
                    rw.AddBond(nbr, newatom, bt)
            if not ok:
                continue
            m2 = rw.GetMol()
            try:
                Chem.SanitizeMol(m2)
            except Exception:  # noqa: BLE001
                continue
            members.append(m2)
        if not members:
            members = [mol]

    seen, uniq = set(), []                     # canonical-SMILES dedup
    for m in members:
        try:
            k = Chem.MolToSmiles(m)
        except Exception:  # noqa: BLE001
            continue
        if k not in seen:
            seen.add(k)
            uniq.append(m)
    members = uniq or [mol]

    verdict = [s for s in (scaffold_to_smarts(m) for m in members) if s]
    dimsets = {}
    for dim in _DIMENSIONS:
        vals = []
        for m in members:
            s = _project_dimension_smarts(m, dim)
            if s and s not in vals:
                vals.append(s)
        if dim == "element":
            # A description rarely pins the heteroatom *positions* (only their kinds and counts), so this expands
            # into a match-any set that leaves the arrangement free. The other dimensions do not depend on the elements and are left alone.
            for m in members:
                b = _project_dimension_smarts(m, "element")
                for s in _element_arrangement_set(m, systems, b):
                    if s and s not in vals:
                        vals.append(s)
            vals = vals[:64]
        dimsets[dim] = vals
    return {"n_members": len(members), "n_free_slots": len(free_slots),
            "verdict_match_any": verdict, "dimension_sets": dimsets,
            "decompositions": _build_decompositions(mol, named, systems)}


# --------------------------------------------------------------------------- #
#  The 'functional-group framework' of an acyclic molecule
#  ---------------------------------------------------------------------------
#  With no rings the Murcko scaffold is empty and cannot provide a skeleton. In that case the groups a medicinal
#  chemist would consider first become the skeleton: the union of the anchors (heteroatoms and unsaturated
#  carbons) and the shortest paths joining them (the linkers between groups). Purely terminal alkyl or halogen
#  branches are neither anchors nor linkers and drop out by themselves.
#  In other words, Murcko's "ring systems + linkers, branches removed" transposed to functional groups.
#
#  An acyclic molecule is a tree (per component), so the path between two atoms is unique and the union of the
#  anchor-pair paths is exactly the minimum spanning subtree joining the anchors — the functional-group framework.
#
#  The framework is emitted in the same shape as the ring route (scaffold_smiles / scaffold_smarts /
#  dimension_smarts / eval_query) so everything downstream — the toolchain builder, the grader — treats it identically.
#  Group *names* are confirmed by SMARTS matching on the framework itself (not on the example molecule, and only
#  what the SMARTS guarantees), which keeps the deterministic description faithful.
# --------------------------------------------------------------------------- #
_HALOGENS = {9, 17, 35, 53, 85}

# Functional group -> SMARTS, most specific first, so a larger group subsumes the smaller ones. A match is skipped
# when its 'core' (heteroatoms + unsaturated carbons) is already fully covered by a group that was accepted.
_FG_SMARTS_RAW: list[tuple[str, str]] = [
    ("carbamate",            "[NX3][CX3](=[OX1])[OX2]"),
    ("urea",                 "[NX3][CX3](=[OX1])[NX3]"),
    ("thiourea",             "[NX3][CX3](=[SX1])[NX3]"),
    ("carbonate",            "[OX2][CX3](=[OX1])[OX2]"),
    ("guanidine",            "[NX3][CX3](=[NX2])[NX3]"),
    ("carboxylic acid",      "[CX3](=[OX1])[OX2H1]"),
    ("ester",                "[CX3](=[OX1])[OX2][#6]"),
    ("amide",                "[CX3](=[OX1])[NX3]"),
    ("thioamide",            "[CX3](=[SX1])[NX3]"),
    ("sulfonamide",          "[$([#16X4](=[OX1])(=[OX1])[NX3]),$([#16X4+2]([OX1-])([OX1-])[NX3])]"),
    ("sulfone",              "[$([#16X4](=[OX1])(=[OX1])[#6]),$([#16X4+2]([OX1-])([OX1-])[#6])]"),
    ("sulfoxide",            "[$([#16X3](=[OX1])),$([#16X3+][OX1-])]"),
    ("phosphate/phosphonate","[PX4](=[OX1])"),
    ("nitro",                "[$([NX3](=[OX1])=[OX1]),$([NX3+](=[OX1])[OX1-])]"),
    ("nitrile",              "[NX1]#[CX2]"),
    ("isothiocyanate",       "[NX2]=[CX2]=[SX1]"),
    ("isocyanate",           "[NX2]=[CX2]=[OX1]"),
    ("amidine",              "[NX3][CX3]=[NX2]"),
    ("imine",                "[CX3]=[NX2]"),
    ("azo",                  "[NX2]=[NX2]"),
    ("aldehyde",             "[CX3H1]=[OX1]"),
    ("ketone",               "[#6][CX3](=[OX1])[#6]"),
    ("alkyne",               "[CX2]#[CX2]"),
    ("alkene",               "[CX3]=[CX3]"),
    ("quaternary ammonium",  "[NX4+]"),
    # The residual C=O cores not caught by the specific carbonyls above (amide/ester/carbamate/ketone/...),
    # where a stripped branch makes anything more specific unassertable. scaffold_smarts still guarantees the C=O.
    ("carbonyl",             "[CX3]=[OX1]"),
    ("ether",                "[#6][OX2][#6]"),
    ("thioether",            "[#6][#16X2][#6]"),
    ("hydroxyl",             "[#6][OX2H1]"),
    ("thiol",                "[#6][SX2H1]"),
    ("hydrazine",            "[NX3]-[NX3]"),
    # amine excludes amide / imine / sulfonamide / ammonium N, so only the branch amines are caught.
    ("amine", "[NX3;!$([NX3][CX3]=[OX1,SX1,NX2]);!$([NX3]=*);!$([NX3][#16X4,PX4]);"
              "!$([NX4+]);!$([NX3][NX3])]"),
]
_FG_SMARTS = [(nm, Chem.MolFromSmarts(sm)) for nm, sm in _FG_SMARTS_RAW]


def _heavy_adj(mol) -> dict[int, list[int]]:
    """Heavy-atom adjacency list, hydrogens excluded."""
    adj: dict[int, list[int]] = {a.GetIdx(): [] for a in mol.GetAtoms()}
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        adj[i].append(j)
        adj[j].append(i)
    return adj


def _bfs_path(adj: dict, src: int, dst: int) -> list[int]:
    """Shortest path src -> dst as a list of atom indices. Empty when unreachable; on an acyclic molecule it is the unique path."""
    if src == dst:
        return [src]
    prev = {src: -1}
    queue = [src]
    while queue:
        nxt = []
        for u in queue:
            for v in adj[u]:
                if v not in prev:
                    prev[v] = u
                    if v == dst:
                        path = [dst]
                        while path[-1] != src:
                            path.append(prev[path[-1]])
                        return path[::-1]
                    nxt.append(v)
        queue = nxt
    return []


def _is_anchor(atom, include_halogen: bool) -> bool:
    """Is this a functional-group anchor: a heteroatom (N/O/S/P/...) or an unsaturated (double- or triple-bonded) carbon.
    Halogens are excluded by default as terminal substituents, and only included with include_halogen=True when no primary anchor exists."""
    z = atom.GetAtomicNum()
    if z == 1:
        return False
    if z == 6:  # carbon: with a multiple bond it is the core of an unsaturated group (C=O/C=N/C#N/C=C/...)
        return any(b.GetBondType() in (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE)
                   for b in atom.GetBonds())
    if z in _HALOGENS:
        return include_halogen
    return True  # any other heteroatom


def _framework_atoms(mol) -> set[int]:
    """The framework atom set: the anchors union the shortest paths between anchor pairs (the linkers). set() when empty."""
    for include_halogen in (False, True):
        anchors = [a.GetIdx() for a in mol.GetAtoms() if _is_anchor(a, include_halogen)]
        if anchors:
            break
    if not anchors:
        return set()
    adj = _heavy_adj(mol)
    fw: set[int] = set(anchors)
    for i in range(len(anchors)):
        for j in range(i + 1, len(anchors)):
            fw.update(_bfs_path(adj, anchors[i], anchors[j]))
    return fw


def _fg_core_atoms(mol, match: tuple) -> set[int]:
    """Only the 'core' of a match: heteroatoms plus carbons carrying a multiple bond. This is the subsumption test."""
    core = set()
    for idx in match:
        a = mol.GetAtomWithIdx(idx)
        if a.GetAtomicNum() != 6:
            core.add(idx)
        elif any(b.GetBondType() in (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE)
                 for b in a.GetBonds()):
            core.add(idx)
    return core


def detect_functional_groups(mol, atoms: set[int]) -> list[tuple[str, frozenset]]:
    """The functional groups inside *atoms* (the framework) as a list of `(name, frozenset of matched atoms)`, in
    backbone order (by first atom index). The same group occurring several times gets a separate entry each time (two amides, say).

    Only SMARTS matches whose atoms all lie inside the framework are accepted, so the description covers only what
    the framework really guarantees (faithful). Once a larger group is accepted, the smaller groups covering its
    core (the amide / ester / ether inside a carbamate) are skipped."""
    covered: set[int] = set()
    units: list[tuple[int, str, frozenset]] = []
    for name, q in _FG_SMARTS:
        if q is None:
            continue
        for match in mol.GetSubstructMatches(q):
            ms = set(match)
            if not ms <= atoms:
                continue                          # a match leaking outside the framework is unverified -> excluded
            core = _fg_core_atoms(mol, match)
            if core and core <= covered:
                continue                          # subsumed by a larger group already accepted
            covered |= core
            units.append((min(match), name, frozenset(match)))
    units.sort()
    return [(nm, ats) for _, nm, ats in units]


def _fg_labels(units: list[tuple[str, frozenset]]) -> list[str]:
    """The display name of a group. Repeats are distinguished as 'amide #1', 'amide #2'."""
    counts: dict[str, int] = {}
    for nm, _ in units:
        counts[nm] = counts.get(nm, 0) + 1
    seen: dict[str, int] = {}
    out = []
    for nm, _ in units:
        if counts[nm] > 1:
            seen[nm] = seen.get(nm, 0) + 1
            out.append(f"{nm} #{seen[nm]}")
        else:
            out.append(nm)
    return out


_NUMWORD = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
            7: "seven", 8: "eight", 9: "nine", 10: "ten"}


def _num_word(n: int) -> str:
    return _NUMWORD.get(n, str(n))


def _join_plain(items: list[str]) -> str:
    """Join with ', ' and 'and', without articles; used when each item is already in 'the ...' form."""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _linker_phrase(n_carbon: int, n_atom: int) -> str:
    """Linker length as a phrase you can rebuild from: 0 atoms is a direct bond, and an all-carbon linker states its carbon count."""
    if n_atom == 0:
        return "directly bonded"
    if n_carbon == n_atom:                        # a pure carbon chain
        annot = {1: " (methylene)", 2: " (ethylene)", 3: " (propylene)"}.get(n_carbon, "")
        return f"connected through a {_num_word(n_carbon)}-carbon{annot} chain"
    return f"connected through a {_num_word(n_atom)}-atom linker"


def _fg_connections(mol, fw: set[int], units: list[tuple[str, frozenset]]) -> list[dict]:
    """The connections between functional groups as a list of `{fgs:[i,j,...], n_carbon, n_atom}`: the topology needed to rebuild it.

    The framework is read as a reduced graph of functional-group nodes joined by carbon linker stretches. Each
    stretch records the groups it joins (two means a chain, three or more a branch point), and direct bonds (linker 0) are collected too."""
    atom2fg: dict[int, int] = {}
    for i, (_, ats) in enumerate(units):
        for a in ats:
            atom2fg.setdefault(a, i)
    fwset = set(fw)
    adj: dict[int, list[int]] = {a: [] for a in fwset}
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in fwset and j in fwset:
            adj[i].append(j)
            adj[j].append(i)

    # Framework atoms belonging to no group are linkers; group them into connected components (stretches).
    linker = {a for a in fwset if a not in atom2fg}
    edges: list[dict] = []
    visited: set[int] = set()
    for a0 in linker:
        if a0 in visited:
            continue
        comp = {a0}
        stack = [a0]
        visited.add(a0)
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v in linker and v not in visited:
                    visited.add(v)
                    comp.add(v)
                    stack.append(v)
        adj_fgs = sorted({atom2fg[v] for u in comp for v in adj[u] if v in atom2fg})
        if len(adj_fgs) >= 2:
            n_c = sum(1 for x in comp if mol.GetAtomWithIdx(x).GetAtomicNum() == 6)
            edges.append({"fgs": adj_fgs, "n_carbon": n_c, "n_atom": len(comp)})

    # direct bonds: pairs of groups attached with no linker between them
    seen_pairs: set[tuple] = set()
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in atom2fg and j in atom2fg and atom2fg[i] != atom2fg[j]:
            pair = tuple(sorted((atom2fg[i], atom2fg[j])))
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                edges.append({"fgs": list(pair), "n_carbon": 0, "n_atom": 0})
    return edges


def _join_with_articles(names: list[str]) -> str:
    """['amide'] → 'an amide'; ['amide','sulfoxide'] → 'an amide and a sulfoxide';
    Three or more take an Oxford comma, and each item gets an indefinite article."""
    items = [f"{_a_an(n)} {n}" for n in names]
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def build_fg_framework(mol) -> Optional[dict]:
    """Acyclic molecule -> the functional-group framework analysis dict, with the same fields as the ring route. None when there is no framework.

    None (a purely saturated hydrocarbon with no functional-group anchor at all, for instance) leaves the caller on
    its existing 'acyclic (no ring scaffold)' path, with has_scaffold=False."""
    if mol is None:
        return None
    fw = _framework_atoms(mol)
    if not fw:
        return None
    scaffold_smiles = _submol_smiles(mol, fw)
    scaffold_smarts = _atoms_to_smarts(mol, fw)
    if not scaffold_smarts:
        return None

    # dimension SMARTS: apply the same per-dimension projection as the ring route to the framework-only submol.
    keep = set(fw)
    rw = Chem.RWMol(mol)
    for idx in sorted((a.GetIdx() for a in mol.GetAtoms() if a.GetIdx() not in keep),
                      reverse=True):
        rw.RemoveAtom(idx)
    sub = rw.GetMol()
    try:
        Chem.FastFindRings(sub)                   # acyclic -> everything is [R0]
    except Exception:  # noqa: BLE001
        pass
    dim_smarts = scaffold_dimension_smarts(sub)

    units = detect_functional_groups(mol, fw)          # [(name, atoms)]
    labels = _fg_labels(units)
    fg_names = [nm for nm, _ in units]
    edges = _fg_connections(mol, fw, units)

    # heteroatom summary over the whole framework, as the fallback when no group is matched at all
    het: dict[str, int] = {}
    for idx in fw:
        e = _PT.GetElementSymbol(_z(mol.GetAtomWithIdx(idx)))
        if e not in ("C", "H"):
            het[e] = het.get(e, 0) + 1

    if labels:
        topo = "acyclic functional-group framework: " + " + ".join(fg_names)
        head = (f"The scaffold is an acyclic (ring-free) framework comprising "
                f"{_join_with_articles(labels)}")
        # Connectivity you can rebuild from: for each linker, the groups it joins and its carbon count (or a direct bond / branch point).
        clauses = []
        for e in edges:
            if len(e["fgs"]) == 2:
                a, b = e["fgs"]
                clauses.append(f"the {labels[a]} is {_linker_phrase(e['n_carbon'], e['n_atom'])} "
                               f"to the {labels[b]}")
            else:                                  # a branch point where 3+ groups meet at one place
                joined = _join_plain([f"the {labels[k]}" for k in e["fgs"]])
                where = (f"a {_num_word(e['n_carbon'])}-carbon branch point"
                         if e["n_atom"] else "a shared branch atom")
                clauses.append(f"{joined} meet at {where}")
        desc = head + (", in which " + "; ".join(clauses) + "." if clauses else ".")
    else:
        topo = f"acyclic framework ({_heteroatom_phrase(het)})"
        desc = (f"The scaffold is an acyclic (ring-free) framework containing "
                f"{_heteroatom_phrase(het)}, with no ring system.")

    eval_query = {
        "n_members": 1, "n_free_slots": 0,
        "verdict_match_any": [scaffold_smarts],
        "dimension_sets": {d: ([dim_smarts[d]] if dim_smarts.get(d) else [])
                           for d in _DIMENSIONS},
        "decompositions": [],
    }

    return {
        "scaffold_kind": "functional_group",
        "framework_atoms": sorted(fw),
        "scaffold_smiles": scaffold_smiles,
        "scaffold_smarts": scaffold_smarts,
        "dimension_smarts": dim_smarts,
        "eval_query": eval_query,
        "functional_groups": fg_names,
        "fg_connections": [{"fgs": [labels[k] for k in e["fgs"]],
                            "n_carbon": e["n_carbon"], "n_atom": e["n_atom"]}
                           for e in edges],
        "heteroatom_summary": _heteroatom_phrase(het),
        "topology_summary": topo,
        "fg_description": desc,
    }


def analyze_scaffold(smiles: str) -> dict:
    """SMILES -> the scaffold analysis dict shared by the template and the validation. A dict is returned even when parsing fails."""
    base = get_murcko_scaffold(smiles)
    mol, scaf = base["mol"], base["scaffold"]
    result = {
        "smiles": Chem.MolToSmiles(mol) if mol is not None else smiles,
        "parse_ok": mol is not None,
        "has_scaffold": base["has_scaffold"],
        "scaffold_kind": "ring" if base["has_scaffold"] else "none",
        "scaffold_smiles": base["scaffold_smiles"],
        "scaffold_smarts": scaffold_to_smarts(scaf) if base["has_scaffold"] else "",
        "dimension_smarts": scaffold_dimension_smarts(scaf) if base["has_scaffold"] else {},
        "ring_systems": [], "connections": [], "attachment": {},
        "n_ring_systems": 0, "n_rings_total": 0, "topology_summary": "",
        "aromaticity": "", "heteroatom_summary": "", "functional_groups": [],
    }
    if not base["has_scaffold"]:
        # With no rings there is no Murcko framework, so the functional-group framework is tried instead.
        fw = build_fg_framework(mol) if mol is not None else None
        if fw is not None:
            result["has_scaffold"] = True
            result.update(fw)                     # scaffold_kind/_smiles/_smarts/dim/eval_query/…
            return result
        result["topology_summary"] = "acyclic (no ring scaffold)"
        return result

    systems = decompose_ring_systems(scaf)
    named = []
    for s in systems:
        info = name_ring_system(scaf, s)
        named.append({
            "id": s["id"], "name": info["name"], "name_source": info["name_source"],
            "n_rings": s["n_rings"], "internal_topology": s["internal_topology"],
            "aromatic": info["aromatic"], "any_aromatic": info["any_aromatic"],
            "n_aromatic_rings": info["n_aromatic_rings"], "aromaticity_desc": info["aromaticity_desc"],
            "ring_sizes": info["ring_sizes"],
            "hetero_counts": info["hetero_counts"], "hetero_phrase": info["hetero_phrase"],
            "smiles": info["smiles"],
            "n_ring_carbonyls": info.get("n_ring_carbonyls", 0),
            "carbonyl_groups": info.get("carbonyl_groups", []),
            "carbonyl_desc": info.get("carbonyl_desc", ""),
        })
    namemap = {ns["id"]: ns["name"] for ns in named}
    connections = find_connections(scaf, systems)
    attach = find_attachment_points(mol, scaf, base["match"], systems, names=namemap)

    # Annotate each connection with where its two ends attach on their ring systems (locant / relation / label).
    sysmap = {s["id"]: s for s in systems}
    for c in connections:
        for end, sid in (("a_position", c["a"]), ("b_position", c["b"])):
            atom = c["a_atom"] if end == "a_position" else c["b_atom"]
            if sid in sysmap:
                c[end] = _atom_ring_position(scaf, sysmap[sid], atom, namemap.get(sid, ""))

    # Merge the attachment information into each system, for convenience when describing it
    for ns in named:
        a = attach["per_system"].get(ns["id"], {})
        ns["n_attachment_points"] = a.get("n_points", 0)
        ns["attachment_points"] = a.get("points", [])
        if a.get("benzene_relations"):
            ns["substitution_relations"] = a["benzene_relations"]

    # Collect the set of locant integers we asserted, so validation can tell an invented locant from one
    # that FACTS provided; validate_text treats only numbers outside this set as violations.
    asserted = set()
    for ns in named:
        # Digits inside a curated or systematic ring name (the 4 and 3 of 'quinazolin-4(3H)-one', the 1 and 3
        # of '1,3-benzodioxole') are computed facts, not violations.
        for d in re.findall(r"\d{1,2}", ns["name"]):
            asserted.add(int(d))
        # NOTE: substituent attachment locants are no longer added to asserted. Substituents fall away from the
        # scaffold and are checked by no evaluation SMARTS — information like "substituted at 2,5" on a ring
        # system is unverifiable — so it stays out of the description and those numbers are not allowed either.
        # (Where a linker attaches IS verified, because the linker is part of the scaffold, so the connection
        # positions below remain in asserted.)
    for c in connections:
        for end in ("a_position", "b_position"):
            if c.get(end) and c[end].get("locant"):
                asserted.add(c[end]["locant"])

    # Overall aromaticity is tallied per ring, so a partly aromatic fused system is reflected exactly.
    tot_rings = sum(s["n_rings"] for s in systems)
    tot_arom = sum(ns["n_aromatic_rings"] for ns in named)
    if tot_rings and tot_arom == tot_rings:
        aromaticity = "all aromatic"
    elif tot_arom == 0:
        aromaticity = "all aliphatic (saturated)"
    else:
        aromaticity = "mixed aromatic/aliphatic"

    # heteroatom summary over the whole scaffold
    total_het: dict[str, int] = {}
    for ns in named:
        for e, c in ns["hetero_counts"].items():
            total_het[e] = total_het.get(e, 0) + c

    result.update(
        ring_systems=named,
        connections=connections,
        attachment=attach,
        linker_attachment_points=attach.get("linker", []),
        asserted_locants=sorted(asserted),
        n_ring_systems=len(named),
        n_rings_total=sum(s["n_rings"] for s in systems),
        aromaticity=aromaticity,
        heteroatom_summary=_heteroatom_phrase(total_het),
        topology_summary=_topology_summary(named, connections),
        eval_query=build_eval_query(scaf, named, connections, systems),
    )
    return result


def _a_an(word: str) -> str:
    """The indefinite article ('a'/'an') fitting the next word: 'an' before a vowel."""
    return "an" if word[:1].lower() in "aeiou" else "a"


def _topology_summary(named: list[dict], connections: list[dict]) -> str:
    """A one-sentence summary of the ring systems and their connections (deterministic; shared by the template and the validation)."""
    if len(named) == 1:
        s = named[0]
        if s["n_rings"] == 1:
            nm = s["name"]
            return nm if nm.endswith("ring") or "carbocycle" in nm else f"a single {nm} ring"
        nm = s["name"]
        # When the name already contains fusion vocabulary (a composite), drop the parenthesised topology to avoid repeating it.
        topo = s["internal_topology"]
        base = topo.replace("ortho-", "")
        extra = "" if base in nm.lower() or "ring" in nm.lower() else f" ({topo})"
        suffix = "" if nm.lower().endswith(("system", "rings")) else " ring system"
        return f"a single {nm}{suffix}{extra}"
    if not connections:
        names = ", ".join(s["name"] for s in named)
        return f"{len(named)} disconnected ring systems ({names})"
    parts = []
    for c in connections:
        a = named[c["a"]]["name"] if c["a"] < len(named) else f"system{c['a']}"
        b = named[c["b"]]["name"] if c["b"] < len(named) else f"system{c['b']}"
        if c["relation"] == "linker-connected":
            parts.append(f"a {a} connected to a {b} through {_a_an(c['type'])} {c['type']}")
        elif "biaryl" in c["type"]:
            parts.append(f"a {a} in a biaryl linkage with a {b}")
        else:
            parts.append(f"a {a} directly bonded to a {b}")
    return "; ".join(parts)


if __name__ == "__main__":
    import json
    import sys
    smis = sys.argv[1:] or ["c1ccc(Nc2ncnc3ccccc23)cc1"]
    for smi in smis:
        print("=" * 70)
        print("SMILES:", smi)
        print(json.dumps(analyze_scaffold(smi), indent=2, ensure_ascii=False, default=list))
