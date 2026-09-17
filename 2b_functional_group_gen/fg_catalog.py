"""fg_catalog.py  —  the functional-group constraint catalog.

One entry per RDKit ``fr_*`` fragment that the benchmark scorer knows about (61 of
them). Each entry carries everything needed to (a) score a constraint and (b) write
a faithful natural-language description of it:

    fr_key       : RDKit fragment function name          (fr_N_O)
    name         : common name used in answer.fragments  ("hydroxylamine")
    smarts       : the scoring pattern                   ("[N!$(N=O)](-O)-C")
    rdkit_desc   : RDKit's own description               ("Number of hydroxylamine groups")
    composition  : elements / bonds / H / charge / ring facts parsed from the SMARTS
    exclusions   : the recursive `!$(...)` clauses spelled out
    probes       : observed match COUNT on a fixed panel of named molecules

Everything here is **derived**, never asserted: `smarts` comes from
``molkit.utils.fragments.fr_catalog()`` (the single source of truth shared with
the grader), and `probes` are measured by running the pattern.

Why probes matter
-----------------
An fr_* name is routinely narrower or wider than it sounds, and a description
written from the name alone is wrong:

    fr_N_O   "hydroxylamine"  [N!$(N=O)](-O)-C
             NH2OH -> 0 (the N must carry a carbon), CH3-N(CH3)-O-CH3 -> 2
    fr_benzene "benzene ring" c1ccccc1
             naphthalene -> 2, anthracene -> 3
    fr_amide  "amide"        C(=O)-N
             urea -> 2
    fr_lactam "beta lactam"  N1C(=O)CC1
             2-pyrrolidinone (5-ring) -> 0

The probe panel makes that behaviour a measured FACT the describer must respect,
instead of something a model has to guess.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

# Import the unified fragment source of truth from the repo root.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# --------------------------------------------------------------------------- #
#  Where the built description catalog lives
# --------------------------------------------------------------------------- #
#  fg_catalog.json is a build ARTIFACT of build_fg_catalog.py (~450 KB, and a
#  paraphrase bank makes it grow), so it lives beside the rest of the generated
#  training data rather than in the repo. Resolution order:
#      $FG_CATALOG  ->  /data/.../fg_catalog.json  ->  the legacy in-repo copy
#  The last hop keeps an old checkout working; nothing writes there any more.
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_CATALOG_JSON = "data/training_data/fg_catalog.json"
LEGACY_CATALOG_JSON = os.path.join(_HERE, "fg_catalog.json")


def catalog_json_path(path: Optional[str] = None) -> str:
    """Resolve the fg_catalog.json to READ. Explicit *path* always wins."""
    if path:
        return path
    env = os.environ.get("FG_CATALOG", "").strip()
    if env:
        return env
    if not os.path.exists(DATA_CATALOG_JSON) and os.path.exists(LEGACY_CATALOG_JSON):
        return LEGACY_CATALOG_JSON
    return DATA_CATALOG_JSON


def _load_fr_catalog():
    """Get ``fr_catalog`` from ``molkit/utils/fragments.py``.

    Plain ``from molkit.utils.fragments import ...`` executes ``molkit/__init__``,
    which imports the LLM clients and therefore needs ``anthropic`` — a dependency this
    stage has no use for and that the data-generation env (molkit) lacks. So fall
    back to loading the module file directly. It is the same file either way, so the
    grader-shared naming stays single-sourced.
    """
    try:
        from molkit.utils.fragments import fr_catalog as _fc
        return _fc
    except ImportError:
        import importlib.util
        path = os.path.join(_ROOT, "molkit", "utils", "fragments.py")
        spec = importlib.util.spec_from_file_location("_molkit_fragments", path)
        if spec is None or spec.loader is None:
            raise
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.fr_catalog


fr_catalog = _load_fr_catalog()


# --------------------------------------------------------------------------- #
#  Probe panel — named molecules whose match counts pin down what a pattern means
# --------------------------------------------------------------------------- #
#  Chosen for discrimination, not coverage: each entry is either a textbook member
#  of some group or a NEAR MISS that a name-based description would get wrong
#  (naphthalene for "benzene ring", 2-pyrrolidinone for "beta lactam", urea for
#  "amide", carboxylate for "carboxylic acid", ...).
PROBE_PANEL: list[tuple[str, str]] = [
    ("benzene",                 "c1ccccc1"),
    ("naphthalene",             "c1ccc2ccccc2c1"),
    ("anthracene",              "c1ccc2cc3ccccc3cc2c1"),
    ("biphenyl",                "c1ccc(-c2ccccc2)cc1"),
    ("pyridine",                "c1ccncc1"),
    ("quinoline",               "c1ccc2ncccc2c1"),
    ("pyrimidine",              "c1cncnc1"),
    ("pyrrole",                 "c1cc[nH]c1"),
    ("imidazole",               "c1c[nH]cn1"),
    ("furan",                   "c1ccoc1"),
    ("thiophene",               "c1ccsc1"),
    ("oxazole",                 "c1ocnc1"),
    ("thiazole",                "c1scnc1"),
    ("tetrazole",               "c1nn[nH]n1"),
    ("piperidine",              "C1CCNCC1"),
    ("piperazine",              "C1CNCCN1"),
    ("morpholine",              "C1COCCN1"),
    ("cyclohexane",             "C1CCCCC1"),
    ("phenol",                  "Oc1ccccc1"),
    ("2-naphthol",              "Oc1ccc2ccccc2c1"),
    ("4-hydroxypyridine",       "Oc1ccncc1"),
    ("ethanol",                 "CCO"),
    ("tert-butanol",            "CC(C)(C)O"),
    ("acetic acid",             "CC(=O)O"),
    ("acetate anion",           "CC(=O)[O-]"),
    ("benzoic acid",            "OC(=O)c1ccccc1"),
    ("methyl acetate",          "COC(=O)C"),
    ("gamma-butyrolactone",     "O=C1CCCO1"),
    ("acetamide",               "CC(=O)N"),
    ("N-methylacetamide",       "CNC(=O)C"),
    ("urea",                    "NC(=O)N"),
    ("methyl carbamate",        "COC(=O)N"),
    ("2-azetidinone",           "O=C1CCN1"),
    ("2-pyrrolidinone",         "O=C1CCCN1"),
    ("acetone",                 "CC(C)=O"),
    ("acetaldehyde",            "CC=O"),
    ("formic acid",             "OC=O"),
    ("anisole",                 "COc1ccccc1"),
    ("diethyl ether",           "CCOCC"),
    ("acetonitrile",            "CC#N"),
    ("nitrobenzene",            "O=[N+]([O-])c1ccccc1"),
    ("nitrosobenzene",          "O=Nc1ccccc1"),
    ("hydroxylamine",           "NO"),
    ("O-methylhydroxylamine",   "CON"),
    ("N,O-dimethylhydroxylamine", "CNOC"),
    ("N,N,O-trimethylhydroxylamine", "CN(C)OC"),
    ("acetoxime",               "CC(C)=NO"),
    ("benzaldehyde hydrazone",  "NN=Cc1ccccc1"),
    ("methylhydrazine",         "CNN"),
    ("azobenzene",              "c1ccc(N=Nc2ccccc2)cc1"),
    ("aniline",                 "Nc1ccccc1"),
    ("methylamine",             "CN"),
    ("dimethylamine",           "CNC"),
    ("trimethylamine",          "CN(C)C"),
    ("benzenesulfonamide",      "NS(=O)(=O)c1ccccc1"),
    ("dimethyl sulfone",        "CS(C)(=O)=O"),
    ("dimethyl sulfoxide",      "CS(C)=O"),
    ("dimethyl sulfide",        "CSC"),
    ("ethanethiol",             "CCS"),
    ("thiourea",                "NC(=S)N"),
    ("thioacetamide",           "CC(N)=S"),
    ("guanidine",               "N=C(N)N"),
    ("acetamidine",             "CC(N)=N"),
    ("methyl isocyanate",       "CN=C=O"),
    ("methyl isothiocyanate",   "CN=C=S"),
    ("methyl thiocyanate",      "CSC#N"),
    ("chlorobenzene",           "Clc1ccccc1"),
    ("styrene",                 "C=Cc1ccccc1"),
    ("phenylacetylene",         "C#Cc1ccccc1"),
    ("barbituric acid",         "O=C1CC(=O)NC(=O)N1"),
    ("diazepam",                "CN1c2ccc(Cl)cc2C(=Nc2ccccc2)CC1=O"),
    ("1,4-benzodiazepine",      "C1CN=Cc2ccccc2N1"),
    ("methyl N-methylcarbamate", "CNC(=O)OC"),
    ("ethylene oxide",          "C1CO1"),
    ("methyl azide",            "CN=[N+]=[N-]"),
    ("benzenediazonium",        "N#[N+]c1ccccc1"),
    ("tetramethylammonium",     "C[N+](C)(C)C"),
    ("phosphoric acid",         "OP(=O)(O)O"),
    ("trimethyl phosphate",     "COP(=O)(OC)OC"),
    ("1,4-dihydropyridine",     "C1C=CNC=C1"),
    ("toluene",                 "Cc1ccccc1"),
    ("cyclohexene",             "C1=CCCCC1"),
]

_PROBE_MOLS: Optional[list[tuple[str, Chem.Mol]]] = None


def probe_mols() -> list[tuple[str, Chem.Mol]]:
    """Parsed probe panel, built once."""
    global _PROBE_MOLS
    if _PROBE_MOLS is None:
        out = []
        for name, smi in PROBE_PANEL:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                out.append((name, mol))
        _PROBE_MOLS = out
    return _PROBE_MOLS


# --------------------------------------------------------------------------- #
#  SMARTS -> structural facts
# --------------------------------------------------------------------------- #
_BOND_WORD = {
    Chem.BondType.SINGLE: "single", Chem.BondType.DOUBLE: "double",
    Chem.BondType.TRIPLE: "triple", Chem.BondType.AROMATIC: "aromatic",
}


def _atom_facts(atom: Chem.Atom) -> str:
    """One query atom -> a plain phrase stating only what the query constrains."""
    sym = atom.GetSymbol()
    if sym in ("*", ""):
        sym = "any atom"
    bits = []
    if atom.GetIsAromatic():
        bits.append("aromatic")
    n_h = atom.GetNumExplicitHs()
    if n_h:
        bits.append(f"{n_h} explicit H")
    if atom.GetFormalCharge():
        bits.append(f"charge {atom.GetFormalCharge():+d}")
    q = atom.GetSmarts()
    if "X" in q:
        bits.append("fixed connectivity")
    if "R" in q or "r" in q:
        bits.append("ring-membership constrained")
    return f"{sym}" + (f" ({', '.join(bits)})" if bits else "")


def smarts_composition(smarts: str) -> dict:
    """Parse *smarts* into element / bond / ring facts. Empty dict if unparsable."""
    patt = Chem.MolFromSmarts(smarts)
    if patt is None:
        return {}
    # Query mols carry no perceived rings until asked; without this a 4-membered
    # beta-lactam pattern reports "no ring", which is the one fact that separates
    # it from an ordinary lactam.
    try:
        Chem.GetSSSR(patt)
    except Exception:  # noqa: BLE001
        pass
    elements: dict[str, int] = {}
    for a in patt.GetAtoms():
        sym = a.GetSymbol() or "*"
        elements[sym] = elements.get(sym, 0) + 1
    bonds: dict[str, int] = {}
    for b in patt.GetBonds():
        w = _BOND_WORD.get(b.GetBondType(), "unspecified/any")
        bonds[w] = bonds.get(w, 0) + 1
    ri = patt.GetRingInfo()
    ring_sizes = sorted(len(r) for r in ri.AtomRings())

    # Per-atom H / aromaticity / charge requirements. Without these a description
    # cannot tell [O&H1] (a hydroxyl) from a bare O (which an ether also satisfies),
    # or c (aromatic carbon) from C — the difference between fr_Ar_OH and fr_Al_OH.
    hydrogens: dict[str, int] = {}
    aromatic: dict[str, int] = {}
    charges: dict[str, int] = {}
    for a in patt.GetAtoms():
        sym = a.GetSymbol() or "*"
        if a.GetNumExplicitHs():
            hydrogens[sym] = max(hydrogens.get(sym, 0), a.GetNumExplicitHs())
        if a.GetIsAromatic():
            aromatic[sym] = aromatic.get(sym, 0) + 1
        if a.GetFormalCharge():
            charges[sym] = a.GetFormalCharge()

    # Ring vs pendant split: fr_phenol is `[OX2H]-c1ccccc1`, a 6-ring *plus* a
    # hanging oxygen. Without the split the oxygen gets described as a ring member.
    ring_elements: dict[str, int] = {}
    pendant_elements: dict[str, int] = {}
    for a in patt.GetAtoms():
        sym = a.GetSymbol() or "*"
        bucket = ring_elements if ri.NumAtomRings(a.GetIdx()) else pendant_elements
        bucket[sym] = bucket.get(sym, 0) + 1

    return {
        "n_atoms": patt.GetNumAtoms(),
        "elements": elements,
        "bonds": bonds,
        "ring_sizes": ring_sizes,
        "ring_elements": ring_elements,
        "pendant_elements": pendant_elements,
        "hydrogens": hydrogens,
        "aromatic": aromatic,
        "charges": charges,
        "atom_phrases": [_atom_facts(a) for a in patt.GetAtoms()],
    }


def smarts_exclusions(smarts: str) -> list[str]:
    """Spell out the `!$(...)` negative-recursive clauses a pattern carries.

    These are the clauses a name-derived description always misses (fr_N_O's
    ``!$(N=O)``, fr_Al_OH's ``!$(C=O)``), so they are surfaced as first-class facts.
    """
    out, i = [], 0
    while True:
        j = smarts.find("!$(", i)
        if j < 0:
            break
        depth, k = 0, j + 2
        while k < len(smarts):
            if smarts[k] == "(":
                depth += 1
            elif smarts[k] == ")":
                depth -= 1
                if depth == 0:
                    break
            k += 1
        out.append(smarts[j + 3:k])
        i = k + 1
    return out


def probe_counts(smarts: str) -> dict[str, int]:
    """Match COUNT of *smarts* on every probe molecule (0 included)."""
    patt = Chem.MolFromSmarts(smarts)
    if patt is None:
        return {}
    return {name: len(mol.GetSubstructMatches(patt, uniquify=True))
            for name, mol in probe_mols()}


# --------------------------------------------------------------------------- #
#  Catalog
# --------------------------------------------------------------------------- #
_CATALOG: Optional[dict[str, dict]] = None


def build_catalog() -> dict[str, dict]:
    """{fr_key: entry} for all 61 scorer-visible fragments. Built once, cached."""
    global _CATALOG
    if _CATALOG is not None:
        return _CATALOG

    out: dict[str, dict] = {}
    for fr_key, meta in fr_catalog(broadest_only=True).items():
        smarts = meta["smarts"]
        counts = probe_counts(smarts)
        out[fr_key] = {
            "fr_key": fr_key,
            "name": meta["name"],
            "smarts": smarts,
            "rdkit_desc": (meta["description"] or "").strip(),
            "composition": smarts_composition(smarts),
            "exclusions": smarts_exclusions(smarts),
            "probes": counts,
            "probe_positive": {k: v for k, v in counts.items() if v},
            "probe_negative": [k for k, v in counts.items() if not v],
        }
    _CATALOG = out
    return out


def by_name() -> dict[str, dict]:
    """{common name: entry} — the key shape used by ``answer.fragments``."""
    return {e["name"]: e for e in build_catalog().values()}


def count_matches(mol: Chem.Mol, smarts_or_patt) -> int:
    """Count *smarts* in *mol* the way the grader counts fr_*.

    ``uniquify=True`` is what makes this identical to RDKit's ``_CountMatches``;
    verified equal to fr_*() on 1500 corpus molecules across all 61 patterns.
    """
    patt = (smarts_or_patt if isinstance(smarts_or_patt, Chem.Mol)
            else Chem.MolFromSmarts(smarts_or_patt))
    if patt is None:
        return 0
    return len(mol.GetSubstructMatches(patt, uniquify=True))


if __name__ == "__main__":
    cat = build_catalog()
    keys = sys.argv[1:] or ["fr_N_O", "fr_benzene", "fr_amide", "fr_lactam"]
    for k in keys:
        e = cat.get(k) or by_name().get(k)
        if not e:
            print(f"unknown: {k}")
            continue
        print("=" * 72)
        print(f"{e['fr_key']}  \"{e['name']}\"\n  SMARTS     {e['smarts']}\n"
              f"  RDKit      {e['rdkit_desc']}\n  exclusions {e['exclusions']}\n"
              f"  composition {e['composition']['elements']} bonds={e['composition']['bonds']}"
              f" rings={e['composition']['ring_sizes']}")
        pos = ", ".join(f"{n}={c}" for n, c in e["probe_positive"].items())
        print(f"  matches    {pos or '(none in panel)'}")
