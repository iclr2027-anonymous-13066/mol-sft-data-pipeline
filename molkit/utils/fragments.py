"""molkit/utils/fragments.py  —  Fragment / functional-group utilities

Functions
---------
fr_catalog                : {fr_key: {func, description, smarts}} for the selected fr_* set
build_fragment_catalog    : {fr_key: count} for the non-zero fr_* fragments in a mol
analyze_fragments         : {common_name: count} for a SMILES
results_to_text           : human-readable summary of fragment counts
list_available_fragments  : sorted list of all recognized common names

Constants
---------
FR_COMMON_NAMES           : fr_key → common name  (derived, not hand-written)
COMMON_NAME_TO_FR         : common name → fr_key
FR_DISPLAY_NAME_OVERRIDES : fr_keys whose common name is pinned rather than derived
FR_PRETTY_DISPLAY_NAMES   : optional Title-Case labels (cosmetic only)

Naming policy
-------------
Common names are **derived** from RDKit's own fragment descriptions using exactly
the policy the benchmark scorer applies: drop the
narrower member of each overlapping fr_* pair (``broadest_only``), then normalize
the description ("Number of benzene rings" → "benzene ring").

So a name produced here means the same thing it means when an answer is graded —
e.g. ``answer.fragments`` keys in ``molkit.jsonl`` such as "carboxylic acid"
(``fr_COO2``) and "phenol" (``fr_phenol``) resolve identically on both sides.

Do NOT reintroduce a hand-maintained fr_key → name table. It silently drifts from
the grader; the previous one lacked "carboxylic acid" entirely and mapped both
``fr_Ar_OH`` and ``fr_phenol`` onto "phenol".
"""

from __future__ import annotations

import inspect
from typing import Iterable, Optional

from rdkit import Chem
from rdkit.Chem import Fragments


# ---------------------------------------------------------------------------
# All fr_* functions exposed by this RDKit build, with description + SMARTS
# ---------------------------------------------------------------------------

_ALL_FR: dict[str, tuple] = {}


def _default_smarts(func) -> Optional[str]:
    """Recover a fr_* function's query pattern as SMARTS, or None."""
    try:
        pattern = inspect.signature(func).parameters.get("pattern").default
        return Chem.MolToSmarts(pattern)
    except Exception:
        return None


def _all_fragment_functions() -> dict[str, tuple]:
    """{fr_key: (func, description, smarts)} for every fr_* in this RDKit build."""
    global _ALL_FR
    if not _ALL_FR:
        _ALL_FR = {
            name: (fn, fn.__doc__ or "", _default_smarts(fn))
            for name, fn in ((n, getattr(Fragments, n)) for n in dir(Fragments))
            if name.startswith("fr_") and callable(fn)
        }
    return _ALL_FR


# ---------------------------------------------------------------------------
# broadest_only dedup — keep the general fr_*, drop the narrower overlapping one
# ---------------------------------------------------------------------------

def _dedup_broadest_only(available: Iterable[str]) -> list[str]:
    """Return *available* minus the fr_keys that a broader fr_key already covers."""
    avail = set(available)
    drop: set[str] = set()

    def _maybe(*names):
        drop.update(n for n in names if n in avail)

    # Carboxylic acids: fr_COO2 subsumes the aliphatic/aromatic/plain variants.
    if "fr_COO2" in avail:
        _maybe("fr_COO", "fr_Al_COO", "fr_Ar_COO")
    _maybe("fr_Al_OH_noTert")                             # hydroxyl: drop 'noTert'
    _maybe("fr_priamide")                                 # amide: drop primary-only
    _maybe("fr_prisulfonamd")                             # sulfonamide: drop primary-only
    _maybe("fr_ketone_Topliss")                           # ketone: drop Topliss variant
    _maybe("fr_nitro_arom", "fr_nitro_arom_nonortho")     # nitro: drop aromatic variants
    _maybe("fr_C_O_noCOO")                                # carbonyl O: drop COOH-excluding
    _maybe("fr_lactone")                                  # lactone ⊂ ester
    _maybe("fr_alkyl_halide")                             # halide specialization
    _maybe("fr_imide", "fr_Imide")                        # imide is a special amide motif
    # Metabolic-site / too-specific descriptors: not functional groups a task can target.
    _maybe("fr_ArN", "fr_NH0", "fr_NH1", "fr_NH2", "fr_bicyclic", "fr_para_hydroxylation",
           "fr_phenol_noOrthoHbond", "fr_unbrch_alkane", "fr_HOCCN",
           "fr_Ndealkylation1", "fr_Ndealkylation2")

    return [n for n in sorted(avail) if n not in drop]


# ---------------------------------------------------------------------------
# description → common name
# ---------------------------------------------------------------------------

_IRREGULAR_PLURALS = {
    "groups": "group", "rings": "ring", "amines": "amine", "amides": "amide",
    "nitrogens": "nitrogen", "halogens": "halogen", "ethers": "ether",
    "esters": "ester", "acids": "acid", "ketones": "ketone", "aldehydes": "aldehyde",
    "coo2": "carboxylic acid",
}


def _singularize_tail(phrase: str) -> str:
    words = phrase.split()
    if not words:
        return phrase
    w = words[-1].lower()
    if w in _IRREGULAR_PLURALS:
        w = _IRREGULAR_PLURALS[w]
    elif w.endswith("ies"):
        w = w[:-3] + "y"
    elif w.endswith(("ses", "xes", "zes", "ches", "shes")):
        w = w[:-2]
    elif w.endswith("s") and not w.endswith("ss"):
        w = w[:-1]
    words[-1] = w
    return " ".join(words)


def _normalize_label_from_desc(desc: str) -> str:
    """"Number of benzene rings" → "benzene ring"."""
    if "coo2" in (desc or "").lower():
        return "carboxylic acid"

    core = (desc or "").strip()
    if core.lower().startswith("number of "):
        core = core[10:].strip()
    core = core.strip().lower()
    if core.endswith(" groups"):
        core = core[:-7].rstrip()
    elif core.endswith(" group"):
        core = core[:-6].rstrip()
    return _singularize_tail(core)


# fr_keys whose RDKit description does not normalize to a usable name.
FR_DISPLAY_NAME_OVERRIDES: dict[str, str] = {
    "fr_alkyl_carbamate": "alkyl carbamate",
    "fr_benzodiazepine":  "benzodiazepine",
    "fr_methoxy":         "methoxy",
    "fr_nitroso":         "nitroso",
}

# Cosmetic Title-Case labels — display only, never used for matching.
FR_PRETTY_DISPLAY_NAMES: dict[str, str] = {
    "fr_Al_OH":     "Aliphatic Hydroxyl",
    "fr_Ar_OH":     "Aromatic Hydroxyl",
    "fr_phenol":    "Phenol",
    "fr_benzene":   "Benzene Ring",
    "fr_amide":     "Amide",
    "fr_ester":     "Ester",
    "fr_halogen":   "Halogen",
    "fr_nitrile":   "Nitrile",
    "fr_nitro":     "Nitro",
    "fr_ketone":    "Ketone",
    "fr_aldehyde":  "Aldehyde",
    "fr_ether":     "Ether",
    "fr_sulfone":   "Sulfone",
    "fr_SH":        "Thiol",
    "fr_COO2":      "Carboxylic Acid",
}


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

_CATALOG_CACHE: dict[bool, dict[str, dict]] = {}


def fr_catalog(broadest_only: bool = True) -> dict[str, dict]:
    """Return ``{fr_key: {"func", "description", "smarts", "name"}}``.

    With *broadest_only* (the default, and what the benchmark scorer uses) the
    narrower member of each overlapping fr_* pair is dropped.
    """
    if broadest_only in _CATALOG_CACHE:
        return _CATALOG_CACHE[broadest_only]

    all_funcs = _all_fragment_functions()
    keys = (_dedup_broadest_only(all_funcs) if broadest_only
            else sorted(all_funcs))

    catalog = {}
    for key in keys:
        fn, desc, smarts = all_funcs[key]
        catalog[key] = {
            "func": fn,
            "description": desc,
            "smarts": smarts,
            "name": FR_DISPLAY_NAME_OVERRIDES.get(key) or _normalize_label_from_desc(desc),
        }
    _CATALOG_CACHE[broadest_only] = catalog
    return catalog


def _name_maps() -> tuple[dict[str, str], dict[str, str]]:
    catalog = fr_catalog(broadest_only=True)
    fr_to_name = {k: meta["name"] for k, meta in catalog.items()}
    return fr_to_name, {v: k for k, v in fr_to_name.items()}


FR_COMMON_NAMES, COMMON_NAME_TO_FR = _name_maps()


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def build_fragment_catalog(mol: Chem.Mol, broadest_only: bool = True) -> dict[str, int]:
    """Return {fr_key: count} for every non-zero fr_* fragment in *mol*."""
    counts: dict[str, int] = {}
    for key, meta in fr_catalog(broadest_only).items():
        try:
            n = int(meta["func"](mol))
        except Exception:
            continue
        if n > 0:
            counts[key] = n
    return counts


def analyze_fragments(
    smiles: str,
    use_display_overrides: bool = False,
    broadest_only: bool = True,
) -> Optional[dict[str, int]]:
    """Return {common_name: count} for a molecule given its SMILES string.

    Parameters
    ----------
    smiles:
        SMILES string of the molecule.
    use_display_overrides:
        If True, apply ``FR_PRETTY_DISPLAY_NAMES`` for prettier labels. Cosmetic
        only — leave it False when the result is compared against benchmark names.
    broadest_only:
        Keep the deduplicated fr_* set the benchmark scorer uses. Default True.

    Returns
    -------
    Dict of common_name → count, or None if the SMILES is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    catalog = fr_catalog(broadest_only)
    out: dict[str, int] = {}
    for key, count in build_fragment_catalog(mol, broadest_only).items():
        name = catalog[key]["name"]
        if use_display_overrides:
            name = FR_PRETTY_DISPLAY_NAMES.get(key, name)
        out[name] = count
    return out


def results_to_text(fragment_counts: dict[str, int]) -> str:
    """Format a fragment-count dict as a human-readable string.

    Example
    -------
    >>> results_to_text({'benzene ring': 2, 'amide': 1})
    '2 benzene ring groups, 1 amide group'
    """
    if not fragment_counts:
        return "No recognized functional groups."
    parts = [
        f"{count} {name} group{'s' if count > 1 else ''}"
        for name, count in sorted(fragment_counts.items())
    ]
    return ", ".join(parts)


def list_available_fragments() -> list[str]:
    """Return a sorted list of all recognized common fragment names."""
    return sorted(FR_COMMON_NAMES.values())
