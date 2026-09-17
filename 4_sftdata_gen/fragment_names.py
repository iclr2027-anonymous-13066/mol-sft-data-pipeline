"""Deterministic, RDKit-verified naming of edit fragments and anchor sites.

The edit-reasoning prompts used to hand the LLM only the raw ``from_smiles`` /
``to_smiles`` strings, so it invented the chemistry vocabulary itself — calling a
pyrrolidine a "piperazine", a primary carboxamide a "urea", a
(tetrahydropyran-4-yl)methyl a "morpholine", or a substitution at an acyclic CH2
"a position on the piperidine ring". Those names then propagated into the
round's intent line and from there into every later round that quotes it.

This module computes the vocabulary instead, so the prompt can hand the model a
closed set of verified names to choose from:

* :func:`describe_fragment` — what a ``[*:n]``-tagged fragment actually is
  (curated substituent name when one exists, plus the functional groups and ring
  systems RDKit can confirm).
* :func:`describe_anchor` — what the edit site actually is (element, aromatic or
  not, and the named ring system it belongs to, if any).

Ring naming reuses the curated tables in ``2_substructure_gen``
(``RING_SMILES_NAMES`` / ``name_ring_system``) so a ring is named the same way
here as in the substructure descriptions the prompts are built from.
"""

from __future__ import annotations

import importlib.util
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ring naming, reused from the stage-2 scaffold analyzer.
# It lives in a package whose name starts with a digit ("2_substructure_gen"),
# so it cannot be imported with a plain import statement — load it by path.
# ---------------------------------------------------------------------------
_ANALYZER = None


def _analyzer():
    global _ANALYZER
    if _ANALYZER is not None:
        return _ANALYZER or None
    path = Path(__file__).resolve().parents[1] / "2_substructure_gen" / "scaffold_analyzer.py"
    try:
        spec = importlib.util.spec_from_file_location("_sft2_scaffold_analyzer", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)          # type: ignore[union-attr]
        _ANALYZER = mod
    except Exception as e:                     # pragma: no cover - optional dep
        logger.warning("scaffold_analyzer unavailable (%s); ring names degrade "
                       "to ring size/aromaticity only", e)
        _ANALYZER = False
        return None
    return _ANALYZER


# ---------------------------------------------------------------------------
# Curated substituent names, keyed by the canonical SMILES of the fragment with
# its attachment point written as a bare ``*``. Keying on the canonical form
# (rather than the raw string) keeps the attachment position significant:
# ``[*]OC`` (methoxy) and ``[*]CO`` (hydroxymethyl) are different keys.
# ---------------------------------------------------------------------------
_RAW_SUBSTITUENTS: list[tuple[str, str]] = [
    ("[*]C", "methyl"), ("[*]CC", "ethyl"), ("[*]CCC", "n-propyl"),
    ("[*]C(C)C", "isopropyl"), ("[*]CCCC", "n-butyl"), ("[*]C(C)(C)C", "tert-butyl"),
    ("[*]CC(C)C", "isobutyl"), ("[*]C1CC1", "cyclopropyl"),
    ("[*]C1CCC1", "cyclobutyl"), ("[*]C1CCCC1", "cyclopentyl"),
    ("[*]C1CCCCC1", "cyclohexyl"), ("[*]CC1CC1", "cyclopropylmethyl"),
    ("[*]C=C", "vinyl"), ("[*]C#C", "ethynyl"), ("[*]CC#C", "propargyl"),
    ("[*]O", "hydroxyl"), ("[*]CO", "hydroxymethyl"), ("[*]CCO", "2-hydroxyethyl"),
    ("[*]OC", "methoxy"), ("[*]OCC", "ethoxy"), ("[*]OC(C)C", "isopropoxy"),
    ("[*]OC(F)(F)F", "trifluoromethoxy"), ("[*]C(F)(F)F", "trifluoromethyl"),
    ("[*]C(F)F", "difluoromethyl"), ("[*]CF", "fluoromethyl"),
    ("[*]F", "fluoro"), ("[*]Cl", "chloro"), ("[*]Br", "bromo"), ("[*]I", "iodo"),
    ("[*]N", "primary amine"), ("[*]NC", "methylamino"), ("[*]N(C)C", "dimethylamino"),
    ("[*]NCC", "ethylamino"), ("[*]CN", "aminomethyl"), ("[*]CCN", "2-aminoethyl"),
    ("[*]C#N", "nitrile"), ("[*]CC#N", "cyanomethyl"),
    ("[*][N+](=O)[O-]", "nitro"),
    ("[*]C=O", "aldehyde"), ("[*]C(C)=O", "acetyl"),
    ("[*]C(N)=O", "primary carboxamide"), ("[*]C(=O)NC", "N-methylcarboxamide"),
    ("[*]NC(C)=O", "acetamido"), ("[*]NC=O", "formamido"),
    ("[*]C(=O)O", "carboxylic acid"), ("[*]C(=O)OC", "methyl ester"),
    ("[*]C(=O)OCC", "ethyl ester"), ("[*]OC(C)=O", "acetoxy"),
    ("[*]S", "thiol"), ("[*]SC", "methylthio"), ("[*]SCC", "ethylthio"),
    ("[*]S(N)(=O)=O", "primary sulfonamide"),
    ("[*]S(C)(=O)=O", "methylsulfonyl"), ("[*]S(=O)C", "methylsulfinyl"),
    ("[*]NS(C)(=O)=O", "methanesulfonamido"),
    ("[*]c1ccccc1", "phenyl"), ("[*]Cc1ccccc1", "benzyl"),
    ("[*]Oc1ccccc1", "phenoxy"), ("[*]Nc1ccccc1", "anilino"),
    ("[*]N1CCCC1", "pyrrolidin-1-yl"), ("[*]N1CCCCC1", "piperidin-1-yl"),
    ("[*]N1CCOCC1", "morpholin-4-yl"), ("[*]N1CCNCC1", "piperazin-1-yl"),
    ("[*]N1CCN(C)CC1", "4-methylpiperazin-1-yl"),
    ("[*]C1CCOCC1", "tetrahydropyran-4-yl"),
    ("[*]CC1CCOCC1", "(tetrahydropyran-4-yl)methyl"),
    ("[*]C(=O)N1CCOCC1", "morpholine-4-carbonyl"),
    ("[*]C(=O)N1CCCC1", "pyrrolidine-1-carbonyl"),
    ("[*][H]", "hydrogen"),
    # Two-attachment-point linkers. Without these the CH2 of a bridge gets
    # called a "methyl" (measured: the residual naming error after the first
    # fix), so name the bridge for what it is.
    ("[*]C[*]", "methylene bridge (–CH2–)"),
    ("[*]CC[*]", "ethylene bridge (–CH2CH2–)"),
    ("[*]C(C)[*]", "methyl-substituted methine bridge (–CH(CH3)–)"),
    ("[*]O[*]", "ether oxygen (–O–)"),
    ("[*]N[*]", "secondary amine nitrogen (–NH–)"),
    ("[*]N(C)[*]", "N-methyl amine nitrogen (–N(CH3)–)"),
    ("[*]S[*]", "thioether sulfur (–S–)"),
    ("[*]C(=O)[*]", "carbonyl bridge (–C(=O)–)"),
    ("[*]C(=O)N[*]", "amide bridge (–C(=O)NH–)"),
    ("[*]NC(=O)[*]", "amide bridge (–NH–C(=O)–)"),
    ("[*]C(=O)O[*]", "ester bridge (–C(=O)O–)"),
    ("[*]S(=O)(=O)[*]", "sulfonyl bridge (–S(=O)2–)"),
    ("[*]S(=O)(=O)N[*]", "sulfonamide bridge (–S(=O)2NH–)"),
    ("[*]CS(=O)(=O)[*]", "methylene-sulfonyl bridge (–CH2S(=O)2–)"),
    ("[*]NS(=O)(=O)[*]", "sulfonamide bridge (–NH–S(=O)2–)"),
    ("[*]c1ccccc1[*]", "ortho-disubstituted benzene ring"),
    ("[*]c1ccc([*])cc1", "para-disubstituted benzene ring"),
    ("[*]c1cccc([*])c1", "meta-disubstituted benzene ring"),
]


def _canon_star(smi: str) -> Optional[str]:
    """Canonical SMILES with every attachment point reduced to a bare ``*``."""
    if not smi:
        return None
    plain = re.sub(r"\[\*(?::\d+)?\]", "[*]", smi)
    m = Chem.MolFromSmiles(plain, sanitize=False)
    if m is None:
        return None
    try:
        Chem.SanitizeMol(m, Chem.SanitizeFlags.SANITIZE_ALL ^
                         Chem.SanitizeFlags.SANITIZE_KEKULIZE)
    except Exception:
        pass
    for a in m.GetAtoms():
        if a.GetAtomicNum() == 0:
            a.SetAtomMapNum(0)
            a.SetIsotope(0)
    try:
        return Chem.MolToSmiles(m)
    except Exception:
        return None


SUBSTITUENT_NAMES: dict[str, str] = {}
for _smi, _name in _RAW_SUBSTITUENTS:
    _k = _canon_star(_smi)
    if _k and _k not in SUBSTITUENT_NAMES:
        SUBSTITUENT_NAMES[_k] = _name


# ---------------------------------------------------------------------------
# Functional groups RDKit can confirm on the fragment itself. Ordered
# most-specific first; only the first match of an overlapping family is kept
# (a sulfonamide must not also be reported as an amine).
# ---------------------------------------------------------------------------
_GROUP_SMARTS: list[tuple[str, str, str]] = [
    # (name, SMARTS, family)
    ("urea",                "[NX3][CX3](=[OX1])[NX3]",              "carbonylN"),
    ("carbamate",           "[NX3][CX3](=[OX1])[OX2][#6]",          "carbonylN"),
    ("sulfonamide",         "[SX4](=[OX1])(=[OX1])[NX3]",           "sulfonyl"),
    ("sulfone",             "[#6][SX4](=[OX1])(=[OX1])[#6]",        "sulfonyl"),
    ("sulfoxide",           "[#6][SX3](=[OX1])[#6]",                "sulfonyl"),
    ("carboxylic acid",     "[CX3](=[OX1])[OX2H1]",                 "carbonylO"),
    ("ester",               "[CX3](=[OX1])[OX2][#6]",               "carbonylO"),
    ("carboxamide",         "[CX3](=[OX1])[NX3]",                   "carbonylN"),
    ("ketone",              "[#6][CX3](=[OX1])[#6]",                "carbonylC"),
    ("aldehyde",            "[CX3H1](=[OX1])[#6]",                  "carbonylC"),
    ("nitrile",             "[CX2]#[NX1]",                          "nitrile"),
    ("nitro",               "[$([NX3](=[OX1])=[OX1]),$([NX3+](=[OX1])[OX1-])]", "nitro"),
    ("trifluoromethyl",     "[CX4](F)(F)F",                         "halide"),
    ("aryl halide",         "[c][F,Cl,Br,I]",                       "halide"),
    ("alkyl halide",        "[CX4][F,Cl,Br,I]",                     "halide"),
    ("phenol",              "[c][OX2H1]",                           "hydroxyl"),
    ("alcohol",             "[CX4][OX2H1]",                         "hydroxyl"),
    ("ether",               "[#6][OX2][#6]",                        "ether"),
    ("thioether",           "[#6][SX2][#6]",                        "ether"),
    ("thiol",               "[SX2H1]",                              "thiol"),
    ("guanidine",           "[NX3][CX3](=[NX2])[NX3]",              "amineN"),
    ("amidine",             "[NX3][CX3]=[NX2]",                     "amineN"),
    ("tertiary amine",      "[NX3;H0;!$(N[#6]=[O,N,S]);!$(N[SX4])]([#6])([#6])[#6]", "amineN"),
    ("secondary amine",     "[NX3;H1;!$(N[#6]=[O,N,S]);!$(N[SX4])]([#6])[#6]",       "amineN"),
    ("primary amine",       "[NX3;H2;!$(N[#6]=[O,N,S]);!$(N[SX4])][#6]",             "amineN"),
]
_GROUP_PATTERNS = [(n, Chem.MolFromSmarts(s), f) for n, s, f in _GROUP_SMARTS]


def _parse_fragment(smi: str):
    if not smi:
        return None
    m = Chem.MolFromSmiles(smi, sanitize=False)
    if m is None:
        return None
    try:
        Chem.SanitizeMol(m, Chem.SanitizeFlags.SANITIZE_ALL ^
                         Chem.SanitizeFlags.SANITIZE_KEKULIZE)
    except Exception:
        pass
    return m


_CARBONYL = Chem.MolFromSmarts("[CX3]=[OX1]")


def _groups_in(mol) -> list[str]:
    """Functional groups present, most specific first, one per family."""
    out: list[str] = []
    seen_family: set[str] = set()
    for name, patt, family in _GROUP_PATTERNS:
        if patt is None or family in seen_family:
            continue
        if mol.HasSubstructMatch(patt):
            out.append(name)
            seen_family.add(family)
    # A carbonyl whose substituents are attachment points matches none of the
    # specific patterns above (they need real neighbours), so report it plainly.
    if (not seen_family & {"carbonylN", "carbonylO", "carbonylC"}
            and _CARBONYL is not None and mol.HasSubstructMatch(_CARBONYL)):
        out.append("carbonyl (C=O)")
    return out


def _ring_names(mol) -> list[str]:
    """Named ring systems in *mol*, e.g. ['pyrrolidine (saturated)']."""
    ana = _analyzer()
    ri = mol.GetRingInfo()
    if ri.NumRings() == 0:
        return []
    names: list[str] = []
    if ana is not None:
        try:
            for system in ana.decompose_ring_systems(mol):
                info = ana.name_ring_system(mol, system)
                name = info.get("name") or "unnamed ring system"
                arom = info.get("aromaticity_desc") or ""
                names.append(f"{name} ({arom})" if arom else name)
            return names
        except Exception as e:
            logger.debug("ring naming failed (%s); falling back", e)
    for ring in ri.AtomRings():
        arom = all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)
        het = sorted({mol.GetAtomWithIdx(i).GetSymbol()
                      for i in ring if mol.GetAtomWithIdx(i).GetSymbol() != "C"})
        kind = "aromatic" if arom else "non-aromatic"
        names.append(f"{len(ring)}-membered {kind} ring"
                     + (f" containing {', '.join(het)}" if het else ""))
    return names


def _attachment_note(mol) -> str:
    """Through which atom the fragment connects to the parent molecule."""
    parts: list[str] = []
    for a in mol.GetAtoms():
        if a.GetAtomicNum() != 0:
            continue
        for nb in a.GetNeighbors():
            kind = "aromatic" if nb.GetIsAromatic() else "aliphatic"
            ring = " (ring atom)" if nb.IsInRing() else ""
            parts.append(f"connects through its {kind} {nb.GetSymbol()}{ring}")
    return "; ".join(dict.fromkeys(parts))


def _linker_name(mol, smi: str) -> Optional[str]:
    """Name a single-atom-linked ring fragment, e.g. ``[*:1]Cc1ccco1`` →
    "methylene-linked furan".

    Without this the model reaches for "methyl-furan" / "1-methyl-pyrazole",
    which puts the methyl in the wrong place — the CH2 is the linker to the
    parent, not a substituent on the ring.
    """
    dummies = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    if len(dummies) != 1:
        return None
    nb = [n for n in dummies[0].GetNeighbors()]
    if len(nb) != 1:
        return None
    link = nb[0]
    if link.GetIsAromatic() or link.IsInRing():
        return None
    heavy_nbrs = [n for n in link.GetNeighbors() if n.GetAtomicNum() != 0]
    if len(heavy_nbrs) != 1 or not heavy_nbrs[0].IsInRing():
        return None
    rings = _ring_names(mol)
    if len(rings) != 1:
        return None
    link_desc = {"C": "methylene", "O": "oxygen", "N": "amino", "S": "thioether"}.get(
        link.GetSymbol())
    if link_desc is None:
        return None
    return f"{link_desc}-linked {rings[0].split(' (')[0]}"


def describe_fragment(smi: Optional[str]) -> str:
    """One-line, RDKit-verified description of an edit fragment.

    ``[*:1]`` (a bare attachment point) means no group at all — the edit replaces
    a hydrogen. Returned text is meant to be pasted into a prompt as the ONLY
    naming vocabulary the model may use for this fragment.
    """
    if not smi:
        return "(none)"
    mol = _parse_fragment(smi)
    if mol is None:
        return f"{smi} — (could not be parsed; refer to it by this SMILES)"

    heavy = [a for a in mol.GetAtoms() if a.GetAtomicNum() != 0]
    if not heavy:
        return (f"{smi} — NO group: a bare attachment point. An edit FROM this "
                f"means a hydrogen is being substituted (nothing is removed); an "
                f"edit TO this means the group is deleted and replaced by hydrogen.")

    key = _canon_star(smi)
    curated = SUBSTITUENT_NAMES.get(key or "") or _linker_name(mol, smi)
    facts: list[str] = []
    if curated:
        facts.append(f"name: {curated}")
    rings = _ring_names(mol)
    facts.append("rings: " + (", ".join(rings) if rings else "none"))
    groups = _groups_in(mol)
    if groups:
        facts.append("groups: " + ", ".join(groups))
    att = _attachment_note(mol)
    if att:
        facts.append(att)
    facts.append(f"{len(heavy)} heavy atom" + ("s" if len(heavy) != 1 else ""))
    if not curated:
        facts.append("no curated common name — refer to it by its SMILES or by "
                     "the ring/group names listed here")
    return f"{smi} — " + "; ".join(facts)


def short_name(smi: Optional[str]) -> str:
    """A compact verified label for a fragment: the curated name when there is
    one, else its ring/group facts. Used for the candidate list, where a full
    fact line per candidate would bury the numbers."""
    if not smi:
        return "(none)"
    mol = _parse_fragment(smi)
    if mol is None:
        return smi
    if not [a for a in mol.GetAtoms() if a.GetAtomicNum() != 0]:
        return "nothing (a bare attachment point — substitutes a hydrogen)"
    curated = SUBSTITUENT_NAMES.get(_canon_star(smi) or "") or _linker_name(mol, smi)
    if curated:
        return curated
    bits = []
    rings = _ring_names(mol)
    if rings:
        bits.append(" + ".join(rings))
    groups = _groups_in(mol)
    if groups:
        bits.append(", ".join(groups))
    return "; ".join(bits) if bits else "no common name (refer to it by SMILES)"


def describe_anchor(mol_smiles: Optional[str], anchors: Optional[dict]) -> str:
    """Describe each anchor atom of *mol_smiles*: element, aromaticity, ring.

    The anchor is the site the edit is pinned to, so a claim like "on the
    piperidine ring" is only true if the anchor atom really is in that ring.
    """
    if not mol_smiles or not anchors:
        return "(no anchors given)"
    mol = Chem.MolFromSmiles(mol_smiles)
    if mol is None:
        return "(current molecule could not be parsed)"
    ana = _analyzer()
    systems = []
    if ana is not None:
        try:
            systems = [(s, ana.name_ring_system(mol, s))
                       for s in ana.decompose_ring_systems(mol)]
        except Exception:
            systems = []
    out: list[str] = []
    for label, idx in sorted(anchors.items(), key=lambda kv: str(kv[0])):
        if not isinstance(idx, int) or idx >= mol.GetNumAtoms():
            out.append(f"anchor {label}: atom {idx} (out of range)")
            continue
        a = mol.GetAtomWithIdx(idx)
        # "aromatic CH1" / "aliphatic NH0" is machine notation, and the reasoning
        # quotes it verbatim (measured: 445 rounds said things like "substituting a
        # hydrogen on aliphatic NH1"). Render the hydrogen count the way a chemist
        # writes it, so the facts stay exact and the prose stays readable.
        n_h = a.GetTotalNumHs()
        sym = a.GetSymbol()
        if n_h == 0:
            env = f"{sym} bearing no hydrogens (fully substituted)"
        elif n_h == 1:
            env = f"{sym}–H"
        else:
            env = f"{sym}H{n_h}"
        bits = [f"{'aromatic' if a.GetIsAromatic() else 'aliphatic'} {env}"]
        if a.IsInRing():
            named = None
            for system, info in systems:
                if idx in system["atoms"]:
                    named = info.get("name")
                    break
            bits.append(f"in the {named} ring system" if named else "in a ring")
        else:
            bits.append("NOT in any ring (acyclic position)")
        nbrs = ", ".join(sorted(
            f"{nb.GetSymbol()}{'(aromatic)' if nb.GetIsAromatic() else ''}"
            for nb in a.GetNeighbors()))
        if nbrs:
            bits.append(f"neighbours: {nbrs}")
        out.append(f"anchor {label} = atom {idx}: " + ", ".join(bits))
    return "\n".join(out)


def _fmt_num(v) -> str:
    if isinstance(v, float) and not float(v).is_integer():
        return f"{v:.3f}"
    return str(int(v)) if isinstance(v, (int, float)) else str(v)


def _ordinal(rank: int, n_ranks: int, n_cands: int) -> str:
    """Tightness label for dense ``rank`` among ``n_cands`` candidates.

    ``n_ranks`` is the number of DISTINCT spreads (so the widest rank can be named
    "widest"), while the denominator quoted to the model is the candidate count —
    "tightest of 2" when four candidates were returned only invited confusion.
    """
    if rank == 1:
        return f"tightest of {n_cands}"
    if rank == n_ranks:
        return f"widest of {n_cands}"
    suffix = "nd" if rank == 2 else "rd" if rank == 3 else "th"
    return f"{rank}{suffix}-tightest of {n_cands}"


def _std_of(c, name: str):
    """The mmpdb Δ std for property ``name`` on candidate ``c``, or None."""
    if not isinstance(c, dict):
        return None
    d = (c.get("delta") or {}).get(name)
    if not isinstance(d, dict):
        return None
    try:
        return float(d.get("std"))
    except (TypeError, ValueError):
        return None


def _spread_ranks(candidates, targets) -> dict:
    """Per property, map candidate index -> a computed tightness label.

    The model is bad at comparing two ± numbers in prose (measured: 26% of its
    "tighter than" claims were numerically inverted), so the comparison is done
    here and handed over as a verdict it only has to repeat.  Ties — very common,
    since many mmpdb rules have ± 0.0 on integer-valued properties — are labelled
    as ties so the reasoning stops inventing a difference that is not there.
    """
    out: dict = {}
    for name in (targets or {}):
        stds = {i: _std_of(c, name) for i, c in enumerate(candidates, 1)}
        vals = sorted({v for v in stds.values() if v is not None})
        if not vals:
            continue
        n_cands = sum(1 for v in stds.values() if v is not None)
        if len(vals) == 1:
            label = f"all tie at ± {vals[0]:g} — no discrimination"
            out[name] = {i: label for i, v in stds.items() if v is not None}
            continue
        rank = {v: k for k, v in enumerate(vals, 1)}
        per: dict = {}
        for i, v in stds.items():
            if v is None:
                continue
            tied = sum(1 for w in stds.values() if w == v) > 1
            label = _ordinal(rank[v], len(vals), n_cands)
            per[i] = f"tied {label}" if tied else label
        out[name] = per
    return out


def _binding_props(props, targets) -> list:
    """Constrained properties whose CURRENT value is outside the target range."""
    out = []
    for name, rng in (targets or {}).items():
        cur = (props or {}).get(name)
        if cur is None:
            continue
        lo = rng[0] if isinstance(rng, (list, tuple)) and len(rng) > 0 else None
        hi = rng[1] if isinstance(rng, (list, tuple)) and len(rng) > 1 else None
        try:
            if (lo is not None and float(cur) < float(lo)) or \
               (hi is not None and float(cur) > float(hi)):
                out.append(name)
        except (TypeError, ValueError):
            continue
    return out


def _same_anchors(a, b) -> bool:
    """Anchor maps equal, compared as ``{str(label): int(index)}``."""
    def norm(x):
        out = {}
        for k, v in (x or {}).items():
            try:
                out[str(k)] = int(v)
            except (TypeError, ValueError):
                out[str(k)] = v
        return out
    return norm(a) == norm(b)


def picked_index(candidates, from_smiles, to_smiles, anchors=None):
    """1-based row of the committed candidate, or ``None``.

    ``suggest_edits`` returns the SAME rule applied at different sites as separate
    candidates — measured on 387 rounds, 1.6% of candidate lists carry a duplicate
    (from, to) pair and in 1.0% the committed pair spans more than one row. Their mmpdb
    Δ is identical (it depends on the fragment pair, not the site), so every column of
    the block reads the same and only ``anchors`` tells them apart. Matching on the pair
    alone made this module disagree with itself: `picked_number` took the first row and
    `landing_safety` the last, so the prompt could name #2 while the table's arrow
    pointed at #4. Every caller goes through here now.
    """
    rows = [(i, c) for i, c in enumerate(candidates or [], 1)
            if isinstance(c, dict)
            and c.get("from_smiles") == from_smiles
            and c.get("to_smiles") == to_smiles]
    if not rows:
        return None
    if anchors:
        for i, c in rows:
            if _same_anchors(c.get("anchors"), anchors):
                return i
    return rows[0][0]


def spread_verdict(candidates, props: Optional[dict], targets: Optional[dict],
                   from_smiles=None, to_smiles=None, anchors=None) -> str:
    """Authoritative, pre-computed spread comparison for the out-of-range properties.

    States for each binding property which candidate has the tightest ± spread and
    where the SELECTED candidate sits, so the reasoning never has to compare two
    ± numbers itself (which it gets wrong a quarter of the time).
    """
    if not candidates:
        return "(no candidates)"
    ranks = _spread_ranks(candidates, targets or {})
    binding = _binding_props(props, targets) or list(targets or {})
    picked = picked_index(candidates, from_smiles, to_smiles, anchors)
    lines: list[str] = []
    for name in binding[:3]:
        per = ranks.get(name)
        if not per:
            continue
        if any("tie at" in v for v in per.values()):
            lines.append(f"{name}: every candidate has the same spread "
                         f"({next(iter(per.values())).split('—')[0].strip()}), so the "
                         f"spread cannot decide this one — compare the Δ instead")
            continue
        best = min(per, key=lambda i: (_std_of(candidates[i - 1], name), i))
        bits = [f"{name}: tightest spread is #{best} "
                f"(± {_std_of(candidates[best - 1], name):g})"]
        if picked is not None and picked in per:
            mine = _std_of(candidates[picked - 1], name)
            bits.append(f"the selected #{picked} is {per[picked]} (± {mine:g})")
        lines.append("; ".join(bits))
    return "\n".join(lines) or "(no comparable spreads)"


# ── numeric guard on the model's own spread comparisons ─────────────────────
# Even with the comparison pre-computed the model occasionally writes its own
# ("±0.14, tighter than #2's ±0.05"), and gets the ordering backwards. Such a
# sentence teaches wrong arithmetic, so it is detected and the reasoning is
# regenerated with the error quoted back.
_TIGHT_W = r"tight(?:er|est)?|narrow(?:er|est)?|smaller|more reliable|safer"
_WIDE_W = r"wid(?:er|est)?|larger|broader|looser|noisier|less reliable"
_CMP_TIGHT = re.compile(
    rf"±\s*(\d+(?:\.\d+)?)[^.]{{0,90}}?\b(?:{_TIGHT_W})\b[^.]{{0,60}}?"
    rf"\bthan\b[^.]{{0,60}}?±\s*(\d+(?:\.\d+)?)", re.I)
_CMP_WIDE = re.compile(
    rf"±\s*(\d+(?:\.\d+)?)[^.]{{0,90}}?\b(?:{_WIDE_W})\b[^.]{{0,60}}?"
    rf"\bthan\b[^.]{{0,60}}?±\s*(\d+(?:\.\d+)?)", re.I)


# ── the size of one edit ────────────────────────────────────────────────────
# `_struct_desc` reads a rule's two fragment SMILES and returns what the block's
# `+heavy` / `ΔMW` columns are built from. `landing_safety` runs twice per round
# (once for the prompt, once for the guard) and a chain revisits the same fragments,
# so both memoise on the fragment string.


@lru_cache(maxsize=16384)
def _capped(smiles: str):
    """The fragment as a real molecule, its ``[*:n]`` attachment points capped with H.

    Capping rather than deleting keeps the ring / H-count descriptors meaning what
    they say, and makes a pure attach (``from_smiles == "[*:1]"``) come out all-zero,
    which is correct — it removes nothing.
    """
    m = Chem.MolFromSmiles(smiles or "")
    if m is None:
        return None
    ed = Chem.RWMol(m)
    for atom in ed.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetAtomicNum(1)
            atom.SetAtomMapNum(0)
    try:
        mol = ed.GetMol()
        Chem.SanitizeMol(mol)
    except Exception:  # noqa: BLE001 - an unsanitisable side contributes nothing
        return None
    return mol


@lru_cache(maxsize=16384)
def _struct_desc(smiles: str) -> tuple:
    """``(heavy atoms, MW, halogen count)`` for one side of a rule."""
    from rdkit.Chem import Descriptors
    mol = _capped(smiles)
    if mol is None:
        return (0, 0.0, 0)
    return (mol.GetNumHeavyAtoms(), float(Descriptors.MolWt(mol)),
            sum(1 for a in mol.GetAtoms()
                if a.GetSymbol() in ("F", "Cl", "Br", "I")))


# column key -> the header the block prints it under, so a verdict can never name a
# column the table does not have.
_LABEL = {"rank": "rank", "safe": "safe", "n_in": "IN after",
          "gap": "gap closed", "over": "overshoot",
          "dheavy": "+heavy", "dmw": "ΔMW", "break": "breaks"}
# lower is better for these; higher for the rest
_LOWER_BETTER = frozenset({"rank", "overshoot", "+heavy", "ΔMW", "breaks"})


def _rank_on(label, cell, i):
    """``"strict"`` / ``"tied"`` / ``None`` — how candidate ``i`` stands on ``label``.

    Measured over 120 rounds: the decision order the prompt states picks the same
    candidate the search committed to only 33% of the time, and in half the rest there
    is no column at all where the committed candidate strictly leads. Which is the
    finding the branch study already reported — the winning move is always among the
    top few but at an essentially uniform rank — so the block must be able to say
    "it merely matches the leader here" rather than manufacture a lead.
    """
    def val(txt):
        txt = txt.split("/")[0]            # "3/5" -> the count, the base is shared
        try:
            return float(txt.rstrip("%").lstrip("+"))
        except ValueError:
            return None
    vals = {k: val(c[label]) for k, c in cell.items()}
    if any(x is None for x in vals.values()) or len(set(vals.values())) < 2:
        return None
    best = (min if label in _LOWER_BETTER else max)(vals.values())
    if vals[i] != best:
        return None
    return "strict" if sum(1 for x in vals.values() if x == best) == 1 else "tied"


def _box(rng):
    """``(lo, hi)`` as floats or None, from a target range in any of its shapes."""
    lo = rng[0] if isinstance(rng, (list, tuple)) and len(rng) > 0 else None
    hi = rng[1] if isinstance(rng, (list, tuple)) and len(rng) > 1 else None
    try:
        lo = None if lo is None else float(lo)
    except (TypeError, ValueError):
        lo = None
    try:
        hi = None if hi is None else float(hi)
    except (TypeError, ValueError):
        hi = None
    return lo, hi


def _gap_share(props, targets) -> float:
    """Total out-of-box distance, each property in its own box widths.

    The same scale-free quantity `c_gap_frac_closed` is a ratio of, reused here so the
    round-context block can say whether the LAST edit actually reduced it.
    """
    tot = 0.0
    for name, rng in (targets or {}).items():
        cur = (props or {}).get(name)
        if cur is None:
            continue
        lo, hi = _box(rng)
        try:
            cur = float(cur)
        except (TypeError, ValueError):
            continue
        width = (hi - lo) if (lo is not None and hi is not None and hi > lo) else 1.0
        tot += max(0.0, (lo - cur) if lo is not None and cur < lo
                   else (cur - hi) if hi is not None and cur > hi else 0.0) / width
    return tot


def round_context(props, targets, candidates=None, mol_smiles=None,
                  prev_props=None, last_pred=None) -> str:
    """What is true of THIS ROUND, before any candidate is considered.

    Every line here is identical for all four candidates, so none of it can pick one —
    it sets how far to trust the numbers below, which is a different job and the reason
    it gets its own block. (In the listwise ranker these are the "A" columns; a
    per-set softmax cancels them exactly, and they only earn their keep by re-weighting
    a per-candidate column. The prose equivalent is posture, not choice.)

    The four quantities are the state / history features that came out positive on BOTH
    held-out sets averaged over depth:

    st_n_props           how many constraints are in play, and how many are still out.
    st_room_hi_min       the tightest CEILING left — the property with the least room
                         below its upper bound. It is what an aggressive edit breaks
                         first, so it says how much overshoot this round can afford.
    h_gap_improved_last  did the previous edit actually reduce the total box distance?
                         When it did not, this round is a correction, not a continuation.
    h_pred_error_last    how far the previous edit's predicted Δ landed from the
                         measured one — the tool's calibration on THIS molecule. A large
                         error is the reason to hedge harder here, and it is the only
                         number in the prompt that says the predictions can be wrong by
                         a specific amount rather than in principle.

    The edit-site line is `site_aromatic`. It is included because it averaged the
    largest positive delta of any feature — but measured over 400 real candidate sets
    it is IDENTICAL for every candidate 83% of the time, so it is stated here, as
    context, and the line itself says whether it separates. Only when it does may the
    prose argue from it.
    """
    props, targets = props or {}, targets or {}
    out = []

    # --- st_n_props / how many are out --------------------------------------
    known = [k for k in targets if props.get(k) is not None]
    viol = []
    for name in known:
        lo, hi = _box(targets[name])
        try:
            cur = float(props[name])
        except (TypeError, ValueError):
            continue
        if (lo is not None and cur < lo) or (hi is not None and cur > hi):
            viol.append(name)
    if known:
        out.append(f"constraints: {len(known)} in play, {len(viol)} still out"
                   + (f" ({', '.join(viol[:6])})" if viol else ""))

    # --- st_room_hi_min: the tightest ceiling -------------------------------
    room = []
    for name in known:
        lo, hi = _box(targets[name])
        if hi is None:
            continue
        try:
            room.append((float(hi) - float(props[name]), name, float(hi)))
        except (TypeError, ValueError):
            continue
    if room:
        r, name, hi = min(room)
        if r < 0:
            out.append(f"tightest ceiling: {name} is already {abs(r):.3f} ABOVE its "
                       f"{hi:g} limit — anything that raises it further is wasted")
        else:
            out.append(f"tightest ceiling: {name} has {r:.3f} left below {hi:g}; that is "
                       f"the least room of any constraint, so it is what an aggressive "
                       f"edit breaks first")

    # --- h_gap_improved_last / h_pred_error_last ----------------------------
    if prev_props:
        was, now = _gap_share(prev_props, targets), _gap_share(props, targets)
        if was > 1e-9 or now > 1e-9:
            moved = "FELL" if now < was - 1e-6 else (
                "did NOT fall" if now > was - 1e-6 else "held")
            out.append(f"last edit: the total box distance {moved}, "
                       f"{was:.3f} → {now:.3f}"
                       + ("" if moved == "FELL" else
                          " — treat this round as a correction, and say so"))
    if last_pred and prev_props:
        worst = None
        for name, avg in last_pred.items():
            try:
                pred = float(avg)
                real = float(props[name]) - float(prev_props[name])
            except (TypeError, ValueError, KeyError):
                continue
            err = abs(pred - real)
            if worst is None or err > worst[3]:
                worst = (name, pred, real, err)
        if worst is not None:
            name, pred, real, err = worst
            if err <= 1e-6:
                out.append(f"last prediction: {name} was predicted {pred:+.3f} and "
                           f"measured {real:+.3f} — exact, the tool is calibrated here")
            else:
                out.append(f"last prediction: {name} was predicted {pred:+.3f} but "
                           f"measured {real:+.3f} — off by {err:.3f}. Every Δ below is "
                           f"the same kind of estimate, so hedge by at least that much")

    # --- site_aromatic ------------------------------------------------------
    site = _site_line(mol_smiles, candidates)
    if site:
        out.append(site)
    return "\n".join("- " + x for x in out) or "(no context)"


def _site_line(mol_smiles, candidates) -> str:
    """Where the candidates attach, and whether that separates them.

    `site_aromatic` averaged the largest positive held-out delta of the whole feature
    set, and it is also the feature the prompt used to forbid arguing from — because on
    83% of real candidate sets (measured, 400 sets) every candidate attaches at the same
    kind of atom, which makes "I chose it because the site is aromatic" a claim about
    nothing. Both are true, so the line states which case this round is.
    """
    if not mol_smiles or not candidates:
        return ""
    mol = Chem.MolFromSmiles(mol_smiles)
    if mol is None:
        return ""
    kinds, per = {}, []
    for i, c in enumerate(candidates, 1):
        if not isinstance(c, dict):
            continue
        idx = []
        for v in (c.get("anchors") or {}).values():
            try:
                k = int(v)
            except (TypeError, ValueError):
                continue
            if 0 <= k < mol.GetNumAtoms():
                idx.append(k)
        if not idx:
            continue
        tag = tuple(sorted(
            ("aromatic " if mol.GetAtomWithIdx(k).GetIsAromatic() else
             "ring " if mol.GetAtomWithIdx(k).IsInRing() else "chain ")
            + mol.GetAtomWithIdx(k).GetSymbol() for k in idx))
        per.append((i, tag))
        kinds.setdefault(tag, []).append(i)
    if not per:
        return ""
    if len(kinds) == 1:
        what = ", ".join(next(iter(kinds)))
        return (f"edit site: every candidate attaches at the same {what} — the site "
                f"cannot tell them apart, so do not argue from it (name it only to say "
                f"WHAT the transformation is)")
    parts = [", ".join("#" + str(i) for i in v) + " at " + ", ".join(k)
             for k, v in kinds.items()]
    return ("edit site: the candidates attach at DIFFERENT places — "
            + "; ".join(parts) + ". Here the site is a real reason, so use it")


def landing_safety(candidates, props, targets, from_smiles=None, to_smiles=None,
                   mol_smiles=None, anchors=None) -> str:
    """Pre-computed decision table + the verdict, for the reasoning to quote.

    Same contract as ``spread_verdict``: the arithmetic happens here, the prose only
    reports it. Every column is a per-candidate feature whose held-out delta came out
    positive on BOTH test sets averaged over depth — the columns that did not clear
    that bar were removed, because a block that carries more than the prose can spend
    words on just crowds out the ones that matter.

    rank        the tool's own order (#1 is its first choice). Last resort by design,
                but it is a real signal — nothing else orders the candidates when the
                measured columns tie.
    safe        constraints that end up INSIDE the box with the nearer edge at least one
                predicted std away. The one column that also beat a conservative
                significance bound on both held-out sets at depth 4-5.
    gap closed  share of the current box distance the edit removes. The absolute
                distance does not transfer between molecules of different size; the
                share does.
    overshoot   how far past the FAR edge of its box the edit carries an axis it was
                supposed to be fixing — the failure a distance-reducing score hides,
                and the reason "closes 100% of the gap" is not automatically good.
    +heavy, ΔMW the size of the edit. Decisive whenever heavy_atoms or MW has an exact
                or tight target, and the tie-break when nothing else separates
                (winning paths run coarse-to-fine: the largest edit first, then
                progressively smaller).
    breaks      constraints satisfied NOW that the edit pushes out. Not a score — an
                elimination, and the first step of the order.

    The verdict lines keep the prose honest. Left to itself the generator claims one
    candidate is better even when every candidate scores the same — measured on 5
    rounds x 11 features, ties were never once reported as ties. So when a column does
    not separate the candidates this block says so in words, lists on one line every
    column that is identical across them, and — because the stated order picks the
    committed candidate only a third of the time — runs the order to the end and emits
    a NOTE when it lands somewhere else.
    """
    if not candidates:
        return "(no candidates)"
    props, targets = props or {}, targets or {}
    picked = picked_index(candidates, from_smiles, to_smiles, anchors)
    rows = []
    for i, c in enumerate(candidates, 1):      # 1-based: `spread_verdict` and
        if not isinstance(c, dict):            # `format_candidate_options` number the
            continue                           # candidates from 1, and a prompt that
                                               # numbers the same candidate two ways gets
                                               # cited two ways ("tightest is #1, so take
                                               # #0" was generated verbatim)
        delta = c.get("delta") or {}
        safe = broke = n_in = 0
        gap_now = gap_post = 0.0
        over = None                # (property, amount past the far edge)
        thin = []                  # (property, room, std) — inside, but room < its std
        for name, rng in targets.items():
            cur = props.get(name)
            if cur is None:
                continue
            lo, hi = _box(rng)
            d = delta.get(name) or {}
            try:
                avg = float(d.get("avg") or 0.0)
                std = float(d.get("std") or 0.0)
                cur = float(cur)
            except (TypeError, ValueError):
                continue
            new = cur + avg
            was = ((lo is None or cur >= lo) and (hi is None or cur <= hi))
            inside = ((lo is None or new >= lo) and (hi is None or new <= hi))
            room = min([v for v in ((new - lo) if lo is not None else None,
                                    (hi - new) if hi is not None else None)
                        if v is not None], default=0.0)
            if inside:
                n_in += 1
                if room >= std:
                    safe += 1
                elif std > 0:
                    thin.append((name, room, std))
            if was and not inside:
                broke += 1
            # overshoot: the axis was OUT and the edit carried it past the FAR edge
            if not was:
                if lo is not None and cur < lo and hi is not None and new > hi:
                    amt = new - hi
                    if over is None or amt > over[1]:
                        over = (name, amt, hi)
                elif hi is not None and cur > hi and lo is not None and new < lo:
                    amt = lo - new
                    if over is None or amt > over[1]:
                        over = (name, amt, lo)
            width = (hi - lo) if (lo is not None and hi is not None and hi > lo) else 1.0
            gap_now += max(0.0, (lo - cur) if lo is not None and cur < lo
                           else (cur - hi) if hi is not None and cur > hi else 0.0) / width
            gap_post += max(0.0, (lo - new) if lo is not None and new < lo
                            else (new - hi) if hi is not None and new > hi else 0.0) / width
        fs, ts = c.get("from_smiles") or "", c.get("to_smiles") or ""
        h_from, mw_from, _ = _struct_desc(fs)
        h_to, mw_to, _ = _struct_desc(ts)
        dh = (delta.get("heavy_atoms") or {}).get("avg")
        rows.append({"i": i, "rank": i, "safe": safe, "n_in": n_in,
                     "break": broke, "thin": thin,
                     "gap": ((gap_now - gap_post) / gap_now) if gap_now > 1e-9 else None,
                     "over": over,
                     # the rule's own atom / mass count, not the mmpdb Δ: it is exact,
                     # while the Δ is a prediction that can come back fractional
                     "dheavy": (h_to - h_from) if (fs or ts) else (
                         None if dh is None else int(round(float(dh)))),
                     "dmw": (mw_to - mw_from) if (fs or ts) else None})
    if not rows:
        return "(no comparable candidates)"

    n_props = sum(1 for k in targets if props.get(k) is not None)

    # Every column is rendered ONCE, into `cell`, and both the table and the
    # "identical for every candidate" verdict read from there. Comparing the raw
    # values instead would let the verdict contradict the table it sits under —
    # 0.9341 and 0.9337 both print "93%" but are not equal.
    cell = {r["i"]: {
        "rank": f"{r['rank']}",
        "safe": f"{r['safe']}/{n_props}",
        "IN after": f"{r['n_in']}/{n_props}",
        "gap closed": "n/a" if r["gap"] is None else f"{r['gap'] * 100:.0f}%",
        "overshoot": "none" if r["over"] is None else
                     f"{r['over'][0]} past {r['over'][2]:g} by {r['over'][1]:.3f}",
        "+heavy": "?" if r["dheavy"] is None else f"{r['dheavy']:+d}",
        "ΔMW": "?" if r["dmw"] is None else f"{r['dmw']:+.2f}",
        "breaks": f"{r['break']}",
    } for r in rows}

    out = [f"{'cand':<6}{'rank':>6}{'safe':>8}{'IN after':>10}{'gap closed':>12}"
           f"{'+heavy':>8}{'ΔMW':>10}{'breaks':>8}   overshoot"]
    for r in rows:
        c = cell[r["i"]]
        mark = "   <-- the candidate to select" if r["i"] == picked else ""
        out.append(f"#{r['i']:<5}{c['rank']:>6}{c['safe']:>8}{c['IN after']:>10}"
                   f"{c['gap closed']:>12}{c['+heavy']:>8}{c['ΔMW']:>10}"
                   f"{c['breaks']:>8}   {c['overshoot']}{mark}")

    # ── verdicts — each says either who wins or that the column is silent ────
    v = []
    breaks = [r for r in rows if r["break"] == 0]
    if len(breaks) == len(rows):
        v.append("breakage: no candidate breaks a satisfied constraint — this criterion "
                 "does NOT separate them")
    elif breaks:
        v.append("breakage: only " + ", ".join("#" + str(r["i"]) for r in breaks)
                 + (" breaks" if len(breaks) == 1 else " break")
                 + " nothing; the rest break "
                 + ", ".join(f"#{r['i']}:{r['break']}" for r in rows if r["break"]))
    else:
        v.append("breakage: every candidate breaks at least one satisfied constraint ("
                 + ", ".join(f"#{r['i']}:{r['break']}" for r in rows) + ")")
    gaps = [c["gap closed"] for c in cell.values()]
    if len(set(gaps)) > 1:
        best = max(rows, key=lambda r: (r["gap"] if r["gap"] is not None else -1))
        v.append(f"gap closed: most is #{best['i']} at "
                 f"{cell[best['i']]['gap closed']}")
    safes = [r["safe"] for r in rows]
    if len(set(safes)) == 1:
        v.append(f"safe landings: every candidate lands {safes[0]}/{n_props} — this "
                 f"criterion does NOT separate them, say so and decide on something else")
    else:
        b = max(safes)
        tied = [r["i"] for r in rows if r["safe"] == b]
        v.append(f"safe landings: most is {b}/{n_props}"
                 + (f" (#{tied[0]})" if len(tied) == 1
                    else f", tied between {', '.join('#' + str(t) for t in tied)}"))
    bad_over = [r for r in rows if r["over"] is not None]
    if bad_over:
        v.append("overshoot: " + "; ".join(
            f"#{r['i']} carries {r['over'][0]} past {r['over'][2]:g} by "
            f"{r['over'][1]:.3f}" for r in bad_over)
            + " — a gap it closes on paper it also flies past, so say that plainly")
    dhs = [r["dheavy"] for r in rows if r["dheavy"] is not None]
    if dhs and len(set(dhs)) > 1:
        sm = min(rows, key=lambda r: (r["dheavy"] if r["dheavy"] is not None else 1 << 20))
        v.append(f"edit size: smallest is #{sm['i']} at {sm['dheavy']:+d} heavy atoms "
                 f"({cell[sm['i']]['ΔMW']} MW) — use this only when the criteria above "
                 f"do not separate")

    same = [label for label in ("rank", "safe", "IN after", "gap closed",
                                "overshoot", "+heavy", "ΔMW", "breaks")
            if len({c[label] for c in cell.values()}) == 1]
    if same:
        v.append("identical for every candidate, so it cannot be a reason: "
                 + ", ".join(same))

    # Where the ORDER lands, and whether that is the candidate actually taken.
    #
    # The order is the one the prompt states. Running it here rather than leaving it
    # to the prose is what stops the two most common inventions: claiming the picked
    # candidate leads on a column it loses (measured — "chosen because it is the
    # least uncertain" written about the candidate with the WIDEST ±), and silently
    # skipping the criterion that actually decided. When the order and the selection
    # disagree — which happens two rounds in three, the search does not follow this
    # order — the block says so, names the criterion, and requires the concession.
    if picked is not None:
        alive = list(rows)
        step = None
        nb = [r for r in alive if r["break"] == 0]
        if nb and len(nb) < len(alive):
            alive = nb
            if picked not in [r["i"] for r in alive]:
                step = ("breakage", [r["i"] for r in alive])
        for label, col, want_max in () if step else (
                ("gap closed", "gap", True), ("safe landings", "safe", True),
                ("edit size", "dheavy", False), ("added mass", "dmw", False),
                ("the tool's own rank", "rank", False)):
            live = [r for r in alive if r[col] is not None]
            if len(live) < 2 or len({cell[r["i"]][_LABEL[col]] for r in live}) < 2:
                continue
            best = (max if want_max else min)(r[col] for r in live)
            keep = [r for r in live if r[col] == best]
            if len(keep) == len(live):
                continue
            alive = keep
            if picked not in [r["i"] for r in alive]:
                step = (label, [r["i"] for r in alive])
                break
            if len(alive) == 1:
                break
        if step is not None:
            crit, leaders = step
            prefer = ("safe", "gap closed", "+heavy", "ΔMW", "rank")
            axis = next((x for x in prefer
                         if _rank_on(x, cell, picked) == "strict"), None)
            tied_axis = None if axis else next(
                (x for x in prefer if _rank_on(x, cell, picked) == "tied"), None)
            v.append(f"NOTE running the order above lands on "
                     + ", ".join("#" + str(x) for x in leaders)
                     + f" — it is {crit} that separates them — but the candidate to "
                     f"select is #{picked}. Sentence 2 has to concede that, in your own "
                     f"wording, before it gives any reason for #{picked}"
                     + (f", and give the axis on which it is still worth trying "
                        f"({axis} {cell[picked][axis]})" if axis else
                        f" and that on {tied_axis} it only equals the leader "
                        f"({cell[picked][tied_axis]}) — then treat it as the next thing "
                        f"to TEST and give the condition for undoing it, in your own "
                        f"words" if tied_axis else
                        ", and that no column favours it — take it as the next thing to "
                        "TEST rather than the better option, and give the condition for "
                        "undoing it")
                     + ".")
        else:
            pr = rows[[r["i"] for r in rows].index(picked)]
            if pr["safe"] < max(safes) or pr["break"] > 0:
                v.append(f"NOTE the selected #{picked} is NOT the best on these counts "
                         f"({pr['safe']}/{n_props} safe, {pr['break']} broken) — "
                         f"sentence 2 must give the axis on which it still wins, and "
                         f"must not claim it leads on the counts")
    return "\n".join(out) + "\n" + "\n".join("- " + x for x in v)


def aromatic_h_note(seed_smiles: Optional[str]) -> str:
    """Which aromatic ring hetero atoms of the scaffold carry a hydrogen.

    A committed SMARTS writes a ring nitrogen as a bare ``[#7]``, which matches both
    ``n`` and ``[nH]`` — the pattern simply does not carry hydrogen counts. So
    "transcribe the SMARTS into SMILES" is not a mechanical operation whenever the
    ring needs an N-H: measured over the 998 committed patterns, a purely mechanical
    transcription leaves 18.9% unkekulisable and turns another 0.8% into a different
    tautomer. Deciding where the H goes is chemistry, and the seed reasoning has to
    say so rather than claiming a straight read-off.

    Returns "" when no aromatic hetero atom bears a hydrogen (the transcription then
    really is mechanical).
    """
    if not seed_smiles:
        return ""
    mol = Chem.MolFromSmiles(seed_smiles)
    if mol is None:
        return ""
    systems = []
    ana = _analyzer()
    if ana is not None:
        try:
            systems = [(s, ana.name_ring_system(mol, s))
                       for s in ana.decompose_ring_systems(mol)]
        except Exception:
            systems = []
    bits: list[str] = []
    seen: set = set()
    for a in mol.GetAtoms():
        if not (a.GetIsAromatic() and a.GetAtomicNum() in (7, 8, 16)
                and a.GetTotalNumHs() > 0):
            continue
        sym = {7: "[nH]", 8: "[oH]", 16: "[sH]"}[a.GetAtomicNum()]
        ring = None
        for system, info in systems:
            if a.GetIdx() in system["atoms"]:
                ring = info.get("name")
                break
        label = f"the {ring} ring {a.GetSymbol()}" if ring else f"a ring {a.GetSymbol()}"
        key = (label, sym)
        if key in seen:
            continue
        seen.add(key)
        bits.append(f"{label} carries a hydrogen — write it {sym}")
    if not bits:
        return ""
    return ("; ".join(bits)
            + ". A bare lowercase n/o/s there gives a SMILES that cannot be kekulised.")


# ── ring locants ───────────────────────────────────────────────────────────
# Measured on 7,289 seed segments: of 559 locant-bearing ring names the seed
# reasoning wrote, 35 (6.3%) were wrong — mostly 1,2,3- against 1,2,4-triazole, and
# a 1,2,4-triazine called "1,3,5-". A seed error is the expensive kind: it is copied
# into that round's intent line and quoted by every later round. So
# the locants are computed and handed to the prompt, and the prose is checked
# against them the same way every other number here is.
_RING_CLASS = {                       # (size, N, O, S) -> (class, {locants: name})
    (5, 3, 0, 0): ("triazole", {(1, 2, 3): "1,2,3-triazole", (1, 2, 4): "1,2,4-triazole"}),
    (5, 2, 1, 0): ("oxadiazole", {(1, 2, 4): "1,2,4-oxadiazole",
                                  (1, 3, 4): "1,3,4-oxadiazole",
                                  (1, 2, 5): "1,2,5-oxadiazole"}),
    (5, 2, 0, 1): ("thiadiazole", {(1, 2, 4): "1,2,4-thiadiazole",
                                   (1, 3, 4): "1,3,4-thiadiazole",
                                   (1, 2, 5): "1,2,5-thiadiazole"}),
    (5, 2, 0, 0): ("diazole", {(1, 2): "pyrazole", (1, 3): "imidazole"}),
    (5, 1, 1, 0): ("azole", {(1, 2): "isoxazole", (1, 3): "oxazole"}),
    (5, 1, 0, 1): ("azole", {(1, 2): "isothiazole", (1, 3): "thiazole"}),
    (5, 4, 0, 0): ("tetrazole", {(1, 2, 3, 4): "tetrazole"}),
    (6, 3, 0, 0): ("triazine", {(1, 2, 3): "1,2,3-triazine", (1, 2, 4): "1,2,4-triazine",
                                (1, 3, 5): "1,3,5-triazine"}),
    (6, 2, 0, 0): ("diazine", {(1, 2): "pyridazine", (1, 3): "pyrimidine",
                               (1, 4): "pyrazine"}),
}
_LOCANT_NAMES = sorted(
    {n for _c, d in _RING_CLASS.values() for n in d.values() if "," in n or n[0].islower()},
    key=len, reverse=True)
_LOCANT_RE = re.compile("|".join(re.escape(n) for n in _LOCANT_NAMES), re.I)


def _ring_locants(mol) -> list:
    """``[(name, class)]`` for every aromatic hetero ring RDKit can name by locant."""
    out = []
    ri = mol.GetRingInfo()
    for ring in ri.AtomRings():
        if not all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            continue
        # fused systems get a fusion name of their own (quinoxaline, not "pyrazine"),
        # so naming their component ring here would push the prose the wrong way
        if any(ri.NumAtomRings(i) > 1 for i in ring):
            continue
        # walk the ring in connectivity order so the positions mean something
        adj = {i: [n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors()
                   if n.GetIdx() in ring] for i in ring}
        order, seen, cur = [], set(), ring[0]
        while len(order) < len(ring):
            order.append(cur)
            seen.add(cur)
            nxt = [x for x in adj[cur] if x not in seen]
            if not nxt:
                break
            cur = nxt[0]
        if len(order) != len(ring):
            continue
        nums = [mol.GetAtomWithIdx(i).GetAtomicNum() for i in order]
        key = (len(ring), nums.count(7), nums.count(8), nums.count(16))
        entry = _RING_CLASS.get(key)
        if entry is None:
            continue
        het = [k for k, z in enumerate(nums) if z in (7, 8, 16)]
        # the locant set is the lexicographically smallest over rotations/reflections
        best = None
        n = len(ring)
        for shift in range(n):
            for direc in (1, -1):
                pos = sorted(((direc * (k - shift)) % n) + 1 for k in het)
                if pos[0] != 1:
                    continue
                if best is None or pos < best:
                    best = pos
        name = entry[1].get(tuple(best)) if best else None
        if name:
            out.append((name, entry[0]))
    return out


def ring_locant_note(seed_smiles) -> str:
    """The VERIFIED locant names of the seed's hetero rings, for the prompt to quote."""
    if not seed_smiles:
        return ""
    mol = Chem.MolFromSmiles(seed_smiles)
    if mol is None:
        return ""
    seen, bits = set(), []
    for name, cls in _ring_locants(mol):
        if name in seen:
            continue
        seen.add(name)
        bits.append(f"the {cls} ring is a {name}")
    if not bits:
        return ""
    return ("; ".join(bits)
            + ". Use these names exactly — a locant guessed from the ring type is wrong "
              "about a third of the time.")


def ring_locant_errors(text: str, seed_smiles) -> list:
    """Locant-bearing ring names in *text* that the seed molecule contradicts."""
    if not text or not seed_smiles:
        return []
    mol = Chem.MolFromSmiles(seed_smiles)
    if mol is None:
        return []
    have = {n for n, _c in _ring_locants(mol)}
    classes = {c for _n, c in _ring_locants(mol)}
    bad, seen = [], set()
    for m in _LOCANT_RE.finditer(text):
        said = m.group(0).lower()
        if said in seen:
            continue
        seen.add(said)
        canon = next((n for n in _LOCANT_NAMES if n.lower() == said), said)
        if canon in have:
            continue
        cls = next((entry[0] for entry in _RING_CLASS.values()
                    if canon in entry[1].values()), None)
        if cls in classes:
            right = ", ".join(sorted(n for n, c in _ring_locants(mol) if c == cls))
            bad.append(f'calls the ring "{canon}" — it is {right}')
        else:
            bad.append(f'names a "{canon}" ring; the molecule has none')
    return bad


def inverted_spread_claims(text: str) -> list:
    """Comparative ± claims in ``text`` whose ordering is numerically false.

    Also reports claims made about two EQUAL spreads — "±0.0 is tighter than
    ±0.0" asserts a difference that does not exist.
    """
    bad = []
    for pat, want_lt in ((_CMP_TIGHT, True), (_CMP_WIDE, False)):
        for hit in pat.finditer(text or ""):
            a, b = float(hit.group(1)), float(hit.group(2))
            if a == b or (a < b) is not want_lt:
                bad.append(hit.group(0).strip())
    return bad


# ── guards on the Landing Safety block being paraphrased rather than quoted ──
# The block states counts, a tie verdict and candidate numbers; the generator was
# measured re-deriving all three. Over 5 rounds x 11 criteria before these guards
# existed: a tie was NEVER once reported as a tie (the prose always claimed one
# candidate led), safe counts were wrong in 4 of 5 rounds, and one sentence picked a
# candidate BECAUSE its spread was the wider one. Each of these is a specific,
# checkable claim, so it is checked and regenerated rather than hoped away.
_LEAD = re.compile(r"\b(highest|most|best|leads?|leading|superior|outperform\w*)\b",
                   re.I)
_TIE_SAID = re.compile(r"not separate|does not decide|tie|ties|tied|same", re.I)
_WORSE_PICK = re.compile(r"\b(select|choose|chose|selecting|take|commit)\b", re.I)
_WORSE_BAD = re.compile(r"\b(wider|widest|larger spread|looser|riskier|less reliable|"
                        r"risking overshoot|risks? overshoot\w*)\b", re.I)


_NOTE_RE = re.compile(r"NOTE running the order above lands on (.+?) — it is (.+?) "
                      r"that separates them — but the candidate to select is #(\d+)")
_NOTE_KW = {"gap closed": r"gap clos|closes? .{0,14}gap|\d+% of the gap",
            "safe landings": r"safe landing|safe count|safety|lands? .{0,12}safe",
            "edit size": r"heavy|size|small(er|est)? edit",
            # `\b` does NOT sit between Δ and MW: Python treats Δ as a word character,
            # so `\bMW\b` misses "ΔMW" — the exact string the block prints and the
            # prose copies. ASCII-only lookarounds instead.
            "added mass": r"(?<![A-Za-z0-9_])(mass|MW)(?![A-Za-z0-9_])|molecular weight",
            "the tool's own rank": r"rank|first choice|top[- ]ranked",
            "breakage": r"break"}
# An unambiguous concession, anywhere in the text.
_CONCEDE = re.compile(r"does ?n[o']t (?:lead|favou?r|win|beat)|"
                      r"match(?:ing|es) the leader|only matches|no column favou?rs|"
                      r"\b(?:conceding|concedes?)\b", re.I)
# A contrastive marker. On its own it proves nothing — "though logS remains out of
# range" concedes a property, not the criterion — so it counts only in a clause that
# also names the criterion or the candidate the order actually lands on.
_CONTRA = re.compile(r"\b(but|though|although|while|whereas|despite|however|yet|"
                     r"accepting|trades?|behind|trails?|loses|lacks|worse|not the)\b",
                     re.I)


# table-1 column order, for binding a tie claim to the column it names
# table column order, for binding a tie claim to the column it names
_TIE_COL = {"rank": 0, "safe landing": 1, "safe count": 1, "in after": 2,
            "in range": 2, "gap clos": 3, "heavy": 4, "mw": 5, "break": 6}


# Which candidate the prose says it is committing. The undo clause names a DIFFERENT
# candidate on purpose ("revert and try #2"), so everything from the first conditional
# or reversal word onward is cut before looking, and "#2-#4" range notation is stripped.
_PICK_VERB = re.compile(
    r"(?:tak(?:e|ing)|try|trying|apply|applying|commit(?:ting)?|select(?:ing)?|"
    r"choos(?:e|ing)|chose|chosen|go with|test(?:ing)?|pick(?:ing)?|attempt(?:ing)?)"
    r"\s+(?:candidate\s+)?#(\d+)"
    r"|candidate\s+#(\d+)\s+is\s+(?:chosen|selected|taken|preferred)"
    r"|#(\d+)\s+is\s+(?:chosen|selected|taken|preferred|a trial|the trial)", re.I)
_PICK_CUT = re.compile(r"\b(revert|undo|back it|switch to|fall ?back|otherwise|"
                       r"instead\b|\bif\b)", re.I)
_PICK_RANGE = re.compile(r"#\d+\s*[\u2013\u2014-]\s*#?\d+")


def _claimed_pick(text: str):
    """Candidate numbers *text* claims, in order, before any undo clause.

    The LAST one is the commitment: a span that weighs the field before deciding names
    the leader first and the candidate it actually takes last ("rank selects #1, but I
    test #2 as a trial"). Taking the set instead lets that span pass, because the right
    number is in it somewhere.
    """
    cut = _PICK_CUT.search(text or "")
    head = (text or "")[:cut.start()] if cut else (text or "")
    head = _PICK_RANGE.sub(" ", head)
    return [int(x) for m in _PICK_VERB.finditer(head) for x in m.groups() if x]


def landing_claim_errors(text: str, block: str) -> list:
    """Claims in ``text`` that contradict the Landing Safety ``block``.

    Returns human-readable strings for the retry note. Empty when the prose only
    reports what the block decided.
    """
    bad, text = [], text or ""
    block = block or ""
    # a count the block does not contain — "9/13" when no row says 9/13
    pairs = set(re.findall(r"(\d+)/(\d+)", block))
    if pairs:
        for m in re.finditer(r"(\d+)\s*/\s*(\d+)", text):
            if m.groups() not in pairs:
                bad.append(f'"{m.group(0)}" is not a count this block reports')
        safes = {a for a, _b in pairs}
        for m in re.finditer(r"(?:for|satisfies|lands?|meets?)\s+(\d+)\s+"
                             r"(?:propert|constraint)", text, re.I):
            if m.group(1) not in safes:
                bad.append(f'"{m.group(0)}" does not match any safe count in the block')
    # asserting a lead on a criterion the block calls a tie
    for crit, kw in (("safe landings", r"safe landing|safety"), ("breakage", r"break")):
        tied = (f"{crit}: every" in block
                or (crit == "breakage" and "no candidate breaks" in block))
        if not tied:
            continue
        for clause in re.split(r";|\.(?:\s|$)", text):
            if re.search(kw, clause, re.I) and _LEAD.search(clause) \
                    and not _TIE_SAID.search(clause):
                bad.append(f'claims a lead on {crit}, which the block calls a tie')
                break
    # candidate numbers outside 1..N (the blocks are 1-based)
    idx = {int(x) for x in re.findall(r"^#(\d+)", block, re.M)}
    n = max(idx) if idx else 0          # each candidate has one row per table, so
    if n:                               # count DISTINCT indices, not rows
        for m in re.finditer(r"#(\d+)", text):
            if not 1 <= int(m.group(1)) <= n:
                bad.append(f'"#{m.group(1)}" is not one of the {n} candidates')
    # picking a candidate BECAUSE its value is the worse one. The "because" must come
    # FIRST: "accepting its wider ±0.543 because the acid adds the required HBD" is the
    # shape we want, and an unanchored search flags it.
    for clause in re.split(r";|\.(?:\s|$)", text):
        low = clause.lower()
        if _WORSE_PICK.search(clause) and "because" in low:
            if _WORSE_BAD.search(clause[low.index("because"):][:70]):
                bad.append("selects a candidate BECAUSE its spread is the "
                           "wider/riskier one")
                break
    # claiming a tie that no column supports. Every count is in the block, so the
    # "number not in the block" rule above passes "safe landings tie #1, #2 and #4 at
    # 8/12" even when #2 is the sole leader at 9/12 — the 8/12 is real, the grouping is
    # invented. So each candidate named in a tie clause must actually carry that cell.
    grid = {}
    for line in block.splitlines():
        if not line.startswith("#"):
            continue
        # the selected candidate's row carries a trailing marker; splitting it in
        # would shift every table-2 cell of that ONE row and silently misalign the
        # column comparison below
        parts = line.split("<--")[0].split()
        grid.setdefault(parts[0].lstrip("#"), []).extend(parts[1:])
    if grid:
        # only the candidates ADJACENT to the tie word count — "safe landings tie, but
        # #1 and #3 satisfy 3/4" ties nothing to #1/#3, it just happens to name them.
        _CANDS = r"((?:#\d+(?:[\s,]+|\s+and\s+)?)+)"
        # the gap around the tie word admits only its natural connectives. A wider gap
        # swallows a contrast — "safe landings tie, but #1 and #3 satisfy 3/4" is not a
        # claim that #1 and #3 tie at 3/4.
        for pat in (r"\btied?\b(?:\s+(?:at|between|among|on))?\s*"
                    + _CANDS + r"[^.;]{0,14}?(\d+)/(\d+)",
                    _CANDS + r"\s*(?:both|all|each|are|is)?\s*\btied?\b"
                    r"[^.;]{0,14}?(\d+)/(\d+)"):
            for m in re.finditer(pat, text, re.I):
                named = [n for n in re.findall(r"#(\d+)", m.group(1)) if n in grid]
                if len(named) < 2:
                    continue
                val = f"{m.group(2)}/{m.group(3)}"
                # bind the claim to the column it names. Without this "safe landings
                # tie #1, #2, #4 at 8/12" passes whenever some OTHER column happens to
                # read 8/12 for all three — which is exactly what safe2 did.
                lead_in = text[max(0, m.start() - 45):m.start() + len(m.group(0))].lower()
                col = next((j for kw, j in _TIE_COL.items() if kw in lead_in), None)
                cols = [col] if col is not None else range(
                    max(len(grid[n]) for n in named))
                if not any(all(grid[n][j:j + 1] == [val] for n in named)
                           for j in cols):
                    bad.append("claims " + ", ".join("#" + n for n in named)
                               + f" tie at {val}, which no column of the block shows")
    # naming a candidate other than the one the tool call actually commits. Measured on
    # 3,817 generated rounds: 10 (0.26%) argued for one candidate and committed another
    # ("all metrics tie, so #2 is selected" with #1 in the tool call). Nearly all of them
    # are rounds where the block's NOTE asked for a concession and the prose wrote the
    # concession the wrong way round. It is the worst defect shape available here — the
    # span teaches saying one thing and doing another — and it is exactly checkable,
    # because the block marks the committed candidate.
    sel = re.search(r"^#(\d+).*<-- the candidate to select", block, re.M)
    if sel:
        picked_i = int(sel.group(1))
        claims = _claimed_pick(text)
        if claims and claims[-1] != picked_i:
            bad.append(f"says it commits #{claims[-1]} but the tool call commits "
                       f"#{picked_i} — name the candidate you are actually applying")
    # ignoring the block's NOTE. The NOTE fires when the stated decision order lands on
    # a different candidate from the one committed — two rounds in three, measured — and
    # it is the one instruction the prose must not skip: without the concession the
    # sentence reads as a claim that the committed candidate won, which the table
    # contradicts one line above.
    nt = _NOTE_RE.search(block)
    if nt:
        leaders, crit = set(re.findall(r"#(\d+)", nt.group(1))), nt.group(2)
        kw = _NOTE_KW.get(crit, "")
        clauses = re.split(r";|\.(?:\s|$)", text)
        credited = bool(kw) and any(
            re.search(kw, c, re.I) and any(f"#{n}" in c for n in leaders)
            for c in clauses)
        tie_said = bool(kw) and any(re.search(kw, c, re.I) and _TIE_SAID.search(c)
                                    for c in clauses)
        contra = any(_CONTRA.search(c)
                     and ((kw and re.search(kw, c, re.I))
                          or any(f"#{n}" in c for n in leaders))
                     for c in clauses)
        if not (credited or tie_said or contra or _CONCEDE.search(text)):
            bad.append(f'ignores the block NOTE: it must concede that the committed '
                       f'candidate does not lead on {crit}')
    return bad


# Chemical names the generator reaches for. Only used NEGATIVELY: a name in this
# list that does not appear in the round's own fact block was invented. Measured
# on 32 authored-edit generations, 16% invented one — and they are wrong, not just
# unsupported ("isopropylphenyl" written as "isobutyl", an acetoxy ester called
# "lipophilic bulk", a bare attachment point described as "the acyclic methyl").
_KNOWN_NAMES = (
    "benzene phenyl pyridine pyrimidine pyrazine pyridazine triazine furan thiophene "
    "pyrrole imidazole pyrazole oxazole isoxazole thiazole isothiazole triazole "
    "tetrazole oxadiazole thiadiazole indole indoline indazole benzimidazole "
    "benzofuran benzothiophene benzothiazole benzoxazole quinoline isoquinoline "
    "quinazoline quinoxaline naphthalene anthracene purine carbazole chromene "
    "coumarin piperidine piperazine pyrrolidine morpholine thiomorpholine azetidine "
    "aziridine oxetane oxirane tetrahydrofuran tetrahydropyran cyclopropane "
    "cyclobutane cyclopentane cyclohexane cycloheptane adamantane "
    "phenol aniline anisole catechol "
    "sulfonamide sulfone sulfoxide sulfonyl thiol thioether disulfide "
    "carboxamide amide urea thiourea carbamate guanidine amidine imine oxime "
    "hydrazone hydrazine azide nitrile nitro ester lactone lactam ketone aldehyde "
    "ether alcohol hydroxyl carboxylic acid amine "
    "methoxy ethoxy propoxy butoxy phenoxy benzyloxy acetoxy "
    "methyl ethyl propyl isopropyl butyl isobutyl tert-butyl benzyl allyl vinyl "
    "trifluoromethyl fluoromethyl aryl halide fluoro chloro bromo iodo alkene alkyne "
    # Acyl / amide / ester family. These are where the generator misnames most
    # often — measured: an N-acetyl written as "isobutyramide" — because every
    # member looks like every other one and the chain length is easy to miscount.
    "formyl acetyl propionyl butyryl isobutyryl benzoyl acryloyl "
    "formamide acetamide propionamide butyramide isobutyramide benzamide acrylamide "
    "sulfonamide methanesulfonamide "
    "formate acetate propionate butyrate benzoate "
    "methylamine dimethylamine ethylamine methylamino dimethylamino "
    "hydroxymethyl aminomethyl cyanomethyl nitrophenyl aminophenyl hydroxyphenyl "
    "methylphenyl dimethylphenyl methoxyphenyl chlorophenyl fluorophenyl"
).split()
# Multi-word names must be checked before their single-word parts.
_KNOWN_NAMES = sorted(set(_KNOWN_NAMES) | {"carboxylic acid", "aryl halide", "tert-butyl"},
                      key=len, reverse=True)


def unknowable_numbers(text: str, allowed: str, non_derivable) -> list:
    """Numeric claims about a NON-DERIVABLE property that ``allowed`` never showed.

    Used by the authored-edit branch, which renders ``suggest_edits`` as an empty
    list. With no candidate table there is no mmpdb Δ to read, so a span that says
    "Δ +0.038 for BBBP" is quoting a number the assistant could not have obtained —
    it would have to fabricate it at inference. Properties whose change is fixed by
    the fragment (MW, heavy_atoms …) are exempt: those the assistant can count, and
    they are listed in ``allowed`` for exactly that reason.

    Returns the offending fragments of text, e.g. ``["BBBP +0.038"]``.
    """
    ok = set(re.findall(r"-?\d+\.?\d*", allowed or ""))
    bad = []
    for name in non_derivable:
        # a number within ~40 chars of the property name, in either order
        for m in re.finditer(
            rf"(?:{re.escape(name)}\W{{0,40}}?([+-]?\d+\.?\d*)"
            rf"|([+-]?\d+\.?\d*)\W{{0,40}}?{re.escape(name)})",
            text or "",
        ):
            val = m.group(1) or m.group(2)
            if val is None:
                continue
            if val.lstrip("+-") in ok or val in ok:
                continue                      # a value the assistant already measured
            bad.append(f"{name} {val}")
    return bad


def invented_chemical_names(text: str, facts: str) -> list:
    """Chemical names ``text`` uses that its own fact block never licensed.

    ``facts`` should be every AUTHORITATIVE string the prompt showed for this
    round — the two ``describe_fragment`` outputs, ``describe_anchor``, and the
    user query (which names the required rings). A name absent from all of them
    was invented by the generator, so the span is rejected and regenerated.
    """
    low = (text or "").lower()
    src = (facts or "").lower()
    return [n for n in _KNOWN_NAMES if n in low and n not in src]


def format_candidate_options(candidates, props: Optional[dict],
                             targets: Optional[dict],
                             from_smiles=None, to_smiles=None, anchors=None) -> str:
    """Render the returned candidates as decision material for the prompt.

    Per candidate and per CONSTRAINED property: current value, the predicted
    ``Δ avg ± std``, the resulting value, and whether it lands in the target
    range. Without this the reasoning could only justify a choice already made —
    the model never saw the alternatives it was supposedly choosing between.
    """
    if not candidates:
        return "(none — suggest_edits returned no candidates)"
    props = props or {}
    targets = targets or {}
    _pick_i = picked_index(candidates, from_smiles, to_smiles, anchors)
    # Tightness ranks are computed here, not left to the model's prose arithmetic.
    ranks = _spread_ranks(candidates, targets)
    lines: list[str] = []
    for i, c in enumerate(candidates, 1):
        if not isinstance(c, dict):
            continue
        picked = (i == _pick_i)
        head = (f"#{i} {c.get('from_smiles')} -> {c.get('to_smiles')} "
                f"(anchors {c.get('anchors')}; predicted total gap "
                f"{c.get('predicted_gap')})")
        lines.append(head + ("   <-- the candidate to select" if picked else ""))
        # Name the ALTERNATIVES too: the reasoning now discusses the candidates it
        # rejects, and without verified names for those it invents them (a
        # hydroxymethyl gets called a "methoxy" again).
        lines.append(f"     removes: {short_name(c.get('from_smiles'))}"
                     f" | attaches: {short_name(c.get('to_smiles'))}")
        delta = c.get("delta") or {}
        for name, rng in targets.items():
            d = delta.get(name)
            cur = props.get(name)
            if d is None:
                if cur is not None:
                    lines.append(f"     {name}: {_fmt_num(cur)} unchanged")
                continue
            avg, std = d.get("avg"), d.get("std")
            lo = rng[0] if isinstance(rng, (list, tuple)) and len(rng) > 0 else None
            hi = rng[1] if isinstance(rng, (list, tuple)) and len(rng) > 1 else None
            tgt = (f"target {_fmt_num(lo) if lo is not None else '-inf'}"
                   f" .. {_fmt_num(hi) if hi is not None else '+inf'}")
            tight = (ranks.get(name) or {}).get(i)
            # Drop the "all tie … no discrimination" verdict here: now that delta
            # reports every constrained property (Δ 0 included), that label lands
            # on most lines of every candidate, and ``spread_verdict`` already
            # states it once per property. Genuine rank labels are kept.
            if tight and "no discrimination" in tight:
                tight = None
            tag = f" [{tight}]" if tight else ""
            if cur is None:
                lines.append(f"     {name}: Δ {avg:+g} ± {std}{tag} ({tgt})")
                continue
            new = float(cur) + float(avg)
            inside = ((lo is None or new >= float(lo))
                      and (hi is None or new <= float(hi)))
            lines.append(
                f"     {name}: {_fmt_num(cur)} -> {_fmt_num(new)} "
                f"(Δ {avg:+g} ± {std}{tag}); {tgt} -> "
                f"{'IN range' if inside else 'still OUT of range'}")
    return "\n".join(lines)


def picked_number(candidates, from_smiles, to_smiles, anchors=None) -> str:
    """``"#3"`` — which entry of the candidate list the round actually commits.

    Stated in the prompt because leaving it implicit costs accuracy: the block marks the
    row with an arrow and the prompt gives the SMILES, but on rounds where the block's
    NOTE asks for a concession the generator loses track and argues for one candidate
    while the tool call applies another. Measured at 0.26% of rounds, and two retries
    were not enough to fix it — the model has to be told the number.
    """
    i = picked_index(candidates, from_smiles, to_smiles, anchors)
    return f"#{i}" if i else "(not in the candidate list)"


def candidate_rank_note(candidates, from_smiles, to_smiles, anchors=None) -> str:
    """Where the committed edit sits in the ranked ``suggest_edits`` list.

    The prompt used to tell the model to say it was taking "the top candidate"
    regardless of which one the chain actually committed (measured: only 53% were
    rank 1), so the rank is now stated explicitly and truthfully.
    """
    if not candidates:
        return ("suggest_edits returned no candidates for this round — do NOT "
                "claim this edit came from the tool's suggestions")
    n = len(candidates)
    for i, c in enumerate(candidates):
        if not isinstance(c, dict):
            continue
        pass
    k = picked_index(candidates, from_smiles, to_smiles, anchors)
    if k == 1:
        return f"the TOP-ranked candidate (#1 of {n}) returned by suggest_edits"
    if k:
        return (f"candidate #{k} of {n} in the ranked suggest_edits list "
                f"(NOT the top-ranked one — do not call it the top candidate)")
    return (f"NOT one of the {n} candidates suggest_edits returned — do NOT claim "
            f"it came from the tool's suggestions; justify it on chemistry alone")
