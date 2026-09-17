#!/usr/bin/env python3
"""
scaffold_describer.py
=====================
The **logic library** that turns a Murcko-scaffold analysis
(scaffold_analyzer.analyze_scaffold) into a FACTS block, a controlled natural-language
draft, and a validation pass. No model calls — purely deterministic.

- format_facts(an)     : analysis dict -> the authoritative FACTS block for the prompt.
- render_template(an)   : analysis dict -> the controlled natural-language draft the LLM polishes.
- validate_text(t, an)  : check the polished text against the analysed facts and return the
                          list of violations (hallucination detection).
- ScaffoldDescriber     : the single entry point that ties these together so the same input
                          reproduces the same prompt and the same validation.

Naming and structural facts come only from what scaffold_analyzer derived
deterministically; IUPAC locants that cannot be asserted with certainty are never
invented.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Optional

from rdkit import Chem

from scaffold_analyzer import analyze_scaffold, RING_SMILES_NAMES
from scaffold_prompts import (
    SYSTEM_PROMPT, USER_TEMPLATE, REVISE_TEMPLATE, CONDENSE_TEMPLATE,
)


_UNICODE_FOLD = str.maketrans({
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-",
    "―": "-", "−": "-",
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"',
    " ": " ", " ": " ", " ": " ", " ": " ", "​": "",
})


def normalize_text(text: str) -> str:
    """Normalize model-specific typography without changing chemical content."""
    if not text:
        return text
    return " ".join(unicodedata.normalize("NFKC", text).translate(_UNICODE_FOLD).split())


_VOWELS = "aeiou"


def _a_an(phrase: str) -> str:
    return "an" if phrase[:1].lower() in _VOWELS else "a"


_NUM = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}


# --------------------------------------------------------------------------- #
#  Noun-phrase helpers: ring-system name -> a natural noun phrase
# --------------------------------------------------------------------------- #
def _is_phrase(name: str) -> bool:
    """Do not append 'ring' when the name is already a complete phrase (a descriptive name)."""
    low = name.lower()
    return (low.endswith(("ring", "rings", "system", "carbocycle"))
            or " to " in low or "membered" in low or "spiro" in low
            or "bridged" in low or "fused" in low)


def _np(rs: dict) -> str:
    """Ring system -> noun phrase. A monocycle becomes '{name} ring'; a named polycycle and a
    descriptive name are both left as they are."""
    name = rs["name"]
    if _is_phrase(name):
        return name
    if rs["n_rings"] == 1:
        return f"{name} ring"
    return name


def _bare(rs: dict) -> str:
    """The short name used in 'the {X}'."""
    return rs["name"]


_RINGCOUNT = {1: "monocyclic", 2: "bicyclic", 3: "tricyclic", 4: "tetracyclic",
              5: "pentacyclic"}


def _the_ref(rs: dict) -> str:
    """How a later sentence refers back to the ring system. A descriptive composite name
    becomes 'this ring system'."""
    name = rs["name"]
    low = name.lower()
    if " to " in low or "spiro" in low or "bridged" in low or "membered" in low:
        return "this ring system"
    return f"the {name}"


def _ring_hetero_phrase(counts: dict) -> str:
    """{'N':2} → 'two ring nitrogens'; {'N':1,'O':1} → 'one ring nitrogen and one ring oxygen'."""
    names = {"N": "ring nitrogen", "O": "ring oxygen", "S": "ring sulfur",
             "P": "ring phosphorus", "B": "ring boron", "Se": "ring selenium",
             "Si": "ring silicon"}
    parts = []
    for e in sorted(counts, key=lambda x: (-counts[x], x)):
        nm = names.get(e, f"ring {e}")
        c = counts[e]
        parts.append(f"{_NUM.get(c, str(c))} {nm}{'' if c == 1 else 's'}")
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


# --------------------------------------------------------------------------- #
#  Controlled natural-language draft renderer (deterministic; the LLM polishes this draft)
# --------------------------------------------------------------------------- #
def _pick_core(rss: list[dict]) -> dict:
    """Pick the ring system with the most rings as the core. Ties break on heteroatom count,
    then on position."""
    return max(rss, key=lambda r: (r["n_rings"], sum(r["hetero_counts"].values()),
                                   -r["id"]))


def _connection_clause(other_rs: dict, conn: dict) -> str:
    """Render one connection, relative to the core, as a 'linked to ... by ...' clause."""
    other = _np(other_rs)
    if conn["relation"] == "linker-connected":
        return f"connected to {_a_an(other)} {other} through {_a_an(conn['type'])} {conn['type']}"
    if "biaryl" in conn["type"]:
        return f"joined to {_a_an(other)} {other} in a biaryl linkage"
    return f"directly bonded to {_a_an(other)} {other}"


def render_template(an: dict) -> str:
    """Analysis dict -> a controlled natural-language draft of 1-4 sentences. Always faithful:
    it states only facts."""
    # Acyclic functional-group skeleton: use the deterministic, faithful sentence that
    # scaffold_analyzer already produced, unchanged.
    if an.get("scaffold_kind") == "functional_group":
        return an.get("fg_description") or an.get("topology_summary", "")
    if not an.get("has_scaffold"):
        return ("The molecule has no ring scaffold — it is an acyclic framework with no "
                "ring system to describe.")
    rss = an["ring_systems"]
    conns = an["connections"]
    sentences: list[str] = []

    # --- (1) skeleton / connections ---
    if len(rss) == 1:
        rs = rss[0]
        np = _np(rs)
        if rs["n_rings"] == 1:
            sentences.append(f"The scaffold is {_a_an(np)} {np}.")
        elif _is_phrase(rs["name"]):
            sentences.append(f"The scaffold is {_a_an(np)} {np}.")
        else:
            bare = _bare(rs)
            topo = rs["internal_topology"]
            poly = _RINGCOUNT.get(rs["n_rings"], f"{rs['n_rings']}-ring")
            sentences.append(
                f"The scaffold is {_a_an(bare)} {bare}, {_a_an(topo)} {topo} {poly} "
                "ring system.")
    else:
        core = _pick_core(rss)
        core_bare = _bare(core)
        core_conns = [c for c in conns if core["id"] in (c["a"], c["b"])]
        clauses = []
        for c in core_conns:
            oid = c["b"] if c["a"] == core["id"] else c["a"]
            other = next((r for r in rss if r["id"] == oid), None)
            if other is not None:
                clauses.append(_connection_clause(other, c))
        head = f"The scaffold contains {_a_an(core_bare)} {core_bare} core"
        if clauses:
            sentences.append(head + " " + " and ".join(clauses) + ".")
        else:
            names = ", ".join(_np(r) for r in rss)
            sentences.append(f"The scaffold comprises {_NUM.get(len(rss), str(len(rss)))} "
                             f"separate ring systems ({names}).")
        # Connections that do not pass through the core are rare, but are not dropped.
        extra = [c for c in conns if core["id"] not in (c["a"], c["b"])]
        for c in extra:
            a = next((r for r in rss if r["id"] == c["a"]), None)
            b = next((r for r in rss if r["id"] == c["b"]), None)
            if a and b:
                sentences.append(f"{_the_ref(a).capitalize()} is " + _connection_clause(b, c) + ".")

    # --- (2) heteroatom pattern ---
    het_systems = [rs for rs in rss if rs["hetero_counts"]]
    if het_systems:
        clauses = []
        for rs in het_systems:
            clauses.append(f"{_the_ref(rs)} contributes {_ring_hetero_phrase(rs['hetero_counts'])}")
        s = "; ".join(clauses)
        sentences.append(s[0].upper() + s[1:] + ".")
    elif len(rss) == 1 and rss[0]["aromatic"]:
        sentences.append("It is an all-carbon aromatic framework.")

    # --- (2b) exocyclic carbonyl on a ring (lactam/lactone/-one), only when the name
    #          does not already carry it ---
    co = [f"{_the_ref(rs)} {rs['carbonyl_desc']}" for rs in rss if rs.get("carbonyl_desc")]
    if co:
        s = "; ".join(co)
        sentences.append(s[0].upper() + s[1:] + ".")

    # --- (3) aromaticity / saturation, only when it adds information ---
    arom = an.get("aromaticity", "")
    if arom == "all aliphatic (saturated)":
        sentences.append("Every ring is saturated (non-aromatic).")
    elif arom == "mixed aromatic/aliphatic":
        sentences.append("Both aromatic and saturated rings are present.")

    # --- (4) substituent attachment positions are no longer described ---
    #     Substituents are stripped by the Bemis-Murcko reduction, so where a substituent
    #     sits (a ring attachment point, a benzene ortho/meta/para pattern) is not checked
    #     by any evaluation SMARTS — neither the verdict set nor the per-dimension ones.
    #     This clause is left empty so the description never carries information that
    #     cannot be verified. (Which ring atom a linker attaches to IS covered, in the
    #     skeleton/connection clause (1): a linker is part of the scaffold, so it is
    #     verified.)

    return " ".join(sentences)


# --------------------------------------------------------------------------- #
#  The authoritative FACTS block for the prompt
# --------------------------------------------------------------------------- #
def format_facts(an: dict) -> str:
    """Analysis dict -> the COMPUTED FACTS text injected into the prompt."""
    if not an.get("has_scaffold"):
        return ("- This molecule has NO ring scaffold (Bemis–Murcko scaffold is empty); it is an "
                "acyclic framework. Describe it as acyclic with no ring system.")
    rss = an["ring_systems"]
    lines = [f"- Scaffold (substituents stripped): {an.get('scaffold_smiles', '')}",
             f"- Ring systems: {len(rss)}"]
    for i, rs in enumerate(rss):
        arom = rs.get("aromaticity_desc") or ("aromatic" if rs["aromatic"]
                                              else "saturated / non-aromatic")
        sizes = "+".join(str(s) for s in rs["ring_sizes"])
        het = _ring_hetero_phrase(rs["hetero_counts"]) if rs["hetero_counts"] \
            else "no heteroatoms (all-carbon)"
        topo = f", {rs['internal_topology']}" if rs["n_rings"] > 1 else ""
        lines.append(
            f"    {i + 1}. {rs['name']} — {rs['n_rings']} ring(s) [{arom}{topo}], "
            f"ring size(s) {sizes}, heteroatoms: {het}")
        # Exocyclic carbonyl on a ring (lactam/lactone/-one): stated separately when the
        # name does not already carry it.
        if rs.get("carbonyl_desc"):
            lines.append(f"         the ring system {rs['carbonyl_desc']}")
        # NOTE: substituent attachment points and substitution patterns (e.g. "substituted
        # at 2,5", benzene ortho/meta/para) are no longer put in FACTS. Substituents are
        # stripped from the scaffold, so no evaluation SMARTS can check them, and
        # unverifiable information must not leak into the description. (Linker attachment
        # positions are still provided, in the connection entries below: a linker is part
        # of the scaffold and is verified.)
    # Connections: kind + shape (the pattern, including =O/=S branches) + total length +
    # the attachment position at each end
    conns = an.get("connections") or []
    if conns:
        lines.append("- How the ring systems are joined (linker identity, shape, length & "
                     "attachment positions):")
        for c in conns:
            a = rss[c["a"]]["name"] if c["a"] < len(rss) else f"system{c['a']}"
            b = rss[c["b"]]["name"] if c["b"] < len(rss) else f"system{c['b']}"
            lk = c.get("linker") or {}
            pat = lk.get("pattern") or ""
            if lk.get("length"):
                shape = (f"; chain shape {pat} ({lk['length']}-atom backbone, "
                         f"{lk['bond_span']} bonds)")
            elif pat and pat not in ("single bond",):
                shape = f"; {pat}"
            else:
                shape = "; a direct single bond"
            ap = c.get("a_position", {}).get("position", "")
            bp = c.get("b_position", {}).get("position", "")
            where = f" — attaches at {ap} (on {a}) and {bp} (on {b})" if (ap or bp) else ""
            lines.append(f"    * {a} ⟷ {b}: {c['relation']} via {c['type']}{shape}{where}")
    # Attribute aromaticity to the rings, not to the scaffold as a whole: a line phrased about
    # the whole gets echoed back as "the scaffold is ...", which this description must not say.
    # No heteroatom total here — total_het sums ring systems only (linkers excluded), so it is a
    # partial sum of the per-ring lines above.
    _arom = an.get("aromaticity", "")
    lines.append("- Aromaticity across the rings: " + {
        "all aromatic": "every ring is aromatic",
        "all aliphatic (saturated)": "every ring is saturated, none aromatic",
        "mixed aromatic/aliphatic": "some rings are aromatic, others saturated",
    }.get(_arom, _arom))
    lines.append("- Positions: the ring locants and 'position …' phrases ABOVE are computed and "
                 "correct — you may state them. Do NOT invent any other ring locant the FACTS do "
                 "not give.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
#  Validation: polished text against the analysed facts (hallucination and style violations)
# --------------------------------------------------------------------------- #
def _build_ring_vocab() -> set:
    """Build the set of 'recognisable ring words' from the analyzer's known ring names, for
    swap detection."""
    base = {
        "benzene", "pyridine", "pyrimidine", "pyrazine", "pyridazine", "triazine",
        "pyrrole", "imidazole", "pyrazole", "furan", "thiophene", "oxazole", "isoxazole",
        "thiazole", "isothiazole", "triazole", "tetrazole", "oxadiazole", "thiadiazole",
        "furazan", "piperidine", "piperazine", "morpholine", "thiomorpholine",
        "pyrrolidine", "tetrahydrofuran", "tetrahydropyran", "oxetane", "azetidine",
        "aziridine", "oxirane", "thiirane", "thietane", "azepane", "oxepane", "azocane",
        "dioxane", "dioxolane", "oxolane", "thiane", "thiolane", "diazepane",
        "indole", "isoindole", "indoline", "indane", "indene", "benzimidazole",
        "indazole", "benzofuran", "benzothiophene", "benzoxazole", "benzothiazole",
        "benzotriazole", "benzodioxole", "benzodioxane", "naphthalene", "quinoline",
        "isoquinoline", "quinazoline", "quinoxaline", "cinnoline", "phthalazine",
        "naphthyridine", "purine", "pteridine", "carbazole", "acridine", "phenazine",
        "phenothiazine", "phenoxazine", "xanthene", "thioxanthene", "dibenzofuran",
        "fluorene", "anthracene", "phenanthrene", "chromene", "chromane", "chromone",
        "coumarin", "indolizine", "quinolizidine", "quinuclidine", "norbornane",
        "adamantane", "tropane", "azaindole", "imidazopyridine", "pyrazolopyridine",
        "cyclopropane", "cyclobutane", "cyclopentane", "cyclohexane", "cycloheptane",
        "cyclooctane", "cyclohexene", "cyclopentene", "azepine", "pyran",
    }
    # Also absorb the extra tokens appearing in the curated dictionary's names, to stay in sync
    for nm in RING_SMILES_NAMES.values():
        for tok in re.findall(r"[a-z]{5,}", nm.lower()):
            if tok not in ("ring", "system", "fused", "spiro", "bridged", "linked",
                           "membered", "saturated", "aromatic", "skeleton", "hexahydro"):
                base.add(tok)
    return base


_RING_VOCAB = _build_ring_vocab()
# Adjectival / substituent forms -> the base ring name
_ADJ_TO_BASE = {
    "phenyl": "benzene", "benzyl": "benzene", "pyridyl": "pyridine",
    "pyridinyl": "pyridine", "pyrimidinyl": "pyrimidine", "pyrazinyl": "pyrazine",
    "furyl": "furan", "furanyl": "furan", "thienyl": "thiophene",
    "thiophenyl": "thiophene", "piperidinyl": "piperidine", "piperazinyl": "piperazine",
    "morpholino": "morpholine", "morpholinyl": "morpholine", "pyrrolidinyl": "pyrrolidine",
    "pyrrolyl": "pyrrole", "imidazolyl": "imidazole", "pyrazolyl": "pyrazole",
    "thiazolyl": "thiazole", "oxazolyl": "oxazole", "indolyl": "indole",
    "naphthyl": "naphthalene", "quinolinyl": "quinoline", "isoquinolinyl": "isoquinoline",
    "quinazolinyl": "quinazoline", "tetrazolyl": "tetrazole", "triazolyl": "triazole",
}
_VOCAB_RE = re.compile(
    r"\b(" + "|".join(sorted(set(_RING_VOCAB) | set(_ADJ_TO_BASE), key=len, reverse=True))
    + r")\w*", re.I)

_NEG = re.compile(r"\b(no|non|not|without|devoid|lack\w*|absen\w*|free|neither|nor|"
                  r"un)\b|\bnon-|\bun-", re.I)
_V_AROM = re.compile(r"\b(aromatic|aryl|benzene|benzo|phenyl|heteroaromatic|biaryl)\w*", re.I)
_V_SATUR = re.compile(r"\b(saturated|aliphatic\s+ring|non-?aromatic|fully\s+saturated)\b", re.I)
_V_CHARGE = re.compile(r"\b(protonat\w*|deprotonat\w*|cation\w*|anion\w*|ammonium|"
                       r"carboxylat\w*|zwitterion\w*|positively\s+charged|negatively\s+charged|"
                       r"charged)\b", re.I)
_V_STEREO = re.compile(r"\b(stereocent\w*|chiral\w*|enantiomer\w*|diastereomer\w*|"
                       r"stereochem\w*)\b|\(R\)-|\(S\)-", re.I)
_V_LOCANT = re.compile(r"\b([CNOS])-?(\d{1,2})\b|\bposition\s+(\d{1,2})\b|\b(\d{1,2})-position\b")
_V_PURPOSE_TAIL = re.compile(
    r"\bfor\s+(?:further|subsequent|additional|easy|ready|sar\b|structure-activity|"
    r"synthetic|medicinal|downstream|future)\s*\w*|\bfor\s+sar\b|"
    r"\b(?:can|could|may)\s+be\s+(?:further\s+)?(?:substitut|elaborat|functionaliz|"
    r"derivatiz|decorat|modif|append)\w*", re.I)
_V_META = re.compile(r"\bthe\s+(?:SMILES|SMARTS|scaffold\s+pattern|query)\b|"
                     r"\bthis\s+(?:SMILES|SMARTS|pattern)\b", re.I)
_V_HETERO_N = re.compile(r"\b(nitrogen|amino|amine|aza|imino)\w*", re.I)
_V_HETERO_O = re.compile(r"\b(oxygen|ether|hydroxyl|epoxide)\w*", re.I)
_V_HETERO_S = re.compile(r"\b(sulfur|sulphur|thio|thia)\w*", re.I)


def _affirmative(text: str, m: "re.Match") -> bool:
    window = text[max(0, m.start() - 42):m.start()]
    window = re.split(r"[.;]", window)[-1]
    return _NEG.search(window) is None


def validate_text(text: str, an: dict) -> list[str]:
    """Check the polished description against the analysed facts and return the violations;
    an empty list means it passed."""
    # An acyclic functional-group skeleton is not subject to the ring-centred checks: with
    # zero rings the aromaticity branch misfires.
    if not text or not an.get("has_scaffold") or an.get("scaffold_kind") == "functional_group":
        return []
    v: list[str] = []
    rss = an["ring_systems"]
    # Heteroatom *presence* is judged over the whole scaffold, linkers included, not just
    # the rings: the N/O/S of a sulfonamide, amino or acylhydrazone linker really is there,
    # so claiming it is not a violation.
    scaf_elems: set = set()
    smol = Chem.MolFromSmiles(an.get("scaffold_smiles", "") or "")
    if smol is not None:
        scaf_elems = {a.GetSymbol() for a in smol.GetAtoms()}
    # Aromaticity is tallied per ring, so a partly aromatic fused system is reflected correctly
    tot_arom = sum(rs.get("n_aromatic_rings", rs["n_rings"] if rs["aromatic"] else 0) for rs in rss)
    tot_rings = sum(rs["n_rings"] for rs in rss)
    any_arom = tot_arom > 0
    any_aliph = tot_arom < tot_rings

    def _hit(rx):
        for m in rx.finditer(text):
            if _affirmative(text, m):
                return m
        return None

    # (1) aromaticity
    if not any_arom:
        m = _hit(_V_AROM)
        if m:
            v.append(f"claims aromatic/aryl ('{m.group(0)}') but every ring in the scaffold is "
                     "saturated (non-aromatic)")
    if not any_aliph:
        m = _hit(_V_SATUR)
        if m:
            v.append(f"calls a ring saturated/non-aromatic ('{m.group(0)}') but every ring in the "
                     "scaffold is aromatic")

    # (2) heteroatom presence: claiming an element that appears nowhere in the scaffold
    #     (rings + linkers) is a violation.
    if "N" not in scaf_elems:
        m = _hit(_V_HETERO_N)
        if m:
            v.append(f"mentions nitrogen ('{m.group(0)}') but the scaffold contains no nitrogen")
    if "O" not in scaf_elems:
        m = _hit(_V_HETERO_O)
        if m:
            v.append(f"mentions oxygen ('{m.group(0)}') but the scaffold contains no oxygen")
    if "S" not in scaf_elems:
        m = _hit(_V_HETERO_S)
        if m:
            v.append(f"mentions sulfur ('{m.group(0)}') but the scaffold contains no sulfur")

    # (3) ring-name swap: an affirmative mention of a ring name that is not in the allowed
    #     vocabulary. Checked only on scaffolds that are entirely monocyclic — fused systems
    #     have many legitimate composite-name synonyms (benzene+imidazole = benzimidazole,
    #     benzene+pyridine = isoquinoline/quinoline, ...). To avoid flagging those as swaps,
    #     the check is skipped as soon as any fused ring system is present. (The aromaticity
    #     and heteroatom-presence checks still apply to fused systems.)
    allowed = set()
    for rs in rss:
        for tok in re.findall(r"[a-z]{4,}", rs["name"].lower()):
            allowed.add(tok)
    for m in (_VOCAB_RE.finditer(text) if rss and all(rs["n_rings"] == 1 for rs in rss) else []):
        w = m.group(0).lower()
        # Strip the suffix, then map to the base name (phenyl -> benzene, ...) and compare stems.
        key = None
        for adj, b in _ADJ_TO_BASE.items():
            if w.startswith(adj):
                key = b
                break
        if key is None:
            # the vocab bases that w starts with
            for vb in _RING_VOCAB:
                if w.startswith(vb):
                    key = vb
                    break
        if key is None or key not in _RING_VOCAB:
            continue
        if key in allowed or any(key.startswith(a) or a.startswith(key) for a in allowed):
            continue
        if _affirmative(text, m):
            v.append(f"names a '{m.group(0)}' ring, but the scaffold contains no such ring "
                     f"(its ring systems are: {', '.join(rs['name'] for rs in rss)})")
            break

    # (4) charge / stereochemistry: outside the scope of a scaffold description, and neutral
    m = _hit(_V_CHARGE)
    if m:
        v.append(f"claims charge ('{m.group(0)}') but the scaffold is neutral")
    m = _hit(_V_STEREO)
    if m:
        v.append(f"claims stereochemistry ('{m.group(0)}') — do not assign configuration in a "
                 "scaffold/topology description")

    # (5) invented ring locants: the locants FACTS computed (asserted_locants) are allowed;
    #     any other specific ring number counts as a violation.
    asserted = set(an.get("asserted_locants") or [])
    for m in _V_LOCANT.finditer(text):
        num = next((int(g) for g in m.groups() if g and g.isdigit()), None)
        if num is not None and num not in asserted:
            v.append(f"states a ring locant ('{m.group(0).strip()}') not given by the FACTS "
                     f"(allowed: {sorted(asserted) or 'none'}) — do not invent ring numbers")
            break

    # (6) style: a purpose tail, or a meta reference
    m = _V_PURPOSE_TAIL.search(text)
    if m:
        v.append(f"contains a banned purpose tail ('{m.group(0).strip()}') — describe what the "
                 "scaffold IS, not what could later be done to it")
    m = _V_META.search(text)
    if m:
        v.append(f"refers to the query/string ('{m.group(0).strip()}') — describe the scaffold, "
                 "not 'the SMILES/SMARTS/pattern'")
    return v


# --------------------------------------------------------------------------- #
#  Public interface: the same input reproduces the same prompt and validation
# --------------------------------------------------------------------------- #
class ScaffoldDescriber:
    """SMILES scaffold -> the single source of FACTS, draft, prompt and validation."""

    @staticmethod
    def analyze(smiles: str) -> dict:
        return analyze_scaffold(smiles)

    @staticmethod
    def facts_text(an: dict) -> str:
        return format_facts(an)

    @staticmethod
    def draft(an: dict) -> str:
        return render_template(an)

    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def user_prompt(self, an: dict, draft: Optional[str] = None) -> str:
        return USER_TEMPLATE.format(facts=format_facts(an),
                                    draft=draft if draft is not None else render_template(an))

    def revise_prompt(self, an: dict, draft: str,
                      violations: Optional[list] = None) -> str:
        vlines = "\n".join(f"  - {x}" for x in (violations or [])) or "  - (unspecified)"
        return REVISE_TEMPLATE.format(facts=format_facts(an), violations=vlines, draft=draft)

    def condense_prompt(self, text: str, max_sentences: int) -> str:
        return CONDENSE_TEMPLATE.format(max_sentences=max_sentences, text=text)

    def validate(self, text: str, an: dict) -> list:
        return validate_text(text, an)


if __name__ == "__main__":
    import sys
    import json
    smis = sys.argv[1:] or ["c1ccc(Nc2ncnc3ccccc23)cc1"]
    d = ScaffoldDescriber()
    for smi in smis:
        an = d.analyze(smi)
        print("=" * 72)
        print("SMILES:", smi)
        print("\nFACTS:\n" + d.facts_text(an))
        print("\nDRAFT:\n" + d.draft(an))
        print("\nvalidate(draft):", d.validate(d.draft(an), an))
