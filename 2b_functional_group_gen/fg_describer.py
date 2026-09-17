"""fg_describer.py  —  FACTS block, deterministic draft, validation for FG constraints.

The functional-group counterpart of ``scaffold_describer``: a single source for
(FACTS, draft, prompts, validation) so the same input always yields the same
contract, whether the text is produced by the LLM or by the template alone.

    format_facts(entry)         -> authoritative FACTS block for one catalog entry
    render_template(entry)      -> deterministic controlled-NL draft
    validate_text(text, entry)  -> list of violations (empty = passes)
    render_constraint(members, catalog) -> the per-instance sentence

Validation is stronger here than on the scaffold side, because a functional group
has an exact oracle: the probe panel. If the text claims a probe molecule contains
the group when the pattern scores it 0 (or denies one the pattern scores nonzero),
that is a measurable contradiction, not a stylistic one.
"""

from __future__ import annotations

import os
import re
import sys
import unicodedata
from typing import Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from fg_prompts import (  # noqa: E402
    BOND_PHRASE, MODE_PHRASE, REVISE_TEMPLATE, SYSTEM_PROMPT, USER_TEMPLATE,
)


# --------------------------------------------------------------------------- #
#  Normalisation — identical intent to scaffold_describer.normalize_text
# --------------------------------------------------------------------------- #
_PUNCT_FOLD = {
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-",
    "―": "-", "−": "-", "‘": "'", "’": "'", "“": '"',
    "”": '"', " ": " ", " ": " ", " ": " ",
}


def normalize_text(text: str) -> str:
    """NFKC + explicit punctuation folding + whitespace collapse."""
    if not text:
        return ""
    out = unicodedata.normalize("NFKC", text)
    for bad, good in _PUNCT_FOLD.items():
        out = out.replace(bad, good)
    out = re.sub(r"\s+", " ", out).strip()
    return out


# --------------------------------------------------------------------------- #
#  FACTS
# --------------------------------------------------------------------------- #
def _elements_phrase(elements: dict[str, int]) -> str:
    parts = []
    for sym, n in sorted(elements.items(), key=lambda kv: (-kv[1], kv[0])):
        label = "any-atom slot" if sym == "*" else sym
        parts.append(f"{n}x {label}")
    return ", ".join(parts) or "(none)"


_BOND_PLURAL = {
    "single": "single bonds", "double": "double bonds", "triple": "triple bonds",
    "aromatic": "aromatic bonds", "unspecified/any": "bonds of unspecified order",
}


def _bonds_phrase(bonds: dict[str, int]) -> str:
    parts = []
    for w, n in sorted(bonds.items()):
        parts.append(BOND_PHRASE.get(w, w) if n == 1
                     else f"{_word(n)} {_BOND_PLURAL.get(w, w + ' bonds')}")
    return ", ".join(parts) or "(no bond constraint)"


_ELEMENT_NAME = {
    "C": "carbon", "N": "nitrogen", "O": "oxygen", "S": "sulfur", "P": "phosphorus",
    "F": "fluorine", "Cl": "chlorine", "Br": "bromine", "I": "iodine", "B": "boron",
    "Si": "silicon", "Se": "selenium",
}


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s


def _atoms_prose(elements: dict[str, int], aromatic: dict[str, int] | None = None) -> str:
    """{'C':3,'N':1,'O':1} -> 'three carbons, one nitrogen and one oxygen'.

    *aromatic* marks how many atoms of each element the pattern requires to be
    aromatic, so ``c[OH1]`` reads as "one aromatic carbon", not "one carbon".
    """
    aromatic = aromatic or {}
    parts = []
    for sym, n in sorted(elements.items()):
        if sym == "*":
            continue
        word = _ELEMENT_NAME.get(sym, sym)
        n_ar = aromatic.get(sym, 0)
        if n_ar >= n:
            word = f"aromatic {word}"
        elif n_ar:
            word = f"{word} ({_word(n_ar)} of them aromatic)"
        parts.append(f"{_word(n)} {word}" if n == 1 else f"{_word(n)} {word}s")
    if not parts:
        return "unconstrained atoms"
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + f" and {parts[-1]}"


def _atom_requirement_lines(comp: dict) -> list[str]:
    """H / aromaticity / charge requirements — the facts that separate a hydroxyl
    from an ether and an aromatic ring atom from an aliphatic one."""
    out: list[str] = []
    hyd = comp.get("hydrogens") or {}
    if hyd:
        parts = [f"the {_ELEMENT_NAME.get(s, s)} must carry "
                 f"{'a hydrogen' if n == 1 else f'{_word(n)} hydrogens'}"
                 for s, n in sorted(hyd.items())]
        out.append("- Hydrogen requirement: " + "; ".join(parts)
                   + " (so an alkylated or acylated variant does NOT count).")
    arom = comp.get("aromatic") or {}
    if arom:
        parts = [f"{_word(n)} aromatic {_ELEMENT_NAME.get(s, s)}"
                 + ("" if n == 1 else "s") for s, n in sorted(arom.items())]
        out.append("- Aromaticity requirement: " + ", ".join(parts)
                   + "; the corresponding saturated atom does NOT count.")
    chg = comp.get("charges") or {}
    if chg:
        parts = [f"the {_ELEMENT_NAME.get(s, s)} carries a formal charge of {c:+d}"
                 for s, c in sorted(chg.items())]
        out.append("- Charge requirement: " + "; ".join(parts) + ".")
    return out


def format_facts(entry: dict) -> str:
    """Catalog entry -> COMPUTED FACTS text for the prompt.

    Everything on these lines is measured or parsed. The catalog NAME is given last
    and explicitly flagged as a label, so the model cannot treat it as a definition.
    """
    comp = entry.get("composition") or {}
    lines = [
        f"- Atoms the group must contain: {_elements_phrase(comp.get('elements') or {})}",
        f"- Bonds between them: {_bonds_phrase(comp.get('bonds') or {})}",
    ]
    for extra in _atom_requirement_lines(comp):
        lines.append(extra)
    rings = comp.get("ring_sizes") or []
    if rings:
        sizes = ", ".join(f"{s}-membered" for s in rings)
        lines.append(f"- Ring requirement: the group is itself a ring ({sizes}).")
    else:
        lines.append("- Ring requirement: none; the group may sit in a chain or on a ring.")

    for ex in entry.get("exclusions") or []:
        lines.append(f"- EXCLUSION: {_exclusion_prose(ex)} does NOT count. This is part of "
                     f"the definition and must be stated.")

    pos = entry.get("probe_positive") or {}
    if pos:
        shown = sorted(pos.items(), key=lambda kv: (-kv[1], kv[0]))[:8]
        lines.append("- Reference molecules that DO contain it, with how many are counted: "
                     + "; ".join(f"{n} -> {c}" for n, c in shown))
        multi = [f"{n} counts as {c}" for n, c in shown if c > 1]
        if multi:
            lines.append("- COUNTING RULE (state this): " + "; ".join(multi)
                         + ". Every distinct occurrence is counted separately.")
    neg = _informative_negatives(entry)
    if neg:
        lines.append("- Reference molecules that do NOT contain it (near misses worth "
                     "distinguishing): " + ", ".join(neg))

    lines.append(f"- Catalog label (a name only, NOT a definition): \"{entry.get('name', '')}\".")
    lines.append("- Do NOT state how many occurrences are required. That number varies per "
                 "task and is appended separately.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
#  Exclusion clauses -> prose (+ the tokens that count as stating them)
# --------------------------------------------------------------------------- #
#  Only six of the 61 patterns carry a `!$(...)` clause, and between them they use
#  the handful of forms below. Rendering them as chemistry (rather than echoing the
#  raw clause) keeps the description free of pattern syntax, which rule 4 forbids.
_EXCLUSION_PROSE: dict[str, tuple[str, tuple[str, ...]]] = {
    "C=O":  ("a carbon that is part of a carbonyl", ("carbonyl", "c=o")),
    "N=O":  ("a nitrogen double-bonded to oxygen, as in a nitroso or nitro group",
             ("nitroso", "nitro", "n=o", "double-bonded to oxygen")),
    "N-O":  ("a nitrogen that already carries a single-bonded oxygen",
             ("single-bonded oxygen", "n-o", "hydroxylamine", "already carries")),
    "N=*":  ("a nitrogen double-bonded to anything, as in an imine",
             ("imine", "double-bonded", "n=")),
}


def _exclusion_prose(clause: str) -> str:
    """Plain-language form of a `!$(...)` clause; falls back to the clause itself."""
    known = _EXCLUSION_PROSE.get(clause)
    if known:
        return known[0]
    return f"an atom in the arrangement {clause}"


def _exclusion_tokens(clause: str) -> tuple[str, ...]:
    """Lowercase substrings any of which counts as having stated *clause*."""
    known = _EXCLUSION_PROSE.get(clause)
    return known[1] if known else (clause.lower(),)


# Near-miss negatives worth naming, per group family. A probe that scores 0 is only
# interesting if a reader would plausibly expect it to score >0.
_NEAR_MISS_HINTS = {
    "fr_lactam":   ["2-pyrrolidinone"],
    "fr_N_O":      ["hydroxylamine", "O-methylhydroxylamine", "nitrosobenzene"],
    "fr_Al_OH":    ["acetic acid", "phenol"],
    "fr_phenol":   ["2-naphthol", "4-hydroxypyridine"],
    "fr_Ar_OH":    ["ethanol"],
    "fr_ester":    ["acetic acid", "methyl carbamate"],
    "fr_ketone":   ["acetaldehyde", "acetic acid"],
    "fr_aldehyde": ["acetone", "formic acid"],
    "fr_COO2":     ["methyl acetate", "acetamide"],
    "fr_amide":    ["methyl acetate"],
    "fr_benzene":  ["pyridine", "cyclohexane"],
    "fr_nitroso":  ["nitrobenzene"],
    "fr_nitro":    ["nitrosobenzene"],
    "fr_sulfone":  ["dimethyl sulfoxide", "dimethyl sulfide"],
    "fr_sulfide":  ["dimethyl sulfone"],
}


def _informative_negatives(entry: dict) -> list[str]:
    """Named probes that score 0 but that a reader would expect to score >0."""
    probes = entry.get("probes") or {}
    hints = _NEAR_MISS_HINTS.get(entry.get("fr_key", ""), [])
    return [h for h in hints if probes.get(h, 0) == 0]


# --------------------------------------------------------------------------- #
#  Deterministic draft
# --------------------------------------------------------------------------- #
def render_template(entry: dict) -> str:
    """Faithful controlled-NL draft, built only from the FACTS above.

    Used as the LLM's wording reference, and as the description itself whenever no
    model is available — so it must stand on its own.
    """
    comp = entry.get("composition") or {}
    elements = comp.get("elements") or {}
    rings = comp.get("ring_sizes") or []
    name = entry.get("name", "the group")

    arom = comp.get("aromatic") or {}
    if rings:
        ring_el = comp.get("ring_elements") or elements
        pend_el = comp.get("pendant_elements") or {}
        head = (f"A {rings[0]}-membered ring made up of "
                f"{_atoms_prose(ring_el, arom)} is required")
        if pend_el:
            head += f", carrying {_atoms_prose(pend_el, arom)} attached to it"
    elif comp.get("n_atoms", 0) <= 1:
        # Single-atom pattern: drop the count word ("one sulfur" -> "a sulfur"), but
        # keep aromaticity — `n` is an AROMATIC nitrogen, and dropping that turns the
        # constraint into "any nitrogen".
        sym = next((s for s in elements if s != "*"), "")
        word = _ELEMENT_NAME.get(sym, sym) or "atom"
        if arom.get(sym):
            word = f"aromatic {word}"
        head = f"A single {word} is required"
    else:
        head = (f"{_cap(_atoms_prose(elements, arom))} joined by "
                f"{_bonds_phrase(comp.get('bonds') or {})}")

    sents = [head + "."]

    hyd = comp.get("hydrogens") or {}
    if hyd:
        parts = [f"the {_ELEMENT_NAME.get(s, s)} must carry "
                 f"{'a hydrogen' if n == 1 else f'{_word(n)} hydrogens'}"
                 for s, n in sorted(hyd.items())]
        sents.append(_cap("; ".join(parts)) + ", so an alkylated or acylated form "
                     "does not qualify.")
    chg = comp.get("charges") or {}
    if chg:
        parts = [f"the {_ELEMENT_NAME.get(s, s)} carries a formal charge of {c:+d}"
                 for s, c in sorted(chg.items())]
        sents.append(_cap("; ".join(parts)) + ".")

    for ex in entry.get("exclusions") or []:
        sents.append(f"{_exclusion_prose(ex).capitalize()} does not count toward it.")

    return " ".join(sents)


def counting_note(entry: dict) -> str:
    """How multiple occurrences are counted, or "" when the panel shows no case.

    Kept OUT of the definition and added only when the requirement actually depends
    on the number (count > 1, or exact mode). Under the usual "at least one" it is
    dead weight: whether naphthalene counts as one benzene ring or two changes
    nothing once one is already enough.
    """
    pos = entry.get("probe_positive") or {}
    multi = sorted(((n, c) for n, c in pos.items() if c > 1), key=lambda kv: -kv[1])
    if not multi:
        return ""
    n0, c0 = multi[0]
    return f"Occurrences are counted separately, so {n0} counts as {c0}."


def needs_count_sentence(count: int, match_mode: str) -> bool:
    """Is the requirement clause carrying information?

    "at least one of them" is the default reading of any constraint, so stating it
    adds nothing — and the scaffold descriptions this dataset sits beside carry no
    requirement sentence at all. Only a number above one, or an exact-count rule,
    changes what the reader has to do.
    """
    return count > 1 or match_mode == "exact"


def count_sentence(count: int, match_mode: str = "exact") -> str:
    """The per-record requirement sentence.

    Deliberately NOT part of the catalog description and never model-generated: it
    is the one clause the grader compares against, and 'exactly two' vs 'at least
    two' decides whether an answer passes. Generated from (count, match_mode) so it
    cannot drift from ``eval_query``.
    """
    noun = "one of them" if count == 1 else f"{_word(count)} of them"
    return f"The molecule must contain {MODE_PHRASE.get(match_mode, match_mode)} {noun}."


_NUMWORD = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
            6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}


def _word(n: int) -> str:
    return _NUMWORD.get(n, str(n))


# --------------------------------------------------------------------------- #
#  Validation
# --------------------------------------------------------------------------- #
_TOOLING = re.compile(r"\b(SMARTS|SMILES|RDKit|substructure|fr_[A-Za-z_]+|pattern|regex)\b",
                      re.IGNORECASE)
_OPENER = re.compile(r"^\s*(this|the)\s+(functional\s+group|group|fragment|moiety|structure)\b",
                     re.IGNORECASE)


def validate_text(text: str, entry: dict, count: Optional[int] = None,
                  match_mode: Optional[str] = None) -> list[str]:
    """Compare a description against the catalog facts. Empty list = passes.

    *count* / *match_mode* are given only when validating an ASSEMBLED per-record
    brief (definition + count sentence). The catalog definition itself is count-free
    by design, so they are omitted there and the count checks are skipped.
    """
    if not text:
        return []
    v: list[str] = []
    low = text.lower()

    # (1) Tooling leakage — the description is chemistry, not a spec of the matcher.
    m = _TOOLING.search(text)
    if m:
        v.append(f"mentions tooling ('{m.group(0)}'); describe the chemistry instead")

    # (2) Banned opener: pointing at the whole object instead of the atoms.
    if _OPENER.match(text):
        v.append("opens by naming the whole group; start from the atoms or the bond")

    # (3a) A catalog DEFINITION (no count/mode given) must be count-free and
    #      mode-neutral. The requirement clause is generated per record, so a wording
    #      that bakes in "exactly one" is wrong for every row that needs "at least
    #      two" — and the per-record validator would only catch it downstream, after
    #      it had already shipped. Reject it here instead.
    if match_mode is None and count is None:
        # Quantifier words only. "of them" is NOT a signal — `_atoms_prose` writes
        # "two carbons (two of them aromatic)", which is composition, not a requirement.
        m = re.search(r"\b(exactly|precisely|at least|at most|no more than|"
                      r"one or more|once|twice)\b", low)
        if m:
            v.append(f"states a requirement ('{m.group(0)}'); the definition must be "
                     "count-free — the count clause is added per record")
        # A garbled sample (high-temperature runs produce them) is not worth keeping.
        words = low.split()
        if not text.strip().endswith(".") or not 4 <= len(words) <= 80:
            v.append("malformed: a definition is 4-80 words and ends in a period")

    # (3) exact/at-least must match the scorer's operator, and the stated number must
    #     be the required one. Only meaningful on an assembled per-record brief.
    if match_mode is not None:
        wants_exact = match_mode == "exact"
        says_atleast = re.search(r"\b(at least|one or more|minimum of)\b", low) is not None
        says_exact = re.search(r"\b(exactly|precisely)\b", low) is not None
        if wants_exact and says_atleast:
            v.append("says 'at least' but this task scores an EXACT count (measured == required)")
        if not wants_exact and says_exact:
            v.append("says 'exactly' but this task scores a MINIMUM count (measured >= required)")
        if wants_exact and not says_exact:
            v.append("does not state that the count must be exact")
    if count is not None:
        want = _word(count)
        stated = re.search(r"\b(?:exactly|at least|precisely)\s+([a-z]+|\d+)\b", low)
        if stated and stated.group(1) not in (want, str(count)):
            v.append(f"states '{stated.group(1)}' occurrence(s) but the constraint requires "
                     f"{want} ({count})")

    # (4) Stated exclusions must survive into the text.
    for ex in entry.get("exclusions") or []:
        if ex not in _EXCLUSION_PROSE:
            continue          # metabolic-site clauses have no compact prose form
        if not any(tok in low for tok in _exclusion_tokens(ex)):
            v.append(f"omits the exclusion ({_exclusion_prose(ex)} does not count), "
                     f"which is part of the definition")

    # (5) Probe contradictions — the measurable check. If the text names a probe the
    #     pattern scores 0 as an example of the group (or vice versa), that is wrong.
    probes = entry.get("probes") or {}
    # A probe molecule can share the catalog label ("hydroxylamine" names both the
    # group and NH2OH, which the pattern scores 0). Using the label is not a probe
    # claim, so exclude it here or every such entry self-flags.
    own_label = (entry.get("name") or "").lower()
    for pname, pcount in probes.items():
        # Whole-name match only: 'pyridine' is a substring of '1,4-dihydropyridine',
        # and substring matching would report a claim the text never made.
        if len(pname) < 6 or not re.search(rf"(?<![\w-]){re.escape(pname)}(?![\w-])", low):
            continue
        if pname == own_label or pname in own_label or own_label in pname:
            continue
        negated = re.search(rf"\b(not|never|excludes?|unlike)\b[^.]{{0,40}}{re.escape(pname)}",
                            low) is not None
        if pcount == 0 and not negated:
            v.append(f"names '{pname}' as an example, but the group does not occur in it")
        if pcount > 0 and negated:
            v.append(f"denies '{pname}', but the group does occur in it ({pcount}x)")

    # (6) Counting rule: if a probe counts >1, the text must not imply one-per-molecule.
    multi = [n for n, c in probes.items() if c > 1]
    if multi and re.search(r"\b(only one|a single|exactly one) (?:such )?(?:group|ring|occurrence)"
                           r" (?:can|may) (?:be present|occur)\b", low):
        v.append("implies at most one occurrence per molecule, but occurrences are "
                 "counted separately")
    return v


# --------------------------------------------------------------------------- #
#  Name-style brief — "the molecule must contain a pyridine ring"
# --------------------------------------------------------------------------- #
#  The definition style spells the group out atom by atom, which is what makes a
#  brief verifiable. But a chemist asking for a compound usually just names the
#  group, and a model trained only on spelled-out definitions never sees the plain
#  request it will actually be given. Both styles are emitted.

_VOWELS = "aeiou"
# Names whose article does not follow the spelling ("a urea", not "an urea").
_ARTICLE_OVERRIDE = {"urea": "a", "oxime": "an"}


def _article(phrase: str) -> str:
    head = (phrase or "").split()[0].lower() if phrase else ""
    if head in _ARTICLE_OVERRIDE:
        return _ARTICLE_OVERRIDE[head]
    return "an" if head[:1] in _VOWELS else "a"


_PLURAL_TAIL = ("ring", "acid", "group", "amine", "ester", "ketone", "aldehyde",
                "amide", "site", "cage", "oxygen", "nitrogen")


def _plural_name(name: str, n: int) -> str:
    """'benzene ring' -> 'benzene rings'; anything else -> '<name> groups'."""
    if n == 1:
        return name
    head = name.rsplit(" ", 1)
    if len(head) == 2 and head[1] in _PLURAL_TAIL:
        return f"{head[0]} {head[1]}s"
    if name.endswith(_PLURAL_TAIL):
        return f"{name}s"
    return f"{name} groups"


# Openers for the name style. Pure templating — no model needed — so the plainest
# phrasing, which is also the most common one a user would type, still varies.
NAME_STYLE_OPENERS: tuple[str, ...] = (
    "The molecule must contain {x}.",
    "The molecule should contain {x}.",
    "Generate a molecule containing {x}.",
    "The compound needs to carry {x}.",
    "Include {x} in the molecule.",
    "The structure must feature {x}.",
)


def render_name_style(members: list[dict], opener: int = 0) -> str:
    """"The molecule must contain a benzene ring and at least two amides."."""
    if not members:
        return ""
    parts = []
    for m in members:
        name = m.get("name", "group")
        count = m.get("count", 1)
        mode = m.get("match_mode", "min")
        if count == 1 and mode == "min":
            parts.append(f"{_article(name)} {name}")
        elif mode == "exact":
            parts.append(f"exactly {_word(count)} {_plural_name(name, count)}")
        else:
            parts.append(f"at least {_word(count)} {_plural_name(name, count)}")
    if len(parts) == 1:
        joined = parts[0]
    elif len(parts) == 2:
        joined = f"{parts[0]} and {parts[1]}"
    else:
        joined = ", ".join(parts[:-1]) + f", and {parts[-1]}"
    return NAME_STYLE_OPENERS[opener % len(NAME_STYLE_OPENERS)].format(x=joined)


# --------------------------------------------------------------------------- #
#  Per-instance sentence
# --------------------------------------------------------------------------- #
def render_constraint(members: list[dict], catalog: dict,
                      descriptions: dict[str, str] | None = None) -> str:
    """Assemble a record's brief: catalog definition + this record's count sentence.

    Per-instance text is assembled, never regenerated. The pattern is fixed (so the
    definition comes from the catalog) and the only per-instance facts are *which*
    groups and *how many* — and the count clause is generated here rather than
    stored, so it always agrees with ``eval_query``.
    """
    return " ".join(t for _, t in render_constraint_parts(members, catalog, descriptions)).strip()


def render_constraint_parts(members: list[dict], catalog: dict,
                            descriptions: dict[str, str] | None = None
                            ) -> list[tuple[dict, str]]:
    """``[(member, its sentence), ...]`` — the pieces ``render_constraint`` joins.

    Kept separate because validation is per member: a two-group brief legitimately
    cites one probe molecule for the first group and another for the second, and
    checking the whole text against either group's probe table alone reports the
    other group's citation as a contradiction.
    """
    descriptions = descriptions or {}
    parts: list[tuple[dict, str]] = []
    for m in members:
        fr = m.get("fr_key")
        count = m.get("count", 1)
        mode = m.get("match_mode", "exact")
        entry = catalog.get(fr) if fr else None
        definition = (descriptions.get(fr) if fr else None) or (
            render_template(entry) if entry else "")
        if definition:
            # The counting rule and the requirement clause only matter when the
            # number does; "at least one" is the default reading and stating it
            # adds nothing (the scaffold descriptions carry no such clause either).
            want_count = needs_count_sentence(count, mode)
            note = counting_note(entry) if (entry and want_count) else ""
            sents = [definition.strip()] + ([note] if note else []) \
                + ([count_sentence(count, mode)] if want_count else [])
            parts.append((m, " ".join(sents)))
        else:
            # Unresolved name: name the requirement rather than inventing chemistry.
            parts.append((m, render_name_style([m])))
    return parts


# --------------------------------------------------------------------------- #
#  Public interface — same input, same prompts/validation
# --------------------------------------------------------------------------- #
class FGDescriber:
    """Catalog entry -> (FACTS · draft · prompts · validation)."""

    @staticmethod
    def facts_text(entry: dict) -> str:
        return format_facts(entry)

    @staticmethod
    def draft(entry: dict) -> str:
        return render_template(entry)

    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def user_prompt(self, entry: dict, draft: str | None = None) -> str:
        return USER_TEMPLATE.format(
            facts=format_facts(entry),
            draft=draft if draft is not None else render_template(entry))

    def revise_prompt(self, entry: dict, draft: str,
                      violations: list | None = None) -> str:
        vlines = "\n".join(f"  - {x}" for x in (violations or [])) or "  - (unspecified)"
        return REVISE_TEMPLATE.format(facts=format_facts(entry),
                                      violations=vlines, draft=draft)

    def validate(self, text: str, entry: dict, count=None, match_mode=None) -> list:
        return validate_text(text, entry, count, match_mode)


if __name__ == "__main__":
    from fg_catalog import build_catalog
    cat = build_catalog()
    d = FGDescriber()
    for key in (sys.argv[1:] or ["fr_N_O", "fr_benzene", "fr_lactam"]):
        e = cat[key]
        print("=" * 72)
        print(f"{key}  \"{e['name']}\"")
        print("\nFACTS:\n" + d.facts_text(e))
        print("\nDEFINITION:\n" + d.draft(e))
        print("\nASSEMBLED (exact 1):\n" + d.draft(e) + " " + count_sentence(1, "exact"))
        print("\nvalidate(definition):", d.validate(d.draft(e), e))
