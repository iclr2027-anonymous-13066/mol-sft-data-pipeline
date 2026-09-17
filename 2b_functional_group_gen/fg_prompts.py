"""fg_prompts.py  —  prompt content for functional-group descriptions.

Prompt content only. Structure analysis lives in ``fg_analyzer.py`` / ``fg_catalog.py``,
FACTS rendering and validation in ``fg_describer.py``, model calls in
``build_fg_catalog.py``. To change a prompt, edit this file and nothing else.

Prompt fg-v1
------------
A functional-group description is read BEFORE any structure exists, and many
structures satisfy it. So it is a brief, not a report — the same contract as the
scaffold prompt (v8), with three differences that come from what an fr_* pattern
actually is:

1. **The pattern, not the name, is the truth.** ``fr_N_O`` is called
   "hydroxylamine" but scores 0 on NH2OH and 2 on (CH3)2N-OCH3. The description
   must describe the FACTS block's bonded arrangement and exclusions, and must not
   promote the catalog name into a claim the pattern does not make.

2. **Counting is part of the constraint.** ``c1ccccc1`` scores naphthalene as 2.
   When the FACTS give a count rule, it belongs in the description, because the
   grader compares counts, not presence.

3. **exact vs at-least is not stylistic.** The scorer requires ``measured == count``
   for generation and ``measured >= count`` otherwise. The FACTS state which; the
   description must use the matching wording.

There is one description per catalog entry (61), not one per instance: the pattern
is fixed, so per-instance generation would add variance carrying no information.
"""

from __future__ import annotations


SYSTEM_PROMPT = """You write one-sentence-to-four-sentence briefs that tell a chemist which functional group a molecule must contain.

Your text is the INPUT to a structure-generation task. It is read before any molecule exists, and more than one molecule can satisfy it. Write the brief a chemist would state when requesting a structure, never a report about a finished object.

RULES
1. Describe the bonded arrangement the FACTS give: which atoms, joined by which bonds, carrying which hydrogens or charges, with which exclusions.
2. Say nothing the FACTS do not give. Leaving out a secondary detail is fine; adding one is not. In particular, do not restate the catalog NAME as if it were the definition — the pattern is often narrower or wider than its name.
3. If the FACTS list an exclusion, state it. If the FACTS give a counting rule, state it.
4. Do not mention SMARTS, SMILES, RDKit, patterns, matches, or fragment keys. Write chemistry, not tooling.
5. Uniform plain register. At most four sentences. ASCII only.
6. Do not open by naming the whole thing ("This functional group is ..."). Start from the atoms or the bond.
"""

USER_TEMPLATE = """COMPUTED FACTS (authoritative):
{facts}

DRAFT (deterministic wording reference, not a template to copy):
{draft}

Write the brief."""

REVISE_TEMPLATE = """COMPUTED FACTS (authoritative):
{facts}

The text below violates the facts or the rules:
{violations}

TEXT:
{draft}

Rewrite it so every violation is gone. Change nothing else."""


# --------------------------------------------------------------------------- #
#  Controlled-NL fragments used by the deterministic draft renderer.
# --------------------------------------------------------------------------- #
BOND_PHRASE = {
    "single": "a single bond",
    "double": "a double bond",
    "triple": "a triple bond",
    "aromatic": "an aromatic bond",
    "unspecified/any": "a bond of unspecified order",
}

MODE_PHRASE = {
    "exact": "exactly",
    "min": "at least",
}
