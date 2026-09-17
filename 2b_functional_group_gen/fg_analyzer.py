"""fg_analyzer.py  —  deterministic functional-group analysis for one record.

The functional-group counterpart of ``scaffold_analyzer.analyze_scaffold``: turns a
molecule (and, when the record already carries one, an authored constraint) into the
structural facts, the authoritative scoring query, and the verification flag.

    analyze_fg(smiles)                  -> full detected FG profile of the molecule
    build_constraint(profile, ...)      -> the FG subset that becomes the constraint
    build_eval_query(constraint, mode)  -> authoritative scorer query
    verify(smiles, eval_query)          -> does ref_smiles actually satisfy its own query?

Two constraint sources
----------------------
``answer``  the record already states the constraint (molkit's
            ``answer.fragments`` = ``[{"hydroxylamine": 1}]``). Use it verbatim; the
            molecule is only used for verification.
``derive``  the record has no FG constraint (generation_2m). Derive one from
            ``ref_smiles`` the way the scaffold pipeline derives a Murcko scaffold.

Count semantics
--------------
This dataset uses **min** semantics throughout: a member is satisfied when the
molecule contains AT LEAST the required number, and a derived member requires
exactly 1 — the group has to be present, and a reference that happens to carry three
copies does not make three a requirement. ``match_mode`` is carried per member so the
description can say "at least two" rather than "exactly two".

The current benchmark scorer does something different
(``evaluate_benchmark.check_individual_fragments_with_dist``)::

    task_type == "generation"   -> satisfied = (measured == count)     EXACT
    otherwise (optimization)    -> satisfied = (measured >= count)     MINIMUM

so scoring a min-mode dataset with that grader unchanged will fail every generation
row whose molecule has a spare copy. Either move its generation branch to ``>=``, or
build with ``--match-mode auto`` to reproduce the old behaviour.
"""

from __future__ import annotations

import hashlib
import os
import random
import sys
from typing import Optional

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from fg_catalog import (  # noqa: E402
    build_catalog, by_name, catalog_json_path, count_matches,
)
from fg_seed import build_seed  # noqa: E402


# --------------------------------------------------------------------------- #
#  Rarity — used to pick the most informative FG when deriving a constraint
# --------------------------------------------------------------------------- #
#  A molecule typically carries 5-15 detectable fr_* groups. "has a benzene ring"
#  constrains almost nothing; "has a beta lactam" constrains a great deal. Corpus
#  frequency is the honest proxy, and it is measured, not guessed — build_fg_catalog.py
#  writes it into the catalog JSON. Absent that file we fall back to a neutral
#  ordering so the pipeline still runs.
_RARITY: dict[str, float] = {}


def load_rarity(path: Optional[str] = None) -> dict[str, float]:
    """Load {fr_key: corpus document frequency} written by build_fg_catalog.py."""
    global _RARITY
    if _RARITY:
        return _RARITY
    path = catalog_json_path(path)
    try:
        import json
        with open(path) as fh:
            blob = json.load(fh)
        _RARITY = {k: float(v.get("doc_freq", 1.0))
                   for k, v in (blob.get("entries") or {}).items()}
    except Exception:  # noqa: BLE001
        _RARITY = {}
    return _RARITY


# --------------------------------------------------------------------------- #
#  Analysis
# --------------------------------------------------------------------------- #
def analyze_fg(smiles: str) -> dict:
    """Detect every scorer-visible functional group in *smiles*, with counts.

    Returns a dict shaped like the scaffold analyzer's output so the two datasets
    stay structurally comparable:

        parse_ok, smiles (canonical), n_functional_groups,
        functional_groups: [{name, fr_key, count, smarts}, ...]  (count-desc, name-asc)
        fg_summary: "two benzene rings, one amide, one nitrile"
    """
    out: dict = {"input_smiles": smiles, "parse_ok": False, "smiles": "",
                 "n_functional_groups": 0, "functional_groups": [], "fg_summary": ""}
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        out["analysis_error"] = "SMILES parse failed"
        return out

    out["parse_ok"] = True
    out["smiles"] = Chem.MolToSmiles(mol)

    found = []
    for fr_key, entry in build_catalog().items():
        n = count_matches(mol, entry["smarts"])
        if n > 0:
            found.append({"name": entry["name"], "fr_key": fr_key,
                          "count": n, "smarts": entry["smarts"]})
    found.sort(key=lambda d: (-d["count"], d["name"]))
    out["functional_groups"] = found
    out["n_functional_groups"] = len(found)
    out["fg_summary"] = summarize(found)
    return out


_NUMWORD = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
            6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}


def _num_word(n: int) -> str:
    return _NUMWORD.get(n, str(n))


def _plural(name: str, n: int) -> str:
    if n == 1:
        return name
    # "benzene ring" -> "benzene rings"; "carbonyl o" -> "carbonyl o groups"
    head = name.rsplit(" ", 1)
    if len(head) == 2 and head[1] in ("ring", "acid", "group", "amine", "ester", "ketone"):
        return f"{head[0]} {head[1]}s"
    return f"{name} groups"


def summarize(fgs: list[dict]) -> str:
    """'two benzene rings, one amide, one nitrile' — plain, no invented grouping."""
    parts = [f"{_num_word(f['count'])} {_plural(f['name'], f['count'])}" for f in fgs]
    if not parts:
        return "no scorer-visible functional group"
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + f", and {parts[-1]}"


# --------------------------------------------------------------------------- #
#  Constraint construction
# --------------------------------------------------------------------------- #
def match_mode_for(task_type: str, override: Optional[str] = None) -> str:
    """Scoring operator for the count: ``min`` (>=) or ``exact`` (==).

    Default is **min** — "the group has to be there", which is the semantics this
    dataset is built for. ``auto`` reproduces the current benchmark scorer instead
    (``exact`` for generation, ``min`` for optimization); note that a min-mode
    dataset and an ``auto`` grader disagree, so if these instances are scored by
    ``evaluate_benchmark.py`` its generation branch has to move to ``>=`` too.
    """
    if override and override != "auto":
        return override
    if override == "auto":
        return "exact" if (task_type or "").lower() == "generation" else "min"
    return "min"


def constraint_from_answer(fragments: list, task_type: str,
                           match_mode: Optional[str] = None) -> list[dict]:
    """Materialize an authored ``answer.fragments`` list into constraint members.

    Unknown names are kept with ``fr_key=None`` and no SMARTS rather than dropped —
    silently discarding a constraint would make the record look satisfiable when it
    is not.
    """
    cat = by_name()
    mode = match_mode_for(task_type, match_mode)
    members = []
    for frag in fragments or []:
        for name, count in (frag or {}).items():
            entry = cat.get(name)
            members.append({
                "name": name,
                "fr_key": entry["fr_key"] if entry else None,
                "count": int(count),
                "match_mode": mode,
                "smarts": entry["smarts"] if entry else None,
                "resolved": bool(entry),
            })
    return members


# fr_* entries that describe a metabolic liability SITE rather than a group a
# generator can be asked to install. They stay in the detected profile (the analysis
# should be honest about what the molecule contains) but must never become the
# constraint — "generate a molecule with exactly one aryl methyl site for
# hydroxylation" is not a chemistry brief. The benchmark's authored constraints
# never use them either.
DESCRIPTOR_FR_KEYS: frozenset = frozenset({
    "fr_allylic_oxid",         # allylic oxidation sites excluding steroid dienone
    "fr_aryl_methyl",          # aryl methyl sites for hydroxylation
})


def _row_rng(seed_text: str, seed: int) -> random.Random:
    """A per-row RNG keyed on the row itself, not on iteration order.

    Selection has to be reproducible AND resume-safe: a shard redone after an
    interrupted run must pick the same groups it picked the first time, which a
    shared global RNG would not guarantee.
    """
    h = hashlib.blake2b(f"{seed}:{seed_text}".encode(), digest_size=8).digest()
    return random.Random(int.from_bytes(h, "big"))


def constraint_from_molecule(profile: dict, task_type: str, n_fg_min: int = 1,
                             n_fg_max: int = 2, select: str = "random",
                             allow_descriptors: bool = False,
                             match_mode: Optional[str] = None,
                             require_count: Optional[int] = 1,
                             seed: int = 0) -> list[dict]:
    """Derive a constraint from a molecule's own FG profile.

    *select*:
      ``random``  uniformly sample between *n_fg_min* and *n_fg_max* distinct groups
                  from those the molecule actually contains (the default).
      ``rarest``  the least common groups in the corpus first (most informative).
      ``common``  the inverse (an easy-mode ablation).
      ``all``     every detected group.

    *require_count*: the number each member must occur at least/exactly. ``1`` (the
    default) means "the group has to be present" and deliberately ignores how many
    copies the reference molecule happens to have — a molecule with three benzene
    rings still only has to yield one. Pass ``None`` to require the molecule's own
    count instead.
    """
    fgs = profile.get("functional_groups") or []
    if not allow_descriptors:
        fgs = [f for f in fgs if f["fr_key"] not in DESCRIPTOR_FR_KEYS]
    if not fgs:
        return []

    if select == "all":
        chosen = list(fgs)
    elif select == "random":
        rng = _row_rng(profile.get("smiles") or profile.get("input_smiles") or "", seed)
        k = rng.randint(max(1, n_fg_min), max(1, n_fg_max))
        k = min(k, len(fgs))
        chosen = rng.sample(fgs, k)
        chosen.sort(key=lambda f: (-f["count"], f["name"]))     # stable output order
    else:
        rarity = load_rarity()
        # Unknown frequency sorts as "common" so a missing catalog never fabricates rarity.
        keyed = sorted(fgs, key=lambda f: (rarity.get(f["fr_key"], 1.0), f["name"]),
                       reverse=(select == "common"))
        chosen = keyed[:max(1, n_fg_max)]

    mode = match_mode_for(task_type, match_mode)
    return [{"name": f["name"], "fr_key": f["fr_key"],
             "count": f["count"] if require_count is None else int(require_count),
             "match_mode": mode, "smarts": f["smarts"], "resolved": True}
            for f in chosen]


def build_eval_query(members: list[dict]) -> dict:
    """Authoritative scoring query — every member must hold (AND).

    Deliberately mirrors the grader rather than re-inventing it: count the SMARTS
    with ``uniquify=True`` and compare with ``==`` (generation) or ``>=``
    (optimization). Verified count-identical to ``fr_*()`` on 1500 corpus molecules.
    """
    return {
        "mode": "count_all",
        "n_members": len(members),
        "members": [{"name": m["name"], "fr_key": m["fr_key"], "smarts": m["smarts"],
                     "op": "==" if m["match_mode"] == "exact" else ">=",
                     "count": m["count"], "uniquify": True}
                    for m in members],
    }


def satisfies(smiles: str, eval_query: dict) -> Optional[bool]:
    """Evaluate *eval_query* against *smiles*. None if unusable (bad SMILES/member)."""
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        return None
    members = (eval_query or {}).get("members") or []
    if not members:
        return None
    for m in members:
        if not m.get("smarts"):
            return None                      # unresolved name: cannot be judged
        n = count_matches(mol, m["smarts"])
        if m["op"] == "==" and n != m["count"]:
            return False
        if m["op"] == ">=" and n < m["count"]:
            return False
    return True


# --------------------------------------------------------------------------- #
#  One-shot entry point used by the driver
# --------------------------------------------------------------------------- #
def build_item(smiles: str, task_type: str = "generation",
               fragments: Optional[list] = None, fg_source: str = "auto",
               n_fg_min: int = 1, n_fg_max: int = 2, select: str = "random",
               allow_descriptors: bool = False, match_mode: Optional[str] = None,
               require_count: Optional[int] = 1, seed: int = 0,
               with_seed: bool = True) -> tuple[dict, dict]:
    """(analysis, item) for one record — the FG analogue of ``analyze_one``.

    *item* holds exactly the fields appended to the output record.
    """
    profile = analyze_fg(smiles)

    use_answer = (fg_source == "answer" or (fg_source == "auto" and fragments))
    if use_answer:
        members = constraint_from_answer(fragments or [], task_type, match_mode)
        source = "answer"
    else:
        members = constraint_from_molecule(
            profile, task_type, n_fg_min=n_fg_min, n_fg_max=n_fg_max, select=select,
            allow_descriptors=allow_descriptors, match_mode=match_mode,
            require_count=require_count, seed=seed)
        source = "derived"

    eq = build_eval_query(members)
    item = {
        "smiles": profile["smiles"],
        "input_smiles": profile["input_smiles"],
        "parse_ok": profile["parse_ok"],
        "has_fg": bool(members),
        "fg_source": source,
        "n_functional_groups": profile["n_functional_groups"],
        "functional_groups": profile["functional_groups"],
        "fg_constraint": members,
        # One SMARTS per member, deliberately NOT joined with '.'. A dot-joined query
        # requires the parts to map to ATOM-DISJOINT matches, so it rejects molecules
        # where the two groups legitimately share an atom — an amide whose N is also a
        # piperazine N satisfies both members but fails `C(=O)-N.N1CCNCC1`. Measured:
        # the joined form disagrees with eval_query on 13.1% of two-group rows.
        "fg_smarts": [m["smarts"] for m in members if m.get("smarts")],
        "n_fg_constraint": len(members),
        "eval_query": eq,
        "fg_summary": profile["fg_summary"],
    }
    if profile.get("analysis_error"):
        item["analysis_error"] = profile["analysis_error"]
    unresolved = [m["name"] for m in members if not m.get("resolved")]
    if unresolved:
        item["unresolved_fg"] = unresolved
    # Does the reference molecule satisfy the constraint it was given? For derived
    # constraints this is true by construction; for authored ones it is a real check
    # (an authored constraint can disagree with its own reference molecule).
    item["fg_verified"] = satisfies(profile["smiles"], eq)

    # Starting molecule for the toolchain: the smallest construction that already
    # carries every required group. Unlike a Murcko scaffold this cannot be read off
    # the SMARTS -- see fg_seed for why -- so it is built and then verified.
    if with_seed and members:
        seed_smiles, seed_source = build_seed(members)
        item["seed_smiles"] = seed_smiles
        item["seed_source"] = seed_source
        item["seed_verified"] = bool(seed_smiles) and satisfies(seed_smiles, eq)
        if seed_smiles:
            m = Chem.MolFromSmiles(seed_smiles)
            item["seed_heavy_atoms"] = m.GetNumHeavyAtoms() if m else None
    return profile, item


if __name__ == "__main__":
    import json
    smis = sys.argv[1:] or ["O=C(c1ccc2nccnc2c1)N1CCCN(CCn2ccnc2)CC1"]
    for smi in smis:
        _, it = build_item(smi, "generation")
        print("=" * 72)
        print(smi)
        print(json.dumps({k: v for k, v in it.items() if k != "functional_groups"},
                         indent=1, ensure_ascii=False))
        print("detected:", it["fg_summary"])
