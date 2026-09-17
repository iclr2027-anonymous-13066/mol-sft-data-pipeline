"""fg_seed.py  —  build a starting molecule that carries the required groups.

The functional-group counterpart of the scaffold pipeline's ``scaffold_smiles``, and
the one piece that does NOT transfer from it.

Why the scaffold recipe does not work here
------------------------------------------
A Murcko scaffold IS a molecule, so stage 4 can tell the model to *transcribe* the
committed SMARTS into SMILES and the two are the same object written twice. A
functional group is not:

    C(=O)-N              transcribed -> formamide, 3 heavy atoms
    C(=O)-N + N1CCNCC1   transcribed -> two DISCONNECTED fragments, not a molecule

So the seed has to be **constructed**: the smallest sensible molecule that contains
every required group. Three strategies, tried in order:

    carrier   one known molecule already contains all required groups
    joined    two carriers bonded together at a site where both groups survive
    (none)    no construction found — caller decides what to do

Every candidate is accepted only after RDKit confirms the finished molecule still
matches every member pattern. That matters more than it sounds: bonding to ethanol's
oxygen turns the hydroxyl into an ether and silently destroys the very group the seed
was built to carry.
"""

from __future__ import annotations

import os
import sys
from typing import Iterable, Optional

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from fg_catalog import PROBE_PANEL, build_catalog, count_matches  # noqa: E402


# --------------------------------------------------------------------------- #
#  Carrier pool
# --------------------------------------------------------------------------- #
#  The probe panel doubles as the carrier pool: it was already chosen to contain each
#  group in a small, unambiguous molecule. A few extra scaffolds are added purely as
#  join partners — they are large enough to have free positions in several places,
#  which the 3-atom carriers do not.
_EXTRA_CARRIERS: tuple[str, ...] = (
    "c1ccccc1", "C1CCCCC1", "CCCC", "c1ccncc1", "C1CCNCC1", "c1ccc2ccccc2c1",
)

_CARRIERS: Optional[dict[str, list[str]]] = None


def carriers(fr_key: str) -> list[str]:
    """Molecules that contain *fr_key*'s group, smallest first."""
    global _CARRIERS
    if _CARRIERS is None:
        cat = build_catalog()
        pool = [smi for _, smi in PROBE_PANEL] + list(_EXTRA_CARRIERS)
        mols = [(s, Chem.MolFromSmiles(s)) for s in pool]
        mols = [(s, m) for s, m in mols if m is not None]
        _CARRIERS = {}
        for key, entry in cat.items():
            hits = [(m.GetNumHeavyAtoms(), s) for s, m in mols
                    if count_matches(m, entry["smarts"])]
            hits.sort()
            _CARRIERS[key] = [s for _, s in hits]
    return _CARRIERS.get(fr_key, [])


# --------------------------------------------------------------------------- #
#  Verification
# --------------------------------------------------------------------------- #
def satisfies_members(smiles: str, members: Iterable[dict]) -> bool:
    """Does *smiles* meet every member's count requirement?"""
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        return False
    for m in members:
        sma = m.get("smarts")
        if not sma:
            return False
        n = count_matches(mol, sma)
        want = int(m.get("count", 1))
        if (n != want) if m.get("match_mode") == "exact" else (n < want):
            return False
    return True


# --------------------------------------------------------------------------- #
#  Joining two carriers
# --------------------------------------------------------------------------- #
_MAX_JOIN_TRIES = 400


def _open_atoms(mol: Chem.Mol) -> list[int]:
    """Heavy atoms with a free valence, carbons first.

    Carbons first because bonding to a heteroatom is what usually destroys the group
    (an alcohol O becomes an ether O, an amine N becomes an amide N); trying carbons
    first finds a surviving join sooner, and the verification below still has the
    final say.
    """
    idx = [(0 if a.GetSymbol() == "C" else 1, a.GetIdx())
           for a in mol.GetAtoms()
           if a.GetAtomicNum() > 1 and a.GetTotalNumHs() > 0]
    idx.sort()
    return [i for _, i in idx]


def join(smi_a: str, smi_b: str, members: Iterable[dict]) -> Optional[str]:
    """Bond *smi_a* to *smi_b* so that every member still holds. None if impossible."""
    a, b = Chem.MolFromSmiles(smi_a), Chem.MolFromSmiles(smi_b)
    if a is None or b is None:
        return None
    members = list(members)
    tries = 0
    for ia in _open_atoms(a):
        for ib in _open_atoms(b):
            tries += 1
            if tries > _MAX_JOIN_TRIES:
                return None
            combo = Chem.RWMol(Chem.CombineMols(a, b))
            try:
                combo.AddBond(ia, a.GetNumAtoms() + ib, Chem.BondType.SINGLE)
                mol = combo.GetMol()
                Chem.SanitizeMol(mol)
            except Exception:  # noqa: BLE001 — an impossible bond is just not a candidate
                continue
            smi = Chem.MolToSmiles(mol)
            if satisfies_members(smi, members):
                return smi
    return None


# --------------------------------------------------------------------------- #
#  Public entry point
# --------------------------------------------------------------------------- #
_SEED_CACHE: dict[tuple, tuple] = {}


def build_seed(members: list[dict]) -> tuple[Optional[str], str]:
    """(seed SMILES, strategy) for a constraint. Strategy is carrier/joined/none.

    The seed is the SMALLEST construction found, so the trajectory that decorates it
    has room to work; a seed that already satisfies the property box would make the
    rest of the task vacuous.

    Cached on the constraint SIGNATURE, not the row: the seed depends only on which
    groups are required and how many of each, so a 2M-row corpus needs at most
    61 + C(61,2) distinct constructions. Measured on 20k rows: 593 signatures, 1.5 ms
    each — the per-row cost after warm-up is a dict lookup.
    """
    members = [m for m in members if m.get("smarts")]
    if not members:
        return None, "none"

    key = tuple(sorted((m["fr_key"], int(m.get("count", 1)),
                        m.get("match_mode", "min")) for m in members))
    hit = _SEED_CACHE.get(key)
    if hit is not None:
        return hit
    out = _build_seed_uncached(members)
    _SEED_CACHE[key] = out
    return out


def _build_seed_uncached(members: list[dict]) -> tuple[Optional[str], str]:

    # 1. A single carrier that happens to hold every required group.
    pool: list[str] = []
    for m in members:
        pool.extend(carriers(m["fr_key"]))
    seen, best = set(), None
    for smi in pool:
        if smi in seen:
            continue
        seen.add(smi)
        if satisfies_members(smi, members):
            mol = Chem.MolFromSmiles(smi)
            n = mol.GetNumHeavyAtoms()
            if best is None or n < best[0]:
                best = (n, smi)
    if best:
        return best[1], "carrier"

    if len(members) == 1:
        return None, "none"          # nothing to join a lone group to

    # 2. Bond two carriers, smallest partners first, keeping the smallest survivor.
    best_join = None
    for sa in carriers(members[0]["fr_key"])[:6]:
        for sb in carriers(members[1]["fr_key"])[:6]:
            smi = join(sa, sb, members)
            if not smi:
                continue
            n = Chem.MolFromSmiles(smi).GetNumHeavyAtoms()
            if best_join is None or n < best_join[0]:
                best_join = (n, smi)
    if best_join:
        return best_join[1], "joined"
    return None, "none"


if __name__ == "__main__":
    import json

    cat = build_catalog()
    by_name = {e["name"]: e for e in cat.values()}
    names = sys.argv[1:] or ["amide", "piperzine ring"]
    members = [{"name": n, "fr_key": by_name[n]["fr_key"],
                "smarts": by_name[n]["smarts"], "count": 1, "match_mode": "min"}
               for n in names if n in by_name]
    smi, how = build_seed(members)
    print(json.dumps({"members": [m["name"] for m in members],
                      "seed_smiles": smi, "seed_source": how,
                      "verified": satisfies_members(smi or "", members)}, indent=1))
