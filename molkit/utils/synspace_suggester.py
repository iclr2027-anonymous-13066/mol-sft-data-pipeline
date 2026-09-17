"""Forward-synthesis edit vocabulary — a ``suggest_edits`` ablation arm built on synspace.

The sibling module :mod:`molkit.utils.random_suggesters` keeps ``suggest_edits``'
enumeration step and throws its ranking away, drawing uniformly from a fixed,
property-blind vocabulary. This module does the same thing with a *different*
vocabulary: instead of "attach one atom" or "attach one group from a catalog", the
moves are **one robust medicinal-chemistry reaction with one purchasable building
block**, enumerated by `synspace <https://github.com/whitead/synspace>`_ (Hartenfeller's
58 robust reactions x ~1.08 M Mcule blocks, REOS-filtered).

Why it is worth an arm: every other vocabulary here is synthesis-blind, and mmpdb's is
a statistical prior over *observed* pairs. synspace's moves are synthesizable by
construction — a reaction that is known to run, plus a block someone sells — so this arm
measures what a synthesizability constraint costs (or buys) at a fixed edit budget.

Two modes:

=================  ==========================================================
``synspace``       forward synthesis only (``steps=(0, 1)``): decorate the
                   molecule with a block wherever a reactant template matches
``synspace_retro`` one retro step first (``steps=(1, 1)``): cut the molecule
                   into precursors, then decorate those. Reaches swap-like
                   moves the forward-only mode cannot, at ~10x the latency
=================  ==========================================================

Everything downstream of the vocabulary is shared with the other arms ON PURPOSE:

* the guard is the same test on the same object (``HasSubstructMatch`` on the product,
  every pattern required separately);
* candidates come back in the same ``edit_fragment``-ready shape (``from_smiles`` /
  ``to_smiles`` / ``anchors``) pinned to one site, and re-applying one with
  ``replace_fragment(anchors=...)`` reproduces the product — **verified per candidate**
  here rather than assumed, because these args are DERIVED from synspace's product
  rather than being the rule that made it (see :func:`_as_edit`);
* ``predicted_gap`` is ``None`` and ``constraints`` is ignored: property-blind, so a
  caller that wants to know whether a candidate helps has to measure it.

Two properties of the vocabulary are worth knowing before reading results:

1. **It does not always fire.** A reaction needs a handle (an amine, an acid, an aryl
   halide, ...). Measured on 300 seeds each, a guard-passing candidate exists for 25%
   (benchmark_fg) / 24% (generation_test_1k) of seeds in ``synspace`` mode and
   41% / 37% in ``synspace_retro`` mode. Where mmpdb offers a move on essentially every
   molecule, this arm runs dry on most, and no pick rule can repair that.
2. **Moves are large and one-directional**: a candidate adds a whole building block
   (median MW +125..139), so there is no fine adjustment and no shrink move.

synspace is an optional dependency. It is imported lazily and
:func:`enumerate_candidates` raises :class:`SynspaceUnavailable` if it is missing, so
importing this module never breaks a harness that does not use these arms.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple, Union

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

MODES = ("synspace", "synspace_retro")

# steps= argument to synspace.chemical_space per mode: (retro steps, forward steps).
_STEPS: Dict[str, Tuple[int, int]] = {"synspace": (0, 1), "synspace_retro": (1, 1)}

# Blocks sampled per reaction template slot, and the cap on what one call returns.
# synspace's own defaults (25 / 250). The similarity threshold is deliberately 0: its
# default of 0.2 is tuned for drug-sized inputs and silently discards nearly every
# candidate on the small fragment seeds of benchmark_fg (measured: mean 0.9
# candidates at 0.2 vs 39.5 at 0.0), which would confound "the vocabulary does not
# apply" with "the similarity filter rejected it".
NBLOCKS = 25
NUM_SAMPLES = 250
THRESHOLD = 0.0
POOL_CAP = 400

Guard = Optional[Union[str, Sequence[str]]]

_OPEN = "[*:1]"


class SynspaceUnavailable(RuntimeError):
    """Raised when the optional ``synspace`` dependency is not importable."""


def _chemical_space():
    try:
        import synspace
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise SynspaceUnavailable(
            "synspace is not installed. pip install synspace (or put it on PYTHONPATH); "
            "it is optional and only the synspace_* ablation arms need it."
        ) from exc
    return synspace.chemical_space


def _guard_mols(scaffold_smarts: Guard) -> List[Chem.Mol]:
    pats = ([scaffold_smarts] if isinstance(scaffold_smarts, str)
            else list(scaffold_smarts or []))
    out = []
    for p in pats:
        if not p:
            continue
        m = Chem.MolFromSmarts(p)
        if m is None:
            raise ValueError(f"Invalid scaffold_smarts: {p}")
        out.append(m)
    return out


def _as_edit(base: Chem.Mol, prod: Chem.Mol) -> Optional[dict]:
    """Express ``base -> prod`` as ONE attach: ``([*:1], fragment, {1: site})``.

    synspace returns a finished product, but the agent's only edit primitive is
    ``edit_fragment`` (an MMP transform pinned by anchors), so a candidate is usable
    only when the whole of ``base`` survives in ``prod`` and the added atoms hang off a
    single site. Returns ``None`` otherwise — which is what happens to ring-forming
    reactions (benzimidazole, Pictet-Spengler: two seam bonds) and to couplings that
    consume an atom of the input (Suzuki eats the halide).
    """
    match = prod.GetSubstructMatch(base)
    if not match:
        return None
    keep = set(match)
    seam = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in prod.GetBonds()
            if (b.GetBeginAtomIdx() in keep) != (b.GetEndAtomIdx() in keep)]
    if len(seam) != 1:
        return None
    i, j = seam[0]
    core_atom, new_atom = (i, j) if i in keep else (j, i)
    bond = prod.GetBondBetweenAtoms(core_atom, new_atom)
    if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
        return None                       # only single-bond attachment is expressible
    rw = Chem.RWMol(prod)
    dummy = rw.AddAtom(Chem.Atom(0))
    rw.GetAtomWithIdx(dummy).SetAtomMapNum(1)
    rw.AddBond(dummy, new_atom, Chem.BondType.SINGLE)
    for idx in sorted(keep, reverse=True):
        rw.RemoveAtom(idx)
    frag = rw.GetMol()
    try:
        Chem.SanitizeMol(frag)
    except Exception:                     # noqa: BLE001 - unrepresentable fragment
        return None
    return {"from_smiles": _OPEN, "to_smiles": Chem.MolToSmiles(frag),
            "anchors": {"1": int(match.index(core_atom))}}


def _seed_int(rng: Optional[random.Random]) -> int:
    """A 32-bit seed for synspace's block sampling (it draws from the global
    ``numpy.random``), taken from the caller's RNG so an arm stays reproducible."""
    return (rng.randrange(2 ** 32) if rng is not None else 0)


def enumerate_candidates(mol_smiles: str, mode: str = "synspace", *,
                         scaffold_smarts: Guard = None,
                         rng: Optional[random.Random] = None,
                         pool_cap: int = POOL_CAP,
                         with_products: bool = True,
                         nblocks: int = NBLOCKS,
                         num_samples: int = NUM_SAMPLES,
                         threshold: float = THRESHOLD) -> List[dict]:
    """Every guard-passing, ``edit_fragment``-expressible synspace product, in random order.

    Each element is ``{label, from_smiles, to_smiles, anchors, mol}`` (plus ``product``
    when *with_products*), where ``label`` names the reaction that produced it.

    The args are derived from the product and then CHECKED by replaying them through
    ``replace_fragment``: a candidate whose replay does not reproduce synspace's own
    product is dropped, so what this returns is exactly what the agent can execute.
    """
    import numpy as np
    from molkit.utils.molecule_edit_utils import replace_fragment

    if mode not in MODES:
        raise ValueError(f"unknown mode '{mode}'; choose from {MODES}")
    if not mol_smiles or not isinstance(mol_smiles, str):
        raise ValueError("mol_smiles must be a non-empty SMILES string")
    base = Chem.MolFromSmiles(mol_smiles)
    if base is None or base.GetNumAtoms() == 0:
        raise ValueError(f"Invalid SMILES: {mol_smiles}")
    guards = _guard_mols(scaffold_smarts)
    chemical_space = _chemical_space()

    # synspace samples blocks from the GLOBAL numpy RNG; seed it per call so the arm is
    # reproducible under the harness's per-instance seed.
    np.random.seed(_seed_int(rng))
    mols, props = chemical_space(mol_smiles, steps=_STEPS[mode], threshold=threshold,
                                 nblocks=nblocks, num_samples=num_samples)

    in_canon = Chem.MolToSmiles(base)
    out: List[dict] = []
    seen: set = set()
    for prod, meta in zip(mols, props):
        if len(out) >= pool_cap:
            break
        canon = Chem.MolToSmiles(prod)
        if canon == in_canon or canon in seen:
            continue
        if guards and not all(prod.HasSubstructMatch(g) for g in guards):
            continue
        edit = _as_edit(base, prod)
        if edit is None:
            continue
        replayed = replace_fragment(mol_smiles, edit["from_smiles"], edit["to_smiles"],
                                    anchors={1: int(edit["anchors"]["1"])})
        if not replayed or Chem.MolToSmiles(Chem.MolFromSmiles(replayed[0])) != canon:
            continue                      # derived args do not reproduce the product
        seen.add(canon)
        cand = dict(edit)
        cand["label"] = (meta.get("rxn-name") or "synspace").strip() or "synspace"
        cand["mol"] = prod
        if with_products:
            cand["product"] = canon
        out.append(cand)
    if rng is not None:
        rng.shuffle(out)
    return out


def suggest_edits_synspace(mol_smiles: str, constraints: Optional[dict] = None,
                           top_k: int = 4, *, mode: str = "synspace",
                           scaffold_smarts: Guard = None,
                           rng: Optional[random.Random] = None,
                           pool_cap: int = POOL_CAP,
                           keep_product: bool = True,
                           **_ignored) -> List[dict]:
    """``top_k`` guard-passing forward-synthesis edits drawn UNIFORMLY AT RANDOM.

    Signature-compatible with :func:`molkit.utils.suggest_edits.suggest_edits` and
    with :func:`molkit.utils.random_suggesters.suggest_edits_random`, so the harness
    swaps one for another. *constraints* is NOT USED — that is the point of the arm.

    Returns fewer than *top_k* when the vocabulary offers fewer guard-passing edits, and
    ``[]`` when no reaction applies (the common case: see the module docstring).
    """
    pool = enumerate_candidates(mol_smiles, mode, scaffold_smarts=scaffold_smarts,
                                rng=rng, pool_cap=pool_cap, with_products=False)
    if not pool:
        return []
    k = max(0, min(int(top_k), len(pool)))
    picked = (rng.sample(pool, k) if rng is not None else pool[:k])
    out = []
    for c in picked:
        cand = {"from_smiles": c["from_smiles"], "to_smiles": c["to_smiles"],
                "anchors": dict(c["anchors"]), "predicted_gap": None,
                "label": c["label"]}
        if keep_product:
            cand["product"] = Chem.MolToSmiles(c["mol"])
        out.append(cand)
    return out
