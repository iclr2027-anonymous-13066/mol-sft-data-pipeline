#!/usr/bin/env python
"""Suggest the MMP edits that most reduce a molecule's distance to a target
property box — returned as ready-to-use ``edit_fragment`` arguments.

Given a molecule, property constraints (each a ``[lo, hi]`` range), and a candidate
count K, this ranks the mmpdb-derived moves (the per-cut move set built by
``0_build_mmpdb``) by the predicted reduction in the normalised distance to the
target box, and returns the top-K as::

    {"from_smiles": "[*:1]Cl", "to_smiles": "[*:1]OC", "anchors": {"1": 3},
     "predicted_gap": 0.41}                               # + "delta" if return_delta

``predicted_gap`` is the normalised (z-score) box distance AFTER applying the move;
candidates are sorted by the reduction from the molecule's current distance.
``delta`` covers exactly the properties named in ``constraints``, in that order. The ``from_smiles`` / ``to_smiles`` / ``anchors``
triple is exactly what ``edit_fragment`` takes, so a candidate applies verbatim.

Δ selection (``context_aware``, default True)
---------------------------------------------
The move set stores, per transform, a context-free radius-0 Δ plus context-specific
Δ at radii 1..5 (each keyed by the mmpdb environment ``context_smarts``). Ranking
uses the deepest-radius context whose SMARTS matches the move's actual site (lower
variance, and the mean can shift meaningfully — e.g. a methyl on O vs on aromatic
C), falling back to radius 0 when nothing more specific matches.

Cost: the O(N) step is only the vectorised radius-0 scoring; the (more expensive)
context match is applied ONLY to the top candidates about to be returned, not all N.
The radius-0 Δ matrix, the per-transform context index, and the compiled context
SMARTS queries are all cached across calls (in-process).

Disk cache: parsing the move set (124MB+ JSON + RDKit fragment checks) takes ~5s.
``build_cache()`` serialises the parsed ``(metas, D, ctx)`` to
``<moves_dir>/.suggest_cache_c{max_cut}.pkl``; ``_load_all`` then loads it in <1s
(and rebuilds automatically if the cut files are newer). ``build_mmp_moves`` writes
these caches at the end of every move-set build; regenerate manually with
``python -m molkit.utils.suggest_edits --build-cache`` (honours ``MMP_MOVES_DIR``).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors, QED

from molkit.utils.molecule_edit_utils import (
    replace_fragment, _map_attachments, _nondummy_heavy)

RDLogger.DisableLog("rdApp.*")

# 16 indexed properties, fixed order (build_mmp_moves PHYS_DEFAULT + ADMET_DEFAULT).
PROP_ORDER = ["MW", "logP", "HBD", "HBA", "TPSA", "rotB", "rings_total", "QED",
              "MR", "heavy_atoms", "formal_charge", "logD", "logS", "BBBP",
              "HIA", "Mutag"]
_PIDX = {p: i for i, p in enumerate(PROP_ORDER)}
_NPROP = len(PROP_ORDER)
_ADMET = frozenset({"logD", "logS", "BBBP", "HIA", "Mutag"})
# Integer-valued properties (Δ shown as int; every other property → 3 decimals).
_INT_PROPS = frozenset({"HBD", "HBA", "rotB", "rings_total", "heavy_atoms", "formal_charge"})

# Per-property z-score scale = population std over the pool. MIRROR of
# search_plan._SCALE / agentic_eval._PROP_SCALE — keep the three in sync.
_SCALE = {"MW": 86.0, "logP": 1.26, "HBD": 0.85, "HBA": 1.62, "TPSA": 26.0,
          "rotB": 1.93, "rings_total": 1.08, "QED": 0.15, "MR": 24.0,
          "heavy_atoms": 6.24, "formal_charge": 1.0, "logD": 1.36, "logS": 1.56,
          "BBBP": 0.125, "HIA": 0.1, "Mutag": 0.19}

# Default points at this node's move library. A missing dir is NOT an error here:
# _load_all just yields no metas and suggest_edits returns [], which is
# indistinguishable from "no edit helps" — so the default has to be a path that
# actually exists. Override with MMP_MOVES_DIR (the Makefile passes it explicitly).
MOVES_DIR = os.environ.get("MMP_MOVES_DIR", "data/mmp_moves")
CUT_FILES = {1: "single_cut.json", 2: "double_cut.json", 3: "triple_cut.json"}
_LONE_ATTACH = "[*:1]"          # edit_fragment's "attach here" from-fragment
_MAX_SITES = 200                # match sites ENUMERATED per from-fragment
_SITE_TRY = 16                  # match sites tried per move (guard-safe ones first)
_ANCHOR_VARIANTS = 2            # anchor variants of the SAME rule allowed in the window
_REPL_BUDGET = 2000             # REACTIONS per call — the safety bound (each
                                # covers every site of its rule, ~7.5 on average)
_GUARD_MATCH_CAP = 20           # guard matches on the input used for the safe-site test
_RESTORE_BUDGET = 1500          # reactions in scaffold-restoration mode
_RESTORE_SMARTS_CACHE = 32      # scaffold SMARTS kept in the to-fragment screen cache
_RXN_CACHE = 20000              # compiled reactions kept across calls
_CTX_MATCH_CAP = 200           # context rows to test per candidate (deepest first)


# ---------------------------------------------------------------------------
# measurement (physchem via RDKit always; ADMET via predict_admet on demand)
# ---------------------------------------------------------------------------
def _measure(mol_smiles: str, keys: List[str]) -> Dict[str, float]:
    """Measure *keys* for *mol_smiles*, byte-for-byte as ``analyze_properties`` does.

    The values MUST match ``molkit.tools.analysis._rdkit_properties`` exactly:
    the ranking here is against a constraint box the agent checks with
    ``analyze_properties``, and the mmpdb Δ tables were built from a property
    table that also uses those definitions. ``MW`` used to be
    ``Descriptors.MolWt`` (AVERAGE mass) against everything else's
    ``CalcExactMolWt`` (monoisotopic) — a systematic +0.2…0.5 Da offset that
    silently re-ranked candidates near a box boundary, and occasionally made
    every candidate look non-improving (empty result).

    "Exactly" includes the ROUNDING: ``analyze_properties`` reports every value to
    3 decimals, so the ADMET floats go through its own ``_safe_round`` rather than
    being passed on raw. A caller that hands us ``props=`` measured with
    ``analyze_properties`` (the stage-3 planner) and a caller that lets us measure
    must see the same numbers, or the two rank the same molecule differently. The
    residual disagreement is only ≤5e-4, but it is enough: candidates that fully
    close the box all score ``predicted_gap`` 0.0, so at the small ``top_k`` stage 3
    runs (4) a third of calls cut the list inside a group of exact ties, and which
    tied candidates survive the slice then turns on that last decimal.
    """
    mol = Chem.MolFromSmiles(mol_smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {mol_smiles}")
    phys = {
        "MW": round(rdMolDescriptors.CalcExactMolWt(mol), 3),
        "logP": round(Crippen.MolLogP(mol), 3),
        "HBD": rdMolDescriptors.CalcNumHBD(mol), "HBA": rdMolDescriptors.CalcNumHBA(mol),
        "TPSA": round(rdMolDescriptors.CalcTPSA(mol), 3),
        "rotB": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "rings_total": rdMolDescriptors.CalcNumRings(mol), "QED": round(QED.qed(mol), 3),
        "MR": round(Crippen.MolMR(mol), 3), "heavy_atoms": mol.GetNumHeavyAtoms(),
        "formal_charge": Chem.GetFormalCharge(mol),
    }
    out = {k: float(phys[k]) for k in keys if k in phys}
    admet_keys = [k for k in keys if k in _ADMET]
    if admet_keys:
        try:
            from molkit.tools.analysis import _safe_round
            from molkit.utils.molmim import predict_admet, COLUMN_MAP
            df = predict_admet([mol_smiles])
            row = df.iloc[0]
            for k in admet_keys:
                col = COLUMN_MAP.get(k)
                if col in df.columns and row[col] is not None:
                    v = _safe_round(row[col])       # None on NaN/Inf -> left unmeasured
                    if v is not None:
                        out[k] = float(v)
        except Exception as e:  # noqa: BLE001
            print(f"[suggest_edits] ADMET unavailable ({e}); {admet_keys} left unmeasured")
    return out


# ---------------------------------------------------------------------------
# cached move set: radius-0 matrices (both directions) + per-transform context index
# ---------------------------------------------------------------------------
# (moves_dir, max_cut) -> (metas, D, S, ctx_index)
#   metas[i]  = (move_from, move_to, support, orig_from, orig_to, sign)
#   D[i]      = 16-vec radius-0 Δ mean (sign already applied)
#   S[i]      = 16-vec radius-0 Δ std (NOT signed — std is always ≥ 0)
#   ctx_index = {(orig_from, orig_to): [(radius, ctx_smarts, avg16, std16, support), ...]}
#               radius >= 1 only, sorted radius desc then support desc
_CACHE: dict = {}
_QCACHE: dict = {}   # ctx_smarts -> (query_mol, [(query_atom_idx, label), ...])
_FCACHE: dict = {}   # from_smiles -> (query_mol, {query_atom_idx: label}) | None
_BYFROM: dict = {}   # (moves_dir, max_cut) -> {mv_from: np.ndarray of move indices}
_GROUPS: dict = {}   # (moves_dir, max_cut) -> (keys, flat move indices, group offsets)
_RXN: dict = {}      # (mv_from, mv_to) -> compiled reaction | None   (bounded)
_NDH: dict = {}      # fragment smiles -> non-dummy heavy-atom count
_TOSET: dict = {}    # (moves_dir, max_cut) -> frozenset of distinct mv_to
_TCORE: dict = {}    # to_smiles -> dummy-stripped Mol | False
_TOGUARD: dict = {}  # guard-SMARTS key -> frozenset of to_smiles inside those guards
_MISSING = object()  # sentinel: _FCACHE stores None as a real "not a fragment" answer


def _heavy_frag(frag: str) -> bool:
    m = Chem.MolFromSmiles(frag)
    return m is not None and any(a.GetAtomicNum() > 1 for a in m.GetAtoms())


def _cache_path(moves_dir: str, max_cut: int) -> str:
    # v2: cache now carries per-property Δ std (S matrix + ctx std vectors).
    return os.path.join(moves_dir, f".suggest_cache_v2_c{max_cut}.pkl")


def _cache_fresh(path: str, moves_dir: str, max_cut: int) -> bool:
    """True iff the pickle exists and is newer than every cut file it derives from
    (so a regenerated move set invalidates a stale cache)."""
    if not os.path.exists(path):
        return False
    cmt = os.path.getmtime(path)
    for cut in range(1, max_cut + 1):
        f = os.path.join(moves_dir, CUT_FILES[cut])
        if os.path.exists(f) and os.path.getmtime(f) > cmt:
            return False
    return True


def _build_index(moves_dir: str, max_cut: int):
    """Parse the per-cut move files into (metas, D, S, ctx_index) — the slow path
    (~5s: 124MB JSON parse + RDKit fragment checks). Cached to disk by build_cache.
    D = per-move Δ mean (signed), S = per-move Δ std (unsigned)."""
    metas: List[tuple] = []
    rows: List[np.ndarray] = []
    srows: List[np.ndarray] = []
    ctx: dict = defaultdict(list)
    for cut in range(1, max_cut + 1):
        path = os.path.join(moves_dir, CUT_FILES[cut])
        if not os.path.exists(path):
            continue
        for r in json.load(open(path)):
            frm, to, rad, sup = r["from"], r["to"], r.get("radius", 0), r.get("support", 0)
            d = r.get("delta", {})
            dv = np.array([d.get(p, {}).get("avg", 0.0) for p in PROP_ORDER], dtype=float)
            sv = np.array([d.get(p, {}).get("std", 0.0) for p in PROP_ORDER], dtype=float)
            if rad == 0:
                if _heavy_frag(frm):                       # forward A -> B
                    metas.append((frm, to, sup, frm, to, 1)); rows.append(dv); srows.append(sv)
                if to == "[*:1][H]":                       # removal reverse -> growth
                    metas.append((_LONE_ATTACH, frm, sup, frm, to, -1)); rows.append(-dv); srows.append(sv)
                elif _heavy_frag(to):                       # swap reverse B -> A
                    metas.append((to, frm, sup, frm, to, -1)); rows.append(-dv); srows.append(sv)
            else:
                ctx[(frm, to)].append((rad, r.get("context_smarts", ""), dv, sv, sup))
    for k in ctx:
        ctx[k].sort(key=lambda t: (-t[0], -t[4]))          # deepest radius, then support
    D = np.vstack(rows) if rows else np.zeros((0, _NPROP))
    S = np.vstack(srows) if srows else np.zeros((0, _NPROP))
    return metas, D, S, dict(ctx)


def build_cache(moves_dir: str = MOVES_DIR, max_cut: int = 3) -> str:
    """(Re)build the fast-load disk cache for one max_cut and return its path.
    Called at the end of a move-set build and by ``--build-cache``; the tool server
    then loads this pickle in <1s instead of re-parsing 124MB of JSON per worker."""
    metas, D, S, ctx = _build_index(moves_dir, max_cut)
    path = _cache_path(moves_dir, max_cut)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump((metas, D, S, ctx), f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)                                   # atomic; safe under many workers
    _CACHE[(moves_dir, max_cut)] = (metas, D, S, ctx)
    return path


def _load_all(moves_dir: str, max_cut: int):
    key = (moves_dir, max_cut)
    if key in _CACHE:                                       # in-process
        return _CACHE[key]
    path = _cache_path(moves_dir, max_cut)                  # fast disk cache
    if _cache_fresh(path, moves_dir, max_cut):
        try:
            with open(path, "rb") as f:
                _CACHE[key] = pickle.load(f)
            return _CACHE[key]
        except Exception:  # noqa: BLE001 — corrupt/old pickle → rebuild
            pass
    _CACHE[key] = _build_index(moves_dir, max_cut)          # slow build (no auto-write)
    return _CACHE[key]


_DT_CACHE: Dict[tuple, np.ndarray] = {}


def _delta_columns(moves_dir: str, max_cut: int, D: np.ndarray) -> np.ndarray:
    """``D`` transposed and made contiguous, so one property is one cache-friendly row.

    ``_red0`` walks the move deltas one PROPERTY at a time. Read out of ``D`` that is a
    strided column of a (279666, 16) array — a fresh gather per property per call. Held
    transposed once per process it is a contiguous (16, 279666) row, which is what makes
    the loop cheaper than the array-at-once form it replaced.
    """
    key = (moves_dir, max_cut, id(D))
    dt = _DT_CACHE.get(key)
    if dt is None:
        dt = _DT_CACHE[key] = np.ascontiguousarray(D.T)
    return dt


def _red0(DT: np.ndarray, cols, v: np.ndarray, lo: np.ndarray, hi: np.ndarray,
          scale: np.ndarray, cur_g: np.ndarray) -> np.ndarray:
    """Radius-0 gap reduction for EVERY move: ``sum_p (cur_g - gap(v+D)) / scale``.

    Identical, to the bit, to the array-at-once expression this replaces::

        ((cur_g[None, :] - _gap(v[None, :] + D[:, cols], lo, hi)) / scale).sum(axis=1)

    but ~3.6x cheaper (measured, 2-14 properties). The expression above materialises
    five (279666, P) float64 temporaries — the fancy-index copy of D alone is 31 MB at
    P=14 — and the run is memory-bound on them. Accumulating one property at a time
    keeps every temporary at (279666,) and reuses two buffers, so the working set stays
    in cache. Same dtype, same order of operations per property, so the ranking it
    feeds is unchanged; float32 would be ~10x but perturbs near-ties, and the ordering
    of near-tied fragments is exactly what this feeds.
    """
    n = DT.shape[1]
    out = np.zeros(n, dtype=np.float64)
    g = np.empty(n, dtype=np.float64)
    over = np.empty(n, dtype=np.float64)            # reused: one alloc, not one per property
    for j, c in enumerate(cols):
        d = DT[c]
        np.add(d, v[j], out=g)                      # v + delta
        np.subtract(g, hi[j], out=over)             # (v+delta) - hi
        np.subtract(lo[j], g, out=g)                # lo - (v+delta)
        np.maximum(g, over, out=g)
        np.maximum(g, 0.0, out=g)                   # = _gap(v+delta, lo, hi)
        np.subtract(cur_g[j], g, out=g)
        np.divide(g, scale[j], out=g)
        np.add(out, g, out=out)
    return out


def _get_query(ctx_smarts: str):
    """Compile a context SMARTS once (cached): (query_mol, [(atom_idx, label), ...])
    for its attachment dummies. Returns (None, None) if it won't parse."""
    hit = _QCACHE.get(ctx_smarts)
    if hit is not None:
        return hit
    qm = Chem.MolFromSmarts(ctx_smarts)
    dums = ([(a.GetIdx(), a.GetAtomMapNum()) for a in qm.GetAtoms() if a.GetAtomicNum() == 0]
            if qm is not None else None)
    _QCACHE[ctx_smarts] = (qm, dums)
    return qm, dums


# ---------------------------------------------------------------------------
# anchor sites + cut-core (constant part with a dummy at each attachment)
# ---------------------------------------------------------------------------
def _query(from_smiles: str):
    """Compile a from-fragment into a match query ONCE (process-wide cache):
    (query_mol, {query_atom_idx: attachment_label}), or None if it can't be one.

    The move set has 279,666 moves but only ~13k distinct from-fragments, so the
    same fragment is compiled tens of times per call without this cache (measured:
    58 us/compile+match vs 8 us matched against a warm cache). The full vocabulary
    costs ~72 MB and ~0.9 s to build, once per process."""
    hit = _FCACHE.get(from_smiles, _MISSING)
    if hit is not _MISSING:
        return hit
    patt0 = Chem.MolFromSmiles(from_smiles)
    val = None
    if patt0 is not None:
        dummies = {a.GetIdx(): (a.GetAtomMapNum() or 0)
                   for a in patt0.GetAtoms() if a.GetAtomicNum() == 0}
        if dummies:
            nxt = max([v for v in dummies.values() if v] or [0])
            for di in dummies:
                if not dummies[di]:
                    nxt += 1
                    dummies[di] = nxt
            qp = Chem.AdjustQueryParameters.NoAdjustments()
            qp.makeDummiesQueries = True
            val = (Chem.AdjustQueryProperties(patt0, qp), dummies)
    _FCACHE[from_smiles] = val
    return val


def _nd_heavy(frag: str) -> int:
    """Cached ``_nondummy_heavy`` — it re-parses the fragment SMILES on every call,
    and the scan asks about the same few thousand fragments over and over."""
    hit = _NDH.get(frag, _MISSING)
    if hit is _MISSING:
        hit = _nondummy_heavy(frag)
        _NDH[frag] = hit
    return hit


def _products_by_anchor(base: Chem.Mol, in_canon: str, mv_from: str, mv_to: str
                        ) -> Dict[tuple, Chem.Mol]:
    """Every clean product of this rule on ``base``, keyed by its anchor map.

    Same semantics as ``replace_fragment`` (verified identical on 962 rule/anchor
    pairs) but run ONCE per rule instead of once per anchor. ``RunReactants``
    enumerates the outcome at every match site regardless, and ``replace_fragment``
    then throws away all but the one anchor it was asked about — so calling it per
    anchor did the same reaction N times and discarded N-1 outcomes each time
    (measured: 18 outcomes per call, 17 discarded; 4.8x more work overall).

    Products are returned as Mols, so the caller's guard test needs no SMILES
    round-trip either."""
    key = (mv_from, mv_to)
    rxn = _RXN.get(key, _MISSING)
    if rxn is _MISSING:
        try:
            rxn = AllChem.ReactionFromSmarts(
                f"{_map_attachments(mv_from)}>>{_map_attachments(mv_to)}")
        except Exception:  # noqa: BLE001
            rxn = None
        if rxn is not None and rxn.GetNumReactantTemplates() != 1:
            rxn = None
        if len(_RXN) >= _RXN_CACHE:
            _RXN.clear()                       # cheap bound; rebuild costs ~0.3 ms/rule
        _RXN[key] = rxn
    if rxn is None:
        return {}
    nd_from, nd_to = _nd_heavy(mv_from), _nd_heavy(mv_to)
    check_heavy = nd_from >= 0 and nd_to >= 0
    exp_heavy = base.GetNumHeavyAtoms() - nd_from + nd_to
    try:
        outcomes = rxn.RunReactants((base,))
    except Exception:  # noqa: BLE001
        return {}
    out: Dict[tuple, Chem.Mol] = {}
    for outcome in outcomes:
        mol = outcome[0]
        lbl = {}
        for a in mol.GetAtoms():
            if a.HasProp("old_mapno") and a.HasProp("react_atom_idx"):
                lbl[int(a.GetProp("old_mapno"))] = a.GetIntProp("react_atom_idx")
        try:
            prod = Chem.RemoveHs(mol)
            Chem.SanitizeMol(prod)
        except Exception:  # noqa: BLE001
            continue
        if check_heavy and prod.GetNumHeavyAtoms() != exp_heavy:
            continue                            # collateral deletion — not a clean swap
        smi = Chem.MolToSmiles(prod)
        if smi == in_canon or "." in smi:
            continue
        out.setdefault(tuple(sorted(lbl.items())), prod)
    return out


def _clean_site(base: Chem.Mol, anchors: Dict[int, int], frag_atoms: frozenset) -> bool:
    """Would a swap here be a CLEAN one — no collateral atoms deleted?

    ``replace_fragment`` runs the swap as a reaction, which deletes every matched
    ``from`` atom and anything hanging off it that is not a declared attachment, and
    then rejects the product unless the heavy-atom count came out exactly right. So a
    site where some fragment atom carries a substituent outside the match is dead for
    EVERY ``to`` — that verdict depends on (from-fragment, site) only, never on what
    we would put there. Deciding it here on the graph keeps dead sites out of the
    window instead of discovering them one wasted reaction at a time."""
    keep = set(anchors.values())
    for idx in frag_atoms:
        for nb in base.GetAtomWithIdx(int(idx)).GetNeighbors():
            j = nb.GetIdx()
            if j not in frag_atoms and j not in keep:
                return False
    return True


def _anchor_sites(base: Chem.Mol, from_smiles: str, limit: int = _MAX_SITES
                  ) -> List[Tuple[Dict[int, int], frozenset]]:
    """EVERY place ``from_smiles`` matches: ({label: anchor-atom-idx}, {fragment atoms}).

    ``limit`` is a blow-up bound, NOT a selection: it used to be 8, which silently
    kept the first 8 matches in ATOM-INDEX order — before the scaffold guard had any
    say. For the bare attachment ``[*:1]`` (1137 moves, matches every atom) that made
    only atoms 0-7 reachable, and measured on real eval calls the anchors that
    actually produced a guard-passing product were routinely outside that prefix.
    The caller now ranks these sites (guard-safe first) and does its own truncation."""
    q = _query(from_smiles)
    if q is None:
        return []
    patt, dummies = q
    out, seen = [], set()
    for m in base.GetSubstructMatches(patt, uniquify=False, maxMatches=2000):
        anch = {lbl: m[di] for di, lbl in dummies.items()}
        if len(set(anch.values())) != len(anch):
            continue
        key = tuple(sorted(anch.items()))
        if key in seen:
            continue
        seen.add(key)
        frag = frozenset(m[i] for i in range(len(m)) if i not in dummies)
        # A fragment with no heavy atoms (the bare `[*:1]`) does not replace anything —
        # it hangs a NEW bond off the anchor, which needs a free hydrogen there.
        # replace_fragment would return [] anyway; skipping is just cheaper.
        if not frag and any(base.GetAtomWithIdx(a).GetTotalNumHs() == 0
                            for a in anch.values()):
            continue
        if not _clean_site(base, anch, frag):
            continue
        out.append((anch, frag))
        if len(out) >= limit:
            break
    return out


def _by_from(moves_dir: str, max_cut: int, metas: List[tuple]) -> Dict[str, np.ndarray]:
    """``mv_from`` -> indices of every move that starts from it (process-wide cache).
    Lets a call ask "which of the ~13k distinct fragments occur in this molecule?"
    instead of re-deriving that from 280k moves."""
    key = (moves_dir, max_cut)
    hit = _BYFROM.get(key)
    if hit is None:
        d: dict = defaultdict(list)
        for i, m in enumerate(metas):
            d[m[0]].append(i)
        hit = {k: np.array(v, dtype=np.int32) for k, v in d.items()}
        # Flat layout for the per-fragment best-score reduction below: the same move
        # indices concatenated in key order, plus each group's start offset.
        ks = list(hit)
        # _BYFROM is published LAST, and it is what every reader tests: the three caches
        # are one logical entry, and a caller that sees _BYFROM populated goes straight
        # on to read _GROUPS (_best_per_from) and _TOSET. Publishing _BYFROM first let a
        # second thread through the door before the other two existed — a KeyError on
        # the very first concurrent call, which the suggesters report as "no candidates".
        _TOSET[key] = frozenset(m[1] for m in metas)
        _GROUPS[key] = (ks,
                        np.concatenate([hit[k] for k in ks]),
                        np.cumsum([0] + [len(hit[k]) for k in ks[:-1]]))
        _BYFROM[key] = hit
    return hit


def _best_per_from(moves_dir: str, max_cut: int, red0: np.ndarray):
    """(from-fragments, their best red0) — one vectorised segment-max.

    The obvious ``{f: red0[idx].max() for f, idx in by_from.items()}`` is 13k
    Python-level numpy calls and measured 21.6 ms, which on the common fast path was
    ~45% of the whole call. ``reduceat`` over the flat layout does the same work in
    2 ms with identical values."""
    keys, flat, offs = _GROUPS[(moves_dir, max_cut)]
    return keys, np.maximum.reduceat(red0[flat], offs)


def _restore_scan(base: Chem.Mol, guards: List[Chem.Mol], cache_key: str,
                  metas: List[tuple], by_from: Dict[str, np.ndarray], tos: frozenset,
                  target: int, all_guards: Optional[List[Chem.Mol]] = None
                  ) -> List[Tuple[int, Dict[int, int], frozenset]]:
    """Scaffold-RESTORATION scan, for a molecule that does not contain the guard yet.

    The normal objective cannot reach these edits at all: measured on the eval dumps,
    the edits that DO create the scaffold score red0 ~= -1.9 (they cost property gap to
    buy the scaffold back) and rank ~180,000 of 279,666, so the gap-ordered scan stops
    long before them. That is why 68-72% of all empty responses come from molecules in
    this state. So here the objective flips: an edit qualifies by making the product
    match the guard, and predicted gap only orders what qualifies.

    Two exact necessary conditions keep it affordable — the from-fragment must occur in
    the molecule, and the to-fragment must lie inside the guard — after which the
    biggest to-fragments are tried first, since the molecule is missing scaffold and a
    bigger insert supplies more of it. Returns (move_index, anchors, frag_atoms)."""
    # *guards* are the MISSING ones (what the edit has to supply); the product is then
    # checked against *all_guards*, so restoring one does not silently break another.
    check = all_guards if all_guards is not None else guards
    allowed = _to_inside_guard(cache_key, guards, tos)
    if not allowed:
        return []
    sites: Dict[str, list] = {}
    cand: List[int] = []
    for fs, idx in by_from.items():
        q = _query(fs)
        if q is None or not base.HasSubstructMatch(q[0]):
            continue
        st = _anchor_sites(base, fs)
        if not st:
            continue
        sites[fs] = st
        cand.extend(int(i) for i in idx if metas[i][1] in allowed)
    if not cand:
        return []
    cand.sort(key=lambda i: -_to_core(metas[i][1]).GetNumAtoms())
    in_canon = Chem.MolToSmiles(base)
    hits: List[Tuple[int, Dict[int, int], frozenset]] = []
    nrxn = 0
    for i in cand:
        if len(hits) >= target or nrxn >= _RESTORE_BUDGET:
            break
        mv_from, mv_to = metas[i][0], metas[i][1]
        nrxn += 1
        pmap = _products_by_anchor(base, in_canon, mv_from, mv_to)
        if not pmap:
            continue
        for anch, frag in sites[mv_from][:_SITE_TRY]:
            prod = pmap.get(tuple(sorted(anch.items())))
            if prod is None or not all(prod.HasSubstructMatch(g) for g in check):
                continue
            hits.append((i, anch, frag))
            break
    return hits


def _to_core(to_smiles: str):
    """The ``to`` fragment with its attachment dummies removed — the heavy part that
    actually lands in the product. Cached; ``False`` when there is nothing left."""
    hit = _TCORE.get(to_smiles, _MISSING)
    if hit is not _MISSING:
        return hit
    tm = Chem.MolFromSmiles(to_smiles)
    val = False
    if tm is not None:
        rw = Chem.RWMol(tm)
        for idx in sorted([a.GetIdx() for a in tm.GetAtoms() if a.GetAtomicNum() == 0],
                          reverse=True):
            rw.RemoveAtom(idx)
        core = rw.GetMol()
        if core.GetNumAtoms():
            val = core
    _TCORE[to_smiles] = val
    return val


def _to_inside_guard(cache_key: str, guards: List[Chem.Mol],
                     tos: frozenset) -> frozenset:
    """The to-fragments that are substructures of the guard pattern.

    This is the screen that makes scaffold RESTORATION affordable. When the input does
    not contain the guard, the product can only contain it by using atoms of ``to`` —
    the rest of the product is input atoms, and those do not contain it. So ``to``
    itself must sit inside the guard. It rejects no real answer, and it keeps ~5% of
    the 13k to-fragment vocabulary (measured), in ~0.7 s for the whole vocabulary.
    Cached per SMARTS key: one rollout reuses its guard set across every segment.
    With several guards the sets are UNIONed — a restoring edit only has to supply the
    guard that is missing, and requiring `to` to sit inside all of them at once would
    reject every real answer."""
    hit = _TOGUARD.get(cache_key)
    if hit is not None:
        return hit
    ok = set()
    for ts in tos:
        core = _to_core(ts)
        if core is False:
            continue
        for g in guards:
            try:
                if g.HasSubstructMatch(core):
                    ok.add(ts)
                    break
            except Exception:  # noqa: BLE001 — an unusable pair just isn't a candidate
                pass
    hit = frozenset(ok)
    if len(_TOGUARD) >= _RESTORE_SMARTS_CACHE:
        _TOGUARD.pop(next(iter(_TOGUARD)))
    _TOGUARD[cache_key] = hit
    return hit


def _split_sites(sites: List[Tuple[Dict[int, int], frozenset]],
                 gmatches: List[List[frozenset]]) -> Tuple[list, list]:
    """(guard-safe, needs-checking). A site whose removed atoms are disjoint from some
    match of a guard leaves every atom AND bond of that match intact, so the product
    still matches that guard whatever we put there — no product needs to be built to
    know it. The bare attachment removes nothing, so all of its sites land here.

    *gmatches* is one match list PER guard. A site is safe only when EVERY guard has
    such an untouched match: preserving one required group says nothing about the
    others."""
    if not gmatches or any(not g for g in gmatches):
        return [], list(sites)
    safe, other = [], []
    for anch, frag in sites:
        ok = all(any(not (frag & m) for m in per_guard) for per_guard in gmatches)
        (safe if ok else other).append((anch, frag))
    return safe, other


def _cut_core(base: Chem.Mol, anchors: Dict[int, int], frag_atoms: frozenset):
    """Constant part of the molecule at this site — the ``from`` fragment removed
    and a dummy ``[*:label]`` placed at each anchor — matching what mmpdb's
    environment ``context_smarts`` describes. Returns (core_mol, {label: dummy_idx})
    or (None, None)."""
    rw = Chem.RWMol(base)
    for lbl, aidx in anchors.items():
        d = rw.AddAtom(Chem.Atom(0))
        rw.GetAtomWithIdx(d).SetAtomMapNum(1000 + int(lbl))   # stable marker across removal
        rw.AddBond(int(aidx), d, Chem.BondType.SINGLE)
    for idx in sorted(frag_atoms, reverse=True):
        rw.RemoveAtom(idx)
    core = rw.GetMol()
    try:
        Chem.SanitizeMol(core)
    except Exception:  # noqa: BLE001
        return None, None
    lbl2dummy = {a.GetAtomMapNum() - 1000: a.GetIdx()
                 for a in core.GetAtoms() if a.GetAtomMapNum() >= 1000}
    return core, lbl2dummy


def _context_delta(core: Chem.Mol, lbl2dummy: Dict[int, int],
                   context_rows: list, sign: int):
    """Deepest-radius context Δ whose SMARTS matches this site (dummies aligned to
    the site's attachments). Returns (avg16, std16, radius) or (None, None, None).
    The mean is signed; the std is not (std ≥ 0 regardless of move direction)."""
    for radius, ctx_smarts, dvec, svec, _sup in context_rows[:_CTX_MATCH_CAP]:
        qm, dums = _get_query(ctx_smarts)
        if qm is None or not dums:
            continue
        for m in core.GetSubstructMatches(qm, uniquify=False, maxMatches=200):
            if all(m[qi] == lbl2dummy.get(lbl) for qi, lbl in dums):
                return sign * dvec, svec, radius
    return None, None, None


def _gap(v: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, np.maximum(lo - v, v - hi))


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------
def suggest_edits(mol_smiles: str, constraints: Dict[str, list], top_k: int = 4, *,
                  props: Optional[dict] = None, max_cut: int = 3,
                  context_aware: bool = True, return_delta: bool = True,
                  scaffold_smarts: Optional[Union[str, List[str]]] = None) -> List[dict]:
    """Rank mmpdb moves by predicted normalised gap-reduction for ``mol_smiles``
    against ``constraints`` ({prop: [lo, hi]}; lo or hi may be null for open), and
    return the best ``top_k`` as edit_fragment-ready dicts {from_smiles, to_smiles,
    anchors, predicted_gap, delta}.

    ``scaffold_smarts`` (optional): when given, only edits whose product still
    matches it are returned — i.e. edits that PRESERVE the substructure you want to
    keep. Omit it to allow any gap-reducing edit.

    A LIST of SMARTS may be given, and then every one of them must survive. That is
    the only correct form for a multi-group functional-group constraint: joining the
    patterns with ``.`` instead would demand ATOM-DISJOINT matches and so reject
    molecules where two required groups legitimately share an atom (an amide whose N
    is also a piperazine N fails ``C(=O)-N.N1CCNCC1`` although both groups are there).

    Each element carries ``delta`` = ``{prop: {"avg": <mean Δ>, "std": <Δ std>}}``
    (the predicted per-property change and its spread over the matched pairs; the
    context-refined value when a context matched, else radius-0). ``delta`` covers
    EXACTLY the properties in ``constraints``, in that order, Δ 0 included — the
    caller asked about those properties, so "this move does not move it" is an
    answer, not a reason to stay silent. ``predicted_gap`` is the total normalised
    box distance AFTER the edit (lower = better)."""
    base = Chem.MolFromSmiles(mol_smiles)
    if base is None:
        raise ValueError(f"Invalid SMILES: {mol_smiles}")
    guard_smarts = ([scaffold_smarts] if isinstance(scaffold_smarts, str)
                    else list(scaffold_smarts or []))
    guard_smarts = [g for g in guard_smarts if g]
    guards: List[Chem.Mol] = []
    for gs in guard_smarts:
        g = Chem.MolFromSmarts(gs)
        if g is None:
            raise ValueError(f"Invalid scaffold_smarts: {gs}")
        guards.append(g)
    guard_key = "\x00".join(guard_smarts)      # cache key for _to_inside_guard
    keys = list(constraints)
    unknown = [k for k in keys if k not in _PIDX]
    if unknown:
        raise ValueError(f"unknown properties {unknown}; choose from {PROP_ORDER}")

    lo = np.array([(-1e18 if constraints[k][0] is None else constraints[k][0]) for k in keys])
    hi = np.array([(1e18 if constraints[k][1] is None else constraints[k][1]) for k in keys])
    scale = np.array([_SCALE[k] for k in keys])
    cols = [_PIDX[k] for k in keys]

    meas = dict(props) if props else _measure(mol_smiles, keys)
    missing = [k for k in keys if meas.get(k) is None]
    if missing:
        raise ValueError(f"could not measure {missing} for the input molecule "
                         f"(ADMET backend down? pass props=... to override)")
    v = np.array([float(meas[k]) for k in keys])
    cur_g = _gap(v, lo, hi)
    current_gap = float(np.sum(cur_g / scale))
    if current_gap <= 1e-9:
        return []                                          # already inside the box

    metas, D, S, ctx_index = _load_all(MOVES_DIR, max_cut)
    if not metas:
        return []

    # 1. cheap vectorised radius-0 ranking (the only O(N) step)
    red0 = _red0(_delta_columns(MOVES_DIR, max_cut, D), cols, v, lo, hi, scale, cur_g)

    # 2. walk the move set FROM-FRAGMENT FIRST, best-red0 fragment first.
    #
    #    The old loop walked all 279,666 moves in red0 order and asked `_anchor_sites`
    #    per move whether its from-fragment occurs here. Measured on the eval dumps:
    #    94.6% of the 18.6k moves it got through died on that question alone, and it
    #    still covered only a third of the gap-reducing pool before the cap stopped it.
    #    But the moves are only ~13k distinct fragments paired up (mean 21 moves per
    #    fragment), so asking the question once per FRAGMENT screens the whole
    #    vocabulary in ~0.07 s and typically leaves a few hundred that apply. That
    #    makes coverage of the positive pool complete and retires the move cap; the
    #    remaining real cost is building products, which `_REPL_BUDGET` bounds.
    by_from = _by_from(MOVES_DIR, max_cut, metas)
    fkeys, fbest = _best_per_from(MOVES_DIR, max_cut, red0)
    pos = np.nonzero(fbest > 1e-9)[0]
    froms = [fkeys[i] for i in pos[np.argsort(-fbest[pos], kind="stable")]]
    in_canon = Chem.MolToSmiles(base)            # computed once, not per product

    # Guard matches on the INPUT; a site disjoint from one of these is safe by
    # construction (see `_split_sites`), which is what lets most candidates skip
    # product construction entirely. Empty when the guard does not match the input —
    # then nothing can be assumed and every site takes the checked path.
    gmatches = [[frozenset(m) for m in
                 base.GetSubstructMatches(g, uniquify=True, maxMatches=_GUARD_MATCH_CAP)]
                for g in guards]

    window = max(40, 4 * top_k)                 # target: collect this many, then rank
    repl_left = _REPL_BUDGET
    collected: List[dict] = []
    per_rule: dict = {}                         # (from,to) -> anchor variants kept

    # The guard is a filter on the PRODUCT, so a molecule that does not contain the
    # scaffold yet can still be one edit away from containing it. That case needs the
    # restoration objective instead of the gap objective — see `_restore_scan`.
    # Restoration is needed as soon as ANY guard is absent from the input; the scan is
    # pointed at exactly the missing ones, and its products are checked against all.
    missing = [g for g, ms in zip(guards, gmatches) if not ms]
    restore = bool(missing)
    if restore:
        for i, anch, frag in _restore_scan(base, missing, guard_key,
                                           metas, by_from, _TOSET[(MOVES_DIR, max_cut)],
                                           max(2 * top_k, 8), all_guards=guards):
            mv_f, mv_to, _sup, of, ot, sign = metas[i]
            dvec, svec = D[i], S[i]
            if context_aware and (of, ot) in ctx_index:
                core, lbl2dummy = _cut_core(base, anch, frag)
                if core is not None:
                    cdvec, csvec, _crad = _context_delta(
                        core, lbl2dummy, ctx_index[(of, ot)], sign)
                    if cdvec is not None:
                        dvec, svec = cdvec, csvec
            cred = float(((cur_g - _gap(v + dvec[cols], lo, hi)) / scale).sum())
            collected.append({"mv_from": mv_f, "mv_to": mv_to, "anch": anch,
                              "cred": cred, "dvec": dvec, "svec": svec,
                              "verified": True})
        froms = ()                              # restoration replaces the gap scan

    for mv_from in froms:
        if len(collected) >= window:
            break
        q = _query(mv_from)
        if q is None or not base.HasSubstructMatch(q[0]):
            continue                            # fragment absent — the cheap screen
        safe, checked = _split_sites(_anchor_sites(base, mv_from), gmatches)
        if not safe and not checked:
            continue
        idxs = by_from[mv_from]
        for i in idxs[np.argsort(-red0[idxs], kind="stable")]:
            if red0[i] <= 1e-9 or len(collected) >= window:
                break
            mv_f, mv_to, _sup, of, ot, sign = metas[i]
            rule = (mv_f, mv_to)
            # The same transform at a DIFFERENT anchor is a different product, so it
            # is a separate candidate — the old code deduped by rule alone and threw
            # those away (measured: that alone is the difference between 6/12 and
            # 10/12 of the hardest calls being able to fill top_k at all). Bounded so
            # one rule cannot crowd the window out.
            if per_rule.get(rule, 0) >= _ANCHOR_VARIANTS:
                continue
            tried = 0
            pmap = None                 # this rule's products, built at most once
            for sites, presumed_ok in ((safe, True), (checked, False)):
                for anch, frag in sites:
                    if (per_rule.get(rule, 0) >= _ANCHOR_VARIANTS
                            or len(collected) >= window or tried >= _SITE_TRY):
                        break
                    tried += 1
                    if not presumed_ok:
                        if pmap is None:
                            if repl_left <= 0:
                                break
                            repl_left -= 1
                            pmap = _products_by_anchor(base, in_canon, mv_f, mv_to)
                        prod = pmap.get(tuple(sorted(anch.items())))
                        if prod is None:
                            continue
                        if guards:                      # substructure filter, on the PRODUCT
                            if not all(prod.HasSubstructMatch(g) for g in guards):
                                continue
                    dvec, svec = D[i], S[i]
                    if context_aware and (of, ot) in ctx_index:
                        core, lbl2dummy = _cut_core(base, anch, frag)
                        if core is not None:
                            cdvec, csvec, _crad = _context_delta(
                                core, lbl2dummy, ctx_index[(of, ot)], sign)
                            if cdvec is not None:
                                dvec, svec = cdvec, csvec
                    cred = float(((cur_g - _gap(v + dvec[cols], lo, hi)) / scale).sum())
                    collected.append({"mv_from": mv_f, "mv_to": mv_to, "anch": anch,
                                      "cred": cred, "dvec": dvec, "svec": svec,
                                      "verified": not presumed_ok,
                                      # The product, when this path already built it.
                                      # It is emitted as cand["product"] so the caller
                                      # does not run the same reaction a second time.
                                      "prod": None if presumed_ok else prod})
                    per_rule[rule] = per_rule.get(rule, 0) + 1

    # 3. re-rank the window by the (context-aware) reduction and emit top_k.
    #    Candidates taken on the guard-safe fast path were never built, so each one
    #    that is actually RETURNED is constructed and re-checked here; a candidate
    #    that fails to apply is dropped and the next one in the ranking takes its
    #    place, which is why the window is collected several times larger than top_k.
    collected.sort(key=lambda c: -c["cred"])
    out: List[dict] = []
    emitted_rules: set = set()
    # Two passes so the anchor variants fill the list rather than crowd it: pass 1 takes
    # the best site of each distinct rule, pass 2 comes back for second sites only if
    # top_k is still short. Same ranking within each pass.
    for allow_repeat in (False, True):
        if len(out) >= top_k:
            break
        for c in collected:
            # In restoration mode the gap gate is off on purpose: buying the scaffold
            # back normally COSTS predicted gap (measured red0 ~= -1.9 on real cases),
            # and a molecule that does not contain the required substructure has no
            # use for an edit that polishes its properties instead.
            if len(out) >= top_k or (not restore and c["cred"] <= 1e-9):
                break
            rule = (c["mv_from"], c["mv_to"])
            if c.get("emitted"):
                continue
            if not allow_repeat and rule in emitted_rules:
                continue
            product = None
            if not c["verified"]:
                prods = replace_fragment(mol_smiles, c["mv_from"], c["mv_to"],
                                         anchors=c["anch"])
                if not prods:
                    c["emitted"] = True         # dead for good; don't retry in pass 2
                    continue
                if guards:
                    pm = Chem.MolFromSmiles(prods[0])
                    if pm is None or not all(pm.HasSubstructMatch(g) for g in guards):
                        c["emitted"] = True
                        continue
                product = prods[0]
            elif c.get("prod") is not None:
                product = Chem.MolToSmiles(c["prod"])
            emitted_rules.add(rule)
            c["emitted"] = True
            # Δ is reported for EXACTLY the constrained properties, in the caller's
            # constraint order — including the ones the move does not move (Δ 0). A
            # property the caller asked about but that is absent from the response
            # reads as "unknown"; an explicit 0 says "this move leaves it alone",
            # which is the load-bearing fact for a constraint you must not break
            # (rings_total on a scaffold, say). Properties outside `constraints` are
            # not reported at all: they never enter the ranking (see `cols` above),
            # so they were pure prompt weight.
            delta = {}
            for p, j in zip(keys, cols):
                avg = (int(round(c["dvec"][j])) if p in _INT_PROPS
                       else round(float(c["dvec"][j]), 3))
                delta[p] = {"avg": avg, "std": round(float(c["svec"][j]), 3)}
            cand = {
                "from_smiles": c["mv_from"], "to_smiles": c["mv_to"],
                "anchors": {str(k): int(val) for k, val in c["anch"].items()},
                "predicted_gap": round(current_gap - c["cred"], 4),
            }
            # Every emitted candidate has had its product built already — either by
            # `_products_by_anchor` during the scan or by `replace_fragment` just
            # above. Handing it back lets a caller that wants the molecule skip
            # re-running the same reaction; `search_plan._apply` does. Advisory: the
            # key is absent if neither path produced one, and a caller must still
            # canonicalise and re-check its own guard, since this one was checked
            # against `scaffold_smarts` and nothing else.
            if product is not None:
                cand["product"] = product
            if return_delta:
                cand["delta"] = delta
            out.append(cand)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mol", help="molecule SMILES")
    ap.add_argument("--constraints",
                    help='JSON {prop: [lo, hi]} (null for open), e.g. '
                         '\'{"logP": [0, 1.0], "MW": [80, 120]}\'')
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--max-cut", type=int, default=3, choices=[1, 2, 3])
    ap.add_argument("--no-context", action="store_true", help="use radius-0 Δ only")
    ap.add_argument("--scaffold-smarts", default=None, action="append",
                    help="only suggest edits whose product still matches this SMARTS. "
                         "Repeat the flag to require several patterns at once (each is "
                         "checked on its own — do NOT join them with '.').")
    ap.add_argument("--props", default=None,
                    help="optional JSON of precomputed current property values")
    ap.add_argument("--build-cache", action="store_true",
                    help="(re)build the disk cache for max_cut 1,2,3 in MMP_MOVES_DIR and exit")
    args = ap.parse_args()

    if args.build_cache:
        for mc in (1, 2, 3):
            print("wrote", build_cache(MOVES_DIR, mc))
        return
    if not args.mol or not args.constraints:
        ap.error("--mol and --constraints are required (unless --build-cache)")
    props = json.loads(args.props) if args.props else None
    out = suggest_edits(args.mol, json.loads(args.constraints), args.top_k, props=props,
                        max_cut=args.max_cut, context_aware=not args.no_context,
                        scaffold_smarts=args.scaffold_smarts)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
