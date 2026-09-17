"""Which candidate does the exhaustive search's WINNING path go through, and what — in
information available BEFORE committing — separates it from the ones that dead-end?

Motivation
----------
On the depth-3 dumps the winning first move is inside ``suggest_edits``' top-4 on
**100%** of the instances where greedy failed and exhaustive succeeded, and its rank
inside that four is essentially uniform (24/29/24/23% on benchmark_fg). So the
candidate list is not the bottleneck and ``predicted_gap`` — the score that produced
the order — carries almost no signal about which branch reaches the box. Something
else in the state or in the rule does, and a model that reasons about THAT can skip
the tree. This module builds the dataset needed to find out what.

The labelled unit is a (molecule, candidate) pair
-------------------------------------------------
"The move exhaustive committed" is the wrong label: exhaustive returns the shallowest
satisfying path, so a second branch that also satisfies is silently a negative. Here
every candidate is labelled on its own merit — expand it, then run the exhaustive
search from the product with the remaining budget, and record whether that SUBTREE
reaches the property box (``y_success``) and how deep it had to go (``y_depth``).
Several candidates per node can be positive, which is the honest structure of the
problem: the question a model faces is not "which one is THE move" but "which of these
lead anywhere".

Features are what a reasoner could actually know
------------------------------------------------
Every column is computable from the current molecule, the constraints, the current
measured properties and the candidate itself — no lookahead, no measurement of the
product. They fall in four families:

``st_*``  state: how far the molecule is from the box, per property and in aggregate,
          how much room is left under the upper bounds, size, free attachment sites,
          functional groups present, edit budget left.
``c_*``   the objective's own view: predicted_gap and its rank, but also the pieces the
          scalar throws away — which constraints the predicted Δ would newly satisfy,
          which SATISFIED ones it would break, whether it moves the binding (worst)
          property, how much of the required direction it covers (cosine), overshoot
          past the far edge of the box, and the move's own Δ spread (reliability).
``r_*``   rule structure: attach / delete / swap, cut count, heavy-atom and ring and
          rotatable-bond and H-bond deltas of the fragment itself, halogens, sp3.
``fg_*``  functional groups created and destroyed by the rule, against the same
          61-pattern catalog the benchmark scorer uses.

``--analyze`` then reports, per feature, the AUC for "this branch leads to the box", a
standardised logistic fit, a depth-3 tree as text rules, and — the number that matters
for prompting — the top-1 accuracy of picking a branch by each score alone.

What it found (900 benchmark_fg + 832 scaffold instances, top_k=4, depth 5)
-------------------------------------------------------------------------------
Nothing available before committing separates the siblings. Within-instance AUC is
0.475-0.53 for every one of the ~40 features, and picking a branch by ANY of them —
the shipped gap, satisfied-count, damage, cosine, overshoot, fragment size, functional
groups, Δ spread, or an out-of-fold logistic on all of them at once — lands at
61.6-65.1% against a 62.3/62.9% random baseline and an 87.6/81.0% oracle. Pooled AUC
looks better than that only because the ``st_*`` columns are constant across the four
siblings: they rank INSTANCES by difficulty, which is not the choice being made.

Two things do carry signal, and they bound what reasoning can buy:
* "how FAST" is mildly predictable where "whether" is not — on ``y_short`` the
  logistic reaches 43.2% / 57.6% against 39.6% / 53.2% random.
* MEASURING the product's real gap after one edit beats every a-priori score, and
  still only reaches 66.4% / 68.8%. So the determinant of a finishable branch is not
  local to the step at all, which is why the beam arm (four measurements per round,
  same candidates) scores 95.1% where greedy scores 61.6%.

The actionable prior is structural, not selective: winning paths are coarse-to-fine
(mean Δheavy +9.1, +5.5, +3.2, +1.5 by step; the largest edit comes first in 94% of
multi-step exhaustive solutions) and shorter than greedy's (2.33 vs 2.91 edits).

Why the ``site_*`` columns are flat, which is a fact about the CANDIDATE SET
--------------------------------------------------------------------------
The attachment site is molecular context that genuinely varies per candidate — except
that it does not, here: ``suggest_edits``' top-4 are four DIFFERENT rules pinned to the
SAME atom on 90% (fg) / 78% (scaffold) of instances. The site columns are therefore
constant inside 73-99% of the choice sets and cannot discriminate by construction, so
their ~0.50 AUC is not evidence that site context is irrelevant — it is evidence that
the tool never offers a choice about it. Of the rule-environment columns only
``ctx_std_r0`` survives (0.469: a tighter radius-0 Δ spread is likelier to be optimal).
Note ``ctx_support`` had to be recovered from ``metas[i][2]``: the MMP pair count exists
in the move index but ``suggest_edits`` does not put it in the candidate dict.

Usage::

    python -m 3_toolchain_gen.branch_features \\
        --input data/training_data/instances/benchmark_fg/generation_benchmark-00000.jsonl \\
        --limit 900 --top-k 4 --max-depth 5 --num-procs 96 --gpus 0,1,2,3,4,5,6,7 \\
        --output data/analysis/branch_features/fg

    python -m 3_toolchain_gen.branch_features --analyze --output <same dir>
"""

from __future__ import annotations

import argparse
import functools
import json
import logging
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Properties whose values are integers on a real molecule; a predicted fractional Δ
# cannot be realised, so the count-style features round it first (same rule as
# molkit.utils.suggest_edits_sat).
_INT_PROPS = frozenset({"HBD", "HBA", "rotB", "rings_total", "heavy_atoms",
                        "formal_charge"})
_EPS = 1e-9

# One line per feature, so a ranking table can be read without going back to the code.
# Availability tier: 1 = already in the prompt (tool response + the last
# analyze_properties + the constraint box), 2 = computable from the SMILES, 3 = inside
# the move index and NOT exposed by suggest_edits today.
FEATURE_DOC: dict[str, tuple[int, str]] = {
    # ---- state of the current molecule (constant across siblings) ----
    "st_n_props": (1, "제약 물성 개수"),
    "st_n_violated": (1, "현재 위반 중인 제약 수"),
    "st_gap": (1, "박스까지 z-정규화 총 거리"),
    "st_worst_z": (1, "가장 나쁜 물성의 거리"),
    "st_room_hi_min": (1, "가장 가까운 상한까지 남은 여유 (음수면 이미 초과)"),
    "st_heavy": (2, "현재 분자의 중원자 수"),
    "st_free_h_sites": (2, "자유 H를 가진 원자 수 (붙일 수 있는 자리)"),
    "st_n_fg": (2, "보유 작용기 종류 수"),
    "st_n_fg_total": (2, "보유 작용기 총 개수"),
    "st_depth_left": (0, "남은 편집 예산 (하네스 설정값 — 추론 시 없음)"),
    # ---- the objective's own view of the candidate ----
    "c_rank": (1, "도구가 매긴 순위 (0이 최상위)"),
    "c_pred_gap": (1, "도구 점수: 편집 후 예측 박스 거리"),
    "c_gap_reduction": (1, "예측 gap 감소량 (현재 - 편집 후)"),
    "c_n_sat_now": (1, "현재 만족 중인 제약 수"),
    "c_n_sat_post": (1, "편집 후 만족할 것으로 예측되는 제약 수"),
    "c_dcount": (1, "만족 제약 수의 증가분"),
    "c_damage": (1, "이미 만족한 제약 중 이 편집이 깨뜨릴 개수"),
    "c_fix_worst": (1, "병목(최악) 물성을 개선하는가 (0/1)"),
    "c_worst_after": (1, "편집 후 최악 물성의 잔여 거리"),
    "c_cos": (1, "예측 Δ 벡터와 '필요한 방향'의 코사인"),
    "c_cover_frac": (1, "필요 이동 거리 중 이 편집이 덮는 비율"),
    "c_overshoot": (1, "고치려던 축을 박스 반대편으로 넘기는 정도"),
    "c_prob": (1, "Δ~N(avg,std) 가정 하 기대 만족 제약 수 (std를 쓰는 유일한 점수)"),
    "c_std_sum": (1, "제약 물성 Δ 표준편차의 합 (불확실성)"),
    "c_std_max": (1, "Δ 표준편차의 최댓값"),
    "c_snr_min": (1, "물성별 |Δ평균|/Δstd 의 최솟값 (최악 신뢰도)"),
    "c_prob_all": (1, "모든 제약이 동시에 만족될 확률 (P의 곱 — 성공의 정의에 부합)"),
    "c_prob_min": (1, "물성별 만족 확률의 최솟값 (가장 약한 고리)"),
    "c_n_props_helped": (1, "박스 쪽으로 움직이는 물성 수"),
    "c_n_props_hurt": (1, "박스 반대로 움직이는 물성 수"),
    "c_worst_prop_delta": (1, "병목 물성에 대한 부호 있는 Δ (z 단위)"),
    "c_room_frac_used": (1, "가장 빡빡한 상한의 남은 여유 중 이 편집이 쓰는 비율"),
    # ---- relative to the other candidates on the table ----
    "c_prob_margin": (1, "차선 형제 대비 c_prob 우위 (결정의 명확도)"),
    "c_prob_z": (1, "형제 평균 대비 c_prob이 몇 sd 위인가"),
    "c_is_unique_best": (1, "c_prob의 유일한 최댓값인가"),
    "set_n_cands": (1, "이 노드에서 적용 가능한 후보 수"),
    "set_prob_spread": (1, "형제 간 c_prob 범위 (좁으면 선택이 무의미)"),
    "cand_similarity": (2, "다른 후보 조각들과의 평균 Tanimoto 유사도"),
    # ---- what the walk already learned on this instance ----
    "h_step_index": (1, "지금까지 커밋한 편집 수"),
    "h_has_history": (1, "이전 편집이 있는가 (seed면 0)"),
    "h_pred_error_last": (1, "직전 편집의 예측 Δ vs 실측 Δ 오차 (도구 신뢰도 보정)"),
    "h_gap_improved_last": (1, "직전 편집이 실제로 gap을 줄였는가"),
    "h_same_rule_as_prev": (1, "직전과 같은 rule을 또 쓰는가"),
    "h_motif_repeat": (2, "붙이려는 조각이 이미 분자 안에 있는가"),
    # ---- structure of the rule ----
    "r_is_attach": (2, "순수 부착 편집인가 (from = [*:1])"),
    "r_is_delete": (2, "삭제 편집인가 (to = [*:1][H])"),
    "r_is_swap": (2, "치환 편집인가"),
    "r_n_cuts": (2, "절단점 개수 (1=단일, 2=이중, 3=삼중)"),
    "r_heavy_from": (2, "떼어내는 조각의 중원자 수"),
    "r_heavy_to": (2, "붙이는 조각의 중원자 수"),
    "r_dheavy": (2, "중원자 수 변화 (to - from)"),
    "r_dmw": (2, "분자량 변화"),
    "r_drings_arom": (2, "방향족 고리 수 변화"),
    "r_drings_aliph": (2, "지방족 고리 수 변화"),
    "r_drotb": (2, "회전 가능 결합 수 변화"),
    "r_dhbd": (2, "수소결합 주개(HBD) 수 변화"),
    "r_dhba": (2, "수소결합 받개(HBA) 수 변화"),
    "r_dtpsa": (2, "TPSA 변화"),
    "r_dlogp": (2, "Crippen logP 변화"),
    "r_dhalogen": (2, "할로겐 원자 수 변화"),
    "r_dsp3": (2, "sp3 탄소 분율 변화"),
    "r_dcharge": (2, "형식 전하 변화"),
    "r_dsa": (2, "합성 난이도(SA score) 변화"),
    "r_dqed": (2, "QED 변화 (조각 수준)"),
    "r_dtpsa_per_heavy": (2, "원자당 TPSA 변화 (크기 보정)"),
    "r_dlogp_per_heavy": (2, "원자당 logP 변화 (크기 보정)"),
    "r_dhba_per_heavy": (2, "원자당 HBA 변화 (크기 보정)"),
    "r_dpolar_frac": (2, "원자당 (HBA+HBD) 변화 — 조각의 극성 밀도"),
    # ---- functional groups the rule creates / destroys ----
    "fg_n_created": (2, "이 편집이 새로 만드는 작용기 개수 (61-패턴 카탈로그)"),
    "fg_n_destroyed": (2, "이 편집이 없애는 작용기 개수"),
    # ---- molecular context of the attachment site ----
    "site_aromatic": (2, "결합 자리 원자가 방향족인가"),
    "site_in_ring": (2, "결합 자리가 고리에 속하는가"),
    "site_ring_size": (2, "그 고리의 최소 크기 (고리 밖이면 0)"),
    "site_fused": (2, "결합 자리가 융합 고리에 속하는가"),
    "site_is_hetero": (2, "결합 자리가 헤테로원자인가"),
    "site_degree": (2, "결합 자리의 결합 차수"),
    "site_num_h": (2, "결합 자리가 가진 수소 수"),
    "site_charge": (2, "결합 자리의 Gasteiger 부분전하"),
    "site_logp_contrib": (2, "결합 자리 원자의 Crippen logP 기여"),
    "site_sym_equiv": (2, "대칭적으로 동등한 자리의 개수"),
    "site_n_hetero_2b": (2, "2결합 이내 헤테로원자 수"),
    "site_on_fg": (2, "결합 자리가 작용기 위에 있는가"),
    "site_dist_to_fg": (2, "가장 가까운 작용기까지의 결합 거리"),
    "site_crowd_2b": (2, "2결합 이내 중원자 수 (혼잡도)"),
    "site_on_guard": (1, "결합 자리가 보존해야 할 guard 영역 위인가"),
    "site_dist_to_guard": (1, "guard 영역까지의 결합 거리 (코어 vs 주변부)"),
    "site_frac_guard": (1, "분자 중 guard가 차지하는 원자 비율"),
    "site_n_anchors": (1, "이 편집이 고정하는 anchor 개수"),
    "site_ok": (0, "자리 계산 성공 플래그 (bookkeeping)"),
    # ---- rule environment, from the move index (not exposed by the tool) ----
    "ctx_support": (3, "이 rule을 뒷받침하는 MMP pair 수"),
    "ctx_log_support": (3, "log(1 + MMP pair 수)"),
    "ctx_std_r0": (3, "radius-0에서의 Δ 산포 (z 단위 합)"),
    "ctx_shift": (3, "문맥 보정이 radius-0 Δ를 움직인 거리"),
    "ctx_fired": (3, "문맥 보정이 실제로 작동했는가"),
}


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------
def _box(targets: dict, p: str) -> tuple:
    lo, hi = targets[p]
    return (-math.inf if lo is None else float(lo),
            math.inf if hi is None else float(hi))


def _dist(v: float, lo: float, hi: float) -> float:
    """Distance to the box on one axis (0 inside)."""
    if v < lo:
        return lo - v
    if v > hi:
        return v - hi
    return 0.0


_SASCORER = None


def _sascore(mol) -> float:
    """RDKit's synthetic-accessibility score (1 easy .. 10 hard), from Contrib."""
    global _SASCORER
    if _SASCORER is None:
        import os
        import sys as _sys
        from rdkit.Chem import RDConfig
        _sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
        import sascorer
        _SASCORER = sascorer
    try:
        return float(_SASCORER.calculateScore(mol))
    except Exception:  # noqa: BLE001 - a fragment SA is best-effort
        return 0.0


@functools.lru_cache(maxsize=32768)
def _frag_desc(smiles: str) -> dict:
    """RDKit descriptors of one side of a rule, with the attachment points capped.

    ``[*:1]`` is replaced by hydrogen rather than deleted so that the fragment is a
    real molecule and its ring/rotatable-bond counts mean what they say. A pure attach
    (``from`` = ``[*:1]``) therefore comes out as all-zero, which is exactly right: it
    removes nothing.

    Cached: the argument is a rule side, and the rule vocabulary is ~13k distinct
    fragments paired up, so a depth-5 expansion asks the same few thousand questions
    tens of thousands of times. QED, Crippen and the SA score inside are the three most
    expensive main-thread entries in a profile of the expansion without this. Callers
    only read the result — nobody mutates it — so one dict per fragment is safe to
    share. Returned dicts must stay read-only for that reason.
    """
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return {}
    ed = Chem.RWMol(m)
    for atom in ed.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetAtomicNum(1)
            atom.SetAtomMapNum(0)
    try:
        mol = ed.GetMol()
        Chem.SanitizeMol(mol)
    except Exception:  # noqa: BLE001 - an unsanitisable side contributes nothing
        return {}
    return {
        "heavy": mol.GetNumHeavyAtoms(),
        "rings_arom": rdMolDescriptors.CalcNumAromaticRings(mol),
        "rings_aliph": rdMolDescriptors.CalcNumAliphaticRings(mol),
        "rotb": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "hbd": rdMolDescriptors.CalcNumHBD(mol),
        "hba": rdMolDescriptors.CalcNumHBA(mol),
        "tpsa": rdMolDescriptors.CalcTPSA(mol),
        "logp": Crippen.MolLogP(mol),
        "mw": Descriptors.MolWt(mol),
        "halogen": sum(1 for a in mol.GetAtoms()
                       if a.GetSymbol() in ("F", "Cl", "Br", "I")),
        "sp3": rdMolDescriptors.CalcFractionCSP3(mol),
        "charge": sum(a.GetFormalCharge() for a in mol.GetAtoms()),
        "sa": _sascore(mol) if mol.GetNumHeavyAtoms() else 0.0,
        "qed": _qed(mol),
    }


def _qed(mol) -> float:
    from rdkit.Chem import QED
    try:
        return float(QED.qed(mol))
    except Exception:  # noqa: BLE001
        return 0.0


_FG_PATTERNS: Optional[list] = None


def _fg_patterns() -> list:
    """The 61 broadest_only ``fr_*`` patterns — the same set the grader scores with."""
    global _FG_PATTERNS
    if _FG_PATTERNS is None:
        from rdkit import Chem

        from molkit.utils.fragments import fr_catalog
        pats = []
        for key, meta in fr_catalog(broadest_only=True).items():
            patt = Chem.MolFromSmarts(meta["smarts"]) if meta.get("smarts") else None
            if patt is not None:
                pats.append((meta.get("name") or key, patt))
        _FG_PATTERNS = pats
    return _FG_PATTERNS


@functools.lru_cache(maxsize=32768)
def _fg_counts(smiles: str) -> dict:
    """Functional-group counts. Cached for the same reason as :func:`_frag_desc`.

    Two of the three call sites pass a rule side (a few thousand distinct strings, hit
    constantly); the third passes the current molecule, which is unique per node and
    only ever evicts. The bound is what keeps that third caller from growing the cache
    without limit while the fragments stay resident. Read-only, like `_frag_desc`.
    """
    from rdkit import Chem
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return {}
    out = {}
    for name, patt in _fg_patterns():
        n = len(m.GetSubstructMatches(patt, uniquify=True))
        if n:
            out[name] = n
    return out


def _move_row(from_smiles: str, to_smiles: str, max_cut: int):
    """``(Δ mean, Δ std, MMP support)`` at radius 0 for this rule, from the move index.

    The candidate dict only carries the CONTEXT-REFINED Δ, so the raw row is the other
    half of a pair: their difference is how much the molecule's own environment moved
    the prediction, and ``support`` is how many matched-molecular-pairs the rule rests
    on. Neither is visible to a caller of ``suggest_edits`` today.
    """
    from molkit.utils.suggest_edits import MOVES_DIR, _by_from, _load_all
    metas, D, S, _ctx = _load_all(MOVES_DIR, max_cut)
    idx = _by_from(MOVES_DIR, max_cut, metas).get(from_smiles)
    if idx is None:
        return None
    for i in idx:
        if metas[i][1] == to_smiles:
            return D[i], S[i], metas[i][2]
    return None


def _site_features(cur: str, cand: dict, guard_smarts=None) -> dict:
    """The MOLECULAR CONTEXT of the attachment site itself.

    Two candidates in one list often carry a similar fragment and differ mainly in
    WHERE it goes, so unlike the state columns these vary between siblings and can
    inform the choice. Everything is read off the anchor atom the candidate pins.
    """
    from rdkit import Chem
    from rdkit.Chem import Crippen, rdmolops
    from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges

    out = {"site_ok": 0, "site_n_anchors": len(cand.get("anchors") or {})}
    m = Chem.MolFromSmiles(cur)
    anchors = [int(v) for v in (cand.get("anchors") or {}).values()]
    if m is None or not anchors:
        return out
    i = anchors[0]
    if i >= m.GetNumAtoms():
        return out
    a = m.GetAtomWithIdx(i)
    ri = m.GetRingInfo()
    try:
        ComputeGasteigerCharges(m)
        q = float(a.GetProp("_GasteigerCharge"))
        q = 0.0 if math.isnan(q) else q
    except Exception:  # noqa: BLE001 - charge is best-effort
        q = 0.0
    try:
        logp_contrib = Crippen.rdMolDescriptors._CalcCrippenContribs(m)[i][0]
    except Exception:  # noqa: BLE001
        logp_contrib = 0.0

    dmat = rdmolops.GetDistanceMatrix(m)
    # Is the edit touching the region the task told us to PRESERVE, or the decoration
    # hanging off it? At the seed everything is the guard, but deeper down this is the
    # core-vs-periphery distinction, and the guard SMARTS is known at inference.
    guard_atoms: set = set()
    for gs in (guard_smarts or []):
        patt = Chem.MolFromSmarts(gs) if isinstance(gs, str) else gs
        if patt is None:
            continue
        for match in m.GetSubstructMatches(patt, uniquify=True):
            guard_atoms.update(match)
    hetero = [x.GetIdx() for x in m.GetAtoms() if x.GetAtomicNum() not in (1, 6)]
    fg_atoms = set()
    for _name, patt in _fg_patterns():
        for match in m.GetSubstructMatches(patt, uniquify=True):
            fg_atoms.update(match)
    # Symmetry-equivalent sites: how many other atoms are indistinguishable from this
    # one. A unique site is a different kind of decision from one of six equivalent
    # aromatic CHs.
    ranks = list(Chem.CanonicalRankAtoms(m, breakTies=False))
    sym = sum(1 for r in ranks if r == ranks[i])

    out.update({
        "site_ok": 1,
        "site_aromatic": int(a.GetIsAromatic()),
        "site_in_ring": int(a.IsInRing()),
        "site_ring_size": min((s for s in (3, 4, 5, 6, 7, 8)
                               if ri.IsAtomInRingOfSize(i, s)), default=0),
        "site_fused": int(ri.NumAtomRings(i) > 1),
        "site_is_hetero": int(a.GetAtomicNum() not in (1, 6)),
        "site_degree": a.GetDegree(),
        "site_num_h": a.GetTotalNumHs(),
        "site_charge": round(q, 4),
        "site_logp_contrib": round(float(logp_contrib), 4),
        "site_sym_equiv": sym,
        "site_n_hetero_2b": sum(1 for h in hetero if h != i and dmat[i][h] <= 2),
        "site_on_fg": int(i in fg_atoms),
        "site_dist_to_fg": (0 if i in fg_atoms else
                            int(min((dmat[i][j] for j in fg_atoms), default=99))),
        # Crowding: heavy atoms within two bonds, i.e. how much room the new fragment
        # has where it is being put.
        "site_crowd_2b": int(sum(1 for j in range(m.GetNumAtoms())
                                 if j != i and dmat[i][j] <= 2)),
        "site_on_guard": int(i in guard_atoms),
        "site_dist_to_guard": (0 if i in guard_atoms else
                               int(min((dmat[i][j] for j in guard_atoms), default=99))),
        "site_frac_guard": (round(len(guard_atoms) / m.GetNumAtoms(), 4)
                            if m.GetNumAtoms() else 0.0),
    })
    return out


def _free_h_sites(smiles: str) -> int:
    from rdkit import Chem
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return 0
    return sum(1 for a in m.GetAtoms() if a.GetTotalNumHs() > 0)


def candidate_features(cur: str, cur_props: dict, targets: dict, cand: dict,
                       rank: int, depth_left: int, scale: dict,
                       state_cache: Optional[dict] = None,
                       max_cut: int = 3, guard_smarts=None,
                       sibling_ctx: Optional[dict] = None) -> dict:
    """One feature row for *cand* at molecule *cur*. No lookahead is used."""
    keys = [p for p in targets if cur_props.get(p) is not None]
    v = {p: float(cur_props[p]) for p in keys}
    box = {p: _box(targets, p) for p in keys}
    sc = {p: float(scale.get(p, 1.0)) or 1.0 for p in keys}

    d_now = {p: _dist(v[p], *box[p]) / sc[p] for p in keys}
    sat_now = {p: d_now[p] <= _EPS for p in keys}
    gap_now = sum(d_now.values())
    worst_p = max(keys, key=lambda p: d_now[p]) if keys else None

    delta = cand.get("delta") or {}
    dv, ds = {}, {}
    for p in keys:
        e = delta.get(p) or {}
        a = float(e.get("avg") or 0.0)
        dv[p] = round(a) if p in _INT_PROPS else a
        ds[p] = float(e.get("std") or 0.0)
    post = {p: v[p] + dv[p] for p in keys}
    d_post = {p: _dist(post[p], *box[p]) / sc[p] for p in keys}
    sat_post = {p: d_post[p] <= _EPS for p in keys}

    # How far, and in which direction, each axis still has to move (z units, signed).
    req = {}
    for p in keys:
        lo, hi = box[p]
        req[p] = (lo - v[p]) / sc[p] if v[p] < lo else ((hi - v[p]) / sc[p]
                                                        if v[p] > hi else 0.0)
    mv = {p: dv[p] / sc[p] for p in keys}
    nr = math.sqrt(sum(x * x for x in req.values()))
    nm = math.sqrt(sum(x * x for x in mv.values()))
    cos = (sum(req[p] * mv[p] for p in keys) / (nr * nm)) if nr > _EPS and nm > _EPS else 0.0
    covered = sum(min(abs(mv[p]), abs(req[p])) * (1 if mv[p] * req[p] > 0 else 0)
                  for p in keys)

    # Overshoot: how far past the FAR edge of the box the move would carry an axis it
    # was supposed to be fixing — the failure mode a distance-reducing score hides.
    over = 0.0
    for p in keys:
        lo, hi = box[p]
        if req[p] > 0 and post[p] > hi:
            over = max(over, (post[p] - hi) / sc[p])
        elif req[p] < 0 and post[p] < lo:
            over = max(over, (lo - post[p]) / sc[p])

    snr = [abs(dv[p]) / max(ds[p], 1e-6) for p in keys if abs(dv[p]) > 0]
    # Per-property probability of landing inside the box under the move's own Δ spread.
    # ``prob`` sums them (expected COUNT satisfied) — but success needs EVERY constraint
    # to hold, so the product is the quantity that actually matters and the minimum is
    # its weakest link. The sum was the best single feature measured so far, and it is
    # the crudest of the three, which is why all three are kept.
    pvec = []
    try:
        from scipy.special import ndtr
        for p in keys:
            lo, hi = box[p]
            sg = max(ds[p], 1e-6)
            up = 1.0 if hi == math.inf else float(ndtr((hi - post[p]) / sg))
            dn = 0.0 if lo == -math.inf else float(ndtr((lo - post[p]) / sg))
            pvec.append(max(0.0, min(1.0, up - dn)))
    except Exception:  # noqa: BLE001 - scipy optional
        pvec = [1.0 if sat_post[p] else 0.0 for p in keys]
    prob = float(sum(pvec))
    prob_all = 1.0
    for pv in pvec:          # not `v`: that name holds the measured property vector
        prob_all *= pv
    prob_min = min(pvec) if pvec else 0.0

    # Direction bookkeeping the aggregate scores throw away.
    helped = sum(1 for p in keys if d_post[p] < d_now[p] - _EPS)
    hurt = sum(1 for p in keys if d_post[p] > d_now[p] + _EPS)
    worst_delta = (mv[worst_p] if worst_p else 0.0)
    # How much of the tightest upper bound's remaining room this edit spends. The
    # earlier stratified analysis found the size preference flips with headroom; this
    # is that interaction as one number.
    room_used = 0.0
    for p in keys:
        lo, hi = box[p]
        if hi == math.inf or dv[p] <= 0:
            continue
        room = (hi - v[p]) / sc[p]
        if room > _EPS:
            room_used = max(room_used, (dv[p] / sc[p]) / room)

    fs, ts = cand.get("from_smiles", ""), cand.get("to_smiles", "")
    f_desc, t_desc = _frag_desc(fs), _frag_desc(ts)

    def d(k):
        return float(t_desc.get(k, 0) or 0) - float(f_desc.get(k, 0) or 0)

    fg_from, fg_to = _fg_counts(fs), _fg_counts(ts)
    created = {k: fg_to[k] - fg_from.get(k, 0) for k in fg_to
               if fg_to[k] - fg_from.get(k, 0) > 0}
    destroyed = {k: fg_from[k] - fg_to.get(k, 0) for k in fg_from
                 if fg_from[k] - fg_to.get(k, 0) > 0}

    state = state_cache if state_cache is not None else {}
    if "st_n_fg" not in state:
        fg_cur = _fg_counts(cur)
        state["st_n_fg"] = len(fg_cur)
        state["st_n_fg_total"] = sum(fg_cur.values())
        state["st_fg_present"] = sorted(fg_cur)
        state["st_free_h_sites"] = _free_h_sites(cur)
        from rdkit import Chem
        m = Chem.MolFromSmiles(cur)
        state["st_heavy"] = m.GetNumHeavyAtoms() if m is not None else 0

    row = {
        # ---- state ----
        "st_n_props": len(keys),
        "st_n_violated": sum(1 for p in keys if not sat_now[p]),
        "st_gap": round(gap_now, 4),
        "st_worst_z": round(max(d_now.values()), 4) if keys else 0.0,
        "st_worst_prop": worst_p,
        "st_depth_left": depth_left,
        "st_heavy": state["st_heavy"],
        "st_free_h_sites": state["st_free_h_sites"],
        "st_n_fg": state["st_n_fg"],
        "st_n_fg_total": state["st_n_fg_total"],
        # Room left before the NEAREST upper bound (z units): small means the molecule
        # can only afford small additions, which is the constraint a size-blind ranking
        # walks straight through.
        "st_room_hi_min": round(min(((box[p][1] - v[p]) / sc[p]
                                     for p in keys if box[p][1] != math.inf),
                                    default=99.0), 4),
        # ---- the objective's own view ----
        "c_rank": rank,
        "c_pred_gap": round(float(cand.get("predicted_gap") or 0.0), 4),
        "c_gap_reduction": round(gap_now - sum(d_post.values()), 4),
        "c_n_sat_now": sum(sat_now.values()),
        "c_n_sat_post": sum(sat_post.values()),
        "c_dcount": sum(sat_post.values()) - sum(sat_now.values()),
        # Constraints that are satisfied NOW and would be pushed out by this move.
        "c_damage": sum(1 for p in keys if sat_now[p] and not sat_post[p]),
        "c_fix_worst": int(bool(worst_p) and d_post[worst_p] < d_now[worst_p] - _EPS),
        "c_worst_after": round(max(d_post.values()), 4) if keys else 0.0,
        "c_cos": round(cos, 4),
        "c_cover_frac": round(covered / nr, 4) if nr > _EPS else 0.0,
        "c_overshoot": round(over, 4),
        "c_prob": round(prob, 4),
        "c_prob_all": round(prob_all, 6),
        "c_prob_min": round(prob_min, 4),
        "c_n_props_helped": helped,
        "c_n_props_hurt": hurt,
        "c_worst_prop_delta": round(worst_delta, 4),
        "c_room_frac_used": round(min(room_used, 99.0), 4),
        "c_std_sum": round(sum(ds.values()), 4),
        "c_std_max": round(max(ds.values()), 4) if keys else 0.0,
        "c_snr_min": round(min(snr), 4) if snr else 0.0,
        # ---- rule structure ----
        "r_is_attach": int(fs.strip() == "[*:1]"),
        "r_is_delete": int(ts.strip() in ("[*:1][H]", "[*:1]")),
        "r_n_cuts": max(fs.count("*:"), ts.count("*:")),
        "r_heavy_from": f_desc.get("heavy", 0),
        "r_heavy_to": t_desc.get("heavy", 0),
        "r_dheavy": d("heavy"),
        "r_dmw": round(d("mw"), 3),
        "r_drings_arom": d("rings_arom"),
        "r_drings_aliph": d("rings_aliph"),
        "r_drotb": d("rotb"),
        "r_dhbd": d("hbd"),
        "r_dhba": d("hba"),
        "r_dtpsa": round(d("tpsa"), 3),
        "r_dlogp": round(d("logp"), 3),
        "r_dhalogen": d("halogen"),
        "r_dsp3": round(d("sp3"), 3),
        "r_dcharge": d("charge"),
        "r_dsa": round(d("sa"), 4),
        "r_dqed": round(d("qed"), 4),
        # Intensive versions: the extensive Δs above are dominated by "the fragment is
        # big", which correlates with everything. Dividing by the atoms added leaves
        # the fragment's CHARACTER, which is the part a chemist would reason about.
        "r_dtpsa_per_heavy": round(d("tpsa") / max(abs(d("heavy")), 1), 4),
        "r_dlogp_per_heavy": round(d("logp") / max(abs(d("heavy")), 1), 4),
        "r_dhba_per_heavy": round(d("hba") / max(abs(d("heavy")), 1), 4),
        "r_dpolar_frac": round((d("hba") + d("hbd")) / max(abs(d("heavy")), 1), 4),
        # ---- functional groups ----
        "fg_n_created": sum(created.values()),
        "fg_n_destroyed": sum(destroyed.values()),
        "fg_created": sorted(created),
        "fg_destroyed": sorted(destroyed),
        "rule": f"{fs}->{ts}",
    }
    row["r_is_swap"] = int(not row["r_is_attach"] and not row["r_is_delete"])

    # ---- molecular context: the site, and how well the rule's own environment fits ----
    row.update(_site_features(cur, cand, guard_smarts))
    mr = _move_row(fs, ts, max_cut)
    if mr is not None:
        d0, s0, sup = mr
        from molkit.utils.suggest_edits import _PIDX
        shift = 0.0
        for p in keys:
            j = _PIDX.get(p)
            if j is None:
                continue
            raw = round(float(d0[j])) if p in _INT_PROPS else float(d0[j])
            shift += abs(dv[p] - raw) / sc[p]
        row.update({
            "ctx_support": int(sup),
            "ctx_log_support": round(math.log1p(float(sup)), 4),
            "ctx_shift": round(shift, 4),
            "ctx_fired": int(shift > 1e-6),
            "ctx_std_r0": round(float(sum(abs(float(s0[_PIDX[p]])) / sc[p]
                                          for p in keys if p in _PIDX)), 4),
        })
    return row


# ---------------------------------------------------------------------------
# labelling (worker side)
# ---------------------------------------------------------------------------
async def _label_instance(rec: dict, idx: int, cfg: dict) -> list:
    import asyncio

    from .builder import guard_smarts, seed_for
    from .constraints import extract_properties
    from .local_tools import local_call_async, local_call_batch
    from . import search_plan
    from .search_plan import (_all_satisfied, _apply, _canon, _guard_mols, _norm_gap,
                              exhaustive_search, measure_with_retry, plan_search)

    scaffold = seed_for(rec)
    smarts = guard_smarts(rec)
    targets = extract_properties(rec)
    if not scaffold or not smarts or not targets:
        return []
    scaffold = _canon(scaffold)
    if scaffold is None:
        return []
    props_list = list(targets)
    memo: dict = {}

    def _complete(res) -> bool:
        return isinstance(res, dict) and all(res.get(p) is not None for p in props_list)

    async def measure(smi: str, props: list) -> dict:
        if smi in memo:
            return dict(memo[smi])
        res = await local_call_async(
            "analyze_properties", {"mol_smiles": smi, "property_names": props})
        res = res if isinstance(res, dict) else {}
        if _complete(res):
            memo[smi] = dict(res)
        return res

    async def measure_batch(smis: list, props: list) -> dict:
        todo = [s for s in dict.fromkeys(smis) if s not in memo]
        got = (await asyncio.to_thread(local_call_batch, "analyze_properties", todo,
                                       property_names=props_list) if todo else {})
        out = {}
        for s in dict.fromkeys(smis):
            res = (got or {}).get(s)
            if not isinstance(res, dict) or not _complete(res):
                res = memo.get(s) or await measure_with_retry(measure, s, props_list,
                                                              retries=cfg["measure_retries"])
            res = res if isinstance(res, dict) else {}
            if _complete(res):
                memo[s] = dict(res)
            out[s] = res
        return out

    seed_props = await measure_with_retry(measure, scaffold, props_list,
                                          retries=cfg["measure_retries"])
    if not seed_props or _all_satisfied(seed_props, targets):
        return []                      # trivial instances decide nothing

    cands = await search_plan._suggest(
        scaffold, targets, seed_props, top_k=cfg["top_k"], max_cut=cfg["max_cut"],
        context_aware=True, scaffold_smarts=smarts)
    if not cands:
        return []

    guard = _guard_mols(smarts)
    from molkit.utils.suggest_edits import _SCALE

    rows, state_cache = [], {}
    for rank, cand in enumerate(cands):
        prod = _apply(scaffold, cand, guard, {scaffold})
        row = candidate_features(scaffold, seed_props, targets, cand, rank,
                                 cfg["max_depth"], _SCALE, state_cache,
                                 max_cut=cfg["max_cut"])
        row.update({"index": idx, "id": rec.get("id"), "seed_smiles": scaffold,
                    "properties": props_list, "seed_gap": round(
                        _norm_gap(seed_props, targets), 4),
                    "applied": prod is not None, "product": prod})
        if prod is None:
            # The rule did not survive the guard/dedup — a negative the model could
            # have known about, so it stays in the table rather than being dropped.
            row.update({"y_success": 0, "y_greedy": 0, "y_short": 0, "y_depth": None,
                        "y_best_gap": None})
            rows.append(row)
            continue
        # The label: does the SUBTREE under this candidate reach the box on the
        # remaining budget? Exhaustive so the answer is about the branch, not about a
        # pick rule applied inside it.
        sub = await exhaustive_search(prod, smarts, targets, measure,
                                      top_k=cfg["top_k"],
                                      max_depth=max(0, cfg["max_depth"] - 1),
                                      max_cut=cfg["max_cut"], context_aware=True,
                                      measure_retries=cfg["measure_retries"],
                                      measure_batch_fn=measure_batch)
        prod_props = await measure(prod, props_list)
        ok_here = _all_satisfied(prod_props, targets) if prod_props else False
        satisfied = bool(ok_here or (sub and sub.get("search_satisfied")))
        depth = 1 if ok_here else (1 + int(sub.get("search_steps_taken", 0))
                                   if satisfied else None)
        best = 0.0 if satisfied else (sub.get("final_gap") if sub else None)
        # The GREEDY label is the decision-relevant one. Under four more levels of
        # exhaustive lookahead almost every branch eventually reaches the box (~90% on
        # a smoke run), so "can this branch ever work" barely discriminates — what a
        # model without search needs to know is whether the branch is finishable by the
        # cheap policy it will actually run.
        greedy = (None if ok_here else
                  await plan_search(prod, smarts, targets, measure,
                                    max_steps=max(0, cfg["max_depth"] - 1),
                                    top_k=cfg["top_k"], max_cut=cfg["max_cut"],
                                    context_aware=True,
                                    measure_retries=cfg["measure_retries"]))
        row.update({
            "y_success": int(satisfied),
            "y_depth": depth,
            # Reached in at most two more edits — "how fast", which is where the signal
            # lives once "eventually" is nearly always true.
            "y_short": int(depth is not None and depth <= 2),
            "y_greedy": int(ok_here or bool(greedy and greedy.get("search_satisfied"))),
            "y_best_gap": None if best in (None, float("inf")) else round(float(best), 4),
            # One-step outcome, kept separate: it is the thing a measurement would tell
            # the model immediately, so features that only predict THIS are cheap.
            "y_gap_after": (round(_norm_gap(prod_props, targets), 4)
                            if prod_props else None),
        })
        rows.append(row)
    return rows


def _worker_main(worker_id: int, records_path: str, gpus: list, cfg: dict,
                 out_path: str, ret_q, cursor, progress=None) -> None:
    import asyncio
    import signal

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    with open(records_path) as fh:
        records = [json.loads(line) for line in fh if line.strip()]

    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(v, "1")
    os.environ.setdefault("MALLOC_ARENA_MAX", "2")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[worker_id % len(gpus)])
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    from . import http_client, local_tools
    http_client.set_local_mode(True)
    try:
        from molkit.utils import suggest_edits as se
        metas, _D, _S, _c = se._load_all(se.MOVES_DIR, cfg["max_cut"])
        se._by_from(se.MOVES_DIR, cfg["max_cut"], metas)
    except Exception as exc:  # noqa: BLE001
        logger.warning("move-index warmup skipped: %s", exc)
    local_tools.preload(with_admet=True)
    _fg_patterns()

    def take_next() -> Optional[int]:
        with cursor.get_lock():
            i = cursor.value
            if i >= len(records):
                return None
            cursor.value = i + 1
        return i

    async def main() -> int:
        lock = asyncio.Lock()
        written = 0

        async def consume():
            nonlocal written
            while True:
                idx = take_next()
                if idx is None:
                    return
                try:
                    rows = await _label_instance(records[idx], idx, cfg)
                except Exception as exc:  # noqa: BLE001
                    logger.exception("instance %d failed: %s", idx, exc)
                    rows = []
                finally:
                    if progress is not None:
                        with progress.get_lock():
                            progress.value += 1
                if not rows:
                    continue
                async with lock:
                    with open(out_path, "a") as fh:
                        for r in rows:
                            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                    written += len(rows)

        await asyncio.gather(*(consume() for _ in range(max(1, cfg["per_proc"]))))
        return written

    ret_q.put((worker_id, asyncio.run(main())))


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------
def _auc(y: list, x: list) -> float:
    """Rank AUC of score *x* for label *y* (0.5 = no information)."""
    import numpy as np
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    pos, neg = y == 1, y == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(1, len(x) + 1)
    # average ranks for ties, else a constant feature scores != 0.5
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2)
                 / (pos.sum() * neg.sum()))


TARGETS = {
    "y_greedy": "branch is FINISHABLE BY THE GREEDY POLICY (the deployment question)",
    "y_short": "branch reaches the box within 2 more edits (how fast, not whether)",
    "y_success": "branch reaches the box under full exhaustive lookahead",
}


def analyze(out_dir: Path, top_n: int = 20, targets: Optional[list] = None) -> None:
    import pandas as pd

    path = out_dir / "candidates.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} not found — run the labelling pass first")
    df = pd.read_json(path, lines=True)
    report: dict = {"n_rows": len(df), "n_instances": int(df["index"].nunique())}
    for target in (targets or [t for t in TARGETS if t in df.columns]):
        print("\n" + "=" * 78)
        print(f"TARGET {target} — {TARGETS.get(target, '')}")
        print("=" * 78)
        report[target] = _analyze_target(df, target, out_dir, top_n)
    with open(out_dir / "feature_report.json", "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {out_dir / 'feature_report.json'}")


def _analyze_target(df, target: str, out_dir: Path, top_n: int) -> dict:
    import pandas as pd

    print(f"{len(df)} candidates over {df['index'].nunique()} instances "
          f"| positives {df[target].mean() * 100:.1f}% "
          f"| instances with >=1 positive "
          f"{df.groupby('index')[target].max().mean() * 100:.1f}%")
    n_pos = df.groupby("index")[target].sum()
    print("positive branches per instance: "
          + ", ".join(f"{k}:{v}" for k, v in sorted(n_pos.value_counts().items())))

    feats = [c for c in df.columns
             if (c.startswith(("st_", "c_", "r_", "fg_n_", "site_", "ctx_"))
                 and c != "st_worst_prop" and pd.api.types.is_numeric_dtype(df[c]))]
    y = df[target].to_numpy()

    print("\n=== single-feature AUC, POOLED (0.5 = no information) ===")
    aucs = sorted(((_auc(y, df[c].fillna(0).to_numpy()), c) for c in feats),
                  key=lambda t: -abs(t[0] - 0.5))
    for a, c in aucs[:top_n]:
        bar = "#" * int(abs(a - 0.5) * 100)
        print(f"  {c:<18} {a:.3f}  {bar}")
    print("  (pooled AUC is mostly BETWEEN-instance: every st_* column and any c_*"
          " column\n   built from the state alone is constant across the four siblings,"
          " so it can\n   rank instances by difficulty while saying nothing about"
          " which branch to take.)")

    # The decision the model actually faces is WITHIN one candidate list, so centre
    # every feature on its instance mean first: what survives is the part that varies
    # between siblings, and only that part can inform a choice.
    print("\n=== single-feature AUC, WITHIN INSTANCE (the choice the model faces) ===")
    cen = df.groupby("index")[feats].transform(lambda s: s - s.mean())
    caucs = sorted(((_auc(y, cen[c].fillna(0).to_numpy()), c) for c in feats),
                   key=lambda t: -abs(t[0] - 0.5))
    for a, c in caucs[:top_n]:
        if math.isnan(a):
            continue
        bar = "#" * int(abs(a - 0.5) * 100)
        print(f"  {c:<18} {a:.3f}  {bar}")

    # Standardised logistic fit — the multivariate view, so a feature that is only
    # a proxy for another drops out.
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier, export_text
    X = df[feats].fillna(0).to_numpy(dtype=float)
    Xs = StandardScaler().fit_transform(X)
    lr = LogisticRegression(max_iter=2000, C=0.5).fit(Xs, y)
    print("\n=== standardised logistic coefficients (multivariate) ===")
    for co, c in sorted(zip(lr.coef_[0], feats), key=lambda t: -abs(t[0]))[:top_n]:
        print(f"  {c:<18} {co:+.3f}")

    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=max(20, len(df) // 100))
    tree.fit(X, y)
    print("\n=== depth-3 tree (human-readable rules) ===")
    print(export_text(tree, feature_names=feats, max_depth=3))

    # The number that matters for prompting: pick ONE branch per instance by a score,
    # how often is it a branch that reaches the box?
    print("\n=== top-1 accuracy: pick one branch per instance by this score alone ===")
    base = df.groupby("index")[target].max().mean()
    scores = {
        "shipped predicted_gap (current tool)": ("c_pred_gap", 1),
        "rank (== predicted_gap order)": ("c_rank", 1),
        "satisfied-count after": ("c_n_sat_post", -1),
        "damage (constraints broken)": ("c_damage", 1),
        "cosine to required direction": ("c_cos", -1),
        "overshoot past far edge": ("c_overshoot", 1),
        "worst-property distance after": ("c_worst_after", 1),
        "expected satisfied count": ("c_prob", -1),
        "fragment heavy atoms added": ("r_dheavy", 1),
    }
    rows = []
    for label, (col, sign) in scores.items():
        if col not in df.columns:
            continue
        pick = df.assign(_k=df[col] * sign).sort_values(
            ["index", "_k"], kind="mergesort").groupby("index").head(1)
        rows.append((label, pick[target].mean()))
    # Held out by instance: an in-sample fit on 40 features would flatter itself.
    from sklearn.model_selection import GroupKFold
    import numpy as np
    oof = np.zeros(len(df))
    groups = df["index"].to_numpy()
    n_splits = min(5, len(np.unique(groups)))
    if n_splits >= 2:
        for tr, te in GroupKFold(n_splits=n_splits).split(Xs, y, groups):
            m = LogisticRegression(max_iter=2000, C=0.5).fit(Xs[tr], y[tr])
            oof[te] = m.predict_proba(Xs[te])[:, 1]
        pick = df.assign(_k=-oof).sort_values(
            ["index", "_k"], kind="mergesort").groupby("index").head(1)
        rows.append(("logistic on all features (out-of-fold)", pick[target].mean()))
        # Same fit on instance-CENTRED features: a conditional model, which is the
        # correctly specified one for a within-list choice.
        Xc = StandardScaler().fit_transform(cen[feats].fillna(0).to_numpy(dtype=float))
        oofc = np.zeros(len(df))
        for tr, te in GroupKFold(n_splits=n_splits).split(Xc, y, groups):
            m = LogisticRegression(max_iter=2000, C=0.5).fit(Xc[tr], y[tr])
            oofc[te] = m.predict_proba(Xc[te])[:, 1]
        pick = df.assign(_k=-oofc).sort_values(
            ["index", "_k"], kind="mergesort").groupby("index").head(1)
        rows.append(("conditional logistic, centred (out-of-fold)", pick[target].mean()))
    rows.append(("random branch (mean positive rate)", df[target].mean()))
    rows.append(("ORACLE (any positive exists)", base))
    for label, acc in rows:
        print(f"  {label:<42} {acc * 100:5.1f}%")

    # Functional groups: which created groups sit on branches that work out.
    fg_stats = {}
    print("\n=== functional groups CREATED by the rule (>=30 occurrences) ===")
    ex = df.explode("fg_created").dropna(subset=["fg_created"])
    if len(ex):
        g = ex.groupby("fg_created")[target].agg(["mean", "count"])
        g = g[g["count"] >= 30].sort_values("mean", ascending=False)
        fg_stats = {str(k): [float(v["mean"]), int(v["count"])] for k, v in g.iterrows()}
        head, tail = list(g.head(6).iterrows()), list(g.tail(6).iterrows())
        for name, r in head + ([("...", None)] if len(g) > 12 else []) + tail:
            if r is None:
                print("  ...")
                continue
            print(f"  {str(name):<28} {r['mean'] * 100:5.1f}%  (n={int(r['count'])})")
    return {"positive_rate": float(df[target].mean()),
            "auc": {c: (None if math.isnan(a) else round(a, 4)) for a, c in aucs},
            "logistic": {c: round(float(co), 4) for c, co in zip(feats, lr.coef_[0])},
            "top1": {k: round(float(v), 4) for k, v in rows},
            "fg_created": fg_stats}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="data/training_data/instances/"
                                       "benchmark_fg/generation_benchmark-00000.jsonl")
    ap.add_argument("--output", default="data/analysis/branch_features")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--scan-limit", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--max-depth", type=int, default=5)
    ap.add_argument("--max-cut", type=int, default=3)
    ap.add_argument("--measure-retries", type=int, default=8)
    ap.add_argument("--num-procs", type=int, default=96)
    ap.add_argument("--per-proc", type=int, default=2)
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--analyze", action="store_true",
                    help="report from an existing --output/candidates.jsonl.")
    args = ap.parse_args(argv)

    out_dir = Path(args.output)
    if args.analyze:
        analyze(out_dir)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    dump = out_dir / "candidates.jsonl"
    if dump.exists():
        if not args.overwrite:
            raise SystemExit(f"{dump} exists; pass --overwrite to replace it.")
        dump.unlink()

    from .compare_strategies import sample_instances
    records = sample_instances(args.input, args.limit, args.seed, args.scan_limit)
    if not records:
        raise SystemExit("no usable instances found")
    logger.info("labelling %d instances, top_k=%d depth=%d", len(records), args.top_k,
                args.max_depth)

    cfg = {"top_k": args.top_k, "max_depth": args.max_depth, "max_cut": args.max_cut,
           "measure_retries": args.measure_retries, "per_proc": args.per_proc}
    num_procs = max(1, min(args.num_procs, len(records)))
    ctx = mp.get_context("spawn")
    ret_q, progress, cursor = ctx.Queue(), ctx.Value("L", 0), ctx.Value("l", 0)
    shards = [out_dir / f".shard_{w:03d}.jsonl" for w in range(num_procs)]
    for s in shards:
        s.unlink(missing_ok=True)
    records_path = out_dir / ".records.jsonl"
    with open(records_path, "w") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    gpus = [int(g) for g in args.gpus.split(",") if g.strip() != ""]
    procs = []
    t0 = time.perf_counter()
    try:
        for w in range(num_procs):
            p = ctx.Process(target=_worker_main,
                            args=(w, str(records_path), gpus, cfg, str(shards[w]),
                                  ret_q, cursor, progress), daemon=False)
            p.start()
            procs.append(p)
        try:
            from tqdm import tqdm
            bar = tqdm(total=len(records), desc="instances", unit="inst",
                       dynamic_ncols=True)
        except Exception:  # pragma: no cover
            bar = None
        import queue as _queue
        done, last = [], 0
        while len(done) < num_procs:
            try:
                while True:
                    done.append(ret_q.get_nowait())
            except _queue.Empty:
                pass
            n = min(int(progress.value), len(records))
            if bar is not None and n != last:
                bar.n = last = n
                bar.refresh()
            if len(done) < num_procs:
                if not any(p.is_alive() for p in procs):
                    break        # a dead worker must not hang the run
                time.sleep(0.5)
        if bar is not None:
            bar.close()
        for p in procs:
            p.join()
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join(timeout=5)

    records_path.unlink(missing_ok=True)
    rows = []
    for s in shards:
        if s.exists():
            with open(s) as fh:
                rows.extend(json.loads(line) for line in fh if line.strip())
            s.unlink()
    rows.sort(key=lambda r: (r["index"], r["c_rank"]))
    with open(dump, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n{len(rows)} candidate rows -> {dump}  ({time.perf_counter() - t0:.0f}s)")
    analyze(out_dir)


if __name__ == "__main__":
    main()
