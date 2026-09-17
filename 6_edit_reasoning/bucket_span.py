# -*- coding: utf-8 -*-
"""v12 spans: a std-aware three-bucket enumeration, rendered deterministically.

WHY THIS SHAPE. Every visible column measured so far sits at chance for the on-path
label (0.49-0.54 within-round AUC), and `predicted_gap`'s argmin scores 0.393/0.251
against floors of 0.453/0.251 -- worth nothing. Bucketing each candidate's predicted
landing by whether it survives ONE SIGMA of its own predicted spread is the first
visible feature to clear those floors:

    n_holds - n_misses at 1 sigma   AUC onpath 0.576   argmax policy 0.481 / 0.320
    the same count ignoring std     AUC onpath 0.539   argmax policy 0.429 / 0.281
    predicted_gap argmin            --                              0.393 / 0.251
    floors (fixed / gap)                                            0.453 / 0.251

The std is the whole difference, and v11b threw it away: it aggregated per-property
std into one scalar per row ("spread +/-0.794"). `delta` carries {avg, std} PER
PROPERTY, so the uncertainty is per property and the bucket is too.

Degeneracy is also far better than the gap column's: four identical rows on 9.5% of
rounds against 44.7% for the gaps, and four distinct rows on 41.7%.

THE MIDDLE BUCKET IS SPLIT BY DIRECTION. Measured over 20,000 candidate rows, the
"mean and sigma disagree" bucket is 32,698 'mean satisfies, sigma escapes' against
10,069 'mean misses, sigma reaches in' -- 76:24. One list would merge two opposite
meanings, so they are `at risk` and `within reach`.

WHAT THIS SPAN DELIBERATELY DOES NOT DO. It names no rival and makes no ranking
claim. When there is a unique bucket leader it is NOT the committed edit 62.4% of the
time, so a text that ranked and then chose otherwise would be v7's false contrast,
measured to cost -0.037/-0.039 with the foil GAINING probability in every cell. The
span enumerates and then states the choice. Nothing in it is a superlative.
"""
from __future__ import annotations

import hashlib as _hashlib
import re as _re

# One sigma. 0.5 and 2.0 were measured too: AUC 0.568 / 0.576 / 0.575, and the share of
# rounds where all four rows collapse to the same text grows with k (9.2 / 13.8 / 19.7),
# so 1.0 is both the most accurate and near the least degenerate.
K = 1.0

SLOTS = ("holds", "at risk", "within reach", "misses")


def bounds(b):
    lo = None if b[0] is None else float(b[0])
    hi = None if b[1] is None else float(b[1])
    return lo, hi


def direction(rnd):
    """The properties out of range right now, and which way."""
    out = []
    for p, b in rnd["targets"].items():
        lo, hi = bounds(b)
        v = float(rnd["props"].get(p) or 0.0)
        if lo is not None and v < lo:
            out.append((p, "below"))
        elif hi is not None and v > hi:
            out.append((p, "above"))
    return out


def classify(rnd, cand, k: float = K):
    """The four slots for one candidate, over ALL constrained properties.

    All of them, not just the violated ones: 30.6% of rounds have a candidate that
    pushes an ALREADY-SATISFIED property out of range, and a span that ranged only over
    the violations would be silent about exactly that. Such a property lands in
    `misses` with a `*`.
    """
    props, targets = rnd["props"], rnd["targets"]
    d = cand.get("delta") or {}
    holds, risk, reach, miss = [], [], [], []
    for p, box in targets.items():
        lo, hi = bounds(box)
        LO = -float("inf") if lo is None else lo
        HI = float("inf") if hi is None else hi
        dp = d.get(p) or {}
        v0 = float(props.get(p) or 0.0)
        m = v0 + float(dp.get("avg") or 0.0)
        s = abs(float(dp.get("std") or 0.0)) * k
        a, b = m - s, m + s
        was_ok = LO <= v0 <= HI
        if a >= LO and b <= HI:
            holds.append(p)
        elif b < LO or a > HI:
            # 1,688 of 18,653 entries here are regressions; they are the actionable ones
            miss.append(p + ("*" if was_ok else ""))
        elif LO <= m <= HI:
            risk.append(p)
        else:
            reach.append(p)
    return holds, risk, reach, miss


def name(rnd, i):
    """`from -> to`, plus the anchor only where that is still ambiguous.

    `to_smiles` alone names two different edits on 2.20% of rounds; `from -> to` on
    0.09%; with the anchor, none. The ANSWER has to be unambiguous even though the rows
    no longer carry it.
    """
    cs = rnd["candidates"]
    base = [f"{(c.get('from_smiles') or '').strip()} -> "
            f"{(c.get('to_smiles') or '').strip()}" for c in cs]
    if base.count(base[i]) == 1:
        return base[i]
    import json as _json
    return f"{base[i]} @{_json.dumps(cs[i].get('anchors'), sort_keys=True)}"


# The v13 closer, kept as constants because the wording is the thing most likely to be
# revised after v11b and v12 report. Neither template makes a COMPARATIVE claim: v7
# established that naming a foil as worse cost -0.037/-0.039 with the foil gaining
# probability in every cell, so mode 2 states that two options are in play and which one
# is taken, and nothing about one being better.
CLOSE_1 = "Taking Edit {n}: {commit}."
CLOSE_2 = ("Edit {first} and {second} both look plausible; "
           "taking Edit {n}: {commit}.")


# ---------------------------------------------------------------- SIFT / WEIGH
# Two observation slots either side of DROP, and they ask MIRROR questions.
#
#   SIFT   before the drop, over all four: the column whose masking most RAISES the
#          dropped edit's q -- the one holding it down. "why is this one out."
#   WEIGH  after the drop, over the survivors: the column whose masking most LOWERS the
#          commit's q -- the one holding it up. "why is this one in."
#
# The margin criterion the earlier arms used cannot express that split: `mdrop > 0` on a
# commit cell means "it holds the commit up" and on a rival cell "it holds that rival
# down", one sign for two readings. Centring q on a chosen edit separates them.
SIFT_HEAD = "What separates them, on {label}:"
WEIGH_HEAD = "What separates the rest, on {label}:"


def _live_cols(occr, meta, pool, targets, over):
    """Positions in `occr["col_idx"]` a block may name, under two filters.

    * the column's property must be one this instance is judged on. Structural columns
      carry no property and stay, as in v19, where this cut unconstrained axes from
      26.3% to 0.0% and made the curve 3.4x more stable (sd 0.0032 against 0.0110).
    * the values must differ ACROSS THE EDITS THE BLOCK WILL PRINT, which is the whole
      four for SIFT and the three survivors for WEIGH. A column that varies over four
      can be constant over those three, and printed there it shows the reader nothing.
      This is why the filter takes `over` rather than reading n_cand.

    A column with a missing value is KEPT, as in v19: only an all-present, all-equal
    column is dropped.
    """
    names, po = meta["rule_names"], (meta.get("prop_of") or {})
    allowed = set(meta["pools"][pool])
    out = []
    for k, j in enumerate(occr["col_idx"]):
        nm = names[j]
        if nm not in allowed:
            continue
        if targets is not None and po.get(nm) is not None and po[nm] not in targets:
            continue
        vals = [occr["col_v"][i][k] for i in over]
        live = [v for v in vals if v is not None]
        if len(live) == len(over) and max(live) - min(live) <= 1e-9:
            continue
        out.append(k)
    return out


def _axis_at(occr, meta, k, over):
    j = occr["col_idx"][k]
    nm = meta["rule_names"][j]
    return {"feature": nm, "col": k,
            "label": meta["labels"].get(nm, nm.replace("_", " ")),
            "over": list(over),
            "per_cand": [{"rule": i,
                          "shown": shown(nm, occr["col_v"][i][k]),
                          "spoken": spoken(nm, occr["col_v"][i][k])} for i in over]}


def choose_sift(occr, meta, dropped: int, pool: str = "primitive", targets=None):
    """argmin over columns of `q(dropped) - q_masked(dropped)`.

    `col_qc[i][k]` is stored as unmasked MINUS masked, so "masking raises it" is the most
    NEGATIVE entry -- argmin, not argmax. Ties break on the lowest column index, because
    the stored values are rounded to 6 dp and np.argsort's quicksort is not stable.
    """
    qc = occr.get("col_qc")
    if not qc:
        return None
    over = list(range(occr["n_cand"]))
    ks = _live_cols(occr, meta, pool, targets, over) or \
        _live_cols(occr, meta, pool, None, over)
    if not ks:
        return None
    return _axis_at(occr, meta, min(ks, key=lambda k: (qc[dropped][k], k)), over)


def choose_weigh(occr, meta, pick: int, dropped: int, pool: str = "primitive",
                 targets=None):
    """argmax over columns of `q(commit) - q_masked(commit)`, over the SURVIVORS.

    Note what this criterion does NOT do: it never looks at the survivors' own q, so the
    axis is the same whether or not a drop happened. It says why the commit stands where
    it does, printed beside the edits still in play.
    """
    qc = occr.get("col_qc")
    if not qc:
        return None
    over = [i for i in range(occr["n_cand"]) if i != dropped]
    ks = _live_cols(occr, meta, pool, targets, over) or \
        _live_cols(occr, meta, pool, None, over)
    if not ks:
        return None
    return _axis_at(occr, meta, max(ks, key=lambda k: (qc[pick][k], -k)), over)


def choose_gate_axis(occr, gates, meta, over, pool: str = "primitive", targets=None):
    """A's gate, argmax over the columns the block will print.

    The cheapest criterion to NAME, which is the property that decides these arms. Over
    the same pool and filter the effective vocabulary runs

        qdrop 39.7 | margin 46.7 | A-gate 15.8      (P0, WEIGH slot, 7,977 rounds)

    and naming is what sank v22: it named its axis wrong on 64.6% of held-out rounds at
    an effective 34.8 choices, and a wrongly-named axis came with onpath 0.513 against
    0.565 for a right one. A's gate is tight because it is ~90% a function of the ROUND
    rather than the candidate (0.134 spread across options against 1.296 across columns)
    -- a defect for telling candidates apart, exactly the virtue wanted here.

    It also leaks LESS: the odd-one-out is the commit 0.394 of the time against 0.450 for
    qdrop and 0.454 for margin, near the 0.333 that three survivors give by chance.

    The gate covers ~14.6 columns per round, so a pool-and-vary filter can empty it; the
    fallback is the q-drop argmax over the same filtered pool (0.4% of rounds at P0).
    """
    ks = _live_cols(occr, meta, pool, targets, over) or \
        _live_cols(occr, meta, pool, None, over)
    if not ks:
        return None
    names = meta["rule_names"]
    gk = [k for k in ks if names[occr["col_idx"][k]] in gates]
    if gk:
        return _axis_at(occr, meta,
                        max(gk, key=lambda k: (gates[names[occr["col_idx"][k]]], -k)),
                        over)
    qc = occr.get("col_qc")
    if not qc:
        return _axis_at(occr, meta, ks[0], over)
    pick = int(occr["pick"])
    return _axis_at(occr, meta, max(ks, key=lambda k: (qc[pick][k], -k)), over)


def sub_pools(meta):
    """P0 split in two, which is the split the WEIGH design asks for.

        P0 = primitive = 132 of the 167 rule columns
           = 41 property columns (r_dmean__/r_dstd__/r_dlogp/r_dmw/...)
           + 91 structural columns (r_dfg__*, site_*, h_motif_repeat, ...)

    The split is exact and it is also the prop_of split: all 41 property columns carry a
    property, none of the 91 structural ones do. So the targets filter bites on the
    property block alone and the structural block never loses a column to it.
    """
    prim, stru = set(meta["pools"]["primitive"]), set(meta["pools"]["structural"])
    return prim - stru, prim & stru


# --------------------------------------------------------------- v30: the cell block
# Every arm from v21 to v29 chose its axis with A's ATTENTION GATE, and measured against
# the labels that gate turns out to carry no axis-selection signal at all. Scored over
# the three survivors on the 4,997 labelled test rounds, with a decoder that is given the
# axis name and each survivor's value on it and fits one direction per axis name:
#
#      criterion (P0)   onpath lift   exact lift   eff.choices
#      A-gate               +0.038       +0.053         15.6
#      random               +0.050       +0.056         62.7
#      col_q                +0.124       +0.138         39.8
#      cell_m               +0.137       +0.158         44.1
#
# The gate is BELOW a coin toss over the same pool. The q-occlusion criteria carry three
# to four times as much -- and `cell_m`, the drop in the commit's q-MARGIN when that
# candidate's own cell is masked, carries the most. That is the criterion here.
#
# WHY IT NEEDS A SHORTLIST. cell_m over the raw pool costs 44.1 effective choices, and
# v22 measured what that regime does: the student named the wrong axis on 64.6% of
# held-out rounds and the arm came in 0.0163 BELOW v20. Restricting the pool to the M
# columns the criterion picks most often over the TRAIN split (never the test split --
# the shortlist is a design constant, not a fit) buys the naming cost back:
#
#      structural pool   M=6    M=10   M=16   all
#      cell_m  onpath   +0.061 +0.075 +0.101 +0.160
#      random  onpath   +0.037 +0.026 +0.035 +0.038
#      eff.choices        5.2    8.2   12.2   27.7
#
# M=16 keeps two thirds of the free-pool lift at 12.2 effective choices -- between v14
# (8.2, reproduced 0.923) and v19 (16.7, 0.749) -- and it is where the teacher-vs-random
# GAP is widest, +0.074 on exact. The structural half is used rather than P0 or the
# property half because the criterion is three times as informative there (+0.103 exact
# at M=16 against the property half's +0.055), which is itself worth saying: what
# separates the survivors is mostly not the constrained properties, which BAND has
# already spoken about two lines earlier.
#
# The shortlist is ordered by selection frequency on 100,000 train rounds and its top is
# stable between the splits (r_heavy_to 0.141/0.150, r_dsa 0.136/0.126, r_dsp3
# 0.090/0.085 train/test), so it is a property of the criterion, not of a sample.
CELL_SHORTLIST_STRU = (
    "r_heavy_to", "r_dsa", "r_dsp3", "r_dhalogen", "r_dfg__carbonyl o",
    "r_dfg__halogen", "r_dfg__benzene ring", "r_dfg__amide",
    "r_dfg__aryl methyl sites for hydroxylation", "site_num_h", "r_dpolar_frac",
    "site_in_ring", "r_dfg__aromatic nitrogen", "r_dfg__aniline",
    "r_dfg__aliphatic hydroxyl", "site_aromatic",
)


def _cell_row(occr, row, key):
    """{column index -> drop} for ONE candidate's row of the cell-occlusion table."""
    out = {}
    for r_i, c_j, d in zip(occr["cell_r"], occr["cell_c"], occr[key]):
        if r_i == row:
            out[c_j] = d
    return out


def choose_axis_cell(occr, meta, pick, over, allowed, targets=None, key="cell_m"):
    """argmax over `allowed` of the drop in q(commit) from masking the COMMIT's own cell.

    Signed and maximised, the same convention `choose_contrast` and `rule_q_lines` use:
    the largest drop is the cell holding the commit's margin up, and a cell whose masking
    RAISES the margin was working the other way. Ties break on the lowest column index --
    the drops are stored at 6 dp and 1.78% of rounds have a tie there.

    NO FALLBACK: if nothing in `allowed` varies across `over`, or the cell table does not
    reach it, this returns None and the caller drops the block rather than printing an
    axis under a criterion its heading does not promise.
    """
    names = meta["rule_names"]
    ks = [k for k in _live_cols(occr, meta, "primitive", targets, over)
          if names[occr["col_idx"][k]] in allowed]
    if not ks:
        return None
    cm = _cell_row(occr, pick, key)
    kk = [k for k in ks if occr["col_idx"][k] in cm]
    if not kk:
        return None
    return _axis_at(occr, meta, max(kk, key=lambda k: (cm[occr["col_idx"][k]], -k)), over)


def choose_axis_uniform(occr, meta, over, allowed, seed: str, targets=None):
    """The matched control for `choose_axis_cell`: uniform over the SAME live columns.

    Same pool, same shortlist, same survivor-vary filter, same block, same heading -- the
    teacher signal is the only thing taken out, so the arm difference prices the
    criterion and not the vocabulary. Measured, the two vocabularies land within 0.2
    effective choices of each other at every M.
    """
    names = meta["rule_names"]
    ks = [k for k in _live_cols(occr, meta, "primitive", targets, over)
          if names[occr["col_idx"][k]] in allowed]
    if not ks:
        return None
    h = _hashlib.blake2b(seed.encode(), digest_size=8).digest()
    return _axis_at(occr, meta, ks[int.from_bytes(h, "big") % len(ks)], over)


# ------------------------------------------------------------------ v31: the panel
# v30 measured why a PER-ROUND teacher choice cannot pay, and the numbers are worth
# keeping because they close the whole family. Its axis (cell_m over a 16-column
# shortlist) is genuinely informative -- given the right axis, a cross-validated decoder
# reads the commit off the three survivors at 0.473 against 0.375 from v20's band alone.
# But the axis is a function of the LABEL, and nothing the student sees predicts it:
#
#     router fitted on the visible round header  agrees with cell_m  0.150
#     a single constant axis                     agrees              0.142
#
# So at epoch 3 the student reproduced the axis on 24.2% of held-out rounds, and the two
# halves of the arm went in opposite directions:
#
#     named the right axis (24.2%)   exact 0.512      (v20 reads 0.414)
#     named the wrong axis (75.8%)   exact 0.343
#     blended                        exact 0.385      (measured 0.385)
#
# Break-even against v20 needs the axis named right ~42% of the time. Its own random
# control (v30r) reproduced 14.0% and scored 0.407 -- unharmed, because a random axis
# carries nothing to be wrong ABOUT. The treatment is worse than its control precisely
# because its axis means something.
#
# THE PANEL takes the naming decision out of the round. Three columns, the same three on
# every round, so there is nothing per-round to reproduce and the block's shape is a
# constant. The teacher signal now selects the PANEL rather than the axis: these are the
# three columns cell_m picks most often over P0 on the TRAIN split (0.180 / 0.104 /
# 0.096). Cross-validated over v20's band, panel +0.044 against a random panel's +0.015.
#
# No vary filter and no fallback: the heading claims a side-by-side, not a contrast, so a
# column that happens to be equal across the three is honest and the block is always
# there. All three columns are present on 100% of rounds.
PANEL_TEACHER = ("r_dmw", "r_dhba", "r_dlogp")
# TWO controls, because "random from the same pool" has two honest readings and they
# measure different things.
#
# PANEL_CONTROL_P0 is the literal one: uniform, seed 20260903, over the 102 primitive
# columns present on every round (the other 30 are `r_dmean__X` / `r_dstd__X` for a
# property this instance is not judged on, and a panel that vanished on 63% of rounds
# would differ from the teacher's in block PRESENCE rather than in content). P0 is 91
# sparse functional-group counts out of 132, so the draw lands on columns that are equal
# across the three survivors on 100%, 100% and 85% of rounds. That is what a uniform
# draw over this pool IS, and it is the comparison as asked -- but it mostly prices
# whether the block says anything at all.
#
# PANEL_CONTROL is the hard one: uniform over the 11 columns that are present on every
# round, vary across the survivors on at least half of them, and are not the teacher's
# own three. These are as usable as the teacher's panel, so v31 - v31u prices the
# CRITERION with content held roughly fixed. Cross-validated over v20's band the three
# panels read +0.024 (teacher) / +0.015 (usable) / ~+0.006 (uniform P0 mean of 4 draws).
PANEL_CONTROL = ("r_dsp3", "r_dqed", "r_dlogp_per_heavy")
PANEL_CONTROL_P0 = ("site_ring_size", "r_dfg__thiol", "r_dfg__isocyanate")

PANEL_HEAD = "Where the rest stand:"
# The column name rides in every cell rather than in the heading, so a row is readable on
# its own and the block does not depend on the student getting an order right -- the one
# thing v30 showed it cannot do.
_PANEL_SHORT = {"r_dmw": "MW", "r_dhba": "HBA", "r_dlogp": "logP", "r_dheavy": "heavy",
                "r_drotb": "rotB", "r_dqed": "QED", "r_dsa": "SA", "r_dsp3": "sp3",
                "r_dtpsa": "TPSA", "r_dhbd": "HBD", "r_dhalogen": "halogens",
                # the edit-shape and history columns, which the generic rules below
                # would render as "r heavy to" / "h motif repeat" -- a stray prefix in
                # text the student has to write back is noise it has to reproduce
                "r_heavy_to": "atoms added", "r_heavy_from": "atoms removed",
                "r_n_cuts": "cut points", "r_is_attach": "attachment",
                "r_is_delete": "deletion", "r_is_swap": "swap",
                "r_dcharge": "charge", "r_dpolar_frac": "polar frac",
                "r_drings_arom": "arom rings", "r_drings_aliph": "aliph rings",
                "h_motif_repeat": "repeats a motif",
                "h_same_rule_as_prev": "same rule as last",
                "site_n_anchors": "anchors", "site_is_hetero": "site heteroatom",
                "site_n_hetero_2b": "heteroatoms within 2"}


def panel_short(nm: str) -> str:
    if nm in _PANEL_SHORT:
        return _PANEL_SHORT[nm]
    if nm.startswith("r_dmean__"):
        return nm[len("r_dmean__"):]
    if nm.startswith("r_dstd__"):
        return nm[len("r_dstd__"):] + " sd"
    if nm.startswith("r_dfg__"):
        return nm[len("r_dfg__"):]
    if nm.startswith("site_"):
        return "site " + nm[len("site_"):].replace("_", " ")
    return nm.replace("r_d", "").replace("_", " ")


def panel_lines(occr, meta, over, cols):
    """One block: `over` in order, each row carrying every panel column as `name value`.

    Returns [] if a column is missing from this round's table, which never happened on
    the 7,977 test rounds -- all three teacher columns and all three control columns are
    present on 100% of them -- but a silently short row would be a different block.
    """
    pos = {meta["rule_names"][j]: k for k, j in enumerate(occr["col_idx"])}
    if any(c not in pos for c in cols):
        return []
    out = [PANEL_HEAD]
    for i in over:
        cells = " | ".join(f"{panel_short(c)} {shown(c, occr['col_v'][i][pos[c]])}"
                           for c in cols)
        out.append(f"  Edit {i + 1} | {cells}")
    return out if len(out) > 1 else []


# ------------------------------------------------------- v32: the invisible column
# `visible_ceiling_rules.py` fits a ranker on every column the student can derive and
# tops out at 0.5438 on-path, which v20 already exceeds -- so no VISIBLE observation has
# anything left to give, and v27/v28/v31 measured exactly that (all three flat). The
# pool marks eight columns INVISIBLE: the student cannot compute them from the prompt at
# any effort, so they are the only place headroom can exist, and they were excluded from
# that ceiling by construction.
#
# Fitting on all 159 visible columns and adding one invisible column at a time (5,000
# labelled rounds, 5-fold grouped by INSTANCE so the same scaffold cannot straddle a
# split), exact among the three survivors:
#
#     all visible                        0.4026
#     + cand_similarity                  0.4192   +0.0166
#     + ctx_log_support                  0.4120   +0.0094
#     + site_logp_contrib                0.4084   +0.0058
#     + ctx_std_r0 / ctx_fired           0.4050 / 0.4060
#     + ctx_shift / site_charge / ctx_support   at or below the baseline
#     + all eight at once                0.4222   +0.0196
#
# One column is 85% of the whole headroom. And the TEACHER SIGNAL finds it: over the
# same pool the q-drop criterion `col_q` takes cand_similarity on 38.2% of rounds,
# roughly twice the next column, while cell_m (0.219) and A's gate (0.058) both go to
# ctx_std_r0, which is worth +0.0024. That is the selection this arm is testing.
#
# WHY A CONSTANT AXIS AND NOT A PER-ROUND ONE. v30 measured the cost of a selection the
# student cannot predict: named right on 33.9% of rounds it scored 0.512, named wrong
# 0.321, and blended to 0.385 against v20's 0.414. Break-even needs the axis named right
# about half the time, and col_q over this pool has a 0.382 majority -- not enough. Held
# constant the name costs nothing to reproduce, and it is the majority pick anyway.
#
# The VALUES still have to be produced, and unlike every earlier arm they cannot be
# copied out of the tool result -- mean Tanimoto to the other candidates is in the
# prompt only in the sense that the four fragments are. That is the bet: value fidelity
# once the block's rows match was 0.907 on v31 and 0.906 on v30, and only the ORDER of
# the three survivors has to be right for the line to do its work.
CAND_TEACHER = "cand_similarity"
# Drawn uniformly, seed 20260903, from the invisible columns that vary across the
# survivors on at least 90% of rounds and are not the teacher's own pick -- so the
# control block is present as often as the treatment's and differs only in which column
# the criterion chose. Its measured headroom over the visible set is -0.0024.
CAND_CONTROL = "ctx_support"


# ---------------------------------------------------------- v33: the ordinal reading
# v33 is v32 with the three values replaced by their ORDER and nothing else changed, so
# v33 - v32 prices the rendering alone.
#
# The reason to try it: the student has to PRODUCE this column, and unlike every earlier
# arm it cannot copy it -- mean Tanimoto to the other candidates is not in the tool
# result. Measured on the arms that could copy, value fidelity was high once the rows
# matched (0.907 on v31, 0.906 on v30), so numbers were never the wall there; here they
# might be. What the line actually has to get right is the ORDER of three survivors, and
# an LM judging "which fragment is the odd one out" is a far easier ask than emitting
# 0.375.
#
# Three words, no superlative about QUALITY -- "lowest" is a statement about the printed
# column and is true by construction, unlike v7's "the weakest", which was false on four
# rounds in five and cost -0.037/-0.039 with the foil gaining probability. Ties collapse
# to two words rather than inventing an order the values do not have.
_ORD3 = ("lowest", "middle", "highest")
_ORD2 = ("lower", "higher")


def ordinal_axis(axis):
    """`axis` from choose_axis_fixed, with each `spoken` replaced by its rank word.

    Ranks over the DISTINCT values, so two survivors that tie read the same word. The
    caller has already guaranteed the column varies, so there are always at least two.
    """
    if not axis:
        return None
    vals = []
    for e in axis["per_cand"]:
        try:
            vals.append(float(e["shown"]))
        except (TypeError, ValueError):
            return None
    uniq = sorted(set(vals))
    words = _ORD3 if len(uniq) >= 3 else _ORD2
    if len(uniq) > 3:                      # never happens at three survivors, but be exact
        words = None
    out = dict(axis)
    out["per_cand"] = [dict(e, spoken=(_ORD3[min(uniq.index(v), 2)] if len(uniq) >= 3
                                       else _ORD2[uniq.index(v)]))
                       for e, v in zip(axis["per_cand"], vals)]
    return out


# ------------------------------------------------- v35 / v36: several criteria at once
# The pool here is P0 with the 28 `r_dmean__`/`r_dstd__` columns removed: 13 property
# columns computed from the fragment, 61 functional-group deltas, 11 edit-shape, 17 site
# and 2 history = 104. What comes out is exactly the columns a RDKit pass over the actual
# edit produces; what goes out is the mmpdb predicted deltas, whose numbers are printed
# verbatim in the tool result and which BAND is already speaking about two lines earlier.
#
# v35 names two of them, by the two halves of A's gate:
#     level   argmax of the gate -- where A is looking this round
#     spread  argmax of (max - min) of the gate ACROSS the live candidates -- where A
#             looks differently depending on which edit it is scoring
# The level has already been measured to carry no axis-selection signal at all (+0.038
# on-path lift against a random column's +0.050). The spread is the only part of the gate
# that is about the candidates rather than the round, it is small -- 0.134 across options
# against 1.296 across columns -- and it has never been used, because `gate_dump.py`
# collapsed the candidate axis with a max before writing. It now stores both.
#
# v36 lets every teacher criterion name one column and prints the union, at most six:
# gate level, gate spread, col_q, col_m, cell_m, cell_q. Deduplicated in that order, so
# a criterion whose argmax is already taken names its runner-up.
#
# Both are drawn as ONE panel rather than one block per column: six headings would be
# twenty-four lines, and the column name rides in each cell anyway.
def pool_104(meta):
    """P0 minus the predicted-delta columns -> the 104 the WEIGH panel draws from."""
    prim = set(meta["pools"]["primitive"])
    return {n for n in prim if not n.startswith(("r_dmean__", "r_dstd__"))}


# The columns a STUDENT could compute from what it actually has at the decision point:
# the indexed molecule, the current property values, and per candidate `from_smiles`,
# `to_smiles`, `anchors` and the mmpdb `delta` for the TARGET properties. pool_A is
# pool_104 minus two kinds of column it cannot get to:
#
#   the six fitted scores -- Crippen logP, QED, SA and TPSA over the fragment. MEASURED:
#     the 1.7B recovers these at 0.738 on a rule it saw in training and 0.222 on one it
#     did not, against a 0.017 chance floor. That is a memorised (from -> to) -> value
#     table, not a computation, and it is 65% of every column the student names.
#   the seventeen site_* columns -- they need the neighbourhood of one atom index inside
#     a SMILES string, which is a graph walk rather than a count. Their signal over
#     chance is the weakest of the three groups (+0.26 against A's +0.45) and
#     `substituents already on the site` is +0.083, i.e. nothing.
#
# What is left is 79 columns that are all differences between two SMILES: 61 functional
# -group counts and 18 fragment-level counts. The student holds these up on rules it has
# never seen (0.779 against 0.870 on seen ones), which is what "can compute" means here.
#
# `r_dmw` stays in even though molecular weight needs per-atom constants rather than a
# pure count. It is the most-selected column in the pool (0.85 of rounds once the fitted
# scores are gone) and the one column above 0.2 whose computability is UNMEASURED -- the
# student names it too rarely to score. Dropping it would have promoted `heavy atoms
# added`, whose 0.623 is the worst measured rate in A.
def pool_A(meta):
    """pool_104 minus the fitted scores and the site features -> the 79 a student can
    count off `from_smiles` and `to_smiles`."""
    fitted = {"r_dlogp", "r_dlogp_per_heavy", "r_dqed", "r_dsa",
              "r_dtpsa", "r_dtpsa_per_heavy"}
    return {n for n in pool_104(meta)
            if n not in fitted and not n.startswith(("site_", "h_"))}


CRITERIA_V35 = ("gate", "gate_spread")
CRITERIA_V36 = ("gate", "gate_spread", "col_q", "col_m", "cell_m", "cell_q")


def choose_cols_multi(occr, meta, over, allowed, criteria, lv, sp, pick,
                      targets=None):
    """One column per criterion, in order, skipping any already taken.

    Every column must vary across `over` -- the panel sits in the WEIGH slot and speaks
    about the survivors, so a column constant over them shows the reader nothing. Returns
    [] when nothing in `allowed` varies, and the caller drops the block.
    """
    names = meta["rule_names"]
    ks = [k for k in _live_cols(occr, meta, "primitive", targets, over)
          if names[occr["col_idx"][k]] in allowed]
    if not ks:
        return []
    cm = _cell_row(occr, pick, "cell_m")
    cq = _cell_row(occr, pick, "cell_q")

    def sc(k, crit):
        j = occr["col_idx"][k]
        nm = names[j]
        if crit == "gate":
            return (lv or {}).get(nm, -1e30)
        if crit == "gate_spread":
            return (sp or {}).get(nm, -1e30)
        if crit == "col_q":
            return occr["col_q"][k]
        if crit == "col_m":
            return occr["col_m"][k]
        if crit == "cell_m":
            return cm.get(j, -1e30)
        if crit == "cell_q":
            return cq.get(j, -1e30)
        raise ValueError(crit)

    out = []
    for crit in criteria:
        rest = [k for k in ks if k not in out]
        if not rest:
            break
        out.append(max(rest, key=lambda k: (sc(k, crit), -k)))
    return [names[occr["col_idx"][k]] for k in out]


def choose_cols_random(occr, meta, over, allowed, n, seed: str, targets=None):
    """The matched control: `n` columns drawn uniformly from the SAME live set.

    Same pool, same vary filter, same count per round -- so the two corpora carry the
    panel on the same rounds with the same number of cells and differ only in which
    criterion picked the columns.
    """
    names = meta["rule_names"]
    ks = [k for k in _live_cols(occr, meta, "primitive", targets, over)
          if names[occr["col_idx"][k]] in allowed]
    if not ks:
        return []
    h = int.from_bytes(_hashlib.blake2b(seed.encode(), digest_size=16).digest(), "big")
    out = []
    pool = list(ks)
    for _ in range(min(n, len(pool))):
        out.append(pool.pop(h % len(pool)))
        h //= 7
    return [names[occr["col_idx"][k]] for k in out]


def choose_axis_fixed(occr, meta, over, col):
    """One named column, the same on every round, if it varies across `over`.

    Returns None where the column is missing or constant over the survivors, and the
    caller drops the block rather than printing three identical rows under a heading
    that promises a contrast. cand_similarity varies on 93.9% of rounds, ctx_support on
    99.2%.
    """
    names = meta["rule_names"]
    for k, j in enumerate(occr["col_idx"]):
        if names[j] != col:
            continue
        vals = [occr["col_v"][i][k] for i in over]
        live = [v for v in vals if v is not None]
        if len(live) < len(over) or max(live) - min(live) <= 1e-9:
            return None
        return _axis_at(occr, meta, k, over)
    return None


def choose_axis_gated(occr, gates, meta, over, allowed, targets=None):
    """A's gate, argmax over the columns of ONE sub-pool that vary across `over`.

    NO FALLBACK, unlike choose_gate_axis: if A gates nothing that varies, this returns
    None and the caller drops the block. Falling back to the q-drop would put a second
    criterion in a block whose heading promises the first, and with a 91-column pool the
    fallback would fire often enough to be the arm rather than an edge case.

    With the truncated `evidence_v10` gates that meant the structural block appeared on
    53.8% of rounds; with gate_dump.py's full vector both blocks are ~100%. The vary
    filter itself only ever drops 0.2%.

    Measured over the three survivors at P0, this criterion is far the cheapest to name
    -- property 8.5 effective choices against qdrop 19.9 and margin 22.3, structural 10.9
    against 29.8 and 30.9 -- and naming cost is what decided every arm in this family.
    """
    names = meta["rule_names"]
    ks = [k for k in _live_cols(occr, meta, "primitive", targets, over)
          if names[occr["col_idx"][k]] in allowed]
    gk = [k for k in ks if names[occr["col_idx"][k]] in gates]
    if not gk:
        return None
    return _axis_at(occr, meta, max(gk, key=lambda k: (gates[names[occr["col_idx"][k]]],
                                                       -k)), over)


def choose_axis_random(occr, meta, over, key: str, pool: str = "primitive",
                       targets=None):
    """The ABLATION chooser: a column drawn uniformly from the SAME filtered pool.

    Seeded from the round id and a per-slot salt, so a corpus is reproducible and v24r's
    two slots do not collide. It is the observation-level twin of choose_drop_random.
    """
    ks = _live_cols(occr, meta, pool, targets, over) or \
        _live_cols(occr, meta, pool, None, over)
    if not ks:
        return None
    h = _hashlib.blake2b(key.encode(), digest_size=8).digest()
    return _axis_at(occr, meta, ks[int.from_bytes(h, "big") % len(ks)], over)


def _axis_block(axis, head: str):
    if not axis:
        return []
    out = [head.format(label=axis["label"])]
    for e in axis["per_cand"]:
        out.append(f"  Edit {int(e['rule']) + 1} | {e.get('spoken') or e['shown']}")
    return out if len(out) > 1 else []


def sift_lines(axis):
    return _axis_block(axis, SIFT_HEAD)


def weigh_lines(axis):
    return _axis_block(axis, WEIGH_HEAD)


# ---------------------------------------------------------------- DROP
# One rival is set aside, and the line says only that. No reason is given, on purpose:
# what DROP is testing is a SMALLER ACTION SPACE, not one more named fact -- CONTRAST
# already tests naming, and which observable it names is worth <=0.007. Attaching a
# reason would mix the two mechanisms with no way to separate them afterwards.
#
# No superlative appears, and none could: the rival q picks is the sole worst on band
# balance only 20.3% of the time (61.9% counting ties), so "the weakest" would be false
# on four rounds in five. v7 measured what a false contrast costs -- naming a foil as
# worse ran -0.037/-0.039 with the foil GAINING probability in every cell.
#
# The second clause names the survivors rather than restating the loser, so the line
# ends on what is still open. The commit is always among them, so the line cannot
# contradict the closer.
DROP_LINE = "Dropping Edit {n}; that leaves {rest}."


# ------------------------------------------------------------------- NEEDS: groups
# v27 adds ONE state observation to NEEDS and changes nothing else, so v27nK - v20
# isolates it and v27n3 - v27n1 isolates how many groups are named.
#
# THE SELECTION IS NOT A STATE GATE, because there is no such thing. The rule-selection
# model's global block is the 14 properties (st_val/st_has_lo/st_has_hi/st_lo_z/st_hi_z)
# plus st_n_props, st_n_violated, st_heavy and the h_* history -- no functional group
# appears in it. The only group attention in the network is `r_dfg__<name>`, which is
# what a CANDIDATE does to that group's count, and `fg_gates.py` aggregates it to the
# round by the max over live candidates -- the same aggregation the A-gate contrast uses.
#
# That gate ranks the groups the edits are MOVING, which is mostly not what the molecule
# HAS: the single highest-gated group is present on 22.5% of rounds and the whole top-8
# contains no present group at all on 33.3%. A state line has to be true of the state, so
# the dump stores every PRESENT group with its gate and this ranks that subset. Naming
# the top present group costs 8.7 effective choices -- v14's regime, where the held-out
# reproduction was 0.923.
#
# The count rides in parentheses rather than as a word. "two aromatic nitrogens" needs a
# plural of a catalog name, and the catalog already contains "halogen" and "ether oxygens
# (including phenoxy)"; the parenthetical also matches the shape of the line directly
# above it, "logD (above), rotB (below)".
FG_HEAD = "Present in the molecule: {body}."


def fg_line(rec, n: int):
    """The n highest-gated groups the molecule actually has, or None.

    Short molecules cap the list rather than pad it: 95.2% of rounds have two groups and
    81.1% have three, so v27n3 prints three on four rounds in five and fewer on the rest.
    Padding with an absent group would make the line false; repeating one would make it
    say nothing.
    """
    fg = (rec or {}).get("fg") or []
    take = fg[:max(int(n), 0)]
    if not take:
        return None
    return FG_HEAD.format(
        body=", ".join(f"{e['name']} ({e['count']})" for e in take))


def fg_line_random(rec, n: int, key: str):
    """The ABLATION: n groups drawn from the SAME pool, with the gate taken out.

    Same pool means the groups the molecule HAS, so the line stays true and its length
    distribution is unchanged -- 3 named on 86.1% of rounds, 2 on 10.3%, 1 on 3.5%. The
    only thing v27n3r removes is which of them the gate put first. Order is randomised
    too: keeping gate order and shuffling only the membership would leave the teacher
    signal in the first slot, which is the slot v27n1 showed is the whole arm.

    Seeded from the round id, like choose_drop_random and choose_axis_random, so the
    corpus is reproducible and does not collide with the other randomised slots.
    """
    fg = (rec or {}).get("fg") or []
    if not fg:
        return None
    idx = sorted(range(len(fg)), key=lambda i: _hashlib.blake2b(
        f"fg-random:{key}:{i}".encode(), digest_size=8).digest())
    take = [fg[i] for i in idx[:max(int(n), 0)]]
    if not take:
        return None
    return FG_HEAD.format(
        body=", ".join(f"{e['name']} ({e['count']})" for e in take))


def fg_line_all(rec):
    """v28: EVERY group the molecule has, with its count. Nothing is selected.

    Three selections were tried in this slot and all three were the wrong question.
    v27 ranked by A's gate on `r_dfg__`, a column whose masking moves the commit's q by
    0.00131 against 0.00476-0.01123 for every other family -- a ranking of noise. v28c
    ranked by the commit label, which put the group the ANSWER acts on in the first line.
    v28m ranked by what the candidate set moves, which is honest but still a ranking.

    Printing all of them removes the question. A cap would put it back: choosing 3 of a
    molecule's 12 groups requires an order, and any order is a criterion again.

    The order is BY COUNT, then alphabetical -- both properties of the molecule, both
    recomputable by the student with match_substructure, neither carrying anything from
    the model. The molecule has a median of 5 groups; 12 is the most seen.
    """
    fg = (rec or {}).get("fg") or []
    if not fg:
        return None
    order = sorted(fg, key=lambda e: (-int(e["count"]), e["name"]))
    return FG_HEAD.format(
        body=", ".join(f"{e['name']} ({e['count']})" for e in order))


FG_MOVED_HEAD = "Groups these edits change: {body}."


def fg_line_moved(occr, meta, n: int):
    """v28: what the CANDIDATE SET collectively moves. No teacher signal at all.

    The gate (v27) and the commit label (v28c) both failed the same test from opposite
    ends. The gate ranks a quantity the teacher's own score barely uses -- masking an
    `r_dfg__` column moves the commit's q by 0.00131 against 0.00476-0.01123 for every
    other column family -- and the commit label is not an observation at all: a line that
    names the group the answer acts on predicts the answer from the first line.

    This ranks by how many of the edits change the group, total movement breaking ties
    and CATALOG ORDER breaking those, so nothing in the ordering comes from the model.
    Every number in it is one the student can recompute from the four candidates it is
    already looking at, with match_substructure.

    IT DOES NOT SINGLE ANYONE OUT, which is the property v28c lacked: the top group is
    changed by exactly one of the four edits on 18.5% of rounds, BELOW the 25% a random
    pick would give. Naming it cannot be read as naming a candidate.

    The count printed is `n of nc` -- how many edits move it -- not the molecule's own
    count, because 43.7% of rounds name three groups the molecule does not have yet:
    these edits are overwhelmingly building groups rather than removing them (12,795
    creations against 924 destructions).

    The edits move at least three distinct groups on 66.3% of rounds, two on 16.0%, one
    on 14.0% and none on 3.7%, where the line is dropped rather than padded.
    """
    names, nc = meta["rule_names"], int(occr["n_cand"])
    rows = []
    for k, j in enumerate(occr["col_idx"]):
        nm = names[j]
        if not nm.startswith("r_dfg__"):
            continue
        vs = [occr["col_v"][i][k] or 0.0 for i in range(nc)]
        ne = sum(1 for v in vs if v != 0)
        if ne:
            rows.append((ne, sum(abs(v) for v in vs), k, nm.split("__", 1)[1]))
    if not rows:
        return None
    rows.sort(key=lambda t: (-t[0], -t[1], t[2]))
    return FG_MOVED_HEAD.format(
        body=", ".join(f"{g} ({ne} of {nc})" for ne, _, _, g in rows[:max(int(n), 0)]))


def fg_line_commit(rec, occr, meta, pick: int, n: int):
    """v28: the same present groups, ordered by whether the COMMIT touches one.

    A's gate ranks the groups the edits are MOVING, and v27 showed that naming the
    highest-gated present one changes nothing. This swaps the teacher signal for the
    teacher LABEL: of the groups the molecule has, the ones the committed edit creates
    or destroys come first, gate order breaking ties inside each bucket. `fg` arrives
    gate-sorted, so using the list index as the secondary key preserves that.

    WHY THE LABEL AND NOT A BETTER SCORE. Masking an `r_dfg__` column moves the commit's
    q by 0.00131 on average against 0.00476-0.01123 for every other column family, so
    the group identity is barely part of the teacher's decision and no re-ranking by q
    can be worth much. The label is the one signal about a group that is not weak.
    The bet is the mechanism v20 measured rather than the information: naming a thing
    early IS the policy on 72% of rounds, and this makes the named group the one the
    answer acts on.

    IT FIRES ON 35.6% OF ROUNDS -- the commit touches a present group that often, one
    such group on 23.7% and two on 9.1%. On the rest the order is unchanged and the line
    is v27's, which caps the effect at roughly a third of whatever the criterion is
    worth. Restricting to groups the molecule HAS is what costs that: the commit touches
    SOME group on 78.2% of rounds, but 12,795 of those touches create a group against
    924 that destroy one, and a created group is by definition not in the molecule yet.
    """
    fg = (rec or {}).get("fg") or []
    if not fg or occr is None:
        return None
    names = meta["rule_names"]
    dv = {}
    for k, j in enumerate(occr["col_idx"]):
        nm = names[j]
        if nm.startswith("r_dfg__"):
            dv[nm.split("__", 1)[1]] = occr["col_v"][pick][k] or 0.0
    order = sorted(range(len(fg)),
                   key=lambda i: (-abs(dv.get(fg[i]["name"], 0.0)), i))
    take = [fg[i] for i in order[:max(int(n), 0)]]
    if not take:
        return None
    return FG_HEAD.format(
        body=", ".join(f"{e['name']} ({e['count']})" for e in take))


def _edit_list(ns) -> str:
    ns = list(ns)
    if len(ns) == 1:
        return f"Edit {ns[0]}"
    return "Edits " + ", ".join(str(x) for x in ns[:-1]) + f" and {ns[-1]}"


def choose_drop(q, pick: int):
    """The rival with the lowest q, or None if there is no rival.

    q is a TEACHER SIGNAL: it gates which rival is named and never reaches the text. The
    student sees an exclusion it must justify from the rows, not a score it cannot check.

    Measured on the 5,000 held-out rounds: fires on every round, and the edit it drops is
    off-path 0.861 of the time against 0.793 for a random rival and 0.834 for the
    position control "drop the first-listed rival". It is not that control in disguise --
    the dropped index runs 23.5 / 23.3 / 24.8 / 28.4% and coincides with the first rival
    on only 30.3% of rounds, so a student that learned position would match the corpus
    three times in ten.
    """
    riv = [i for i in range(len(q)) if i != pick]
    if not riv:
        return None
    return min(riv, key=lambda i: q[i])


def choose_drop_random(n_cand: int, pick: int, key: str):
    """The ABLATION chooser: a uniformly random rival, no teacher signal at all.

    v20 asks whether naming an exclusion helps. It cannot answer WHY on its own, because
    the corpus's drop carries two things at once -- a smaller action space and q's actual
    opinion. This chooser keeps the first and removes the second.

    What it is NOT is "the same rule with less information". A random rival is off-path
    0.793 of the time against 0.861 for q's pick, so this corpus asserts a FALSE
    exclusion on 20.7% of rounds rather than 13.9%, and a student may learn to discount
    the line rather than to use it. That is part of the question.

    Seeded from the round id, not from `random`: the corpus has to be reproducible, and
    `hash()` on a str is salted per process, so it would render a different corpus on
    every run. blake2b is stable across interpreters and machines.
    """
    riv = [i for i in range(n_cand) if i != pick]
    if not riv:
        return None
    h = _hashlib.blake2b(f"drop-random:{key}".encode(), digest_size=8).digest()
    return riv[int.from_bytes(h, "big") % len(riv)]


def drop_order(q, pick: int):
    """The rivals, weakest q first -- the order v26 eliminates them in.

    One drop asks "is anything clearly out"; three ask the student to walk the field
    down to a single edit, so the COMMIT that follows is already determined by the last
    line. That is the point: v20 vs v20r said the drop's CONTENT is worth nothing
    (+0.0076, t=1.82) while the drop's ACT is worth +0.028 over v12, so the natural next
    question is whether three acts beat one.
    """
    riv = [i for i in range(len(q)) if i != pick]
    return sorted(riv, key=lambda i: (q[i], i))


def drop_order_random(n_cand: int, pick: int, key: str):
    """The same rivals in a seeded order, with q taken out of it entirely.

    Sorted on a per-rival hash rather than shuffled with `random`: a corpus has to render
    the same way on every machine, and `hash()` on a str is salted per process.
    """
    riv = [i for i in range(n_cand) if i != pick]
    return sorted(riv, key=lambda i: _hashlib.blake2b(
        f"drop-order-random:{key}:{i}".encode(), digest_size=8).digest())


def drop_lines(order, n_cand: int):
    """One line per elimination, each naming what is still standing after it.

    The last line reads `that leaves Edit 1` and so announces the answer a line before
    COMMIT does. That is not a leak to hide -- it is what eliminating three of four
    MEANS, and the span would be dishonest if it pretended otherwise.
    """
    alive = list(range(n_cand))
    out = []
    for d in order:
        alive = [i for i in alive if i != d]
        if not alive:
            break
        out.append(DROP_LINE.format(n=d + 1,
                                    rest=_edit_list(i + 1 for i in alive)))
    return out


def drop_line(drop: int, n_cand: int) -> str:
    return DROP_LINE.format(n=drop + 1,
                            rest=_edit_list(i + 1 for i in range(n_cand)
                                            if i != drop))


_LBL = _re.compile(r"^(?P<label>.*?):\s*(?P<val>[^:]*)$")


def _as_eq(text: str) -> str:
    """"label: value" -> "label = value". The evidence dump already writes each feature as a
    human sentence ending in its value; only the separator changes, so nothing is
    paraphrased and nothing is invented."""
    m = _LBL.match((text or "").strip())
    return f"{m.group('label').strip()} = {m.group('val').strip()}" if m else (text or "")


def _whole(x):
    return x is not None and abs(x - round(x)) < 1e-9


def is_integer_prop(rnd, p) -> bool:
    """A property that steps by one -- its distance to a bound is a count, not a margin.
    Local to this module rather than imported from struct_span: bucket_span is the v12+
    renderer and must not depend on the v11b one."""
    lo, hi = bounds(rnd["targets"][p])
    if not (_whole(lo) and _whole(hi)):
        return False
    vals = [float(rnd["props"].get(p) or 0.0)] + [
        float(((c.get("delta") or {}).get(p) or {}).get("avg") or 0.0)
        for c in rnd["candidates"]]
    return all(_whole(v) for v in vals)


def fmt(v, integer: bool = False) -> str:
    if integer:
        return str(int(round(v)))
    return f"{v:.3f}".rstrip("0").rstrip(".") or "0"


_Z = _re.compile(r"^(?P<p>\w+) sits [\d.]+ scaled units (?:above|below) "
                 r"its (?P<which>lower|upper) bound")


def _restate_z(text: str, rnd) -> str:
    """Rewrite a `st_hi_z` / `st_lo_z` sentence into the property's OWN units.

    A prints these as "logP sits 0.39 scaled units above its lower bound 2.29", and that
    0.39 is not derivable: it is neither |v - bound| (0.737 here) nor that over the box
    width (1.170) -- the selector scales by a per-property constant the student has never
    seen. The axis A is pointing at is worth keeping, so the distance is recomputed from
    the visible value and bound and the phrase "scaled units" is dropped. Same fact,
    checkable arithmetic.
    """
    m = _Z.match((text or "").strip())
    if m is None:
        return _as_eq(text)
    p, which = m.group("p"), m.group("which")
    if p not in rnd.get("targets", {}):
        return _as_eq(text)
    lo, hi = bounds(rnd["targets"][p])
    b = lo if which == "lower" else hi
    if b is None:
        return _as_eq(text)
    v = float(rnd["props"].get(p) or 0.0)
    it = is_integer_prop(rnd, p)
    d = v - b
    # A distance of zero has no direction, and the code had to invent one: `d >= 0` made
    # every such line read "sits 0 above", which the student has no way to predict. 14 of
    # 3,353 lines, but a span must not contain a coin flip.
    if fmt(abs(d), it) in ("0", "-0"):
        return f"{p} sits exactly at its {which} bound {fmt(b, it)}"
    return (f"{p} sits {fmt(abs(d), it)} "
            f"{'above' if d > 0 else 'below'} its {which} bound {fmt(b, it)}")


def a_lines(ev, ncand: int, only_varying: bool = False,
            n_state: int = 1, per_rule: bool = True, rnd=None):
    """A's top-gated feature: one for the state, then one per rule.

    The selection is A's argmax over its own gates -- the network's, not a visible rule.
    That is a real limitation and it is measured: a visible predictor of A's argmax tops
    out at 0.465 (constant guess 0.412), so the student cannot know at test time which
    feature it is supposed to name. What it CAN do is reproduce the value once the label
    is written. Judge this arm on whether that is enough.
    """
    out = []
    st = [e for e in (ev.get("state") or []) if e.get("text")]
    # State features are the reproducible half of A's vocabulary: the five kinds are
    # st_val, st_has_hi, st_has_lo, st_hi_z and st_lo_z, all of them functions of the
    # visible props and targets, and every one of the 48,663 in the test split refers to
    # a property the instance actually constrains. Unlike the candidate features they are
    # also not round-level constants in the damaging sense -- they do not need to
    # discriminate between candidates, only to say what the state turns on.
    for e in sorted(st, key=lambda e: -float(e.get("gate") or 0.0))[:max(n_state, 0)]:
        txt = _restate_z(e.get("text"), rnd) if rnd is not None else _as_eq(e.get("text"))
        out.append(f"What the state turns on: {txt}" if not out else f"Also: {txt}")
    if not per_rule:
        return out
    cs = sorted(ev.get("candidates") or [], key=lambda c: c.get("index", 0))
    # `only_varying` restricts each rule's pick to features whose VALUE differs across
    # the candidates. Without it the four lines are byte-identical on 45.4% of rounds and
    # name the same feature on 62.9%: A's highest-gated candidate features are largely
    # round-level constants (c_prob_all is flat on 97.7% of rounds, site_num_h 95.3%,
    # c_worst_after 75%), so four lines cost tokens and say one thing. A feature that
    # does vary exists on 99.8% of rounds, so the restriction almost never falls back.
    ok = None
    if only_varying:
        seen = {}
        for c in cs:
            for e in (c.get("features") or []):
                seen.setdefault(e.get("feature"), []).append((e.get("text") or "").strip())
        ok = {k for k, v in seen.items() if len(v) == len(cs) and len(set(v)) > 1}
    for i in range(ncand):
        c = cs[i] if i < len(cs) else None
        fs = [e for e in ((c or {}).get("features") or []) if e.get("text")]
        if ok:
            fs = [e for e in fs if e.get("feature") in ok] or fs
        if not fs:
            continue
        top = max(fs, key=lambda e: float(e.get("gate") or 0.0))
        out.append(f"  Edit {i + 1}: {_as_eq(top.get('text'))}")
    return out


def occl_lines(occ, rnd, n: int = 2, key: str = "top_margin"):
    """The v15 lines: the globals whose masking costs the commit the most MARGIN.

    `occ` is one record from occlude_globals.py. Its `text` came from the same
    `reasoning.pretty` that produced v14's wording, so the two arms differ only in WHICH
    feature is named -- which is the comparison. The z-kinds are restated in the
    property's own units for the same reason as in v14: A prints "scaled units" divided
    by a per-property constant the student has never seen.
    """
    out = []
    for e in (occ.get(key) or [])[:max(n, 0)]:
        txt = _restate_z(e.get("text"), rnd)
        out.append(f"What the state turns on: {txt}" if not out else f"Also: {txt}")
    return out


_BOOL_SUFFIX = ("_ok", "_aromatic", "_in_ring", "_fused", "_is_hetero", "_on_fg",
                "_on_guard", "_fired", "_is_unique_best", "_is_swap", "_is_delete",
                "_is_attach")


def shown(feature: str, v) -> str:
    """One option's value on a rule column, as the student would have to write it back.

    Three significant digits rather than the raw repr: a line reading 4.7387 has to be
    transcribed exactly, and a wrongly-copied state line was measured on v14 to come with
    onpath -0.046 / exact -0.037. Indicators become yes/no -- printed as 1 they read as a
    count ("edit site is in a ring: 1"). Kept byte-identical to occlude_rules._shown so a
    re-ranked axis and a stored one render the same.
    """
    if v is None:
        return "n/a"
    v = float(v)
    if feature.endswith(_BOOL_SUFFIX):
        return "yes" if v > 0.5 else "no"
    signed = feature.startswith(("r_d", "c_d"))
    if abs(v - round(v)) < 1e-9:
        return f"{v:+.0f}" if signed else f"{int(round(v))}"
    return f"{v:+.3g}" if signed else f"{v:.3g}"


# ---------------------------------------------------------------- CONTRAST values
_NUMWORD = ("none", "one", "two", "three", "four", "five", "six", "seven",
            "eight", "nine", "ten")

# Columns whose delta is a COUNT, so it reads "one more" and not "up 1". `r_dmean__*` is
# excluded on purpose even where the property counts things: it is a mean over an mmpdb
# move set, so HBA lands on +0.0224 as readily as on +1.
_COUNT_DELTA = {"r_drings_arom", "r_drings_aliph", "r_drotb", "r_dhbd", "r_dhba",
                "r_dhalogen", "r_dsp3", "r_dcharge", "r_dheavy"}
# Exact names, NOT prefixes: `r_dhba_per_heavy` starts with `r_dhba` and is a rate, not a
# count, so a startswith test would speak it as "one more".
_PLAIN_COUNT = {"r_n_cuts", "r_heavy_from", "r_heavy_to", "site_n_anchors",
                "site_ring_size", "site_degree", "site_num_h", "site_sym_equiv",
                "site_n_hetero_2b", "site_crowd_2b", "site_dist_to_fg",
                "site_dist_to_guard"}
# Indicators `_BOOL_SUFFIX` does not catch -- their names end in neither `_ok` nor an
# `_is_*` -- and printed as 1/0 they read as counts.
_BOOL_EXTRA = {"h_motif_repeat", "h_same_rule_as_prev"}


def _count_word(n: int) -> str:
    return _NUMWORD[n] if 0 <= n <= 10 else str(n)


def spoken(feature: str, v) -> str:
    """One edit's value on the CONTRAST axis, as a clause rather than a bare number.

    The heading already names the observable, so the line completes it and never repeats
    the noun: "the change in amide groups: Edit 2 | one more (+1)". Repeating it would
    cost four transcriptions of a phrase the student can already read one line up, and
    naming cost is the thing that sank v18 (rows 0.300 against v17b's 0.772).

    THE NUMBER STAYS IN THE LINE for every family that has one. Speaking the value
    without it would force the student to paraphrase a quantity it could otherwise copy,
    and a wrongly-copied value was measured on v14 to come with onpath -0.046 /
    exact -0.037.

    `shown` is untouched and stays byte-identical to `occlude_rules._shown`, so the BAND
    rows and the stored dumps render exactly as before; this is a second rendering of the
    same number for the one block a reader has to interpret rather than transcribe.
    """
    if v is None:
        return "n/a"
    v = float(v)
    if feature.endswith(_BOOL_SUFFIX) or feature in _BOOL_EXTRA:
        return "yes" if v > 0.5 else "no"
    if feature.startswith("r_dstd__"):
        # a spread has no direction; "up 0.16" would assert one. Zero reads as prose
        # rather than "0 wide", which parses as a width of nothing.
        return "no spread" if abs(v) < 1e-9 else f"{v:.3g} wide"
    if feature.startswith("r_dfg__") or feature in _COUNT_DELTA:
        n = int(round(v))
        if n == 0:
            return "unchanged (+0)"
        return f"{_count_word(abs(n))} {'more' if n > 0 else 'fewer'} ({n:+d})"
    if feature in _PLAIN_COUNT:
        return _count_word(int(round(v)))
    if feature.startswith(("r_d", "c_d")):
        return "no change" if abs(v) < 1e-9 else \
            f"{'up' if v > 0 else 'down'} {abs(v):.3g}"
    return shown(feature, v)


def choose_contrast_gate(occr, gates, meta, pool: str = "primitive",
                     require_vary: bool = True, targets=None):
    """v17b's axis: of the columns A GATES, the highest-gated one whose values DIFFER.

    Two facts force the filter rather than a plain argmax of the gate. Measured over
    6,000 rounds of the evidence dump, A's gate on a rule column is ~90% a property of
    the ROUND, not of the candidate: the same column's gate varies 0.134 across the four
    options while gates vary 1.296 across columns, and the single top-gated column is
    identical for all four options on 65.1% of rounds. So "the column whose attention
    differs most between options" ranks on a tenth of the signal and is degenerate on two
    thirds of rounds -- it is not a usable criterion. What IS usable is attention for
    salience and the value spread for discrimination: gate says which column this state
    turns on, `require_vary` keeps only the ones that can separate these options.

    `gates` is {feature: gate} for the round, from the evidence dump (the union of each
    candidate's top-gated features and `table4`). It covers 14.6 columns per round, 11.57
    of them varying, and at least one on 100.0% of rounds (99.9% within `primitive`).

    -> the same shape as `choose_contrast`, so `contrast_lines` renders either.
    """
    names = meta["rule_names"]
    allowed = set(meta["pools"][pool])
    n_cand = occr["n_cand"]
    # v19: drop a column whose property THIS INSTANCE does not constrain. `r_dmean__p` is
    # already live only for a constrained p, but the RDKit deltas (`r_dmw`, `r_dlogp`,
    # ...) exist whether or not anyone asked about that property, so A can gate one and
    # the axis ends up about a quantity the round is not judged on. Measured on v17b:
    # 26.3% of rounds named an unconstrained property, and 19.8 of those points are
    # `molecular weight added by the fragment` alone. Structural columns -- functional
    # groups, the edit site, cut count, edit kind -- have no property to constrain and
    # are KEPT: they carry distinctions the property numbers cannot show (a round where
    # every candidate `holds HBD, HBA` but only two introduce a carbamate).
    if targets is not None:
        po = meta.get("prop_of") or {}
        allowed = {n for n in allowed if po.get(n) is None or po[n] in targets}

    def scan(need_gate: bool, rank):
        best = None
        for k, j in enumerate(occr["col_idx"]):
            nm = names[j]
            if nm not in allowed or (need_gate and nm not in gates):
                continue
            vals = [occr["col_v"][i][k] for i in range(n_cand)]
            live = [v for v in vals if v is not None]
            if require_vary and len(live) == n_cand and max(live) - min(live) <= 1e-9:
                continue
            sc = rank(k, nm)
            if best is None or sc > best[0]:
                best = (sc, nm, vals)
        return best

    # THE BLOCK IS FIXED FURNITURE, so it may not vanish. The gate table covers only
    # ~14.6 columns a round (each candidate's top-gated, plus `table4`) and `primitive`
    # discards the c_* aggregates that dominate those gates, so on 0.07% of rounds the
    # intersection was empty and the span lost the heading AND all four lines -- it
    # silently became a v12 span. Those rounds now fall back to the SAME pool under the
    # SAME varying requirement, ranked by the margin drop instead of the gate: `col_m`
    # exists for every live column and a varying primitive column exists on 100.0% of
    # rounds, so the fallback cannot fail. The fallback is NOT the gate criterion, so an
    # arm-level claim about attention is a claim about the other 99.93%.
    best = scan(True, lambda k, nm: gates[nm])
    if best is None:
        best = scan(False, lambda k, nm: occr["col_m"][k])
    if best is None:
        return None
    g, nm, vals = best
    return {"feature": nm, "label": meta["labels"].get(nm, nm.replace("_", " ")),
            "drop": g,
            "per_cand": [{"rule": i, "shown": shown(nm, vals[i]),
                          "spoken": spoken(nm, vals[i])}
                         for i in range(n_cand)]}


COMMIT_LINE = "The edit taken rests on: {text}"


def _phrase(label: str, feature: str, v) -> str:
    """`label` + value as one readable clause.

    An indicator's label is already a whole predicate -- "edit site is in a ring" -- so
    appending the value gives "edit site is in a ring yes". Booleans are negated in place
    instead, and fall back to a yes/no suffix only where the label has no copula to negate.
    """
    if feature.endswith(_BOOL_SUFFIX):
        true = float(v) > 0.5
        if true:
            return label
        return (label.replace(" is ", " is not ", 1) if " is " in label
                else f"not: {label}")
    return f"{label} {shown(feature, v)}"


def rule_q_lines(occr, meta, pool: str = "primitive", key: str = "cell_q"):
    """v18's block: for EACH option, that option's own feature whose masking drops
    q(commit) the most. Four lines, in rule order.

    The criterion is the drop in the COMMITTED candidate's q, so the reading differs by
    row and it is worth being exact about it:

        the commit's own row   masking it lowers q(commit) -> the quantity its score
                               rests on
        a rival's row          masking it lowers q(commit) too, by way of the candidate
                               self-attention -> that option's value is part of the
                               context the commit scores well against

    A rival's effect is real but a fifth the size: max |drop| 0.0167 against the commit's
    own 0.0811 over 4,000 rounds, and only 6.5% of rounds have a rival effect under a
    tenth of the commit's. Signed, not absolute -- the largest DROP is what holds
    q(commit) up, and a feature whose masking RAISES it was working the other way.

    Rule order, not effect order: sorting by effect would put the commit's row first
    wherever its own cells dominate, and "take the one named first" is a form shortcut
    the earlier arms already showed the student will learn (first mention == the call on
    72% of rounds).
    """
    names = meta["rule_names"]
    allowed = set(meta["pools"][pool])
    pos = {j: k for k, j in enumerate(occr["col_idx"])}
    best = {}
    for r_i, c_j, d in zip(occr["cell_r"], occr["cell_c"], occr[key]):
        nm = names[c_j]
        if nm not in allowed:
            continue
        cur = best.get(r_i)
        if cur is None or d > cur[0]:
            best[r_i] = (d, nm, c_j)
    out = ["What each edit turns on:"]
    for i in range(occr["n_cand"]):
        e = best.get(i)
        if e is None:
            continue
        _, nm, c_j = e
        k = pos.get(c_j)
        v = occr["col_v"][i][k] if k is not None else None
        if v is None:
            continue
        out.append(f"  Edit {i + 1} | "
                   f"{_phrase(meta['labels'].get(nm, nm.replace('_', ' ')), nm, v)}")
    return out if len(out) > 1 else []


def choose_contrast(occr, meta, criterion: str = "margin", pool: str = "primitive",
                require_vary: bool = True):
    """Re-rank the RAW attribution into one axis. No GPU, no stored top-K.

    `occlude_rules.py` writes `col_idx` / `col_m` / `col_q` / `col_v` per round precisely
    so the criterion can be chosen after the hour of GPU is already spent. `meta` is that
    run's `rule_names.json`, which carries the index -> name table, the named pools and
    the label per column.

        criterion   "margin" = q(commit) - max q(rival)   (v15's B)
                    "qdrop"  = q(commit) alone            (v16's A)
        pool        "primitive" evidence columns, "all", or "dprop" (the predicted delta
                    of a constrained property only)
        require_vary  skip columns whose live values are equal across the options -- a
                    constant column still carries a large drop, because masking it moves
                    every q together, and without this filter the top column has four
                    identical values on 20.3% of rounds

    Ties are broken by the LOWEST column index, deliberately. The dump's own `top_col*`
    lists come from `np.argsort`, whose default quicksort is not stable, and 1.78% of
    rounds have two columns tied at the 6-dp resolution the drops are stored at -- so the
    stored order is not reproducible and this function, not that list, is what a corpus
    is rendered from. On the rounds without a tie the two agree exactly, feature and every
    printed value (7,835 of 7,977 test rounds; the remaining 142 are exactly those ties).

    -> {"feature", "label", "drop", "per_cand": [{"rule", "shown"}]} or None.
    """
    names = meta["rule_names"]
    allowed = set(meta["pools"][pool])
    key = "col_m" if criterion == "margin" else "col_q"
    n_cand = occr["n_cand"]
    best = None
    for k, j in enumerate(occr["col_idx"]):
        nm = names[j]
        if nm not in allowed:
            continue
        vals = [occr["col_v"][i][k] for i in range(n_cand)]
        live = [v for v in vals if v is not None]
        if require_vary and len(live) == n_cand and max(live) - min(live) <= 1e-9:
            continue
        d = occr[key][k]
        if best is None or d > best[0]:
            best = (d, nm, vals)
    if best is None:
        return None
    d, nm, vals = best
    return {"feature": nm, "label": meta["labels"].get(nm, nm.replace("_", " ")),
            "drop": d,
            "per_cand": [{"rule": i, "shown": shown(nm, vals[i]),
                          "spoken": spoken(nm, vals[i])}
                         for i in range(n_cand)]}


def contrast_lines(contrast, n_cand: int):
    """`choose_contrast` output -> the CONTRAST block. Same shape `column_lines` prints."""
    if not contrast:
        return []
    per = {int(e["rule"]): e for e in contrast["per_cand"]}
    out = [f"What separates them, on {contrast['label']}:"]
    for i in range(n_cand):
        if i in per:
            # `spoken` where the chooser produced one; `shown` is the fallback for a
            # contrast read back out of a stored dump, which carries only the number.
            out.append(f"  Edit {i + 1} | {per[i].get('spoken') or per[i]['shown']}")
    return out if len(out) > 1 else []


def column_lines(occr, n_cand: int, key: str = "top_col_varying"):
    """v17's block, ONE axis: the rule column whose masking costs the commit the most
    margin, then every option's value on that column, in rule order.

    This is the version that reads as a comparison. The cell-level alternative -- each
    option's own strongest feature -- puts the four lines on the same column 0.0% of the
    time (mean 3.18 distinct columns over 1,500 rounds), so it lists four unrelated
    quantities under a heading that promises a contrast.

    `top_col_varying` is the pool filtered to columns whose values actually DIFFER across
    the options. Without that filter the top column has four identical values on 20.3% of
    rounds: masking a constant column still moves the margin through the interaction
    terms, so mdrop alone does not imply a contrast. 48.5 columns vary per round and none
    has zero, so the filter never empties the pool.

    The heading names the axis and carries no value and no superlative; the lines carry no
    ranking claim, for the reason v7 established -- naming a foil as worse cost
    -0.037/-0.039 with the foil GAINING probability in every cell.
    """
    for c in (occr.get(key) or []):
        per = {int(e["rule"]): e for e in (c.get("per_cand") or [])}
        if not per or not c.get("label"):
            continue
        out = [f"What separates them, on {c['label']}:"]
        for i in range(n_cand):
            e = per.get(i)
            if e is not None:
                out.append(f"  Edit {i + 1} | {e['shown']}")
        return out if len(out) > 1 else []
    return []


def rule_lines(occr, n_cand: int, key: str = "per_rule_primitive"):
    """v17's block: one line per option, IN RULE ORDER, naming that option's own
    strongest rule feature by the margin criterion.

    `occr` is one record from occlude_rules.py. Rule order -- not effect order -- is
    deliberate and is the same reason v13's pair mention was ordered by number: sorting
    by effect would put the pivotal option first and teach "take the one named first",
    which is a form shortcut rather than a reason. The lines carry no superlative and no
    ranking claim; each states one measured quantity about one option, exactly as the
    bucket rows above them do.

    `per_rule_primitive` restricts each option's pick to a PRIMITIVE column. The
    unrestricted ranking is dominated by the `c_*` aggregates -- c_pred_gap 20.5%,
    c_prob 17.6%, c_n_props_helped 10.9%, c_n_sat_now 4.8% on a 1,500-round sample --
    which are arithmetic on the same delta and box the rows already show, so naming one
    is reciting a score. That is the `allfeat` arm's measured failure (12 of 18 spans
    read a score column and committed in one sentence). Restricting costs 29% of the
    attribution magnitude (median primitive mdrop / unrestricted mdrop = 0.71) and
    covers all four options on 100% of rounds.
    """
    ent = {int(e["rule"]): e for e in (occr.get(key) or [])}
    if not ent:
        return []
    out = ["What separates them:"]
    for i in range(n_cand):
        e = ent.get(i)
        if e is not None:
            out.append(f"  Edit {i + 1} | {e['text']}")
    return out if len(out) > 1 else []


def render(rnd, pick: int, q=None, ev=None, a_vary: bool = False,
           n_state: int = 1, per_rule: bool = True, occ=None,
           occ_key: str = "top_margin") -> str:
    """The span. Rows carry NO fragment name -- `Edit N` refers to the Nth candidate in
    the `suggest_edits` result the student is looking at.

    THE CLOSER NAMES BOTH, `Taking Edit 2: <from> -> <to>.`, reversing the earlier rule
    that it name the fragment alone. That rule existed because on-path membership by
    index runs 0.330/0.379/0.431/0.453, so a bare index as the answer would teach
    "always Edit 4". The index is back because DROP names edits by number and a closer
    that switched to fragments mid-argument would not read as the same argument -- and
    the shortcut it re-opens is capped: prepare.py quotas the COMMITTED index, so a
    constant index scores 0.453 on-path / 0.250 exact, both below noreason's 0.499. It
    is not a winning policy, only a cheap one, and the fragment still has to follow.
    """
    un = direction(rnd)
    L = []
    L.append("Out of range now: "
             + (", ".join(f"{p} ({d})" for p, d in un) if un else "nothing") + ".")
    L.append("Each edit, at one sigma of its predicted shift:")
    for i, c in enumerate(rnd["candidates"]):
        h, r, w, m = classify(rnd, c)
        parts = []
        if h: parts.append("holds " + ", ".join(h))
        if r: parts.append("at risk " + ", ".join(r))
        if w: parts.append("within reach " + ", ".join(w))
        if m: parts.append("misses " + ", ".join(m))
        L.append(f"  Edit {i + 1} | " + " | ".join(parts))
    if occ is not None:
        L.extend(occl_lines(occ, rnd, n_state, occ_key))
    elif ev is not None:
        L.extend(a_lines(ev, len(rnd["candidates"]), a_vary, n_state, per_rule, rnd))
    nm = name(rnd, pick)
    qt = None
    if q:
        qt = max(range(len(q)), key=lambda j: q[j])
    if qt is None or qt == pick:
        # q agrees with the commit (43% of rounds), or there is no q: v12's closer.
        L.append(CLOSE_1.format(n=pick + 1, commit=nm))
    else:
        # q disagrees. BOTH are mentioned by number only, ORDERED BY RULE NUMBER rather
        # than by role, and the fragment appears once, in the clause that states the
        # choice. Two properties follow from that and both matter:
        #   - the pair mention leaks nothing. A fixed "commit first" would teach "take
        #     whichever is named first"; and naming the commit by fragment INSIDE the
        #     pair would make the answer identifiable by form. Numbers in rule order
        #     put the commit first on 46.6% of mode-2 rounds -- a coin flip.
        #   - the ANSWER is still a fragment, never a bare index, because on-path
        #     membership by index runs 0.330/0.379/0.431/0.453 and "always Rule 4" would
        #     otherwise already score 0.453.
        lo, hi = sorted((pick, qt))
        L.append(CLOSE_2.format(first=lo + 1, second=hi + 1,
                                n=pick + 1, commit=nm))
    return "\n".join(L)


# THE n40 FAMILY SAYS IT IN A DIFFERENT VOICE, so the anchor is an ALTERNATION
# rather than a rewrite: every arm from n8 to n35 is still read by the first
# branch, byte for byte. Replacing the literal instead would have switched off
# drop_present, drop_match and drop_avoid_commit on the whole back catalogue
# without a word in the log -- the trap nat_span's n25 header describes.
_DROP = _re.compile(
    r"^(?:Dropping Edit (?P<a>\d+); that leaves "
    r"|.*?\bI set Edit (?P<b>\d+) aside(?: first| next)?[^;]*; that leaves )", _re.M)


def dropped_edit(text: str):
    """The 0-based index a span drops, or None if it drops nothing.

    Used on BOTH sides at eval time: on the reference to decide whether this arm has a
    DROP block at all, and on the student's own text to score what IT set aside. The two
    questions are different -- "did it drop the same edit" is fidelity, "is the edit it
    dropped actually droppable" is the thing the block exists for -- and neither is
    readable without the other.
    """
    m = _DROP.search(text or "")
    return int(m.group("a") or m.group("b")) - 1 if m else None


# ---------------------------------------------------------------- fidelity marking
_HEAD = _re.compile(r"^Out of range now:\s*(?P<body>.*?)\.\s*$", _re.M)
_ROW = _re.compile(r"^\s{2}Edit\s+(?P<n>\d+)\s*\|\s*(?P<body>.*?)\s*$", _re.M)
_PROP = _re.compile(r"[A-Za-z_][A-Za-z_0-9]*\*?")


def _slots_from(body: str) -> dict:
    """Split one row body back into its four slots. Absent slot -> empty set.

    TWO ROW SHAPES, and this has to read both or n63's comp_* metrics silently report
    zero for every slot -- the failure mode that looks like "the student learned
    nothing" when the truth is "the parser learned nothing".

        n49..n62   one chunk per SLOT, word first:      `holds MW, logP`
        n63        one chunk per PROPERTY, word last:   `MW 310.091 holds`

    They cannot collide: no property is named after a slot word, and no slot word is
    followed by a number. The property is the FIRST `_PROP` match in the n63 chunk --
    `to` in `85.421 to 86.773` matches too, which is why it is [0] and not the set.
    """
    out = {s: set() for s in SLOTS}
    chunks = [c.strip() for c in body.split("|")]
    tail = False
    for chunk in chunks:
        for s in ("within reach", "at risk", "holds", "misses"):   # longest first
            if chunk == s or chunk.endswith(" " + s):
                names = _PROP.findall(chunk[:len(chunk) - len(s)])
                if names:
                    out[s].add(names[0])
                tail = True
                break
    if tail:
        return out
    for chunk in chunks:
        for s in ("within reach", "at risk", "holds", "misses"):   # longest first
            if chunk.startswith(s):
                out[s] = set(_PROP.findall(chunk[len(s):]))
                break
    return out


def check_components(rnd, text: str) -> dict:
    """Per-component fidelity: is each piece of the enumeration actually right?

    Reported separately per slot because they fail for different reasons -- `holds`
    needs the box arithmetic, `at risk` and `within reach` need the SIGN of the sigma
    excursion, and `misses` needs the regression check. One aggregate number would hide
    which of those the student never learned.
    """
    # The header is checked as an EXACT body match, not as a set of names. Two earlier
    # attempts were not checks at all: a bare `findall` over the body also collects the
    # direction words `above`/`below`, and matching only name SETS scores an injected
    # "FAKE," as correct because it carries no direction to collect. The body is a pure
    # function of the round, so equality is the honest test and it catches insertion,
    # omission, reordering and a flipped direction alike.
    un = direction(rnd)
    truth_head = ", ".join(f"{p} ({d})" for p, d in un) if un else "nothing"
    got_head = _HEAD.search(text)
    head_ok = int(bool(got_head) and
                  " ".join(got_head.group("body").split()) == truth_head)

    truth = []
    for c in rnd["candidates"]:
        h, r, w, m = classify(rnd, c)
        truth.append({"holds": set(h), "at risk": set(r),
                      "within reach": set(w), "misses": set(m)})

    seen = {}
    for mo in _ROW.finditer(text):
        n = int(mo.group("n"))
        if 1 <= n <= len(truth) and n not in seen:
            seen[n] = _slots_from(mo.group("body"))

    out = {"head_ok": head_ok, "rounds": 1,
           "rows_named": len(seen), "rows_expected": len(truth), "rows_parsed": 0,
           "all_ok": 0}
    for s in SLOTS:
        out[f"{s.replace(' ', '_')}_ok"] = 0
    for n, got in seen.items():
        t = truth[n - 1]
        out["rows_parsed"] += 1
        every = True
        for s in SLOTS:
            ok = int(got[s] == t[s])
            out[f"{s.replace(' ', '_')}_ok"] += ok
            every &= bool(ok)
        out["all_ok"] += int(every)
    return out


_N19_CRIT = _re.compile(r"^We should (?:select an edit|narrow to the edits|"
                        r"at least rule out the edits) which .*\.$", _re.M)
# n20's per-edit block. `--` and not `|`, because `_ROW` above scans the WHOLE span for
# `  Edit N | ...` and a second block in that shape would be read as more band rows.
_N20_ROW = _re.compile(r"^\s{2}Edit\s+(?P<n>\d+)\s+--\s+(?P<ph>.*?)\s*$", _re.M)
_N20_NONE = "no feature of its own here"
# n21's clause, appended to the decision line it closes. Anchored so that the reference
# line and the student's are compared on the SAME split -- the part up to the full stop
# that `_DROP` and `parse_edit` already read, and the clause after it.
_N21_CL = _re.compile(
    r"^(?P<head>(?:Dropping|Taking) Edit \d+[;:] .*?\."
    r"|.*?\bI (?:set Edit \d+ aside(?: first| next)?[^;]*;|take Edit \d+:) .*?\.)"
    r"(?: It (?P<cl>.*?)\.)?$")
# n25 puts the same clause on the line ABOVE the decision instead of behind it, so the
# reason has to be picked up from either side. Matched as a WHOLE LINE and compared as
# one: the edit index is inside the sentence, and a lead naming Edit 3 over a line that
# drops Edit 2 is the incoherence this arm can produce, so it has to score wrong rather
# than be normalised away.
#
# A SHAPE TEST, not a literal. n25 draws its reason from a bank of 29 sentences and the
# bank is edited, so anything anchored on one wording rots the first time a frame is
# reworded -- silently, by reporting the arm has no clauses at all. What actually defines
# the line is its position and what it is NOT: flush left (the band rows and the block
# rows are indented two), not a decision line, not one of the four headers, naming an
# edit, ending in a full stop.
# The three n40 exclusions are the prose the family writes flush left: its ANALYSIS
# paragraph opens `Edit 1 holds ...`, its SUMMARY opens `What separates them:`, and its
# decisions open `Of Edit(s) ...`. All three name an edit and end in a full stop, so
# without these the reason for a bare decision would be read off the paragraph above it.
_N25_LEAD = _re.compile(
    r"^(?!\s)(?!Dropping Edit \d+;)(?!Taking Edit \d+:)(?!Out of range now:)"
    r"(?!Present in the molecule:)(?!Each edit,)(?!What sets each edit apart:)"
    r"(?!Edit \d+ )(?!What separates them:)"
    r"(?!Of Edits? \d)(?!Among Edits? \d)(?!Between Edits? \d)"
    r".*\bEdit \d+\b.*\.$")


_N44_CL = (
    _re.compile(r"^(?P<head>Of Edits? .*?, I set Edit \d+ aside(?: first)?"
                r"(?: -- (?P<cl>.*?))?; that leaves .*\.)$"),
    _re.compile(r"^(?P<head>Among Edits? .*?, (?:Edit \d+ (?P<cl>.*?), so )?"
                r"I set Edit \d+ aside next; that leaves .*\.)$"),
    _re.compile(r"^(?P<head>(?:Between|Of) Edits? .*?, (?:Edit \d+ (?P<cl>.*?), so )?"
                r"I take Edit \d+: .*\.)$"),
)


def _n21_pairs(span: str):
    """{the decision line: its reason, or None} -- from either placement.

    One pass over the lines, because the lead form is defined by ADJACENCY: the reason
    for a decision is the line directly above it, and nothing else on the span can be
    read as one.
    """
    out, ls = {}, (span or "").split("\n")
    for j, ln in enumerate(ls):
        m = _N21_CL.match(ln)
        head = None if m is None else m.group("head")
        cl = None if m is None else m.group("cl")
        if cl is None:
            # `_N21_CL` matches n44's lines too, and reports no clause on every one of
            # them, because the reason is inside the sentence rather than behind it.
            for rx in _N44_CL:
                m4 = rx.match(ln)
                if m4 is not None:
                    head, cl = m4.group("head"), m4.group("cl")
                    break
        if head is None:
            continue
        if cl is None and j and _N25_LEAD.match(ls[j - 1]):
            cl = ls[j - 1]
        out[head] = cl
    return out
# The four families a row's facts can come from, told apart by how the phrase opens.
# Order matters: `holds the most` is a COUNT over the whole band and `holds logP` is one
# slot of it, and a prefix test on "holds " alone would file the first as the second.
_N20_TALLY = ("holds the most", "holds the least",
              "gives up the most", "gives up the least")


def _n20_family(ph: str) -> str:
    if ph in _N20_TALLY:
        return "cnt"                                    # the band, counted
    if ph.startswith(("puts on ", "takes off ")):
        return "fg"
    if ph.startswith(("adds no ", "adds the most ", "adds the fewest ")):
        return "atom"
    return "band"                                       # holds/exposed/gives up/reach
_N19_FGC = ("adds ", "drops ", "no change to any named group")


def _n19_cells(body: str):
    """(the functional-group cell, the atom-count cell) out of one n19 row.

    Keyed on how each cell OPENS rather than on position: a row carries four slots at
    most and any of them can be absent, so the fg cell is not at a fixed index.
    """
    fg = cnt = None
    for chunk in body.split("|"):
        ch = chunk.strip()
        # `-` as well as `+`: n25 prints the sign, so a swap that takes two carbons off
        # opens the cell with `-2 carbon`. Keyed on `+` alone this scored 8,312 of 8,724
        # cells and said nothing about the other 412 -- not wrong, just blind, which is
        # worse in a metric.
        if ch.startswith(("+", "-")) or ch == "adds no atoms":
            cnt = ch
        elif any(ch.startswith(x) for x in _N19_FGC):
            fg = ch
    return fg, cnt


def check_n21(ref_span: str, text: str) -> dict:
    """n21's clauses, marked line by line against the reference.

    Keyed on the decision line's own head, not on position: a student that drops a
    different edit writes a different head, and scoring by position would then compare
    its clause for Edit 2 against the reference's for Edit 3 and call it wrong for the
    second time. A clause is scored only where the two agree on what the line says, and
    `cl_head` carries how often that was.
    """
    out = {"cl_lines": 0, "cl_head": 0, "cl_ref": 0, "cl_ok": 0, "cl_spurious": 0,
           "cl_drop_ref": 0, "cl_drop_ok": 0, "cl_take_ref": 0, "cl_take_ok": 0}
    got = _n21_pairs(text)
    for head, cl in _n21_pairs(ref_span).items():
        drop = head.startswith("Dropping")
        out["cl_lines"] += 1
        if head not in got:
            continue                    # the student did not write this line at all
        out["cl_head"] += 1
        g = got[head]
        if cl is None:
            # 4.9% of drop joints carry no clause -- nothing was sayable on the drop
            # side. Writing one there is an invention, and it is the only way this arm
            # can put a contradiction on screen, so it is counted on its own.
            out["cl_spurious"] += int(g is not None)
            continue
        out["cl_ref"] += 1
        ok = int(g == cl)
        out["cl_ok"] += ok
        out["cl_drop_ref" if drop else "cl_take_ref"] += 1
        out["cl_drop_ok" if drop else "cl_take_ok"] += ok
    return out


def check_n19(ref_span: str, text: str) -> dict:
    """n19's three additions, each marked against the reference span.

    Marked against the REFERENCE, not recomputed: the fg cell needs the 61-pattern
    catalog and the count cell needs RDKit, and neither belongs inside a per-step eval
    loop. The reference row is the same string the corpus was built from, so equality
    with it is the same test.

    The criterion is the one line here the student cannot get from the table in front of
    it -- it names the edit the teacher committed -- so `crit_exact` is the arm's whole
    question in one number.
    """
    out = {"rounds": 1, "crit_ref": 0, "crit_present": 0, "crit_exact": 0,
           "fg_cells": 0, "fg_ok": 0, "cnt_cells": 0, "cnt_ok": 0, "row_all": 0,
           "feat_ref": 0, "feat_ok": 0, "feat_all": 0,
           "feat_facts": 0, "feat_fact_ok": 0, "feat_extra": 0,
           "feat_none_ref": 0, "feat_none_got": 0, "feat_none_ok": 0,
           "feat_band": 0, "feat_band_ok": 0, "feat_cnt": 0, "feat_cnt_ok": 0,
           "feat_fg": 0, "feat_fg_ok": 0, "feat_atom": 0, "feat_atom_ok": 0}
    # n20 replaces the criterion with one phrase per edit. Marked row by row and not as
    # a block: the arm's question is whether a fact about an edit is derivable, and a
    # single all-or-nothing number over four rows answers it four times less precisely.
    rf = {int(m.group("n")): m.group("ph") for m in _N20_ROW.finditer(ref_span or "")}
    if rf:
        gf = {int(m.group("n")): m.group("ph")
              for m in _N20_ROW.finditer(text or "")}
        out["feat_ref"] = len(rf)
        out["feat_ok"] = sum(int(gf.get(k) == v) for k, v in rf.items())
        out["feat_all"] = int(out["feat_ok"] == len(rf))
        # PER FACT as well as per row. A row carrying more of its edit's facts than the
        # cap allows is cut down by a draw the student cannot reproduce, so whole-row
        # equality is capped below 1 by construction and would read as a failure the
        # arm never had. This counts the facts themselves, in either order.
        for k, v in rf.items():
            deny = v == _N20_NONE
            want = [] if deny else [x.strip() for x in v.split(";")]
            g = gf.get(k)
            got = ([] if g is None or g == _N20_NONE
                   else [x.strip() for x in g.split(";")])
            out["feat_facts"] += len(want)
            out["feat_fact_ok"] += sum(1 for x in want if x in got)
            # Facts the student wrote that the reference row does not carry. Without it
            # `feat_fact` is pure recall and a row that lists everything scores 1.
            out["feat_extra"] += sum(1 for x in got if x not in want)
            # The denial is 29.5% of rows -- the single largest class -- so a student
            # that learns to write it everywhere would carry `feat_row` to 0.295 while
            # having learned nothing. Scored in both directions for that reason.
            out["feat_none_ref"] += int(deny)
            out["feat_none_got"] += int(g == _N20_NONE)
            out["feat_none_ok"] += int(deny and g == _N20_NONE)
            for x in want:
                f = _n20_family(x)
                out[f"feat_{f}"] += 1
                out[f"feat_{f}_ok"] += int(x in got)
    rc = _N19_CRIT.search(ref_span or "")
    gc = _N19_CRIT.search(text or "")
    if rc:
        out["crit_ref"] = 1
        out["crit_present"] = int(bool(gc))
        out["crit_exact"] = int(bool(gc) and gc.group(0).strip() == rc.group(0).strip())
    ref = {int(m.group("n")): _n19_cells(m.group("body"))
           for m in _ROW.finditer(ref_span or "")}
    got = {int(m.group("n")): _n19_cells(m.group("body"))
           for m in _ROW.finditer(text or "")}
    for n, (rf, rc2) in ref.items():
        gf, gc2 = got.get(n, (None, None))
        every = True
        if rf is not None:
            out["fg_cells"] += 1
            ok = int(gf == rf); out["fg_ok"] += ok; every &= bool(ok)
        if rc2 is not None:
            out["cnt_cells"] += 1
            ok = int(gc2 == rc2); out["cnt_ok"] += ok; every &= bool(ok)
        out["row_all"] += int(every and (rf is not None or rc2 is not None))
    return out


# The two rule-block headings. Both are followed by one `  Rule N | ...` line per option,
# and the v12 bucket rows above them share that row form -- so a checker that grepped for
# `Rule N |` over the whole span would mark the BUCKET rows as the block and always score
# a v17b/v18 span as perfect. The rows are taken only from the run that follows the
# heading, and the heading identifies which arm's block it is.
_BLK_HEADS = ("What separates them, on ", "What separates the rest, on ",
              "What each edit turns on:")
_BLK_ROW = _re.compile(r"^\s{2}Edit\s+(\d+)\s*\|\s*(.*?)\s*$")


def _blocks_of(text: str):
    """[(heading, {edit number: line body})] for EVERY observation block, in order.

    Plural since v24, which carries SIFT and WEIGH at once. Returning only the first
    would have scored SIFT and silently ignored WEIGH -- and a metric that reports on
    half a span reads as if the other half were fine.
    """
    lines = (text or "").split("\n")
    out = []
    for i, ln in enumerate(lines):
        if not ln.startswith(_BLK_HEADS):
            continue
        rows = {}
        for nxt in lines[i + 1:]:
            m = _BLK_ROW.match(nxt)
            if m is None:
                break                       # the run ends at the first non-row line
            rows[int(m.group(1))] = m.group(2)
        out.append((ln.strip(), rows))
    return out


def check_rule_block(ref_text: str, gen_text: str) -> dict:
    """Did the student reproduce the rule block the training data would have shown?

    Marked against the REFERENCE span rather than recomputed, because the block's content
    comes from the occlusion dump -- which column A gated, or which cell moved q(commit)
    -- and none of that is a function of the round the way the bucket rows are. The
    reference is the same string the corpus carried for this (group_id, depth), so this
    measures exactly "did it name the feature it was trained to name, with the right
    values".

    `blk_rounds` counts rounds whose REFERENCE has a block, so an arm without one (v12,
    v15, noreason) contributes nothing and the metric is simply not reported for it.
    Everything else is counted over those rounds whether or not the student emitted
    anything -- a span that drops the block scores 0, not "not applicable".
    """
    ref = _blocks_of(ref_text)
    if not ref:
        return {}
    gen = _blocks_of(gen_text)
    out = {"blk_rounds": 1,
           "blk_present": int(bool(gen)),
           # the headings, as a SEQUENCE: an arm with two blocks that emits them in the
           # wrong order has not reproduced the span, and comparing sets would say it had
           "blk_head_ok": int([h for h, _ in gen] == [h for h, _ in ref]),
           "blk_rows_expected": sum(len(r) for _, r in ref),
           "blk_rows_ok": 0,
           "blk_all_ok": int(gen == ref)}
    # blocks paired BY POSITION, so a missing first block does not credit its rows to
    # the second one
    for (rh, rrows), (gh, grows) in zip(ref, gen):
        if gh != rh:
            continue
        for n, body in rrows.items():
            out["blk_rows_ok"] += int(grows.get(n) == body)
    return out
