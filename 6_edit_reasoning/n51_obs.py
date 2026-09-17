# -*- coding: utf-8 -*-
"""The two observations n51 prints, precomputed for every round of a work dir.

    PYTHONPATH=. ARMS_RECIPE=main_fg python 6_edit_reasoning/n51_obs.py
    PYTHONPATH=. ARMS_RECIPE=main_fg python 6_edit_reasoning/n51_obs.py --arm n52

WHAT IS SELECTED. Over the pool that survives the filters below, the first observation
is the one on which the network's own top-q candidate stands furthest from the rest:

    obs1 = argmax_p  V[argmax q][p] - mean(V[others][p])          ("qmax_gap")

and the second is the next such column that is not saying the same thing, |rho| < 0.8
against the first across the candidates. Measured on the study corpus, qmax_gap picked
observations that took a GBM from a random column's 0.3728 to 0.4148 (z +21) on COMMIT,
and the permutation control -- q shuffled inside the round, everything else held -- put
that gain back at 0.3709 (z -0.97). The selection is the thing, not the printing.

WHY `base_cols` AND NOT `evidence_full`. Three reasons, and the third is a bug fix:

  size      `evidence_full` is 87 KB a round, so the main corpus would be 390 GB of
            intermediate. `base_cols` is 4.2 KB a round -- 19 GB for all 4.5M -- and
            `base_dump.py` writes it, `evidence_q` and `gates_full` from ONE forward.

  contents  `col_v` is [n_cand][len(col_idx)]: every candidate's value on every column
            the round has, which is exactly what a per-round argmax needs. The
            `evidence_*` dumps store each candidate's top-N gated features instead, and
            a top-N per candidate makes the ACROSS-candidate intersection collapse --
            that is what held the visible pool at 2.05 columns a round before the caps
            came off.

  the props `col_idx` is already conditioned on the round: a round carries
            `r_dmean__<prop>` / `r_dstd__<prop>` only for the properties it constrains
            (measured: 141 columns with two constrained properties, 147 with five, 149
            with six). Selecting off `evidence_full` needed a property filter bolted on
            top, keyed on a `targets` field that dump does not carry -- so `cons` came
            out empty on every round and all 35 property-keyed observations died
            silently. Here the filter is the data.

CANDIDATE COUNT. 1..4, not 4. A round with one candidate has no comparison to make and
gets no observations; the span is MOL_INFO, a one-row table and the commit. A round with
two gets one observation and skips the correlation test -- with two points every pair of
varying columns is perfectly correlated, so |rho| < 0.8 would reject every second slot
rather than deduplicate it, and the second column would add magnitude without adding
ordering. Saying one true thing is better than saying the same thing twice.

NEVER DROPS A ROUND. A round whose pool comes out empty gets an empty tuple and prints
no observation line. The alternative -- dropping it -- would give this arm a different
record set from the corpus it is built out of, and for a training corpus that is a
straight loss of trajectories rather than a controlled comparison.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import importlib
import json
import os
import pickle
import re
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
rp = importlib.import_module("6_edit_reasoning.recipe")

# What a forward pass could not produce, so the student could never write it. Same list
# `arms.NOT_VISIBLE` uses, plus the tier-6 demotion and the RDKit descriptors that are
# not recomputable from the page, plus the per-heavy normalisations and the three
# q-derived columns -- printing those would be printing q.
NOT_VISIBLE = ("c_prob", "c_is_unique_best", "ctx_", "site_", "cand_similarity")
TIER6 = {"c_snr_min"}
RDKIT = {"r_dsa", "r_dqed", "r_dlogp", "r_dtpsa", "r_dcharge"}
PER_HEAVY = {"r_dtpsa_per_heavy", "r_dlogp_per_heavy", "r_dhba_per_heavy"}
Q_DERIVED = {"c_rank", "c_prob_margin", "c_prob_z"}
RHO_MAX = 0.8


def visible(name: str) -> bool:
    return not (name.startswith(NOT_VISIBLE) or name in TIER6 or name in RDKIT
                or name in PER_HEAVY or name in Q_DERIVED)


def _rho(a: np.ndarray, b: np.ndarray) -> float:
    """|Pearson| of two columns across the candidates, 0 when either is flat."""
    a = a - a.mean()
    b = b - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(abs(np.dot(a, b) / (na * nb)))


def select(names, V, q, rng=None, mode="raw", k=2):
    """-> up to `k` column indices of V, by qmax_gap; random if `rng` is given.

    Returns a LIST, shorter than `k` when the round has nothing left to say that is not
    already said: the pool can hold fewer than `k` columns, and the dedup below can
    reject every remaining one. Callers pad; nobody drops a round over it.

    `V` is [n_cand, n_col] and already filtered to the visible, varying pool.

    `mode="z"` divides the gap by the column's spread ACROSS THE CANDIDATES before the
    argmax. Raw is what n51 and the main corpora were built with, and it makes the
    argmax a contest of UNITS: `r_dmw` is daltons and `fg_n_created` is a count, so the
    weight column wins whenever it varies at all -- MEASURED, `r_dmw` is the first
    observation on 36.4% of main-corpus rounds against 0 of the study's 110,158, which
    is the whole difference in one column. Dividing by the spread asks which column
    separates argmax(q) by the most of ITS OWN scale instead, which is the comparison
    the rule was meant to make. n56 is that; n51/n55 keep raw so the pair is readable.
    """
    n_cand = V.shape[0]
    if not names:
        return []
    if rng is not None:
        s = rng.random(len(names))
    else:
        hi = int(np.argmax(q))
        oth = np.delete(np.arange(n_cand), hi)
        # A one-candidate round never reaches here (`pool` is empty), so `oth` is
        # non-empty and the mean is defined.
        s = V[hi] - V[oth].mean(0)
        if mode == "z":
            # Population std over the candidates. Every column here varies (`pool_of`
            # dropped the flat ones), but a two-point column can still round to a
            # spread below the epsilon, so guard rather than divide.
            sd = V.std(0)
            s = np.where(sd > 1e-12, s / np.maximum(sd, 1e-12), 0.0)
    # THE FIRST SLOT IS `argmax`, NOT `argsort(-s)[0]`, and the two are not the same
    # function. `argmax` returns the FIRST maximal index; numpy's default argsort is
    # introsort and is not stable, so on a tied score it can put a different one in
    # front -- and ties are ordinary here, because most of the pool is integer counts.
    # Taking the head of the sort instead changed the first observation on 7% of the
    # study rounds while every other line of this function stayed the same.
    j1 = int(np.argmax(s))
    picked = [j1]
    order = [int(j) for j in np.argsort(-s) if int(j) != j1]
    # DEDUP AGAINST EVERY SLOT ALREADY TAKEN, not just the first. With k=2 that is the
    # original rule unchanged; with k=3 checking only slot 1 would let slots 2 and 3 be
    # the same reading twice, which is the thing the test exists to stop.
    for j in order:
        if len(picked) >= k:
            break
        if n_cand <= 2:
            # See the header: with two candidates |rho| is 1 for every varying pair, so
            # the test would reject everything instead of deduplicating.
            picked.append(j)
        elif all(_rho(V[:, p], V[:, j]) < RHO_MAX for p in picked):
            picked.append(j)
    return picked


def select_gate(names, V, G, mode="max", k=2, rng=None):
    """Pick by A's ATTENTION rather than by the column's values.

    `G` is [candidate, column], A's renormalised gate -- 1.0 is "attended as usual".
    `mode="max"` scores a column by the round's own gate (the max over the live
    candidates, which is what `gates_full` has always stored); `mode="spread"` scores
    it by max - min, "where does A look DIFFERENTLY depending on which edit it is
    scoring". The two were the whole of `CRITERIA_V35`.

    NEITHER READS `q`, AND THAT IS THE POINT. Every criterion that has been trained so
    far keys on the selector's argmax or on the reference, and the screen puts their
    leak term at 0.36-0.94 against a 0.25 floor. These two sit at 0.23-0.25 -- at the
    floor -- because A's gate is ~90% a property of the ROUND rather than of the
    candidate, so the choice cannot encode which edit wins. That also caps what they
    can carry: measured marginal gain +0.001 and +0.006 on pool B.

    The dedup is `select`'s, so the two slots are still not one reading twice.
    """
    if not names:
        return []
    if rng is not None:
        s = rng.random(len(names))
    elif mode == "spread":
        s = G.max(0) - G.min(0)
    else:
        s = G.max(0)
    n_cand = V.shape[0]
    j1 = int(np.argmax(s))
    picked = [j1]
    for j in [int(x) for x in np.argsort(-s) if int(x) != j1]:
        if len(picked) >= k:
            break
        if n_cand <= 2:
            picked.append(j)
        elif any(V[:, p].std() < 1e-12 or V[:, j].std() < 1e-12 for p in picked):
            # A FLAT COLUMN IS UNCORRELATED WITH EVERYTHING, not undefined. `_rho`
            # returns 0.0 on a zero-norm vector, so this branch only makes that
            # explicit -- pool B puts flat columns in reach and the dedup must not
            # reject a whole round because one of them came first.
            picked.append(j)
        elif all(_rho(V[:, p], V[:, j]) < RHO_MAX for p in picked):
            picked.append(j)
    return picked


# ---------------------------------------------------------------- n64: need-bearing
# THE OBSERVATION HAS TO BE ABOUT THE THING THAT IS OUT OF RANGE.
#
# MEASURED on n63's 7,976 test spans: the first observation names a property that is
# currently out of range on 17.1% of rounds. The page opens with `Out of range now:
# logD (above), MR (below)` and then argues about `the carbonyl o it adds` -- two
# unrelated arguments on one sheet. The cause is the selection rule: `qmax_gap` ranks
# columns by how far they separate argmax(q), and never looks at the box at all. The
# breakdown of the other 82.9%: 45.6% of picks are a functional-group or structural
# count with no property behind them, 22.0% are a property this round does not
# constrain, 14.8% are constrained but already IN range (where the honest direction is
# "stay", not "more"), 0.5% a spread.
#
# n64 restricts the pool instead of changing the rule: qmax_gap still chooses, but only
# out of the columns that bear on a property the round says is out of range. That keeps
# n61 - n64 readable as ONE change, and the restriction is computable by the student --
# the box is in the user turn and the current value is in the analyze_properties result,
# so nothing here needs `q`.
#
# THREE TIERS, SO NO ROUND IS DROPPED and the record set stays identical to every other
# arm. MEASURED: tier 0 alone holds >= 2 columns on 76.7% of rounds and >= 1 on 96.1%;
# adding tier 1 takes >= 2 to 94.3%. The fallback is the rest of the pool, so a round
# with nothing out of range still renders.
#
#   tier 0   about an out-of-range property AND "more/less is better" is defined
#   tier 1   about an out-of-range property, but a SPREAD -- no direction
#   tier 2   everything else, i.e. n57's pool unchanged
# NO DIRECT STRUCTURAL COUNTS HERE, and that is the point. `r_drotb` is the rotatable
# bonds the FRAGMENT carries; the table's `rotB` cell is `now + delta.rotB.avg`, mmpdb's
# average over the matched pairs. They disagree -- AUDITED: a span read `rotB 6 holds`
# (1 + 5) beside "the rotatable bonds it adds: Edit 1 adds 4", two numbers for one
# property on one page. Same for `r_dmw` against `delta.MW.avg`. Only the `r_dmean__X`
# and `r_dstd__X` family is the quantity the band cell is built from, so only that
# family may be presented as being ABOUT an out-of-range property. The structural counts
# are still in tier 2, where the span claims no such link.
_NEED_PROP = {}
# A STRUCTURAL COUNT THAT SHADOWS A CONSTRAINED PROPERTY IS OUT OF THE POOL ENTIRELY.
# `r_dmw` is the fragment's exact-mass delta and `r_drotb` the rotatable bonds it
# carries; the table's `MW` / `rotB` cells are `now + delta.avg`, mmpdb's average over
# the matched pairs. AUDITED: a span read `rotB 6 holds` (1 + 5) beside "the rotatable
# bonds it adds: Edit 1 adds 4" -- two numbers for one property on one page. Dropping
# them from the need tiers was not enough; they came back through tier 2 on 1.90% of
# slots. They are only dropped where the property is CONSTRAINED, because that is the
# only case where the table prints a rival number.
_SHADOW = {"r_dmw": "MW", "r_drotb": "rotB", "r_dhba": "HBA", "r_dhbd": "HBD",
           "r_dheavy": "heavy_atoms", "r_drings_arom": "rings_total",
           "r_drings_aliph": "rings_total"}
_MEAN_RE = re.compile(r"r_dmean__(.+)$")
_STD_RE = re.compile(r"r_dstd__(.+)$")


def need_of(nm):
    """(the property this column is about, whether a direction is defined) or (None, _)."""
    m = _MEAN_RE.match(nm)
    if m:
        return m.group(1), True
    m = _STD_RE.match(nm)
    if m:
        return m.group(1), False
    p = _NEED_PROP.get(nm)
    return (p, True) if p else (None, False)


def oor_props(rnd):
    """The properties the round's own NEEDS line names, as a set."""
    out = set()
    for p, b in (rnd.get("targets") or {}).items():
        v = float((rnd.get("props") or {}).get(p) or 0.0)
        lo = None if b[0] is None else float(b[0])
        hi = None if b[1] is None else float(b[1])
        if (lo is not None and v < lo) or (hi is not None and v > hi):
            out.add(p)
    return out


# THE PRINTED VALUE HAS TO BE THE ONE THE LABEL NAMES.
#
# `r_dmean__X` in `base_cols` is a NORMALISED feature, not the predicted shift.
# MEASURED against the round's own `delta`: it agrees on HBD and rings_total and on
# nothing else -- MW 1.3788 where the shift is 158.956, MR 1.1514 where it is 35.58,
# logD 0.5149 where it is 0.899, heavy_atoms divided by 7, rotB by 3. So a span that
# says "the MW shift it predicts: Edit 1 predicts 1.38" beside a table cell reading
# `MW 419.107` is stating a number that is neither the shift nor anything the reader
# can check. This has been true since n51.
#
# The SELECTION still runs on the normalised column -- that is what qmax_gap was
# defined on and changing it would move two things at once -- but what gets PRINTED is
# `delta[X]["avg"]` / `["std"]`, which is exactly the number the band interval
# `now + avg +- std` is built from. The observation line and the table then agree.
def real_values(rnd, nm):
    """the per-candidate values a `r_dmean__X`/`r_dstd__X` observation should print."""
    m = _MEAN_RE.match(nm) or _STD_RE.match(nm)
    if not m:
        return None
    prop, key = m.group(1), ("avg" if _MEAN_RE.match(nm) else "std")
    out = []
    for c in rnd["candidates"]:
        d = (c.get("delta") or {}).get(prop) or {}
        v = d.get(key)
        if v is None:
            return None
        out.append(float(v))
    return out


# What pool P+F holds. The property deltas are the numbers the band cell's interval is
# built from; the functional-group counts are the ones the student writes most
# accurately (0.992 exact against 0.980 for a property delta and 0.955 for a structural
# count, measured on n63 epoch 4). Everything else -- `r_dsp3`, `r_heavy_from`,
# `r_n_cuts`, the edit-kind flags -- stays out: it is neither on the page nor countable
# off the two SMILES.
_POOL_PF = ("r_dmean__", "r_dstd__", "r_dfg__", "fg_n_")


def prop_tiers(names, constrained=()):
    """POOL P+F: property deltas and functional-group counts first, the rest as top-up.

    n64's first pool was cut to the columns about a property the round says is OUT of
    range. MEASURED with the GBM pre-test (the two chosen columns handed to a
    HistGradientBoosting over the on-path label, GroupKFold 5): that pool separates the
    criterion from a random draw by +0.068, this one by +0.125, and the unrestricted
    pool by +0.139. P keeps 90% of the separation the full pool offers and gives up
    almost nothing that matters:

    MEASURED with the GBM pre-test (the two chosen columns handed to a
    HistGradientBoosting over the on-path label, GroupKFold 5), P+F at 15.86 columns a
    round: gap_z +0.1283 against its random draw, gap_raw +0.1225, A_riv_QMAX +0.1030.
    Adding F to P is worth +0.0006 for gap_z and costs A_riv_QMAX 0.030 -- F is in the
    pool because the counts are the ones the student can actually write, not because
    they carry selection signal (F alone is +0.064, half of P).

    THE SCALE SKEW IS WHY `gap_raw` IS NOT THE DEFAULT HERE. A group count is a raw
    integer and `r_dstd__X` is a normalised fraction, so the raw argmax takes F on
    0.585 of slots against a random draw's 0.297 and `std` on 0.084 against 0.314. With
    `mode="z"` the mix comes back to 0.383/0.260/0.357 against 0.389/0.314/0.297 -- a
    matched control the arm can actually be read against.
    """
    p, rest = [], []
    for j, nm in enumerate(names):
        if _SHADOW.get(nm) in constrained:
            continue
        (p if nm.startswith(_POOL_PF) else rest).append(j)
    return [p, rest]


def need_tiers(names, oor, constrained=()):
    t0, t1, t2 = [], [], []
    for j, nm in enumerate(names):
        if _SHADOW.get(nm) in constrained:
            continue                      # see _SHADOW: the table prints a rival number
        p, directed = need_of(nm)
        if p is not None and p in oor:
            (t0 if directed else t1).append(j)
        else:
            t2.append(j)
    return [t0, t1, t2]


def select_tiered(names, V, q, tiers, mode="raw", k=2, rng=None, rival=None):
    """`select`'s rule, walked tier by tier. NOT a refactor of `select`: that function
    is what seven trained arms were built with and it stays byte-for-byte what it was.

    Ties are broken by column index here rather than by `np.argsort`, which is
    introsort and unstable -- the same trap the comment in `select` records, and the
    reason this sorts on `(-s[j], j)` instead.
    """
    if not names:
        return []
    n_cand = V.shape[0]
    if rng is not None:
        # THE MATCHED CONTROL DRAWS FROM THE SAME TIERS. A random arm that drew from the
        # unrestricted pool would move the selection AND the pool at once, and
        # "selection beats random" is not a claim that comparison can make -- the
        # mistake n52 vs n57 made, which is why n58 had to be built.
        s = rng.random(len(names))
    else:
        hi = int(np.argmax(q))
        oth = np.delete(np.arange(n_cand), hi)
        sd = np.maximum(V.std(0), 1e-12)
        if mode == "rival" and rival is not None and 0 <= int(rival) < n_cand \
                and int(rival) != hi:
            # AGAINST ONE NAMED RIVAL, not against the mean of the other three. The
            # rival is the one `evidence_ac` records -- the candidate the trained
            # selector's own candidate-attention puts against the pick -- so this asks
            # which column separates argmax(q) from the edit the model itself thinks is
            # its closest competitor. Pre-test: +0.1329 against a random draw, the best
            # of thirteen criteria, with the highest AUC (0.6326) of any of them.
            s = (V[hi] - V[int(rival)]) / sd
        else:
            s = V[hi] - V[oth].mean(0)
            if mode == "z":
                s = s / sd
    # NO PASS CONDITIONS. Walk the tiers in order and take the top `k` DISTINCT
    # columns by score, nothing else. The two dedups that used to sit here -- one
    # property per span, and |rho| < 0.8 across the candidates -- are gone: they made
    # the arm's selection a function of what had already been picked, so n64 and n65
    # were not choosing out of the same set at the second slot. With the filters off,
    # both arms score the SAME tiers and differ only in the score, which is what
    # `n64 - n65` is supposed to isolate.
    picked = []
    for tier in tiers:
        for j in sorted(tier, key=lambda t: (-s[t], t)):
            if len(picked) >= k:
                break
            if j not in picked:
                picked.append(j)
        if len(picked) >= k:
            break
    return picked


def pool_of(names_all, row, drop_scores=False, keep_flat=False):
    """The round's visible, discriminating columns -> (names, V, q).

    `col_idx` has already applied the property filter (see the header), so the only
    cuts here are visibility and "does this column tell the candidates apart at all".

    `drop_scores` additionally removes the whole `c_*` family -- the box arithmetic:
    how much gap the move closes, how many constraints it helps, what it leaves behind.
    Those are honestly visible in the sense `visible()` means, and the student still
    cannot produce them: MEASURED on the free heldout rollout, the value vector it
    writes for a `c_*` axis is right 64.7% of the time against 99.0% for a structural
    count and 99.5% for a functional-group count, with `c_cos` at 1 of 59 and
    `c_gap_reduction` at 0 of 19. `visible_ceiling.py` had already priced the family at
    0.347 against a 0.398 floor. n57 is the pool without it.
    """
    V_all = np.asarray(row["col_v"], dtype=np.float64)         # [n_cand, n_col]
    q = np.asarray(row["q_all"], dtype=np.float64)
    keep, names = [], []
    for k, gi in enumerate(row["col_idx"]):
        nm = names_all[gi]
        if not visible(nm):
            continue
        if drop_scores and nm.startswith("c_"):
            continue
        col = V_all[:, k]
        # POOL B keeps it. A column with one value across the candidates cannot RANK
        # them, but the page already has a slot for that kind of line -- `_cols_for`
        # prints the constants under "these are the same for all four edits, so they
        # say what the edits have in common and cannot separate them". Whether an
        # observation that RULES AN AXIS OUT is worth a sentence is the question POOL B
        # asks, and the screen says it raises every criterion's marginal gain (gap_z
        # +0.044 -> +0.067, gate_ihat -0.010 -> +0.014) while leaving the leak term
        # untouched. Default OFF: pool A is what n51..n75 were built on.
        if not keep_flat and len(set(np.round(col, 6))) < 2:
            continue
        keep.append(k)
        names.append(nm)
    return names, (V_all[:, keep] if keep else V_all[:, :0]), q


# WHAT EACH ARM CHANGES, and it is exactly one thing each. The page, the wording, the
# dedup and the record set are the same for all five, so every pair below is readable.
#
#   n51  qmax_gap over the visible pool                    the shipped selection
#   n52  the same pool, drawn at random                    n51 - n52 = the selection
#   n55  n51's rule, and nothing else changed              n51 - n55 = the POOL, once
#        -- it exists to be built on a work dir whose      the source is fixed (this is
#        `base_cols` was joined rather than gate-capped    only distinguishable on the
#                                                          study dir; see commit 0843ff2)
#   n56  n55 with the gap divided by the column's spread   n55 - n56 = raw units
#   n57  n55 with the `c_*` box arithmetic out of the pool n55 - n57 = the axes the
#                                                          student cannot reproduce
#   n58  n57's POOL, drawn at RANDOM                       n57 - n58 = the selection,
#                                                          on the pool n57 actually has
#   n59  n57 with THREE observations instead of two        n57 - n59 = a third reading
#
# n55 IS NOT A NO-OP ON A MAIN CORPUS. There it selects identically to n51 and the two
# pickles come out equal; it earns its name only where the two sources differ.
#
# WHY n58 IS THE ONE THAT WAS MISSING. n52 is the random control for n51's POOL -- the
# 4.43-column `evidence_full` one, where `r_dmw` and every property delta are absent
# (measured: 73 axes, r_dmw 0.0000). Reading n57 against n52 therefore moves the pool
# AND the rule at once, and "selection beats random" is not a claim that comparison can
# make. n58 is the same draw on n57's own pool, so n57 - n58 isolates the rule.
#
# WHY n59. On the free rollout the commit follows the FIRST observation the span names,
# and a wrong first axis is where n57's whole free-vs-forced gap sits (matched-axis
# rounds: free == forced, p=0.715). A third reading is the cheapest way to ask whether
# the span needs more evidence in front of the commit or a better first pick.
#   n64  n57's pool RESTRICTED to columns that bear on an   n61 - n64 = whether the
#        out-of-range property, same qmax_gap inside it      observation being ABOUT
#                                                            the stated need is worth
#                                                            anything
#   n65  n64's POOL AND TIERS, drawn at RANDOM               n64 - n65 = THE SELECTION,
#                                                            and this pair is the one
#                                                            the arm has to win
_MODE = {"n51": "raw", "n52": "raw", "n55": "raw", "n56": "z", "n57": "raw",
         "n58": "raw", "n59": "raw",
         # POOL P FAMILY. One pool, four scoring rules, and nothing else differs --
         # same page, same wording, same records. n65 is the matched random draw the
         # other three have to beat.
         "n64": "raw", "n65": "raw", "n66": "z", "n67": "rival", "n68": "z", "n69": "raw",
         # n70: n57 WITH ONE KNOB MOVED. Same pool, same dedup, same page, same
         # normalised printed values, no reorder -- only the score is divided by the
         # column's spread across the candidates. n57 - n70 is the criterion and
         # nothing else, which is the contrast n56 was supposed to be and was not
         # (n56 sits on n55's pool and keeps the `c_*` score columns).
         "n70": "z",
         # n75: n70's RULE AND POOL, KEYED ON THE REFERENCE EDIT. `select` reads the
         # privileged vector only through `argmax(q)`, so handing it a one-hot at
         # `pick` puts the beam-search answer where the selector's argmax used to be
         # and changes nothing else -- same pool, same dedup, same normalisation, same
         # page. n70 - n75 is therefore the STRENGTH of the teacher signal and nothing
         # else: i-hat agrees with the reference on 0.418 of rounds, the reference with
         # itself on all of them.
         #
         # WHY IT IS WORTH RUNNING AND NOT JUST ARGUING ABOUT. The pre-test will score
         # n75 near 1.0 by construction, which is exactly why the pre-test cannot
         # settle it: on n72 the same pre-test predicted +0.126 over random and the
         # trained students came out at -0.009. If n75 lands BELOW n72 then "the
         # stronger the teacher signal, the worse the student" is monotone over three
         # points (random 0.504 > argmax-q 0.495 > reference), which is a claim no
         # single pair can make.
         "n75": "z",
         # POOL B FAMILY. One pool -- the n70 pool with the flat columns kept -- and
         # three ways to choose out of it. n78 is the matched random draw the other two
         # have to beat, and it is drawn from THE SAME POOL, which is the mistake n52
         # vs n57 made and n58 had to be built to fix.
         #
         #   n76  A's round-level gate      `gates_full`'s `rule_gate`, restored per
         #                                  candidate and maxed back over them
         #   n77  A's gate SPREAD           max - min over the candidates
         #   n78  uniform over pool B       the control
         #
         # WHY THESE TWO AND NOT A SHARPER ONE. The screen puts every criterion that
         # keys on `q` or on the reference at a leak term of 0.36-0.94 against a 0.25
         # floor, and all four trained arms ordered by that term and nothing else
         # (none 0.510 > random 0.504 > gap_z 0.495 > reference 0.423). n76 and n77 sit
         # AT the floor (0.233 and 0.248) because A's gate is ~90% a property of the
         # round rather than of the candidate. They are the first arms that ask what an
         # observation is worth when the choice carries no information about the answer.
         "n76": "gate_max", "n77": "gate_spread", "n78": "raw"}
# Pool B: the flat columns are kept. See `pool_of`.
_KEEP_FLAT = {"n76", "n77", "n78"}
# Arms scored on A's gate rather than on the column's values. See `select_gate`.
_GATE_MODE = {"n76": "max", "n77": "spread"}
_DROP_SCORES = {"n57", "n58", "n59", "n64", "n65", "n66", "n67", "n68", "n69", "n70",
                "n75", "n76", "n77", "n78"}
# Arms whose criterion is keyed on the reference edit instead of on `q`. See `_MODE`.
_REF_PICK = {"n75"}
_RANDOM = {"n52", "n58", "n65", "n69", "n78"}   # drawn from the pool rather than scored
_N_OBS = {"n59": 3}                      # observations per span; 2 everywhere else
_PROP_POOL = {"n64", "n65", "n66", "n67"}   # pool P+F: property deltas and FG counts
# n68: n57's POOL AND n57's DEDUP, with everything else the later arms changed.
#
# n57 selects over the whole live pool -- 22.8 columns, every visible non-`c_*` column
# that tells the candidates apart -- and takes the second slot by walking the score
# order until it finds one with |rho| < 0.8 against the first. The arms after it cut the
# pool to 15.9 columns and dropped the dedup. Those two are the only changes since n57
# that have never been trained, and this arm puts them back while keeping the six that
# were asked for: clause values, the table description line, the interval band cells,
# the real `delta` numbers, `gap_z`, and the concessive closer.
#
# n56 IS THE MATCHED COMPARATOR AND ALREADY EXISTS. It is gap_z over the same pool with
# the same dedup and the OLD page, so n56 - n68 is the page and n57 - n56 is the
# criterion -- two readable contrasts out of runs that are already on disk.
# n69 IS n68's MATCHED CONTROL: the same pool, the same |rho| dedup, the same page,
# drawn uniformly instead of by gap_z. n68 - n69 is the criterion and nothing else --
# the pair n57 never had, because n57's own random twin was never built.
_FULLPOOL = {"n68", "n69"}
# Arms that PRINT `delta[X]["avg"]`/`["std"]` instead of the normalised feature. Not the
# same set as the pool choice: n68 takes n57's pool and still has to quote numbers the
# band cell's interval is built from.
_REAL_VALUES = _PROP_POOL | _FULLPOOL
_NEED_ONLY = _PROP_POOL | _FULLPOOL          # these arms join base_cols with the round


def _git_rev():
    """The commit the corpus was built at, or None outside a checkout."""
    import subprocess
    try:
        return subprocess.run(["git", "-C", os.path.dirname(os.path.abspath(__file__)),
                               "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:
        return None


def recipe_of(arm):
    """EVERY KNOB THAT DEFINES THIS ARM'S SELECTION, as plain data.

    The pickle used to be the whole record: which pool, which criterion, whether the
    dedup ran -- all of it lived in this file and could only be recovered by finding the
    commit the build ran at. Six arms in, `n56 keeps c_* and n57 does not` was a fact
    nobody could read off the artefacts, and it invalidated a comparison. This is
    written beside the pickle so the artefact answers for itself.
    """
    if arm in _FULLPOOL:
        pool = ("base_cols -> visible() -> drop c_* -> drop columns flat across the "
                "candidates -> drop _SHADOW columns whose property is constrained")
        dedup = "|rho| < %.2f against every slot already taken" % RHO_MAX
    elif arm in _PROP_POOL:
        pool = ("base_cols -> visible() -> drop c_* -> drop flat -> drop _SHADOW -> "
                "r_dmean__/r_dstd__/r_dfg__/fg_n_ first, the rest only as a top-up")
        dedup = "none -- the top k distinct columns, in score order"
    elif arm in _KEEP_FLAT:
        pool = ("base_cols -> visible()%s -- POOL B: the columns FLAT across the "
                "candidates are KEPT" % (" -> drop c_*" if arm in _DROP_SCORES else ""))
        dedup = ("|rho| < %.2f against every slot taken, a flat column counting as "
                 "uncorrelated" % RHO_MAX)
    else:
        pool = "base_cols -> visible()%s -> drop columns flat across the candidates" % (
            " -> drop c_*" if arm in _DROP_SCORES else "")
        dedup = "|rho| < %.2f against every slot already taken" % RHO_MAX
    mode = _MODE.get(arm, "raw")
    crit = {"raw": "qmax_gap: V[argmax q] - mean(V[others])",
            "z": "qmax_gap / the column's spread across the candidates",
            "rival": "(V[argmax q] - V[A_c rival]) / the column's spread",
            "gate_max": "A's gate, max over the candidates (reads no q)",
            "gate_spread": "A's gate, max - min over the candidates (reads no q)",
            }[mode]
    if arm in _RANDOM:
        crit = "UNIFORM RANDOM, seeded on the round key (the matched control)"
    return {"arm": arm, "pool": pool, "criterion": crit, "mode": mode,
            "random": arm in _RANDOM, "observations_per_span": _N_OBS.get(arm, 2),
            "dedup": dedup, "drops_c_star": arm in _DROP_SCORES,
            "prints_real_delta": arm in _REAL_VALUES,
            # gated to the new family so the back catalogue rebuilds byte for byte
            "mean_before_its_own_spread": arm in _NEED_ONLY,
            "rival_source": "evidence_ac" if mode == "rival" else None}


def build(work, arm, splits, out_path, src="base_cols", limit=0):
    names_all = json.load(open(f"{work}/{src}/rule_names.json"))["rule_names"]
    labels = json.load(open(f"{work}/{src}/rule_names.json")).get("labels") or {}
    mode = _MODE.get(arm, "raw")
    drop_scores = arm in _DROP_SCORES
    k = _N_OBS.get(arm, 2)
    out, stat = {}, {"rounds": 0, "none": 0, "ncand": {}, "slots": {}}
    for split in splits:
        paths = sorted(glob.glob(f"{work}/{src}/{split}.jsonl")) or \
                sorted(glob.glob(f"{work}/{src}/{split}.part*.jsonl"))
        if not paths:
            print(f"  {split}: no {src} shards, skipped", flush=True)
            continue
        # THE BOX AND THE CURRENT VALUE LIVE IN `rounds/`, NOT IN `base_cols` -- the
        # dump carries col_idx/col_v/q_all and nothing about the targets -- so the
        # need-bearing arms join the two here. Loaded once per split, and only the
        # property SET is kept, so this is a few MB rather than the rounds themselves.
        oor_of, rnd_of, con_of, riv_of, gate_of = {}, {}, {}, {}, {}
        gidx = {n_: k_ for k_, n_ in enumerate(names_all)}
        if arm in _GATE_MODE:
            # A's [candidate][rule] gate, from `gate_dump.py --per-cand`. The folded
            # `gates_full` cannot serve here: its `rule_gate` is the round-level max and
            # is identical for every edit, so `spread` would come out all zeros.
            import glob as _glob
            for gp in sorted(_glob.glob(f"{work}/gates_cand/{split}.part*.jsonl")
                             or _glob.glob(f"{work}/gates_cand/{split}.jsonl")):
                with open(gp) as gfh:
                    for gl in gfh:
                        if not gl.strip():
                            continue
                        ge = json.loads(gl)
                        m = ge.get("rule_gate_cand")
                        if m:
                            gate_of[f'{ge["group_id"]}|{ge["depth"]}'] = np.asarray(
                                m, dtype=np.float64)
            print(f"  {split}: per-candidate gates for {len(gate_of):,} rounds",
                  flush=True)
        if arm in _NEED_ONLY:
            for rp in sorted(glob.glob(f"{work}/rounds/{split}/*.jsonl")):
                with open(rp) as rfh:
                    for rl in rfh:
                        if not rl.strip():
                            continue
                        rr = json.loads(rl)
                        kk = f'{rr["group_id"]}|{rr["depth"]}'
                        oor_of[kk] = oor_props(rr)
                        con_of[kk] = set(rr.get("targets") or ())
                        # only the deltas are kept, so this is a fraction of the rounds
                        rnd_of[kk] = {"candidates": [
                            {"delta": c.get("delta") or {}} for c in rr["candidates"]]}
            if mode == "rival":
                # the candidate-attention rival, from the same GPU pass the `model`
                # arm's prompt is rendered from
                for ep in sorted(glob.glob(f"{work}/evidence_ac/{split}/*.jsonl")):
                    with open(ep) as efh:
                        for el in efh:
                            if not el.strip():
                                continue
                            ee = json.loads(el)
                            riv_of[f'{ee["group_id"]}|{ee["depth"]}'] = ee.get("rival")
                print(f"  {split}: rivals for {len(riv_of):,} rounds", flush=True)
            print(f"  {split}: out-of-range sets for {len(oor_of):,} rounds", flush=True)
        got = 0
        for path in paths:
            with open(path) as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    key = f'{row["group_id"]}|{row["depth"]}'
                    names, V, q = pool_of(names_all, row, drop_scores,
                                          arm in _KEEP_FLAT)
                    nc = int(row.get("n_cand") or V.shape[0])
                    if arm in _REF_PICK and q.size:
                        # ONE-HOT AT THE REFERENCE. `select` uses q only as
                        # `hi = argmax(q)`, so this substitutes the answer for the
                        # selector's pick without touching the pool or the score.
                        _p = int(row.get("pick", 0))
                        if 0 <= _p < q.size:
                            q = np.zeros_like(q)
                            q[_p] = 1.0
                    stat["ncand"][nc] = stat["ncand"].get(nc, 0) + 1
                    rng = None
                    if arm in _RANDOM:
                        # The matched control: the SAME pool, the same dedup, the same
                        # wording, drawn at random instead of by qmax_gap. Seeded on the
                        # round key so the corpus is a fixed object and a re-render
                        # reproduces it exactly. Keyed on the ROUND alone, so n52 and
                        # n58 draw the same ordinal on a round -- they differ only by
                        # the pool they draw it out of, which is the point of n58.
                        rng = np.random.default_rng(int.from_bytes(
                            hashlib.blake2b(key.encode(), digest_size=8).digest(),
                            "little"))
                    if arm in _FULLPOOL:
                        # n57's `select` VERBATIM -- same argmax, same |rho| < 0.8 dedup
                        # -- over n57's pool minus only the columns that shadow a
                        # constrained property (see `_SHADOW`: those print a rival
                        # number for a property the table already gives).
                        con = con_of.get(key, ())
                        kp = [j for j, nm_ in enumerate(names)
                              if _SHADOW.get(nm_) not in con]
                        sub = select([names[j] for j in kp], V[:, kp], q, rng, mode, k)
                        js = [kp[j] for j in sub]
                    elif arm in _PROP_POOL:
                        js = select_tiered(
                            names, V, q,
                            prop_tiers(names, con_of.get(key, ())),
                            mode, k, rng, riv_of.get(key))
                        stat["tier0"] = stat.get("tier0", 0) + sum(
                            1 for j in js if need_of(names[j])[0]
                            in oor_of.get(key, set()) and need_of(names[j])[1])
                        stat["need"] = stat.get("need", 0) + sum(
                            1 for j in js
                            if need_of(names[j])[0] in oor_of.get(key, set()))
                        stat["slots_tot"] = stat.get("slots_tot", 0) + len(js)
                    elif arm in _GATE_MODE:
                        gk = f'{row["group_id"]}|{row["depth"]}'
                        Gr = gate_of.get(gk)
                        if Gr is None:
                            stat["nogate"] = stat.get("nogate", 0) + 1
                            continue
                        Gp = Gr[:, [gidx[n_] for n_ in names]]
                        if Gp.shape[0] != nc:
                            stat["nogate"] = stat.get("nogate", 0) + 1
                            continue
                        js = select_gate(names, V, Gp, _GATE_MODE[arm], k, rng)
                    else:
                        js = select(names, V, q, rng, mode, k)
                    # THE MEAN COMES BEFORE ITS OWN SPREAD. Pool P is flat, so a
                    # score can rank `r_dstd__X` above `r_dmean__X` and the span then
                    # states the uncertainty of a number it has not given yet --
                    # MEASURED on 5.5% of rounds, and `nat_span`'s "And the spread on
                    # that shift" only recognises the mean-first order, so those also
                    # lost the one connective that describes the pair. This reorders
                    # the PRINTING and nothing else: the same two columns are chosen.
                    #
                    # GATED TO THE NEW FAMILY, so n51..n63 rebuild byte for byte. The
                    # reorder is a 2026-09-11 change and every arm before it was trained
                    # on pickles that did not have it; leaving it ungated would have
                    # made `n57` mean one thing on disk and another in this file.
                    if arm in _NEED_ONLY and len(js) == 2:
                        a_, b_ = names[js[0]], names[js[1]]
                        if (a_.startswith("r_dstd__") and b_.startswith("r_dmean__")
                                and a_[len("r_dstd__"):] == b_[len("r_dmean__"):]):
                            js = [js[1], js[0]]
                            stat["reordered"] = stat.get("reordered", 0) + 1
                    stat["rounds"] += 1
                    stat["slots"][len(js)] = stat["slots"].get(len(js), 0) + 1
                    if not js:
                        out[key] = ()
                        stat["none"] += 1
                    else:
                        rec = []
                        for j in js:
                            vals = V[:, j].tolist()
                            if arm in _REAL_VALUES and key in rnd_of:
                                rv = real_values(rnd_of[key], names[j])
                                if rv is not None:
                                    vals = rv
                                    stat["real"] = stat.get("real", 0) + 1
                            rec += [names[j], labels.get(names[j], names[j]), vals]
                        out[key] = tuple(rec)
                    got += 1
                    if limit and got >= limit:
                        break
            if limit and got >= limit:
                break
        print(f"  {split}: {got} rounds", flush=True)
    with open(out_path, "wb") as fh:
        pickle.dump(out, fh, protocol=4)
    # THE SETTINGS, BESIDE THE ARTEFACT. Stats as well as knobs: a family mix that has
    # drifted is the first sign a pool changed under an arm, and it is cheap to record.
    import datetime as _dt
    fam = {}
    for _v in out.values():
        for _i in range(0, len(_v), 3):
            _n = _v[_i]
            _f = ("property delta (mean)" if _n.startswith("r_dmean__") else
                  "property delta (spread)" if _n.startswith("r_dstd__") else
                  "functional group" if _n.startswith(("r_dfg__", "fg_n_")) else
                  "structural")
            fam[_f] = fam.get(_f, 0) + 1
    _t = max(sum(fam.values()), 1)
    meta = {"recipe": recipe_of(arm), "git_commit": _git_rev(),
            "written_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "work_dir": work, "source": src, "splits": list(splits),
            "rounds": len(out),
            "slots_per_span": {str(k): v / max(stat["rounds"], 1)
                               for k, v in sorted(stat["slots"].items())},
            "family_share": {k: v / _t for k, v in sorted(fam.items())},
            "reordered_mean_first": stat.get("reordered", 0) / max(stat["rounds"], 1),
            "need_bearing_slots": (stat.get("need", 0) / stat["slots_tot"]
                                   if stat.get("slots_tot") else None)}
    with open(out_path + ".meta.json", "w") as fh:
        json.dump(meta, fh, indent=1)
    print(f"  settings -> {out_path}.meta.json")
    n = max(stat["rounds"], 1)
    print(f"wrote {out_path}  ({len(out)} rounds, "
          f"{os.path.getsize(out_path)/1e6:.1f} MB)")
    print("  observations per span: " + "   ".join(
        f"{s}: {c/n:.4f}" for s, c in sorted(stat["slots"].items())))
    print(f"  candidate counts {dict(sorted(stat['ncand'].items()))}")
    if stat.get("reordered"):
        print(f"  mean put back in front of its own spread on "
              f"{stat['reordered']/max(stat['rounds'],1):.4f} of rounds")
    if stat.get("slots_tot"):
        t = stat["slots_tot"]
        print(f"  observations ABOUT an out-of-range property: "
              f"{stat.get('need', 0)/t:.4f}  "
              f"(with a direction: {stat.get('tier0', 0)/t:.4f})  of {t:,} slots")
    return stat


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default=None, help="default: recipe.work_dir(DEFAULT)")
    ap.add_argument("--arm", default="n51",
                    choices=["n51", "n52", "n55", "n56", "n57", "n58", "n59", "n64", "n65", "n66", "n67", "n68", "n69", "n70", "n75", "n76", "n77", "n78"],
                    help="n52/n58 draw from the pool at random (the controls); "
                         "n55..n59 are the pool-fix cycle, see _MODE; n64 restricts "
                         "the pool to columns about an out-of-range property")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--src", default="base_cols",
                    help="the base_dump output to select from")
    ap.add_argument("--out", default=None, help="default: <work>/<arm>_obs.pkl")
    ap.add_argument("--limit", type=int, default=0, help="rounds per split; 0 = all")
    a = ap.parse_args()
    work = a.work or rp.work_dir(rp.DEFAULT)
    out = a.out or f"{work}/{a.arm}_obs.pkl"
    print(f"n51_obs: work={work} arm={a.arm} src={a.src}")
    build(work, a.arm, a.splits, out, a.src, a.limit)


if __name__ == "__main__":
    main()
