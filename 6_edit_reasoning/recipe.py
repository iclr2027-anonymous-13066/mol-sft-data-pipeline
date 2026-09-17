# -*- coding: utf-8 -*-
"""The frozen data recipe — everything that must not move between arms.

Four arms train on the SAME rounds, the SAME candidate ordering and the SAME
hyperparameters. Only the reasoning span between `suggest_edits` and `edit_fragment`
differs. That is only true if the selection is deterministic, so it lives here rather
than in the scripts: `Recipe` is hashed into a version id and the id goes into every
output path.

**The `suggest_edits` response is used exactly as the tool returned it.** No reordering,
no reranking, no rewriting. An earlier version of this recipe permuted the candidate
list to move the committed edit to a hash-chosen position; that is gone, and removing it
was an improvement rather than a concession. In the untouched corpus, POSITION and
`predicted_gap` RANK are very nearly the same variable — the tool returns its candidates
ranked — so balancing the committed candidate's index balances both shortcuts at once.
MEASURED on 1000 index-balanced rounds per corpus, natural order:

    always #1                            25.0%   (by construction)
    argmin(predicted_gap), ties->first   25.0%
    argmin(predicted_gap), ties->last    21.5% scaffold / 20.0% fg

Under permutation the two decouple and `argmin(gap)` climbed back to 35.9%, because
moving a candidate moves its numbers with it. So the natural order plus an index quota
is the design that leaves the span as the only route to the answer.

**Balance is by SELECTION, with a per-index quota.** The committed candidate sits at #1
in 48% of scaffold rounds and 42% of fg rounds, and at #4 in 14% / 17%. `prepare.py`
therefore accepts a record only while every index its rounds contribute still has quota
left. Two consequences worth stating: the corpus is deliberately NOT representative of
the natural index distribution, and records whose rounds land on rare indices are
favoured. Both are the point of balancing, and every arm gets the identical record set,
so neither biases the comparison.

**The split is by seed-molecule Murcko scaffold**, not by instance id and not at random.
Instance ids are not comparable across instance files, and a random split puts the same
scaffold on both sides, where memorisation reads as accuracy.
"""
from __future__ import annotations

import hashlib
import json
import os as _os
from dataclasses import asdict, dataclass


# A corpus NAME is what `Recipe.corpora` carries and therefore what the digest -- and so
# the work and data directory -- is keyed on.  The PATH is not, so pointing an existing
# name at a different directory would silently append to that name's finished work dir,
# whose stage markers would then skip the rebuild.  A new corpus is always a NEW NAME.
CORPORA = {
    "scaffold": "data/training_data/sftdata_fullctx/"
                "generation_2m_scaffold_naive_satonly",
    "fg": "data/training_data/sftdata_fullctx/"
          "generation_2m_fg_naive_satonly",
    # The beam-recovered trajectories: instances whose stage-3 search had failed, re-run
    # at beam_width 16 / expand 4 / depth 8 and solved.  Same instances, same seeds and
    # the same tool set as the two above -- what differs is DEPTH.  Recovery keeps 56.6%
    # of fg and 44.8% of scaffold, and the chains it keeps average 5.33 edits against
    # 3.22 and 1.87, so they fill the edit>=4 tier the base corpora barely reach
    # (scaffold tops out at 7 edits and puts 49.2% of its records at exactly 1).
    "scaffold_rec": "data/training_data/sftdata_fullctx/"
                    "generation_2m_scaffold_recovered_dummy",
    "fg_rec": "data/training_data/sftdata_fullctx/"
              "generation_2m_fg_recovered_dummy",
    # THE SAME TWO PATHS AS `scaffold`/`fg`, UNDER A SECOND NAME. A corpus name is a
    # digest input and a path is not, which is what makes this the safe way to change
    # which RECORDS a build takes without touching the finished work dirs of the name
    # it shares a path with.  `DROP_RECOVERED` below is what the second name means.
    "scaffold_keep": "data/training_data/sftdata_fullctx/"
                     "generation_2m_scaffold_naive_satonly",
    "fg_keep": "data/training_data/sftdata_fullctx/"
               "generation_2m_fg_naive_satonly",
}

# Corpora that drop ONLY the seed stubs a beam re-run has already replaced.
#
# `make_satisfied_only` keeps a satisfied chain whole and cuts an unsatisfied one down
# to its seed segment, so a satonly corpus holds two kinds of record that a round count
# cannot tell apart -- both have zero edit rounds:
#
#   normal, 0 edits          the seed already satisfies the box and the record ends on
#                            <ANSWER>.  A complete trajectory, and the one case where
#                            "no edit" IS the answer.  `min_rounds_per_record=0` exists
#                            to keep these.
#   no <ANSWER>              the planner FAILED; what survives is the SMARTS and the
#                            seed SMILES with no answer at the end.
#
# `min_rounds_per_record=1` used to drop both.  Keeping the first means the second has
# to be decided on what it is rather than on how many rounds it has -- and NOT all of
# them can go.  `3_toolchain_gen` was re-run at beam 16 on the failed instances and got
# through 34,004 of fg but only 117,088 of scaffold's 197,310 before it was stopped, so
# a blanket drop would throw away the 80,222 scaffold stubs nothing has replaced yet.
#
# So the rule is REPLACEMENT, not `ends_with_answer`: a stub goes only when a whole trajectory
# for the same instance now exists in `*_recovered_dummy`.  The id files are written
# from those corpora and are the record of what was actually built.
DROP_RECOVERED = {
    # `WORK_ROOT` is defined below this block, so the path is written out.
    "scaffold_keep": "data/analysis/reasoning_arms/_recovered_ids/scaffold.txt",
    "fg_keep": "data/analysis/reasoning_arms/_recovered_ids/fg.txt",
}


def recovered_ids(corpus: str) -> frozenset:
    """`group_id`s already replaced by a recovered trajectory, or empty."""
    path = DROP_RECOVERED.get(corpus)
    if not path or not _os.path.exists(path):
        return frozenset()
    with open(path) as fh:
        return frozenset(x.strip() for x in fh if x.strip())

ARMS = ("naive", "allfeat", "model", "noreason")

# Intermediates (rounds, evidence tensors, generated spans) and the assembled training
# corpora live apart: the second is what a training config points at, and it should be
# possible to delete the first without touching a training run.
WORK_ROOT = "data/analysis/reasoning_arms"
DATA_ROOT = "data/training_data/reasoning_arms"


@dataclass(frozen=True)
class Recipe:
    """Everything that fixes WHICH tokens each arm sees, minus the span text."""

    # decision rounds per split, PER CORPUS (x2 corpora for the totals)
    train_rounds: int = 50_000
    val_rounds: int = 1_000
    test_rounds: int = 4_000

    # scaffold-disjoint split, by Murcko scaffold of the seed molecule
    split_seed: int = 20260828
    val_frac: float = 0.02
    test_frac: float = 0.04

    # balance the committed candidate's index by quota. The tool's own candidate order
    # is never touched.
    balance_index: bool = True
    n_positions: int = 4

    # a record is kept only if EVERY one of its decision rounds has exactly this many
    # candidates — the rule-selection checkpoint is built for 4, and a mixed corpus
    # would make the index quota meaningless.
    # 0 IS A SENTINEL for "any count 1..max_candidates". Use it only with
    # `balance_index=False`: the quota counts committed indices, and a round with two
    # candidates cannot contribute to index #3 or #4, so a mixed corpus really would
    # make the quota mean something different per round. The main-model recipes below
    # set both together for that reason.
    require_candidates: int = 4
    min_rounds_per_record: int = 1
    corpora: tuple = ("scaffold", "fg")

    # the span generator; part of the recipe because a different model is a different
    # corpus even at the same prompt
    gen_model: str = "Qwen/Qwen3.6-27B"
    gen_temperature: float = 0.7

    notes: str = "natural suggest_edits order; index balanced by selection"

    def digest(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    @property
    def version(self) -> str:
        return f"v{self.digest()}"

    def to_json(self) -> str:
        return json.dumps({**asdict(self), "digest": self.digest()},
                          indent=1, default=str)


def _h(*parts) -> int:
    return int(hashlib.sha256("::".join(map(str, parts)).encode()).hexdigest()[:16], 16)


def split_of(scaffold_key: str, rec: Recipe) -> str:
    """train / val / test for one scaffold, stable under any scan order."""
    x = (_h(rec.split_seed, scaffold_key) % 10_000) / 10_000.0
    if x < rec.test_frac:
        return "test"
    if x < rec.test_frac + rec.val_frac:
        return "val"
    return "train"


def quota(budget_rounds: int, rec: Recipe) -> list:
    """Rounds allowed per committed index, so the split comes out flat."""
    n = rec.n_positions
    if not rec.balance_index:
        return [budget_rounds] * n
    per = -(-budget_rounds // n)
    return [per] * n


def fits(counts: list, remaining: list) -> bool:
    """Can this record's per-index round counts be taken without over-filling a bucket?

    A record is accepted whole or not at all. Dropping one round out of the middle of a
    trajectory would break the conversation, and training is on records — so the balance
    is achieved by choosing records, not by trimming them.
    """
    return all(c <= r for c, r in zip(counts, remaining))


def work_dir(rec: Recipe) -> str:
    return f"{WORK_ROOT}/{rec.version}"


def data_dir(rec: Recipe, arm: str) -> str:
    return f"{DATA_ROOT}/{rec.version}/{arm}"


# --------------------------------------------------------------------------- #
# THE FULL-CORPUS RECIPES. The study recipe above takes 50,000 rounds per corpus with the
# committed index balanced by quota; these take everything, per corpus, for SFT training
# rather than for an A/B.
#
# TWO THINGS CHANGE AND BOTH ARE DELIBERATE:
#
#   balance_index=False   The quota is what stops "always #1" from being a policy: the
#                         commit sits at index 1 in 48% of scaffold rounds and 42% of fg
#                         rounds, and 34.5% of the records the study scanned were turned
#                         away by it. Keeping it would cost ~36% of the corpus (3.6M
#                         rounds -> 2.3M), because the budget divides evenly and the
#                         rarest index (#4, 14-17%) fills last. A training corpus wants
#                         the data; what it buys with the data is a 0.45 index floor
#                         instead of 0.25, and any number measured on it has to say so.
#
#   corpora=("x",)        One corpus each, so `work_dir` and `data_dir` come out separate
#                         from the start -- no post-hoc glob over
#                         `toolchains_arms_<corpus>-*.jsonl`, and fg (a quarter the size)
#                         finishes first and can start training while scaffold runs.
#
# The round budgets are CEILINGS, not targets: with `balance_index=False` the quota is
# `[budget] * 4` and every record fits, so any number above the corpus size takes the
# whole corpus. They are written large and left alone.
FULL_SCAFFOLD = Recipe(train_rounds=8_000_000, val_rounds=400_000,
                       test_rounds=800_000, balance_index=False,
                       corpora=("scaffold",),
                       notes="full scaffold corpus, natural index distribution")
FULL_FG = Recipe(train_rounds=8_000_000, val_rounds=400_000,
                 test_rounds=800_000, balance_index=False, corpora=("fg",),
                 notes="full fg corpus, natural index distribution")

# The MAIN-MODEL recipes: the full corpus, natural index distribution, and every round
# KEPT REGARDLESS OF CANDIDATE COUNT.
#
# `require_candidates=0` is a sentinel for "any count the checkpoint can encode", i.e.
# 1..FeatureSpec.max_candidates, rather than exactly one number. Nothing downstream
# needed changing to support it: `FeatureSpec.encode` already takes
# `n = min(len(cands), self.max_candidates)` and carries a per-candidate `cmask`, and
# `bucket_span.render` was already generic -- verified by rendering the same round at
# n_cand 4, 3, 2 and 1, which gives a 7/6/5/4-line span and the right `Taking Edit N`.
# The only gate that had to move is `_frame`'s two-elimination requirement, which the
# nodrop arms discard anyway (see NAT_ALLOW_SHORT in nat_span).
#
# It costs a digest, so these are NEW names rather than edits to FULL_*: `full_fg`
# stays v46b1f809a01e and the 79 GB already prepared under it stays valid.
#
# Measured on the corpora: rounds with a candidate count other than 4 are 1.17% of
# scaffold rounds and 0.36% of fg, but a record is taken whole or not at all, so the
# exact-4 rule costs 1.63% / 0.99% of RECORDS -- about 62,000 trajectories.
MAIN_SCAFFOLD = Recipe(train_rounds=8_000_000, val_rounds=400_000,
                       test_rounds=800_000, balance_index=False,
                       require_candidates=0, corpora=("scaffold",),
                       notes="main model: full scaffold, any candidate count")
MAIN_FG = Recipe(train_rounds=8_000_000, val_rounds=400_000,
                 test_rounds=800_000, balance_index=False,
                 require_candidates=0, corpora=("fg",),
                 notes="main model: full fg, any candidate count")

# THE RECOVERED RECIPES. `main_*` over the beam-recovered corpora and nothing else
# changed: same budgets, same split seed, same `require_candidates=0`, same natural index
# distribution -- only `corpora` moves, which is exactly what has to move for the digest
# to give these their own work and data directories.
#
# The split seed is deliberately UNCHANGED. `split_of` hashes the record's Murcko
# scaffold, so a scaffold that is in test under `main_scaffold` is in test here too: the
# recovered trajectories are re-runs of the SAME instances, and a recovered record whose
# scaffold sits in the base corpus's training set must not turn up in a recovered test
# split. Keeping the seed is what makes the two corpora mixable.
REC_SCAFFOLD = Recipe(train_rounds=8_000_000, val_rounds=400_000,
                      test_rounds=800_000, balance_index=False,
                      require_candidates=0, corpora=("scaffold_rec",),
                      notes="beam-recovered scaffold, any candidate count")
REC_FG = Recipe(train_rounds=8_000_000, val_rounds=400_000,
                test_rounds=800_000, balance_index=False,
                require_candidates=0, corpora=("fg_rec",),
                notes="beam-recovered fg, any candidate count")

# THE BASE CORPORA, REBUILT FOR THE n72 SET. `main_*` minus the failed seed stubs, plus
# the satisfied-at-seed records `min_rounds_per_record=1` was dropping along with them.
# Everything else is `main_*`: same budgets, same split seed, same natural index.
BASE_SCAFFOLD = Recipe(train_rounds=8_000_000, val_rounds=400_000,
                       test_rounds=800_000, balance_index=False,
                       require_candidates=0, min_rounds_per_record=0,
                       corpora=("scaffold_keep",),
                       notes="base scaffold: recovered stubs dropped, 0-edit kept")
BASE_FG = Recipe(train_rounds=8_000_000, val_rounds=400_000,
                 test_rounds=800_000, balance_index=False,
                 require_candidates=0, min_rounds_per_record=0,
                 corpora=("fg_keep",),
                 notes="base fg: recovered stubs dropped, 0-edit kept")

STUDY = Recipe()
NAMED = {"study": STUDY, "full_scaffold": FULL_SCAFFOLD, "full_fg": FULL_FG,
         "main_scaffold": MAIN_SCAFFOLD, "main_fg": MAIN_FG,
         "rec_scaffold": REC_SCAFFOLD, "rec_fg": REC_FG,
         "base_scaffold": BASE_SCAFFOLD, "base_fg": BASE_FG}

# Every script in this package reads `rp.DEFAULT`, and the recipe decides the work and
# data directories. Selecting it by environment rather than by a flag on nine scripts
# keeps one source of truth and makes a mixed run impossible: `ARMS_RECIPE` is set once,
# for the whole pipeline, and a shell that forgets it gets the study corpus rather than a
# silent half-and-half.
_SEL = _os.environ.get("ARMS_RECIPE", "study")
if _SEL not in NAMED:
    raise SystemExit(f"ARMS_RECIPE={_SEL!r} is not one of {sorted(NAMED)}")
DEFAULT = NAMED[_SEL]

if __name__ == "__main__":
    print(DEFAULT.to_json())
    print("\nversion :", DEFAULT.version)
    print("work    :", work_dir(DEFAULT))
    print("data    :", data_dir(DEFAULT, "<arm>"))
    from collections import Counter
    s = Counter(split_of(f"scaf{i}", DEFAULT) for i in range(40_000))
    print("split shares:", {k: f"{100*v/40_000:.1f}%" for k, v in s.items()})
    print("train quota per index:", quota(DEFAULT.train_rounds, DEFAULT))
