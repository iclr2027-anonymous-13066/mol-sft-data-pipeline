# -*- coding: utf-8 -*-
"""The GENERATION PROMPTS for the natural-language v34 family (n1, n2, n3, n4, n6).

    PYTHONPATH=. python 6_edit_reasoning/nat_span.py --arm n1 --splits val
    PYTHONPATH=. python 6_edit_reasoning/generate.py --arm model_n1 \
        --evidence nat_prompt/n1 --splits val

THE FIRST CUT OF THIS FAMILY REPLACED THE TEMPLATE AND LOST. n1..n4 were written as
continuous prose that said in words what `Dropping Edit 4; that leaves ...` says in a
line, and eight epochs of n1 (exact .357/.368/.375 over three seeds) landed under
noreason's two (.347) and under ONE epoch of the v34 template it was modelled on (.375).
n1 and n2 -- the selected six columns against six random ones -- came out identical on
every seed. The prose did not carry the signal; it deleted the skeleton that did.

So the skeleton is now literal and the prose goes in the GAPS. The five v34 beats are
rendered by `bucket_span` exactly as `render_bucket.py` renders them for v34, and three
passages are generated into the three joints:

    Out of range now: ...                                   NEEDS    (rendered)
    Each edit, at one sigma of its predicted shift: ...      BAND     (rendered)
    <<<1>>>  why the first edit is about to go               (written)
    Dropping Edit 1; that leaves Edits 2, 3 and 4.           DROP1    (rendered)
    <<<2>>>  why the second goes, out of the three left      (written)
    Dropping Edit 4; that leaves Edits 2 and 3.              DROP2    (rendered)
    <<<3>>>  what decides the last two                       (written)
    Taking Edit 3: <from> -> <to>.                           COMMIT   (rendered)

Every rendered line is byte-identical to v34's, so v34 IS the control: n1 - v34 is the
inserted prose and nothing else, and `bucket_span.dropped_edit` reads the corpus and the
student's rollouts with the same regex it always did.

    n1   q-ordered drops, the six columns the six teacher criteria pick
    n2   the same with six columns drawn UNIFORMLY from the same live pool (the control
         for n1: same rounds, same count, only the criterion differs)
    n3   the same with ALL of pool_104 -- no selection at all
    n4   n1's observations with the two drops chosen at RANDOM (the control for the
         teacher signal in the ACTION space, as v34r is to v34)
    n6   n3's observations with no skeleton and no drops: reason freely to the commit.
         UNCHANGED by the rewrite -- it is the arm that asks what the skeleton is worth,
         so it has to stay the arm it was.

    n1a  n1 restricted to pool_A: the same six teacher criteria choosing out of the 79
         columns a student can COUNT off the two SMILES, instead of out of all 104.
    n2a  the matched random control for n1a, in the same pool.

WHY n1a/n2a EXIST. n1 and n2 came out identical -- exact .4017 against .4020 over three
seeds, against a seed noise of 0.012 -- and the diagnostics say why. The student names a
column on 4-11% of joints, and when it does the KIND is one of that round's six 45% of
the time for n1 against 38% for n2 (chance is 6/19 = 32%), so the teacher signal does
reach the text; there is just too little of it to move a corpus mean. And 65% of what the
student names is a fitted score it recovers by having memorised the rule: 0.738 on a rule
seen in training, 0.222 on one that was not. pool_A takes those away, so an arm can only
argue from quantities the reader can reach.

WHAT THIS CANNOT FIX, and it should be said here. A GBM reading ALL 104 columns scores
0.347 on the four-way pick where the student scores 0.402, and it is right on only 0.291
of the rounds the student gets wrong (0.250 is chance, 0.350 is independence). The pool
is not a source of information the student lacks, so no selection out of it can buy more
than about two points. pool_A does not change that -- its ceiling is 0.313 against
pool_104's 0.311. What n1a/n2a test is whether reasoning the student can EXECUTE beats
reasoning it can only RECALL, which is a claim about faithfulness, not about the ceiling.

`noreason` (arm 5) is not generated; assemble.py writes it as an empty span.

WHAT IS NOT IN THE PROMPT. `q` never appears, in any arm. It decides WHICH two edits are
set aside and nothing else. v5 printed a quantity the student cannot compute and 98.3% of
its trained spans went on to invent one, so the teacher signal enters only as an
instruction ("you set these aside") and the writer has to find the reason in the block a
student can also read.

THE OBSERVATION TABLE IS COLUMN-MAJOR and covers all four edits, not the survivors. v36's
panel sat after the drop and spoke about the three left standing; here the block has to
support the FIRST elimination too, and n1 and n4 have to carry the same table so that
only the drop order differs between them. Columns are therefore selected over the whole
live field.
"""
from __future__ import annotations

import argparse
import glob
import importlib
import json
import os
import collections as _collections
import hashlib as _hashlib
import re as _re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

import bucket_span as bs                                            # noqa: E402
import render_bucket as rb                                          # noqa: E402
rp = importlib.import_module("6_edit_reasoning.recipe")          # noqa: E402

# KEEP ROUNDS THE STUDY WOULD HAVE DROPPED. Off by default, so every arm already on the
# curve renders byte-identical spans on byte-identical record sets. On, two study-only
# gates lift: the two-elimination requirement (meaningless below three candidates, and
# discarded by the nodrop arms regardless) and the empty-functional-group drop. Set it
# for a training corpus, never for an arm comparison.
_ALLOW_SHORT = os.environ.get("NAT_ALLOW_SHORT", "0") == "1"

ARMS = ("n1", "n2", "n3", "n4", "n6", "n1a", "n2a", "n7", "n7r",
        "n8", "n8r", "n9", "n9r", "n10", "n10c", "n11", "n15", "n16", "n17", "n18", "n19",
        "n20", "n21", "n22", "n23", "n24", "n25", "n26", "n27", "n28",
        "n29", "n30", "n35", "n40", "n41", "n42", "n43", "n44",
        "n40f", "n41f", "n42f", "n43f", "n44f", "n45f", "n46f", "n47f",
        "n48", "n49", "n50", "n51", "n52", "n53", "n54",
        "n55", "n56", "n57", "n58", "n59", "n60", "n62", "n61", "n63",
        "n64", "n65", "n66", "n67", "n68", "n69", "n70", "n71", "n72", "n73",
        "n74", "n75", "n76", "n77", "n78")
# The arms whose span is a rendered v34 skeleton with three written joints. n6 is
# not one of them: it has no skeleton, which is what it is for.
GAP_ARMS = ("n1", "n2", "n3", "n4", "n1a", "n2a", "n7", "n7r")
GAP_MARKS = ("<<<1>>>", "<<<2>>>", "<<<3>>>")
GAP_SEP = "###"
# ONE POOL FOR EVERY ARM: the 104 v35/v36 drew from, which is P0 minus the 28
# `r_dmean__`/`r_dstd__` predicted deltas. n1/n2/n4 select two-to-six columns out of it
# and n3/n6 print the whole thing, so "selected" against "not selected" is the only
# difference and neither side can win on a column the other never had. The first cut
# gave n3/n6 all 132 of P0 and it showed immediately in the spans -- they argued almost
# entirely from "the predicted change in BBBP" and "the spread of the predicted logP
# change", columns n1 cannot reach, which would have priced the pool rather than the
# selection.
POOL = "104"
# n1a/n2a repeat the n1/n2 contrast -- teacher-chosen columns against a matched random
# draw -- inside pool_A, the 79 the student can count off the two SMILES. The teacher
# criteria select for what predicts the ANSWER and are blind to whether the reader can
# reach the quantity, which is how 24% of n1's selection came to sit on fitted scores
# the student can only recall. See bucket_span.pool_A for the measurements.
POOL_OF = dict({a: POOL for a in ARMS}, n1a="A", n2a="A", n7="A", n7r="A",
               n9="A", n9r="A", n10="A", n10c="A", n11="A", n15="A", n16="A",
               n17="A", n18="A", n19="A", n20="A", n21="A", n22="A", n23="A", n24="A",
               n25="A", n26="A", n27="A", n28="A", n29="A", n30="A", n35="A",
               n40="A", n41="A", n42="A", n43="A", n44="A",
               n40f="A", n41f="A", n42f="A", n43f="A", n44f="A", n45f="A",
               n46f="A", n47f="A", n48="A", n49="A", n50="A")
# n7/n7r drop the SELECTION axis entirely and keep only the ORDER of elimination. Both
# arms get the SAME six columns on every round -- one draw, shared -- so the observation
# is a constant of the pair rather than a second variable, and the only thing that
# differs is whether the two edits set aside are q's weakest or a seeded pick. This is
# v34 - v34r with an argument attached to each elimination.
#
# And the argument carries NO NUMBERS. Every failure mode this family has had came
# through the value: invented quantities, a memorised (rule -> score) table that fell
# from 0.738 to 0.222 on a rule never seen, false magnitude claims on two values a
# hundredth apart. An ordinal claim -- the most, the only one, ahead of Edit 3 -- is
# still checkable against the block, which is what `false_ordinal` does, and it is
# something a reader can reproduce by comparing four candidates rather than by recalling
# a number it cannot compute.
ABS_ARMS = ("n7", "n7r")
N_ABS_COLS = 6
# n8/n8r are the same two orderings with NOTHING written between the beats -- the v34
# frame exactly as `render_bucket.py` renders it, q's weakest two against a pair that is
# never q's. No model is involved, so they render in minutes rather than generating in
# half an hour, and they are the control the prose arms are missing: n7 - n8 is what the
# passages add, on identical eliminations, and n8 - n8r is whether the ORDER carries
# anything at all once no argument is attached to it.
RENDER_ARMS = ("n8", "n8r", "n9", "n9r", "n10", "n10c", "n11", "n19", "n20", "n21", "n22", "n23", "n24", "n25",
               "n26", "n27", "n28", "n29", "n30", "n35",
               "n40", "n41", "n42", "n43", "n44",
               "n40f", "n41f", "n42f", "n43f", "n44f", "n45f", "n46f", "n47f",
               "n48", "n49", "n50", "n51", "n52", "n53", "n54",
               "n55", "n56", "n57", "n58", "n59", "n60", "n62", "n61", "n63",
               "n64", "n65", "n66", "n67", "n68", "n69", "n70", "n71", "n72", "n73",
               "n74", "n75", "n76", "n77", "n78")
# n10/n10c: THE TEACHER'S THREE COLUMN NAMES AND NOTHING ELSE -- no values. Diagnostic
# arms, not candidates: `rand_labels` measured -0.013 against the bare frame, so names
# without numbers are not expected to help, and n10's accuracy is not the point.
#
# WHAT THEY ARE FOR. n9 reaches tbl_own 0.487 at epoch 4 and is still climbing, but it
# must also produce twelve numbers, so the selection is a small share of its loss.
# Stripping the values puts the whole gradient on the choice and asks the question
# cleanly: IS THE TEACHER'S COLUMN SET LEARNABLE AT ALL?
#
# THE TWO POSITIONS ARE THE EXPERIMENT. `cell_m` is the drop in the COMMIT's margin when
# the COMMIT's own cell is masked -- it is a function of the answer, and at the point n9
# prints its table the model does not have one yet.
#   n10   names before the drops -- learnability where the table actually sits
#   n10c  names after the commit -- learnability once the answer is (approximately) known
# n10c also keeps everything up to COMMIT byte-identical to n8, so its `exact` is a
# control that should reproduce n8's and its tbl_own is uncontaminated by the damage a
# badly chosen table does to the drops.
NAME_ARMS = ("n10", "n10c")
NAME_HEAD = "What separates the edits:"
# n11: n9 EXACTLY, with the three column names announced on the heading line before any
# value is written. Everything else -- the columns, their order, the values, the drops,
# the commit -- is byte-identical to n9, so n11 - n9 is the announcement and nothing else.
#
# WHAT IT COULD DO, and it is one thing only. n9 emits `  label:  a | b | c | d` three
# times, so each label is chosen with ~40 value tokens standing between it and the last
# one. Announcing all three first makes the selection a single joint decision. That
# matters if -- and only if -- a PARTLY correct set is worth something: the model already
# reproduces 1.45 of the teacher's three and scores at the no-table baseline (free .377
# against band .374), which is what the k-of-3 dose-response was being run to settle.
#
# WHAT IT CANNOT DO. It hands the model no information it did not have. The three names
# still have to be produced, from a criterion (cell_m/col_m/cell_q) that needs the
# selection network and the answer, and tbl_own is 0.487 against 0.373 for a random draw.
N11_JOIN = ", "
# n15: THE JOINTS ARGUE FROM THE BAND, and from nothing the student cannot recount.
#
# Every prose arm so far argued from the observation table, and every one of them lost to
# the bare frame: n1a .410, n1 .402, n2 .402 against n8's .426. The reason was never the
# prose. It was the SOURCE -- cell_m/col_m/cell_q read q at the commit, so the student
# reproduces the columns on 0.497 of rounds and then argues from the wrong ones. Measured
# on the same 4-way scale the criteria are scored on:
#
#     the teacher's three observation columns   0.3411   needs the answer
#     THE BAND ALONE                            0.3438   needs nothing
#     one rule, "fewest misses"                 0.3168   needs nothing
#     the best commit-free criterion, any 3     0.2614   at chance
#     three columns at random                   0.2620
#
# The band carries the teacher's signal ENTIRELY, and the student already writes it --
# 42% byte-exact, 0.68 a row. What no arm has done is SAY the comparison: n8 prints four
# band rows and drops in silence, and the top band feature is `z_misses`, which is a
# comparison ACROSS the rows rather than anything in one of them.
#
# So the bet here is not that the band is informative -- forcing a correct band moved
# nothing (`band` - `free` = -0.004, p=0.21), which says the model already has it. The bet
# is that PERFORMING the comparison in words is worth something the information alone is
# not. No experiment has separated those two, and this is the one that does.
N15_KINDS = ("drop", "drop", "hold")
# n16 is n15 with v28's line in front of the band: EVERY functional group the molecule
# has, with its count, ordered by count then alphabetically. Nothing is selected, and
# that is the design -- v27 ranked these by A's gate on `r_dfg__`, a column whose masking
# moves the commit's q by 0.00131 against 0.00476-0.01123 for every other family, which
# is a ranking of noise; v28c ranked by what the COMMIT acts on and so named the answer
# in the first line. Printing all of them removes the question, and a cap would put it
# back: choosing 3 of a molecule's 12 groups needs an order and any order is a criterion.
#
# Both properties of the line are recomputable by the student with match_substructure, so
# it is the same kind of evidence the band is -- present, checkable, and owing nothing to
# the selection network. n16 - n15 is that one line.
N16_ARMS = ("n16",)
# n9/n9r put the EVIDENCE ITSELF in the span, and no prose at all: three observations
# with their four values, written between the band and the first elimination. Nothing is
# argued -- the student writes the table and then drops.
#
# This is the one form that escapes the trap every prose arm fell into. n1-n2 and
# n1a-n2a came out identical and the reason was never that the selection is worthless,
# it was that the student names an observation on 4-11% of joints, so a difference that
# lives in the selection reaches a twentieth of the corpus. Enumeration makes that 100%
# by construction, and the null it produces (or does not) finally means one thing.
#
# THREE, not six. Measured with a GBM over the four-way pick, criteria z-scored within
# the round: all six score 0.4026, and cell_m + col_m + cell_q alone score 0.3987 --
# 99.0% of it. `gate` and `gate_spread`, which were the whole of CRITERIA_V35, score
# 0.278 and 0.276 alone against a 0.250 floor and add 0.001 between them. Three also
# halves what the student has to copy (12 numbers, not 24) and drops the overlap with a
# random draw from 0.58 at six columns to 0.32 at three, because the live pool_A set is
# about twelve columns wide.
CRITERIA_V9 = ("cell_m", "col_m", "cell_q")
N_V9_COLS = 3
TABLE_HEAD = "What separates the edits:"
_ORDER_OF = {"n7": "q", "n8": "q", "n7r": "anti", "n8r": "anti"}


def anti_q_order(q, pick: int, n_cand: int, key: str):
    """A seeded (drop1, drop2) that is never the pair q would have chosen.

    `drop_order_random` draws uniformly over the orderings, so with three rivals it
    reproduces q's own first two on one round in six -- 15.3% measured -- and those
    rounds carry no contrast at all. Excluding that one pair makes every round of the
    control a round where the two arms eliminate differently.

    It does change what the control IS, and the change should be stated rather than
    buried: n7r is no longer "q against a uniform draw" but "q against anything but q".
    That is the sharper test of whether the ORDER carries the teacher signal, and it is
    a slightly pessimistic control -- it can never accidentally agree with the teacher.
    """
    riv = [i for i in range(n_cand) if i != pick]
    if len(riv) < 2:
        return None
    qp = bs.drop_order(q, pick)[:2] if q else []
    qpair = tuple(qp) if len(qp) == 2 else None
    pairs = [(a, b) for a in riv for b in riv if a != b and (a, b) != qpair]
    if not pairs:
        return None
    h = _hashlib.blake2b(f"nat-antiq:{key}".encode(), digest_size=8).digest()
    return list(pairs[int.from_bytes(h, "big") % len(pairs)])


# --------------------------------------------------------------------------- #
SKELETON = """You are a molecular-design AI assistant working through a task. Below is
the conversation so far and the tool call you are about to make. Write the assistant's
turn that goes between them -- the reasoning that leads from the tool output you just
received to the call you are about to make.

## The task you were given
{user_prompt}

## The molecule you are working on
{mol_smiles}

{evidence}
## The tool call you are about to make
Edit {pick_no} -- edit_fragment(from_smiles={from_smiles}, to_smiles={to_smiles}, anchors={anchors})

{closer}"""


# --------------------------------------------------------------------------- #
# THE GAP FORM (n1..n4). The turn is shown to the writer with its three joints marked,
# so it is completing an argument it can see rather than composing one from a
# description. Three things follow from that and each is deliberate:
#
#  * THE RENDERED LINES ARE IN THE PROMPT, ONCE. They carry NEEDS and BAND, so the two
#    blocks the old prompt printed above the observation table are gone -- printing them
#    twice would only invite the writer to restate them.
#
#  * NOTHING IS LEAKED BY SHOWING THE FRAME. The drops and the commit were already in
#    the old prompt (as `## The two edits you set aside` and the tool call); the frame
#    replaces that block with the literal lines those decisions become.
#
#  * THE WRITER NEVER STATES THE ACT. The line after each passage does. What is asked
#    for is the ground, and a passage that announces "so I drop Edit 1" would make the
#    rendered line a restatement of itself -- which is exactly what the first cut of
#    this family did to the whole skeleton.
SKELETON_GAP = """You are a molecular-design AI assistant working through a task. Below
is the conversation so far, the assistant's turn that follows it, and the tool call that
turn ends in. Three passages are missing from the turn and you are writing them.

## The task you were given
{user_prompt}

## The molecule you are working on
{mol_smiles}

{evidence}
## The turn, with the three passages marked
{frame}

## The tool call the turn ends in
Edit {pick_no} -- edit_fragment(from_smiles={from_smiles}, to_smiles={to_smiles}, anchors={anchors})

{closer}"""


# Identical for all four gap arms, word for word. n1, n2 and n3 differ only in which
# columns "{obs_head}" holds and n4 only in which two edits the rendered lines drop, so
# any difference in the instruction would confound the thing being measured.
#
# The permission clause is what makes n4 possible at all: its drops are seeded, not
# q-ordered, so on a fifth of rounds the block does not condemn the edit that is about to
# go. Told to persuade, a writer invents; told to name what the edit is behind on and
# move on, it reports. Prohibition has backfired twice in this codebase -- "claim nothing
# the numbers do not show" RAISED false claims 0.33 -> 0.38 per span -- so the clause
# grants rather than forbids.
CLOSER_GAP = """Write the three missing passages, marked <<<1>>>, <<<2>>> and <<<3>>>,
and nothing else. The lines around them are already written; do not repeat them and do
not rewrite them.

<<<1>>> stands between the four edits and the line that drops {drop1_name}. Say what in
"{obs_head}" puts {drop1_name} behind the other three.
<<<2>>> stands after {drop1_name} is gone. Weigh only the three still standing, and say
what puts {drop2_name} behind the other two.
<<<3>>> stands where it is down to {pair_text}. Say what decides it for {commit_name}.

Each passage has to rest on a value read out of "{obs_head}", and every one of the three
has to quote at least one such value. That block is the only place a reason can come
from: the lines already written say WHAT happens, and you are writing WHY, and what
holds or is at risk is already said above you. Name the observation and quote its value; do not gesture at an impression,
and do not use a number that is not printed above. Write every number in digits, exactly
as it is printed there -- "+1", "-0.296", "+133" -- and never spell one out in words. A
column whose four values are the same cannot separate anything, so do not build a
sentence on one, and do not lean on the same observation in two passages. If two values
are level, say they are level -- never call one higher or lower than the other.

Do not announce the elimination and do not announce the commitment -- the line after
each passage does that, and a passage that says "so I drop Edit 3" makes it say the same
thing twice. Give the ground it stands on, and stop there.

Where nothing in the block clearly condemns the edit that is about to go, say the one
thing it IS behind on, or the trade-off you accept in letting it go, and move on. Do not
go looking for something that overturns the order, and do not re-rank the edits: which
two go, and which one is taken, are given to you.

Write in the first person, as the model working this out -- one sentence per passage,
about 25 words and never more than 35, plain prose -- placing four edits takes a few
more words than naming one. Two values are enough to make a
comparison; do not walk through all four. No bullet points, no numbered lists, no headings, no
markdown, no backticks and no tool-call syntax. Do not think out loud, do not enumerate
values to yourself and do not correct yourself mid-sentence. Never ask yourself a
question, never mention these instructions or "the prompt", and never write about the
blocks as blocks -- you are looking at a molecule, not at a document.

Refer to each edit as "Edit 1", "Edit 2", "Edit 3", "Edit 4" -- the numbering in the
lines above. You may name the chemistry alongside it ("Edit 2, the added amide"), but
the number has to be there every time so the reader knows which edit you mean.

Output the three passages in order, separated by a line containing only ###. Nothing
before the first, nothing after the third, and no marker of your own."""


# n6 has the same three information blocks and no skeleton: no drop order is given, no
# beats are named, and the writer reaches the commit however it likes. n3 minus n6 is
# therefore the skeleton plus the two teacher-chosen eliminations, which is the only
# clean reading available once the drops are part of what the skeleton IS.
CLOSER_FREE = """Write that turn as continuous prose in the first person -- the voice of
the model working this out, not a report about a decision already made. No bullet points,
no numbered lists, no headings, no markdown, no backticks and no tool-call syntax. Do not
think out loud, do not enumerate values to yourself and do not correct yourself
mid-sentence. Never ask yourself a question in the text, never mention these
instructions or "the prompt", and never write about the blocks as blocks -- you
are looking at a molecule, not at a document.

Reason from what you are looking at to the edit you are about to make: what this round
needs, and what in the blocks above makes {commit_name} the edit to make rather than one
of the other three. Put the reason before the commitment.

Every claim has to rest on a value printed above and has to tell the edits apart. Name
the observation and quote its value; do not gesture at an impression, and do not use a
number that is not printed. Write every number in digits, exactly as it is printed there
-- "+1", "-0.296", "+133" -- and never spell one out in words. A column whose four
values are the same cannot separate anything, so do not build a sentence on one. Where
the blocks do not favour the edit you are making, that is the round you are writing, not
a mistake to fix: name the one point it does win on, or the trade-off you accept in
taking it, and let that be the sentence that commits. Do not re-rank the edits -- the
edit you take is given to you.

Refer to each edit as "Edit 1", "Edit 2", "Edit 3", "Edit 4" -- the numbering in the
blocks above. You may name the chemistry alongside it ("Edit 2, the added amide"), but
the number has to be there every time so the reader knows which edit you mean.

About 130 words in the whole turn. It is a working note, not an essay: give the reason
and move on.

Your last sentence must be exactly this, and nothing may follow it:
Taking Edit {pick_no}: {commit_pair}."""


OBS_HEAD = "What separates the four edits"

# --------------------------------------------------------- reading a span back
# The template family is parsed by `bucket_span.dropped_edit`, a regex over the literal
# `Dropping Edit 4; that leaves ...` line. Prose has no such line, and the diagnostic
# that explained v30 and v36 -- did the STUDENT set aside the same edits the corpus did
# -- is the whole reason to keep parsing. So the elimination is read grammatically
# instead, and the same function reads the corpus and the student's rollouts.
# "so I set IT aside" and "I set aside Edit 4" are the same act, so the particle verbs
# tolerate up to two words between the verb and its particle.
_PART = r"(?:\w+\s+){0,2}"
_VERB = (r"(?:set\w*\s+" + _PART + r"aside|put\w*\s+" + _PART + r"aside|"
         r"leav\w*\s+" + _PART + r"out|rul\w*\s+" + _PART + r"out|"
         r"discard\w*|drop(?:s|ped|ping)?|eliminat\w+|reject\w*)")
_V_RE = _re.compile(_VERB, _re.I)
_E_RE = _re.compile(r"\bEdit (\d+)")
_SENT_RE = _re.compile(r"(?<=[.!?])\s+")
_TAKE_RE = _re.compile(
    r"^(?:Taking Edit (\d+):|.*?\bI take Edit (\d+):)", _re.M)


def parse_drops(text: str, limit: int = 2):
    """The edits a span says it set aside, in order, 0-based.

    Both voices occur and they put the edit on opposite sides of the verb -- "I set aside
    Edit 4" and "Edit 4 ... so I set it aside" -- and a sentence that eliminates one edit
    routinely NAMES another as the thing it is behind ("set aside Edit 4 because its +2
    is under Edit 2's +3"). Reading every `Edit N` in an elimination sentence would score
    that as two drops. So: the subject wins when it sits right before the verb (the
    passive "Edit 1 and Edit 3 are discarded", which is also how a pair gets eliminated
    in one sentence), otherwise the first edit named after the verb, otherwise the last
    one named before it.
    """
    out = []
    for sent in _SENT_RE.split(text.strip()):
        if _TAKE_RE.match(sent.strip()):
            continue
        v = _V_RE.search(sent)
        if not v:
            continue
        pre = [(m.start(), int(m.group(1)) - 1) for m in _E_RE.finditer(sent)
               if m.end() <= v.start()]
        post = [int(m.group(1)) - 1 for m in _E_RE.finditer(sent)
                if m.start() >= v.end()]
        near = [x for x in pre if v.start() - x[0] <= 25]
        if near:
            take = [x[1] for x in near]
        elif post:
            take = post[:1]
        elif pre:
            take = [pre[-1][1]]
        else:
            continue
        for i in take:
            if i not in out:
                out.append(i)
        if limit and len(out) >= limit:
            break
    return out[:limit] if limit else out


def parse_commit(text: str):
    """The edit a span commits to, 0-based, or None."""
    m = None
    for m in _TAKE_RE.finditer(text):
        pass
    return int(m.group(1) or m.group(2)) - 1 if m else None


_TAKE = _re.compile(r"(?<!\n)(?<!^)(\s*)(Taking Edit \d+:)")


def normalize_span(text: str) -> str:
    """Put the commit sentence on its own line, and nothing else.

    The closer asks for it there and the writer complies about half the time -- the rest
    of the time it ends the paragraph with it, which the shape gate accepts because it IS
    the last sentence. Re-rolling over a line break would spend generation on a
    difference that carries no meaning; moving it is deterministic and arm-symmetric.

    The v34 template the whole family is modelled on puts `Taking Edit 1: ...` on its own
    last line, and the diagnostics that explained v30 and v36 read the corpus and the
    student's rollouts with the same parser. This keeps one parser working on both.
    """
    t = "\n".join(x.rstrip() for x in text.strip().split("\n"))
    t = _re.sub(r"\n{3,}", "\n\n", t)
    m = None
    for m in _TAKE.finditer(t):
        pass
    if m is not None:
        t = t[:m.start()] + "\n" + t[m.start(2):]
    return t



# ------------------------------------------------------- reading the gap form back
_SEPLINE = _re.compile(r"(?m)^[ \t]*#{3,}[ \t]*$")
_MARKER = _re.compile(r"^\s*(?:<{2,}\s*\d\s*>{2,}|\(?\d[.):]|passage\s+\d[:.]?)\s*",
                      _re.I)
_RENDERED = _re.compile(r"Dropping Edit \d+;|Taking Edit \d+:|Out of range now:")


def parse_gaps(text: str, n: int = 3):
    """The three written passages, or None if the answer is not three passages.

    Rejecting rather than repairing, because a writer that did not produce three
    passages did not do the task -- it summarised the turn, or it ran past max_tokens
    mid-argument -- and the retry costs one generation while a repaired half-answer
    costs a training record. The two things that ARE repaired are pure formatting: a
    leading "<<<2>>>" or "2." that the writer echoed back, and the newlines inside a
    passage, which have to go because the assembled span is one line per beat.
    """
    if not text:
        return None
    parts = [x.strip() for x in _SEPLINE.split(text.strip())]
    parts = [x for x in parts if x]
    if len(parts) != n:
        return None
    out = []
    for x in parts:
        x = _MARKER.sub("", x)
        x = " ".join(x.split())
        # A passage that writes a rendered line has restated the skeleton instead of
        # arguing into it, and the assembled span would then say the same thing twice.
        if not x or _RENDERED.search(x) or GAP_SEP in x or "<<<" in x:
            return None
        out.append(x)
    return out


def split_span(span: str, frame: str):
    """The three written passages back out of an assembled span, or None.

    The inverse of `fill_frame`, and the only way to audit the gap arms: what the writer
    produced is not what lands on disk, so shape, invented numbers and off-block columns
    have to be measured on the passages alone -- the rendered lines around them carry
    digits of their own (edit numbers, SMILES) and would dilute every rate.
    """
    fl = [x for x in frame.split("\n") if x not in GAP_MARKS]
    out, i = [], 0
    for line in span.split("\n"):
        if i < len(fl) and line == fl[i]:
            i += 1
        else:
            out.append(line)
    return out if (i == len(fl) and len(out) == len(GAP_MARKS)) else None


# A magnitude claim about two printed numbers, and whether it is true. Deliberately
# narrow: only "<num> ... higher/lower than ... <num>", with the comparative between the
# two and each number the nearest one on its side. Evaluative comparatives -- "less
# favorable", "more compact" -- are not claims about the numbers and are left alone; a
# first cut that counted them reported a 25-39% error rate that was almost entirely the
# detector. What is left is unambiguous and is wrong on 9-33% of the claims that make it,
# most often on two values a hundredth apart or on two that are equal. It is a small
# share of the corpus and it is the WORST share: a span that says +0.112 is lower than
# -0.595 teaches the student that the direction of a comparison does not matter, on the
# one axis this whole family is about.
_CMP = _re.compile(r"\b(higher|greater|larger|bigger|more|above|smaller|lower|less|"
                   r"fewer|below)\s+than\b", _re.I)
_UP = {"higher", "greater", "larger", "bigger", "more", "above"}
_NUM = _re.compile(r"[-+]?\d+(?:\.\d+)?")
_EDIT_N = _re.compile(r"\bEdit \d+")
# the writer arguing with itself in the output -- what the free-form arms did at length
_RUNAWAY = _re.compile(r"\b(wait|hmm)\b|\bis not true\b|\bActually,", _re.I)


def false_compare(text: str) -> bool:
    """True if the text says one printed number is above or below another, wrongly."""
    y = _EDIT_N.sub("  ", text)
    for m in _CMP.finditer(y):
        pre = [z for z in _NUM.finditer(y) if z.end() <= m.start()]
        post = [z for z in _NUM.finditer(y) if z.start() >= m.end()]
        if not pre or not post:
            continue
        a, b = float(pre[-1].group()), float(post[0].group())
        if a == b or (a > b) != (m.group(1).lower() in _UP):
            return True
    return False


def fill_frame(frame: str, parts) -> str:
    """The rendered v34 span with the three passages in its joints."""
    for m, x in zip(GAP_MARKS, parts):
        frame = frame.replace(m, x)
    return None if any(m in frame for m in GAP_MARKS) else frame


# The abstract closer (n7/n7r). Same five beats, same three joints, same instruction
# everywhere it can be the same -- what changes is that the passage may not carry a
# number. It says WHERE an edit stands on an observation, not what the observation reads.
#
# Why the numbers go. Every measured failure in this family came through the value:
# spans that invented a quantity, a student recovering fitted scores at 0.738 on a rule
# it had seen and 0.222 on one it had not, and false magnitude claims on 9-33% of the
# explicit comparisons. An ordinal claim survives all three -- it is checkable against
# the block (see `false_ordinal`), it cannot be recalled from a rule the way a score can,
# and a reader reproduces it by ranking four candidates rather than by computing a value
# it has no way to compute.
CLOSER_ABS = """Write the three missing passages, marked <<<1>>>, <<<2>>> and <<<3>>>,
and nothing else. The lines around them are already written; do not repeat them and do
not rewrite them.

<<<1>>> stands between the four edits and the line that drops {drop1_name}. Say what in
"{obs_head}" puts {drop1_name} behind the other three.
<<<2>>> stands after {drop1_name} is gone. Weigh only the three still standing, and say
what puts {drop2_name} behind the other two.
<<<3>>> stands where it is down to {pair_text}. Say what decides it for {commit_name}.

Each passage has to rest on one observation out of "{obs_head}". Name the observation
and say WHERE the edit stands on it against the others -- it adds the most, it adds the
least, it is the only one that adds any, it is behind {commit_name} on that count, the
two are level. That block is the only place a reason can come from: the lines already
written say WHAT happens and you are writing WHY, and what holds or is at risk is
already said above you.

PLACE EVERY EDIT, NOT ONLY THE ONE THIS PASSAGE IS ABOUT. Open with the observation and
put each of the edits still standing somewhere on it:

  On heavy atoms removed, Edit 1, Edit 2 and Edit 3 all take off the same, and Edit 4
  alone takes off far less.

Not "Edit 4 removes the fewest heavy atoms" -- a sentence that opens with an edit has
already picked it and is only explaining itself afterwards -- and not "Edit 4 takes off
less than the others", which never compared the other three at all. No passage may begin
with the word "Edit", and every edit still standing has to appear in it.

WRITE NO NUMBERS AT ALL, and counting words are numbers. Say where an edit sits in one
of these ways and in no other:

  it adds none                        (never "adds zero")
  it is the only one that adds any    (never "adds one")
  it adds the most / it adds the least
  it adds more than Edit 2 / it adds less than Edit 2
  Edit 1 and Edit 2 are level

If you were about to write "Edit 2 adds one and Edit 3 adds two", write "Edit 3 adds
more than Edit 2" instead. Never write one, two, three or four as a quantity of what an
edit adds, removes or changes. The only digits allowed in your answer are the ones in
"Edit 1", "Edit 2", "Edit 3" and "Edit 4". A column whose four values are the same
cannot separate anything, so do not build a sentence on one, and do not lean on the same
observation in two passages.

TIES ARE THE COMMON CASE, so choose around them. Put an edit at the top or the bottom of
an observation only when it is there ALONE: on values of 6, 5, 5 and 5 the edit at 5 is
not "the lowest", it is level with two others. If the edit you are placing ties with
another on the observation you were going to use, either say the two are level and name
what else separates them, or use a different observation where it does stand alone.
Never put one edit ahead of another when the block has them equal.

Do not announce the elimination and do not announce the commitment -- the line after
each passage does that, and a passage that says "so I drop Edit 3" makes it say the same
thing twice. Give the ground it stands on, and stop there.

Where nothing in the block clearly condemns the edit that is about to go, say the one
thing it IS behind on, or the trade-off you accept in letting it go, and move on. Do not
go looking for something that overturns the order, and do not re-rank the edits: which
two go, and which one is taken, are given to you.

Write in the first person, as the model working this out -- one sentence per passage,
about 18 words and never more than 25, plain prose. No bullet points, no numbered lists,
no headings, no markdown, no backticks and no tool-call syntax. Do not think out loud,
do not enumerate the edits to yourself and do not correct yourself mid-sentence. Never
ask yourself a question, never mention these instructions or "the prompt", and never
write about the blocks as blocks -- you are looking at a molecule, not at a document.

Refer to each edit as "Edit 1", "Edit 2", "Edit 3", "Edit 4" -- the numbering in the
lines above. You may name the chemistry alongside it ("Edit 2, the added amide"), but
the number has to be there every time so the reader knows which edit you mean.

Output the three passages in order, separated by a line containing only ###. Nothing
before the first, nothing after the third, and no marker of your own."""


# --------------------------------------------------------- the DRAFT (n7/n7r)
# What a passage says is not a judgement call: pick one observation, put the live edits
# in its order, and the sentence is written. So it IS written -- here, deterministically
# -- and the model is left with the one thing it is better at, which is making it read
# like prose.
#
# What that buys, measured on the version where the model wrote the passage itself:
#   correctness      every gate passes by construction. The writer's own attempts were
#                    false on 11-17% of ordinal claims, wrote a numeral on 45%, and left
#                    an edit unplaced on 13%; one attempt in three survived all of it.
#   throughput       the prompt drops from 6.6 KB to under 1 KB -- no task, no molecule,
#                    no frame, no rules -- and this pipeline is prefill-bound.
#   the arms         the column is chosen to support THAT arm's own drop, so each argues
#                    as well as its ordering allows. How often a supporting column exists
#                    at all is then a corpus statistic, and it is the teacher signal
#                    itself: see `draft_stats`.
def _fmt_edits(idxs) -> str:
    n = [f"Edit {i + 1}" for i in sorted(idxs)]
    if len(n) == 1:
        return n[0]
    return ", ".join(n[:-1]) + " and " + n[-1]


def _v(idxs, sing: str, plur: str) -> str:
    return sing if len(idxs) == 1 else plur


def _groups(vals, live):
    """[(value, [edits])] over the live set, highest value first."""
    g = {}
    for i in live:
        g.setdefault(vals[i], []).append(i)
    return sorted(g.items(), key=lambda kv: -kv[0])


def _pick_col(own, live, target, used, key: str):
    """The observation this joint argues from: one that separates the live set, and by
    preference one where `target` stands alone at an end. Ties are broken on a hash of
    the round so the corpus does not lean on whichever column sorts first."""
    best = []
    for lb, raw in own.items():
        if lb in used:
            continue
        v = _nums_of(raw)
        if v is None or len(v) < max(live) + 1:
            continue
        vv = [v[i] for i in live]
        if len(set(vv)) < 2:
            continue
        alone = ((v[target] == max(vv) or v[target] == min(vv))
                 and vv.count(v[target]) == 1)
        best.append((0 if alone else 1, lb, v))
    if not best:
        return None, None
    rank = min(b[0] for b in best)
    same = sorted(b for b in best if b[0] == rank)
    h = int.from_bytes(_hashlib.blake2b(key.encode(), digest_size=8).digest(), "big")
    _, lb, v = same[h % len(same)]
    return lb, v


def draft_passage(own, live, target, used, key: str):
    """(sentence, label) for one joint -- true by construction, dull by design."""
    lb, v = _pick_col(own, live, target, used, key)
    if lb is None:
        return (f"On every observation here, {_fmt_edits(live)} are level, so this comes "
                f"down to what holds and what is at risk above."), None
    gs = _groups(v, live)
    (hv, hi), (lv_, lo) = gs[0], gs[-1]
    if len(gs) == 2 and lv_ == 0 and len(hi) == 1:
        body = (f"{_fmt_edits(hi)} is the only one that adds any and "
                f"{_fmt_edits(lo)} {_v(lo, 'adds', 'add')} none")
    elif len(gs) == 2:
        low = "none" if lv_ == 0 else "the least"
        body = (f"{_fmt_edits(hi)} {_v(hi, 'adds', 'add')} the most and "
                f"{_fmt_edits(lo)} {_v(lo, 'adds', 'add')} {low}")
    else:
        mid = [i for _, g in gs[1:-1] for i in g]
        low = "none" if lv_ == 0 else "the least"
        body = (f"{_fmt_edits(hi)} {_v(hi, 'adds', 'add')} the most, "
                f"{_fmt_edits(lo)} {_v(lo, 'adds', 'add')} {low}, and "
                f"{_fmt_edits(mid)} {_v(mid, 'sits', 'sit')} between them")
    return f"On {lb}, {body}.", lb


def _band_of(rnd, i):
    """(holds, at risk, within reach, misses) as sets, for one edit.

    The trailing `*` the band prints on a starred property is stripped: it is a mark on
    the ROW, and a sentence that says "the second gives up logD*" has copied a piece of
    table notation into prose.
    """
    h, r, w, m = bs.classify(rnd, rnd["candidates"][i])
    st = lambda xs: {str(x).rstrip("*") for x in xs}                # noqa: E731
    return st(h), st(r), st(w), st(m)


def _sole(rnd, live, target, unp, used, polarity="any"):
    """A property on which `target` is ALONE among the live edits.

    POLARITY IS NOT OPTIONAL. A drop joint that cites the one thing an edit is BEST at and
    then drops it teaches the reverse of the policy it is meant to teach, and the first
    version of this did exactly that on 18.4% of drop joints -- "Edit 1 is the only one
    bringing BBBP back within reach", and the next line dropped Edit 1. Worse, `misses`
    and `holds` both scored oor+1, so on an out-of-range property "alone in holding it"
    could outrank "alone in missing it". A drop takes "bad" and sees only the two
    condemning relations; the deliberative joint takes "any", because there it is
    describing two survivors rather than arguing one out.

    Preference order otherwise: out of range beats in range, and a clean miss beats a
    risk. `used` keeps the three joints off the same property.
    """
    B = {i: _band_of(rnd, i) for i in live}
    out = []
    for p in sorted({x for b in B.values() for s_ in b for x in s_}):
        if p in used:
            continue
        miss = {i for i in live if p in B[i][3]}
        risk = {i for i in live if p in B[i][1]}
        hold = {i for i in live if p in B[i][0]}
        reach = {i for i in live if p in B[i][2]}
        oor = 2 if p in unp else 0
        if miss == {target} and len(live) > 1:
            out.append((oor + 1, "misses", p, live - {target}))
        elif miss | risk == {target} and len(live) > 1:
            out.append((oor + 0, "risk", p, live - {target}))
        elif polarity == "any" and hold == {target} and len(live) > 1:
            out.append((oor + 1, "holds", p, live - {target}))
        elif polarity == "any" and reach == {target} and len(live) > 1:
            out.append((oor + 0, "reach", p, live - {target}))
    return max(out, default=None)


_REL = {
    # F: the rest are merely NOT missing it -- they may be at risk or within reach, and
    # "keep it in hand" claimed they hold it, which the band often did not say.
    "misses":  ("is the only one that misses it", "does", "do", "not"),
    "risk":    ("is the only one left exposed", "is", "are", "not"),
    "holds":   ("is the only one still holding it", "does", "do", "not"),
    "reach":   ("is the only one bringing it back within reach", "leaves", "leave",
                "it where it is"),
}
_SAYS = {"misses": "gives up {p}", "risk": "is exposed on {p}",
         "holds": "still holds {p}", "reach": "brings {p} back within reach"}


def _counts(rnd, live, unp):
    """{edit: dict of band counts}. BOTH scopes: over every property the band reports and
    over the out-of-range ones alone. The all-property counts are the ones that carry the
    signal -- `z_misses` and `z_holds` are the top two band features by permutation
    importance and the out-of-range versions are an order of magnitude below them."""
    out = {}
    for i in live:
        h, r, w, m = _band_of(rnd, i)
        out[i] = {"missed": len(m), "exposed": len(m) + len(r), "held": len(h),
                  "missed_oor": len(m & unp), "held_oor": len(h & unp)}
    return out


# (key, IS THE MAXIMUM THE BAD END, what the bad end reads as, what the good end reads as)
# The flag is the fix for the second contradiction: `held` and `held_oor` run the other
# way from the miss counts, and labelling the maximum "holds the least in range" made the
# sentence flatly false rather than merely unhelpful.
# (key, is the MAXIMUM the bad end, bad end singular, good end singular, bad end PLURAL)
# The plural is not decoration: the tie branch has a plural subject, and "Edit 3 and Edit
# 2 are the weakest -- both misses the most" is what came out without it.
_CNT = (("missed",     True,  "misses the most", "misses the fewest", "miss the most"),
        ("held",       False, "holds the least in range", "holds the most in range",
                              "hold the least in range"),
        ("exposed",    True,  "leaves the most missed or exposed",
                              "leaves the least missed or exposed",
                              "leave the most missed or exposed"),
        ("missed_oor", True,  "gives up the most of what is out of range",
                              "gives up the least of what is out of range",
                              "give up the most of what is out of range"),
        ("held_oor",   False, "holds the least of what is out of range",
                              "holds the most of what is out of range",
                              "hold the least of what is out of range"))


def _count_note(rnd, live, target, unp, polarity="any"):
    """A true sentence about the live set's band counts, or None.

    For a DROP joint the sentence has to point the same way as the drop, so it is emitted
    only when the target is alone at the BAD end. If the band separates the edits but not
    against this one, there is nothing here to argue with and the caller says so -- which
    is honest, where the earlier version wrote "Edit 3 misses the least, where the others
    do better" and then dropped Edit 3.
    """
    C = _counts(rnd, live, unp)
    rest = sorted(set(live) - {target})
    if not rest:
        return None
    for key, max_is_bad, bad, good, bad_pl in _CNT:
        v = {i: C[i][key] for i in live}
        if len(set(v.values())) < 2:
            continue
        worst = [i for i in live if v[i] == (max if max_is_bad else min)(v.values())]
        best = [i for i in live if v[i] == (min if max_is_bad else max)(v.values())]
        if target in worst and len(worst) == 1:
            return (f"Across the band, Edit {target + 1} {bad}, where "
                    f"{_fmt_edits(rest)} {_v(rest, 'does', 'do')} better.")
        if target in worst:
            # TIED FOR WORST is not "not bad", which is how the earlier text read it.
            return (f"Across the band, {_fmt_edits(sorted(worst))} are the weakest here "
                    f"-- they {bad_pl} -- and Edit {target + 1} is the first to go.")
        if polarity == "bad":
            return None       # the band separates them, but not against this edit
        if target in best and len(best) == 1:
            return (f"Across the band, Edit {target + 1} {good}, where "
                    f"{_fmt_edits(rest)} {_v(rest, 'does', 'do')} worse.")
        return (f"Across the band, {_fmt_edits(worst)} {_v(worst, 'is', 'are')} the one "
                f"that {bad}, {_fmt_edits(best)} the one that {good}, and Edit "
                f"{target + 1} sits between.")
    return None


# WHEN THE BAND DOES NOT ARGUE FOR THIS DROP. Two different things, and they must not be
# said with the same words. `_NULL_LEVEL` is a real tie -- every count equal -- and
# `_NULL_UNSINGLED` is the commoner case where the band does separate the edits but says
# nothing against the one on its way out. Several phrasings each, drawn per round, because
# a corpus that says one sentence a fifth of the time teaches the sentence.
_NULL_LEVEL = (
    "Counting what the band reports, {live_and} come out level, so nothing there settles "
    "it; going on the fragment itself, and Edit {t} is the first to go.",
    "The band gives {live_and} the same tally, so it has no call to make here -- setting "
    "Edit {t} aside first and seeing where that leaves the rest.",
    "There is no daylight between {live_and} on anything the band counts, so this one "
    "rests on what the fragment actually brings; Edit {t} goes first.",
)
_NULL_UNSINGLED = (
    "Nothing on the band singles Edit {t} out as the weakest, so it is the first to go "
    "on what the fragment brings rather than on the properties.",
    "The band has no complaint about Edit {t} in particular, so what sends it out first "
    "is the fragment rather than anything in the table above.",
    "On the band Edit {t} is not the one that stands out badly, so it goes out first on "
    "the fragment rather than on the properties.",
)
_NULL_HOLD = (
    "The two left sit the same way on everything the band reports, so the choice rests on "
    "the fragment itself.",
    "Counting what the band reports, the two that remain come out level, so what decides "
    "it is what the fragment actually puts in.",
    "The band has nothing left to separate these two, so it comes down to the fragment "
    "rather than the properties.",
)


def _pick_null(templates, key: str, **kw) -> str:
    h = int.from_bytes(_hashlib.blake2b(key.encode(), digest_size=8).digest(), "big")
    return templates[h % len(templates)].format(**kw)


def band_drop_order(rnd, pick: int, key: str):
    """Which two rivals to drop, worst on the band first. THE COMMIT IS NOT TOUCHED.

    The order axis is free: n8 against n8r put q's order against anything-but-q's and the
    difference was +0.002 over three seeds at every matched epoch. So it can be spent on
    making the prose true instead.

    It has to be spent, because q's order and the band do not agree. Under q, the edit on
    its way out was the one the band condemns on well under half of joints, and the
    honest draft then had nothing to say -- 30 spans read by hand had a real argument in
    the first joint 9 times. Ordering by the band makes the target the weakest edge by
    construction, so every joint has the thing it is supposed to say, and the prose and
    the eliminations finally follow the same rule.

    Ranked on `_CNT` lexicographically -- misses, then holds, then missed-or-exposed,
    then the two out-of-range counts -- and ties broken on a hash of the round so the
    corpus does not lean on whichever edit sorts first.
    """
    rivals = [i for i in range(len(rnd["candidates"])) if i != pick]
    if len(rivals) < 2:
        return []
    unp = {p for p, _ in bs.direction(rnd)}
    C = _counts(rnd, rivals, unp)
    h = int.from_bytes(_hashlib.blake2b(key.encode(), digest_size=8).digest(), "big")

    def badness(i):
        return tuple((C[i][k] if max_is_bad else -C[i][k]) for k, max_is_bad, *_ in _CNT)

    return sorted(rivals, key=lambda i: (tuple(-x for x in badness(i)),
                                         (h + i) % len(rivals)))[:2]


def draft_band(rnd, live, target, unp, used, kind: str):
    """One joint, from the band alone. True by construction, dull by design.

    THREE TIERS. A property where `target` stands alone AND BADLY is the sharpest thing
    the band can say. Failing that, the count comparison -- how much of what the band
    reports each edit gives up -- which is not a fallback so much as the signal itself:
    `z_misses` is the top band feature and "fewest misses" alone scores 0.3168 against
    0.250. Only when neither condemns the target does the joint say so.

    `kind` "hold" is the LAST joint and is deliberative: it lays the two survivors side by
    side and stops. It must not say which wins -- an arm that announces the commit before
    the commit line teaches the student to decide before it has looked.
    """
    live = set(live)
    if kind == "hold":
        a, b = sorted(live)[:2]
        ga = _sole(rnd, {a, b}, a, unp, used)
        gb = _sole(rnd, {a, b}, b, unp, used)
        if ga and gb:
            return (f"Between Edit {a + 1} and Edit {b + 1}, the first "
                    f"{_SAYS[ga[1]].format(p=ga[2])} while the second "
                    f"{_SAYS[gb[1]].format(p=gb[2])}, so it comes down to which of those "
                    f"the target can afford."), {ga[2], gb[2]}
        g, who = (ga, a) if ga else (gb, b) if gb else (None, None)
        if g:
            return (f"Of the two left, Edit {who + 1} {_SAYS[g[1]].format(p=g[2])} and "
                    f"nothing else on the band separates them."), {g[2]}
        # NO COUNT NOTE HERE. `_count_note` ranks, and a ranking of the two survivors is
        # a verdict: "Edit 2 holds the most in range, where Edit 4 does worse" with
        # `Taking Edit 2` underneath announces the commit before the commit line, which
        # is the one thing this joint exists not to do.
        return _pick_null(_NULL_HOLD, f'{rnd["group_id"]}:{rnd["depth"]}:hold'), set()

    got = _sole(rnd, live, target, unp, used, polarity="bad")
    if got is not None:
        _, rel, p, rest = got
        alone, sing, plur, tail = _REL[rel]
        lead = (f"On {p}, which is out of range, " if p in unp else f"On {p}, ")
        return (lead + f"Edit {target + 1} {alone}, while {_fmt_edits(rest)} "
                f"{_v(rest, sing, plur)} {tail}."), {p}

    note = _count_note(rnd, live, target, unp, polarity="bad")
    if note is not None:
        return note, set()

    key = f'{rnd["group_id"]}:{rnd["depth"]}:{target}'
    C = _counts(rnd, live, unp)
    level = all(len({C[i][k] for i in live}) < 2 for k, *_ in _CNT)
    if level:
        return _pick_null(_NULL_LEVEL, key, live_and=_fmt_edits(sorted(live)),
                          t=target + 1), set()
    return _pick_null(_NULL_UNSINGLED, key, t=target + 1), set()


_EDNUM = _re.compile(r"\bEdit\s+(\d+)")


def edits_in(text: str):
    """The Edit numbers a passage names, as a set."""
    return {int(x) for x in _EDNUM.findall(text or "")}


# A DROP JOINT SAYS SOMETHING BAD ABOUT ITS EDIT, and the reword must not turn that into
# something good. Measured on 30 spans read by hand: one polish took "Edit 1 is the only
# one LEFT EXPOSED on Mutag" to "Edit 1 is the only one STILL HOLDING it", which is false
# against the band and puts the corpus back to praising the edit it then drops -- the very
# thing the drafter was fixed to stop doing. Nothing else in the gate could see it: the
# edit numbers were kept and the sentence did not open with "Edit".
_GOOD_MARK = ("still hold", "bringing it back", "brings it back", "back within reach",
              "holds the most", "misses the fewest", "leaves the least",
              "retain", "maintain", "preserv")
# Words the band cannot support at all. "spectral coverage" and "the full spectral range"
# both appeared: the reword reached for a chemistry that is not in front of it, and these
# are ADMET properties, not spectra.
_INVENTED = ("spectral", "spectrum", "spectra", "nmr", "chromatograph", "bandwidth",
             "wavelength", "frequency", "signal", "peak", "absorb")
# The fourteen names the band can print. A reword may drop one, and often should -- "while
# the others are not" is better prose -- but it may not INTRODUCE one: "There is no
# daylight between Edit 1..4 on anything the band counts" came back as "ON MR, there is
# no daylight...", and MR was not in that round at all.
_PROPS = ("BBBP", "HBA", "HBD", "MR", "MW", "Mutag", "QED", "TPSA", "heavy_atoms",
          "logD", "logP", "logS", "rings_total", "rotB")
# The head verb of every comparison `_CNT` can make. A reword that keeps "Edit 1 and Edit
# 2 are the weakest here" and drops "they leave the most missed or exposed" has kept the
# verdict and thrown away the reason, which is the half that makes it reasoning.
_CNT_STEMS = {"missed": ("miss",), "held": ("hold",), "exposed": ("leave", "expos"),
              "missed_oor": ("give", "giving"), "held_oor": ("hold",)}


def props_kept(part: str, note: str) -> bool:
    """No property name the note did not have."""
    inn = {p for p in _PROPS if p in (note or "")}
    return all(p in inn for p in _PROPS if p in (part or ""))


def reason_kept(part: str, note: str) -> bool:
    """A count comparison must keep its reason, not just its verdict."""
    low_n, low_p = (note or "").lower(), (part or "").lower()
    if "the weakest here" not in low_n and "across the band" not in low_n:
        return True
    for key, stems in _CNT_STEMS.items():
        if any(st in low_n for st in stems):
            return any(st in low_p for st in stems)
    return True


def polarity_kept(part: str, i: int) -> bool:
    """A drop joint may not read as praise for the edit it is about."""
    if i >= 2:
        return True          # the deliberative joint weighs both sides and may say either
    low = (part or "").lower()
    return not any(m in low for m in _GOOD_MARK)


def no_invention(part: str) -> bool:
    low = (part or "").lower()
    return not any(m in low for m in _INVENTED)


def body_kept(part: str, note: str, frac: float = 0.6) -> bool:
    """The reword must not drop the note's second half.

    Measured on the same 30: "On the band Edit 1 is not the one that stands out badly, so
    this comes down to what the fragment brings" came back as the first clause alone,
    three joints out of three on one span. What is left is a bare assertion sitting above
    a drop line with nothing joining them, which is worse prose and a worse example than
    the note it was made from.
    """
    return len((part or "").split()) >= frac * len((note or "").split())


def edits_kept(part: str, note: str, must=None) -> bool:
    """Did the reword keep the edits its note named, and invent none?

    The one check n15 cannot do without. Its notes carry no observation values, so
    `has_number` and `false_ordinal` have nothing to read and the whole abstract gate is
    skipped -- which let a polish turn "Edit 3 goes first" into "starting with Edit 2"
    while the drop line below it still said Edit 3.

    NOT SET EQUALITY, though it was at first, and that cost 23.1% of passages and half the
    throughput. Most of those were not renumberings: "while Edit 1, Edit 3 and Edit 4 are
    not" came back as "while the others are not", which is better prose and loses nothing
    -- the frame lists all four immediately above. What must not happen is an edit
    appearing that the note never named, or the one the joint is ABOUT going missing.
    """
    got, have = edits_in(part), edits_in(note)
    if not got <= have:
        return False
    return must is None or must not in have or must in got


def draft_band_all(rnd, lives, targets, unp):
    """The three joints. Each takes a different property where one is available."""
    out, used = [], set()
    for live, tgt, kind in zip(lives, targets, N15_KINDS):
        sent, took = draft_band(rnd, live, tgt, unp, used, kind)
        used |= took
        out.append(sent)
    return out


# ---------------------------------------------------------------------------
# n17: THE THREE THINGS n16's JOINTS GOT WRONG. The frame is n16's, byte for byte.
#
# (1) A DROP JOINT MAY NOT RANK. `_count_note`'s tie branch ranked over the whole live
#     set while `band_drop_order` chose the target from the RIVALS ONLY, so the two
#     scopes disagreed and the sentence could call the commit weak -- "Across the band,
#     Edit 1, Edit 3 and Edit 4 are the weakest here ... and Edit 4 is the first to go",
#     with `Taking Edit 3` two lines below. Every n17 drop joint is an ABSOLUTE fact
#     read off the outgoing edit's own row: no superlative, no second edit named. That
#     also retires `_sole`'s "while Edit 2 and Edit 3 do not", which cleared the commit
#     on every joint the teacher wrote -- a tell the student has no way to reproduce.
#
# (2) THE CLOSER MAY NOT COMPARE ON THE BAND. Dropping the two band-worst rivals leaves
#     a band-strong rival by construction, and the teacher's own pick is the band-weaker
#     of the final pair on 69.1% of val rounds. n16's closer therefore argued for the
#     edit the teacher then declined, and the student learned it: it takes the
#     band-stronger survivor on 43.5% of rounds where the base rate is 60.8%.
#
# (3) THE FUNCTIONAL GROUPS EARN THEIR LINE. n16 printed them and never used them. They
#     are the right material for the closer BECAUSE they are not a criterion: ranked
#     every way tried -- most groups, most new groups, most the molecule already has,
#     longest fragment -- the argmax is the commit on 0.260-0.280 of rounds against
#     0.250 for a coin. Naming them describes the choice without claiming to make it,
#     and the closer is the one slot that can carry prose for free: n15's conversion
#     given the commit survived was 0.609 against n8's 0.598.
# n17 IS THE CONTRAST THAT ISOLATES THE DROP JOINT. Its frame is n8's, line for line --
# no functional-group line, no joint before COMMIT -- so `n17 - n8` is two sentences and
# nothing else, and whatever the pair measures is the drop reasoning on its own. n15/n16
# moved three things at once and could not say which of them cost the 0.053.
#
# It carries v28's functional-group line all the same, so `n17 - n8` is that line and two
# sentences rather than two sentences alone. The line is not a criterion -- ranked every
# way tried the groups pick the commit on 0.260-0.300 against 0.250 for a coin, and added
# to the band features they move a GBM by -0.005 -- but it is chemistry the student can
# recompute with match_substructure, and the joints below are allowed to use it.
#
# AND IT TAKES n8's q ORDER, not `band_drop_order`. The band order exists because n15's
# notes were superlatives and a superlative needs its target to be the worst; n17's are
# absolute facts off one row, which any order supports. Under q the target still has a
# miss or a risk to name on 92.0% of joints against 94.5% under the band -- 2.5 points,
# and it buys a pair that differs by two sentences instead of two sentences and an
# ordering.
#
# The commit joint is left empty on purpose even though it is the one slot that measured
# free (conversion 0.609 against n8's 0.598): free is not the same as helpful, and a
# second sentence in the same arm would put the two effects back together.
N17_KINDS = ("drop", "drop")
N17_ARMS = ("n17",)
# n18 is n17 plus the two things n17 leaves out: v28's functional-group line, and a
# closer built from it. The groups go THERE and nowhere else -- ranked every way tried
# they pick the commit on 0.260-0.280 of rounds against 0.250 for a coin, so they can
# describe a choice but must never appear to make one.
N18_KINDS = ("drop", "drop", "hold")
N18_ARMS = ("n18",)

_FRAG_FG = {}
# --------------------------------------------------------------------------- #
# THE GROUPS THE STUDENT CANNOT READ OFF A MOLECULE, AND WHAT IT COSTS TO KEEP THEM.
#
# Measured on n40 at epoch 3, over 25,767 reference FG mentions in the 4,980 held-out
# rounds: the share of mentions the model reproduces with the right name AND the right
# count. The catalog's frequent groups are essentially free -- the ten commonest carry
# 73% of all mentions and every one of them is above 0.985 -- and everything below 0.90
# is a rare group, a count over a repeated site, or both:
#
#   guanidine            0.250 (n=4)     amidine              0.474 (n=19)
#   nitroso              0.500 (n=2)     hydroxylamine        0.583 (n=24)
#   alkyl carbamate      0.625 (n=24)    allylic oxidation…   0.632 (n=87)
#   benzodiazepine       0.750 (n=4)     hydrazine            0.804 (n=51)
#   quaternary nitrogen  0.000 (n=1)
#
# Together 216 of 25,767 mentions (0.84%). Dropping them takes the FG line's whole-line
# accuracy from 0.913 to 0.926 and changes 3.4% of MOL_INFO lines, 1.2% of ANALYSIS fg
# cells and 0.9% of SUMMARY rows; 0.30% of SUMMARY rows lose their last fact and fall to
# `has nothing of its own here`.
#
# NOT IN THE SET, deliberately: `aryl methyl sites for hydroxylation` at 0.915 (n=645).
# It is the largest single source of what error remains -- 645 x 0.085 is 55 misses
# against 76 for all nine above -- but it is right nine times in ten and dropping it
# rewrites 18.8% of MOL_INFO lines and 15.3% of the fg cells. Add it to `FG_DROP` to buy
# 0.913 -> 0.938 at that price.
#
# THE FILTER HAS TO REACH BOTH SIDES. `_fg_change` calls a group NEW when it is in the
# fragment and not in `molfg`; filtering the molecule's list without filtering the
# fragment's would make the page claim a group is new when the molecule already has it.
# `new here` is true on 2,583 of 2,583 claims today and that is worth not breaking, so
# the filter lives inside `_frag_groups` and is applied to `molfg` at the same time.
FG_NONE = "Present in the molecule: nothing the catalog names."
FG_DROP = frozenset((
    "guanidine", "amidine", "nitroso", "hydroxylamine", "alkyl carbamate",
    "allylic oxidation sites excluding steroid dienone", "benzodiazepine",
    "hydrazine", "quaternary nitrogen",
))
# Set for the arm being rendered, in `build_prompt`. A render is one arm in one process,
# so a module flag is enough -- but the memo below is keyed on it anyway, because a
# stale cache entry would be silent and this is exactly the kind of thing that hides.
_FG_FILTER = False


def fg_keep(d):
    """`d` without the dropped groups, when the arm being rendered asks for that."""
    return {k: v for k, v in d.items() if k not in FG_DROP} if _FG_FILTER else dict(d)


def _frag_groups(smi: str) -> dict:
    """{group: count} for one candidate fragment, from the SAME 61-pattern catalog the
    n16 line is built from -- the closer and the line above it must not use two
    vocabularies for the same chemistry."""
    key = (smi, _FG_FILTER)
    if key not in _FRAG_FG:
        try:
            import fg_gates as _fgg
            _FRAG_FG[key] = fg_keep(_fgg.counts(smi) or {})
        except Exception:
            _FRAG_FG[key] = {}
    return _FRAG_FG[key]


def _and_list(xs) -> str:
    xs = list(xs)
    if not xs:
        return ""
    if len(xs) == 1:
        return xs[0]
    return ", ".join(xs[:-1]) + " and " + xs[-1]


def _fg_clause(g: dict, mol: dict) -> str:
    """What one fragment puts on, and whether the molecule already carries it."""
    if not g:
        return "nothing the catalog names"
    body = _and_list(f"{k} ({v})" for k, v in
                     sorted(g.items(), key=lambda kv: (-kv[1], kv[0])))
    new = sorted(k for k in g if k not in mol)
    if not new:
        return f"{body}, all of it already in the molecule"
    if len(new) == len(g):
        return f"{body}, none of it already in the molecule"
    return f"{body}, of which {_and_list(new)} {_v(new, 'is', 'are')} new here"


# "IT IS THE FIRST TO GO" UNDER THE SECOND `Dropping` LINE. Both joints said it, because
# the sentence was written without knowing which joint it was for. `_count_note` and the
# `_NULL_*` templates n15/n16 draw from have the same fault and it went unread there too.
_GOES = ("it is the first to go", "it is the next to go")
_GOES_NULL = ("it is the first to go all the same", "it goes next all the same")


# THE FLOOR IN THE SAME REGISTER AS THE SHEET. A joint that will not come back clean used
# to fall to "On logD, which is out of range, Edit 4 is left exposed, and it is the first
# to go" -- true, but written in a different voice from the joints around it. At a 28.2%
# fall-back rate that is a quarter of the corpus in a second register, which is a
# difference the student can read and nothing the arm means to teach. Built from the same
# against-facts the writer was given, so the two are the same sentence in two hands.
_GER = {"moves": "moving", "leaves": "leaving", "breaks": "breaking",
        "keeps": "keeping", "carries": "carrying", "adds": "adding",
        "drops": "dropping", "is": "being", "no": "making no"}


def _gerund(f: str) -> str:
    head, _, rest = f.partition(" ")
    return f"{_GER.get(head, head)} {rest}".strip()


def floor_from_facts(rnd, target: int, unp, molfg, j: int = 0) -> str:
    """The fall-back joint, cut from the target's AGAINST facts and nothing else."""
    facts = dict(fact_sheet(rnd, unp, molfg)).get(target) or []
    bad = [x for x in facts if _polarity(x) == "against"]
    tail = _GOES[min(j, 1)]
    if not bad:
        return (f"With no property counting against Edit {target + 1}, {tail} on "
                f"what the fragment brings rather than on the properties.")
    # ONE FACT WHEN THE FIRST ALREADY CARRIES A CLAUSE. "breaking heavy_atoms, which the
    # molecule satisfies now and carrying the widest spread" reads as one run-on where
    # the relative clause and the conjunction collide.
    # MERGE, DO NOT REPEAT. "moving MW the right way but nowhere near far enough and
    # moving QED the right way but nowhere near far enough" is one fact about two
    # properties, and the sheet lists it once per property.
    byt = {}
    for x in bad:
        for pr in _PROPS:
            if x.startswith(("moves " + pr + " ", "leaves " + pr + " ")):
                byt.setdefault(x.replace(pr, "{}", 1), []).append(pr)
                break
        else:
            byt.setdefault(x, [])
    merged = [(t.replace("{}", _and_list(ps), 1) if ps else t) for t, ps in byt.items()]
    take = [_gerund(x) for x in merged[:2]]
    if "," in merged[0] or len(take) == 1:
        take = take[:1]
    return (f"With Edit {target + 1} " + " and ".join(take) + f", {tail}.")


def draft_drop17(rnd, target: int, unp, used, j: int = 0):
    """One DROP joint, about the edit on its way out AND NOTHING ELSE.

    Out of range beats in range and a clean miss beats a risk -- `_sole`'s preference
    order, with the comparison taken out. `used` keeps the joints off one property.

    5.5% of drop targets have neither a miss nor a risk to name. That branch says so.
    `_NULL_UNSINGLED` used to say "it goes out first on what the fragment brings", which
    names a criterion that did not make the choice -- the band did -- and it was the
    most expensive sentence in the arm: paired against n8 on the same rounds it cost
    +0.093 on P(drops the commit) where the argued joints cost +0.042.
    """
    h, r, w, m = _band_of(rnd, target)
    for pool, oor, rel in ((m & unp, True, "miss"), (r & unp, True, "risk"),
                           (m, False, "miss"), (r, False, "risk")):
        if not pool:
            continue
        ps = sorted(pool - used) or sorted(pool)
        lead = (f"On {_and_list(ps)}, which {_v(ps, 'is', 'are')} out of range, "
                if oor else f"On {_and_list(ps)}, ")
        tail = (f"Edit {target + 1} gives {_v(ps, 'it', 'them')} up"
                if rel == "miss" else f"Edit {target + 1} is left exposed")
        return f"{lead}{tail}, and {_GOES[min(j, 1)]}.", set(ps)
    return (f"Nothing the band reports counts against Edit {target + 1}; "
            f"{_GOES_NULL[min(j, 1)]}."), set()


def draft_hold17(rnd, live, molfg):
    """The closer: what each of the two puts on the molecule, and no verdict.

    Ordered by edit number and symmetric on purpose. A clause hung on the commit alone
    would be the "first mention is the policy" shortcut from the other end, and the
    student reads that at 72%.
    """
    a, b = sorted(live)[:2]
    ga = _frag_groups((rnd["candidates"][a] or {}).get("to_smiles") or "")
    gb = _frag_groups((rnd["candidates"][b] or {}).get("to_smiles") or "")
    return (f"Of the two left, Edit {a + 1} puts on {_fg_clause(ga, molfg)}, while "
            f"Edit {b + 1} puts on {_fg_clause(gb, molfg)}; which of those the target "
            f"can carry is what settles it."), set()


def draft_band17_all(rnd, lives, targets, unp, molfg, kinds=N18_KINDS,
                     facts_floor=False):
    out, used = [], set()
    for j, (live, tgt, kind) in enumerate(zip(lives, targets, kinds)):
        if kind == "hold":
            sent, took = draft_hold17(rnd, live, molfg)
        elif facts_floor:
            sent, took = floor_from_facts(rnd, tgt, unp, molfg, j), set()
        else:
            sent, took = draft_drop17(rnd, tgt, unp, used, j)
        used |= took
        out.append(sent)
    return out


# n17's DROP joints make no comparison, and the reword may not add one. THE PROMPT'S OWN
# WORKED EXAMPLE TAUGHT IT: "In terms of MR, Edit 1 is THE ONE THAT gives it up" put a
# definite article in front of a note that had claimed nothing of the kind. Measured on
# the first 2,171 n17 val spans it landed on 51.2% of drop joints, and 64.4% of those
# were false -- another live edit did the very same thing. An example is a rule, and this
# one was teaching the opposite of the arm.
# "the one" ON ITS OWN IS THE WHOLE LEAK. Naming the wordings one at a time caught
# "is the one that" and missed "is the one LEFT EXPOSED", which then arrived on 29.2% of
# joints. No n17 note contains the word "one" at all, so any occurrence is an addition
# and the definite article in front of it is a uniqueness claim either way.
_COMPARE = ("the one", "the ones", "the only", "sole", "solely",
            "alone in", "alone among", "the weakest", "weakest here",
            "the most", "the least", "more than the", "fewer than the", "worse than",
            "better than", "unlike edit", "whereas the other", "while the other",
            "than the others", "than the rest", "stands out", "singles out")


def no_compare(part: str) -> bool:
    """A drop joint says what its edit does, never how it ranks against the others."""
    return not any(m in (part or "").lower() for m in _COMPARE)


POLISH_BAND18 = """Rewrite each note as one natural sentence a chemist would say while
working through the options.

Keep the whole note, including the clause after the semicolon -- a sentence that stops
at the first clause is not a rewrite of it. A note about an edit's weakness stays about
its weakness; never turn "left exposed" into "still holding".

Open the way the note opens, never with "Edit". "On MR, Edit 1 gives it up" may become
"Where MR is concerned, Edit 1 gives it up" -- but never "Edit 1 gives up MR". No
sentence may begin with the word "Edit".

MAKE NO COMPARISON. The note says what ONE edit does. It does not say that edit is the
only one, the weakest one, or worse than any other, and the rewrite may not say so
either: "Edit 1 is THE ONE THAT gives it up" is not a rewording of "Edit 1 gives it up",
it is a new claim, and usually a false one -- another edit is doing the same thing. No
"the only one", no "the one that", no "the one left exposed" -- do not write the word
"one" at all -- no "the weakest", no "unlike", no "the most".

NAME NO EDIT THE NOTE DOES NOT NAME. The first two notes each speak about ONE edit; do
not add a clause saying what the others do, however natural that reads. Change no Edit
number, use no digits, and add no property the note does not have.

The third note lists what each of the two remaining edits puts on the molecule. Keep it
even-handed, keep it off the property table, and never say which of the two wins.

Three sentences in order, separated by a line of ###, nothing else.

{notes}"""


POLISH_BAND = """Rewrite each note as one natural sentence a chemist would say while
working through the options.

Keep the whole note, including the clause after the comma that says what it comes down
to -- a sentence that stops at the first clause is not a rewrite of it. A note about an
edit's weakness stays about its weakness; never turn "left exposed" into "still holding".

Open with the property, never with "Edit". "On MR, Edit 1 is the only one still holding
it" may become "In terms of MR, Edit 1 is the one that still holds it" -- but never
"Edit 1 still holds MR". Moving the edit to the front is the one change that is not
allowed, and no sentence may begin with the word "Edit". Never change an Edit number: the same numbers,
no more, no fewer. No digits. Add no chemistry and no verdict about which edit is right.
A note that already moves provisionally ("Edit 2 goes first") keeps that; do not add one.
The third sentence weighs the two that are left and stops -- it never says which wins.

Three sentences in order, separated by a line of ###, nothing else.

{notes}"""


# n17's prompt is n18's minus the closer paragraph. Naming a third sentence that is not
# coming is not harmless: the writer supplies one, the gap parse finds three passages
# where the frame has two marks, and the round is thrown away.
# --------------------------------------------------------------------------- #
# n17's JOINTS ARE WRITTEN FROM THE WHOLE TURN, NOT FROM THE BAND.
#
# The band is a four-way bucketing of something the student can already see in full: the
# `suggest_edits` result carries every candidate's delta as avg +/- std per property, and
# the user turn carries the target ranges. Drafting from the buckets threw all of that
# away and then argued from what was left, which is why 68.8% of the rule-based notes
# were true of some other live edit as well -- the buckets tie where the numbers do not.
#
# TWO CALLS, NOT ONE. The second joint is written with the first joint's finished text in
# front of it, the way the student will meet it: a single call that fills both gaps has
# to guess what it already said. The rule-based note stays as the floor for a joint that
# never comes back clean, so no round is lost to the writer's prose.
# HIGH-LEVEL, AND NO ARITHMETIC. Handing the writer the exact deltas produced sentences
# that were specific and wrong: 8 of 16 joints read by hand asserted something the numbers
# printed two lines above contradict -- "Edit 1's +56.026 shift overshoots the 354.0
# ceiling" where 278.117 + 56.026 lands at 334, inside. `numbers_ok` cannot see that; it
# checks that a digit string appears above, not that the sum comes out where the sentence
# says. `false_compare` only relates two printed numbers to each other.
#
# So the numbers come out of the sentence entirely. The writer still SEES them -- it needs
# them to know which way each property is wrong and by how much it matters -- but it may
# only say what they mean. A claim with no arithmetic in it cannot get the arithmetic
# wrong, and what is left is checkable against the band, which is derived from the same
# numbers: the joint may name a property only where the band already condemns this edit.
# --------------------------------------------------------------------------- #
# THE FACT SHEET. Everything the turn supports, written as sentences with no values in
# them, by rule, and the writer sees ONLY this.
#
# The two earlier shapes each failed at one end. Drafting from the band alone was true by
# construction and said nothing: 68.8% of notes were equally true of another live edit.
# Handing over the raw deltas bought specificity and lost the truth -- 8 of 16 joints read
# by hand asserted an arithmetic the numbers two lines above contradict, because a gate
# can check that a digit was printed above but not that the sum lands where the sentence
# says.
#
# A fact sheet is both. Every line on it is computed, so nothing on it can be false; the
# lines carry no numbers, so nothing can be got wrong by arithmetic; and there are enough
# of them -- direction, distance, spread, regression, chemistry -- that the writer has a
# real choice about which to argue from, which is where the prose comes from.
_MOVE = {("toward", "holds"):  "moves {p} the right way and far enough",
         ("toward", "reach"):  "moves {p} the right way, close but not all the way",
         ("toward", "risk"):   "moves {p} the right way, though it may not hold",
         ("toward", "miss"):   "moves {p} the right way but nowhere near far enough",
         ("away",   "holds"):  "moves {p} the wrong way, though not enough to matter",
         ("away",   "reach"):  "moves {p} the wrong way",
         ("away",   "risk"):   "moves {p} the wrong way",
         ("away",   "miss"):   "moves {p} the wrong way and out",
         ("still",  "holds"):  "leaves {p} where it is",
         ("still",  "reach"):  "leaves {p} where it is",
         ("still",  "risk"):   "leaves {p} where it is",
         ("still",  "miss"):   "leaves {p} where it is, which is outside"}


def _slot_of(rnd, i):
    """{property: holds|risk|reach|miss} for one edit, stars stripped."""
    h, r, w, m = _band_of(rnd, i)
    out = {}
    for s, k in ((h, "holds"), (r, "risk"), (w, "reach"), (m, "miss")):
        for p in s:
            out[p] = k
    return out


def _fg_change(rnd, i, molfg):
    """(added groups, removed groups, which of the added are new to the molecule)."""
    c = rnd["candidates"][i]
    to = _frag_groups(c.get("to_smiles") or "")
    fr = _frag_groups(c.get("from_smiles") or "")
    add = sorted(k for k in to if to[k] > fr.get(k, 0))
    rem = sorted(k for k in fr if fr[k] > to.get(k, 0))
    return add, rem, [k for k in add if k not in molfg]


def fg_row(rnd, i, molfg) -> str:
    """The band row's last cell: what this edit puts on and takes off."""
    add, rem, new = _fg_change(rnd, i, molfg)
    bits = []
    if add:
        bits.append("adds " + _and_list(add)
                    + (", new here" if len(new) == len(add)
                       else f", of which {_and_list(new)} new here" if new
                       else ", already present"))
    if rem:
        bits.append("drops " + _and_list(rem))
    return "; ".join(bits) if bits else "no change to any named group"


# A FACT IS FOR THE EDIT OR AGAINST IT, and the drop joint may only argue from the ones
# against. Left unmarked, the writer reached for "carries the narrowest spread on BBBP"
# -- the most certain of the four, the best thing on its sheet -- and dropped the edit
# for it. This is `_sole`'s polarity bug in a new coat: the first n15 praised the edit it
# then dropped on 18.4% of joints, and an unlabelled sheet re-opens exactly that.
# ORDERED, BECAUSE THE PHRASES NEST. "moves p the wrong way, though not enough to
# matter" contains "the wrong way", and an unordered scan called it a reason to drop --
# the band says that edit HOLDS p, so the move is harmless and the sentence would have
# dropped an edit for nothing. Longest and most specific first; the bare cases last.
_POL = (("though not enough to matter", "neutral"),
        ("and far enough",              "for"),
        ("keeps ",                      "for"),
        ("narrowest spread",            "for"),
        ("close but not all the way",   "against"),
        ("though it may not hold",      "against"),
        ("nowhere near far enough",     "against"),
        ("the wrong way",               "against"),
        ("leaves ",                     "against"),
        ("breaks ",                     "against"),
        ("widest spread",               "against"))


def _polarity(f: str) -> str:
    """Is this fact a reason to drop the edit, a reason to keep it, or neither?

    EVERY `_MOVE` fact is about a property that is ALREADY out of range -- `fact_sheet`
    emits them only for `bs.direction`. So "leaves p where it is" is never neutral: the
    property is wrong and this edit does not touch it. The one genuinely harmless entry
    is the wrong-way move the band still scores as holding.
    """
    for m, k in _POL:
        if m in f:
            return k
    return "neutral"


def fact_sheet(rnd, unp, molfg) -> list:
    """[(edit index, [fact, ...])] for every candidate. No number appears anywhere."""
    n = len(rnd["candidates"])
    slot = {i: _slot_of(rnd, i) for i in range(n)}
    # spread is comparative, so it is a RANK and never a value
    std = {}
    for p in (rnd.get("targets") or {}):
        v = []
        for i, c in enumerate(rnd["candidates"]):
            dp = ((c.get("delta") or {}).get(p) or {})
            v.append(abs(float(dp.get("std") or 0.0)))
        if len(set(v)) > 1:
            std[p] = (max(range(n), key=lambda j: v[j]), min(range(n), key=lambda j: v[j]))
    size = [len((c.get("to_smiles") or "")) - len((c.get("from_smiles") or ""))
            for c in rnd["candidates"]]
    big, small = max(range(n), key=lambda j: size[j]), min(range(n), key=lambda j: size[j])
    out = []
    for i in range(n):
        f = []
        for p, d in bs.direction(rnd):                  # the properties still wrong
            dp = ((rnd["candidates"][i].get("delta") or {}).get(p) or {})
            a = float(dp.get("avg") or 0.0)
            want_up = (d == "below")
            way = "still" if a == 0 else ("toward" if (a > 0) == want_up else "away")
            f.append(_MOVE[(way, slot[i].get(p, "risk"))].format(p=p))
        h, r, w, m = _band_of(rnd, i)
        broke = sorted(x for x in _band_raw_stars(rnd, i))
        if broke:
            f.append("breaks " + _and_list(broke) + ", which the molecule satisfies now")
        safe = sorted(set(h) - unp)
        if safe:
            f.append("keeps " + _and_list(safe) + " inside")
        for p, (hi, lo) in std.items():
            if i == hi:
                f.append(f"carries the widest spread here on {p}")
            elif i == lo:
                f.append(f"carries the narrowest spread here on {p}")
        f.append(fg_row(rnd, i, molfg))
        if i == big:
            f.append("is the largest fragment here")
        elif i == small:
            f.append("is the smallest fragment here")
        out.append((i, f))
    return out


def _band_raw_stars(rnd, i):
    """The properties this edit BREAKS -- in range now, out of range after. The `*` the
    band prints, which reads as `already wrong` to anyone who has not been told."""
    h, r, w, m = bs.classify(rnd, rnd["candidates"][i])
    return {str(x)[:-1] for x in m if str(x).endswith("*")}


# THE STATIC HALF COMES FIRST. The rules were 37.5% of a 6.5 KB prompt and identical on
# every round, sitting at the END where a prefix cache cannot reach them -- the servers
# run with `--enable-prefix-caching` and it was hitting nothing. In front, they are one
# shared prefix for the whole corpus.
#
# AND BOTH JOINTS IN ONE CALL. Sequential was the honest shape -- the second joint sees
# what the first said -- but it doubles the calls, and the frame below already shows the
# writer both drop lines, so it knows what the other joint is about. Measured cost of
# the split: 6.14 calls a span against 39 KB of prefill each.
#
# AND ONLY THE TWO EDITS THAT GO. The sheet carried all four, and the other two were
# never once used: over 936 joints the rate at which a joint names a SURVIVING edit is
# 0.000. Three quarters of the sheet was prefill for nothing.
# --------------------------------------------------------------------------- #
# n19: THE CRITERION COMES FIRST AND THE DROPS FALL OUT OF IT.
#
# Every earlier arm eliminated first and justified afterwards, and the elimination was
# irrevocable: measured over 5,000 heldout rounds, once a student had written "Dropping
# Edit N" it committed N on 0.000 of them. So `exact = (1 - P(drops the commit)) x
# P(commits it given it survives)`, and n8 lost 0.334 in the first term, n15 0.389. Here
# the drops are the edits the criterion excludes, so the commit CANNOT be among them and
# that term is 1 by construction.
#
# NO MODEL WRITES ANY OF THIS. Every line is computed. The last three arms each bought
# prose and paid for it in truth -- 8 of 16 joints arithmetically false when the writer
# had the numbers, 68.8% true-of-another-edit when it had only the band -- and the
# criterion needs neither: 91.9% of rounds have a phrase that is true of the commit, of
# no other edit, and pointing the way the molecule has to move.
#
# WHAT IS AND IS NOT SAYABLE AS A CRITERION. A phrase may be the criterion only if it is
# a reason to WANT the edit. "gives QED up" singles the commit out on plenty of rounds
# and the unconstrained search picked it, which produced "we should select an edit which
# gives QED up" under a NEEDS line saying QED is short. Polarity depends on the property:
# moving one that is out of range is good and moving one that is already in range is not,
# so `barely moves X` flips sign with X.
N19_ARMS = ("n19",)
# how far the criterion narrows the field, and how the line reads at each strength
N19_LINE = {3: "We should select an edit which {c}.",
            2: "We should narrow to the edits which {c}.",
            1: "We should at least rule out the edits which do not {c}."}
# `which do not {c}` needs the bare verb: "does not holds logP" is what the inflected
# phrase gives, and the phrases are written for the third person singular because that is
# how the other two lines read.
_N19_BARE = ((" is left exposed", " left exposed"), ("holds ", "hold "),
             ("brings ", "bring "), ("gives ", "give "), ("breaks ", "break "),
             ("puts on ", "put on "), ("takes off ", "take off "), ("adds ", "add "),
             ("moves ", "move "), ("raises ", "raise "), ("lowers ", "lower "),
             ("is the ", "rank "), ("carries ", "carry "))


def _bare(ph: str) -> str:
    for a, b in _N19_BARE:
        if ph.startswith(a.strip() + " ") or ph.startswith(a):
            return ph.replace(a, b, 1)
    return ph
_N19_NUM = ("carbon", "nitrogen", "oxygen", "halogen", "heavy atoms", "rings",
            "rotatable bonds")
# The subset the row actually PRINTS. heavy atoms is the sum of the elements and
# rotatable bonds rarely separates anything, so both were left out of the cell -- and
# n20 may not call a fact on-table when the cell it would be checked against is not
# there. `adds the fewest heavy atoms` was the first row this caught.
_N19_CELL = ("carbon", "nitrogen", "oxygen", "halogen", "rings")


def _atom_delta(rnd, i):
    """{element or ring or rotatable-bond: how many this edit adds}, off the fragment."""
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    out = {}
    for key, smi in (("to", rnd["candidates"][i].get("to_smiles") or ""),
                     ("fr", rnd["candidates"][i].get("from_smiles") or "")):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return {k: 0 for k in _N19_NUM}
        a = _collections.Counter(x.GetSymbol() for x in m.GetAtoms()
                                 if x.GetSymbol() != "*")
        out[key] = {"carbon": a.get("C", 0), "nitrogen": a.get("N", 0),
                    "oxygen": a.get("O", 0),
                    "halogen": sum(a.get(x, 0) for x in ("F", "Cl", "Br", "I")),
                    "heavy atoms": sum(a.values()),
                    "rings": m.GetRingInfo().NumRings(),
                    "rotatable bonds": rdMolDescriptors.CalcNumRotatableBonds(m)}
    return {k: out["to"][k] - out["fr"][k] for k in _N19_NUM}


def n19_facts(rnd, molfg, reach_out=False):
    """{edit: {key: (phrase, tier, is a reason to WANT it)}}.

    `tier` orders the choice when several phrases single the commit out: what is out of
    range first, then the rest of the band, then chemistry, then bare counts. The
    shortest phrase breaks a tie -- and only after the tier, because sorting on length
    alone is what put "barely moves QED" in front of "moves QED the right way" on 37.3%
    of rounds.
    """
    n = len(rnd["candidates"])
    props = sorted(rnd.get("targets") or {})
    dirs = bs.direction(rnd)
    unp = {p for p, _ in dirs}
    up = {p: (w == "below") for p, w in dirs}
    F = {i: {} for i in range(n)}
    S, AV, SD = {}, {p: [] for p in props}, {p: [] for p in props}
    for i in range(n):
        h, r, w, m = bs.classify(rnd, rnd["candidates"][i])
        st = lambda xs: {str(x).rstrip("*") for x in xs}
        S[i] = {"holds": st(h), "risk": st(r), "reach": st(w), "miss": st(m)}
        for p in S[i]["holds"]:
            if p in props: F[i][("holds", p)] = (f"holds {p}", 0 if p in unp else 1, True)
        for p in S[i]["reach"]:
            if p in props:
                # `reach` is "the mean sits outside the box and the one-sigma band still
                # touches it", and `classify` never asks whether the property was inside
                # to begin with. On 0.2640 of them it WAS -- the edit pushes an already
                # satisfied property out, recoverably -- and `brings p back within reach`
                # then says the reverse of what happened. `N21_LIFT` has split the two
                # since it was fitted (0.2747 against 0.2695); only the phrase had not.
                # Behind a flag so every arm already assembled renders byte-identical.
                F[i][("reach", p)] = (
                    f"brings {p} back within reach" if p in unp or not reach_out
                    else f"puts {p} out, though not out of reach", 0, p in unp)
        for p in S[i]["risk"]:
            if p in props: F[i][("risk", p)] = (f"is left exposed on {p}", 0, False)
        for p in S[i]["miss"]:
            # `miss` is the whole band outside the box, and 0.0794 of them are properties
            # the molecule SATISFIED before this edit -- `gives p up` then says the edit
            # failed to fetch something that was never lost. Those already have an exact
            # fact of their own two lines down (`breaks p, which the molecule satisfies
            # now`, the `*` in the cell), so the duplicate is dropped rather than
            # reworded. It reached the reason line only 3 times in 4,362 -- the two tie
            # on lift and the longer phrase wins -- but the BLOCK printed it on 212 of
            # 905 rows, and the block is the part the student is scored against.
            if p in props and not (reach_out and p not in unp):
                F[i][("miss", p)] = (f"gives {p} up", 0, False)
        for x in m:
            if str(x).endswith("*"):
                q = str(x)[:-1]
                F[i][("break", q)] = (f"breaks {q}, which the molecule satisfies now",
                                      0, False)
        add, rem, new = _fg_change(rnd, i, molfg)
        for g in add:
            F[i][("adds", g)] = ("puts on " + g + ("" if g in molfg else
                                 ", which the molecule does not have yet"), 2, True)
        for g in rem:
            F[i][("drops", g)] = (f"takes off {g}", 2, True)
        for p in props:
            dp = (rnd["candidates"][i].get("delta") or {}).get(p) or {}
            AV[p].append(float(dp.get("avg") or 0.0))
            SD[p].append(abs(float(dp.get("std") or 0.0)))
    for g in {k for i in range(n) for t, k in F[i] if t == "adds"}:
        for i in range(n):
            if ("adds", g) not in F[i]:
                F[i][("notadds", g)] = (f"puts on no {g}", 2, True)

    def sup(vals, hi, lo):
        if vals.count(max(vals)) == 1: yield vals.index(max(vals)), hi
        if vals.count(min(vals)) == 1: yield vals.index(min(vals)), lo

    for key, fn, hi, lo in (
            ("miss", lambda j: len(S[j]["miss"]),
             ("gives up the most", False), ("gives up the least", True)),
            ("hold", lambda j: len(S[j]["holds"]),
             ("holds the most", True), ("holds the least", False))):
        for i, (ph, good) in sup([fn(j) for j in range(n)], hi, lo):
            F[i][("cnt", key, ph)] = (ph, 1, good)
    dv = {i: _atom_delta(rnd, i) for i in range(n)}
    for k in _N19_NUM:
        for i in range(n):
            if dv[i][k] == 0: F[i][("no", k)] = (f"adds no {k}", 3, True)
        for i, ph in sup([dv[j][k] for j in range(n)],
                         f"adds the most {k}", f"adds the fewest {k}"):
            F[i][("num", k, ph)] = (ph, 3, True)
    for p in props:
        oor = p in unp
        for i, (ph, good) in sup([abs(x) for x in AV[p]],
                                 (f"moves {p} the most", oor),
                                 (f"barely moves {p}", not oor)):
            F[i][("mv", p, ph)] = (ph, 0 if oor else 2, good)
        for i, (ph, good) in sup(AV[p],
                                 (f"raises {p} the most", oor and up.get(p, False)),
                                 (f"lowers {p} the most", oor and not up.get(p, True))):
            F[i][("dir", p, ph)] = (ph, 0 if oor else 2, good)
        for i, (ph, good) in sup(SD[p], (f"is the least certain on {p}", False),
                                        (f"is the most certain on {p}", True)):
            F[i][("sd", p, ph)] = (ph, 1 if oor else 3, good)
        if oor:
            for i in range(n):
                v = AV[p][i]
                if v != 0 and ((v > 0) == up[p]):
                    F[i][("way", p)] = (f"moves {p} the right way", 0, True)
                elif v != 0:
                    F[i][("way", p)] = (f"moves {p} the wrong way", 0, False)
    return F


def n19_criterion(rnd, pick, molfg):
    """(the criterion line or None, the rivals it excludes).

    Only a phrase that is a reason to WANT the edit, and among those the one that rules
    the most rivals out -- a criterion that settles the commit beats one that only
    narrows, and the frame has exactly two eliminations to spend.
    """
    F = n19_facts(rnd, molfg)
    n = len(rnd["candidates"])
    best = None
    for k, (ph, tier, good) in F[pick].items():
        if not good:
            continue
        out = [j for j in range(n) if j != pick and k not in F[j]]
        c = (-len(out), tier, len(ph))
        if best is None or c < best[0]:
            best = (c, ph, out)
    if best is None or not best[2]:
        return None, []
    _, ph, out = best
    k = min(len(out), 3)
    return N19_LINE[k].format(c=_bare(ph) if k == 1 else ph), out


def n19_order(rnd, pick, out, key):
    """The two eliminations: edits the criterion excluded, then -- only if it excluded
    fewer than two -- whatever is left, drawn on a hash of the round so the padding is
    reproducible and does not lean on the lowest index."""
    rivals = [i for i in range(len(rnd["candidates"])) if i != pick]
    h = int.from_bytes(_hashlib.blake2b(key.encode(), digest_size=8).digest(), "big")
    rest = sorted(set(rivals) - set(out), key=lambda i: (h + i) % len(rivals))
    return (sorted(out) + rest)[:2]


# --------------------------------------------------------------------------- #
# n20: THE SAME FACTS, ONE PER EDIT, AND THE DECISION LEFT WHERE n8 HAS IT.
#
# n19 asked for ONE line -- the commit's distinguishing phrase -- and made the two
# eliminations follow from it. Epoch 1 says the mechanism was learned and the line was
# not: the student writes a criterion on 0.998 of rounds, obeys its own criterion on
# 0.946, never commits an edit it has dropped (0.000), and gets the line itself right on
# 0.196. The two halves that came out of that are the whole reason this arm exists.
#
# 1. THE LINE WAS ANSWER-DEPENDENT, SO IT COULD NOT BE DERIVED. Beside it, the two cells
#    n19 added to every band row -- the functional-group change and the atom counts --
#    scored 0.981 and 0.981, because each is a function of its own row. The criterion is
#    a function of WHICH EDIT WINS, and no amount of table in front of it makes that
#    derivable: band+fg+count argmax predicts the commit at 0.350. Here every edit gets
#    its own phrase, chosen from its own facts, so the block joins the 0.98 family.
#
# 2. A WRONG CRITERION WAS WORSE THAN NO CRITERION. exact | criterion right = 0.956;
#    exact | criterion wrong = 0.129, BELOW the 0.250 random floor, because the drops
#    were bound to the criterion and a wrong one carried the commit out with it. n20
#    unbinds them: the eliminations are q's, exactly as in n8, and a wrong phrase costs
#    a wrong phrase.
#
# What is left is n8 plus a block of derivable observations, which is the comparison the
# programme wants: n8's curve is measured, and the only difference is whether naming
# each edit's own distinguishing fact helps the elimination that follows.
#
# NO POLARITY FILTER. n19 could only say things that were a reason to WANT the edit --
# "we should select an edit which gives QED up" is not a sentence -- and that cost 3.5pp
# of identifiability. A block that DESCRIBES has no such problem: sole-phrase coverage
# goes 0.9082 -> 0.9665 per edit, and rounds with all four covered 0.714 -> 0.902.
N20_ARMS = ("n20",)
# WHICH FACTS THE STUDENT CAN CHECK AGAINST WHAT IT JUST WROTE. Everything in this set
# is a function of the three cells of the band row above it -- the slots, the group
# change, the atom counts -- or of counting those cells down the column. Everything
# outside it (`mv`, `dir`, `sd`, `way`) is a cross-candidate argmax over the predicted
# shifts, which live in the tool result and never appear in the span. Both are honest
# observations and neither needs the answer, but only the first kind can be marked
# against the table the model has already produced, so it is preferred and the other is
# a fallback. Without the preference the block filled with `barely moves X` -- three
# rows of four on a six-property round, because one edit per property is the argmin by
# construction -- which is distinguishing and says nothing.
N20_ONTABLE = frozenset(("holds", "reach", "risk", "miss", "break",
                         "adds", "drops", "notadds", "cnt", "no", "num"))


def _n20_rank(k, molfg, cell=None):
    """(is it on the table, does it earn its place inside its tier).

    `cell` is the element list the band row's count cell actually prints, and it is an
    ARGUMENT because it is not the same for every arm: n25 drops the cell entirely, so
    on that arm every count fact is off the table and -- with `N20_MSG_FALLBACK` off --
    out of the block. Passing the wrong one is the `adds the fewest heavy atoms` bug at
    arm scale: a block full of claims the row underneath it cannot settle.

    The second half is only for the group facts, which all share tier 2 and are then
    separated by phrase length -- and the longest of them is the informative one, since
    `, which the molecule does not have yet` is what makes it long. Left alone the sort
    preferred `puts on amide` on a molecule that already has an amide over `puts on
    sulfonamide, which the molecule does not have yet` on the same edit.
    """
    cell = _N19_CELL if cell is None else cell
    on = 0 if k[0] in N20_ONTABLE else 1
    if k[0] == "adds":
        return on, 0 if k[1] not in molfg else 1
    if k[0] == "notadds":                       # `puts on no X` -- true, and thin
        return on, 2
    if k[0] in ("no", "num") and k[1] not in cell:
        return 1, 0                             # the cell does not print this element
    return on, 0
N20_HEAD = "What sets each edit apart:"
# AN EDIT WITH NOTHING OF ITS OWN SAYS SO, AND SAYS NOTHING ELSE. The version in between
# printed its least-shared true fact with a suffix admitting the fact was shared, and
# that was the worst of the three: on a round whose commit and one rival both only
# `hold logP`, the block gave THE SAME SENTENCE to the right edit and the wrong one. A
# block headed "what sets each edit apart" may carry only what does -- so the row stays,
# because every edit is answered for, and it carries no fact at all.
N20_NONE = "no feature of its own here"
# The most facts a row may carry. FOUR covers the edit completely on 0.8245 of rows --
# that is the share with four or fewer of their own under the band/fg/count vocabulary --
# and the rest are cut down to four by a draw keyed on the round, so the cut is the same
# every time this corpus is built. WHAT THE DRAW COSTS: the student cannot recompute a
# blake2b digest, so on the 0.1755 of rows that are cut it cannot know WHICH four were
# kept, and `feat_row` (whole-row string equality) is capped near 0.82 by construction.
# `feat_fact` is scored beside it for that reason -- see `check_n19`. The ORDER inside a
# row is never drawn: the kept facts are printed in the arm's fixed ranking, so a row
# that was not cut is fully determined.
N20_FACTS = 4
# Whether a row may fall back to a fact that lives only in the tool result -- the
# cross-candidate argmax over the predicted shifts (`moves/raises/lowers {p} the most`,
# `barely moves {p}`, `is the most/least certain on {p}`). Everything else in the
# vocabulary can be checked against the row printed above it. See the measurement in the
# n20 header: turning this off takes all-four coverage from 0.902 to 0.340.
N20_MSG_FALLBACK = False
# NOT `  Edit N | ...`. That is the band row's own form, and `bucket_span._ROW` scans the
# WHOLE span for it -- a second block in the same shape would be read as more band rows.
N20_ROW = "  Edit {n} -- {ph}"


def n20_features(rnd, molfg, cap=None, cell=None, reach_out=False):
    """{edit: the facts true of it and of no other edit, most telling first}.

    Chosen on-table first (see `N20_ONTABLE`), then by the order n19's criterion used --
    what is out of range first, then the rest of the band, then chemistry, then bare
    counts, shortest phrase inside a tier -- minus the `is a reason to want it` filter,
    and with the phrase itself as the last tie-break so the choice does not ride on dict
    order.

    `pick` is not an argument and `n19_facts` never sees it: the block is a function of
    the table alone. That is the point of the arm -- an observation that depends on the
    answer is the thing that scored 0.196.
    """
    F = n19_facts(rnd, molfg, reach_out)
    n = len(rnd["candidates"])
    cap = N20_FACTS if cap is None else cap
    out = {}
    for i in range(n):
        ranked, shared = [], []
        for k, (ph, tier, _g) in F[i].items():
            on, pref = _n20_rank(k, molfg, cell)
            if on and not N20_MSG_FALLBACK:
                continue
            others = sum(1 for j in range(n) if j != i and k in F[j])
            (ranked if others == 0 else shared).append(
                (others, on, tier, pref, len(ph), ph))
        # `others` is 0 for every entry in `ranked`, so it drops out of the sort and
        # the order is the one the arm is built on: on-table, then tier, then the group
        # tie-break, then length. `shared` is collected and discarded -- kept in the
        # loop only because the soleness test is the same pass.

        if len(ranked) > cap:
            ranked = sorted(ranked, key=lambda x: _hashlib.blake2b(
                f'{rnd["group_id"]}:{rnd["depth"]}:{i}|{x[5]}'.encode(),
                digest_size=8).digest())[:cap]
        out[i] = [x[5] for x in sorted(ranked)]
    return out


# --------------------------------------------------------------------------- #
# n21: n20, AND A CLAUSE AFTER EACH DECISION LINE.
#
# The bare frame beats every arm that argues: noreason 0.385 < n8 0.427, and the three
# arms that put prose in front of the eliminations came in at 0.334, 0.317 and 0.291.
# n21 is not an attempt to beat n8. It is an attempt to write the reasoning a reader
# expects to see WITHOUT paying for it -- a model that emits nothing but `Dropping Edit
# 3` reads as a lookup table, and that is the only problem this arm solves.
#
# TWO THINGS MAKE THE CLAUSE FREE.
#
# 1. IT COMES AFTER THE WHOLE LINE. The edit index is emitted first, so in an
#    autoregressive decode the clause cannot reach the token it justifies. Every earlier
#    arm put the argument in FRONT of the elimination, and that is exactly where the
#    policy moved: in n15/n16 the edit the sentence named was the edit that went on
#    1.000 of joints, so the decision became "which property to talk about", taken
#    earlier and from a narrower basis than the bare drop token. After the line there is
#    no such choice to make.
#
#    It also has to be after the WHOLE line, not spliced into it. `bucket_span._DROP` is
#    anchored on `^Dropping Edit N; that leaves `, and a clause put before "that leaves"
#    makes `dropped_edit` return None -- which switches off drop_present, drop_match and
#    drop_avoid_commit without a word in the log. Those are the numbers this arm lives
#    or dies by.
#
# 2. GIVEN THE EDIT, THE CLAUSE IS DETERMINED. It quotes a fact already computed for
#    that edit -- by preference the one printed in its block row above, so the model is
#    copying its own output rather than deriving anything new. Nothing is chosen at the
#    joint except the edit, exactly as in n8.
#
# POLARITY, AND WHY IT IS THE ONLY CONTENT RULE. Non sequiturs are accepted here -- a
# clause does not have to be a REASON, only true -- but a contradiction is not: `Dropping
# Edit 2. It holds the most.` is worse than silence. The split is the 0.250 floor of
# N21_LIFT: a drop clause may not use a template that points at the commit MORE than
# chance, a commit clause may not use one that points at it LESS. Measured over the test
# split this leaves 0 violations, 0.804/0.807 of drop clauses strongly negative and
# 0.909 of commit clauses strongly positive; 0.049 of drop joints have nothing sayable
# on the right side and get no clause at all, which after the drop token costs nothing.
# --------------------------------------------------------------------------- #
# n22: THE ENRICHED TABLE AND NOTHING ELSE.
#
# n8 with the molecule's groups named and two cells added to every band row -- the group
# change and the atom counts -- and no other difference: q's drop order, no criterion, no
# per-edit block, no clause. It exists because the forced-prefix experiment on n20 said
# the block is not the lever. Handed the block PERFECTLY, at no cost, the arm went 0.3914
# -> 0.3994; handed the same block with its rows attached to the WRONG edits it went to
# 0.4012, higher than the truthful one. The content is not read. What that leaves
# untested is the table itself: n19/n20/n21 all carry the enriched table AND something
# else, so no arm on the curve prices the two cells on their own against n8.
#
# The prediction from the same experiment is that this is flat too -- a correct band
# table forced for free bought 0.0066 -- and the point of running it is that `drops`
# bought 0.2339 on the same rounds, so the gap between an observation and an elimination
# is the result, and it needs the observation arm to be clean.
# --------------------------------------------------------------------------- #
# n23: n20 WITH THE CAP AT EIGHT.
#
# The same block, the same ranking, the same draw -- only the number of facts a row may
# carry. FOUR was costing the arm's own metric more than it was costing the span: an
# edit has more than four facts of its own on 0.1755 of rows, and a student that knows
# the fact set but cannot recompute a blake2b digest tops out at `feat_fact` 0.8675 and
# `feat_row` 0.8409 by construction. At eight, only 0.0266 of rows are cut and those
# ceilings go to 0.9814 and 0.9749, so what the number says about learning is no longer
# mixed with what the draw took away.
#
# The span barely grows -- the k distribution is piled at 0-2, so facts per row goes
# 1.90 -> 2.30 -- and the accuracy prediction is that nothing moves: handed the cap-4
# block PERFECTLY the arm went 0.3914 -> 0.3994, and handed the same block with its rows
# attached to the WRONG edits it went to 0.4012. Content that is not read does not become
# read by there being more of it. This arm is here to price the METRIC, not the answer.
N23_ARMS = ("n23",)
N23_FACTS = 8
# --------------------------------------------------------------------------- #
# n24: n22 WITHOUT THE ATOM COUNTS.
#
# The band slots and the group change, and no third cell. n22 is statistically tied with
# n8 -- paired over the same 4,980 rounds, five epochs, the largest |z| is 1.58 and it is
# in n22's FAVOUR -- so the two cells cost nothing, but neither has been priced on its
# own. n24 - n22 is the counts alone.
#
# The lift table says they are the emptier of the two. `adds the most carbon` points at
# the commit 0.2555 of the time and `adds the fewest carbon` 0.2565, over 98k pointings
# between them: the largest count family in the corpus separates nothing at all. Only
# nitrogen, rings and halogen move off the 0.250 floor, and only to ~0.31/~0.23.
N24_ARMS = ("n24",)
# --------------------------------------------------------------------------- #
# n25: n23, WITH THE REASON IN FRONT OF THE DECISION INSTEAD OF BEHIND IT.
#
# Everything n23 has -- the enriched table with all three cells, the per-edit block at
# cap 8 -- and n21's clause moved to the OTHER SIDE of the line it belongs to. That is
# the whole arm: n25 - n23 is what a sentence costs, and n25 - n21 is what the placement
# costs, on a clause built from the same facts under the same polarity filter.
#
# WHAT THE PROGRAMME ALREADY KNOWS ABOUT THIS SIDE OF THE LINE, and why the arm is still
# worth running. Every earlier arm that argued IN FRONT of an elimination lost badly --
# n15 0.334, n16 0.317, n19 0.291 against n8's 0.427 -- and the mechanism was visible in
# the spans: the edit the sentence named was the edit that went on 1.000 of joints, so
# the real decision moved OFF the drop token and ONTO "which property to talk about",
# taken earlier and from a narrower basis. n21 was built to dodge exactly that by
# emitting the index first. But those arms wrote a FREE sentence, chosen from a
# vocabulary that could name any edit for any reason; n25's line is pinned to one edit's
# own on-table facts and filtered for polarity, so the two failures are not the same
# experiment. This one prices the placement with everything else held still.
#
# THE LINE IS ITS OWN LINE, AND THAT IS NOT A STYLE CHOICE. `bucket_span._DROP` is
# anchored `^Dropping Edit (\d+); that leaves ` under re.M. A reason spliced onto the
# FRONT of that line makes `dropped_edit` return None and switches off drop_present,
# drop_match and drop_avoid_commit without a word in the log -- the same trap the n21
# header describes from the other direction. So the reason goes on the line above, which
# is also the form n15/n16 already use ("Across the band, ... -- and Edit 4 is the first
# to go." / "Dropping Edit 4; that leaves Edits 2 and 3.").
#
# FACT FIRST, EDIT SECOND. The sentence leads with the observation and names the edit
# after it, which is the house rule the generation arms are given in SEQ_HEAD ("NEVER
# OPEN WITH Edit"). It is also what makes the arm a real test: a line reading `Edit 2
# holds the least.` would put the index ahead of the fact and would be n21 with extra
# whitespace, since the index token is what the decision actually is.
N25_ARMS = ("n25",)
N25_FACTS = 8
# --------------------------------------------------------------------------- #
# n26: n25 WITH THE TWO ELIMINATIONS TAKEN OUT.
#
# Everything n25 has -- the table with all three cells, the per-edit block at cap 8, the
# reason line -- and then the span goes straight from the block to the commit. One
# sentence and one answer, no ladder.
#
# WHAT IT PRICES. The ladder is the most expensive thing on the page and nothing has
# isolated it against a span that also observes. Two measurements bracket it. n8, the
# bare frame with the eliminations and nothing written between them, scores 0.427
# against `noreason`'s 0.385, so writing them is worth +0.042 with no prose at all. And
# handing the eliminations to a trained arm FOR FREE, in the forced-prefix runs, was
# worth +0.2339 on n20 and +0.2333 on n24 -- by a distance the largest effect in the
# programme. n25 - n26 is the same quantity from the other side: what the arm loses by
# not writing them, with every observation left in place.
#
# THE PREDICTION IS NOT OBVIOUS, WHICH IS WHY IT IS WORTH RUNNING. Every arm on the
# curve satisfies `exact = (1 - P(drops the pick)) x P(commit = pick | pick survives)`
# with ex|dropped measured at 0.000 -- an edit this family sets aside is never taken
# back. So the ladder carries a term that can only HURT: two chances to throw the answer
# away irrevocably. n26 deletes that term. If the ladder's value were only that it
# rarely drops the right edit, n26 should come out AHEAD. It comes out behind only if
# writing the eliminations is doing search rather than risking the answer.
#
# NOTHING ELSE MOVES. Same records, same rounds, same table, same block, same 46-frame
# bank, same commit line. The two lines are removed AFTER the frame is built, so the
# round is still gated on having two eliminations to draw -- `build_prompt` returns None
# below two, and lifting that gate here would hand n26 a larger and easier corpus than
# every arm it is read against.
N26_ARMS = ("n26",)
# --------------------------------------------------------------------------- #
# n27: n25 WITHOUT THE PER-EDIT BLOCK.
#
# The table, the eliminations, the commit and every reason line exactly as n25 writes
# them -- and no `What sets each edit apart`. n25 - n27 is what PRINTING the block is
# worth, and it is the last of the three pieces to be priced on its own: n22 - n8 took
# the table's two extra cells (flat), n26 takes the ladder, this takes the block.
#
# WHAT IS ALREADY KNOWN, AND WHY IT IS WORTH ASKING AGAIN. Handed the cap-4 block
# PERFECTLY and for free, n20 went 0.3914 -> 0.3994; handed the same block with its rows
# attached to the WRONG edits it went to 0.4012, higher than the truthful one. The
# content was not read. But n20's block stood alone, and here it feeds the reason lines:
# the fact each line quotes is chosen with block membership as a tie-break, so the block
# is the working of a sentence the student can see the answer to. Printing your working
# is a different thing from being handed someone else's, and only this arm separates the
# two.
#
# THE SELECTION DOES NOT MOVE. `feat` is still computed and still breaks ties inside
# `n21_clause`, so n27 quotes the SAME fact n25 quotes on every joint -- one difference
# between the arms, not two. That tie-break stays derivable without the block, because
# block membership is the soleness test over rows the table still prints; what the
# student loses is having it worked out on the page, which is precisely the question.
N27_ARMS = ("n27",)
# The n25 FAMILY: everything true of n25's span is true of these, up to the one thing
# each removes. One name so a change to the arm cannot reach one and miss the others.
# --------------------------------------------------------------------------- #
# n28: THE REASON MAY ONLY QUOTE A FACT FROM THE EDIT'S OWN BLOCK ROW.
#
# n25 chose the fact from every on-table fact the edit had, with block membership only a
# tie-break -- `block_first=False`, put in because the block gate was making the COMMIT
# quote `adds the most carbon` (0.2555) in front of `holds heavy_atoms` (0.3297) on
# 0.7715 of rounds. Reading the rollouts back says that trade was the wrong way round.
#
# WHAT THE ROLLOUTS SAID. 0.6831 of n25's first-drop reasons quote a fact that is NOT in
# the block row, and such a sentence fits 3.25 edits on average -- it does not name one.
# Split by that:
#
#     quoted fact IS exclusive   0.3169 of joints   P(edit right) 0.4348   (1.00 edits fit)
#     quoted fact is shared      0.6831 of joints   P(edit right) 0.3496   (3.25 edits fit)
#
# and overall P(edit right | shape right) = 0.5087 against 0.5425 for a coin flip inside
# the set the sentence admits. The sentence narrows and then the model guesses. n28 makes
# the sentence NAME the edit: an exclusive fact is true of exactly one candidate, so
# choosing the sentence IS choosing the edit.
#
# THREE CASES, AND NO SHARED FACT IN ANY OF THEM.
#   plain    the block row has a fact on the right side of the floor -- quote it.
#   concede  the block row has facts but all of them point the other way -- grant the
#            strongest and decide anyway. Present on 0.2577 of drops and 0.0926 of
#            commits.
#   none     the block row is `no feature of its own here` (0.2877 of drops, 0.2705 of
#            commits). n25 fell through to a shared fact here; n28 says there is nothing
#            and names the edit anyway. A contentless line is honest and, unlike a shared
#            one, it does not point at three edits at once.
# n29 is n28 with the block hidden and n30 is n28 with the eliminations taken out --
# the same two cuts n27 and n26 make to n25, so the four arms read as a 2x2 and the
# effect of the exclusive-fact rule can be told apart from the effect of the page it
# sits on. In n29 the sentence still quotes an EXCLUSIVE fact, so it still names one
# edit; what goes is only the reader seeing the working.
N28_ARMS = ("n28",)
N29_ARMS = ("n29",)
N30_ARMS = ("n30",)
N35_ARMS = ("n35",)
N40_ARMS = ("n40", "n40f")          # MOL_INFO -> EDIT_INFO -> DROP1 -> DROP2 -> COMMIT
N41_ARMS = ("n41", "n41f")          # ablation 1: neither paragraph, the decisions alone
N42_ARMS = ("n42", "n42f")          # ablation 2: no elimination, straight to the commit
N43_ARMS = ("n43", "n43f")          # ablation 3: the eliminations, in a SEEDED order
N44_ARMS = ("n44", "n44f")
# n45f: the FG-filtered page with ONE elimination instead of two.
#   MOL_INFO -> EDIT_INFO -> DROP1 -> COMMIT, both decisions in n44's voice.
# The round is still GATED on having drawn two eliminations, so the record set is
# byte-for-byte the one n40-n44 are read against; only the printed page is shorter.
# Between n42 (no elimination at all) and n44 (two), this is the middle rung: it says
# whether the ACT of eliminating pays or whether it is the SECOND one that costs.
N45_ARMS = ("n45f",)
# n46f: n45f WITHOUT THE SUMMARY PARAGRAPH.
#
# The block is still BUILT -- the decisions pick their facts out of it exactly as
# n45f does -- it is simply not printed, the way `_NOBLOCK` does it for n27 and n29.
# One variable, and the reasons stay checkable: `n44_reasons` draws from rank-0 facts,
# which are the ones the ANALYSIS row above settles.
#
# WHY THE PARAGRAPH IS WORTH REMOVING. Measured on n40 at epoch 3 over 19,860 rows:
#   0.182 of rows carry MORE than the cap of four true facts, and which four survive
#   is a blake2b draw on the round key. Whole-row agreement on those rows is 0.071
#   against 0.785 on the rows that were not cut -- eleven times worse -- and 0.879 of
#   what the model writes there that the reference does not is TRUE, just not the four
#   that were drawn. So a fifth of the paragraph is a target the student cannot
#   reproduce, and the loss spends itself pushing at an arbitrary subset.
# What removing it should NOT be expected to do is move `exact`: SUMMARY's marginal
# GBM contribution is 0 (everything in it is derived from ANALYSIS), handing the model
# a correct one is worth -0.002, and v28 -- which never had one -- is still the best
# arm on the curve at 0.444.
N46_ARMS = ("n46f",)
# n47f: n46f with the decisions cut loose from the block as well.
#
# n46f stops PRINTING the block but still picks its reasons out of it -- the pool is
# filtered to the edit's own exclusive facts and the concession is required to be one
# of them. So the paragraph is gone and its content is not: every reason n46f gives is
# a SUMMARY fact. n47f drops that filter, and the reasons are drawn from the on-table
# pool at large, ranked on lift alone.
#
# WHAT THIS COSTS, and it is not nothing. `n21_clause`'s docstring is explicit: the
# block's row is what makes the quoted fact DISCRIMINATING, and a shared fact "fits
# 3.25 candidates and points at all of them" (the n28 header). Cut loose, a drop line
# may read `Edit 4 holds MR` on a round where all four hold MR. The arm is here to
# price exactly that: n46f - n47f is what the block is worth to the DECISIONS once it
# is no longer worth anything to the page.
N47_ARMS = ("n47f",)
_FREEREASON = N47_ARMS       # reasons ignore the block entirely
# n48: n22 WITH ONE ELIMINATION INSTEAD OF TWO -- the cell the ladder was missing.
#
# v28 is still the best arm on the curve at 0.444 and everything built after it sits
# below, but every step away from v28 moved two things at once:
#     v28   NEEDS + FG + the BAND CELLS ONLY, one elimination      0.444
#     n8    the same, TWO eliminations                             0.427
#     n24   two eliminations + the fg cell                         0.422
#     n22   two eliminations + the fg cell + the count cell        0.410
# So the 0.034 between v28 and n22 is the two cells AND the second elimination, mixed.
# n48 is v28's frame -- one elimination -- carrying n22's two extra cells: v28 - n48
# is the cells alone, n48 - n22 the second elimination alone. Worth separating,
# because the GBM puts the extractable signal in those cells at +0.002 (fg) and
# +0.011 (count) while the corpus charges 0.017 of accuracy for the pair.
N48_ARMS = ("n48",)
# n49 and n50 take the elimination out of the two ends of that ladder.
#   n49  n48 without the drop -- NEEDS + FG + the three-cell table, then the commit
#   n50  v28 without the drop -- NEEDS + FG + the BAND CELLS ONLY, then the commit
# n42 already showed that dropping the elimination is worth something on the prose
# side (0.432 against n40's 0.390), and the elimination is where the corpus throws
# the answer away: on n45f the first drop kills the committed edit on 0.199 of rounds
# and the recovery from that is 0 of 992. These two ask what the table is worth once
# that loss is off the board, at both cell counts.
N49_ARMS = ("n49",)
N50_ARMS = ("n50",)
# n51: n49 PLUS two observations the table does not carry, chosen by `qmax_gap` --
# the observation on which argmax(q) stands furthest from the other three -- and a
# closing line that names the commit without claiming the two readings point at it.
#
# WHY IT IS WORTH A RUN WHEN THE GBM SAYS ZERO. On the visible pool the rule scores
# COMMIT 0.4148 against a random 0.3728, and the permutation control -- the same rule
# with q shuffled inside the round -- gives 0.3709. So the GBM gain is q's ordering
# carried through the choice of observation, not the observations themselves, and
# `arms.py` already records that no wording carries q (q_recovery: r=0.833 on the value,
# 38.5% on the within-round argmax). Against that: the fg and count cells were priced
# at +0.013 by the same GBM and cost -0.017 in the corpus (n8 0.427 -> n22 0.410), so a
# GBM zero is not a corpus zero either. One run settles which way this one falls.
#
# THE POOL IS WHAT THE STUDENT COULD PRODUCE. arms.NOT_VISIBLE (c_prob*, site_*, ctx_*,
# cand_similarity) plus the tier-6 and RDKit demotions (c_snr_min, r_dsa, r_dqed,
# r_dlogp, r_dtpsa, the per-heavy normalisations) and the selector's own columns
# (c_rank, c_prob_margin, c_prob_z) are all out; 89 names remain, 17 of them varying
# across the four on an average round, and 46% of those are structural.
N51_ARMS = ("n51",)
# n52: n51's MATCHED CONTROL. Same pool, same dedup, same wording, same link -- the two
# observations are drawn at random instead of by qmax_gap. n51 - n52 is the value of the
# selection with "having two observations at all" subtracted out, which is the only part
# the GBM and its permutation control disagree about.
N52_ARMS = ("n52",)
# n53/n54: THE OBSERVATIONS WITH NOTHING TO BYPASS. n51's page is MOL_INFO (the
# out-of-range line and the group line) plus EDIT_INFO (the three-cell table) plus the
# two observations plus the commit, and the two observations are the only part that
# differs from n49. n53 keeps ONLY the observations and the commit; n54 is the same
# strip applied to n52's random pair.
#
# WHY THIS PAIR IS WORTH A RUN. On the n51/n52 cycle the commit tracked the argmax of
# the page's first observation with lift +0.148 on n51 and +0.003 on n52 -- n52 emitted
# the two lines and routed the decision around them, through the table it shares with
# n49, while n51 followed whatever axis it had written. n51's loss then came from
# writing the WRONG axis and following it faithfully: stratified by whether the free
# rollout reproduced the reference pair, free equals forced on the rounds it got right
# (0.521 vs 0.517) and the whole gap sits on the rounds it did not (0.398 vs 0.419).
# Taking the table away removes the thing n52 bypassed TO. n54 then has nothing to
# decide on but two random numbers, so it prices the floor of the stripped page, and
# n53 - n54 is the selection's value with the bypass route closed.
#
# THE OBSERVATIONS ARE THE SAME ONES, not a fresh draw: n53 reads `n51_obs.pkl` and n54
# reads `n52_obs.pkl` (see `_OBS_SRC`). So n51 - n53 is the page removal with the pair
# held fixed, and n53 - n54 is the selection under the stripped page -- neither contrast
# moves two things at once.
N53_ARMS = ("n53",)
N54_ARMS = ("n54",)
# n55/n56/n57: THE POOL n51 WAS SELECTED FROM WAS WRONG, AND THESE PRICE THE FIX.
# n51's observations came out of `evidence_full`, whose `table4` is the A-gate's top 8
# rows and carries no `targets` -- so the property filter emptied on every round and
# every `r_dmean__*`/`r_dstd__*` died silently (commit 0843ff2). MEASURED on the study
# test split: table4 is 8 rows, 44.7% of them fail `visible()`, and the pool the
# selection actually saw was 4.43 columns against the 34.4 `base_cols` gives. The main
# corpora were rebuilt on `base_cols`; the study arms never were, so every n51 number
# in the README was measured on the broken pool.
#
#   n55  n51 exactly, over the FIXED pool          n51 - n55 = the bug, priced
#   n56  n55 with qmax_gap divided by the column's spread across the candidates
#   n57  n55 with the `c_*` box arithmetic out of the pool
#
# WHY n56 IS IN THE SET. The fix has a side effect nobody priced: qmax_gap compares raw
# values, so a column in daltons beats a column in counts, and `r_dmw` is the first
# observation on 36.4% of main-corpus rounds against 0 of the study's 110,158. n56 asks
# whether that is a signal or a unit.
#
# WHY n57 IS IN THE SET. The free heldout rollout writes a correct value vector for
# 99.0% of structural axes and 64.7% of `c_*` ones (`c_cos` 1 of 59). n57 takes the
# family the student demonstrably cannot reproduce out of the pool.
#
# All three share n51's page, wording, dedup, record set and hyperparameters, so each
# pair differs in one thing. They select their own observations -- unlike n53/n54,
# which reuse n51's and n52's -- because the selection IS what they change.
# n58: n57's POOL, DRAWN AT RANDOM -- the matched control that was missing. n52 is
# random on n51's pool (the 4.43-column `evidence_full` one: 73 axes, `r_dmw` 0.0000,
# no property delta at all), so n57 read against n52 moves the pool AND the rule and
# cannot say anything about the rule. n57 - n58 is the selection on the pool n57 has.
# n59: n57 with a THIRD observation. The commit follows the first axis the span names
# and n57's whole free-forced gap sits on the rounds where that axis is wrong (on the
# matched ones free == forced, p=0.715), so this asks whether the span wants more
# evidence in front of the commit or a better first pick.
# n60: n57 WITH THE VALUES PUNCTUATED, and nothing else. `Edit 1 141, Edit 2 126`
# runs a label straight into its number with no separator, so on an integer column
# `Edit 1 3` reads as one token as easily as two. n60 writes `Edit 1 = 141` instead.
#
# IT READS n57's PICKLE (see `_OBS_SRC`), so the axes and the values are the SAME
# OBJECT, round for round -- the way n53 reuses n51's. n57 - n60 is therefore the
# punctuation alone, which is the only way to find out whether the format costs
# anything: n57 already transcribes 0.9522 of its value vectors correctly, so the
# headroom is small and worth measuring before the whole set is rebuilt on a guess.
# n62: n60 PLUS THE ONE LINK THE PAGE NEVER MAKES -- the table.
#
# MEASURED on n60's 7,957 reference spans: on 40.5% of rounds some other edit reads
# strictly better in the three-cell table (>= holds and <= misses, one strict) than
# the one the span commits to, and the span never mentions it. On 65.4% of those the
# first observation does put the commit at a unique extreme the preferred edit is not
# at -- so the material for "the table says Y, but on Z my pick stands apart" is
# already there, in 26.5% of all rounds, and is thrown away.
#
# WHY THIS LINK AND NOT A BETTER OBSERVATION. The observation vocabulary is
# informationally dead: a GBM over the whole `noagg` pool tops out at 0.4646 on this
# metric, BELOW noreason's ~0.497, while the arms live at 0.52-0.56 and drop to
# 0.52-0.53 when the table is removed (n53/n54). The table is where the signal is and
# the observation floats free of it. n62 makes the span restate the table before
# overriding it, the first time any arm reasons about the part that carries accuracy.
#
# IT CHANGES 26.5% OF ROUNDS AND NOTHING ELSE. Where the table does not contradict the
# commit, or where the observation cannot honestly overturn it, n62 renders exactly
# what n60 does -- an "override" with no ground would be a worse sentence than silence.
N55_ARMS = ("n55", "n56", "n57", "n58", "n59", "n60", "n62", "n61", "n63",
            "n64", "n65", "n66", "n67", "n68", "n69", "n70", "n71", "n72",
            "n73", "n74", "n75", "n76", "n77", "n78")
# The arms that name the table's preference before the observation that overrides it.
_TABLE_ARMS = ("n62",)
# n61: n60, PLUS A TABLE VERDICT THAT CANNOT LEAK, plus the two corrections.
#
# n62's override names the committed edit -- `Edit {pick+1} stands apart` -- and the
# forced pass feeds everything up to `Taking Edit N:`, so the answer sits in the
# prompt. MEASURED: n62 forced scores 0.9811 on its 739 override rounds against n60's
# 0.4682 on the SAME rounds, +0.42, while n62 FREE falls to 0.2882 there against n60's
# 0.3748. It is the circularity the README's "Why this metric" section exists to avoid,
# and it also teaches a shortcut that does not exist at inference.
#
# n61's line is a pure function of the TABLE, which is already in the prompt: it never
# reads `pick`. So it cannot leak by construction, and a model at inference can compute
# it for itself. The verdict ranks the edits on the four cells, in this order:
#
#     most `holds` -> fewest `misses` -> fewest `at risk` -> most `within reach`
#
# Everything tied on all four is named together; if that is every edit, the line says
# the table does not separate them rather than pretending it favours something.
_TABLE2_ARMS = ("n61", "n63")   # n64/n65 describe instead -- see _DESC_ARMS
# (cell, sign for "better is smaller", singular reason, plural reason). The reason names
# the cell that DECIDED it, and says so cumulatively, because "misses the least" alone
# would be false when something else misses as few but holds more.
_VERDICT_KEYS = (
    ("holds", -1, "it holds the most", "they hold the most"),
    ("misses", +1, "it ties on holds and misses the least",
                   "they tie on holds and miss the least"),
    ("at risk", +1, "it ties on holds and misses, and has the least at risk",
                    "they tie on holds and misses, and have the least at risk"),
    ("within reach", -1,
     "it ties on holds, misses and risk, and has the most within reach",
     "they tie on holds, misses and risk, and have the most within reach"))


# n63: THE BAND CELL SHOWS ITS ARITHMETIC.
#
# n61 prints the CONCLUSION grouped by slot -- `holds MW, logP | at risk BBBP`. The
# student has to have done `now + avg +- std` against the box in its head, and can only
# be scored on the answer. MEASURED on n61 at epoch 2, per band assignment: 0.9616
# overall, and the error is ENTIRELY on the properties whose std is non-zero --
# rings_total and HBD 1.000, HBA 0.9998, rotB 0.9989, heavy_atoms 0.9871 against logD
# 0.9154, BBBP 0.9301, logS 0.9309, logP 0.9330, QED 0.9387, Mutag 0.9455. A count is
# read off the two SMILES; a predicted shift has to be recalled and then added. That
# split is the signature of arithmetic the model is guessing rather than doing, and
# 75.9% of the errors are one slot off -- `holds` <-> `at risk` above all, which is
# exactly the pair the sigma decides.
#
# So n63 writes the interval out, per property, and puts the label after it. IT ADDS NO
# INFORMATION: `props` is in the analyze_properties result, the box is in the user turn
# and avg/std are in the suggest_edits result the row already summarises. It adds a
# step. (n62's lesson is the other side of this -- an added line that names the answer
# is a leak; an added line the student can compute is not.)
#
# THREE DECIMALS, WHICH IS LOSSLESS. Every input is at most 3 dp in the corpus
# (measured over 143,804 cells: 0/1/2/3 only), so `round(now + avg +- std, 3)` removes
# binary dust and nothing else -- there is no rounding decision and no round-off that
# could put the printed interval at odds with the printed label. Trailing zeros go, so
# a zero-std cell prints one number: `rings_total 2 holds`.
#
# THE LABEL COMES FROM `bs.classify`, NOT FROM THESE NUMBERS. One authority, so n63's
# slots are the same slots n61 prints and the arms stay paired: VERIFIED, the verdict
# line is byte-identical on all 7,957 test spans and the labels match `classify` on all
# 31,908 rows. 132 cells in 251,496 (0.05%) land exactly ON a bound in decimal and are
# then decided by float dust -- 0.89 - 0.362 - 0.097 is 0.431 exactly, and `>` against
# a 0.431 bound turns on the 0.43100000000000005 the double holds -- so there the
# printed interval touches the bound and the label is the one `classify` gave. Making
# `classify` decimal-exact would move n61's labels too and cost the paired contrast.
# n64/n65 carry the interval cells too: the band label is the thing the observation
# lines are then read against, so the arm that finally asks "does the SELECTION matter"
# should not also be the one that hides the band's arithmetic. `_cell_counts` already
# reads both row shapes, so `_table_desc` describes the interval table unchanged.
_INTERVAL_ARMS = ("n63", "n64", "n65", "n66", "n67", "n68", "n69")
_BAND_WORDS = ("holds", "at risk", "within reach", "misses")
# Longest first: `at risk` has to be tested before `risk` would ever be, and a chunk
# ending in `within reach` must not be read as ending in `reach`.
_BAND_LONGEST = ("within reach", "at risk", "holds", "misses")


def _n63_num(x, nd=3):
    s = f"{round(x, nd):.{nd}f}".rstrip("0").rstrip(".")
    return "0" if s in ("", "-0") else s


def _interval_row(rnd, i, k=bs.K):
    """n63's band cells for one edit: the predicted interval, then `classify`'s label.

    Target order, not slot order, so a property sits in the same column down the table.
    The `*` stays on the PROPERTY as it does in n61 -- it marks a property this edit
    pushes out of a range it was already in, which is a fact about the property and not
    about the word `misses` -- or the two arms' regression marks would not be
    comparable.
    """
    h, r, w, m = bs.classify(rnd, rnd["candidates"][i], k)
    lab = {}
    for xs, nm in ((h, "holds"), (r, "at risk"), (w, "within reach"), (m, "misses")):
        for x in xs:
            lab[str(x).rstrip("*")] = (nm, str(x).endswith("*"))
    d = rnd["candidates"][i].get("delta") or {}
    out = []
    for p in rnd["targets"]:
        if p not in lab:                      # classify ranges over every target
            continue
        nm, star = lab[p]
        dp = d.get(p) or {}
        mid = float(rnd["props"].get(p) or 0.0) + float(dp.get("avg") or 0.0)
        sd = abs(float(dp.get("std") or 0.0)) * k
        a, b = _n63_num(mid - sd), _n63_num(mid + sd)
        head = p + ("*" if star else "")
        out.append(f"{head} {a} {nm}" if a == b else f"{head} {a} to {b} {nm}")
    return " | ".join(out)


def _interval_table(rnd, lines):
    """Rewrite every band row in place. Called BEFORE the fg and count cells are
    appended, so `m.group(2)` is the band cells and nothing else."""
    for j, ln in enumerate(lines):
        mo = _TBL_ROW.match(ln)
        if mo:
            lines[j] = f"  Edit {mo.group(1)} | " + _interval_row(rnd, int(mo.group(1)) - 1)
    return lines


def _cell_counts(lines):
    """-> {edit: (n_holds, n_misses, n_at_risk, n_within_reach)} off the rendered table.

    Reads both row shapes. n49..n62 group by slot and put the word FIRST
    (`holds MW, logP`); n63 prints one chunk per property and puts it LAST
    (`MW 310.091 holds`). No property is named after a slot word and no slot word is
    followed by a number, so the shape is unambiguous -- and getting this wrong would
    not raise, it would silently count zero and hand `_table_verdict` a flat table.
    """
    out = {}
    for ln in lines or ():
        m = _TBL_ROW.match(ln)
        if not m:
            continue
        cell = m.group(2)
        chunks = [c.strip() for c in cell.split("|")]
        tail = [c for c in chunks
                if any(c == b or c.endswith(" " + b) for b in _BAND_WORDS)]
        if tail:
            c2 = dict.fromkeys(_BAND_WORDS, 0)
            for c in tail:
                for b in _BAND_LONGEST:
                    if c == b or c.endswith(" " + b):
                        c2[b] += 1
                        break
            out[int(m.group(1)) - 1] = tuple(c2[k[0]] for k in _VERDICT_KEYS)
            continue
        def _n(word):
            mm = _re.search(word + r" ([^|]*)", cell)
            return len([x for x in mm.group(1).split(",") if x.strip()]) if mm else 0
        out[int(m.group(1)) - 1] = tuple(_n(k[0]) for k in _VERDICT_KEYS)
    return out


def _edit_list(idx):
    names = [f"Edit {i + 1}" for i in sorted(idx)]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


# n64/n65: THE TABLE LINE IS A DESCRIPTION, AND THE CONCESSION MOVES INTO THE CLOSER.
#
# `The table favours Edit X since ...` is arithmetically right -- the student reproduces
# it from its own table 0.9938 of the time -- but the IMPLICATURE is false. "favours"
# says the table decides, and MEASURED: following it is worth 0.4500 on-path, against
# 0.4536 for the constant "always Edit 4" and 0.9781 for the commit the span then takes.
# The span goes on to take something else on 54.8% of the rounds that have a verdict,
# with no sentence in between. So the line states the fact and stops claiming a verdict:
#
#     The table favours Edit 2 since it holds the most.   ->   Edit 2 holds the most.
#
# THE CONCESSION GOES IN THE CLOSER LINE AND NOWHERE ELSE. `_COMMIT_RE` cuts the forced
# prefix at the EARLIEST announcement, and `.*?\bI'll go with Edit \d+\.` is anchored at
# `^`, so the whole closer line -- concession included -- is outside the forced prompt.
# A separate "but I'll take Edit 2" line ABOVE the closer would sit inside it, which is
# exactly how n62 leaked (forced 0.9811 against n60's 0.4682 on the same 739 rounds).
_DESC_ARMS = ("n64", "n65", "n66", "n67", "n68", "n69")
# NO "WEIGHING", because no sentence above performed one. The two observations quote
# values and never say which direction helps; MEASURED, the commit is a unique extreme
# of the first observation on 0.365 of rounds. "Weighing all of that together" and "on
# balance" both assert a derivation the page does not contain, so the closer states the
# choice and claims nothing about how it was reached.
_CLOSE_WITH = "I'll go with Edit {n}."
_CLOSE_AGAINST = "The table leans the other way, but I'll go with Edit {n}."
# ---------- (1) the seam between the table line and the first observation
# Before this the two sat side by side with no connective at all -- a flat fact about
# the table, then a flat observation, and nothing saying whether the second supports,
# challenges or merely follows the first. Both lead-ins below are true on every round
# they fire on: where the table named somebody the tally is genuinely not the whole of
# it (the span goes on to take something else on 54.8% of them), and where it named
# nobody the observation genuinely is what is left.
_DESC_TAIL = ", but the tally is not the whole of it."
_OBS1_AFTER_NAMED = "The thing to look at here is "
_OBS1_AFTER_FLAT = "So the thing to look at here is "


def _table_desc(lines):
    """(the descriptive line, the set the cascade puts first) -- or (None, None).

    Same cascade as `_table_verdict`, same reason strings; only the frame changes, so
    n61 and n64 describe the same edits and differ in what the sentence claims about
    them. The set comes back because the CLOSER needs it: a concession is only honest
    where the commit is actually outside it.
    """
    S = _cell_counts(lines)
    if len(S) < 2:
        return None, None
    key = lambda i: tuple(k[1] * S[i][j] for j, k in enumerate(_VERDICT_KEYS))
    best = min(key(i) for i in S)
    win = [i for i in S if key(i) == best]
    if len(win) == len(S):
        h, ms = S[win[0]][0], S[win[0]][1]
        return (f"Nothing in the table separates them: all {len(S)} hold {h} "
                f"and miss {ms}."), set(win)
    why = None
    alive = set(S)
    for j, (_w, sign, sg, pl) in enumerate(_VERDICT_KEYS):
        wv = S[win[0]][j]
        rest = [i for i in alive if i not in win]
        if rest and all(sign * S[i][j] > sign * wv for i in rest):
            why = pl if len(win) > 1 else sg
            break
        alive = {i for i in alive if S[i][j] == wv}
    if why is None:
        return None, None
    # the reason strings are written to follow "it"/"they"; as a bare description the
    # subject is the edit list itself, so the pronoun goes.
    body = why.split(" ", 1)[1] if why.split(" ", 1)[0] in ("it", "they") else why
    return f"{_edit_list(win)} {body}.", set(win)


def _table_verdict(lines):
    """The table's own verdict, read off the table and NOTHING else. Never sees `pick`."""
    S = _cell_counts(lines)
    if len(S) < 2:
        return None
    key = lambda i: tuple(k[1] * S[i][j] for j, k in enumerate(_VERDICT_KEYS))
    best = min(key(i) for i in S)
    win = [i for i in S if key(i) == best]
    if len(win) == len(S):
        h, ms = S[win[0]][0], S[win[0]][1]
        return f"The table does not separate them: all {len(S)} hold {h} and miss {ms}."
    # WHICH CELL DECIDED IT. Only the edits still tied on every EARLIER key are in
    # contention, so the test walks the cascade instead of comparing against edits that
    # were already out -- otherwise an edit that lost on `holds` but happens to miss as
    # few blocks `misses` from being named, and the reason given is the wrong cell.
    why = None
    alive = set(S)
    for j, (_w, sign, sg, pl) in enumerate(_VERDICT_KEYS):
        wv = S[win[0]][j]
        rest = [i for i in alive if i not in win]
        if rest and all(sign * S[i][j] > sign * wv for i in rest):
            why = pl if len(win) > 1 else sg
            break
        alive = {i for i in alive if S[i][j] == wv}
    if why is None:                       # unreachable: `win == S` is handled above
        return None
    return f"The table favours {_edit_list(win)} since {why}."
# THE TWO CORRECTIONS, gated so n59 and n60 -- already trained -- re-render byte for
# byte if anyone rebuilds them.
#
# H. `adds -18`. `r_dmw` and the net-change columns go negative, and an additive verb
#    on a negative number is neither English nor true. MEASURED: 8.1% of n62's rounds
#    carried at least one. The verb flips and the number drops its sign, PER EDIT, so a
#    mixed column reads "Edit 1 removes 18, Edit 2 adds 0". Only the count verbs flip;
#    `predicts` / `changes it by` / `shifts it by` carry the sign on purpose.
#
# D. "The other side of it is". MEASURED over 7,821 spans: the two observations rank
#    the candidates the SAME way on 72.4% of rounds (mean rho +0.34, median +0.47) and
#    genuinely oppositely on 12.4%. The one discourse marker the span owns asserts a
#    tension that is not there in seven rounds out of ten. The dedup already thresholds
#    |rho| at 0.8, so the number is to hand -- the connective just has to read it.
_FIXED_ARMS = ("n62", "n61", "n63", "n64", "n65", "n66", "n67", "n68", "n69")
_INVERSE_VERB = {"adds": "removes", "removes": "adds",
                 "creates": "destroys", "destroys": "creates"}


def _pearson(a, b):
    """Correlation across the candidates. Pure python: this file imports no numpy."""
    n = len(a)
    if n < 2 or len(b) != n:
        return 0.0
    ma, mb = sum(a) / n, sum(b) / n
    da = [x - ma for x in a]
    db = [x - mb for x in b]
    na = sum(x * x for x in da) ** 0.5
    nb = sum(x * x for x in db) ** 0.5
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return sum(x * y for x, y in zip(da, db)) / (na * nb)
# The arms whose numeric observation values are written as a clause per edit --
# `Edit 1 adds 3, Edit 2 adds 3, Edit 3 adds 2 and Edit 4 adds 0` -- instead of the
# label run `Edit 1 3, Edit 2 3, ...`. The verb comes from `_N51_VERB`. The boolean
# branch is already a clause and does not change, or n57 - n60 would move two things.
_SENT_ARMS = ("n60", "n62", "n61", "n63", "n64", "n65", "n66", "n67", "n68", "n69",
              # n71: n57 WITH ONE THING CHANGED -- the values read as clauses.
              # `Edit 1 0, Edit 2 1` becomes `Edit 1 adds 0 and Edit 2 adds 1`. Same
              # pool, same criterion, same dedup, same table, same connective, same
              # closer, and the observations come out of n57's own pickle, so the two
              # arms cannot differ in which columns were chosen.
              #
              # n72 is n70's selection with n71's wording -- gap_z AND the clause form,
              # rendered from n70's pickle. The four together are a 2x2: n57 raw/run,
              # n70 z/run, n71 raw/clause, n72 z/clause, so the criterion and the
              # wording can be read apart and together.
              #
              # n73 is the MATCHED RANDOM cell of that page: n58's pickle, which is the
              # uniform draw over the same pool with the same dedup, rendered in the
              # same clause form. n72 - n73 is the criterion with the wording held
              # fixed, and n71 - n73 is the same question for gap_raw.
              #
              # n74 IS THAT PAGE WITH THE EVIDENCE TAKEN OUT: the same NEEDS line, the
              # same group line, the same band table and the same closer, with no
              # observation between them. It sits in this tuple and in `_OBS_SRC` below
              # so that everything upstream of the PRINTING -- the pool, the dedup, the
              # selection, the gates -- runs exactly as it does for n72, and the arms
              # keep one record set. Only the two lines are withheld, so `n72 - n74`
              # and `n73 - n74` price the evidence block itself rather than the page.
              #
              # n75 IS n72 WITH THE CRITERION KEYED ON THE ANSWER: `n51_obs._REF_PICK`
              # hands `select` a one-hot at the reference edit, so the same rule over
              # the same pool now asks which observation separates the BEAM-SEARCH
              # PICK. Everything printed is otherwise n72's. `n72 - n75` is therefore
              # the STRENGTH of the teacher signal with the page held fixed: measured
              # on the test split, argmax of n72's first observation lands on a
              # shortest path 0.591 of the time and n75's 0.970.
              #
              # n76/n77/n78 ARE THE POOL B FAMILY, and the first arms whose criterion
              # reads no `q` at all. n76 scores a column by A's own gate, n77 by how
              # much that gate MOVES across the candidates, and n78 is the uniform draw
              # over the same pool -- the matched control, from THE SAME POOL, which is
              # the thing n52-vs-n57 got wrong. The pool keeps the columns that are flat
              # across the candidates, so an observation may now say that an axis does
              # NOT separate the edits.
              "n71", "n72", "n73", "n74", "n75", "n76", "n77", "n78")
# Arms that flip the verb rather than print `adds -18`. `_FIXED_ARMS` implies it; this
# is for an arm that wants the flip without the pearson connective.
_INVVERB_ARMS = ("n71", "n72", "n73", "n74", "n75", "n76", "n77", "n78")   # n61/n62/n63 inherit n60's clause form
# WHICH PICKLE AN ARM'S OBSERVATIONS COME FROM. n53/n54 do not select their own; they
# reuse n51's and n52's so the stripped arms are paired with the full-page ones round by
# round. `n51_obs.py` writes only `n51_obs.pkl` and `n52_obs.pkl` and needs no new run.
_OBS_SRC = {"n51": "n51", "n52": "n52", "n53": "n51", "n54": "n52",
            "n55": "n55", "n56": "n56", "n57": "n57",
            "n58": "n58", "n59": "n59", "n60": "n57", "n62": "n57",
            "n61": "n57", "n63": "n57",
            # n64 SELECTS ITS OWN, because the restriction IS the arm: the pool is
            # n57's minus every column that is not about a property the round says is
            # out of range. n61 - n64 is that restriction and nothing else -- same
            # page, same verdict line, same clause form, same connectives.
            "n64": "n64", "n65": "n65", "n66": "n66", "n67": "n67", "n68": "n68", "n69": "n69",
            # n70 selects its own: n57's pool and dedup, scored with mode="z"
            "n70": "n70",
            # n71 RENDERS FROM n57's PICKLE. Nothing about the selection changes, so
            # there is no second pickle to drift.
            "n71": "n57",
            # n72 RENDERS FROM n70's PICKLE: same pool, same dedup, same gap_z.
            "n72": "n70",
            # n73 RENDERS FROM n58's PICKLE -- n57's pool and dedup, drawn at random.
            "n73": "n58",
            # n74 READS n70's PICKLE AND PRINTS NONE OF IT. The entry is not dead: the
            # selection still runs, so n74 keeps and drops exactly the rounds n72 does.
            "n74": "n70",
            # n75 SELECTS ITS OWN, because the criterion IS the arm -- n70's rule and
            # n70's pool with the reference edit in place of argmax(q).
            "n75": "n75",
            # Each POOL B arm selects its own: the pool differs from every earlier
            # arm's, so none of the existing pickles can be reused.
            "n76": "n76", "n77": "n77", "n78": "n78"}
# THE PAGE WITHOUT THE EVIDENCE. `n51_lines` returns the observation lines followed by
# the closer, so the closer is its LAST element and `[-1:]` keeps the page's last beat
# while withholding the readings. KEEPING THE CLOSER IS THE POINT: it is the line
# `inloop_eval._COMMIT_RE` cuts the forced prefix at, so n74 keeps a gold-prefix
# condition meaning what n72's means, and `n72 - n74` stays ONE change.
_NOOBS = ("n74",)
# The arms whose page is the observations and the commit, and nothing else.
_OBSONLY = N53_ARMS + N54_ARMS
_NOCELLS = N50_ARMS          # the band row carries no fg cell and no count cell
_NOSUMMARY = N46_ARMS + N47_ARMS   # the block is built and used, but not printed
_ONEDROP = (N45_ARMS + N46_ARMS + N47_ARMS + N48_ARMS)   # keep the first, drop the second
_N40F = (N40_ARMS + N41_ARMS + N42_ARMS + N43_ARMS + N44_ARMS + N45_ARMS
         + N46_ARMS + N47_ARMS)
# The same five arms with `FG_DROP` applied to every group name on the page.
_FGF = ("n40f", "n41f", "n42f", "n43f", "n44f", "n45f", "n46f", "n47f")
# THE FOUR SWITCHES, named once. Every arm below carries n25's page; each of the three
# after it turns exactly one thing off or swaps exactly one rule, and a change that
# reached one member and missed another would be invisible in the result.
_N25F = (N25_ARMS + N26_ARMS + N27_ARMS + N28_ARMS + N29_ARMS + N30_ARMS
         + N35_ARMS)
_NOBLOCK = N27_ARMS + N29_ARMS + _N40F  # the block is built but not printed
# n40 is in `_NOBLOCK` for a different reason than n27/n29: it does not suppress the
# block, it RE-SAYS it as the second half of EDIT_INFO. The rows are still built,
# because every reason the decisions quote is drawn from them.
_NODROP = (N26_ARMS + N30_ARMS + N42_ARMS + N49_ARMS + N50_ARMS + N51_ARMS
           + N52_ARMS + N53_ARMS + N54_ARMS + N55_ARMS)   # no elimination
_N28SEL = N28_ARMS + N29_ARMS + N30_ARMS   # exclusive facts only, never a shared one

# n35: THE REASON IS AN AXIS NAME AND THE EDITS THAT CARRY IT, NOTHING ELSE.
#
# n28 writes one sentence about one edit, so the sentence IS the choice: the model was
# measured naming the corpus's opening token at .466 against a .482 random-emission
# baseline, and when it named the wrong one the round was lost at .084 -- a wrong first
# token forecloses the answer because the sentence has to be true of the edit the
# decision line then names. n35 breaks that lock. The line names one of five AXES and
# then lists every live edit whose block row carries a fact of that axis, so a wrong
# axis still leaves the answer in the list 45.7% of the time instead of never.
#
# The axis is the axis of the SAME fact n28 would have quoted -- `n28_pick`, unchanged --
# so the two arms differ in the shape of the line and in nothing else. The listing is a
# read-off: every name in it can be checked against the block printed three lines above,
# which is why the block stays.
#
# A round where every live edit shares the axis prints the whole live set and narrows
# nothing. That is left in (4.9% of drop1, 12.5% of drop2, 30.2% of commits): the
# alternative was to fall back to whichever of the target's axes has the smallest group,
# and that biases the named axis toward small groups in a way the student can read off
# the block and follow without ever choosing an edit -- n30's empty-row shortcut again.
N35_NONE = "None"                       # the block row is empty; not an axis but a slot
_N35_AXIS = {
    "holds {p}": "Band",
    "brings {p} back within reach": "Band",
    "is left exposed on {p}": "Band",
    "gives {p} up": "Band",
    "puts {p} out, though not out of reach": "Band",
    "breaks {p}, which the molecule satisfies now": "Band",
    "puts on {g}, which the molecule does not have yet": "Group",
    "puts on no {g}": "Group",
    "takes off {g}": "Group",
    "puts on {g}": "Group",
    "adds no {k}": "Count",
    "adds the fewest {k}": "Count",
    "adds the most {k}": "Count",
    "holds the most": "Tally",
    "gives up the least": "Tally",
    "holds the least": "Tally",
    "gives up the most": "Tally",
}
# Band/Group/Count/Tally/None take five DISTINCT first tokens under the Qwen3 vocabulary
# ('Band', 'Group', 'Count', 'T', 'None'), which is the whole point of naming them: the
# axis is decided at token one and five ways is as coarse as the line ever gets.


def n35_axes(molfg, unp, F, feat, i):
    """The axes present on edit i's PRINTED block row -- a set, not one value.

    An edit averages 1.79 of them; half the rows carry two or more. That is why the
    list can hold an edit the axis is not that edit's own max-lift choice.
    """
    out = {_N35_AXIS[_n25_shape(k, molfg, unp)[0]] for k, v in F[i].items()
           if _n20_rank(k, molfg)[0] == 0 and v[0] in feat[i]}
    return out or {N35_NONE}


def _n35_list(idxs) -> str:
    """`Edit 3` / `Edits 1 and 3` -- the plural the elimination lines already use.

    `_fmt_edits` repeats the word ("Edit 1 and Edit 4"), which reads as two sentences
    joined and does not match "that leaves Edits 2, 3 and 4" three lines down.
    """
    n = [str(i + 1) for i in sorted(idxs)]
    if len(n) == 1:
        return f"Edit {n[0]}"
    return "Edits " + ", ".join(n[:-1]) + " and " + n[-1]


def n35_line(rnd, molfg, unp, F, feat, e, side, live):
    """`Band: Edits 1 and 3.` -- (the line, the axis, the listed edits)."""
    kb, _mode = n28_pick(rnd, molfg, unp, F, feat, e, side)
    ax = N35_NONE if kb is None else _N35_AXIS[_n25_shape(kb, molfg, unp)[0]]
    # `| {e}` is belt and braces: the target's own row is what the axis came off, so it
    # is already in. It matters only if `feat` and `n28_pick` ever disagree, and a list
    # that omitted the edit the next line names would be a contradiction, not a hint.
    grp = sorted({i for i in live if ax in n35_axes(molfg, unp, F, feat, i)} | {e})
    return f"{ax}: {_n35_list(grp)}.", ax, grp


# --------------------------------------------------------------------------- #
# n40: THE SAME EVIDENCE, IN PROSE, WITH THE DECISION DELIBERATED AND THEN GIVEN.
#
# Nothing new is shown and nothing is taken away. n40 is n21's corpus -- the same rounds,
# the same q-ordered eliminations, the same cap-4 block, the same on-table facts -- laid
# out as four paragraphs instead of a table:
#
#   MOLECULE_INFO   NEEDS and FG in two sentences
#   EDIT_INFO       ANALYSIS as one sentence per edit, then SUMMARY as one sentence
#   DROP1/2/COMMIT  the live pool, then the choice, then one to three reasons
#
# WHY THE POOL COMES FIRST AND THE REASON LAST. Every arm in this programme that put its
# reason IN FRONT of the index lost -- n15 0.334, n16 0.317, n19 0.291, n25 0.347,
# n28 0.354 -- against 0.41-0.42 for the arms that emit the index first. The measured
# mechanism is that the reason's FIRST TOKEN becomes the decision: in n28 the sentence
# names exactly one edit, the model is 1.000 self-consistent with it, and a wrong first
# token forecloses the answer (P(exact | first token wrong) = 0.084, recovery 0/2157).
# So the deliberative preamble here is written to name NO EDIT INDIVIDUALLY: `Of Edits 1,
# 2, 3 and 4, weighing all of it together, ...` is the live set, which is a fact about
# the page above it and not a choice. The first token that commits to anything is the
# digit in `I set Edit 4 aside`, exactly as in n21.
#
# ONE TO THREE REASONS, AND THE FIRST ONE IS n21'S. The count is drawn per decision from
# a hash of (round, joint), so the length of the justification carries nothing about the
# round; the pool has two or more usable facts on 0.65 of eliminations and 0.77 of
# commits, so the draw is rarely clipped. The FIRST reason is `n21_clause`'s own pick --
# the on-table fact furthest from the 0.250 floor on the right side -- so n40's clause is
# n21's clause plus zero to two seeded extras, and the two arms stay readable against
# each other. The extras sit AFTER the index and so cannot foreclose anything.
#
# THE ANCHORS MOVED, AND THEY MOVED IN `bucket_span` TOO. `_DROP`, `_N21_CL` and
# `_N25_LEAD` are matched against this family's wording as an ALTERNATION next to the old
# literal, never as a replacement -- see the note over `_DROP`. Left alone they would
# have reported n40 as an arm with no eliminations and no clauses, silently.
N40_NUMW = ("no", "one", "two", "three", "four", "five", "six", "seven", "eight",
            "nine", "ten", "eleven", "twelve")
N40_KMAX = 3                             # reasons per decision, 1..N40_KMAX
# The block's phrases are written to stand alone in a bulleted row. Two of them carry a
# trailing `which ...` clause that reads as a comma pile-up once three are joined into a
# sentence, so those get a short form. Same fact, same polarity, one clause fewer.
_N40_SHORT = (
    (_re.compile(r"^puts on (.*), which the molecule does not have yet$"),
     r"newly puts on \1"),
    (_re.compile(r"^breaks (.*), which the molecule satisfies now$"),
     r"breaks the already-satisfied \1"),
    (_re.compile(r"^puts (.*) out, though not out of reach$"),
     r"pushes \1 out but not out of reach"),
)
_N40_BAND = ("holds {}", "puts {} at risk", "brings {} back within reach", "misses {}")
_N40_ORD = {"drop1": "first", "drop2": "next"}


def _n40_num(k: int) -> str:
    return N40_NUMW[k] if k < len(N40_NUMW) else str(k)


def _n40_short(ph: str) -> str:
    for rx, rep in _N40_SHORT:
        if rx.match(ph):
            return rx.sub(rep, ph)
    return ph


def n40_molecule(rnd, molfg) -> str:
    """MOLECULE_INFO: what is out of range, and what the molecule already carries."""
    un = bs.direction(rnd)
    by = _collections.OrderedDict()
    for p_, d in un:
        by.setdefault(d, []).append(p_)
    bits = [f"{_and_list(v)} {'sits' if len(v) == 1 else 'sit'} {d} "
            f"{'its window' if len(v) == 1 else 'their windows'}"
            for d, v in by.items()]
    s1 = (f"{_n40_num(len(un)).capitalize()} "
          f"{'property is' if len(un) == 1 else 'properties are'} out of range: "
          f"{_and_list(bits)}." if un else "Nothing is out of range.")
    gs = [f"{_n40_num(c)} {g}" + ("s" if c > 1 and not g.endswith("s") else "")
          for g, c in sorted(molfg.items(), key=lambda kv: (-kv[1], kv[0]))]
    return s1 + " " + (f"What the molecule already carries is {_and_list(gs)}."
                       if gs else "The molecule carries no group the catalog names.")


def n40_analysis(rnd, molfg, n_cand: int) -> str:
    """EDIT_INFO, first half: one sentence per edit, carrying all three cells."""
    out = []
    for i in range(n_cand):
        # `_band_of` returns SETS and the row prints the round's own property order --
        # MR, HBD, HBA, rotB ... -- which is the order the constraints came in. Sorting
        # would put `HBA, HBD, MR, heavy_atoms` on the page and make the paragraph
        # disagree with every other arm's table on the same round.
        bands = [t.format(_and_list(str(x).rstrip("*") for x in v))
                 for t, v in zip(_N40_BAND,
                                 bs.classify(rnd, rnd["candidates"][i])) if v]
        dv = _atom_delta(rnd, i)
        got = [f"{dv[x]:+d} {x}" for x in _N19_CELL if dv[x] != 0]
        # `fg_row`'s empty case is a NOUN PHRASE -- "no change to any named group" --
        # written for a table cell, and `; it ` in front of it makes "it no change to
        # any named group". Rare on the unfiltered arms and not rare once `FG_DROP`
        # empties a fragment's list, so the prose needs its own verb here.
        fgc = fg_row(rnd, i, molfg)
        s = (f"Edit {i + 1} " + _and_list(bands)
             + ("; it changes no group the catalog names"
                if fgc == "no change to any named group" else "; it " + fgc))
        out.append(s + (f", for {_and_list(got)}." if got
                        else ", with no change in atom count."))
    return " ".join(out)


def n40_summary(feat, n_cand: int) -> str:
    """EDIT_INFO, second half: the cap-4 block, as one sentence."""
    bits = [(f"Edit {i + 1} alone " + _and_list(_n40_short(x) for x in feat[i]))
            if feat[i] else f"Edit {i + 1} has nothing of its own here"
            for i in range(n_cand)]
    return "What separates them: " + "; ".join(bits) + "."


def n40_reasons(rnd, molfg, unp, F, feat, i, side, key: str, pos: str):
    """One to three on-table facts for this decision, strongest first, or [].

    The pool and the first pick are `n21_clause`'s, exactly -- see its docstring for why
    the block row filters rather than gates. What is new is only how many follow it.
    """
    on = {k: v for k, v in F[i].items() if _n20_rank(k, molfg)[0] == 0}
    lf = lambda k: N21_LIFT.get(_n21_tmpl(k, molfg, unp), N21_FLOOR)   # noqa: E731
    ok = ((lambda k: lf(k) <= N21_FLOOR) if side == "drop"
          else (lambda k: lf(k) >= N21_FLOOR))
    pool = ([k for k in on if on[k][0] in feat[i] and ok(k)]
            or [k for k in on if ok(k)])
    if not pool:
        return []
    inb = lambda k: on[k][0] in feat[i]                               # noqa: E731
    rank = lambda k: (lf(k), inb(k), -_n20_rank(k, molfg)[1], -len(on[k][0]))
    best = (min(pool, key=lambda k: (lf(k), not inb(k),
                                     -_n20_rank(k, molfg)[1], -len(on[k][0])))
            if side == "drop" else max(pool, key=rank))
    h = lambda t: _hashlib.blake2b(f"n40:{key}:{pos}:{t}".encode(),
                                   digest_size=8).digest()            # noqa: E731
    want = 1 + int.from_bytes(h("k"), "big") % N40_KMAX
    rest = sorted((k for k in pool if k != best), key=h)[:want - 1]
    tail = sorted(rest, key=lambda k: lf(k), reverse=side != "drop")
    return [_n40_short(on[k][0]) for k in [best] + tail]


def n40_decision(e: int, side: str, pos: str, live, reasons, commit=None) -> str:
    """One decision: the pool it is taken from, the choice, then the reasons."""
    head = f"Of {_n35_list(live)}, weighing all of it together, "
    if side == "drop":
        head += (f"I set Edit {e + 1} aside {_N40_ORD[pos]}; "
                 f"that leaves {_n35_list([x for x in live if x != e])}.")
    else:
        head += f"I take Edit {e + 1}: {commit}."
    return head + (" It " + _and_list(reasons) + "." if reasons else "")


# --------------------------------------------------------------------------- #
# n44: n40's PAGE, with the three decisions argued in one sentence each.
#
# Everything above the decisions is n40's, byte for byte. What changes is the shape of
# the joint: n40 states the choice and appends the fact behind it, n44 states the choice
# and then says WHY, with a concession and, where the page supports one, a tie back to
# the property the round is actually short of.
#
#   Of Edits 1, 2, 3 and 4, I set Edit 4 aside first -- it adds the most rings, even
#   though it newly puts on aliphatic hydroxyl; that leaves Edits 1, 2 and 3.
#   Among Edits 1, 2 and 3, Edit 3 lowers MR the most, so I set Edit 3 aside next; ...
#   Between Edits 1 and 2, Edit 1 newly puts on methoxy, so I take Edit 1: ... .
#
# THE INDEX IS STILL THE FIRST THING THAT COMMITS. `Of Edits 1, 2, 3 and 4` and `Among
# Edits 1, 2 and 3` are the live set -- a fact about the page above, naming no candidate
# -- so the first token that chooses anything is the digit, exactly as in n21 and n40.
# That is the whole reason the drop2/commit sentences may lead with `Edit 3 ...`: the
# index precedes the fact, which is the ordering the 0.41-0.42 arms share and the
# 0.35 ones invert.
#
# THE FACT VOCABULARY IS WIDER THAN n40'S, AND IT IS STILL THE CORPUS'S. n40 quotes only
# facts checkable against the printed row; n44 also quotes the three families that live
# in the `suggest_edits` result -- `barely moves {p}`, `is the most certain on {p}`,
# `moves/raises/lowers {p} the most`. Those are in the prompt, four candidates side by
# side, so the student can still settle them; they are simply not on the page's own
# table. `N21_LIFT` has no entry for any of them (0 of 32 templates), so they default to
# the 0.250 floor, pass both sides of the polarity filter, and take their sign from
# `n19_facts`'s `good` flag instead. Sitting exactly on the floor they can never outrank
# a real fact -- they land as extras, never as the reason that leads.
#
# FOUR THINGS THE FIRST CUT GOT WRONG, all found by reading ninety rendered joints:
#   * the need clause ran backwards on 0.597 of the lines that carried one -- `adds the
#     most rings` was quoted as a reason to DROP `with heavy_atoms still below its
#     window`, when adding the most is what that need asks for. `_need_for` now attaches
#     the clause only when the count works AGAINST the decision's direction, and
#     `_NEED_CELL` stops an element count from speaking about the ring total at all.
#   * the concession repeated the reason's subject on 0.064 -- `newly puts on halogen,
#     even though it adds the most halogen`.
#   * the concession opposed a structural fact to a structural fact on 0.077 -- `adds
#     the most rings, even though it adds the most carbon` sets two facts against each
#     other that push the same way. A contrast needs a property on one side of it.
#   * a structural alert was conceded as a merit on 0.090, a third of them on rounds
#     whose unmet property is Mutag. `even though it newly puts on nitro` is not a
#     concession. They stay quotable as REASONS; they may not be conceded.
# And the round's own unmet property went unmentioned across all three joints on 0.236
# of rounds. Lift still picks which fact LEADS -- that is the ranking the arm is read
# against -- but if nothing quoted names a property `bs.direction` named, one is added.
_N44_MSG = ("mv", "dir", "sd")          # lives in the tool result, not in the row
# The families that speak about a PROPERTY, by name or as a tally over the whole band.
_N44_PROP = ("holds", "reach", "risk", "miss", "break", "mv", "dir", "sd", "way", "cnt")
_N44_ALERT = ("nitro", "nitroso", "aniline", "aromatic amine", "azo", "hydrazine",
              "epoxide", "aldehyde", "michael acceptor", "alkyl halide", "thiol",
              "aryl methyl sites for hydroxylation", "allylic oxidation",
              "quaternary nitrogen", "isocyanate", "azide")
# Which count may speak to which need. A ring count moves the ring total; an element
# count moves the atom total and the size measures with it. Nitrogen says nothing at all
# about rings_total, and the first cut attached it anyway.
_N44_NEED_CELL = {"rings_total": ("rings",),
                  "heavy_atoms": ("carbon", "nitrogen", "oxygen", "halogen"),
                  "MW":          ("carbon", "nitrogen", "oxygen", "halogen"),
                  "MR":          ("carbon", "nitrogen", "oxygen", "halogen", "rings")}
_N44_MOST = _re.compile(r"^adds the most (\w+)$")
_N44_FEW = _re.compile(r"^adds (?:the fewest|no) (\w+)$")
N44_KMAX = 3
_N44_VERBS = ("newly puts on ", "puts on ", "takes off ", "adds the most ",
              "adds the fewest ", "adds ", "is left exposed on ", "holds the ",
              "holds ", "gives up the ", "gives ")


def _n44_need(ph: str, side: str, dirs):
    """The need this COUNT fact works against, or None."""
    m = _N44_MOST.match(ph) or _N44_FEW.match(ph)
    if not m:
        return None
    grows = bool(_N44_MOST.match(ph))            # does the fact push the count UP?
    for p_, w in dirs:
        if p_ not in _N44_NEED_CELL or m.group(1) not in _N44_NEED_CELL[p_]:
            continue
        serves = grows == (w == "below")         # below wants more, above wants less
        if serves == (side == "commit"):
            return (p_, w)
    return None


def _n44_subject(k):
    """What a fact is ABOUT. Loose on purpose: `oxygen` and `ether oxygens (including
    phenoxy)` are one subject said two ways, and quoting one as the reason and the other
    as the concession reads as a contradiction."""
    return (k[1] if len(k) > 1 and isinstance(k[1], str) else k[0]).lower()


def _n44_clash(a: str, b: str) -> bool:
    return a == b or a in b or b in a


def _n44_merge(rs) -> str:
    """`holds rotB, holds HBA` is a list with a stutter in it. Collapse the repeats.

    Negations are left alone: `puts on no thiazole ring and aryl methyl sites` reads as
    if the second were put ON, so those stay two clauses.
    """
    out, i = [], 0
    while i < len(rs):
        v = next((v for v in _N44_VERBS if rs[i].startswith(v)), None)
        j, tails = i + 1, ([rs[i][len(v):]] if v else [])
        while v and j < len(rs) and rs[j].startswith(v):
            tails.append(rs[j][len(v):])
            j += 1
        out.append(v + _and_list(tails) if v and j > i + 1 else rs[i])
        i = j
    return _and_list(out)


def n44_reasons(rnd, molfg, unp, F, feat, i, side, key: str, pos: str,
                block: bool = True):
    """(1-3 facts, a concession or None, a need or None). Facts verbatim; joins are prose."""
    lf = lambda k: N21_LIFT.get(_n21_tmpl(k, molfg, unp), N21_FLOOR)   # noqa: E731
    bk = lambda k: round(lf(k) / 0.02)          # lift, bucketed, so tier breaks near-ties
    rk = lambda k: _n20_rank(k, molfg)[0]       # noqa: E731
    on = {k: v for k, v in F[i].items() if rk(k) == 0 or k[0] in _N44_MSG}
    want = side == "commit"

    def ok(k, good):
        if k[0] in _N44_MSG:
            return on[k][2] is good
        return lf(k) >= N21_FLOOR if good else lf(k) <= N21_FLOOR

    # `block=False` cuts the decisions loose from the block: the pool is the on-table
    # facts at large, and nothing prefers the edit's own exclusive ones.
    pool = (([k for k in on if on[k][0] in feat[i] and ok(k, want)] if block else [])
            or [k for k in on if ok(k, want)]) or []
    if not pool:
        return [], None, None
    inb = (lambda k: on[k][0] in feat[i]) if block else (lambda k: False)
    # NOT `on[k][1]`. That field is 0 for every risk/miss/break fact whether or not the
    # property was ever out of range, which left `is left exposed on Mutag` in front of
    # `is left exposed on rotB` on a round short of rotB.
    def tier(k):
        return 0 if (k[0] in _N44_PROP and len(k) > 1 and k[1] in unp) else 1
    best = (min(pool, key=lambda k: (bk(k), tier(k), not inb(k), rk(k),
                                     -_n20_rank(k, molfg)[1], -len(on[k][0])))
            if side == "drop" else
            max(pool, key=lambda k: (bk(k), -tier(k), inb(k), -rk(k),
                                     -_n20_rank(k, molfg)[1], -len(on[k][0]))))
    h = lambda t: _hashlib.blake2b(f"n44:{key}:{pos}:{t}".encode(),
                                   digest_size=8).digest()            # noqa: E731
    n_want = 1 + int.from_bytes(h("k"), "big") % N44_KMAX
    # One fact per subject, and one tool-result fact in all: `raises Mutag the most,
    # raises logS the most and raises logP the most` clears a per-subject filter and is
    # still one observation said three times.
    seen, msg_used, rest = [_n44_subject(best)], best[0] in _N44_MSG, []
    for k in sorted((k for k in pool if k != best), key=h):
        if len(rest) >= n_want - 1:
            break
        if any(_n44_clash(_n44_subject(k), x) for x in seen):
            continue
        if k[0] in _N44_MSG:
            if msg_used:
                continue
            msg_used = True
        seen.append(_n44_subject(k))
        rest.append(k)
    ks = [best] + sorted(rest, key=lf, reverse=side != "drop")
    hits = lambda k: k[0] in _N44_PROP and len(k) > 1 and k[1] in unp  # noqa: E731
    if not any(hits(k) for k in ks):
        cand = [k for k in pool if hits(k)
                and not any(_n44_clash(_n44_subject(k), _n44_subject(x)) for x in ks)]
        if cand:
            ks.append(min(cand, key=lf) if side == "drop" else max(cand, key=lf))
            seen.append(_n44_subject(ks[-1]))
    # THE CONCESSION. A drop concedes the best thing true of the edit it drops; a commit
    # concedes the worst thing true of the edit it takes. It must be on the printed
    # block, must not repeat a subject already quoted, must supply the property side of
    # the contrast when the reason did not, and may never be a structural alert.
    struct_led = best[0] not in _N44_PROP
    cp = [k for k in on if ok(k, not want) and k not in ks
          and (on[k][0] in feat[i] if block else True)
          and not any(_n44_clash(_n44_subject(k), x) for x in seen)
          and not (struct_led and k[0] not in _N44_PROP)
          and not any(a in on[k][0].lower() for a in _N44_ALERT)]
    con = None
    if cp:
        con = on[min(cp, key=lambda k: (lf(k), rk(k))) if want
                 else max(cp, key=lambda k: (lf(k), -rk(k)))][0]
    return ([_n40_short(on[k][0]) for k in ks],
            _n40_short(con) if con else None,
            _n44_need(on[ks[0]][0], side, bs.direction(rnd)))


def n44_line(e: int, side: str, pos: str, live, rs, con, need, commit=None,
             ordinal: bool = True) -> str:
    """One decision, argued. The pool, then the index, then why."""
    pool, k = _n35_list(live), len(live)
    rest = _n35_list([x for x in live if x != e])
    body = _n44_merge(rs)
    if need and need[0] not in body:
        body += f", with {need[0]} still {need[1]} its window"
    if con:
        body += f", even though it {con}"
    if pos == "drop1":
        # `first` is a lie on an arm with only one elimination.
        ord_ = " first" if ordinal else ""
        if not rs:
            return f"Of {pool}, I set Edit {e + 1} aside{ord_}; that leaves {rest}."
        return (f"Of {pool}, I set Edit {e + 1} aside{ord_} -- it {body}; "
                f"that leaves {rest}.")
    if pos == "drop2":
        if not rs:
            return f"Among {pool}, I set Edit {e + 1} aside next; that leaves {rest}."
        return (f"Among {pool}, Edit {e + 1} {body}, so I set Edit {e + 1} "
                f"aside next; that leaves {rest}.")
    head = "Between" if k == 2 else "Of"
    if not rs:
        return f"{head} {pool}, I take Edit {e + 1}: {commit}."
    return f"{head} {pool}, Edit {e + 1} {body}, so I take Edit {e + 1}: {commit}."
# ONE SENTENCE PER (OBSERVATION SHAPE, SIDE, JOINT), WRITTEN ONCE AND FROZEN.
#
# The first cut spelled the reason mechanically -- `Holding the least -- Edit 2 is the
# one to drop.` -- which is a table cell with a comma in it. These read the way a person
# would say the same thing, and they were drafted by a 27B given `bucket_span.classify`
# in words and told to write the MEANING of the house shorthand rather than quote it:
# `is left exposed on {p}` becomes "the central estimate for {p} is inside the window but
# the uncertainty band crosses the edge". Unpacking a label is its definition, not an
# addition, which is why this is allowed where saying anything else is not.
#
# FROZEN, AND KEYED ON WHAT THE SPAN ALREADY SHOWS. Generating a fresh sentence per round
# would put text in the middle of the span that the student cannot reproduce -- the same
# cost the blake2b cap draw imposed on n20, and a direct hit on the premise this arm
# shares with n21, that the line is determined once the edit is. So the sentence is a
# function of (what the fact is, which side, which joint), all three readable off the
# page, and the round only fills the slots. Three joints x sixteen shapes is variety
# enough that a reader sees a different construction at every joint.
#
# WHAT THE DRAFTS GOT WRONG, since the screens are the reusable part: 30 of 174 opened
# with "Edit" or ran to two sentences; 9 conceded on a joint that was not a concession;
# 8 put the index in front of the fact behind a subordinator ("Since Edit 2 adds ...");
# 7 opened on a pronoun with no antecedent yet; 5 slipped into the first person; 4 made
# the fact the stated CAUSE of the drop, which is the one inference this arm may not
# draw. Earlier passes also produced `Edit {n} is the one to go` on the COMMIT side --
# in this trace an edit that goes is an edit that was thrown out -- and `Keeping {p}
# below the target window`, false whenever the window is breached from the other side.
_N25_FRAMES_DOC = None
N25_FRAMES = {
    ('adds no {k}',
     'concede', 'drop1'):
        'Even though the swap brings no net change to {k}, Edit {n} is set aside.',
    ('adds no {k}',
     'concede', 'drop2'):
        'Even though the swap is {k}-neutral, Edit {n} is the second to be dropped.',
    ('adds no {k}',
     'keep', 'commit'):
        'The swap is {k}-neutral, so Edit {n} is the one to take.',
    ('adds the fewest {k}',
     'concede', 'drop1'):
        'Even though it adds the fewest {k} of all the candidates, Edit {n} is the first to come out.',
    ('adds the fewest {k}',
     'concede', 'drop2'):
        'Even though it adds the fewest {k} of all the candidates, Edit {n} is set aside.',
    ('adds the fewest {k}',
     'keep', 'commit'):
        'The smallest increase in {k} is added by Edit {n}, which is taken.',
    ('adds the most {k}',
     'concede', 'drop1'):
        'Even though it adds the most {k}, Edit {n} is the first to come out.',
    ('adds the most {k}',
     'concede', 'drop2'):
        'Even though it adds the most {k}, Edit {n} is the second to be dropped.',
    ('adds the most {k}',
     'drop', 'drop1'):
        'The fragment that adds the most {k} is found in Edit {n}, which is set aside.',
    ('adds the most {k}',
     'drop', 'drop2'):
        'No other candidate adds as many {k} as Edit {n}, which is set aside.',
    ('adds the most {k}',
     'keep', 'commit'):
        'Adding the most {k}, Edit {n} is the one to take.',
    ('breaks {p}, which the molecule satisfies now',
     'drop', 'drop1'):
        'The molecule satisfies {p} as it stands, but this edit pushes it out of range, so Edit {n} is set aside.',
    ('breaks {p}, which the molecule satisfies now',
     'drop', 'drop2'):
        'The molecule satisfies {p} as it stands, but this edit breaks it, so Edit {n} is set aside.',
    ('brings {p} back within reach',
     'concede', 'drop1'):
        'Even though {p} is brought back within reach, Edit {n} is set aside.',
    ('brings {p} back within reach',
     'concede', 'drop2'):
        'For all that {p} is brought back within reach, Edit {n} is set aside.',
    ('brings {p} back within reach',
     'keep', 'commit'):
        'With {p} pulled back within reach, Edit {n} is the one to take.',
    ('gives up the least',
     'concede', 'drop1'):
        'Even though it gives up the least among the candidates, Edit {n} is set aside.',
    ('gives up the least',
     'concede', 'drop2'):
        'For all that it sacrifices fewer targets than any other option, Edit {n} is the second to go.',
    ('gives up the least',
     'keep', 'commit'):
        'The candidate that sacrifices the fewest target properties is Edit {n}, which is taken.',
    ('gives up the most',
     'drop', 'drop1'):
        'Leaving the most properties out of range is what Edit {n} does, so it is set aside.',
    ('gives up the most',
     'drop', 'drop2'):
        'With the highest count of properties left outside their windows, Edit {n} is set aside.',
    ('gives {p} up',
     'drop', 'drop1'):
        'With {p} remaining outside its target window, Edit {n} is set aside.',
    ('gives {p} up',
     'drop', 'drop2'):
        'Because this step fails to bring {p} within range, Edit {n} is set aside.',
    ('holds the least',
     'drop', 'drop1'):
        'With fewer target properties held safely in range than any other candidate, Edit {n} is set aside.',
    ('holds the least',
     'drop', 'drop2'):
        'The candidate that keeps the fewest target properties safely in range is Edit {n}, so it is set aside.',
    ('holds the most',
     'concede', 'drop1'):
        'Even though it holds the most target properties safely in range, Edit {n} is set aside.',
    ('holds the most',
     'concede', 'drop2'):
        'For all that it keeps the highest number of target properties in range, Edit {n} is dropped.',
    ('holds the most',
     'keep', 'commit'):
        'Keeping the most properties in range is what Edit {n} does, so it is taken.',
    ('holds {p}',
     'concede', 'drop1'):
        'Even though {p} is held safely in range, Edit {n} is set aside.',
    ('holds {p}',
     'concede', 'drop2'):
        'For all that {p} remains within its window, Edit {n} is the one to drop.',
    ('holds {p}',
     'keep', 'commit'):
        'With {p} held safely within its target window, Edit {n} is taken.',
    ('is left exposed on {p}',
     'drop', 'drop1'):
        'The central estimate for {p} is inside the window but the uncertainty band crosses the edge, so Edit {n} is set aside.',
    ('is left exposed on {p}',
     'drop', 'drop2'):
        'With {p} in range but the error margin touching the edge, Edit {n} is set aside.',
    ('puts on no {g}',
     'concede', 'drop1'):
        'Even though no new {g} is added by Edit {n}, it is set aside.',
    ('puts on no {g}',
     'concede', 'drop2'):
        'For all that no {g} comes in with this fragment, Edit {n} is set aside.',
    ('puts on no {g}',
     'keep', 'commit'):
        'The absence of {g} in the incoming fragment means Edit {n} is taken.',
    ('puts on {g}, which the molecule does not have yet',
     'concede', 'drop1'):
        'Even though the incoming fragment brings {g}, a group the molecule currently lacks, Edit {n} is set aside.',
    ('puts on {g}, which the molecule does not have yet',
     'concede', 'drop2'):
        'For all that it introduces {g}, which is absent from the starting structure, Edit {n} is dropped.',
    ('puts on {g}, which the molecule does not have yet',
     'keep', 'commit'):
        'The introduction of {g}, which the molecule does not have yet, makes Edit {n} the one taken.',
    ('puts on {g}',
     'drop', 'drop1'):
        'Adding {g}, a group the molecule already possesses, Edit {n} is set aside.',
    ('puts on {g}',
     'drop', 'drop2'):
        'With the incoming fragment carrying {g}, which the molecule already has, Edit {n} is set aside.',
    ('puts {p} out, though not out of reach',
     'drop', 'drop1'):
        'The central estimate for {p} moves out of range, so Edit {n} is set aside.',
    ('puts {p} out, though not out of reach',
     'drop', 'drop2'):
        'With {p} moved out of the window but remaining recoverable, Edit {n} is set aside.',
    ('takes off {g}',
     'concede', 'drop1'):
        'Even though the swap removes {g} from the structure, Edit {n} is set aside.',
    ('takes off {g}',
     'concede', 'drop2'):
        'True as it is that {g} is stripped away by the change, Edit {n} is dropped.',
    ('takes off {g}',
     'keep', 'commit'):
        'The removal of {g} means Edit {n} is the one to take.',
}


def _n25_shape(k, molfg, unp):
    """(the frame key for this fact, the slot it fills).

    Off `_n21_tmpl`, with two differences. The count family is keyed per ELEMENT there
    -- the lift of `adds the most carbon` and `adds the most nitrogen` point opposite
    ways and merging them hid both -- and per SHAPE here, because the sentence is the
    same whichever element it is. And the two phrases that carry a relative clause get
    their own key, since a frame has to know whether it is wrapping one.
    """
    t = _n21_tmpl(k, molfg, unp).split("   ")[0]
    m = _re.match(r"^adds (no|the most|the fewest) (.+)$", t)
    if m:
        return f"adds {m.group(1)} {{k}}", {"k": m.group(2)}
    if t == "brings {p} back within reach" and k[1] not in unp:
        return "puts {p} out, though not out of reach", {"p": k[1]}
    if t == "breaks {p}":
        return "breaks {p}, which the molecule satisfies now", {"p": k[1]}
    if t == "puts on {g}" and k[1] not in molfg:
        return "puts on {g}, which the molecule does not have yet", {"g": k[1]}
    for sl in ("{p}", "{g}"):
        if sl in t:
            return t, {sl[1]: k[1]}
    return t, {}


def n25_lead(k, molfg, unp, i, side, pos):
    """The reason line that goes ABOVE the decision line, or None if the shape is new."""
    sh, sl = _n25_shape(k, molfg, unp)
    f = N25_FRAMES.get((sh, side, pos))
    if f is None:
        # The bank is the complete closure of what the polarity gate can hand this
        # function -- 8 drop shapes x 2 joints, 10 keep shapes at the commit, and the
        # same 10 as concessions at 2 joints. A miss means the gate changed and the bank
        # did not, and the first version of this returned None and left 14 decision lines
        # in 299,367 with no reason above them, which nothing downstream would report.
        raise KeyError(f"n25: no frame for {(sh, side, pos)}")
    return f.format(n=i + 1, **sl)



# n28's bank. The 46 sentences n25 uses are REUSED, re-keyed by mode, so the two arms
# say the same thing wherever they say anything -- the only difference between them is
# WHICH FACT gets quoted, and a second difference in wording would make that unreadable.
# Eleven are new: eight for a concession on the COMMIT, which n25 never reaches because
# its wider pool always found something above the floor, and three for the case n25 has
# no answer to at all -- the block row is empty and a shared fact is not allowed.
_N28_COMMIT_CONCEDE = {
    "is left exposed on {p}":
        "Even with {p} inside its window but not clear of the edge, "
        "Edit {n} is still the one to take.",
    "holds the least":
        "For all that it holds the fewest target properties safely in range, "
        "Edit {n} is still the one to take.",
    "adds the most {k}":
        "Despite raising the {k} count more than any other candidate, "
        "Edit {n} is the one taken.",
    "puts on {g}":
        "Even bringing in {g}, which the molecule already carries, "
        "Edit {n} is still the one to take.",
    "puts {p} out, though not out of reach":
        "With {p} pushed out of its window and only just recoverable, "
        "Edit {n} is nonetheless the one to take.",
    "gives {p} up":
        "Although {p} is left outside its target window, Edit {n} is still the one taken.",
    "breaks {p}, which the molecule satisfies now":
        "Granting that {p}, which the molecule satisfies as it stands, is broken here, "
        "Edit {n} is still the one to take.",
    "gives up the most":
        "True as it is that more target properties are written off here than by any "
        "other candidate, Edit {n} is the one taken.",
}
# THE EMPTY ROW. It quotes nothing, because there is nothing of the edit's own to quote
# and the whole point of the arm is that a shared fact is worse than silence -- it fits
# 3.25 candidates and points at all of them. What is left is true, checkable against the
# block row two lines up (`Edit N -- no feature of its own here`), and names the edit.
N28_NONE = {
    "drop1": "Nothing in the table sets Edit {n} apart, and it is the first to come out.",
    "drop2": "With nothing of its own to point to, Edit {n} is the one set aside.",
    "commit": "With no feature of its own on the table, Edit {n} still looks like "
              "the one to take.",
}
N28_FRAMES = {}
for (_sh, _sd, _pos), _v in N25_FRAMES.items():
    if _sd == "drop":
        N28_FRAMES[(_sh, "drop", "plain", _pos)] = _v
    elif _sd == "keep":
        N28_FRAMES[(_sh, "commit", "plain", "commit")] = _v
    else:                                                   # n25's concede side
        N28_FRAMES[(_sh, "drop", "concede", _pos)] = _v
for _sh, _v in _N28_COMMIT_CONCEDE.items():
    N28_FRAMES[(_sh, "commit", "concede", "commit")] = _v
del _sh, _sd, _pos, _v


def n28_pick(rnd, molfg, unp, F, feat, i, side):
    """(the fact key or None, one of 'plain' / 'concede' / 'none').

    EXCLUSIVE facts only -- the pool is the edit's own block row, never the wider set of
    on-table facts n25 drew from. An exclusive fact is true of exactly one candidate, so
    the sentence built on it names the edit instead of narrowing to 3.25 of them.
    """
    on = {k: v for k, v in F[i].items()
          if _n20_rank(k, molfg)[0] == 0 and v[0] in feat[i]}
    if not on:
        return None, "none"

    def lf(k):
        t = _n21_tmpl(k, molfg, unp)
        return N25_LIFT_TIE.get(t, N21_LIFT.get(t, N21_FLOOR))
    neg = lambda k: _n25_reads_negative(k, unp)
    ok = ((lambda k: lf(k) <= N21_FLOOR or neg(k)) if side == "drop"
          else (lambda k: lf(k) >= N21_FLOOR and not neg(k)))
    pool = [k for k in on if ok(k)]
    rank = lambda k: (-_n20_rank(k, molfg)[1], -len(on[k][0]))
    if pool:
        b = (min(pool, key=lambda k: (lf(k),) + rank(k)) if side == "drop"
             else max(pool, key=lambda k: (lf(k),) + rank(k)))
        return b, "plain"
    # Nothing on the right side of the floor. Grant the strongest thing the row has and
    # decide anyway -- on a drop that is the most favourable fact, on a commit the least.
    b = (max(on, key=lambda k: (lf(k),) + rank(k)) if side == "drop"
         else min(on, key=lambda k: (lf(k),) + rank(k)))
    return b, "concede"


def n28_lead(k, molfg, unp, i, side, mode, pos):
    """The reason line above the decision. Never None -- every case has a frame."""
    if mode == "none":
        return N28_NONE[pos].format(n=i + 1)
    sh, sl = _n25_shape(k, molfg, unp)
    f = N28_FRAMES.get((sh, side, mode, pos))
    if f is None:
        raise KeyError(f"n28: no frame for {(sh, side, mode, pos)}")
    return f.format(n=i + 1, **sl)


N22_ARMS = ("n22",)
N21_ARMS = ("n21",)
# Anchored on the FULL line each one closes, so a clause can only ever be appended.
_N21_DROP = _re.compile(r"^Dropping Edit (\d+); that leaves .*\.$")
_N21_TAKE = _re.compile(r"^Taking Edit (\d+): .*\.$")
# P(the edit a template singles out is the committed one), fitted on the 99k train
# rounds, read back on test with no ranking change. The floor is 0.250 -- four
# candidates -- so a template above it is evidence FOR an edit and one below it is
# evidence AGAINST. Templates with fewer than 200 train pointings are absent and default
# to the floor when looked up; over the whole test split no clause was ever built from
# one, so the default is not load-bearing.
N21_FLOOR = 0.25
N21_LIFT = {
    'gives up the least':                                 0.3767,   # n=14,391
    'holds the most':                                     0.3672,   # n=37,715
    'holds {p}   [p is IN range now]':                    0.3415,   # n=20,550
    'holds {p}':                                          0.3297,   # n=24,786
    'adds no rings':                                      0.3184,   # n=7,323
    'adds the fewest halogen':                            0.3121,   # n=8,719
    'adds no nitrogen':                                   0.3099,   # n=11,810
    'adds the fewest rings':                              0.3089,   # n=9,691
    'adds the fewest nitrogen':                           0.3087,   # n=16,321
    'adds no halogen':                                    0.3086,   # n=7,415
    'takes off {g}':                                      0.3050,   # n=4,800
    'puts on no {g}':                                     0.2900,   # n=29,837
    'adds no carbon':                                     0.2779,   # n=11,866
    'brings {p} back within reach':                       0.2747,   # n=22,610
    'puts on {g}   [new to the molecule]':                0.2744,   # n=33,341
    'brings {p} back within reach   [p is IN range now]': 0.2695,   # n=14,422
    'adds no oxygen':                                     0.2650,   # n=17,073
    'adds the fewest oxygen':                             0.2635,   # n=22,954
    'adds the fewest carbon':                             0.2565,   # n=49,440
    'adds the most oxygen':                               0.2556,   # n=28,956
    'adds the most carbon':                               0.2555,   # n=48,656
    'puts on {g}   [already present]':                    0.2471,   # n=26,018
    'adds the most halogen':                              0.2447,   # n=23,821
    'adds the most rings':                                0.2433,   # n=14,935
    'adds the most nitrogen':                             0.2330,   # n=28,264
    'gives {p} up':                                       0.2253,   # n=19,502
    'is left exposed on {p}   [p is IN range now]':       0.2144,   # n=27,752
    'is left exposed on {p}':                             0.2134,   # n=25,221
    'gives up the most':                                  0.1885,   # n=20,985
    'holds the least':                                    0.1803,   # n=39,926
    'gives {p} up   [p is IN range now]':                 0.1617,   # n=8,976
    'breaks {p}   [the * mark]':                          0.1617,   # n=8,976
}


def _n21_tmpl(k, molfg, unp):
    """The fact's key with its property or group name replaced by a placeholder.

    The table is keyed on the SHAPE of the claim, not on which property it names: there
    are fourteen properties and sixty-one groups, and splitting the table by them would
    leave most cells too thin to fit. The count family IS split by element, because
    `adds the most carbon` (0.2555) and `adds the most nitrogen` (0.2330) turned out to
    point opposite ways and merging them hid both.
    """
    t = k[0]
    if t in ("holds", "reach", "risk", "miss"):
        base = {"holds": "holds {p}", "reach": "brings {p} back within reach",
                "risk": "is left exposed on {p}", "miss": "gives {p} up"}[t]
        return base + ("" if k[1] in unp else "   [p is IN range now]")
    if t == "break":
        return "breaks {p}   [the * mark]"
    if t == "adds":
        return "puts on {g}" + ("   [new to the molecule]" if k[1] not in molfg
                                else "   [already present]")
    if t == "drops":
        return "takes off {g}"
    if t == "notadds":
        return "puts on no {g}"
    if t == "cnt":
        return k[2]
    if t == "no":
        return f"adds no {k[1]}"
    if t == "num":
        return k[2]
    return ""


# n25 ONLY. The two `holds` templates measure 0.3415 where the property was already
# inside its window against 0.3297 where this edit brought it in -- 0.012 apart on 20,550
# and 24,786 train pointings, inside each other's noise -- and the order that imposes is
# backwards on the one thing separating them. Naming a property the edit REPAIRED says
# something; naming one that was never in trouble is equally true of every candidate that
# left it alone. Taking the maximum, the clause took the non-event on 0.7514 of the
# `holds` lines whose own row offered a repair. The number is not refitted, it is put
# back on the right side of a tie it should never have decided.
N25_LIFT_TIE = {"holds {p}   [p is IN range now]": 0.3290}


def _n25_reads_negative(k, unp):
    """Facts a reader hears as an objection whatever the lift says.

    One entry, and it is the one the split above created. `puts {p} out, though not out
    of reach` measures at 0.2695 -- a hair ABOVE the 0.250 floor, so the polarity gate
    would let it close a COMMIT, and "putting logD out, Edit 3 is the one to take" is
    the exact sentence the gate exists to prevent. The statistic is inside noise of the
    floor and the wording is not, so the wording decides.
    """
    return k[0] == "reach" and k[1] not in unp


def n21_clause(rnd, molfg, unp, F, feat, i, side, cell=None, block_first=True,
               reads_negative=None, tie=False):
    """The clause for edit `i` under `Dropping` or `Taking`, or None.

    On-table facts only, so the claim can be checked against the row printed above it.

    Returns the fact's KEY, not its phrase: n21 wants the phrase and n25 wants to know
    which SHAPE it is so it can pick a frame, and only the key carries both.

    `block_first` is how n21 chose: the block's own row FILTERS the pool, so the model
    copies a line it has already written, and only an edit whose row says nothing falls
    through to the rest. It costs more than it looked: on 0.7715 of commits the row held
    a weaker fact than the table did, and the clause took the weaker one -- `adds the
    most carbon` at 0.2555 in front of `holds heavy_atoms` at 0.3297. With it off the
    row becomes a TIE-BREAK instead of a gate, so the clause still prefers what the
    model wrote whenever the two are equally far from the floor.
    """
    on = {k: v for k, v in F[i].items() if _n20_rank(k, molfg, cell)[0] == 0}
    def lf(k):
        t = _n21_tmpl(k, molfg, unp)
        return (N25_LIFT_TIE if tie else {}).get(t, N21_LIFT.get(t, N21_FLOOR))
    neg = (lambda k: False) if reads_negative is None else (lambda k: reads_negative(k, unp))
    ok = ((lambda k: lf(k) <= N21_FLOOR or neg(k)) if side == "drop"
          else (lambda k: lf(k) >= N21_FLOOR and not neg(k)))
    pool = ([k for k in on if on[k][0] in feat[i] and ok(k)] if block_first else []) \
        or [k for k in on if ok(k)]
    if not pool:
        return None
    # The furthest from the floor on this side, then the block row, then the arm's usual
    # order. Filtering the candidates is not enough on its own: on a row whose facts all
    # sit near the KEEP end, ranking by tier still promoted the most positive survivor.
    inb = lambda k: on[k][0] in feat[i]
    b = (min(pool, key=lambda k: (lf(k), not inb(k),
                                  -_n20_rank(k, molfg, cell)[1], -len(on[k][0])))
         if side == "drop" else
         max(pool, key=lambda k: (lf(k), inb(k),
                                  -_n20_rank(k, molfg, cell)[1], -len(on[k][0]))))
    return b


def n25_concede(rnd, molfg, unp, F, feat, i, cell=None, tie=False):
    """The BEST thing true of an edit that has nothing against it, or None.

    Reached only where `n21_clause` returns None on a drop -- every on-table fact the
    edit has points the other way, so the honest sentence grants one and drops the edit
    anyway. It grants the STRONGEST, not the weakest: taking the minimum conceded `adds
    the most carbon` on 0.9835 of these joints while the same row said `holds the most`,
    which is the fact a reader is looking at. Conceding the best point and going anyway
    is a sentence; conceding the thinnest one is an evasion.
    """
    on = {k: v for k, v in F[i].items() if _n20_rank(k, molfg, cell)[0] == 0}
    if not on:
        return None
    def lf(k):
        t = _n21_tmpl(k, molfg, unp)
        return (N25_LIFT_TIE if tie else {}).get(t, N21_LIFT.get(t, N21_FLOOR))
    inb = lambda k: on[k][0] in feat[i]
    b = max(on, key=lambda k: (lf(k), inb(k), -_n20_rank(k, molfg, cell)[1],
                               -len(on[k][0])))
    return b


SEQ_HEAD = """You are writing two steps of a molecular-design agent's reasoning: the
sentence in front of each of two eliminations.

RULES

- NEVER OPEN WITH "Edit". This is the rule that gets ignored more than all the others put
  together, so here it is twice. The sentence names the edit -- it just does not START
  there. Put the reason first and let the edit arrive after it.

      NO   "Edit 1 is set aside because it moves BBBP the wrong way."
      YES  "With BBBP already over and this change pushing it further the wrong way,
            Edit 1 is set aside."

      NO   "Edit 4 moves logP the right way only tentatively, so it is dropped."
      YES  "Moving logP the right way but not surely enough, Edit 4 is dropped."

- One sentence each, at most 32 words. No markdown, no bullet, no line break. Do not
  write the "Dropping Edit ..." lines -- they are already there.

- ARGUE FROM WHAT IS LISTED **AGAINST** THE EDIT THAT GOES. Those are the reasons it
  goes. A fact under "in its favour" is the opposite of a reason to set it aside, and
  using one makes the sentence say the reverse of what happens under it. You may not add
  a fact of your own, and you may not name a property or a group that is not listed for
  that edit. If nothing is listed against it, say that no property counts against it and
  let it go on the fragment.

- NO NUMBERS. Not one digit, not one written-out quantity.

- A property under "misses" with a STAR is one the molecule currently satisfies and this
  edit would break. It is not already wrong -- do not say it is. The properties that are
  already wrong are the ones on the "Out of range now" line, and only those.

- Use the listing's own words for a property or a group -- BBBP, logP, nitrile -- never a
  reading of them like "the blood-brain barrier score" or "the hydroxyl".

- You do not know which edit will finally be taken, and you must not guess: no "best",
  no "the winner", no "I would take".

- WHERE THIS COMES FROM DOES NOT EXIST FOR THE READER. The reader sees only the molecule,
  the tool's suggestions and the lines above. Never write "the sheet", "listed", "no
  listed drawback" or anything else about where you got this.

      NO   "With no listed drawbacks on the sheet, Edit 2 is set aside."
      YES  "With no property counting against Edit 2, it goes on the fragment instead."

- Fragment size is not a reason on its own. Do not drop an edit for being the largest or
  the smallest.

- Do not begin with "It" or "This" -- name what the fact is about.

- The two sentences must not read as the same sentence twice. Where the same fact is the
  reason both times, say it a different way and lead with a different part of it.

A COMPLETE ANSWER, for two edits whose listings read

      Edit 1   AGAINST it: moves BBBP the right way but nowhere near far enough;
                           carries the widest spread here on logP
      Edit 4   AGAINST it: breaks MR, which the molecule satisfies now;
                           moves BBBP the right way, though it may not hold

looks like this and nothing else:

      With BBBP barely shifting and the spread on logP the widest here, Edit 1 goes first.
      ###
      Breaking MR, which the molecule holds today, Edit 4 follows it out.

Note what that answer does NOT contain: no digit, no "the sheet", no edit that stays, no
sentence opening with "Edit", and no second sentence that is the first one again.

Two sentences in order, separated by a line of ###, nothing else.

"""


def _fact_block(rnd, unp, molfg, only=None) -> str:
    """The whole context a joint is written from: the ask, and the sheet.

    NO DELTAS AND NO MEASUREMENTS. They were in this block for one run and the writer
    used them to do arithmetic it got wrong half the time. What it needs from them --
    which way each property has to move, whether this edit moves it that way, how far,
    how surely -- is on the sheet already, computed rather than inferred.
    """
    # NO `user_prompt`. It was 10.9% of the prompt and the sentence may not use it: every
    # claim has to come off the listing below, which already carries which way each
    # property must move.
    L = ["WHAT IS STILL WRONG WITH THE MOLECULE: "
         + (", ".join(f"{p} is {d} its range" for p, d in bs.direction(rnd)) or "nothing"),
         "", "WHAT THE TWO EDITS THAT GO ACTUALLY DO. Every line is computed from the",
         "tool's numbers; nothing else is known about them."]
    for i, facts in fact_sheet(rnd, unp, molfg):
        if only is not None and i not in only:
            continue
        L.append(f"  Edit {i + 1}")
        for lab, key in (("AGAINST it", "against"), ("in its favour", "for"),
                         ("neither", "neutral")):
            got = [x for x in facts if _polarity(x) == key]
            if got:
                L.append(f"    {lab}:")
                L += [f"      - {x}" for x in got]
    return "\n".join(L)


def _seq_task_block(rnd) -> str:
    """The turn as the student meets it: the ask, where the molecule stands, and what the
    tool proposed -- with the numbers the band rounds off."""
    tg = rnd.get("targets") or {}
    pr = rnd.get("props") or {}
    lines = ["THE TASK", (rnd.get("user_prompt") or "").strip(), "",
             f"CURRENT MOLECULE: {rnd.get('smiles')}"]
    if pr:
        lines.append("MEASURED NOW: " + ", ".join(
            f"{k} {v}" for k, v in pr.items()))
    if tg:
        lines.append("TARGET RANGE:  " + ", ".join(
            f"{k} {v[0]} to {v[1]}" for k, v in tg.items()))
    lines += ["", "WHAT suggest_edits PROPOSED, one line per edit:"]
    for i, c in enumerate(rnd["candidates"]):
        d = c.get("delta") or {}
        shift = ", ".join(f"{k} {v.get('avg')} +/- {v.get('std')}" for k, v in d.items())
        lines.append(f"  Edit {i + 1} | {c.get('from_smiles')} -> {c.get('to_smiles')}"
                     + (f" | {shift}" if shift else ""))
    return "\n".join(lines)


def build_seq_prompts(rnd, frame: str, order, unp=None, molfg=None):
    """([one prompt], the fact text its gates check against).

    Still a list, because that is what dispatches the sequential path -- one element now
    instead of two. The frame travels whole, marks and all, so the writer sees where each
    sentence lands and which line follows it.
    """
    facts = _fact_block(rnd, unp or set(), molfg or {}, only=set(order[:2]))
    # THE FRAME WITHOUT ITS BAND ROWS. They are in the prompt twice -- the listing above
    # is the same four rows with more said about them -- and the rows are the largest
    # round-specific block there is.
    skel = "\n".join(x for x in frame.split("\n") if not x.startswith("  Edit "))
    body = (f"{facts}\n\nWHAT THE AGENT IS WRITING. Your two sentences go where the "
            f"marks are:\n{skel}\n\nWrite the sentence for {GAP_MARKS[0]} "
            f"(about Edit {order[0] + 1}), then a line of ###, then the sentence for "
            f"{GAP_MARKS[1]} (about Edit {order[1] + 1}).")
    return [SEQ_HEAD + body], facts


_DIGIT = _re.compile(r"(?<![A-Za-z])[-+]?\d")
_QTY = _re.compile(r"\b(one|two|three|four|five|six|seven|eight|nine|ten|"
                   r"half|twice|double|triple|zero|nil)\b", _re.I)


def no_digits(text: str) -> bool:
    """No quantity at all. `Edit 3` is a label and `sp3` is part of a chemical token."""
    y = _re.sub(r"\bedit\s+\d+", " ", text or "", flags=_re.I)
    return not (_DIGIT.search(y) or _QTY.search(y))


# THE BAND ARBITER HAS A PARAPHRASE HOLE. "the blood-brain barrier score is already past
# its ceiling" names BBBP without writing BBBP, and a scan for the literal names sees
# nothing to check. Every reading the writer actually produced, mapped back.
_ALIAS = {
    "blood-brain barrier": "BBBP", "blood brain barrier": "BBBP",
    "brain penetration": "BBBP", "brain permeability": "BBBP",
    "lipophilicity": "logP", "partition coefficient": "logP", "hydrophobicity": "logP",
    "distribution coefficient": "logD", "solubility": "logS", "aqueous solubility": "logS",
    "mutagenicity": "Mutag", "ames": "Mutag", "polar surface area": "TPSA",
    "molecular weight": "MW", "molar refractivity": "MR", "refractivity": "MR",
    "rotatable bond": "rotB", "hydrogen-bond donor": "HBD", "hydrogen bond donor": "HBD",
    "hydrogen-bond acceptor": "HBA", "hydrogen bond acceptor": "HBA",
    "heavy atom": "heavy_atoms", "drug-likeness": "QED", "druglikeness": "QED",
    "ring count": "rings_total", "number of rings": "rings_total",
}


def props_named(text: str) -> set:
    """The properties a sentence names, by their band name or by a reading of it."""
    low = (text or "").lower()
    out = {p for p in _PROPS if p in (text or "")}
    return out | {v for k, v in _ALIAS.items() if k in low}


def band_only_bad(text: str, bad: set) -> bool:
    """A drop joint may name a property only where the band condemns THIS edit.

    The band is derived from the same deltas the writer is reading, so it is the arbiter
    that costs nothing. Three of the eight false joints found by hand were of exactly one
    shape -- naming a property the band lists under `holds` and then calling it a problem
    -- and this is the check that ends them.
    """
    return all(p in bad for p in props_named(text))


_STOP = {"the", "a", "an", "its", "it", "is", "are", "and", "or", "of", "to",
         "with", "at", "in", "on", "this", "that", "so", "as", "for", "by",
         "already", "while", "which", "be", "not", "into", "out", "from"}


# THE APPARATUS MUST NOT REACH THE CORPUS. 5.2% of joints wrote "the sheet lists no
# facts against Edit 3" -- a document the student will never see, described to it as
# though it were part of the turn. Fragment size rides along here because it is a neutral
# fact used as a verdict, and the corpus said both "dropped for being the largest" and
# "dropped for being the smallest" two lines apart.
_META = ("sheet", "listed", "the list", "table above", "no argument", "downside",
         "largest fragment", "smallest fragment", "fragment size", "as given",
         "the facts given", "provided")


def no_meta(text: str) -> bool:
    return not any(m in (text or "").lower() for m in _META)


def is_ascii(text: str) -> bool:
    """The rest of the corpus is ASCII. A curly apostrophe or a real minus sign in one
    joint and the student learns two spellings of the same mark."""
    return all(ord(c) < 128 for c in (text or ""))


# ONLY AN OUT-OF-RANGE CLAIM. "Breaking heavy_atoms, which the molecule CURRENTLY
# SATISFIES" is the star read exactly right, and a trigger on the bare word "currently"
# threw it away. What must be on the NEEDS line is a claim that a property is already
# WRONG, not that it is already fine.
_ALREADY = ("already out", "already over", "already past", "already above",
            "already below", "already exceed", "already violat", "already wrong",
            "currently out", "currently over", "currently above", "currently below",
            "currently exceed", "still out", "still above", "still below",
            "already at the ceiling", "already at its ceiling", "still wrong")


def already_ok(text: str, unp: set) -> bool:
    """A property called ALREADY out of range must be on the NEEDS line.

    `misses logP*` means the opposite of what the writer read it as: the star marks a
    property the molecule currently satisfies and THIS EDIT would break. "With logP
    already at the ceiling" inverts that -- logP is fine until the edit lands.
    """
    if not any(m in (text or "").lower() for m in _ALREADY):
        return True
    return all(p in unp for p in props_named(text))


def not_echo(text: str, prev: str) -> bool:
    """The second joint may not open with the first joint's words.

    Measured on the first high-detail run: a round whose two joints differ only in the
    edit number and the value, "With BBBP at 0.973 exceeding the 0.961 ceiling, Edit 1..."
    then the identical clause for Edit 2. The writer is handed the first joint as context
    and copies it when nothing tells it not to.
    """
    strip = lambda t: set(_re.sub(r"\bedit\s+\d+", " ", (t or "").lower(),
                                  flags=_re.I).split()) - _STOP
    A, B = strip(text), strip(prev)
    if not A or not B:
        return True
    # FIRST-k-WORDS WAS TOO WEAK: "With logP already at THE upper limit" against "...at
    # ITS upper limit" differs at word five and sailed through, and the two joints were
    # otherwise the same sentence. Content-word overlap does not care which article.
    # 0.7 let through a pair differing only in "sulfonamide" against
    # "heteroaromatic" -- eight content words shared out of twelve once the edit
    # numbers were counted, which is why they come out first.
    return len(A & B) / len(A | B) < 0.6


_SEQ_EDIT = _re.compile(r"\bedit\s+\d+", _re.I)
_VERDICT = ("the best", "best option", "best choice", "the winner", "i would take",
            "i will take", "should be taken", "the strongest candidate", "the right one",
            "we should choose", "the preferred", "is the answer", "will be chosen")
_NUMTOK = _re.compile(r"\d+(?:\.\d+)?")


def no_verdict(text: str) -> bool:
    """A drop joint may not say which edit wins. The writer does not know, so a verdict
    from it is a guess the corpus would teach as if it were the teacher's."""
    return not any(m in (text or "").lower() for m in _VERDICT)


def numbers_ok(text: str, ctx: str) -> bool:
    """Every number in the sentence must be one the turn printed.

    Edit indices are stripped first -- they are labels, not measurements -- and so is any
    ordinal word, which carries no digit. What is left is deltas, spreads and bounds, and
    a writer that rounds 2.452 to 2.45 has invented a number the student cannot check.
    """
    y = _SEQ_EDIT.sub(" ", text or "")
    return all(n in (ctx or "") for n in _NUMTOK.findall(y))


POLISH_BAND17 = """Rewrite each note as one natural sentence a chemist would say while
working through the options.

Keep the whole note, including the clause after the semicolon -- a sentence that stops
at the first clause is not a rewrite of it. A note about an edit's weakness stays about
its weakness; never turn "left exposed" into "still holding".

Open the way the note opens, never with "Edit". "On MR, Edit 1 gives it up" may become
"Where MR is concerned, Edit 1 gives it up" -- but never "Edit 1 gives up MR". No
sentence may begin with the word "Edit".

MAKE NO COMPARISON. The note says what ONE edit does. It does not say that edit is the
only one, the weakest one, or worse than any other, and the rewrite may not say so
either: "Edit 1 is THE ONE THAT gives it up" is not a rewording of "Edit 1 gives it up",
it is a new claim, and usually a false one -- another edit is doing the same thing. No
"the only one", no "the one that", no "the one left exposed" -- do not write the word
"one" at all -- no "the weakest", no "unlike", no "the most".

KEEP THE ORDINAL. The first note ends "it is the first to go" and the second ends "it is
the next to go", because the second elimination is not the first. Do not turn "next"
into "first".

TWO sentences in order, separated by a line of ###, nothing else.

{notes}"""


def draft_all(own, lives, targets, key: str):
    """The three joints, each on its own observation where one is available."""
    out, used = [], set()
    for j, (live, tgt) in enumerate(zip(lives, targets)):
        sent, lb = draft_passage(own, live, tgt, used, f"{key}:{j}")
        if lb:
            used.add(lb)
        out.append(sent)
    return out


# The POLISH prompt. No task, no molecule, no frame, no rules about what to argue --
# the argument is already made and correct, and all that is left is to make it read like
# a person wrote it. Under 1 KB against the 6.6 KB the writing prompt needed, which is
# most of the speed: this pipeline is prefill-bound.
POLISH = """Rewrite each of these three notes as one natural sentence, the way a chemist
would say it while working through the options.

EVERY SENTENCE MUST OPEN THE WAY ITS NOTE OPENS -- with the observation, never with an
edit. "On heavy atoms added, Edit 2 adds the most and Edit 3 adds the least" may become
"In terms of heavy atoms added, Edit 2 adds the most while Edit 3 adds the fewest", but
never "Edit 2 adds the most heavy atoms". Moving the edit to the front is the one change
that is not allowed, and no sentence may begin with the word "Edit".

Keep every Edit number that appears in a note, keep which edit is higher and which is
lower, and add nothing that is not there -- no chemistry you were not given, no
conclusion, no mention of dropping or taking anything. Do not write any number, and do
not write "one", "two", "three" or "four" as a quantity of what an edit adds. No
markdown, no lists, no quotation marks.

Write the three sentences in order, separated by a line containing only ###, and nothing
else.

{notes}"""


# ------------------------------------------------- marking an ORDINAL claim
# The abstract arms' equivalent of `false_compare`. Two forms are checked and no others,
# because these are the two the writer actually produces and both have an unambiguous
# truth value against the block:
#
#   superlative   "Edit 3 adds the most <observation>"      -> unique argmax/argmin
#   comparative   "Edit 3 is behind Edit 1 on <observation>" -> a pairwise ordering
#
# "the only one that adds any" is left alone: it is true of a column where one edit is
# non-zero and the rest are zero, which the superlative rule already covers whenever the
# writer also says "the most", and reading it directly would need the sign as well.
_SUP_HI = r"(?:the\s+)?(?:most|highest|largest|greatest|biggest)"
_SUP_LO = r"(?:the\s+)?(?:least|lowest|smallest|fewest)"
_CMP_HI = r"(?:more|higher|larger|greater|above|ahead\s+of)"
_CMP_LO = r"(?:less|fewer|lower|smaller|below|behind|under)"
_SUP_RE = _re.compile(rf"\b({_SUP_HI}|{_SUP_LO})\b", _re.I)
_CMP_RE = _re.compile(rf"\b({_CMP_HI}|{_CMP_LO})\b", _re.I)
_HI_WORDS = {"most", "highest", "largest", "greatest", "biggest",
             "more", "higher", "larger", "greater", "above", "ahead of"}


# "Edits 2 and 3" and "Edits 1, 2 and 4" are how a passage names more than one edit, and
# `\bedit (\d+)` sees neither -- the plural has no space before the number in the first
# case and the second number is behind an "and" in both. Rewritten to the singular form
# the finders already read, rather than teaching every finder the list grammar.
# "Edits 2 and 3", "Edit 2, 3 and 4", "edits 2 3 and 4" -- all three occur and none of
# them is `\bedit (\d+)`. A bare space is allowed as a separator because a numeral
# cannot legally appear in these arms for any other reason.
_EDIT_LIST = _re.compile(
    r"\bedits?\s+(\d+(?:\s*(?:,\s*|\s+and\s+|\s+)\d+)+)", _re.I)


def _expand_edits(text: str) -> str:
    def _one(m):
        return " ".join(f"edit {d}" for d in _re.findall(r"\d+", m.group(1)))
    return _EDIT_LIST.sub(_one, text or "")


def _nums_of(vals):
    try:
        return [float(v) for v in vals]
    except (TypeError, ValueError):
        return None


def live_sets(prompt: str, n_cand: int = 4):
    """The candidates still standing at each of the three joints.

    The second passage weighs three edits and the third weighs two, so "the least of the
    remaining" is a claim about a SUBSET. Marking it against all four fails a true
    sentence -- which, since this is also the generation gate, would re-roll correct
    spans and push the corpus off whichever column the dropped edit happened to lead.
    """
    drops = [int(x) - 1 for x in _re.findall(r"Dropping Edit (\d+);", prompt or "")]
    live = list(range(n_cand))
    out = [list(live)]
    for d in drops[:2]:
        live = [i for i in live if i != d]
        out.append(list(live))
    while len(out) < 3:
        out.append(list(live))
    return out[:3]


def targets_of(prompt: str):
    """The edit each of the three joints is about: drop1, drop2, then the commit."""
    d = [int(x) - 1 for x in _re.findall(r"Dropping Edit (\d+);", prompt or "")][:2]
    t = _re.findall(r"Taking Edit (\d+):", prompt or "")
    while len(d) < 2:
        d.append(-1)
    return d + [int(t[-1]) - 1 if t else -1]


def false_ordinal(text: str, own: dict, live=None) -> bool:
    """True if a passage puts an edit somewhere the block does not put it.

    Only sentences that name exactly one observation are marked: with two in a sentence
    there is no way to say which one the superlative belongs to, and guessing would fail
    the writer for a sentence that is true. `live` restricts the comparison to the edits
    still standing at this joint.
    """
    t = _expand_edits((text or "").lower())
    for sent in _MSENT.split(t):
        labs = [lb for lb in own if lb in sent]
        if len(labs) != 1:
            continue
        v = _nums_of(own[labs[0]])
        if v is None:
            continue
        keep = [i for i in (live if live is not None else range(len(v)))
                if 0 <= i < len(v)]
        if len(keep) < 2:
            continue
        eds = [int(x) - 1 for x in _re.findall(r"\bedit (\d+)", sent)]
        eds = [e for e in eds if e in keep]
        m = _SUP_RE.search(sent)
        if m and len(set(eds)) == 1:
            e = eds[0]
            hi = m.group(1).split()[-1] in _HI_WORDS
            vv = [v[i] for i in keep]
            best = max(vv) if hi else min(vv)
            if v[e] != best or vv.count(best) > 1:
                return True
            continue
        m = _CMP_RE.search(sent)
        # EXACTLY two, because a survey sentence names every edit: "Edit 2, Edit 3 and
        # Edit 4 all add more than Edit 1, which adds the least" compares a GROUP with
        # one edit, and reading its first two mentions as the compared pair marked a
        # true sentence false.
        if m and len(set(eds)) == 2 and eds[0] != eds[1]:
            a, b = eds[0], eds[1]
            hi = m.group(1).split()[-1] in _HI_WORDS
            if v[a] == v[b] or (v[a] > v[b]) != hi:
                return True
    return False


_STARTS_EDIT = _re.compile(r"^\W*edits?\s+\d", _re.I)


def starts_with_edit(text: str) -> bool:
    """A passage that opens by naming an edit.

    MEASURED at 1.000 of passages across n1a, n7 and n7r before this gate existed: every
    single one read "Edit 4 removes the fewest heavy atoms ...". At training time that is
    a span whose first token commits to an edit and whose remaining eighteen words
    explain the commitment -- so at test time the student must choose before it can
    reason, and the reasoning cannot be what does the choosing. The order of the two
    clauses is the whole mechanism this family is testing.
    """
    return bool(_STARTS_EDIT.match((text or "").strip()))


_EDIT_N = _re.compile(r"\bedit\s+(\d+)", _re.I)


def survey_ok(text: str, live, target=None) -> bool:
    """Does the passage place EVERY edit still standing? With `target`, must it also be
    the last one named?

    The gate that makes the passage a comparison rather than a verdict with a reason
    attached. Two failures it catches, and both were the whole of what the writer
    produced before it existed:

      "Edit 4 removes the fewest heavy atoms"  -- the decision is the first token and
      the eighteen words after it are the explanation. Measured at 1.000 of passages,
      across every arm in this family.

      "Edit 4 takes off less than the others"  -- the other three are never placed, so
      nothing was compared; "the others" is a word, not a comparison.

    TARGET-LAST IS OFF FOR n7/n7r, and the reason is worth recording. It and the no-
    numbers rule are the two expensive clauses -- 52.4% and 55.6% of single attempts fail
    one of them -- and together they leave a 13.2% pass rate, because pool_A's values are
    small integers and placing four edits on one without writing a numeral takes a
    sentence the writer will not reliably produce. Keeping both would have cost 15
    re-rolls and seven hours of generation. Of the two, no-numbers was kept.
    """
    got = [int(x) - 1 for x in _EDIT_N.findall(_expand_edits((text or "").lower()))]
    if not got:
        return False
    if target is not None and got[-1] != target:
        return False
    return set(live) <= set(got)


# "adds two", "removes one" -- the value written as a word, which the digit rule cannot
# see. Restricted to a numeral directly after a verb of change, because the same words
# count CANDIDATES all over a survey sentence ("the other three", "among the four", "the
# only one that") and those are not values.
_NUMWORD = _re.compile(
    r"\b(?:adds?|adding|removes?|removing|takes?\s+off|gains?|loses?|"
    r"contributes?|brings?)\s+(?:in\s+)?"
    r"(one|two|three|four|five|six|seven|eight|nine|ten)\b", _re.I)


def has_numword(text: str) -> bool:
    return bool(_NUMWORD.search(text or ""))


def has_number(text: str, catalog) -> bool:
    """A digit that is not part of `Edit N` and not inside an observation's own name.

    `sp3-carbon change` carries a 3 and is a legitimate thing to write, so the labels
    come out before the check rather than the check tolerating stray digits.
    """
    t = _expand_edits((text or "").lower())
    for lb in catalog:
        if lb in t:
            t = t.replace(lb, " ")
    t = _re.sub(r"\bedit\s+\d+", " ", t)
    # A digit glued to a letter is part of a chemical token, not a quantity: the writer
    # paraphrases `sp3-carbon change` as "sp3 carbons", which the label strip above
    # cannot catch, and rejecting it both wasted generations and pushed the corpus off
    # that column. A quoted VALUE always stands on its own or behind a sign.
    return bool(_re.search(r"(?<![A-Za-z])[-+]?\d", t))


# --------------------------------------------------- scoring a STUDENT's observations
# The three questions the post-hoc analysis of n1/n2 had to answer, moved in-loop so an
# arm reports them per epoch instead of after it has finished:
#
#   does the student name an observation at all   -- 4-11% per joint, and this is the
#                                                    binding constraint, not accuracy
#   is the KIND one of that round's own columns   -- n1 0.451, n2 0.377, chance 6/19
#   is the VALUE it quotes that column's value    -- A columns 0.88, fitted scores 0.67,
#                                                    and 0.22 on a rule not in training
#
# Read out of the GENERATION PROMPT rather than the occlusion dump: the prompt already
# carries the block as rendered, so this needs no npz, no bucket_span and no second
# source that could drift from what the corpus actually said.
_OBS_ROW = _re.compile(r"^\s{2}(?P<lab>.+?):\s{2}(?P<vals>.+)$")
_HEAD_KV = _re.compile(rb'^\{"group_id":\s*"(?P<g>[^"]*)",\s*"depth":\s*(?P<d>\d+)')
_MNUM = _re.compile(r"[-+]?\d+(?:\.\d+)?")
_MEDIT = _re.compile(r"\bedit \d+", _re.I)
_MSENT = _re.compile(r"(?<=[.!?])\s+")


def obs_of_prompt(prompt: str) -> dict:
    """{label -> [value per edit]} for the block one generation prompt printed."""
    body = prompt.split(f"## {OBS_HEAD}", 1)
    if len(body) < 2:
        return {}
    body = body[1].split("\n## ", 1)[0]
    out = {}
    for line in body.split("\n"):
        m = _OBS_ROW.match(line)
        if not m or m.group("lab").startswith("("):
            continue
        out[m.group("lab").strip().lower()] = [v.strip().lstrip("+")
                                               for v in m.group("vals").split("|")]
    return out


def load_obs(work: str, arm: str, split: str, keys) -> dict:
    """The observation block of every round in `keys`, off nat_prompt.

    Streams the shards and pulls the id off the head of the line, so the 7,000 rounds an
    eval tick scores do not cost a json.loads over the 100,000 in the split.
    """
    arm = arm[len("model_"):] if arm.startswith("model_") else arm
    need = {(str(g), int(d)) for g, d in keys}
    out = {}
    for path in sorted(glob.glob(f"{work}/nat_prompt/{arm}/{split}/*.jsonl")):
        with open(path, "rb") as fh:
            for raw in fh:
                m = _HEAD_KV.match(raw)
                if m is None:
                    continue
                k = (m.group("g").decode(), int(m.group("d")))
                if k not in need:
                    continue
                d = json.loads(raw)
                # A RENDER arm (n8/n8r/n9/n9r) writes no generation prompt: nothing was
                # ever sent to a model, so the block travels on `own` instead. Reading
                # `own` first also keeps the two paths from drifting -- it is the very
                # dict `_obs_values` handed the renderer.
                out[k] = d["own"] if d.get("own") else obs_of_prompt(d.get("prompt", ""))
    return out


def obs_stats(text: str, own: dict, catalog) -> dict:
    """What one rollout says about observations. `catalog` is every label in the pool,
    longest first, so "fragment TPSA change" is not counted inside the per-heavy one."""
    t = _re.sub(r"<think>.*?</think>", " ", text or "", flags=_re.S).lower()
    t = t.split("each edit, at one sigma", 1)[-1]
    n_ment = n_own = n_val = n_val_ok = 0
    for sent in _MSENT.split(t):
        seen = []
        nums = [z.lstrip("+") for z in _MNUM.findall(_MEDIT.sub("  ", sent))]
        for lb in catalog:
            if lb not in sent or any(lb in s for s in seen):
                continue
            seen.append(lb)
            n_ment += 1
            tv = own.get(lb)
            if tv is not None:
                n_own += 1
                if nums:
                    n_val += 1
                    n_val_ok += int(any(z in tv for z in nums))
    return {"mention": int(n_ment > 0), "own": int(n_own > 0), "n_mentions": n_ment,
            "n_own": n_own, "n_val": n_val, "n_val_ok": n_val_ok}


def tbl_stats(text: str, own: dict, n_cand: int = 4) -> dict:
    """What a rollout wrote back of the OBSERVATION TABLE (n9/n9r).

    `obs_stats` above scores prose -- it hunts labels sentence by sentence and asks
    whether any number nearby matches. That is the wrong instrument here: n9 prints a
    fixed block the student has to reproduce verbatim, so what matters is whether the
    block is there, whether its rows are this round's rows, and whether each of the
    4 x rows CELLS carries the right value. A row copied with one wrong cell is a
    different failure from a row invented whole, and prose metrics cannot tell them
    apart.

    rows_ok    rows named that are this round's own
    rows_extra rows named that are not (a fabricated observation)
    cells      cells printed on an own row
    cells_ok   of those, the ones equal to the truth, string-compared after the same
               `+`-strip the renderer applies -- an exact transcription, not a tolerance
    """
    t = text or ""
    t = _re.sub(r"<think>.*?</think>", " ", t, flags=_re.S)
    if TABLE_HEAD not in t:
        return {"present": 0, "n_rows": 0, "rows_ok": 0, "rows_extra": 0,
                "cells": 0, "cells_ok": 0, "all_ok": 0,
                "ann": 0, "ann_n": 0, "ann_own": 0, "ann_match": 0}
    body = t.split(TABLE_HEAD, 1)[1]
    if own and all(not v for v in own.values()):
        # NAMES ONLY (n10/n10c): there are no values to score, so a row is a bare label
        # and `cells` stays 0 -- tbl_cell is meaningless here and reads 0.0 by
        # construction. tbl_own and tbl_all are the numbers these arms exist for.
        n_rows = rows_ok = rows_extra = 0
        for line in body.split("\n"):
            if not line.strip():
                continue
            lab = line.strip().lower()
            if line[:2] != "  " or ":" in line:
                break
            n_rows += 1
            if lab in own:
                rows_ok += 1
            else:
                rows_extra += 1
        return {"present": 1, "n_rows": n_rows, "rows_ok": rows_ok,
                "rows_extra": rows_extra, "cells": 0, "cells_ok": 0,
                "all_ok": int(rows_extra == 0 and rows_ok == len(own)),
                "ann": 0, "ann_n": 0, "ann_own": 0, "ann_match": 0}
    # THE FIRST ELEMENT IS WHATEVER FOLLOWED THE HEADING ON ITS OWN LINE -- empty for
    # n9, and n11's announced name list. Dropping it is right for both: without this the
    # n11 scan breaks on that remainder before reaching a single row, and every table
    # metric reads 0 on an arm whose tables are in fact fine.
    body_lines = body.split("\n")
    ann = [x.strip().lower() for x in body_lines[0].split(N11_JOIN) if x.strip()]
    # The table runs until the first line that is not an indented "label:  a | b | c"
    # row -- in the rendered frame that is `Dropping Edit ...`.
    n_rows = rows_ok = rows_extra = cells = cells_ok = 0
    seen = []
    for line in body_lines[1:]:
        if not line.strip():
            continue
        m = _OBS_ROW.match(line)
        if m is None:
            break
        n_rows += 1
        lab = m.group("lab").strip().lower()
        seen.append(lab)
        vals = [v.strip().lstrip("+") for v in m.group("vals").split("|")]
        tv = own.get(lab)
        if tv is None:
            rows_extra += 1
            continue
        rows_ok += 1
        for i in range(min(len(vals), len(tv))):
            cells += 1
            cells_ok += int(vals[i] == str(tv[i]).lstrip("+"))
    all_ok = int(rows_extra == 0 and rows_ok == len(own)
                 and cells_ok == cells == len(own) * n_cand)
    # n11 announces its three names on the heading line before writing a value. Two
    # things are worth separating there and `rows_ok` separates neither: whether the
    # ANNOUNCED names are the teacher's -- the selection, now made as one joint decision
    # rather than three interleaved with values -- and whether the rows the model then
    # writes are the ones it just announced. A high ann_own with a low ann_match would
    # mean the announcement is decoration the model does not itself follow, which is a
    # different failure from choosing badly.
    return {"present": 1, "n_rows": n_rows, "rows_ok": rows_ok,
            "rows_extra": rows_extra, "cells": cells, "cells_ok": cells_ok,
            "all_ok": all_ok, "ann": int(bool(ann)), "ann_n": len(ann),
            "ann_own": sum(1 for lb in ann if lb in own),
            "ann_match": int(bool(ann) and ann == seen)}


# --------------------------------------------------------------------------- #
def _needs_block(rnd) -> str:
    un = bs.direction(rnd)
    body = ", ".join(f"{p} ({d})" for p, d in un) if un else "nothing"
    return f"## What is out of range now\n{body}.\n"


def _band_block(rnd) -> str:
    L = ["## Where each edit lands, at one sigma of its predicted shift"]
    for i, c in enumerate(rnd["candidates"]):
        h, r, w, m = bs.classify(rnd, c)
        parts = []
        if h: parts.append("holds " + ", ".join(h))
        if r: parts.append("at risk " + ", ".join(r))
        if w: parts.append("within reach " + ", ".join(w))
        if m: parts.append("misses " + ", ".join(m))
        L.append(f"  Edit {i + 1} | " + " | ".join(parts))
    return "\n".join(L) + "\n"


def _table_lines(occ, meta, cols, n_cand: int):
    """The observation table as the STUDENT has to write it back.

    One line per observation, four values in edit order, the same shape the generation
    prompts print -- so one parser reads the corpus, the prompt and the rollout. No
    header note: every token here is a token the student must reproduce on every round,
    and "(one line per observation; the values are Edit 1 | ...)" is twenty of them.
    """
    pos = {meta["rule_names"][j]: k for k, j in enumerate(occ["col_idx"])}
    lab = meta["labels"]
    out = [TABLE_HEAD]
    for c in cols:
        if c not in pos:
            continue
        vals = " | ".join(bs.shown(c, occ["col_v"][i][pos[c]]) for i in range(n_cand))
        out.append(f"  {lab.get(c, c.replace('_', ' '))}:  {vals}")
    return out


def _obs_values(occ, meta, cols, n_cand: int) -> dict:
    """{label -> [value per edit]} straight off the occlusion row -- what `obs_of_prompt`
    reads back out of a printed block, for the arms whose prompt no longer prints one."""
    pos = {meta["rule_names"][j]: k for k, j in enumerate(occ["col_idx"])}
    lab = meta["labels"]
    out = {}
    for c in cols:
        if c not in pos:
            continue
        out[lab.get(c, c.replace("_", " ")).lower()] = [
            bs.shown(c, occ["col_v"][i][pos[c]]).lstrip("+") for i in range(n_cand)]
    return out


def _obs_block(occ, meta, cols, n_cand: int, rest=()) -> str:
    """Column-major: one line per observation, the four values in edit order.

    Row-major (v36's panel) puts every column on every edit's line, which reads fine at
    six columns and not at the hundred-odd n3 prints. Column-major also puts the four
    values of ONE observation next to each other, which is the comparison every beat
    after the second has to make.
    """
    pos = {meta["rule_names"][j]: k for k, j in enumerate(occ["col_idx"])}
    lab = meta["labels"]

    def _row(c):
        k = pos[c]
        vals = " | ".join(bs.shown(c, occ["col_v"][i][k]) for i in range(n_cand))
        return f"  {lab.get(c, c.replace('_', ' '))}:  {vals}"

    L = [f"## {OBS_HEAD}",
         "  (one line per observation; the values are Edit 1 | Edit 2 | Edit 3 | Edit 4"
         f"{'' if n_cand == 4 else ' -- fewer edits, fewer values'})"]
    L += [_row(c) for c in cols if c in pos]
    if rest:
        L += ["", "  (these are the same for all four edits, so they say what the edits "
                  "have in common and cannot separate them)"]
        L += [_row(c) for c in rest if c in pos]
    return "\n".join(L) + "\n"


def _cols_for(arm, occ, meta, lv, sp, pick, n_cand, targets=None):
    """(the columns the block prints, the constant ones it prints under them)."""
    over = list(range(n_cand))
    allow = (bs.pool_A if POOL_OF.get(arm) == "A" else bs.pool_104)(meta)
    if arm in ("n3", "n6"):
        names = meta["rule_names"]
        present = [names[j] for j in occ["col_idx"] if names[j] in allow]
        live = {names[occ["col_idx"][k]]
                for k in bs._live_cols(occ, meta, "primitive", None, over)}
        return ([c for c in present if c in live],
                [c for c in present if c not in live])
    if arm in ("n9", "n9r", "n10", "n10c", "n11"):
        # THE RELEVANCE FILTER IS `targets`, and it was already here -- `_live_cols` has
        # taken it since v19, where cutting unconstrained axes took them from 26.3% to
        # 0.0% and made the curve 3.4x more stable (sd 0.0032 against 0.0110). Every
        # call in this file had been passing None. Measured on val before it was turned
        # on: 29.6% of n9's printed rows were about a property the instance is not
        # judged on at all, and only 19.1% about one that was out of range. A column
        # with no property -- the 61 functional-group counts, sp3, halogen, cut points --
        # carries no relevance to lose and stays.
        if arm == "n9r":
            # a plain random draw from the whole live set, NOT the complement of the
            # teacher's three: the control is "a random three", and drawing from what
            # the teacher left would make it "the three the teacher rejected", which is
            # a different and harder-to-read claim.
            return bs.choose_cols_random(
                occ, meta, over, allow, N_V9_COLS,
                f"nat-v9r:{occ['group_id']}:{occ['depth']}", targets=targets), ()
        return bs.choose_cols_multi(occ, meta, over, allow, CRITERIA_V9,
                                    lv, sp, pick, targets=targets), ()

    if arm in ABS_ARMS:
        # ONE draw for the pair. Drawing per arm would leave the two corpora arguing
        # from different evidence as well as eliminating in a different order, and the
        # contrast this pair exists for is the order alone.
        return bs.choose_cols_random(
            occ, meta, over, allow, N_ABS_COLS,
            f"nat-abs:{occ['group_id']}:{occ['depth']}"), ()
    cols = bs.choose_cols_multi(occ, meta, over, allow, bs.CRITERIA_V36, lv, sp, pick)
    if arm in ("n2", "n2a"):
        # the matched control: same pool, same vary filter, same COUNT, same rounds.
        # n2a gets its own seed prefix so its draw is not n2's draw filtered down to
        # pool_A -- that would correlate the two controls on the columns they share.
        key = ("nat-random-a" if arm == "n2a" else "nat-random")
        cols = bs.choose_cols_random(occ, meta, over, allow, len(cols),
                                     f"{key}:{occ['group_id']}:{occ['depth']}")
    return cols, ()


def _frame(rnd, pick: int, order, n_cand: int) -> str:
    """The v34 span with the three joints marked -- rendered, not written.

    Built out of `bs.render` and `bs.drop_lines` exactly as `render_bucket.py` builds
    v34, so every line the student is trained to emit is byte-identical to the line the
    v34 corpus carries. That is the whole point of the gap form: the only difference
    between this arm and v34 is what stands in the joints.
    """
    base = bs.render(rnd, pick).split("\n")
    dl = bs.drop_lines(order, n_cand)
    if len(dl) < 2:
        return None
    return "\n".join(base[:-1] + [GAP_MARKS[0], dl[0], GAP_MARKS[1], dl[1],
                                  GAP_MARKS[2], base[-1]])


def _frame17(rnd, pick: int, order, n_cand: int) -> str:
    """n8's span with a joint in front of each elimination and NOTHING before COMMIT."""
    base = bs.render(rnd, pick).split("\n")
    dl = bs.drop_lines(order, n_cand)
    if len(dl) < 2:
        return None
    return "\n".join(base[:-1] + [GAP_MARKS[0], dl[0], GAP_MARKS[1], dl[1], base[-1]])


def _shuf_cols(cols, rnd):
    """The per-round row permutation n9 uses, so n10 carries the identical one."""
    h = _hashlib.blake2b(f"v9-order:{rnd['group_id']}:{rnd['depth']}".encode(),
                         digest_size=8).digest()
    g = int.from_bytes(h, "big")
    cols = list(cols)
    for i in range(len(cols) - 1, 0, -1):
        j = g % (i + 1); g //= 7
        cols[i], cols[j] = cols[j], cols[i]
    return cols



# ---------------------------------------------------------------- n51: two observations
# Precomputed by `scripts/n51_table.py` because the choice needs q AND the uncapped
# evidence, and re-reading 9.4 GB of it per render process would dominate the pass.
_N51_OBS = {}
# WHERE n51's OBSERVATIONS COME FROM. This used to be the study work dir, frozen as a
# literal, which is right for n51/n52 themselves and wrong for every other recipe: the
# main-model corpora have their own rounds and their own q, so their observations live in
# their own work dir. `N51_DIR` overrides it, `ARMS_RECIPE` picks it up automatically,
# and the study path stays the fallback so a bare n51 render is unchanged.
_N51_DIR = os.environ.get("N51_DIR") or (
    rp.work_dir(rp.DEFAULT) if rp.DEFAULT.version != "v231318b2cfa1"
    else "data/analysis/reasoning_arms/v231318b2cfa1")

def _n51_obs(gid, depth, arm="n51"):
    # n53/n54 REUSE n51's AND n52's PICKLES rather than selecting again -- see
    # `_OBS_SRC`. Resolved here, not at the call site, so every path that reaches an
    # observation (render, `force_span`, the audits) maps the arm the same way and a
    # stripped arm can never silently read a pickle of its own name that does not exist.
    src = _OBS_SRC.get(arm, arm)
    if src not in _N51_OBS:
        import pickle
        with open(f"{_N51_DIR}/{src}_obs.pkl", "rb") as fh:
            _N51_OBS[src] = pickle.load(fh)
    return _N51_OBS[src].get(f"{gid}|{depth}")

# Labels as noun phrases, so "The other side of it is <L>: ..." parses. The raw label is
# a caption ("does it move the worst-violated property") and reads as a question in the
# middle of a sentence.
_N51_NOUN = {
    "c_std_max": "its widest single spread",
    "c_std_sum": "the total spread across its predicted shifts",
    "c_cos": "how well its shift lines up with the direction needed",
    "c_worst_prop_delta": "how far it moves the worst-violated property",
    "c_room_frac_used": "the share of the remaining headroom it spends",
    "c_pred_gap": "the gap it leaves",
    "c_gap_reduction": "the gap it closes",
    "c_worst_after": "the worst violation it leaves behind",
    "c_dcount": "the change in satisfied-constraint count",
    "c_n_sat_post": "the constraints satisfied afterwards",
    "c_n_sat_now": "the constraints already satisfied",
    "c_cover_frac": "how much of the needed direction it covers",
    "c_overshoot": "how far it overshoots the far edge of the box",
    "c_damage": "the satisfied constraints it would break",
    "c_n_props_helped": "the constraints it moves the right way",
    "c_n_props_hurt": "the constraints it moves the wrong way",
    "c_fix_worst": "whether it moves the worst-violated property",
    "fg_n_created": "the functional groups it creates",
    "fg_n_destroyed": "the functional groups it destroys",
    "r_dsp3": "the sp3 carbons it adds",
    "r_heavy_to": "the heavy atoms it adds",
    "r_heavy_from": "the heavy atoms it removes",
    "r_dheavy": "its net heavy-atom change",
    "r_dpolar_frac": "the shift in polar-atom fraction",
    "r_dhalogen": "the halogens it adds",
    "r_dhba": "the acceptors it adds",
    "r_dhbd": "the donors it adds",
    "r_drotb": "the rotatable bonds it adds",
    "r_drings_arom": "the aromatic rings it adds",
    "r_drings_aliph": "the aliphatic rings it adds",
    "r_dmw": "the weight it adds",
    "r_n_cuts": "the number of cut points",
    "r_is_attach": "whether it only adds",
    "r_is_delete": "whether it only deletes",
    "r_is_swap": "whether it swaps",
    # NEVER SELECTED in any arm built so far (0 of 110,158 rounds in every pickle), but
    # they are IN the pool, and without an entry `_n51_noun` falls through to
    # "the h_motif_repeat" -- a raw column name in the middle of a sentence. Naming them
    # changes no existing corpus and stops the next one from printing that.
    "h_motif_repeat": "whether the fragment repeats one already in the molecule",
    "h_same_rule_as_prev": "whether it reuses the previous edit's rule",
}
_N51_BOOL = {"c_fix_worst", "r_is_attach", "r_is_delete", "r_is_swap"}

# THE VERB EACH AXIS TAKES, so n60 can write a clause per edit instead of a label run.
# `Edit 1 3, Edit 2 3` puts a number against a label with nothing between them, and on
# an integer column "Edit 1 3" reads as one token as easily as two. n60 writes
# "Edit 1 adds 3, Edit 2 adds 3, Edit 3 adds 2 and Edit 4 adds 0" -- which needs a verb,
# and the verb is a property of the AXIS, not something a template can derive: the pool
# holds counts that are added, counts that are removed, a fraction that is shifted, a
# predicted delta and a spread. One entry per name, three prefixes for the 89 columns
# that are per-property or per-functional-group.
#
# The BOOLEAN branch is untouched: "true of only Edit 1" is already a clause.
_N51_VERB = {
    "r_n_cuts": "has",
    "r_heavy_from": "removes",
    "r_heavy_to": "adds",
    "r_dheavy": "changes it by",
    "r_dmw": "adds",
    "r_drings_arom": "adds",
    "r_drings_aliph": "adds",
    "r_drotb": "adds",
    "r_dhbd": "adds",
    "r_dhba": "adds",
    "r_dhalogen": "adds",
    "r_dsp3": "adds",
    "r_dpolar_frac": "shifts it by",
    "fg_n_created": "creates",
    "fg_n_destroyed": "destroys",
    "h_motif_repeat": "scores",
    "h_same_rule_as_prev": "scores",
    # the `c_*` family is out of n57's pool and so out of n60's, but n60's renderer is
    # shared, so give them verbs rather than let a future arm fall through to "is".
    "c_std_max": "sits at", "c_std_sum": "sits at", "c_cos": "scores",
    "c_worst_prop_delta": "moves it by", "c_room_frac_used": "spends",
    "c_pred_gap": "leaves", "c_gap_reduction": "closes", "c_worst_after": "leaves",
    "c_dcount": "changes it by", "c_n_sat_post": "ends with", "c_n_sat_now": "starts with",
    "c_cover_frac": "covers", "c_overshoot": "overshoots by", "c_damage": "breaks",
    "c_n_props_helped": "helps", "c_n_props_hurt": "hurts",
}
_N51_VERB_PRE = (("r_dfg__", "adds"),          # the <group> it adds
                 ("r_dmean__", "predicts"),    # the <prop> shift it predicts
                 ("r_dstd__", "sits at"))      # the spread on its predicted <prop> shift


def _n51_verb(name):
    if name in _N51_VERB:
        return _N51_VERB[name]
    for pre, v in _N51_VERB_PRE:
        if name.startswith(pre):
            return v
    return "is"          # a column with no entry still reads as a clause

def _n51_noun(name):
    if name in _N51_NOUN:
        return _N51_NOUN[name]
    for pre, tmpl in (("r_dfg__", "the {} it adds"),
                      ("r_dmean__", "the {} shift it predicts"),
                      ("r_dstd__", "the spread on its predicted {} shift")):
        if name.startswith(pre):
            return tmpl.format(name[len(pre):])
    return "the " + name

def _n51_body(name, v, sent=False, inv_ok=False, zfix=False):
    n = len(v)                       # 4 in the study; 1..4 once short rounds are kept
    if name in _N51_BOOL and set(round(x, 6) for x in v) <= {0.0, 1.0}:
        yes = [i + 1 for i in range(n) if v[i] > .5]
        no = [i + 1 for i in range(n) if v[i] <= .5]
        if len(yes) == 1:
            return f"true of only Edit {yes[0]}"
        if len(no) == 1:
            return f"true of every edit but Edit {no[0]}"
        return ("true of Edits " + " and ".join(str(x) for x in yes)) if yes \
               else "true of none of them"
    def f(x):
        t = f"{x:g}" if abs(x - round(x)) < 1e-9 else f"{x:.3g}"
        # NEGATIVE ZERO. `-0.0 < 0` is False in python, so the verb flip below never
        # sees it and the span reads `adds -0` -- 293 of 109,928 n71 spans before this.
        # The value IS zero; only the sign survived the float.
        #
        # `zfix` IS ITS OWN GATE AND NOT `inv_ok`. The first cut reused the flip's flag
        # and moved n63, which is trained: `_FIXED_ARMS` implies the flip, so every arm
        # from n60 on would have re-rendered differently. n60..n69 keep the `-0` they
        # were built with; only an arm that opts in is corrected.
        return "0" if zfix and t in ("-0", "-0.0") else t
    if not sent:
        return ", ".join(f"Edit {i+1} {f(v[i])}" for i in range(n))
    vb = _n51_verb(name)
    # `inv_ok` USED TO BE `fix`, i.e. membership of `_FIXED_ARMS`, which also decides
    # the connective. Those are two different things and n71 wants one without the
    # other: the clause form with `removes 18` instead of `adds -18`, and n57's fixed
    # `The other side of it is` untouched. Every earlier arm passes the same value it
    # always did, so none of them move.
    inv = _INVERSE_VERB.get(vb) if inv_ok else None
    parts = []
    for i in range(n):
        if inv is not None and v[i] < 0:
            parts.append(f"Edit {i+1} {inv} {f(-v[i])}")
        else:
            parts.append(f"Edit {i+1} {vb} {f(v[i])}")
    # "a, b and c" -- the last comma replaced, so the run reads as a list of clauses
    # rather than trailing off. One candidate needs no conjunction.
    return parts[0] if n == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]

_TBL_ROW = _re.compile(r"^  Edit (\d+) \| (.*)$")


def _tbl_counts(lines):
    """-> {edit index: (n_holds, n_misses)} read back off the rendered table."""
    out = {}
    for ln in lines or ():
        m = _TBL_ROW.match(ln)
        if not m:
            continue
        cell = m.group(2)
        def _n(word):
            mm = _re.search(word + r" ([^|]*)", cell)
            return len([x for x in mm.group(1).split(",") if x.strip()]) if mm else 0
        out[int(m.group(1)) - 1] = (_n("holds"), _n("misses"))
    return out


def _table_override(lines, pick, name, v):
    """The line naming the table's preference, or None when there is no honest one.

    HONEST MEANS THREE THINGS AT ONCE, and all three are checked rather than assumed:
    some edit D dominates the commit in the table (>= holds, <= misses, one strict);
    the commit sits at a UNIQUE extreme of the first observation; and D is not at that
    same value. Miss any one and n62 renders what n60 does -- an override with no
    ground reads worse than saying nothing, and it would also be false.
    """
    S = _tbl_counts(lines)
    if pick not in S or v is None or any(x is None for x in v) or len(v) < 2:
        return None
    ph, pm = S[pick]
    dom = [i for i in S if i != pick and S[i][0] >= ph and S[i][1] <= pm
           and (S[i][0] > ph or S[i][1] < pm)]
    if not dom:
        return None
    # the strongest of them, deterministically: most holds, then fewest misses
    d = sorted(dom, key=lambda i: (-S[i][0], S[i][1], i))[0]
    hi = [i for i in range(len(v)) if v[i] == max(v)]
    lo = [i for i in range(len(v)) if v[i] == min(v)]
    if not ((hi == [pick]) or (lo == [pick])) or v[d] == v[pick]:
        return None
    dh, dm = S[d]
    # say the TRUE half of the dominance, not both halves on faith
    if dh > ph and dm < pm:
        why = "holds more of the box and misses less of it"
    elif dh > ph:
        why = "holds more of the box"
    else:
        why = "misses less of the box"
    return f"The table favours Edit {d + 1}: it {why}."


def n51_lines(gid, depth, pick, arm="n51", lines=None):
    """the observation lines and the link.

    NEVER None. 1.4% of rounds have fewer than two observations that survive the pool
    filters, and dropping them would give n51 a different record set from n49 -- the one
    arm it exists to be read against -- and take `test_items` below the 4,980 the in-loop
    eval needs. Those rounds print one line, or none, and keep the link.
    """
    got = _n51_obs(gid, depth, arm) or ()
    sent = arm in _SENT_ARMS
    fix = arm in _FIXED_ARMS
    # the verb flip on a negative value, which `_FIXED_ARMS` used to carry alone
    inv = fix or arm in _INVVERB_ARMS
    zfx = arm in _INVVERB_ARMS          # the `-0` correction, opt-in only
    out = []
    if not got:
        # NOTHING TO WEIGH, SO NO CLOSER. A one-candidate round, or one whose pool came
        # out empty, has no observation in front of the commit -- and "Weighing all of
        # that together, I'll go with Edit 1" in front of a one-row table is a claim
        # about a comparison that did not happen. `Taking Edit N:` stands on its own,
        # and inloop_eval's _COMMIT_RE already matches it.
        return []
    if len(got) >= 3:
        n1, _t1, v1 = got[0], got[1], got[2]
        if arm in _DESC_ARMS:
            # DESCRIPTION, NOT VERDICT -- and the set it names is carried to the closer.
            # The lead-in for the observation is decided here too, because it depends on
            # whether the table named anybody: after a name the observation is what the
            # tally leaves out, after "nothing separates them" it is what is left.
            txt, _win = _table_desc(lines)
            if txt:
                if txt.startswith("Nothing in the table"):
                    _lead1 = _OBS1_AFTER_FLAT
                else:
                    txt = txt[:-1] + _DESC_TAIL       # drop the period, add the clause
                    _lead1 = _OBS1_AFTER_NAMED
                out.append(txt)
            else:
                _lead1 = "I think the thing to look at here is "
            out.append(f"{_lead1}{_n51_noun(n1)}: {_n51_body(n1, v1, sent, inv, zfx)}.")
        elif arm in _TABLE2_ARMS:
            # NO `pick` ANYWHERE IN THIS BRANCH. The verdict is the table's, the opener
            # is n60's unchanged -- which is what keeps the forced prompt free of the
            # answer while still making the span reference the part that carries it.
            vd = _table_verdict(lines)
            if vd:
                out.append(vd)
        ovr = (_table_override(lines, pick, n1, v1)
               if arm in _TABLE_ARMS else None)
        if arm in _DESC_ARMS:
            pass                      # obs1 already written, with its own lead-in
        elif ovr:
            # THE CONTRAST IS THE POINT, so the opener changes with it: "I think the
            # thing to look at here is X" states a reading, "But on X, Edit N stands
            # apart" answers the line above it. Both name the same axis and the same
            # numbers -- only the discourse relation is added.
            out.append(ovr)
            out.append(f"But on {_n51_noun(n1)}, Edit {pick + 1} stands apart: "
                       f"{_n51_body(n1, v1, sent, inv, zfx)}.")
        else:
            out.append(f"I think the thing to look at here is "
                       f"{_n51_noun(n1)}: {_n51_body(n1, v1, sent, inv, zfx)}.")
    if len(got) >= 6:
        n2, _t2, v2 = got[3], got[4], got[5]
        # A SPREAD IS NOT "SEPARATE" FROM ITS OWN MEAN. With the selection filters off,
        # `r_dmean__X` followed by `r_dstd__X` is 20.3% of rounds -- the tiers put the
        # mean first, so this is the only order it occurs in -- and every one of the
        # three connectives below misdescribes it: the spread does not go the same way,
        # pull the other way, or stand separately from the shift it is the spread OF.
        # It also reads straight off the band cell above, where the interval is the
        # mean plus and minus exactly this number. So the pair gets its own line.
        # GATED TO `_DESC_ARMS`, THE FAMILY IT WAS WRITTEN FOR. The first cut gated it
        # on `fix`, which is `_FIXED_ARMS` and therefore includes n60..n63 -- all
        # trained. Re-rendering n63 then moved one span in 159, silently, which is the
        # same drift the settings sidecars exist to catch.
        if (arm in _DESC_ARMS and n1.startswith("r_dmean__")
                and n2.startswith("r_dstd__")
                and n1[len("r_dmean__"):] == n2[len("r_dstd__"):]):
            out.append(f"And the spread on that shift: {_n51_body(n2, v2, sent, inv, zfx)}.")
        else:
            if fix:
                # THE CONNECTIVE FOLLOWS THE NUMBERS. `_pearson` over the candidates is
                # the same quantity the dedup used to threshold, so this costs nothing
                # and stops the span asserting a contrast its own values deny.
                r = _pearson(v1, v2)
                lead = ("It goes the same way for" if r > 0.2 else
                        "Pulling the other way is" if r < -0.2 else
                        "Separately, there is")
            else:
                lead = "The other side of it is"
            out.append(f"{lead} {_n51_noun(n2)}: {_n51_body(n2, v2, sent, inv, zfx)}.")
    if len(got) >= 9:
        # n59's third slot. "Against that" and "on the other hand" both assert a
        # DIRECTION between the readings, and the selection does not establish one --
        # the dedup only says this column is not the previous one restated. So the
        # opener adds a reading without ranking it, the way the second line does.
        n3, _t3, v3 = got[6], got[7], got[8]
        out.append(f"There is also {_n51_noun(n3)}: {_n51_body(n3, v3, sent, inv, zfx)}.")
    if arm in _DESC_ARMS:
        # THE CONCESSION, ON THE CLOSER LINE. Honest only where the table really does
        # point elsewhere, so it is computed from the table and compared with the
        # commit; where they agree the closer is the one every other arm writes.
        _t, win = _table_desc(lines)
        # The flat case returns EVERY edit, so `pick not in win` is already False
        # there -- no candidate-count guard is needed and none is hard-coded.
        against = bool(win) and pick not in win
        return out + [(_CLOSE_AGAINST if against
                       else _CLOSE_WITH).format(n=pick + 1)]
    return out + [
            # NAMES THE COMMIT, CLAIMS NOTHING ABOUT THE TWO READINGS. They point at the
            # committed edit on only part of the corpus -- picked == argmax(q) is 0.378
            # and the selection is built around argmax(q) -- so "both of those favour it"
            # would be false more often than true. It also has to miss
            # force_span._TAKE (`^Taking Edit N:` / `I take Edit N:`).
            f"Weighing all of that together, I'll go with Edit {pick + 1}."]

def build_prompt(arm, rnd, occ, meta, lv, sp, q, fg=None):
    """One arm's full generation prompt for one round, or None if the round is unusable."""
    # ARMED FIRST, because `_frag_groups` is called from six places below and every one
    # of them has to see the same catalog. `one_split` renders one arm per process.
    global _FG_FILTER
    _FG_FILTER = arm in _FGF
    pick = int(rnd["picked"])
    n_cand = len(rnd["candidates"])
    if arm in (("n8", "n8r") + N20_ARMS + N21_ARMS + N22_ARMS + N23_ARMS
               + N24_ARMS + N48_ARMS + N49_ARMS + N50_ARMS + N51_ARMS + N52_ARMS
               + N53_ARMS + N54_ARMS + N55_ARMS + _N25F + _N40F):
        # n20 selects no columns for the same reason n8 does not: its block is built
        # from the round, not from the occlusion draw, and going through `_cols_for`
        # would drop rounds on a gate this arm never reads.
        cols, rest = [], ()
    else:
        cols, rest = _cols_for(arm, occ, meta, lv, sp, pick, n_cand,
                               targets=set(rnd.get("targets") or ()))
        if not cols:
            return None
    order = None
    molfg, crit, out = {}, None, []
    if arm in (N19_ARMS + N20_ARMS + N21_ARMS + N22_ARMS + N23_ARMS + N24_ARMS
               + N48_ARMS + N49_ARMS + N50_ARMS + N51_ARMS + N52_ARMS
               + N53_ARMS + N54_ARMS + N55_ARMS + _N25F + _N40F):
        molfg = fg_keep({e["name"]: int(e["count"])
                         for e in ((fg or {}).get("fg") or [])})
    if arm in N19_ARMS:
        crit, out = n19_criterion(rnd, pick, molfg)
        order = n19_order(rnd, pick, out, f'{rnd["group_id"]}:{rnd["depth"]}')
        if len(order) < 2:
            return None
    elif arm in ("n15",) + N16_ARMS + N18_ARMS:
        order = band_drop_order(rnd, pick, f'{rnd["group_id"]}:{rnd["depth"]}')
        if len(order) < 2:
            return None
    elif arm != "n6":
        if _ORDER_OF.get(arm) == "anti":
            order = anti_q_order(q, pick, n_cand,
                                 f'{rnd["group_id"]}:{rnd["depth"]}')
            order = order[:2] if order else []
        else:
            # n43 is n40 with q taken out of the ELIMINATION ORDER and nothing else:
            # same rounds, same commit, same two rivals set aside, drawn in a seeded
            # order instead of weakest-q-first. n40 - n43 is what the order is worth.
            order = (bs.drop_order_random(n_cand, pick,
                                          f'{rnd["group_id"]}:{rnd["depth"]}')
                     if arm in ("n4",) + N43_ARMS
                     else bs.drop_order(q, pick))[:2]
        if len(order) < 2 and not (_ALLOW_SHORT and arm in _NODROP):
            # Two rivals to set aside means three candidates. Below that there is no
            # elimination to draw -- and a nodrop arm never prints one, so requiring it
            # only threw the round away. See NAT_ALLOW_SHORT.
            return None
    a = rnd["committed"]
    call = dict(pick_no=pick + 1, from_smiles=a.get("from_smiles"),
                to_smiles=a.get("to_smiles"), anchors=a.get("anchors"))
    info = {"cols": cols, "drops": list(order or ()), "pick": pick}

    if arm in RENDER_ARMS:
        # No prompt at all: the span IS the frame, with the joint markers taken out and,
        # for n9/n9r, the observation table standing where the passages would have been.
        frame = _frame(rnd, pick, order, n_cand)
        if frame is None and _ALLOW_SHORT and arm in _NODROP:
            # A ROUND WITH FEWER THAN THREE CANDIDATES CANNOT BE DROPPED TWICE, and the
            # nodrop arms throw the eliminations away three lines below anyway -- the
            # gate was there so n26/n49/n51 kept the same record set as the arms they
            # are read against, which is a study concern and not a corpus one. With
            # ARMS_RECIPE=main_* the corpus keeps its 1.17%/0.36% short rounds instead,
            # and they render as a two- or one-row table.
            frame = "\n".join(bs.render(rnd, pick).split("\n"))
        if frame is None:
            return None
        lines = [x for x in frame.split("\n") if x not in GAP_MARKS]
        if arm in _INTERVAL_ARMS:
            # BEFORE the fg/count cells go on, so the row body is still the band cells
            # alone and the rewrite cannot eat a cell it does not own.
            lines = _interval_table(rnd, lines)
        if arm in _ONEDROP:
            # KEEP THE FIRST ELIMINATION, DROP THE SECOND. Taken out AFTER `_frame`, so
            # the round is still gated on having drawn two and n45f trains on exactly the
            # records n40-n44 do. `info["drops"]` is trimmed with it, because every
            # downstream diagnostic reads the drop count from there.
            seen, keep = 0, []
            for x in lines:
                if _N21_DROP.match(x):
                    seen += 1
                    if seen > 1:
                        continue
                keep.append(x)
            lines = keep
            info["drops"] = list(order or ())[:1]
        if arm in _NODROP:
            # The ablation, and the only line of it. Taken out AFTER `_frame` so the
            # round is still gated on having drawn two eliminations and n26 keeps the
            # same corpus as everything it is read against.
            lines = [x for x in lines if not _N21_DROP.match(x)]
            info["drops"] = []
        if arm in (N19_ARMS + N20_ARMS + N21_ARMS + N22_ARMS + N23_ARMS
                   + N24_ARMS + N48_ARMS + N49_ARMS + N50_ARMS + N51_ARMS
                   + N52_ARMS + N53_ARMS + N54_ARMS + N55_ARMS + _N25F + _N40F):
            # THE GATE READS THE UNFILTERED RECORD, ON PURPOSE. `fg_line_all` returns
            # None for a molecule with no catalog group, and `build_prompt` drops that
            # round. On a filtered arm a molecule whose ONLY groups are in `FG_DROP`
            # would empty out and the round would vanish -- 1 in 2,000, enough to give
            # the f-arms a different record set from the arms they are read against.
            # So the gate stays on what the molecule really has; only the printed line
            # is filtered. (The n40 family rebuilds the page and never prints `fgl` at
            # all -- its MOL_INFO comes from `n40_molecule`, which says "carries no
            # group the catalog names" for the empty case.)
            if bs.fg_line_all(fg) is None:
                if not _ALLOW_SHORT:
                    return None
                # ~1 round in 2,000 has a molecule with no group the catalog names. In
                # the study that round is dropped, to keep the record sets identical; a
                # training corpus says so instead, in n40's words for the same case.
                fg = None
            fgl = bs.fg_line_all(
                {"fg": [e for e in ((fg or {}).get("fg") or [])
                        if e["name"] not in FG_DROP]} if _FG_FILTER else fg) or FG_NONE
            for j, ln in enumerate(lines):
                if ln.startswith("  Edit ") and arm not in _NOCELLS:
                    k = int(ln.split("|")[0].split()[1]) - 1
                    dv = _atom_delta(rnd, k)
                    # heavy atoms is the sum of the element counts and rotatable
                    # bonds rarely separates anything -- both make the row longer
                    # without making it say more.
                    # WITH THE SIGN, on n25. Printing only the positives made the
                    # cell unable to tell 0 from -1, and the whole count family is
                    # checked against it: a row reading `+1 halogen` is as consistent
                    # with `adds no oxygen` as with taking an oxygen away, and `adds
                    # the fewest carbon` cannot be settled either when a rival's -2 is
                    # invisible. Negatives are 3.9% of carbon cells and 0.6% of rings.
                    sgn = arm in _N25F + _N40F
                    got = ", ".join(f"{dv[x]:+d} {x}" if sgn else f"+{dv[x]} {x}"
                                    for x in _N19_CELL
                                    if (dv[x] != 0 if sgn else dv[x] > 0))
                    lines[j] = ln + " | " + fg_row(rnd, k, molfg)
                    if arm not in N24_ARMS:
                        lines[j] += " | " + (got or "adds no atoms")
            lines.insert(1, fgl)
            # The block goes in front of the first DECISION -- the first elimination
            # everywhere but n26, which has none, so there the commit is the first thing
            # the span decides. Keyed on `Dropping` alone this raised StopIteration on
            # every n26 round.
            if arm in N51_ARMS + N52_ARMS + N55_ARMS + _OBSONLY:
                # Straight in front of the commit, after the table. A round with fewer
                # than two usable observations is DROPPED rather than printed short, so
                # n51 trains on one record set and every line it prints is the same
                # shape.
                obs = n51_lines(rnd["group_id"], rnd["depth"], pick, arm, lines)
                if arm in _NOOBS:
                    # The closer alone. On a round whose pool came out empty `obs` is
                    # already [] and there is no closer to keep, exactly as on n72.
                    obs = obs[-1:]
                t = next(i for i, x in enumerate(lines) if x.startswith("Taking Edit "))
                lines[t:t] = obs
                if arm in _OBSONLY:
                    # EVERYTHING ABOVE THE OBSERVATIONS GOES: the out-of-range line, the
                    # group line and the whole three-cell table. `t` is where the block
                    # was just inserted, so the slice is exactly the observations, the
                    # link and the commit -- there is nothing after the commit in a
                    # rendered frame.
                    #
                    # THE GATES ABOVE STAY ON. n53/n54 are in the same fg and
                    # short-round tuples n51/n52 are, and the table cells are still
                    # built, because the point of the pair is that n51 - n53 differs in
                    # WHAT IS PRINTED and in nothing else. Dropping a round here that
                    # n51 keeps would cost the paired contrast and take `test_items`
                    # below the 4,980 the in-loop eval reads.
                    #
                    # 19 rounds in 110,158 have no observation at all and 1.4% have one;
                    # those print a bare commit, or one line and the link, exactly as
                    # they do on n51, where `n51_lines` is documented never to be None
                    # for this same reason.
                    lines = lines[t:]
            head = next(i for i, x in enumerate(lines)
                        if x.startswith(("Dropping Edit ", "Taking Edit ")))
            if arm in N20_ARMS + N21_ARMS + N23_ARMS + _N25F + _N40F:
                # One row per edit, in the table's own order, whether or not the edit
                # has anything to say. A block that silently omits the edits with no
                # sole fact would make its own LENGTH a signal, and length is not an
                # observation -- it is a count of how identifiable the round is, which
                # is exactly the kind of thing the student cannot recompute.
                # n40 takes n21's block exactly -- cap 4, no `within reach` facts --
                # so that its prose SUMMARY is the same content n21 prints as rows and
                # the two arms differ in layout alone.
                feat = n20_features(
                    rnd, molfg,
                    N25_FACTS if arm in _N25F else
                    N23_FACTS if arm in N23_ARMS else N20_FACTS,
                    reach_out=arm in _N25F)
                # Every edit is answered for, including on the 1.6% of rounds where
                # the printed rows separate none of the four and the block is four
                # denials. Dropping the block there would make its ABSENCE the round's
                # hardest signal, and an arm cannot be measured on spans it declines to
                # write.
                if arm not in _NOBLOCK:
                    lines[head:head] = [N20_HEAD] + [
                        N20_ROW.format(n=i + 1, ph="; ".join(feat[i]) or N20_NONE)
                        for i in range(n_cand)]
                info["n20_feat"] = [feat[i] for i in range(n_cand)]
                if arm in N21_ARMS + _N25F:
                    # SAME fact, same polarity filter, opposite side of the line. n21
                    # appends `It {ph}.` to the whole line -- see its header: after the
                    # index, so the clause cannot reach the token it justifies, and
                    # after "that leaves", so `bucket_span._DROP` still matches. n25
                    # puts the reason on the line ABOVE instead, which is the one other
                    # placement that leaves that anchor alone.
                    lead = arm in _N25F
                    unp = {p_ for p_, _w in bs.direction(rnd)}
                    Fx, cl, ins, nd = n19_facts(rnd, molfg, lead), [], [], 0
                    for j, ln in enumerate(lines):
                        md = _N21_DROP.match(ln)
                        mt = _N21_TAKE.match(ln) if md is None else None
                        if md is None and mt is None:
                            continue
                        e = int((md or mt).group(1)) - 1
                        side = "drop" if md else "commit"
                        if arm in N35_ARMS:
                            # The live set is the pool this decision is taken from: all
                            # four at the first elimination, minus what the earlier
                            # lines already took out. Listing a dropped edit would
                            # contradict the page above it.
                            live = [x for x in range(n_cand)
                                    if x not in list(order or ())[:nd]]
                            ld, ax, grp = n35_line(rnd, molfg, unp, Fx, feat, e,
                                                   side, live)
                            ins.append((j, ld))
                            cl.append((ax, e, len(grp)))
                            nd += bool(md)
                            continue
                        if arm in _N28SEL:
                            # EXCLUSIVE facts only, and a frame for all three cases, so
                            # this branch never falls through to a shared fact and never
                            # leaves a decision line bare.
                            pos = ("commit" if mt else
                                   "drop1" if nd == 0 else "drop2")
                            kb, mode = n28_pick(rnd, molfg, unp, Fx, feat, e, side)
                            ins.append((j, n28_lead(kb, molfg, unp, e, side, mode, pos)))
                            cl.append((mode, e, None if kb is None else Fx[e][kb][0]))
                            nd += bool(md)
                            continue
                        kb = n21_clause(
                            rnd, molfg, unp, Fx, feat, e, side,
                            block_first=not lead, tie=lead,
                            reads_negative=_n25_reads_negative if lead else None)
                        # Nothing is sayable against it, so the line grants the best
                        # thing that IS true and drops the edit anyway. n21 stays silent
                        # here -- after the index that costs nothing, but n25's line
                        # comes BEFORE the index and a missing one is a hole in the page.
                        if kb is None and lead and side == "drop":
                            kb = n25_concede(rnd, molfg, unp, Fx, feat, e,
                                             tie=True)
                            side = "concede"
                        ph = None if kb is None else Fx[e][kb][0]
                        if kb is not None and lead:
                            pos = ("commit" if mt else
                                   "drop1" if nd == 0 else "drop2")
                            ld = n25_lead(kb, molfg, unp, e,
                                          "keep" if mt else side, pos)
                            if ld:
                                ins.append((j, ld))
                        elif ph:
                            lines[j] = f"{ln} It {ph}."
                        nd += bool(md)
                        cl.append((side, e, ph))
                    # Back to front, so an insertion does not move the joints after it.
                    for j, ld in reversed(ins):
                        lines.insert(j, ld)
                    info["n21_clauses"] = cl
                if arm in _N40F:
                    # The whole page, rebuilt. `_frame` is still what produced the
                    # decisions -- the same eliminations in the same order -- but every
                    # line of it is re-said, so the arm cannot inherit a table it does
                    # not mean to print.
                    unp = {p_ for p_, _w in bs.direction(rnd)}
                    Fx = n19_facts(rnd, molfg, False)
                    key = f'{rnd["group_id"]}:{rnd["depth"]}'
                    dec, cl, live, nd = [], [], list(range(n_cand)), 0
                    for ln in lines:
                        md = _N21_DROP.match(ln)
                        mt = _N21_TAKE.match(ln) if md is None else None
                        if md is None and mt is None:
                            continue
                        e = int((md or mt).group(1)) - 1
                        side = "drop" if md else "commit"
                        pos = ("commit" if mt else
                               "drop1" if nd == 0 else "drop2")
                        smarts = None if md else ln.split(": ", 1)[1][:-1]
                        if arm in N44_ARMS + N45_ARMS + N46_ARMS + N47_ARMS:
                            rs, con, nd_ = n44_reasons(
                                rnd, molfg, unp, Fx,
                                ({} if arm in _FREEREASON else feat),
                                e, side, key, pos, block=arm not in _FREEREASON)
                            dec.append(n44_line(e, side, pos, live, rs, con, nd_,
                                                smarts,
                                                ordinal=arm not in _ONEDROP))
                            cl.append((pos, e, rs, con))
                        else:
                            rs = n40_reasons(rnd, molfg, unp, Fx, feat, e, side,
                                             key, pos)
                            dec.append(n40_decision(e, side, pos, live, rs, smarts))
                            cl.append((pos, e, rs))
                        if md:
                            live.remove(e)
                            nd += 1
                    page = [] if arm in N41_ARMS else (
                        [n40_molecule(rnd, molfg),
                         n40_analysis(rnd, molfg, n_cand)]
                        + ([] if arm in _NOSUMMARY
                           else [n40_summary(feat, n_cand)]))
                    lines = page + dec
                    info["n40_reasons"] = cl
            elif crit:
                lines.insert(head, crit)
            if arm in N19_ARMS:
                info["n19_crit"] = crit
                info["n19_excl"] = out
            info["span"] = "\n".join(lines)
            info["frame"] = frame
            return None, info
        if arm in NAME_ARMS:
            # NAMES ONLY, one per line, shuffled the same way n9 shuffles its rows so the
            # first name is not always cell_m's -- molecular weight on 54% of rounds --
            # which would let the arm score on a positional habit rather than a choice.
            cols = _shuf_cols(cols, rnd)
            info["cols"] = cols
            lab = meta["labels"]
            block = [NAME_HEAD] + [f"  {lab.get(c, c.replace('_', ' '))}" for c in cols]
            if arm == "n10":
                head = next(i for i, x in enumerate(lines)
                            if x.startswith("Dropping Edit "))
                lines = lines[:head] + block + lines[head:]
            else:                                   # n10c: after the commit
                lines = lines + block
            # keyed LOWERCASE, exactly as `_obs_values` keys n9's -- `tbl_stats`
            # lowercases the label it reads back, and "H-bond donor change" is printed
            # with a capital on 9.9% of rows.
            info["own"] = {lab.get(c, c.replace("_", " ")).lower(): [] for c in cols}
        elif arm in ("n9", "n9r", "n11"):
            # ROW ORDER IS SHUFFLED, per round, in both arms. Printed in criterion order
            # the first row would always be cell_m's pick -- molecular weight on 54% of
            # rounds -- and n9 would carry a positional regularity its control does not,
            # which is a difference between the arms that has nothing to do with which
            # columns were chosen.
            cols = _shuf_cols(cols, rnd)
            info["cols"] = cols
            head = next(i for i, x in enumerate(lines)
                        if x.startswith("Dropping Edit "))
            tbl = _table_lines(occ, meta, cols, n_cand)
            if arm == "n11":
                # the announcement rides ON the heading line, so no new line type enters
                # the frame and `_OBS_ROW` -- which needs a two-space indent -- cannot
                # match it. Same order as the rows below it.
                lab = meta["labels"]
                tbl[0] = (TABLE_HEAD + " "
                          + N11_JOIN.join(lab.get(c, c.replace("_", " ")) for c in cols))
            lines = lines[:head] + tbl + lines[head:]
            info["own"] = _obs_values(occ, meta, cols, n_cand)
        info["span"] = "\n".join(lines)
        info["frame"] = frame
        return None, info

    if arm in N17_ARMS:
        # n8's frame with a joint in front of each elimination, plus v28's line.
        frame = _frame17(rnd, pick, order, n_cand)
        if frame is None:
            return None
        unp = {p for p, _ in bs.direction(rnd)}
        fgl = bs.fg_line_all(fg)
        if fgl is None:
            return None              # a molecule with no groups has no line to add
        molfg = {e["name"]: int(e["count"]) for e in ((fg or {}).get("fg") or [])}
        fl = frame.split("\n")
        # the band row gains a last cell: what this edit puts on and takes off. The
        # groups do not predict the commit -- added to the band features they move a GBM
        # by -0.005 -- but the joints argue from chemistry and this is where the
        # chemistry has to be visible and checkable.
        for j, ln in enumerate(fl):
            if ln.startswith("  Edit "):
                k = int(ln.split("|")[0].split()[1]) - 1
                fl[j] = ln + " | " + fg_row(rnd, k, molfg)
        fl.insert(1, fgl)            # after NEEDS, before the band header
        frame = "\n".join(fl)
        lives = live_sets(frame, n_cand)[:2]
        tgts = [order[0], order[1]]
        notes = draft_band17_all(rnd, lives, tgts, unp, molfg, kinds=N17_KINDS,
                                 facts_floor=True)
        # The rule-based notes stay, but as the FLOOR: `build_seq_prompts` asks the writer
        # for a sentence off the whole turn, and a joint that never passes falls back to
        # the note that was already true.
        prompts, seq_ctx = build_seq_prompts(rnd, frame, order, unp, molfg)
        # what the band condemns for each joint's edit -- the only properties that joint
        # is allowed to name, and the arbiter `band_only_bad` reads
        # WHAT THE AGAINST FACTS NAME, not the band's `at risk | misses`. A spread fact
        # -- "carries the widest spread here on logD" -- is a reason to drop and names a
        # property the band may well score as holding, and taking the band's set as the
        # vocabulary rejected 36.9% of otherwise clean joints for citing a fact the sheet
        # had just handed them.
        sheet = dict(fact_sheet(rnd, unp, molfg))
        bad = []
        for t in tgts:
            ag = [x for x in (sheet.get(t) or []) if _polarity(x) == "against"]
            bad.append(sorted({p for p in _PROPS if any(p in x for x in ag)}))
        info.update(frame=frame, own={}, lives=lives, draft=notes, band_source=True,
                    seq_ctx=seq_ctx, seq_bad=bad, seq_unp=sorted(unp),
                    tgts=[t + 1 for t in tgts])
        return prompts, info

    if arm in ("n15",) + N16_ARMS + N18_ARMS:
        # Same shape as the ABS arms -- draft here, the model only rewords -- but the
        # draft is cut from the BAND, so `own` is not needed and is not passed.
        frame = _frame(rnd, pick, order, n_cand)
        if frame is None:
            return None
        molfg = {}
        if arm in N16_ARMS + N18_ARMS:
            fgl = bs.fg_line_all(fg)
            if fgl is None:
                return None          # a molecule with no groups has no line to add
            fl = frame.split("\n")
            # AFTER the NEEDS line and BEFORE the band header, which is where v28 put it.
            fl.insert(1, fgl)
            frame = "\n".join(fl)
            # n17's closer names the groups the two survivors PUT ON, and has to know
            # which of them the molecule already carries -- the same record the line
            # above is rendered from, so the two can never disagree.
            molfg = {e["name"]: int(e["count"]) for e in ((fg or {}).get("fg") or [])}
        lives = live_sets(frame, n_cand)
        tgts = [order[0], order[1], pick]
        unp = {p for p, _ in bs.direction(rnd)}
        if arm in N18_ARMS:
            notes, polish = (draft_band17_all(rnd, lives, tgts, unp, molfg),
                             POLISH_BAND18)
        else:
            notes, polish = draft_band_all(rnd, lives, tgts, unp), POLISH_BAND
        # the edit each joint is ABOUT, so the gate can insist the reword keeps it
        info.update(frame=frame, own={}, lives=lives, draft=notes, band_source=True,
                    tgts=[t + 1 for t in tgts])
        return polish.format(notes="\n".join(f"{i + 1}. {x}"
                                              for i, x in enumerate(notes))), info

    if arm in ABS_ARMS:
        # The passages are drafted here and the model only rewords them, so the prompt
        # carries the three notes and nothing else. `own` and `lives` ride along in the
        # record because the gates need them and the prompt no longer has them.
        frame = _frame(rnd, pick, order, n_cand)
        if frame is None:
            return None
        own = _obs_values(occ, meta, cols, n_cand)
        lives = live_sets(frame, n_cand)
        tgts = [order[0], order[1], pick]
        notes = draft_all(own, lives, tgts, f'{rnd["group_id"]}:{rnd["depth"]}')
        info.update(frame=frame, own=own, lives=lives, draft=notes)
        return POLISH.format(notes="\n".join(f"{i + 1}. {x}"
                                             for i, x in enumerate(notes))), info

    if arm in GAP_ARMS:
        frame = _frame(rnd, pick, order, n_cand)
        if frame is None:
            return None
        # the two edits still standing at the last joint, in edit order
        alive = sorted([pick] + [i for i in range(n_cand) if i not in order
                                 and i != pick])[:2]
        if len(alive) < 2:
            return None
        closer = (CLOSER_ABS if arm in ABS_ARMS else CLOSER_GAP).format(
            obs_head=OBS_HEAD, commit_name=f"Edit {pick + 1}",
            drop1_name=f"Edit {order[0] + 1}", drop2_name=f"Edit {order[1] + 1}",
            pair_text=f"Edit {alive[0] + 1} against Edit {alive[1] + 1}")
        info["frame"] = frame
        # NEEDS and BAND are in the frame; printing them again above it would only ask
        # the writer to restate what it can already see.
        return SKELETON_GAP.format(
            user_prompt=rnd["user_prompt"], mol_smiles=rnd["smiles"],
            evidence=_obs_block(occ, meta, cols, n_cand, rest),
            frame=frame, closer=closer, **call), info

    ev = _needs_block(rnd) + "\n" + _band_block(rnd) + "\n" \
        + _obs_block(occ, meta, cols, n_cand, rest)
    kw = dict(pick_no=pick + 1, commit_pair=bs.name(rnd, pick),
              commit_name=f"Edit {pick + 1}", obs_head=OBS_HEAD)
    return SKELETON.format(
        user_prompt=rnd["user_prompt"], mol_smiles=rnd["smiles"], evidence=ev,
        closer=CLOSER_FREE.format(**kw), **call), info


# --------------------------------------------------------------------------- #
def one_split(work, split, arm, rule_dir="occlusion_rules2", evd="evidence_ac",
              limit_shards=0, shard=0, nshards=1):
    out_dir = f"{work}/nat_prompt/{arm}/{split}"
    os.makedirs(out_dir, exist_ok=True)
    # A render-only arm needs no model, so it writes its spans here and is done. The
    # prompt file is still written, because every downstream diagnostic keys off it.
    span_dir = None
    if arm in RENDER_ARMS:
        span_dir = f"{work}/spans/model_{arm}/{split}"
        os.makedirs(span_dir, exist_ok=True)
        # THE PAGE'S SETTINGS, BESIDE THE SPANS. Which membership tuples this arm is
        # in IS the page: whether the band cells carry intervals, whether the table
        # line is a verdict or a description, whether the values read as clauses, which
        # pickle the observations came from. None of that was recoverable from the
        # artefact before -- it lived in this file and had to be read at the right
        # commit, which is how `n56 keeps c_* and n57 does not` went unnoticed long
        # enough to invalidate a comparison.
        import datetime as _dt
        import subprocess as _sp
        try:
            _rev = _sp.run(["git", "-C", os.path.dirname(os.path.abspath(__file__)),
                            "rev-parse", "HEAD"], capture_output=True, text=True,
                           timeout=10).stdout.strip() or None
        except Exception:
            _rev = None
        _page = {
            "band_cells": ("interval, `MW 419.107 holds`" if arm in _INTERVAL_ARMS
                           else "grouped by slot, `holds MW, logP`"),
            "table_line": ("description + concessive closer" if arm in _DESC_ARMS
                           else "verdict, `The table favours Edit N since ...`"
                           if arm in _TABLE2_ARMS
                           else "override, names the commit" if arm in _TABLE_ARMS
                           else None),
            "observation_values": ("clauses, `Edit 1 adds 3`" if arm in _SENT_ARMS
                                   else "label run, `Edit 1 3`"),
            "connectives": ("computed from the pearson between the two readings"
                            if arm in _FIXED_ARMS else "fixed, `The other side of it is`"),
            "observations_from": _OBS_SRC.get(arm, arm),
            "fg_filter": arm in _FGF, "no_cells": arm in _NOCELLS,
            "observations_only": arm in _OBSONLY,
        }
        with open(f"{work}/spans/model_{arm}/prompt_hash.json", "w") as fh:
            json.dump({"arm": f"model_{arm}", "rendered": True,
                       "order": _ORDER_OF.get(arm),
                       "page": _page, "git_commit": _rev,
                       "written_at": _dt.datetime.now().isoformat(timespec="seconds"),
                       "note": "v34 frame, no written passages -- no model involved"},
                      fh, indent=1)
    occ_all, meta = rb._rule_occlusion(work, split, rule_dir)
    # n1, n2 and n4 choose their columns with CRITERIA_V36, four of whose six criteria
    # are re-ranks of the occlusion drops -- and n2 needs them too, because its random
    # draw takes its COUNT from what the criteria selected. n3 and n6 print the whole
    # live pool and `_live_cols` reads `col_v` alone, so `base_dump.py` is enough for
    # them and `occlude_rules.py` -- ~20x the CPU -- is not.
    need_gate = arm in ("n1", "n2", "n4", "n1a", "n2a")   # n7/n7r select at random
    if need_gate and occ_all:
        probe = next(iter(occ_all.values()))
        if "cell_m" not in probe:
            raise SystemExit(
                f"{rule_dir} has no occlusion drops (cell_m/col_m): arm {arm} selects "
                "its columns with CRITERIA_V36 and needs them. Either run "
                "occlude_rules.py over this work dir, or build n3/n6, which read col_v "
                "alone.")
    lv_all = rb._gates_full(work, split, "gates_full") if need_gate else {}
    sp_all = rb._gates_spread(work, split, "gates_spread") if need_gate else {}
    fg_all = (rb._fg_gates(work, split, "fg_gates")
              if arm in N16_ARMS + N17_ARMS + N18_ARMS + N19_ARMS + N20_ARMS
                        + N21_ARMS + N22_ARMS + N23_ARMS + N24_ARMS
                        + N48_ARMS + N49_ARMS + N50_ARMS + N51_ARMS + N52_ARMS
                        + N53_ARMS + N54_ARMS + N55_ARMS + _N25F + _N40F else {})
    n = miss = 0
    lens = []
    paths = sorted(glob.glob(f"{work}/rounds/{split}/*.jsonl"))
    if limit_shards:
        paths = paths[:limit_shards]
    # A FULL-CORPUS SPLIT IS ONE PROCESS OTHERWISE. Rendering is per rounds-shard and
    # every output file is named after its shard, so a stride over `paths` splits the
    # work with nothing to merge afterwards. The per-split tables above (`occ_all`,
    # `fg_all`) are still loaded whole by each worker -- that is the cost of the fan,
    # and it is why this is a stride rather than 96 workers: 3M scaffold rounds is
    # ~40 GB of `base_cols` per process.
    if nshards > 1:
        paths = paths[shard::nshards]
    for path in paths:
        base = os.path.basename(path)
        q_of = {}
        with open(f"{work}/{evd}/{split}/{base}") as fh:
            for line in fh:
                e = json.loads(line)
                cs = sorted(e["candidates"], key=lambda c: c["index"])
                q_of[(e["group_id"], e["depth"])] = [float(c.get("q") or 0.0)
                                                     for c in cs]
        sp_out = open(f"{span_dir}/{base}", "w") if span_dir else None
        with open(path) as fh, open(f"{out_dir}/{base}", "w") as out:
            for line in fh:
                r = json.loads(line)
                k = (r["group_id"], r["depth"])
                occ, q = occ_all.get(k), q_of.get(k)
                if occ is None or (q is None and arm != "n6"):
                    miss += 1
                    continue
                built = build_prompt(arm, r, occ, meta, lv_all.get(k), sp_all.get(k),
                                     q, fg_all.get(k))
                if built is None:
                    miss += 1
                    continue
                prompt, info = built
                # a SEQUENTIAL arm returns a list of prompts, and len() of
                # that is 2, which made the size report read "0.0 KB".
                _p = ("".join(prompt) if isinstance(prompt, list)
                      else prompt or info.get("span") or "")
                lens.append(len(_p))
                rec = {"group_id": k[0], "depth": k[1], **info}
                if prompt is not None:
                    rec["prompt"] = prompt
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if sp_out is not None:
                    sp_out.write(json.dumps({"group_id": k[0], "depth": k[1],
                                             "text": info["span"]},
                                            ensure_ascii=False) + "\n")
                n += 1
        if sp_out is not None:
            sp_out.close()
    lens.sort()
    print(f"# {arm} {split}: {n:,} prompts, {miss} skipped, "
          f"median {lens[len(lens)//2]/1024:.1f} KB, max {lens[-1]/1024:.1f} KB"
          if lens else f"# {arm} {split}: nothing built", flush=True)
    return {"arm": arm, "split": split, "n": n, "miss": miss}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--work", default=None)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--rule-dir", default="occlusion_rules2")
    ap.add_argument("--evidence", default="evidence_ac")
    ap.add_argument("--limit-shards", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0,
                    help="render only rounds-shards [shard::nshards]")
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--show", type=int, default=0, help="print N built prompts and stop")
    a = ap.parse_args(argv)
    work = a.work or rp.work_dir(rp.DEFAULT)
    for split in a.splits:
        one_split(work, split, a.arm, a.rule_dir, a.evidence, a.limit_shards,
                  a.shard, a.nshards)
        if a.show:
            p = sorted(glob.glob(f"{work}/nat_prompt/{a.arm}/{split}/*.jsonl"))[0]
            for i, line in enumerate(open(p)):
                if i >= a.show:
                    break
                print("=" * 78)
                d = json.loads(line)
                # a RENDER arm has no generation prompt -- the span IS the artefact
                print(d.get("prompt") or d["span"])
            return


if __name__ == "__main__":
    main()
