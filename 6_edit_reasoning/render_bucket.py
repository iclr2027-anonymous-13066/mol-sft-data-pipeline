# -*- coding: utf-8 -*-
"""Write the v12 (bucket) spans without a language model.

    PYTHONPATH=. python 6_edit_reasoning/render_bucket.py --dst model_v12

Output form matches `generate.py` byte for byte -- {group_id, depth, text} per line, one
shard per rounds shard -- so `assemble.py --arms model_v12` needs no change.

Unlike render_struct (v11b) this reads NO evidence dump: the v12 span has no mode
structure and never consults `q`. It is a pure function of the round.
"""
from __future__ import annotations

import argparse
import glob
import importlib
import json
import os
import statistics
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

import bucket_span as bs                                            # noqa: E402
rp = importlib.import_module("6_edit_reasoning.recipe")          # noqa: E402


def _occlusion(work: str, split: str, sub: str) -> dict:
    """{(group_id, depth): record} from occlude_globals.py, for the v15 lines."""
    p = f"{work}/{sub}/{split}.jsonl"
    if not os.path.exists(p):
        raise SystemExit(f"no occlusion file at {p} -- run occlude_globals.py first")
    out = {}
    with open(p) as fh:
        for line in fh:
            r = json.loads(line)
            out[(r["group_id"], r["depth"])] = r
    return out


def _rule_occlusion(work: str, split: str, sub: str):
    """({(group_id, depth): record}, meta) from occlude_rules.py, for the v17 block.

    `meta` is the run's rule_names.json: the index -> name table the per-round `col_idx`
    refers to, the named pools, and one label per column. Loading it here keeps
    bucket_span free of any dependency on the rule-selection package.
    """
    # Sharded runs write `<split>.partNN.jsonl`, one per worker, and a full-corpus pass
    # has to be sharded -- 3.6M rounds is 96 workers, not one. Both layouts are read
    # here, the same way `_gates_full` already reads both.
    paths = ([f"{work}/{sub}/{split}.jsonl"]
             if os.path.exists(f"{work}/{sub}/{split}.jsonl")
             else sorted(glob.glob(f"{work}/{sub}/{split}.part*.jsonl")))
    m = f"{work}/{sub}/rule_names.json"
    if not paths or not os.path.exists(m):
        raise SystemExit(f"no rule occlusion at {work}/{sub}/{split} -- "
                         "run occlude_rules.py (or base_dump.py) first")
    out = {}
    for p in paths:
        with open(p) as fh:
            for line in fh:
                r = json.loads(line)
                out[(r["group_id"], r["depth"])] = r
    with open(m) as fh:
        return out, json.load(fh)


def _gates(work: str, split: str, sub: str) -> dict:
    """{(group_id, depth): {feature: gate}} from an evidence dump, for v17b.

    The dump keeps each candidate's top-gated features plus `table4`, not the whole
    167-wide gate vector -- 14.6 columns per round. That is enough: A's rule gate is ~90%
    a property of the round rather than the candidate (0.134 spread across options
    against 1.296 across columns), so the per-candidate maximum is the round's gate, and
    at least one gated column varies across the options on 100.0% of rounds.
    """
    out = {}
    for path in sorted(glob.glob(f"{work}/{sub}/{split}/*.jsonl")):
        with open(path) as fh:
            for line in fh:
                r = json.loads(line)
                g = {}
                for c in (r.get("candidates") or []):
                    for x in (c.get("features") or []):
                        g[x["feature"]] = max(g.get(x["feature"], -1e30), x["gate"])
                for e in (r.get("table4") or []):
                    if e.get("gate") is not None:
                        g[e["feature"]] = max(g.get(e["feature"], -1e30), e["gate"])
                out[(r["group_id"], r["depth"])] = g
    if not out:
        raise SystemExit(f"no evidence gates under {work}/{sub}/{split}")
    return out


def _gates_full(work: str, split: str, sub: str) -> dict:
    """{(group_id, depth): {feature: gate}} from gate_dump.py -- ALL 167 columns.

    The same shape `_gates` returns, so `choose_gate_axis` and `choose_contrast_gate`
    take either. The difference is coverage: `_gates` reads `evidence_v10`, which keeps
    each candidate's top-6 plus `table4` -- 14.6 columns a round -- and that is enough
    for a block drawing on all of P0 (A gates a varying column on 99.6% of rounds) but
    not once the pool is split. Over the three survivors A gates a varying PROPERTY
    column on 97.7% of rounds and a varying STRUCTURAL one on only 53.8%, and the
    shortfall is the truncation, not A. With the full vector both are ~100% and no
    block has to fall back to a criterion it was not supposed to use.

    The numbers are on the same scale (`gate_vectors` renormalises within the mask, so
    1.0 is "attended as usual") but they are NOT bit-identical to the cached ones: a
    fresh forward differs on 74.7% of rounds because only 68.6% of this corpus's state
    molecules and 35.7% of its candidate products are in the prebuilt ctx cache, and the
    rest are encoded inline by a molbert path that has changed since evidence_v10 was
    written. That difference does not move a decision -- paired on the same rounds, the
    fresh q and the stored q pick an on-path option equally often (+0.0000, t=0.00 over
    1,867 differing rounds) -- but an arm should read gates from ONE of the two, never
    a mixture, or its two blocks are scored on subtly different numbers.
    """
    paths = ([f"{work}/{sub}/{split}.jsonl"]
             if os.path.exists(f"{work}/{sub}/{split}.jsonl")
             else sorted(glob.glob(f"{work}/{sub}/{split}.part*.jsonl")))
    names = json.load(open(f"{work}/{sub}/names.json"))["rule_names"]
    out = {}
    for p in paths:
        with open(p) as fh:
            for line in fh:
                r = json.loads(line)
                out[(r["group_id"], r["depth"])] = dict(zip(names, r["rule_gate"]))
    if not out:
        raise SystemExit(f"no full gates under {work}/{sub}/{split} -- "
                         "run gate_dump.py first")
    return out


def _gates_spread(work: str, split: str, sub: str) -> dict:
    """{(group_id, depth): {feature: gate spread}} from gate_dump.py.

    `rule_gate_spread` is max - min of the gate over the LIVE candidates: the part of A's
    attention that depends on which edit it is scoring. `rule_gate` in the same file is
    the max, which is the round-level level every earlier arm used.
    """
    paths = ([f"{work}/{sub}/{split}.jsonl"]
             if os.path.exists(f"{work}/{sub}/{split}.jsonl")
             else sorted(glob.glob(f"{work}/{sub}/{split}.part*.jsonl")))
    names = json.load(open(f"{work}/{sub}/names.json"))["rule_names"]
    out = {}
    for p in paths:
        with open(p) as fh:
            for line in fh:
                r = json.loads(line)
                out[(r["group_id"], r["depth"])] = dict(
                    zip(names, r["rule_gate_spread"]))
    if not out:
        raise SystemExit(f"no gate spread under {work}/{sub}/{split} -- "
                         "re-run gate_dump.py (it now writes rule_gate_spread)")
    return out


def _fg_gates(work: str, split: str, sub: str) -> dict:
    """{(group_id, depth): record} from fg_gates.py, for the v27 NEEDS line.

    Reads the merged `<split>.jsonl` and falls back to the per-worker `part*` files, so
    a sharded dump can be rendered without a merge step.
    """
    paths = ([f"{work}/{sub}/{split}.jsonl"]
             if os.path.exists(f"{work}/{sub}/{split}.jsonl")
             else sorted(glob.glob(f"{work}/{sub}/{split}.part*.jsonl")))
    out = {}
    for p in paths:
        with open(p) as fh:
            for line in fh:
                r = json.loads(line)
                out[(r["group_id"], r["depth"])] = r
    if not out:
        raise SystemExit(f"no fg gates under {work}/{sub}/{split} -- "
                         "run fg_gates.py first")
    return out


def one_split(work: str, split: str, dst: str, evd: str = "",
              style: str = "v12", occ_dir: str = "occlusion",
              occ_key: str = "top_margin", n_state: int = 1,
              rule_dir: str = "occlusion_rules", criterion: str = "margin",
              pool: str = "primitive", require_vary: bool = True,
              fg_dir: str = "fg_gates", n_fg: int = 0,
              gates_dir: str = "gates_full") -> dict:
    """Render one split. With `evd`, q is read and the v13 two-name closer is used.

    q is NEVER printed -- it only decides whether the closer names one option or two,
    exactly as in v11b. The student cannot see it at test time.
    """
    out_dir = f"{work}/spans/{dst}/{split}"
    os.makedirs(out_dir, exist_ok=True)
    n = miss = 0
    words = []
    modes = {"mode1": 0, "mode2": 0, "commit_first": 0}
    occ_all = _occlusion(work, split, occ_dir) if style in ("v15", "v17") else {}
    rule_all, rule_meta = ({}, None)
    if style in ("v17", "v17b", "v18", "v19", "v21",
                 "v22", "v22r", "v23", "v23r", "v24", "v24r", "v25", "v25r",
                 "v28c", "v28m", "v29", "v29s", "v30", "v30r",
                 "v31", "v31r", "v31u", "v32", "v32r", "v33", "v33r",
                 "v35", "v35r", "v36", "v36r"):
        rule_all, rule_meta = _rule_occlusion(work, split, rule_dir)
    # v29 reads the FULL gate vector; every earlier arm reads evidence_v10's top-6.
    # Never both in one span -- the two are the same scale but not the same numbers.
    spread_all = (_gates_spread(work, split, "gates_spread")
                  if style in ("v35", "v35r", "v36", "v36r") else {})
    gates_all = (_gates_full(work, split, gates_dir)
                 if style in ("v29", "v29s", "v35", "v35r", "v36", "v36r")
                 else _gates(work, split, evd) if style in ("v17b", "v19", "v21",
                                                            "v25", "v25r") else {})
    # v28 needs only the candidates' own deltas, which the occlusion dump carries
    fg_all = (_fg_gates(work, split, fg_dir)
              if style.startswith("v27") or style in ("v28", "v28c") else {})
    no_contrast = 0
    for path in sorted(glob.glob(f"{work}/rounds/{split}/*.jsonl")):
        base = os.path.basename(path)
        ev = {}
        if evd:
            ep = f"{work}/{evd}/{split}/{base}"
            if not os.path.exists(ep):
                raise SystemExit(f"no evidence shard for {split}/{base} in {evd}")
            with open(ep) as fh:
                for line in fh:
                    e = json.loads(line)
                    # sorted by index rather than trusted positionally -- verified to be
                    # already in order on 4,577 records, but the alignment of q to
                    # candidates is silent if it is ever wrong
                    cs = sorted(e["candidates"], key=lambda c: c["index"])
                    ev[(e["group_id"], e["depth"])] = (
                        [float(c.get("q") or 0.0) for c in cs], e)
        with open(path) as fh, open(f"{out_dir}/{base}", "w") as out:
            for line in fh:
                r = json.loads(line)
                pick = int(r["picked"])
                q = None; ent = None
                if evd:
                    rec = ev.get((r["group_id"], r["depth"]))
                    if rec is None:
                        miss += 1
                        continue
                    q, ent = rec
                # `picked` is the beam's committed candidate -- the label the tool call
                # after this span will carry. The span must not contradict it.
                #   v12  rows + plain closer
                #   v13  rows + the two-name closer when q disagrees
                #   v14  rows + A's top-gated state line and one line per rule
                #   v14c same, but each rule's line is restricted to features whose
                #        value VARIES across the candidates and that the student can
                #        actually derive
                if style == "v13":
                    text = bs.render(r, pick, q)
                elif style in ("v14", "v14c"):
                    text = bs.render(r, pick, None, ent, style == "v14c")
                elif style == "v14g":
                    # global (state) features only, top 2, no per-rule lines
                    text = bs.render(r, pick, None, ent, False, 2, False)
                elif style == "v15":
                    oc = occ_all.get((r["group_id"], r["depth"]))
                    if oc is None:
                        miss += 1
                        continue
                    # ONE line, not two. The second line is both harder to name and
                    # thinner: measured on v14 at epoch 3, the student reproduced slot
                    # 1's feature identity 0.923 of the time and slot 2's only 0.748,
                    # and a wrongly-named line came with onpath -0.046 / exact -0.037.
                    # The occlusion criteria are more diffuse than A's gate (27.1 and
                    # 17.6 effective choices in slot 1 against A's 8.2), so slot 2 here
                    # would be harder still.
                    text = bs.render(r, pick, None, None, False, n_state, False, oc,
                                     occ_key)
                elif style in ("v20", "v20r", "v21"):
                    # NEEDS -> BAND [-> CONTRAST] -> DROP -> COMMIT. v20 and v21 differ
                    # by the CONTRAST block and nothing else, so v21 - v20 isolates it
                    # exactly the way v20 - (this renderer with no DROP) isolates DROP.
                    # v20r is v20 with the teacher signal taken out of the chooser
                    # and nothing else changed. `--evidence` is still required so the
                    # two corpora keep the SAME rounds: v20 drops a round whose evidence
                    # shard is missing, and a control on a different round set is not a
                    # control.
                    d = (bs.choose_drop_random(
                             len(r["candidates"]), pick,
                             f'{r["group_id"]}:{r["depth"]}')
                         if style == "v20r" else bs.choose_drop(q, pick))
                    if d is None:
                        miss += 1
                        continue
                    base = bs.render(r, pick).split("\n")
                    lines = []
                    if style == "v21":
                        orl = rule_all.get((r["group_id"], r["depth"]))
                        g = gates_all.get((r["group_id"], r["depth"]))
                        if orl is None or g is None:
                            miss += 1
                            continue
                        contrast = bs.choose_contrast_gate(
                            orl, g, rule_meta, pool, require_vary, r.get("targets"))
                        lines = bs.contrast_lines(contrast, len(r["candidates"]))
                        if not lines:
                            no_contrast += 1
                    text = "\n".join(base[:-1] + lines
                                     + [bs.drop_line(d, len(r["candidates"]))]
                                     + [base[-1]])
                elif style.startswith(("v27", "v28")):
                    # v20 with ONE extra line before the band, and nothing else
                    # changed: BAND, the q-argmin DROP and the closer stay byte-identical
                    # to v20, so each of these arms minus v20 is that line alone.
                    #
                    #   v27nK   the K highest-gated groups the molecule HAS
                    #   v27n3r  three of the same, gate removed (the control)
                    #   v28     EVERY group the molecule has -- no selection at all
                    #   v28m    what the four edits collectively change  (no teacher)
                    #   v28c    the groups the COMMIT changes -- built, then rejected:
                    #           it names what the answer does, in the first line
                    d = bs.choose_drop(q, pick)
                    if d is None:
                        miss += 1
                        continue
                    orl = rule_all.get((r["group_id"], r["depth"]))
                    fgr = fg_all.get((r["group_id"], r["depth"]))
                    need_occ = style in ("v28m", "v28c")
                    need_fg = style.startswith("v27") or style in ("v28", "v28c")
                    if (need_occ and orl is None) or (need_fg and fgr is None):
                        miss += 1
                        continue
                    if style == "v28":
                        fgl = bs.fg_line_all(fgr)
                    elif style == "v28m":
                        fgl = bs.fg_line_moved(orl, rule_meta, n_fg)
                    elif style == "v28c":
                        fgl = bs.fg_line_commit(fgr, orl, rule_meta, pick, n_fg)
                    elif style.endswith("r"):
                        fgl = bs.fg_line_random(
                            fgr, n_fg, f'{r["group_id"]}:{r["depth"]}')
                    else:
                        fgl = bs.fg_line(fgr, n_fg)
                    if fgl is None:
                        no_contrast += 1
                    base = bs.render(r, pick).split("\n")
                    # inserted AFTER the out-of-range header and before the band head,
                    # so the state block reads as one unit
                    text = "\n".join(base[:1] + ([fgl] if fgl else []) + base[1:-1]
                                     + [bs.drop_line(d, len(r["candidates"]))]
                                     + [base[-1]])
                elif style in ("v34", "v34r"):
                    # TWO eliminations, leaving two. The one dimension in this family
                    # with a measured effect is the ACTION SPACE, not the observation:
                    # v20 (drop one) beat v12 (drop none) by +0.028, while v20 - v20r
                    # said the drop's CONTENT is worth +0.0076 (t=1.82). v26 walked it
                    # all the way down to one survivor and came apart at epoch 3 (0.492
                    # against v20's 0.536) -- its last line announces the answer, so a
                    # student whose own eliminations differ commits to the wrong one.
                    # Two drops is the untested midpoint: it halves the choice without
                    # naming it. No observation block at all, so v34 - v20 is the second
                    # elimination alone.
                    #
                    # v34r takes q out of WHICH two and keeps the count, so v34 - v34r
                    # prices the teacher signal in the action space exactly as
                    # v20 - v20r did for one drop. q's rival is off-path 0.861 of the
                    # time against a random rival's 0.793, and over two drops that gap
                    # is what the contrast is made of.
                    d = bs.choose_drop(q, pick)
                    if d is None:
                        miss += 1
                        continue
                    nc = len(r["candidates"])
                    order = (bs.drop_order_random(
                                 nc, pick, f'{r["group_id"]}:{r["depth"]}')
                             if style == "v34r" else bs.drop_order(q, pick))[:2]
                    base = bs.render(r, pick).split("\n")
                    text = "\n".join(base[:-1] + bs.drop_lines(order, nc) + [base[-1]])
                elif style in ("v26", "v26r"):
                    # NEEDS -> BAND -> DROP1 -> DROP2 -> DROP3 -> COMMIT. No observation
                    # block at all: this varies the NUMBER of eliminations, which is the
                    # one dimension v20/v20r showed carries the effect.
                    d = bs.choose_drop(q, pick)
                    if d is None:
                        miss += 1
                        continue
                    nc = len(r["candidates"])
                    order = (bs.drop_order_random(
                                 nc, pick, f'{r["group_id"]}:{r["depth"]}')
                             if style == "v26r" else bs.drop_order(q, pick))
                    base = bs.render(r, pick).split("\n")
                    text = "\n".join(base[:-1] + bs.drop_lines(order, nc) + [base[-1]])
                elif style in ("v35", "v35r", "v36", "v36r"):
                    # ONE panel over the three survivors, its columns chosen per round by
                    # the teacher criteria. v35 uses two -- A's gate LEVEL and the gate's
                    # SPREAD across the live candidates; v36 lets all six name one each
                    # and prints the union. See pool_104 / CRITERIA_V35 in bucket_span.
                    #
                    # The r-arms draw the SAME NUMBER of columns uniformly from the same
                    # live set, so the two corpora carry the panel on the same rounds
                    # with the same number of cells and only the criterion differs.
                    orl = rule_all.get((r["group_id"], r["depth"]))
                    lv = (gates_all or {}).get((r["group_id"], r["depth"]))
                    sp = (spread_all or {}).get((r["group_id"], r["depth"]))
                    if orl is None or lv is None or sp is None:
                        miss += 1
                        continue
                    d = bs.choose_drop(q, pick)
                    if d is None:
                        miss += 1
                        continue
                    nc = len(r["candidates"])
                    surv = [i for i in range(nc) if i != d]
                    allow = bs.pool_104(rule_meta)
                    crit = (bs.CRITERIA_V35 if style.startswith("v35")
                            else bs.CRITERIA_V36)
                    cols = bs.choose_cols_multi(orl, rule_meta, surv, allow, crit,
                                                lv, sp, pick)
                    if style.endswith("r") and cols:
                        cols = bs.choose_cols_random(
                            orl, rule_meta, surv, allow, len(cols),
                            f'panel-random:{r["group_id"]}:{r["depth"]}')
                    lines = bs.panel_lines(orl, rule_meta, surv, cols) if cols else []
                    if not lines:
                        no_contrast += 1
                    base = bs.render(r, pick).split("\n")
                    text = "\n".join(base[:-1] + [bs.drop_line(d, nc)] + lines
                                      + [base[-1]])
                elif style in ("v32", "v32r", "v33", "v33r"):
                    # v20 with ONE weigh block on a column the student CANNOT derive --
                    # the only kind with headroom left over the visible ceiling. The
                    # column is constant across the corpus, so nothing has to be
                    # reproduced but the three values; v32 uses the one the teacher's
                    # q-drop ranks first in that pool and v32r one drawn at random from
                    # it. See CAND_TEACHER in bucket_span for the measurement.
                    orl = rule_all.get((r["group_id"], r["depth"]))
                    if orl is None:
                        miss += 1
                        continue
                    d = bs.choose_drop(q, pick)
                    if d is None:
                        miss += 1
                        continue
                    nc = len(r["candidates"])
                    surv = [i for i in range(nc) if i != d]
                    # BOTH columns are required to vary, in both arms, so the two
                    # corpora carry the block on exactly the same rounds and differ only
                    # in which column it names. Ungated they would not: cand_similarity
                    # varies on 93.9% of rounds and ctx_support on 99.2%, and a 5-point
                    # difference in block PRESENCE is the same size as the effect being
                    # measured.
                    axes = [bs.choose_axis_fixed(orl, rule_meta, surv, c)
                            for c in (bs.CAND_TEACHER, bs.CAND_CONTROL)]
                    ax = None if any(a is None for a in axes) else \
                        axes[1 if style in ("v32r", "v33r") else 0]
                    # v33 keeps the same column and the same rounds and replaces the
                    # three values by their order -- the rendering, and nothing else.
                    if ax and style.startswith("v33"):
                        ax = bs.ordinal_axis(ax)
                    lines = bs.weigh_lines(ax) if ax else []
                    if not lines:
                        no_contrast += 1
                    base = bs.render(r, pick).split("\n")
                    text = "\n".join(base[:-1] + [bs.drop_line(d, nc)] + lines
                                      + [base[-1]])
                elif style in ("v31", "v31r", "v31u"):
                    # v20 with ONE fixed three-column panel after DROP. Nothing is
                    # selected per round, so there is no axis to reproduce; the teacher
                    # signal picks the three columns once, for the whole corpus, and
                    # v31r is the same block with three columns drawn at random from the
                    # usable pool instead. See PANEL_TEACHER in bucket_span for what v30
                    # measured and why the choice moved out of the round.
                    orl = rule_all.get((r["group_id"], r["depth"]))
                    if orl is None:
                        miss += 1
                        continue
                    d = bs.choose_drop(q, pick)
                    if d is None:
                        miss += 1
                        continue
                    nc = len(r["candidates"])
                    surv = [i for i in range(nc) if i != d]
                    cols = (bs.PANEL_CONTROL_P0 if style == "v31r" else
                            bs.PANEL_CONTROL if style == "v31u" else bs.PANEL_TEACHER)
                    lines = bs.panel_lines(orl, rule_meta, surv, cols)
                    if not lines:
                        no_contrast += 1
                    base = bs.render(r, pick).split("\n")
                    text = "\n".join(base[:-1] + [bs.drop_line(d, nc)] + lines
                                      + [base[-1]])
                elif style in ("v30", "v30r"):
                    # v20 with ONE observation block after DROP, over the three
                    # survivors, and nothing else changed -- so v30 - v20 is that block
                    # and v30 - v30r is the CRITERION alone.
                    #
                    # The criterion is `cell_m`: the drop in the commit's q-margin when
                    # that candidate's own cell of the rule table is masked. It replaces
                    # A's attention gate, which every arm from v21 to v29 used and which
                    # scores BELOW random at picking an informative axis (+0.038 onpath
                    # lift against random's +0.050 over P0). See CELL_SHORTLIST_STRU in
                    # bucket_span for the measurement and for why the pool is the
                    # structural half cut to 16 columns.
                    #
                    # No gate is read here, so `--gates-dir` is not needed and no round
                    # is lost to a missing gate vector.
                    orl = rule_all.get((r["group_id"], r["depth"]))
                    if orl is None:
                        miss += 1
                        continue
                    d = bs.choose_drop(q, pick)
                    if d is None:
                        miss += 1
                        continue
                    nc = len(r["candidates"])
                    surv = [i for i in range(nc) if i != d]
                    allow = set(bs.CELL_SHORTLIST_STRU)
                    ax = (bs.choose_axis_uniform(
                              orl, rule_meta, surv, allow,
                              f'weigh-uniform:{r["group_id"]}:{r["depth"]}')
                          if style == "v30r" else
                          bs.choose_axis_cell(orl, rule_meta, pick, surv, allow))
                    lines = bs.weigh_lines(ax) if ax else []
                    if not lines:
                        no_contrast += 1
                    base = bs.render(r, pick).split("\n")
                    text = "\n".join(base[:-1] + [bs.drop_line(d, nc)] + lines
                                      + [base[-1]])
                elif style in ("v29", "v29s"):
                    # TWO observation blocks over the three survivors, one per half of
                    # P0: a PROPERTY axis (41 columns, cut to this instance's targets)
                    # and a STRUCTURAL one (91 columns, no property to cut on). Both are
                    # A's gate over the columns that vary across the survivors, and
                    # either block is DROPPED when A gates nothing that varies rather
                    # than falling back to a criterion its heading does not promise.
                    #
                    # Naming cost, measured over the same pool and filter on 7,977
                    # rounds -- and naming cost is what decided every arm in this family:
                    #   property   A-gate  8.5 | qdrop 19.9 | margin 22.3
                    # structural   A-gate 10.9 | qdrop 29.8 | margin 30.9
                    #
                    # The gates come from gate_dump.py, not evidence_v10: the cached dump
                    # keeps 14.6 columns a round, which fills the property block on 97.7%
                    # of rounds but the structural one on only 53.8%, and a block whose
                    # presence is decided by a gate the student cannot see is a coin flip
                    # it cannot win. With the full vector both are ~100%.
                    #
                    # v29 puts them in the WEIGH slot, after DROP; v29s puts them in the
                    # SIFT slot, before it. DROP is the only block with a measured
                    # effect and it currently sits against COMMIT, so v29 - v29s is
                    # whether that adjacency was carrying anything.
                    orl = rule_all.get((r["group_id"], r["depth"]))
                    g = gates_all.get((r["group_id"], r["depth"]))
                    if orl is None or g is None:
                        miss += 1
                        continue
                    d = bs.choose_drop(q, pick)
                    if d is None:
                        miss += 1
                        continue
                    nc = len(r["candidates"])
                    surv = [i for i in range(nc) if i != d]
                    tg = r.get("targets")
                    P_PROP, P_STRU = bs.sub_pools(rule_meta)
                    lines = []
                    for allowed, targ in ((P_PROP, tg), (P_STRU, None)):
                        ax = bs.choose_axis_gated(orl, g, rule_meta, surv, allowed, targ)
                        lines += bs.weigh_lines(ax) if ax else []
                    if len(lines) < 2:
                        no_contrast += 1
                    base = bs.render(r, pick).split("\n")
                    dl = [bs.drop_line(d, nc)]
                    text = "\n".join(base[:-1] + (dl + lines if style == "v29"
                                                  else lines + dl) + [base[-1]])
                elif style in ("v25", "v25r"):
                    # v23 with the WEIGH criterion swapped for A's gate and nothing else:
                    # same pool, same survivor-vary filter, same DROP, so the text through
                    # DROP stays byte-identical to v20's and only the axis moves.
                    #
                    # Effective vocabulary over the SAME pool and filter, 7,977 rounds:
                    #   qdrop 39.7 | margin 46.7 | A-gate 15.8
                    # Naming is what sank v22 -- wrong axis on 64.6% of held-out rounds
                    # at an effective 34.8, with onpath 0.513 against 0.565 when right.
                    orl = rule_all.get((r["group_id"], r["depth"]))
                    g = gates_all.get((r["group_id"], r["depth"]))
                    if orl is None or g is None:
                        miss += 1
                        continue
                    d = bs.choose_drop(q, pick)
                    if d is None:
                        miss += 1
                        continue
                    nc = len(r["candidates"])
                    surv = [i for i in range(nc) if i != d]
                    tg = r.get("targets")
                    weigh = (bs.choose_axis_random(
                                 orl, rule_meta, surv,
                                 f'weigh-random:{r["group_id"]}:{r["depth"]}', pool, tg)
                             if style == "v25r" else
                             bs.choose_gate_axis(orl, g, rule_meta, surv, pool, tg))
                    wl = bs.weigh_lines(weigh)
                    if not wl:
                        no_contrast += 1
                    base = bs.render(r, pick).split("\n")
                    text = "\n".join(base[:-1] + [bs.drop_line(d, nc)] + wl + [base[-1]])
                elif style in ("v22", "v22r", "v23", "v23r", "v24", "v24r"):
                    # NEEDS -> BAND -> [SIFT] -> DROP -> [WEIGH] -> COMMIT.
                    #
                    # THE PREFIX NESTS, on purpose. All six use v20's DROP line -- the
                    # same q chooser, so the same edit -- which makes v23's text through
                    # DROP byte-identical to v20's and v24's identical to v22's. Only the
                    # observation slots differ, and in the `r` variants only the way the
                    # observation is chosen. The `r` variants keep the q-based DROP:
                    # v20r already ablates the drop identity, and folding both ablations
                    # into one arm would leave neither readable.
                    orl = rule_all.get((r["group_id"], r["depth"]))
                    if orl is None or not orl.get("col_qc"):
                        miss += 1
                        continue
                    d = bs.choose_drop(q, pick)
                    if d is None:
                        miss += 1
                        continue
                    nc = len(r["candidates"])
                    tg = r.get("targets")
                    rk = f'{r["group_id"]}:{r["depth"]}'
                    rnd_pick = style.endswith("r")
                    sift = weigh = None
                    if style.startswith(("v22", "v24")):
                        sift = (bs.choose_axis_random(
                                    orl, rule_meta, list(range(nc)),
                                    f"sift-random:{rk}", pool, tg)
                                if rnd_pick else
                                bs.choose_sift(orl, rule_meta, d, pool, tg))
                    if style.startswith(("v23", "v24")):
                        surv = [i for i in range(nc) if i != d]
                        weigh = (bs.choose_axis_random(
                                     orl, rule_meta, surv,
                                     f"weigh-random:{rk}", pool, tg)
                                 if rnd_pick else
                                 bs.choose_weigh(orl, rule_meta, pick, d, pool, tg))
                    sl, wl = bs.sift_lines(sift), bs.weigh_lines(weigh)
                    if (style.startswith(("v22", "v24")) and not sl) or \
                       (style.startswith(("v23", "v24")) and not wl):
                        no_contrast += 1
                    base = bs.render(r, pick).split("\n")
                    text = "\n".join(base[:-1] + sl + [bs.drop_line(d, nc)]
                                     + wl + [base[-1]])
                elif style in ("v17b", "v18", "v19"):
                    # "rule feature INSTEAD OF the global one": v12's rows, then the rule
                    # content, then the closer. No `What the state turns on:` line, so
                    # these are a same-slot swap against v15 rather than an extra line.
                    orl = rule_all.get((r["group_id"], r["depth"]))
                    if orl is None:
                        miss += 1
                        continue
                    base = bs.render(r, pick).split("\n")
                    if style in ("v17b", "v19"):
                        g = gates_all.get((r["group_id"], r["depth"]))
                        if g is None:
                            miss += 1
                            continue
                        # v19 is v17b with the pool cut to what this instance is judged
                        # on; passing `targets` is the only difference between them.
                        contrast = bs.choose_contrast_gate(
                            orl, g, rule_meta, pool, require_vary,
                            (r.get("targets") if style == "v19" else None))
                        lines = bs.contrast_lines(contrast, len(r["candidates"]))
                    else:
                        lines = bs.rule_q_lines(orl, rule_meta, pool)
                    if not lines:
                        no_contrast += 1
                    text = "\n".join(base[:-1] + lines + [base[-1]])
                elif style == "v17":
                    # v15's block, then ONE rule contrast: the column whose masking costs the
                    # commit the most margin, with every option's value on it. The contrast is
                    # re-ranked from the raw vectors rather than read off the dump's
                    # top_col* lists, which come from an unstable sort -- see
                    # bucket_span.choose_contrast.
                    oc = occ_all.get((r["group_id"], r["depth"]))
                    orl = rule_all.get((r["group_id"], r["depth"]))
                    if oc is None or orl is None:
                        miss += 1
                        continue
                    base = bs.render(r, pick, None, None, False, n_state, False, oc,
                                     occ_key).split("\n")
                    contrast = bs.choose_contrast(orl, rule_meta, criterion, pool, require_vary)
                    lines = bs.contrast_lines(contrast, len(r["candidates"]))
                    if not lines:
                        no_contrast += 1
                    text = "\n".join(base[:-1] + lines + [base[-1]])
                else:
                    text = bs.render(r, pick)
                out.write(json.dumps({"group_id": r["group_id"],
                                      "depth": r["depth"],
                                      "text": text}, ensure_ascii=False) + "\n")
                n += 1
                words.append(len(text.split()))
                if q:
                    qt = max(range(len(q)), key=lambda j: q[j])
                    modes["mode1" if qt == pick else "mode2"] += 1
                    if qt != pick and pick < qt:
                        modes["commit_first"] += 1
    return {"split": split, "spans": n, "missing_evidence": miss,
            "median_words": statistics.median(words) if words else 0,
            "no_contrast": no_contrast, **modes}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default=None)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--dst", default="model_v12", help="spans/<dst>/ to write")
    ap.add_argument("--style", default="v12",
                    choices=["v12", "v13", "v14", "v14c", "v14g", "v15", "v17",
                             "v17b", "v18", "v19", "v20", "v20r", "v21",
                             "v22", "v22r", "v23", "v23r", "v24", "v24r",
                             "v25", "v25r", "v26", "v26r",
                             "v27n1", "v27n2", "v27n3", "v27n3r",
                             "v28", "v28c", "v28m",
                             "v29", "v29s", "v30", "v30r",
                             "v31", "v31r", "v31u",
                             "v32", "v32r", "v33", "v33r", "v34", "v34r",
                             "v35", "v35r", "v36", "v36r"],
                    help="which closer / extra lines. v13 and v14* need --evidence; "
                         "v17 adds one RULE contrast under v15's state line; v17b "
                         "and v18 REPLACE that state line with rule content (v17b: the "
                         "highest-gated varying column, all four values; v18: the "
                         "committed edit's own feature by q(commit) drop). v17b needs "
                         "--evidence for the gates.")
    ap.add_argument("--n-state", type=int, default=1,
                    help="how many global lines to print (v15 style). 1 by default -- "
                         "see the comment at the v15 branch for why the second line is a "
                         "liability rather than a bonus.")
    ap.add_argument("--occ-key", default="top_margin",
                    choices=["top_margin", "top_q"],
                    help="which occlusion ranking to print. top_margin (criterion B) is "
                         "the drop in q(commit) - max q(rival); top_q (criterion A) is "
                         "the drop in q(commit) alone, which names a round-level scalar "
                         "on 52.6%% of rounds. Both are stored by occlude_globals.py, so "
                         "either corpus is a render away -- no second GPU pass.")
    ap.add_argument("--occ-dir", default="occlusion",
                    help="v15: {work}/<dir>/<split>.jsonl from occlude_globals.py")
    ap.add_argument("--rule-dir", default="occlusion_rules",
                    help="v17: {work}/<dir>/ from occlude_rules.py (+ rule_names.json)")
    ap.add_argument("--criterion", default="margin", choices=["margin", "qdrop"],
                    help="v17 contrast ranking. margin = q(commit) - max q(rival), which is "
                         "what v15 beat v16 by +0.011 on-path with; qdrop = q(commit) "
                         "alone, kept so the same A/B can be run over the rule pool.")
    ap.add_argument("--pool", default="primitive",
                    choices=["primitive", "all", "dprop"],
                    help="which rule columns may be named. `primitive` is the evidence "
                         "(per-property deltas, group changes, the site, fragment size); "
                         "`all` re-admits the c_* aggregates, which are arithmetic on the "
                         "same delta and box the rows already show -- q-drop over `all` "
                         "names c_prob on 21%% of rounds, which is reciting a score and is "
                         "the allfeat arm's measured failure; `dprop` is the predicted "
                         "delta of a constrained property only (coverage 99.9%%).")
    ap.add_argument("--no-vary", action="store_true",
                    help="v17: do NOT require the contrast to differ across the options. Off "
                         "by default: without the filter the top column has four "
                         "identical values on 20.3%% of rounds, under a heading that "
                         "promises a contrast.")
    ap.add_argument("--evidence", default="",
                    help="evidence dump to read q from, e.g. evidence_ac. Empty = the "
                         "v12 closer (one name). Set it for v13's two-name closer.")
    ap.add_argument("--gates-dir", default="gates_full",
                    help="v29: gate_dump.py's full 167-column gate vector.")
    ap.add_argument("--fg-dir", default="fg_gates",
                    help="v27: the fg_gates.py dump the NEEDS group line is read from.")
    args = ap.parse_args(argv)

    work = args.work or rp.work_dir(rp.DEFAULT)
    print(f"# work {work}\n# dst  spans/{args.dst}\n"
          f"# q    {args.evidence or 'NOT USED (v12 closer)'}", flush=True)
    for split in args.splits:
        if args.style not in ("v12", "v15", "v17", "v18") and not args.evidence:
            raise SystemExit(f"--style {args.style} needs --evidence")
        # v27nK names K groups; the arm's whole design lives in that digit
        n_fg = (int(args.style[4:].rstrip("r"))
                if args.style.startswith("v27n") else
                3 if args.style in ("v28c", "v28m") else 0)
        r = one_split(work, split, args.dst, args.evidence, args.style, args.occ_dir,
                      args.occ_key, args.n_state, args.rule_dir, args.criterion,
                      args.pool, not args.no_vary, args.fg_dir, n_fg,
                      args.gates_dir)
        line = (f"# {r['split']:5s} {r['spans']:>7,} spans  "
                f"median {r['median_words']:.0f} words")
        if args.style in ("v17", "v17b", "v18", "v19", "v21",
                          "v22", "v22r", "v23", "v23r", "v24", "v24r",
                          "v25", "v25r"):
            line += f"  | rounds with no contrast {r['no_contrast']}"
        if args.style.startswith(("v27", "v28")) or args.style.startswith("v29"):
            line += f"  | rounds with no group line {r['no_contrast']}"
        # v17b reads --evidence for the GATES and never consults q, so its closer is
        # always the one-name form. Printing mode1/mode2 there would report a closer the
        # spans do not contain.
        if args.evidence and args.style in ("v13", "v14", "v14c", "v14g"):
            m1, m2 = r["mode1"], r["mode2"]
            t = max(m1 + m2, 1)
            line += (f"  | mode1 {m1 / t:.3f} mode2 {m2 / t:.3f}"
                     f"  commit named first in mode2 "
                     f"{r['commit_first'] / max(m2, 1):.3f}"
                     f"  missing evidence {r['missing_evidence']}")
        print(line, flush=True)


if __name__ == "__main__":
    main()
