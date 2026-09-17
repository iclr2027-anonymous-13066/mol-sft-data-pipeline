"""Reasoning-span variants to try, and the machinery to price them against `naive`.

`reasoning.py` ships two prompts: `prompt_naive` (the deterministic candidate tables
`4_sftdata_gen/fragment_names.py` builds) and `prompt_model` (A + A_c). Measured, they
are complements — naive is the better chemistry description, model the better decision
description — so the interesting settings are in between, and there are many of them.
This module holds them as one registry so `scripts/reasoning_utility.py` can generate
every arm on ONE instance set and probe them side by side.

Each variant is `(system, prompt)` built from `(row, ev, res, spec, naive)`:

    row     the decision state (state_row / rows_from_record shape)
    ev      reasoning.evidence(...) with pick_override = the COMMITTED rule
    res     the raw Scorer.run output — needed for the FULL gate vectors, since
            `ev` only keeps the top few per candidate
    spec    FeatureSpec, for the feature names the gates index
    naive   (cands, material, spread) from reasoning.naive_material

**The committed rule is the target.** Every prompt argues for `row["committed"]`, not
for the model's argmax — the beam's choice is the label the SFT corpus teaches. On the
real corpus the model's argmax agrees with it only ~39% of the time, so most spans are
arguing for a rule the ranker did not rank first; `ev["pick_is_argmax"]` says which case
a state is in and several variants use it to pick their hedge.

**One arm leaks on purpose.** `product` puts the post-edit molecule's own computed
properties in the prompt. At inference the assistant has not called `edit_fragment` and
cannot know them, so a corpus written this way teaches it to assert what it cannot
derive — and the probe will reward that arm for the wrong reason. It is here to measure
the size of the temptation, and it is reported separately from the honest arms.
"""
from __future__ import annotations

import re

import numpy as np

# --------------------------------------------------------------------------- #
#  gates: the full vectors, renormalised the way evidence() reports them
# --------------------------------------------------------------------------- #
AVERAGE_GATE = 1.05


def gate_vectors(res: dict, spec) -> dict:
    """-> {"state": {name: gate}, "cand": [{name: gate}, ...]}.

    `evidence()` renormalises within the mask before turning alpha into a multiplier,
    so 1.0 means "treated as usual" among the features this instance defines. It keeps
    only the top few per candidate; the variants here need all of them, so the same
    arithmetic is repeated over the full vector.
    """
    ag, gm = res["A_g"], res["g_mask"]
    out = {"state": {}, "cand": []}
    if gm.any():
        a = ag[gm] / max(ag[gm].sum(), 1e-12)
        g = np.zeros_like(ag)
        g[gm] = a * gm.sum()
        out["state"] = {n: float(g[j]) for j, n in enumerate(spec.global_names) if gm[j]}
    ar = res["A_r"]
    for i in range(ar.shape[0]):
        m = res["r_mask"][i]
        d = {}
        if m.any():
            a = ar[i][m] / max(ar[i][m].sum(), 1e-12)
            g = np.zeros_like(ar[i])
            g[m] = a * m.sum()
            d = {n: float(g[j]) for j, n in enumerate(spec.rule_names) if m[j]}
        out["cand"].append(d)
    return out


# feature-name -> the property it speaks about
_PROP_RX = re.compile(r"^(st_val|st_has_lo|st_has_hi|st_lo_z|st_hi_z|r_dmean|r_dstd)__(.+)$")


def property_gates(gv: dict, pick: int) -> dict:
    """-> {property: gate} for the picked candidate, over BOTH blocks.

    A property is talked about in the state block (`st_lo_z__logP` = how far below the
    floor it sits) and in the rule block (`r_dmean__logP` = what this edit does to it).
    Either can be what the gate raised, so the property's weight is the max of the two.
    """
    out: dict = {}
    for src in (gv["state"], gv["cand"][pick] if pick < len(gv["cand"]) else {}):
        for name, g in src.items():
            m = _PROP_RX.match(name)
            if m:
                p = m.group(2)
                out[p] = max(out.get(p, 0.0), g)
    return out


# the interpretable non-property columns, grouped by what kind of argument they are.
# Which family wins the gate says what the decision was made ON, which is what several
# variants use to choose the shape of the sentence rather than its content.
FAMILIES = {
    "reliability": ["c_std_sum", "c_std_max", "c_snr_min", "ctx_std_r0", "ctx_support",
                    "ctx_log_support"],
    "magnitude": ["c_gap_reduction", "c_pred_gap", "c_cover_frac", "c_prob",
                  "c_prob_all", "c_prob_min", "c_n_sat_post", "c_n_props_helped"],
    "damage": ["c_damage", "c_dcount", "c_overshoot", "c_n_props_hurt",
               "c_worst_prop_delta", "c_worst_after"],
    "size": ["r_dheavy", "r_dmw", "r_heavy_to", "r_heavy_from"],
    "site": ["site_ok", "site_aromatic", "site_in_ring", "site_fused", "site_on_fg",
             "site_dist_to_fg", "site_crowd_2b", "site_on_guard", "site_frac_guard",
             "site_dist_to_guard", "site_degree", "site_num_h", "site_sym_equiv"],
    "sibling": ["c_rank", "c_prob_margin", "c_prob_z", "c_is_unique_best",
                "cand_similarity", "set_prob_spread"],
}
_FAM_OF = {c: f for f, cs in FAMILIES.items() for c in cs}

FAMILY_HEDGE = {
    "reliability": ("the spread on the matched pairs is what this turns on, so the hedge "
                    "belongs on reliability: say the predicted shift may not transfer"),
    "magnitude": ("the size of the move is what this turns on, so hedge on the number "
                  "itself: it is an average over matched pairs, not a measurement"),
    "damage": ("what this edit COSTS elsewhere is what it turns on, so the last sentence "
               "must name the constraint it puts at risk"),
    "size": ("the size of the edit is what separates these, so hedge on whether the "
             "smaller change is enough"),
    "site": ("where the edit lands is what separates these, so hedge on whether that "
             "site behaves like the matched pairs it was learned from"),
    "sibling": ("nothing about this candidate alone decides it — it is chosen relative "
                "to the others, so say the margin is what makes the call and how thin"),
}


def family_weights(gv: dict, pick: int) -> list:
    """[(family, mean gate), ...] worst-to-best, for the picked candidate."""
    acc: dict = {}
    for name, g in (gv["cand"][pick] if pick < len(gv["cand"]) else {}).items():
        f = _FAM_OF.get(name)
        if f:
            acc.setdefault(f, []).append(g)
    return sorted(((f, float(np.mean(v))) for f, v in acc.items()),
                  key=lambda kv: -kv[1])


# --------------------------------------------------------------------------- #
#  naive-material surgery
# --------------------------------------------------------------------------- #
_MAT_PROP = re.compile(r"^\s{4,}([A-Za-z_]+):\s")


def prune_material(material: str, pgates: dict, keep_min: float = 1.0,
                   keep_at_least: int = 2) -> str:
    """Drop the per-property lines the gate left below `keep_min`.

    `format_candidate_options` tables every constrained property for every candidate.
    The prompt already tells the LLM to ignore columns that are identical across
    candidates; the gate gives the per-state version of that — a property can differ
    and still be one the model did not weigh. Always keeps the top `keep_at_least` so a
    flat gate cannot empty the table.
    """
    if not pgates:
        return material
    order = sorted(pgates, key=lambda p: -pgates[p])
    keep = {p for p in order if pgates[p] >= keep_min} | set(order[:keep_at_least])
    out = []
    for line in material.splitlines():
        m = _MAT_PROP.match(line)
        if m and m.group(1) not in keep:
            continue
        out.append(line)
    return "\n".join(out)


# --------------------------------------------------------------------------- #
#  blocks
# --------------------------------------------------------------------------- #
def gtag(g: float) -> str:
    if g >= 1.25:
        return f"x{g:.2f} decisive here"
    if g >= AVERAGE_GATE:
        return f"x{g:.2f} important here"
    return f"x{g:.2f}"


def order_block(gv: dict, pick: int, pgates: dict, top: int = 3) -> str:
    """What to argue FROM, in the order the gate put it — the learned replacement for
    the fixed eliminate -> gap -> safe -> overshoot -> size -> rank ladder."""
    L = ["WHAT DECIDES THIS STATE (in order; the first line is the criterion the "
         "argument should be built on)"]
    fams = family_weights(gv, pick)[:top]
    props = sorted(pgates, key=lambda p: -pgates[p])[:top]
    for f, g in fams:
        L.append(f"  - {f}: {gtag(g)}")
    if props:
        L.append("  properties that matter here, most first: "
                 + ", ".join(f"{p} ({gtag(pgates[p])})" for p in props))
    L.append("  Anything not listed decided nothing in this state: do not argue from it.")
    return "\n".join(L)


# The contrast block and the naive candidate table are two DIFFERENT measurement
# systems, and mixing them makes the writer contradict itself. `r_drotb` is how many
# rotatable bonds the FRAGMENT SWAP itself contributes; the table's `rotB: 2 -> 4
# (Δ +2)` is what mmpdb PREDICTS for the whole molecule. Measured on the first smoke
# state they disagreed in surface reading — fragment ΔrotB 2 vs 0 while the predicted
# molecule ΔrotB was +2 for both — and the model duly wrote "#2 adds no rotatable
# bonds", faithful to the contrast block and false against the table. So a variant that
# carries both blocks has to either drop these rows or label them.
_COLLIDES = re.compile(r"^(r_d(?!fg__)|r_heavy_)")


def split_contrast(contrast: list) -> list:
    """Only the rows the naive table does NOT already speak about."""
    return [d for d in contrast if not _COLLIDES.match(d["feature"])]


_FRAG_NOTE = ("  These are what the FRAGMENT SWAP itself contributes, not the "
              "mmpdb-predicted change for the whole molecule in the table above. Never "
              "write one as if it were the other.")


def rival_block(ev: dict, pick: int, contrast=None, note: bool = False) -> str:
    rival = ev["rival"]
    L = [f"THE ONE TO ARGUE AGAINST: #{rival+1}"
         + ("  (chosen by the model's own candidate comparison)"
            if "candidate-attention" in ev.get("rival_rule", "") else
            "  (the nearest on score — the comparison did not single anyone out)"),
         "WHAT SEPARATES THEM"]
    for d in (ev["contrast"] if contrast is None else contrast):
        if d["favours"]:
            who = f"  -> favours #{pick+1 if d['favours']=='picked' else rival+1}"
        else:
            who = f"  -> larger for #{pick+1 if d['larger']=='picked' else rival+1}"
        L.append(f"  - #{pick+1}: {d['picked']}   |   #{rival+1}: {d['rival']}{who}   "
                 f"{gtag(d['gate'])}")
    if note:
        L.append(_FRAG_NOTE)
    return "\n".join(L)


def q_block(ev: dict, pick: int) -> str:
    qs = [c["q"] for c in ev["candidates"]]
    L = ["PREDICTED CHANCE that a search through each edit reaches the property box"]
    for i, q in enumerate(qs):
        tag = "   <-- the one to take" if i == pick else ""
        L.append(f"  #{i+1} {q:.2f}{tag}")
    best = int(np.argmax(qs))
    if best != pick:
        L.append(f"  NOTE #{best+1} scores higher ({qs[best]:.2f} against {qs[pick]:.2f}). "
                 f"Argue for #{pick+1} on the facts and CONCEDE that #{best+1} scores "
                 f"better rather than pretending it leads on everything.")
    else:
        m = sorted(qs)[-2] if len(qs) > 1 else 0.0
        L.append(f"  #{pick+1} is the highest; its margin over the next is "
                 f"{qs[pick]-m:+.2f}"
                 + (" — thin, so say the call is close." if qs[pick] - m < 0.05 else "."))
    return "\n".join(L)


def hedge_block(gv: dict, pick: int) -> str:
    fams = family_weights(gv, pick)
    if not fams:
        return ""
    f, g = fams[0]
    return f"HOW TO HEDGE\n  - {FAMILY_HEDGE.get(f, '')} ({f}, {gtag(g)})"


def product_block(row: dict, pick: int) -> str:
    """The post-edit molecule itself. LEAKY — see the module docstring."""
    c = row["candidates"][pick]
    smi = c.get("smiles")
    if not smi:
        return ""
    return (f"WHAT THE MOLECULE BECOMES (computed, not predicted)\n"
            f"  {smi}\n"
            f"  This is the exact product of the edit, so anything you say about it is "
            f"a fact rather than a prediction.")


# --------------------------------------------------------------------------- #
#  system prompts
# --------------------------------------------------------------------------- #
import importlib

_RS = importlib.import_module("5_rule_selection.reasoning")

SYS_BASE = _RS.SYSTEM          # the shipped one: 3 sentences, reason-first, no meta
SYS_TERSE = SYS_BASE.replace("3 sentences, 65 words or fewer IN TOTAL",
                             "2 sentences, 45 words or fewer IN TOTAL")


# --------------------------------------------------------------------------- #
#  the arms
# --------------------------------------------------------------------------- #
def _head(row: dict, ev: dict) -> list:
    return [f"MOLECULE: {row['state_smiles']}",
            f"(search depth {row.get('depth')}, {len(ev['candidates'])} candidate edits)",
            ""]


def _tail(pick: int, extra: str = "") -> list:
    return ["", f"WRITE THAT TURN for taking #{pick+1}: 3 sentences, 65 words or fewer "
                f"IN TOTAL, first person, and name the candidate you take only in the "
                f"LAST sentence." + (" " + extra if extra else "")]


def v_naive(row, ev, res, spec, naive):
    _c, material, spread = naive
    return SYS_BASE, _RS.prompt_naive(row, material, "", spread, ev["pick"])


def v_model(row, ev, res, spec, naive):
    return SYS_BASE, _RS.prompt_model(row, ev)


def v_model_a(row, ev, res, spec, naive):
    return SYS_BASE, _RS.prompt_model_a(row, ev)


def v_q_only(row, ev, res, spec, naive):
    """Control: the ranker's SCORE and nothing else from it.

    If this matches the attention arms, then what helps is knowing which rule the model
    likes — not knowing what it looked at — and every A/A_c prompt is overhead.
    """
    _c, material, spread = naive
    pick = ev["pick"]
    L = _head(row, ev) + ["CANDIDATES", material]
    if spread:
        L += ["", "SPREAD", spread]
    L += ["", q_block(ev, pick)] + _tail(pick)
    return SYS_BASE, "\n".join(L)


def v_gate_order(row, ev, res, spec, naive):
    """A as a decision ORDER over the shipped material. No facts added or removed."""
    gv = gate_vectors(res, spec)
    pick = ev["pick"]
    pg = property_gates(gv, pick)
    _c, material, spread = naive
    L = _head(row, ev) + [order_block(gv, pick, pg), "", "CANDIDATES", material]
    if spread:
        L += ["", "SPREAD", spread]
    L += _tail(pick, "Build the argument on the FIRST line of the deciding block and "
                     "take its numbers from the candidate table.")
    return SYS_BASE, "\n".join(L)


def v_gate_prune(row, ev, res, spec, naive):
    """A as a FILTER: the property lines it left below average are removed outright."""
    gv = gate_vectors(res, spec)
    pick = ev["pick"]
    pg = property_gates(gv, pick)
    _c, material, spread = naive
    L = _head(row, ev) + ["CANDIDATES  (only the properties that weigh on this state)",
                          prune_material(material, pg)]
    L += _tail(pick)
    return SYS_BASE, "\n".join(L)


def v_rival(row, ev, res, spec, naive):
    """A_c only: the shipped material plus a named opponent and what separates them."""
    _c, material, spread = naive
    pick = ev["pick"]
    L = _head(row, ev) + ["CANDIDATES", material]
    if spread:
        L += ["", "SPREAD", spread]
    L += ["", rival_block(ev, pick)]
    L += _tail(pick, f"Run the argument against #{ev['rival']+1} specifically, on a "
                     f"separating line above.")
    return SYS_BASE, "\n".join(L)


def v_hybrid(row, ev, res, spec, naive):
    """naive chemistry + A order + A_c rival + q. The predicted best of the honest arms."""
    gv = gate_vectors(res, spec)
    pick = ev["pick"]
    pg = property_gates(gv, pick)
    _c, material, spread = naive
    L = _head(row, ev) + [order_block(gv, pick, pg), "", "CANDIDATES", material]
    if spread:
        L += ["", "SPREAD", spread]
    L += ["", rival_block(ev, pick), "", q_block(ev, pick)]
    L += _tail(pick, f"Build it on the first line of the deciding block, run it against "
                     f"#{ev['rival']+1} on a separating line, and let the choice fall out.")
    return SYS_BASE, "\n".join(L)


def v_hybrid_hedge(row, ev, res, spec, naive):
    """`hybrid`, plus the KIND of hedge the winning feature family implies."""
    gv = gate_vectors(res, spec)
    pick = ev["pick"]
    _sys, base = v_hybrid(row, ev, res, spec, naive)
    hb = hedge_block(gv, pick)
    if not hb:
        return _sys, base
    parts = base.rsplit("\n\nWRITE THAT TURN", 1)
    return _sys, parts[0] + "\n\n" + hb + "\n\nWRITE THAT TURN" + parts[1]


def v_hybrid_pruned(row, ev, res, spec, naive):
    """`hybrid` on the PRUNED table: does removing the ignored columns help or hurt?"""
    gv = gate_vectors(res, spec)
    pick = ev["pick"]
    pg = property_gates(gv, pick)
    _c, material, spread = naive
    L = _head(row, ev) + [order_block(gv, pick, pg), "",
                          "CANDIDATES  (only the properties that weigh on this state)",
                          prune_material(material, pg)]
    L += ["", rival_block(ev, pick), "", q_block(ev, pick)]
    L += _tail(pick, f"Build it on the first line of the deciding block and run it "
                     f"against #{ev['rival']+1}.")
    return SYS_BASE, "\n".join(L)


def v_hybrid_terse(row, ev, res, spec, naive):
    """`hybrid` at 2 sentences. Prices verbosity separately from content."""
    _sys, p = v_hybrid(row, ev, res, spec, naive)
    return SYS_TERSE, p.replace("3 sentences, 65 words or fewer IN TOTAL",
                                "2 sentences, 45 words or fewer IN TOTAL")


def v_contrast_only(row, ev, res, spec, naive):
    """No candidate table at all: the deciding block, the rival contrast and q.

    The extreme of "A decides which facts appear". If this holds up, most of the table
    was never doing work.
    """
    gv = gate_vectors(res, spec)
    pick = ev["pick"]
    pg = property_gates(gv, pick)
    L = _head(row, ev) + [order_block(gv, pick, pg), ""]
    L.append("WHAT THE STATE STILL NEEDS")
    for s_ in (ev["state"] or []):
        L.append(f"  - {s_['text']}   {gtag(s_['gate'])}")
    L.append("")
    L.append("THE CANDIDATES, on what the gate raised")
    for c in ev["candidates"]:
        tag = ("   <-- the one to take" if c["index"] == pick else
               ("   <-- the closest alternative" if c["index"] == ev["rival"] else ""))
        L.append(f"  #{c['index']+1} {c['rule']}{tag}")
        for f in c["features"]:
            L.append(f"        {f['text']}   {gtag(f['gate'])}")
    L += ["", rival_block(ev, pick), "", q_block(ev, pick)]
    L += _tail(pick)
    return SYS_BASE, "\n".join(L)


def v_product(row, ev, res, spec, naive):
    """LEAKY ARM — reported separately. See the module docstring."""
    _sys, base = v_hybrid(row, ev, res, spec, naive)
    pb = product_block(row, ev["pick"])
    if not pb:
        return _sys, base
    parts = base.rsplit("\n\nWRITE THAT TURN", 1)
    return _sys, parts[0] + "\n\n" + pb + "\n\nWRITE THAT TURN" + parts[1]


def v_hybrid_split(row, ev, res, spec, naive):
    """`hybrid` with the colliding contrast rows REMOVED.

    The property argument then comes only from the naive table, and A_c supplies only
    what the table cannot say — the site, the fragment identity, the rule's support,
    the sibling margin.
    """
    gv = gate_vectors(res, spec)
    pick = ev["pick"]
    pg = property_gates(gv, pick)
    _c, material, spread = naive
    L = _head(row, ev) + [order_block(gv, pick, pg), "", "CANDIDATES", material]
    if spread:
        L += ["", "SPREAD", spread]
    L += ["", rival_block(ev, pick, contrast=split_contrast(ev["contrast"])),
          "", q_block(ev, pick)]
    L += _tail(pick, f"Take every property number from the candidate table, run the "
                     f"argument against #{ev['rival']+1}, and let the choice fall out.")
    return SYS_BASE, "\n".join(L)


def v_hybrid_label(row, ev, res, spec, naive):
    """`hybrid` with the colliding rows KEPT but labelled as fragment-level.

    The paired alternative to `hybrid_split`: is the fix to hide the second measurement
    system, or to name it?
    """
    gv = gate_vectors(res, spec)
    pick = ev["pick"]
    pg = property_gates(gv, pick)
    _c, material, spread = naive
    L = _head(row, ev) + [order_block(gv, pick, pg), "", "CANDIDATES", material]
    if spread:
        L += ["", "SPREAD", spread]
    L += ["", rival_block(ev, pick, note=True), "", q_block(ev, pick)]
    L += _tail(pick, f"Run the argument against #{ev['rival']+1}.")
    return SYS_BASE, "\n".join(L)


def v_rival_split(row, ev, res, spec, naive):
    """A_c alone, filtered — isolates the rival from the gate ordering."""
    _c, material, spread = naive
    pick = ev["pick"]
    L = _head(row, ev) + ["CANDIDATES", material]
    if spread:
        L += ["", "SPREAD", spread]
    L += ["", rival_block(ev, pick, contrast=split_contrast(ev["contrast"]))]
    L += _tail(pick, f"Run the argument against #{ev['rival']+1} specifically.")
    return SYS_BASE, "\n".join(L)


def v_naive_tail(row, ev, res, spec, naive):
    """CONTROL. The naive material with THIS module's head and tail wording, no A at all.

    Every arm below `naive` uses `_tail` ("name the candidate you take only in the LAST
    sentence") while `reasoning.prompt_naive` ends with "The candidate to select is #N.
    WRITE THE REASON for taking it over the others". Measured, that difference alone
    moves the judge's `order` score from 0.41 to ~0.95, so without this control every
    A-based arm gets credit for a wording change. Whatever this arm beats `naive` by is
    the prompt; whatever an A arm beats THIS by is the attention.
    """
    _c, material, spread = naive
    pick = ev["pick"]
    L = _head(row, ev) + ["CANDIDATES", material]
    if spread:
        L += ["", "SPREAD", spread]
    L += _tail(pick)
    return SYS_BASE, "\n".join(L)


def v_prune_only(row, ev, res, spec, naive):
    """CONTROL's partner: the PRUNED table under the ORIGINAL naive wording.

    With `naive`, `naive_tail` and `gate_prune` this closes a 2x2 — {full, pruned}
    material x {original, new} wording — so the gate's filtering effect is identified
    rather than entangled with the instruction.
    """
    gv = gate_vectors(res, spec)
    pick = ev["pick"]
    pg = property_gates(gv, pick)
    _c, material, spread = naive
    return SYS_BASE, _RS.prompt_naive(row, prune_material(material, pg), "", spread, pick)


# ---- round 2: built from what round 1 measured --------------------------- #
# Round 1, judged on 300 states for factual correctness (grounded AND direction AND
# comparison), against the CONTROL `naive_tail` = 0.840:
#     model 0.847  gate_prune 0.843  q_only 0.840  rival 0.826  ...  hybrid 0.630
#     model_a 0.477
# Two things came out of that. (1) `model_a` is the worst arm and `model` the best, and
# they differ by exactly the rival + contrast block: A ALONE is harmful, A WITH A_c is
# not. (2) every arm that carried `order_block` landed at 0.59-0.66, so naming the
# deciding feature families up front is what poisons the hybrids — the writer argues
# from the family label instead of from the numbers.
# So round 2 builds the hybrid the other way round: `model`'s structure as the frame,
# with the property table added back only where naive is strong, and no order block.


def _pair_table(row, naive, pick: int, rival: int) -> str:
    """The naive per-property rows for the two candidates that are actually in play.

    `model` has no property table at all and is still the cleanest arm, but it is also
    the arm that talks about concrete property values least. Tabling all four candidates
    is what `hybrid` did and it drowned; two is the comparison the text is allowed to
    make anyway.
    """
    _c, material, _s = naive
    keep, cur = [], None
    for line in material.splitlines():
        m = re.match(r"^#(\d+)\s", line)
        if m:
            cur = int(m.group(1)) - 1
        if cur in (pick, rival):
            keep.append(line)
    return "\n".join(keep)


def v_model_table(row, ev, res, spec, naive):
    """`model` + the naive property rows for the picked candidate and its rival."""
    pick, rival = ev["pick"], ev["rival"]
    base = _RS.prompt_model(row, ev)
    tbl = _pair_table(row, naive, pick, rival)
    if not tbl:
        return SYS_BASE, base
    parts = base.rsplit("\n\nWRITE THAT TURN", 1)
    add = ("THE TWO IN PLAY, property by property (mmpdb predictions for the whole "
           "molecule — these are the numbers to quote)\n" + tbl)
    return SYS_BASE, parts[0] + "\n\n" + add + "\n\nWRITE THAT TURN" + parts[1]


def v_model_hedge(row, ev, res, spec, naive):
    """`model` + the hedging demand. Round 1 scored `hedged` at 0.00-0.04 for EVERY arm:
    reasoning.SYSTEM never asks for it, and stage 4's own prompt requires it."""
    base = _RS.prompt_model(row, ev)
    parts = base.rsplit("\n\nWRITE THAT TURN", 1)
    add = ("EVERY NUMBER ABOVE IS A PREDICTION\n"
           "  The deltas are mmpdb averages over matched pairs, not measurements. The "
           "commitment has to read as an attempt whose result will be measured — "
           "'should', 'ought to land near', 'try ... and re-measure' — never as a "
           "settled outcome, and the LAST sentence must name something still wrong "
           "after the edit or the measurement that would send you back.")
    return SYS_BASE, parts[0] + "\n\n" + add + "\n\nWRITE THAT TURN" + parts[1]


def v_model_table_hedge(row, ev, res, spec, naive):
    """Both round-2 additions at once."""
    _sys, base = v_model_table(row, ev, res, spec, naive)
    parts = base.rsplit("\n\nWRITE THAT TURN", 1)
    add = ("EVERY NUMBER ABOVE IS A PREDICTION\n"
           "  The deltas are mmpdb averages over matched pairs, not measurements. Hedge "
           "the commitment as an attempt to be measured, and let the LAST sentence name "
           "what is still wrong after the edit.")
    return SYS_BASE, parts[0] + "\n\n" + add + "\n\nWRITE THAT TURN" + parts[1]


def v_naive_hedge(row, ev, res, spec, naive):
    """CONTROL for the two above: the same hedging demand on the naive_tail baseline."""
    _sys, base = v_naive_tail(row, ev, res, spec, naive)
    parts = base.rsplit("\n\nWRITE THAT TURN", 1)
    add = ("EVERY NUMBER ABOVE IS A PREDICTION\n"
           "  The deltas are mmpdb averages over matched pairs, not measurements. Hedge "
           "the commitment as an attempt to be measured, and let the LAST sentence name "
           "what is still wrong after the edit.")
    return SYS_BASE, parts[0] + "\n\n" + add + "\n\nWRITE THAT TURN" + parts[1]


# ---- round 3: two leaks the winner still had ------------------------------ #
# Reading `model_hedge`'s own output on 300 states, twice:
#   "logS ... sits 0.03 SCALED UNITS below its lower bound"  -> st_lo_z is IQR-scaled,
#      an internal normalisation with no chemical meaning, and `pretty()` prints it
#      as if it were a measurement;
#   "while #1 has the highest Q"                             -> the bare symbol for the
#      predicted success probability, leaked into prose.
# reasoning.SYSTEM already forbids naming the model, but not its notation. Both are
# one-line prompt fixes, and neither costs any of the content that made the arm win.
_NOTATION = (
    "TWO THINGS NOT TO WRITE\n"
    "  - Never write \"q\", or any other symbol, for the success probability. It is a "
    "plain number: \"a 0.94 chance\", \"0.94 against 0.97\".\n"
    "  - Never write \"scaled units\", \"z\", \"normalised\" or any distance in them. A "
    "bound distance given in scaled units is an internal normalisation, not a "
    "measurement: say the property is below/above its bound, or quote the property's "
    "own value and the bound, and nothing about the scale.")


def _inject(base: str, block: str) -> str:
    parts = base.rsplit("\n\nWRITE THAT TURN", 1)
    if len(parts) != 2:
        return base + "\n\n" + block
    return parts[0] + "\n\n" + block + "\n\nWRITE THAT TURN" + parts[1]


_HEDGE = ("EVERY NUMBER ABOVE IS A PREDICTION\n"
          "  The deltas are mmpdb averages over matched pairs, not measurements. The "
          "commitment has to read as an attempt whose result will be measured — "
          "'should', 'ought to land near', 'try ... and re-measure' — never as a "
          "settled outcome, and the LAST sentence must name something still wrong "
          "after the edit or the measurement that would send you back.")


def v_model_hedge_clean(row, ev, res, spec, naive):
    """`model_hedge` with the notation leaks closed. The candidate for shipping."""
    return SYS_BASE, _inject(_inject(_RS.prompt_model(row, ev), _HEDGE), _NOTATION)


def v_model_clean(row, ev, res, spec, naive):
    """CONTROL: the notation fix WITHOUT the hedging demand, to keep the two separable."""
    return SYS_BASE, _inject(_RS.prompt_model(row, ev), _NOTATION)


def v_naive_hedge_clean(row, ev, res, spec, naive):
    """CONTROL: both round-2/3 additions on the naive baseline, no attention at all.

    If this matches `model_hedge_clean`, then everything measured is the prompt and the
    attention is decoration — which is the null this whole sweep has to be able to state.
    """
    _sys, base = v_naive_tail(row, ev, res, spec, naive)
    return SYS_BASE, _inject(_inject(base, _HEDGE), _NOTATION)


# ---- round 4: the one axis the control actually won on -------------------- #
# Judged on 300 states, `naive_hedge_clean` (no attention) beat `model_hedge_clean` by
# 0.034 on `clean`. Broken out by axis the gap is almost entirely ONE thing:
#     grounded    0.907 vs 0.910   (level)
#     comparison  0.903 vs 0.913   (level)
#     direction   0.963 vs 0.890   <- the whole gap
# and the cause is visible in the prompts. The naive table states the box for every
# property on the line ("target 2.580 .. 4.010 -> IN range"), so which way a property
# has to move is unambiguous. `prompt_model`'s state block renders st_lo_z / st_hi_z as
# a DISTANCE to a bound with the bound's own range nowhere in sight, so the writer has
# to infer the direction and sometimes gets it backwards. That is a missing block, not
# a limit of attention.


def box_block(row: dict) -> str:
    """Per constrained property: where it is, where it has to be, which way to move.

    Not new information — the naive table carries all of it — and not attention. It is
    here so the two families of prompt are held to the same standard on direction.
    """
    props = row.get("state_props") or {}
    L = ["THE BOX (current value, the range it must end in, and which way that is)"]
    any_row = False
    for t in (row.get("targets") or []):
        p = t.get("property")
        v = props.get(p)
        lo, hi = t.get("min"), t.get("max")
        if v is None:
            continue
        rng = (f"{lo:g} .. {hi:g}" if lo is not None and hi is not None else
               (f"at least {lo:g}" if lo is not None else f"at most {hi:g}"))
        if lo is not None and v < lo:
            need = "must RISE"
        elif hi is not None and v > hi:
            need = "must FALL"
        else:
            need = "already inside — keep it there"
        L.append(f"  - {p}: {v:g}, target {rng} -> {need}")
        any_row = True
    return "\n".join(L) if any_row else ""


def v_model_hedge_box(row, ev, res, spec, naive):
    """`model_hedge_clean` + the explicit box. The round-4 shipping candidate."""
    base = _inject(_inject(_RS.prompt_model(row, ev), _HEDGE), _NOTATION)
    bb = box_block(row)
    return SYS_BASE, (_inject(base, bb) if bb else base)


def v_model_box(row, ev, res, spec, naive):
    """CONTROL: the box alone on top of `model`, so its effect is separable from hedging."""
    bb = box_block(row)
    base = _RS.prompt_model(row, ev)
    return SYS_BASE, (_inject(base, bb) if bb else base)


def v_naive_hedge_clean_box(row, ev, res, spec, naive):
    """CONTROL: the same box on the no-attention winner, in case it helps there too."""
    _sys, base = v_naive_hedge_clean(row, ev, res, spec, naive)
    bb = box_block(row)
    return SYS_BASE, (_inject(base, bb) if bb else base)


# ---- round 5: the trade-off, and whether q alone buys the good half ------- #
# After four rounds the two families split cleanly on the judge, and NOT on the axis
# this sweep set out to test:
#     naive_hedge_clean (no attention)  clean 0.897  direction 0.963  order 0.713
#     model_hedge_box   (A + A_c)       clean 0.866  direction 0.910  order 0.910
# The control is more often FACTUALLY right; the attention arm is far more often built
# reason-first, which is what stage 4's own prompt demands and what `clean` does not
# score. The question left is whether the reason-first discipline needs the attention at
# all, or only the score and the concession line that comes with it — `q_only` was
# already level with the control back in round 1, on the weaker wording.


def v_naive_q(row, ev, res, spec, naive):
    """The no-attention winner + the ranker's q and its concede-the-leader line.

    No gated features, no contrast rows: the only thing taken from the model is the
    number it puts on each candidate and the instruction to concede when the committed
    rule is not its favourite. If this reaches the attention arm's `order` while keeping
    the control's `direction`, then nothing in this sweep needs A or A_c.
    """
    _sys, base = v_naive_tail(row, ev, res, spec, naive)
    parts = base.rsplit("\n\nWRITE THAT TURN", 1)
    body = parts[0] + "\n\n" + q_block(ev, ev["pick"])
    body = body + "\n\n" + _HEDGE + "\n\n" + _NOTATION
    return SYS_BASE, body + "\n\nWRITE THAT TURN" + parts[1]


def v_naive_q_rival(row, ev, res, spec, naive):
    """`naive_q` plus ONE line of A_c: which candidate to argue against, no contrast rows.

    The minimal use of the candidate comparison — a name, not a table. Round 1 showed the
    contrast ROWS are what break the property talk; this keeps the opponent and drops
    them.
    """
    _sys, base = v_naive_q(row, ev, res, spec, naive)
    parts = base.rsplit("\n\nWRITE THAT TURN", 1)
    line = (f"THE ONE TO ARGUE AGAINST: #{ev['rival']+1}"
            + ("  (the model's own candidate comparison singled it out)"
               if "candidate-attention" in ev.get("rival_rule", "")
               else "  (nearest on score)")
            + "\n  Use the candidate table above for every number about it.")
    return SYS_BASE, parts[0] + "\n\n" + line + "\n\nWRITE THAT TURN" + parts[1]


# ---- round 6: what `naive_hedge_clean` was still missing ------------------ #
# `reasoning.prompt_naive` takes a `block` argument and `reasoning.naive_material`
# never fills it, so every naive arm in rounds 1-5 ran WITHOUT the two computed blocks
# the real stage-4 prompt is built on: `fragment_names.landing_safety` (the decision
# table — rank / safe landings / IN-after / gap closed / +heavy / ΔMW / breaks /
# overshoot, plus the verdict) and `fragment_names.round_context` (how far to trust the
# numbers: how many constraints are still out, the tightest ceiling, whether the edit
# site separates the candidates). The baseline was therefore weaker than what stage 4
# actually does, which makes every "naive" number in this file a floor rather than the
# real comparison.
#
# It also settles a claim made earlier in this README. `landing_safety` ends with, when
# the committed rule is not the one its own ordering picks:
#     "NOTE running the order above lands on #3 ... but the candidate to select is #2.
#      Sentence 2 has to concede that ... take it as the next thing to TEST"
# so the concession the ranker's `q` was credited with is already available WITHOUT the
# ranker, computed from the same table the prose quotes.


def _targets(row: dict) -> dict:
    return {t["property"]: (t.get("min"), t.get("max"))
            for t in (row.get("targets") or [])}


def _fn():
    import importlib
    import os
    import sys
    d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "4_sftdata_gen")
    if d not in sys.path:
        sys.path.insert(0, d)
    return importlib.import_module("fragment_names")


def landing_block(row: dict, naive, pick: int) -> str:
    cands = naive[0]
    try:
        c = cands[pick]
        return _fn().landing_safety(cands, row.get("state_props"), _targets(row),
                                    from_smiles=c["from_smiles"],
                                    to_smiles=c["to_smiles"],
                                    mol_smiles=row.get("state_smiles"),
                                    anchors=c.get("anchors")) or ""
    except Exception:  # noqa: BLE001
        return ""


def round_block(row: dict, naive) -> str:
    try:
        return _fn().round_context(row.get("state_props"), _targets(row),
                                   candidates=naive[0],
                                   mol_smiles=row.get("state_smiles")) or ""
    except Exception:  # noqa: BLE001
        return ""


def _nhc_body(row, ev, res, spec, naive, extra_blocks=(), blocks_first=False):
    """`naive_hedge_clean`'s prompt with blocks inserted, and the write instruction last."""
    _c, material, spread = naive
    pick = ev["pick"]
    L = _head(row, ev)
    pre = [b for b in extra_blocks if b]
    if blocks_first:
        L += [_HEDGE, "", _NOTATION, ""]
    L += ["CANDIDATES", material]
    if spread:
        L += ["", "SPREAD", spread]
    for b in pre:
        L += ["", b]
    if not blocks_first:
        L += ["", _HEDGE, "", _NOTATION]
    return L, pick


def v_nhc_landing(row, ev, res, spec, naive):
    """+ the pre-computed decision table the prose is supposed to quote."""
    pick = ev["pick"]
    L, _ = _nhc_body(row, ev, res, spec, naive,
                     extra_blocks=["THE COUNTING, ALREADY DONE (quote these numbers; "
                                   "do not recompute any of them)\n"
                                   + landing_block(row, naive, pick)])
    return SYS_BASE, "\n".join(L + _tail(pick))


def v_nhc_round(row, ev, res, spec, naive):
    """+ posture: how far to trust the numbers, and whether the site separates them."""
    L, pick = _nhc_body(row, ev, res, spec, naive,
                        extra_blocks=["THIS ROUND (identical for every candidate, so it "
                                      "cannot pick one — it says how far to trust the "
                                      "numbers)\n" + round_block(row, naive)])
    return SYS_BASE, "\n".join(L + _tail(pick))


def v_nhc_full(row, ev, res, spec, naive):
    """Both computed blocks: the closest this sweep gets to stage 4's real material."""
    pick = ev["pick"]
    L, _ = _nhc_body(row, ev, res, spec, naive, extra_blocks=[
        "THIS ROUND (identical for every candidate, so it cannot pick one)\n"
        + round_block(row, naive),
        "THE COUNTING, ALREADY DONE (quote these numbers; do not recompute any)\n"
        + landing_block(row, naive, pick)])
    return SYS_BASE, "\n".join(L + _tail(pick))


_PLAN = ("  Sentence 1 — the gap that drives it: the property, its value, its target and "
         "the direction it must move. Sentence 2 — the ONE criterion that decides, and "
         "the candidate it rules out. Sentence 3 — take it as a trial and name what is "
         "STILL wrong afterwards. The candidate you take is named in sentence 3 and "
         "NOWHERE earlier.")


def v_nhc_plan(row, ev, res, spec, naive):
    """An explicit 3-sentence skeleton. `order` is the arm's weakest axis (0.713) and
    stage 4's own prompt fixes it this way rather than by asking."""
    L, pick = _nhc_body(row, ev, res, spec, naive)
    return SYS_BASE, "\n".join(L + _tail(pick) + [_PLAN])


def v_nhc_blocksfirst(row, ev, res, spec, naive):
    """The hedging and notation blocks BEFORE the table, so the ordering instruction is
    the last thing read. Round 3 showed those two blocks buy grounding and cost ordering
    (0.977 -> 0.713); this tests whether the cost is placement rather than content."""
    L, pick = _nhc_body(row, ev, res, spec, naive, blocks_first=True)
    return SYS_BASE, "\n".join(L + _tail(pick))


_QUOTE = ("EVERY NUMBER YOU WRITE MUST APPEAR ABOVE\n"
          "  Copy it exactly — 48.150, never 48.2. If a number is not in the material, "
          "do not write it, and do not derive a new one by subtracting or dividing two "
          "that are. Compare two candidates only on a line that already carries both of "
          "their values; never build a comparison out of two separate blocks.")


def v_nhc_quote(row, ev, res, spec, naive):
    """+ the exact-quotation rule. `grounded` 0.907 / `comparison` 0.903 are the two
    remaining error modes and both are fabrication rather than misreading."""
    L, pick = _nhc_body(row, ev, res, spec, naive, extra_blocks=[_QUOTE])
    return SYS_BASE, "\n".join(L + _tail(pick))


# ---- round 7: combine what round 6 separated ----------------------------- #
# Round 6, judged on the same 300 states against `naive_hedge_clean` (clean 0.897):
#   nhc_blocksfirst  0.910  order .810  hedged .393   the hedging/notation blocks moved
#                                                     BEFORE the table. Better on every
#                                                     axis at identical content and
#                                                     length — the cost those two blocks
#                                                     carried in round 3 was PLACEMENT,
#                                                     not content: they were sitting
#                                                     between the table and the write
#                                                     instruction and burying it.
#   nhc_landing      0.897  order .803  hedged .467   the computed decision table. Free
#                                                     on clean, large on the other two.
#   nhc_plan         0.847  order .740  hedged .893   the 3-sentence skeleton makes
#                                                     hedging land almost always and
#                                                     costs 0.050 of clean (significant).
#   nhc_quote        0.723                            FORBIDDING derivation collapsed
#                                                     `comparison` .903 -> .737. The rule
#                                                     meant to stop fabrication stopped
#                                                     the correct comparisons instead.
#   nhc_round        0.870  order .633                the posture block hurts ordering.
# So: keep the placement, keep the counting, and find out whether the placement pays for
# the plan's clean cost.


def v_nhc_bf_landing(row, ev, res, spec, naive):
    """The two free wins together: block placement + the computed decision table."""
    pick = ev["pick"]
    L, _ = _nhc_body(row, ev, res, spec, naive, blocks_first=True,
                     extra_blocks=["THE COUNTING, ALREADY DONE (quote these numbers; "
                                   "do not recompute any of them)\n"
                                   + landing_block(row, naive, pick)])
    return SYS_BASE, "\n".join(L + _tail(pick))


def v_nhc_bf_plan(row, ev, res, spec, naive):
    """Does the placement pay for the skeleton's cost? `nhc_plan` lost 0.050 of clean
    with the blocks in the old position; this is the same skeleton with them moved."""
    L, pick = _nhc_body(row, ev, res, spec, naive, blocks_first=True)
    return SYS_BASE, "\n".join(L + _tail(pick) + [_PLAN])


def v_nhc_bf_landing_plan(row, ev, res, spec, naive):
    """All three. The most-instructed arm in the sweep."""
    pick = ev["pick"]
    L, _ = _nhc_body(row, ev, res, spec, naive, blocks_first=True,
                     extra_blocks=["THE COUNTING, ALREADY DONE (quote these numbers; "
                                   "do not recompute any of them)\n"
                                   + landing_block(row, naive, pick)])
    return SYS_BASE, "\n".join(L + _tail(pick) + [_PLAN])


def v_nhc_bf_full(row, ev, res, spec, naive):
    """Placement + both computed blocks, no skeleton."""
    pick = ev["pick"]
    L, _ = _nhc_body(row, ev, res, spec, naive, blocks_first=True, extra_blocks=[
        "THIS ROUND (identical for every candidate, so it cannot pick one)\n"
        + round_block(row, naive),
        "THE COUNTING, ALREADY DONE (quote these numbers; do not recompute any)\n"
        + landing_block(row, naive, pick)])
    return SYS_BASE, "\n".join(L + _tail(pick))


# ---- round 8: three axes nothing in rounds 1-7 varied --------------------- #
# Every arm so far was generated at temperature 0.3 and asked for "3 sentences, 65 words
# or fewer". Round 7 left `nhc_bf_landing` and `nhc_blocksfirst` tied at clean 0.910, so
# the next thing to try is not another block but the knobs around them:
#   temperature  0.3 -> 0.0. Sampling noise is a plausible share of the ~9% of spans that
#                still fabricate, and greedy costs nothing.
#   budget       65 -> 90 words. Every arm lands at 41-46 words, so the cap is not what
#                is binding; a bigger one tests whether the model is compressing away
#                the qualifications it needs to be correct.
#   hedge        the full block (5 lines) -> one clause on the last sentence. `nhc_plan`
#                got hedged .970 from a skeleton and `nhc_bf_landing` .448 from the
#                block; a one-line requirement is the cheap middle.


def v_nhc_bf_landing_t0(row, ev, res, spec, naive):
    """Identical to `nhc_bf_landing`. Exists only so one arm can be GENERATED at
    temperature 0 while the rest of the cache stays at 0.3 — the sweep applies its
    --temperature per run, and `--append --variants` regenerates only the named arm."""
    return v_nhc_bf_landing(row, ev, res, spec, naive)


def v_nhc_bf_landing_w90(row, ev, res, spec, naive):
    """`nhc_bf_landing` with the word budget raised from 65 to 90."""
    sysmsg, p = v_nhc_bf_landing(row, ev, res, spec, naive)
    return (sysmsg.replace("3 sentences, 65 words or fewer IN TOTAL",
                           "3 sentences, 90 words or fewer IN TOTAL"),
            p.replace("3 sentences, 65 words or fewer IN TOTAL",
                      "3 sentences, 90 words or fewer IN TOTAL"))


_HEDGE_LITE = ("ONE REQUIREMENT ON THE LAST SENTENCE\n"
               "  It takes the edit as an attempt to be measured, not a settled result, "
               "and it names one thing still wrong afterwards.")


def v_nhc_bf_landing_lite(row, ev, res, spec, naive):
    """`nhc_bf_landing` with the five-line hedging block replaced by one clause."""
    sysmsg, p = v_nhc_bf_landing(row, ev, res, spec, naive)
    return sysmsg, p.replace(_HEDGE, _HEDGE_LITE)


VARIANTS = {
    "naive": v_naive,                    # the baseline to beat
    "naive_tail": v_naive_tail,          # CONTROL: naive material, new wording
    "prune_only": v_prune_only,          # CONTROL: pruned material, old wording
    "model": v_model,                    # shipped A + A_c prompt
    "model_a": v_model_a,                # A alone
    "q_only": v_q_only,                  # control: the score, no attention
    "gate_order": v_gate_order,          # A as a decision order
    "gate_prune": v_gate_prune,          # A as a column filter
    "rival": v_rival,                    # A_c alone, on the shipped table
    "hybrid": v_hybrid,                  # naive + order + rival + q
    "hybrid_split": v_hybrid_split,      # the same, colliding contrast rows removed
    "hybrid_label": v_hybrid_label,      # the same, colliding rows labelled instead
    "rival_split": v_rival_split,
    "hybrid_hedge": v_hybrid_hedge,
    "hybrid_pruned": v_hybrid_pruned,
    "hybrid_terse": v_hybrid_terse,
    "contrast_only": v_contrast_only,    # no table at all
    "model_table": v_model_table,        # round 2: model frame + the two rows in play
    "model_hedge": v_model_hedge,        # round 2: model frame + hedging demand
    "model_table_hedge": v_model_table_hedge,
    "naive_hedge": v_naive_hedge,        # round 2 CONTROL for the hedging demand
    "model_hedge_clean": v_model_hedge_clean,   # round 3: the shipping candidate
    "model_clean": v_model_clean,               # round 3: notation fix alone
    "naive_hedge_clean": v_naive_hedge_clean,   # round 3 CONTROL: no attention
    "model_hedge_box": v_model_hedge_box,       # round 4: the shipping candidate
    "model_box": v_model_box,                   # round 4: the box alone
    "naive_hedge_clean_box": v_naive_hedge_clean_box,   # round 4 CONTROL
    "naive_q": v_naive_q,                # round 5: control + q, no attention maps
    "naive_q_rival": v_naive_q_rival,    # round 5: + one line of A_c
    "nhc_landing": v_nhc_landing,        # round 6: + the computed decision table
    "nhc_round": v_nhc_round,            # round 6: + the posture block
    "nhc_full": v_nhc_full,              # round 6: + both (stage 4's real material)
    "nhc_plan": v_nhc_plan,              # round 6: explicit 3-sentence skeleton
    "nhc_blocksfirst": v_nhc_blocksfirst,
    "nhc_quote": v_nhc_quote,
    "nhc_bf_landing": v_nhc_bf_landing,          # round 7: placement + counting
    "nhc_bf_plan": v_nhc_bf_plan,                # round 7: placement + skeleton
    "nhc_bf_landing_plan": v_nhc_bf_landing_plan,
    "nhc_bf_full": v_nhc_bf_full,
    "nhc_bf_landing_t0": v_nhc_bf_landing_t0,      # round 8: greedy
    "nhc_bf_landing_w90": v_nhc_bf_landing_w90,    # round 8: 90-word budget
    "nhc_bf_landing_lite": v_nhc_bf_landing_lite,  # round 8: one-clause hedge
    "product": v_product,                # LEAKY, reported apart
}
LEAKY = {"product"}
