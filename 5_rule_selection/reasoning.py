"""Reasoning text for one rule-selection decision — naive material, or the model's own attention.

Two ways to hand an LLM the evidence for "why this edit":

``--mode naive``  the shipped path, unchanged: the deterministic candidate tables from
                  ``4_sftdata_gen/fragment_names.py`` (current value, predicted
                  Δ ± std, whether it lands in the box, what the fragment is). Every
                  candidate contributes every constrained property, and nothing says
                  which of those numbers actually drove the decision — the LLM has to
                  guess that from the table. That module is imported, never modified.

``--mode model``  the evidence is read out of the trained rule-selection model:

                  * **A — context-feature relevance.** The gating softmax over the
                    G + N·R interpretable feature tokens, conditioned on the molecule.
                    Only the top-weighted features enter the prompt, each with the share
                    of attention it received, so the prompt states WHICH facts the
                    decision was actually made on instead of listing all of them.
                  * **A_c — candidate comparison.** The candidate self-attention row of
                    the selected edit says which rival its representation was built
                    against. That rival — not all three others — becomes the contrast
                    the text has to make, and its own top features are supplied for it.

                  The model also supplies `q̂`, its predicted success probability per
                  candidate, so the text can be explicit about how big the margin is.

Both modes call the same vLLM endpoint and emit the same record shape, so the two texts
can be read side by side for the same state.

Usage:
    python -m 5_rule_selection.reasoning \
        --states .../dataset_fixed/states.jsonl --ckpt .../ckpt/best.pt \
        --tensors .../tensors --ctx-dir .../ctx_morgan \
        --mode both --limit 40 --out .../reasoning_compare.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np
import torch

from .features import FeatureSpec
from .model import RuleSelector

# One URL or a comma-separated fleet. Requests are handed out round-robin, so two
# servers halve the wall clock of a comparison run without any other change.
VLLM_URLS = [u.strip() for u in os.environ.get(
    "VLLM_URLS", os.environ.get("VLLM_URL", "http://localhost:8082/v1")).split(",")
    if u.strip()]
VLLM = VLLM_URLS[0]
MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen3.6-27B")

# --------------------------------------------------------------------------- #
#  turning column names into something a chemist reads
# --------------------------------------------------------------------------- #
GLOSS = {
    "c_pred_gap": "predicted remaining gap after the edit",
    "c_gap_reduction": "predicted gap closed by the edit",
    "c_n_sat_now": "constraints already satisfied",
    "c_n_sat_post": "constraints satisfied after the edit",
    "c_dcount": "change in satisfied-constraint count",
    "c_damage": "satisfied constraints the edit would break",
    "c_n_props_helped": "constraints the edit moves the right way",
    "c_n_props_hurt": "constraints the edit moves the wrong way",
    "c_fix_worst": "does it move the worst-violated property",
    "c_worst_after": "worst remaining violation after the edit",
    "c_cos": "alignment of the edit's Δ with the direction needed",
    "c_cover_frac": "fraction of the needed direction covered",
    "c_overshoot": "predicted overshoot past the far edge of the box",
    "c_std_max": "widest Δ spread among the properties (reliability)",
    "c_prob": "expected satisfied count from the move's own Δ spread",
    "c_prob_min": "weakest per-constraint success probability",
    "c_prob_all": "probability all constraints land",
    "c_rank": "rank in the suggest_edits ordering",
    "r_heavy_to": "heavy atoms added",
    "r_heavy_from": "heavy atoms removed",
    "r_dheavy": "net heavy-atom change",
    "r_is_attach": "pure attachment (no atoms removed)",
    "r_is_delete": "deletion",
    "r_n_cuts": "number of cut points",
    "r_drings_arom": "aromatic-ring change",
    "r_drings_aliph": "aliphatic-ring change",
    "r_dhalogen": "halogen-count change",
    "r_dsp3": "sp3-carbon change",
    "r_dtpsa": "fragment TPSA change",
    "r_dtpsa_per_heavy": "fragment TPSA change per heavy atom",
    "r_dhbd": "H-bond donor change",
    "site_aromatic": "edit site is aromatic",
    "site_is_hetero": "edit site is a heteroatom",
    "site_ring_size": "ring size at the edit site",
    "site_fused": "edit site sits on a fused ring",
    "site_num_h": "hydrogens on the edit site",
    "site_sym_equiv": "symmetry-equivalent sites",
    "site_crowd_2b": "heavy atoms within 2 bonds of the site (crowding)",
    "site_degree": "substituents already on the site",
    "site_dist_to_fg": "bond distance to the nearest functional group",
    "site_dist_to_guard": "bond distance to the guard substructure",
    "site_frac_guard": "share of the molecule the guard occupies",
    "site_logp_contrib": "the site's own logP contribution",
    "cand_similarity": "mean Tanimoto similarity to the other candidates",
    "set_n_cands": "candidates available here",
    "set_prob_spread": "spread of success probability across candidates",
    "st_worst_z": "worst constraint violation in the current molecule",
    "st_n_violated": "constraints currently violated",
    "st_gap": "total normalised distance to the box",
    "st_heavy": "heavy atoms in the current molecule",
    "st_free_h": "free attachment sites",
    "st_budget": "edits left in the budget",
    "h_step_index": "edits committed so far",
    "h_gap_improved_last": "did the previous edit help",
    "c_prob_margin": "success-probability margin over the next candidate",
    "c_prob_z": "how many spreads of headroom the move leaves",
    "c_is_unique_best": "is it the only candidate predicted to satisfy",
    "c_room_frac_used": "fraction of the remaining headroom the move consumes",
    "c_worst_prop_delta": "how much it moves the worst-violated property",
    "c_snr_min": "weakest signal-to-noise across the constrained properties",
    "r_dmw": "molecular weight added by the fragment",
    "r_dlogp": "logP added by the fragment",
    "r_dqed": "QED change from the fragment",
    "r_dsa": "synthetic-accessibility change",
    "r_dhba": "H-bond acceptor change",
    "r_dhba_per_heavy": "H-bond acceptors added per heavy atom",
    "r_drotb": "rotatable-bond change",
    "r_dpolar_frac": "change in the polar-atom fraction",
    "site_in_ring": "edit site is in a ring",
    "site_n_hetero_2b": "heteroatoms within 2 bonds of the site",
    "site_on_guard": "the edit site sits on the guard substructure",
    "st_depth_left": "edits left before the depth budget runs out",
    "st_free_h_sites": "free attachment sites on the molecule",
    "h_has_history": "there is a previous edit to learn from",
    "h_motif_repeat": "the fragment repeats one already in the molecule",
    "ctx_shift": "mmpdb context shift of the move",
    "ctx_support": "how many mmpdb pairs support this move",
    "ctx_log_support": "log support count for this move",
    "fg_n_created": "functional groups created",
    "fg_n_destroyed": "functional groups destroyed",
    "c_std_sum": "total spread across the predicted Δs",
    "ctx_fired": "the mmpdb context correction actually applied",
    "ctx_std_r0": "spread of the context-free (radius-0) Δ estimate",
    "h_pred_error_last": "how far the last edit's predicted Δ was from the measurement",
    "h_same_rule_as_prev": "the same rule as the previous edit",
    "r_dcharge": "formal-charge change from the fragment",
    "r_dlogp_per_heavy": "logP added per heavy atom",
    "r_is_swap": "a swap (atoms both removed and added)",
    "site_charge": "Gasteiger partial charge at the edit site",
    "site_n_anchors": "attachment points the rule cuts at",
    "site_ok": "the edit site was located in the molecule",
    "site_on_fg": "the edit site sits on a functional group",
    "st_n_fg": "distinct functional groups in the current molecule",
    "st_n_fg_total": "functional groups in the current molecule (with repeats)",
    "st_n_props": "properties this instance constrains",
    "st_room_hi_min": "room left before the nearest upper bound",
}


# Which way is better, where the column has a direction. Without this the LLM has to
# infer it from the name and gets it backwards — the first draft wrote "#3 breaks fewer
# constraints" off a line that said #3 breaks 2 and #1 breaks 1.
BETTER = {
    "c_pred_gap": -1, "c_gap_reduction": +1, "c_n_sat_post": +1, "c_dcount": +1,
    "c_damage": -1, "c_n_props_helped": +1, "c_n_props_hurt": -1, "c_worst_after": -1,
    "c_cos": +1, "c_cover_frac": +1, "c_overshoot": -1, "c_std_max": -1,
    "c_prob": +1, "c_prob_min": +1, "c_prob_all": +1, "c_rank": -1,
    "site_crowd_2b": -1, "fg_n_destroyed": -1,
}


# A bare multiplier invites the LLM to editorialise: a draft called a x1.04 gate
# "a factor the model weights heavily". The number alone does not say what counts as
# large, so the tier says it instead, and anything at or below AVERAGE_GATE is either
# dropped or explicitly labelled ordinary.
AVERAGE_GATE = 1.05


def tier(g: float) -> str:
    """Only the raised bands get a word.

    Labelling the ordinary band too made the LLM narrate the metadata instead of the
    chemistry — "a distinction treated as about as usual" — so a feature the model did
    not raise now carries no word at all, and the prompt says such a line is a fact to
    use, not something to comment on.
    """
    if g >= 1.60:
        return "decisive here"
    if g >= 1.25:
        return "important here"
    if g >= AVERAGE_GATE:
        return "mildly relevant"
    return ""


def gtag(g: float) -> str:
    t = tier(g)
    return f"[x{g:.2f}, {t}]" if t else f"[x{g:.2f}]"


def favours(name: str, a: float, b: float):
    """-> 'picked' | 'rival' | None, for the contrast line."""
    d = BETTER.get(name.partition("__")[0] if "__" in name else name)
    if d is None or abs(a - b) < 1e-9:
        return None
    return "picked" if (a > b) == (d > 0) else "rival"



# --------------------------------------------------------------------------- #
#  센티널: 값이 아니라 "해당 없음" 을 뜻하는 자리
# --------------------------------------------------------------------------- #
def _sentinel(name: str, value: float):
    """숫자로 읽으면 거짓이 되는 자리를 말로 바꾼다. 아니면 None."""
    stem = name.partition("__")[0]
    if stem == "c_snr_min" and value >= 1000:
        return ("the weakest per-property signal-to-noise is unbounded: every "
                "constrained property's predicted change has zero spread here")
    if stem == "c_room_frac_used" and value >= 90:
        return "no upper bound is close enough for this edit to eat into"
    if stem == "st_room_hi_min" and value >= 90:
        return "no upper bound is anywhere near binding in this molecule"
    if stem in ("site_dist_to_fg",) and value >= 90:
        return "the molecule has no functional group to measure a distance to"
    if stem in ("site_dist_to_guard",) and value >= 90:
        return "the edit site is not on or near the substructure that must be kept"
    if stem in ("ctx_support", "ctx_log_support", "ctx_std_r0", "ctx_shift",
                "ctx_fired"):
        # availability tier 3 — suggest_edits 가 노출하지 않는 값이라 추론 시점에
        # 재현할 수 없다. 프롬프트에 넣으면 그걸 근거로 쓰는 span 이 나온다.
        return None
    return None


def pretty(name: str, value: float, targets: dict) -> str:
    """'st_lo_z__logP' + value -> 'logP is 0.09 below its lower bound 3.66'."""
    stem, _, prop = name.partition("__")
    if prop:
        t = targets.get(prop) or {}
        lo, hi = t.get("min"), t.get("max")
        if stem == "st_val":
            return f"{prop} is {value:g}"
        if stem == "st_lo_z":
            side = "above" if value >= 0 else "below"
            return (f"{prop} sits {abs(value):.2f} scaled units {side} its lower bound"
                    + (f" {lo:g}" if lo is not None else ""))
        if stem == "st_hi_z":
            side = "below" if value >= 0 else "above"
            return (f"{prop} sits {abs(value):.2f} scaled units {side} its upper bound"
                    + (f" {hi:g}" if hi is not None else ""))
        if stem == "st_has_lo":
            return f"{prop} has a lower bound" if value > 0.5 else f"{prop} has no lower bound"
        if stem == "st_has_hi":
            return f"{prop} has an upper bound" if value > 0.5 else f"{prop} has no upper bound"
        if stem == "r_dmean":
            return f"predicted Δ{prop} {value:+.3g}"
        if stem == "r_dstd":
            return f"spread of the Δ{prop} estimate ±{value:.3g}"
        if stem == "r_dfg":
            if abs(value) < 0.5:
                return f"{prop}: unchanged"
            return f"{prop}: {'+' if value > 0 else '-'}{abs(value):.0f}"
    # 센티널을 숫자로 인쇄하면 산문이 그 숫자를 인용한다 — "a signal-to-noise of
    # 1e+06" 은 |Δ|/max(std, 1e-6) 이 std=0 을 만난 것이고, 99 는 "그런 것이 없다" 는
    # 뜻이지 99 라는 값이 아니다. 뜻을 쓴다.
    sent = _sentinel(name, value)
    if sent is not None:
        return sent
    g = GLOSS.get(name)
    if g:
        return f"{g}: {value:g}"
    # never leak a raw column name into the prompt — the LLM copies it verbatim
    # ("maintaining a manageable r_dmw of 127") and the sentence stops being readable.
    return f"{name.replace('_', ' ')}: {value:g}"


# --------------------------------------------------------------------------- #
#  the model side
# --------------------------------------------------------------------------- #
class Scorer:
    def __init__(self, ckpt: str, tensors: str, ctx_dir: str, device: str = "cuda",
                 ctx_extra: str = ""):
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.meta = blob["meta"]
        a = blob["args"]
        if a.get("arch") == "flat":
            raise SystemExit(
                "this checkpoint is the flat-MLP ablation arm: it has no feature gating "
                "and no candidate attention, so there is no A or A_c to write the "
                "reasoning from. Use a --arch gated checkpoint.")
        self.spec = FeatureSpec.load(os.path.join(tensors, "spec.json"))
        with open(os.path.join(ctx_dir, "index.json")) as fh:
            idx = json.load(fh)
        self.ctx = np.load(os.path.join(ctx_dir, "ctx.npy"), mmap_mode="r")
        self.ctx_row = {s: i for i, s in enumerate(idx["smiles"])}
        # which encoder built this table — `ensure_ctx` has to match it, and until the
        # default moved to MolBERT every caller happened to be on morgan
        self.encoder = idx.get("encoder", "morgan")
        self.dev = torch.device(device if torch.cuda.is_available() else "cpu")
        # a --product-ctx checkpoint also scores from each candidate's post-edit
        # molecule. The evidence still comes from A over the interpretable columns
        # only — the model splits the product's attention off into `attn_prod`, whose
        # mass is carried through as `product_attention_mass` so a reader can see how
        # much of the decision the printed features do NOT explain.
        # prod_dims=0 in the args means "the whole context vector", which is what the
        # MolBERT arms record; morgan records 2048 so its descriptor tail is left out.
        # Reading the 0 literally would silently disable the product path and then fail
        # on the state_dict.
        self.prod_dims = ((int(a.get("prod_dims") or 0) or int(self.ctx.shape[1]))
                          if a.get("product_ctx") else 0)
        self.n_prod = int(a.get("n_prod", 128))
        self.model = RuleSelector(self.meta["G"], self.meta["R"], self.ctx.shape[1],
                                  d_model=a["d_model"], d_key=a["d_key"],
                                  n_heads=a["heads"], mlp_hidden=a["mlp_hidden"],
                                  dropout=0.0,
                                  # an ablation checkpoint has no cand_attn weights
                                  candidate_attention=not a.get("no_cand_attn", False),
                                  prod_dim=self.prod_dims, n_prod=self.n_prod,
                                  ).to(self.dev).eval()
        self.model.load_state_dict(blob["model"])
        self.gc, self.gsd, self.rc, self.rsd = self.spec.norm_vectors()
        self._molbert = None                 # built on first use, then cached

        # ── THE SIDECAR. Everything `ensure_ctx` computes used to live in an
        # in-memory dict and die with the process. That is why the 2M-corpus dumps
        # cost a full MolBERT pass every time and left nothing behind: v231318b2cfa1's
        # rounds hit `ctx.npy` on 0.4% of molecules, yet its gates exist, so 99.6% was
        # embedded at dump time and thrown away.
        #
        # `ctx_extra` (or $RS_CTX_EXTRA) names a directory of append-only shards,
        # `part-*.npy` beside `part-*.json`. Each worker writes its own, so sharded
        # runs need no lock, and every shard is mmapped at load, so a 15M-molecule
        # store costs its SMILES index in RAM and nothing else.
        self.extra_dir = ctx_extra or os.environ.get("RS_CTX_EXTRA") or ""
        self.extra_ctx = {}                  # computed this session, not yet flushed
        self.extra_arr, self.extra_row = [], {}
        self.ctx_refused = set()
        self._extra_n = 0
        if self.extra_dir:
            self._extra_load()
        # ── (b) A MISS IS AN ERROR. `_ctx_of` used to answer an unknown molecule with
        # a zero vector, and the gate is CONDITIONED on that vector -- a whole round
        # scored from zeros, silently, and the only symptom is a worse arm. Callers are
        # supposed to run `ensure_ctx_rows` first; if one does not, say so. A molecule
        # MolBERT itself refuses is the one legitimate zero and is tracked separately.
        self.strict_ctx = os.environ.get("RS_CTX_LENIENT", "") not in ("1", "true", "yes")

    # The context table was built over the training trees. A molecule from the real
    # SFT corpus is usually NOT in it, and a zero context vector is not a neutral
    # input — the gate is conditioned on it. So fingerprint the missing ones with the
    # same encoder, standardising the descriptor tail against the training corpus
    # rather than against the handful of molecules in this batch.
    def _extra_load(self) -> None:
        """mmap every shard already in `extra_dir` and index it by SMILES."""
        import glob as _g
        os.makedirs(self.extra_dir, exist_ok=True)
        for j in sorted(_g.glob(os.path.join(self.extra_dir, "part-*.json"))):
            npy = j[:-5] + ".npy"
            if not os.path.exists(npy):
                continue
            k = len(self.extra_arr)
            self.extra_arr.append(np.load(npy, mmap_mode="r"))
            for i, smi in enumerate(json.load(open(j))["smiles"]):
                self.extra_row[smi] = (k, i)
        rp = os.path.join(self.extra_dir, "refused.txt")
        if os.path.exists(rp):
            self.ctx_refused = {x.rstrip("\n") for x in open(rp) if x.strip()}
        self._extra_n = len(self.extra_row)
        logger = __import__("logging").getLogger(__name__)
        logger.info("[ctx] sidecar %s: %s molecules in %d shards, %s refused",
                    self.extra_dir, f"{len(self.extra_row):,}", len(self.extra_arr),
                    f"{len(self.ctx_refused):,}")

    def _extra_flush(self, refused=()) -> int:
        """Append what this session computed to its own shard, then let it mmap.

        Written to a `.part` and renamed, so a reader globbing mid-write never sees a
        half file, and the index json is written AFTER the array so a shard is only
        discoverable once its rows exist.
        """
        if refused:
            self.ctx_refused |= set(refused)
            if self.extra_dir:
                with open(os.path.join(self.extra_dir, "refused.txt"), "a") as fh:
                    for s_ in refused:
                        fh.write(s_ + "\n")
        if not self.extra_dir or not self.extra_ctx:
            return 0
        smis = sorted(self.extra_ctx)
        arr = np.stack([self.extra_ctx[s_] for s_ in smis]).astype(np.float32)
        stem = os.path.join(self.extra_dir,
                            f"part-{os.getpid():06d}-{len(self.extra_arr):04d}")
        # np.save APPENDS `.npy` to a path that lacks it, so `stem + ".npy.part"`
        # lands at `...npy.part.npy` and the rename below finds nothing. A file object
        # is written verbatim.
        tmp = stem + ".npy.tmp"
        with open(tmp, "wb") as fh:
            np.save(fh, arr)
        os.replace(tmp, stem + ".npy")
        with open(stem + ".json", "w") as fh:
            json.dump({"smiles": smis, "dim": int(arr.shape[1])}, fh)
        k = len(self.extra_arr)
        self.extra_arr.append(np.load(stem + ".npy", mmap_mode="r"))
        for i, s_ in enumerate(smis):
            self.extra_row[s_] = (k, i)
        n = len(smis)
        self.extra_ctx = {}          # the rows live in the mmap now
        return n

    def ensure_ctx(self, smiles: list) -> int:
        """`smiles` should include the candidates' post-edit molecules on a
        --product-ctx checkpoint; `ensure_ctx_rows(rows)` does that for you."""
        import importlib
        ce = importlib.import_module("5_rule_selection.context_embed")
        missing = sorted({s for s in smiles if s and s not in self.ctx_row
                          and s not in self.extra_row and s not in self.extra_ctx
                          and s not in self.ctx_refused})
        if not missing:
            return 0
        if self.encoder == "molbert":
            n = self._ensure_molbert(ce, missing)
            self._extra_flush()
            return n
        rng = np.random.default_rng(0)
        pool = list(self.ctx_row)
        ref = [pool[i] for i in rng.choice(len(pool), size=min(20000, len(pool)),
                                           replace=False)]
        nb = 2048
        tail_ref = ce._morgan_chunk(ref)[:, nb:]
        mu = tail_ref.mean(0, keepdims=True)
        sd = tail_ref.std(0, keepdims=True) + 1e-6
        arr = ce._morgan_chunk(missing)
        arr[:, nb:] = (arr[:, nb:] - mu) / sd
        for s_, v in zip(missing, arr):
            self.extra_ctx[s_] = v.astype(np.float32)
        self._extra_flush()
        return len(missing)

    def _ensure_molbert(self, ce, missing: list) -> int:
        """The same thing for a MolBERT table.

        Without this the morgan branch above would hand a 2058-dim vector to a model
        whose `ctx_query` is 768 wide. That was latent while the shipped default was a
        morgan arm; it stopped being latent when the default became `molbert-*`.

        A SMILES the tokenizer refuses keeps NO entry, so `_ctx_of` returns the zero
        vector — which for a candidate product is what training saw (`ctx.npy` holds a
        zero row for every molecule MolBERT rejected). For a STATE molecule a zero
        context is not neutral, since the gate is conditioned on it, so the count of
        those is returned separately in `self.n_molbert_refused`.
        """
        import importlib
        DF = importlib.import_module("5_rule_selection.defaults")
        dtype = torch.bfloat16 if self.dev.type == "cuda" else torch.float32
        # $RS_TOK_WORKERS caps the tokeniser pool. The default asks for 32 processes,
        # which is right for one Scorer on the node and wrong for thirty-two of them:
        # 32 x 32 forks oversubscribe a 288-core box three-fold and every one of them
        # pays MolBERT's featurizer import.
        _tw = int(os.environ.get("RS_TOK_WORKERS") or 0) or min(32, os.cpu_count() or 8)
        ids, _length, valid = ce.tokenize_molbert(
            missing, DF.DEFAULT["molbert_repo"], workers=_tw)
        if self._molbert is None:
            self._molbert = ce._build_molbert(DF.DEFAULT["molbert_ckpt"], dtype,
                                              self.dev)
        (word, ttype, ln, encoder), pos, *_ = self._molbert
        keep = [k for k, v in enumerate(valid) if v]
        self.n_molbert_refused = getattr(self, "n_molbert_refused", 0) \
            + (len(missing) - len(keep))
        # A SMILES the tokenizer refuses gets NO row, and `_ctx_of` has to answer it
        # with the zero vector that `ctx.npy` holds for exactly these -- so name them,
        # or the strict check below cannot tell a refusal from a missing call.
        self._extra_flush(refused=[missing[k] for k, v in enumerate(valid) if not v])
        with torch.no_grad():
            for i in range(0, len(keep), 512):
                sel = keep[i:i + 512]
                chunk = ids[sel]
                width = max(int((chunk != 0).sum(1).max()), 1)
                t = torch.from_numpy(chunk[:, :width].astype(np.int64)).to(self.dev)
                attn = (t != 0)
                h = word(t) + pos(width, self.dev, word.weight.dtype) \
                    + ttype(torch.zeros_like(t))
                h = ln(h)
                ext = (~attn)[:, None, None, :].to(h.dtype) * torch.finfo(h.dtype).min
                h = encoder(h, attention_mask=ext).last_hidden_state
                m = attn.unsqueeze(-1).to(h.dtype)
                emb = ((h * m).sum(1) / m.sum(1).clamp(min=1)).float().cpu().numpy()
                for k, v in zip(sel, emb):
                    self.extra_ctx[missing[k]] = v.astype(np.float32)
        return len(keep)

    def ensure_ctx_rows(self, rows: list) -> int:
        """Every molecule these decisions stand on: the states, plus — when the
        checkpoint reads the post-edit molecule — every candidate product."""
        want = [r.get("state_smiles") for r in rows]
        if self.prod_dims:
            want += [c.get("smiles") for r in rows for c in (r.get("candidates") or [])]
        return self.ensure_ctx(want)

    def _ctx_of(self, smi):
        j = self.ctx_row.get(smi, -1)
        if j >= 0:
            return np.array(self.ctx[j], dtype=np.float32)
        e = self.extra_row.get(smi)
        if e is not None:
            k, i = e
            return np.array(self.extra_arr[k][i], dtype=np.float32)
        v = self.extra_ctx.get(smi)
        if v is not None:
            return v
        if smi in self.ctx_refused or not smi:
            # MolBERT will not tokenise it; `ctx.npy` holds a zero row for these and
            # that is what training saw.
            return np.zeros(self.ctx.shape[1], np.float32)
        if self.strict_ctx:
            raise KeyError(
                f"no context vector for {smi!r}. The gate is CONDITIONED on it, so a "
                f"zero here scores the whole round from nothing and looks only like a "
                f"worse arm. Call ensure_ctx_rows(rows) before run(), or set "
                f"RS_CTX_LENIENT=1 to restore the old silent zero.")
        return np.zeros(self.ctx.shape[1], np.float32)

    def run(self, row: dict) -> dict:
        g, gm, r, rm, y, cm = self.spec.encode(row)
        gn = (g - self.gc) / self.gsd
        rn = (r - self.rc) / self.rsd
        c = self._ctx_of(row.get("state_smiles"))
        pv = None
        if self.prod_dims:
            cands = row.get("candidates") or []
            pv = np.zeros((self.spec.max_candidates, self.prod_dims), np.float32)
            for i, cd in enumerate(cands[:self.spec.max_candidates]):
                pv[i] = self._ctx_of(cd.get("smiles"))[:self.prod_dims]
        t = lambda x, d=torch.float32: torch.as_tensor(x, dtype=d, device=self.dev)[None]
        with torch.no_grad():
            out = self.model(t(gn), t(gm, torch.bool), t(rn), t(rm, torch.bool),
                             t(cm, torch.bool), t(c), None if pv is None else t(pv))
        return {"score": out["score"][0].cpu().numpy(), "q": out["q"][0].cpu().numpy(),
                "A_g": out["attn_global"][0].cpu().numpy(),
                "A_r": out["attn_rule"][0].cpu().numpy(),
                "A_p": out["attn_prod"][0].cpu().numpy(),
                "A_c": out["attn_cand"][0].cpu().numpy(),
                "g_raw": g, "g_mask": gm, "r_raw": r, "r_mask": rm,
                "cmask": cm, "y": y,
                # the pre-MLP representations, for the injection study. `x` is what the
                # shared scoring head reads; `u_hat` is its only candidate-varying part.
                "u_hat": out["u_hat"][0].cpu().numpy(),
                "u_g": out["u_g"][0].cpu().numpy(),
                "c_proj": out["c_proj"][0].cpu().numpy(),
                "x": out["x"][0].cpu().numpy()}


def table4(row: dict, res: dict, spec: FeatureSpec, top_k: int = 8,
           pick: int | None = None) -> list:
    """One row per column, with EVERY candidate's value — a 4-way contrast.

    `evidence`'s `contrast` is pairwise by construction: it diffs the pick against one
    rival and keeps only columns where those two differ. That is the right shape for
    "why this edit over that one" and the wrong shape for "which of four", and the
    difference shows up in the trained student. Across five span versions on the same
    5,000 held-out rounds, every one landed at or above the no-reasoning floor on
    `set_acc` (which credits any of ~1.59 correct options, so a pairwise argument
    suffices) and every one landed BELOW it on `beam_acc` (which needs the exact option
    out of four):

        floor 0.479 / 0.373    v5 0.468 / 0.350    v7 0.494 / 0.360
        v8 0.480 / 0.347       v9 0.474 / 0.344

    Reconstructing a 4-way table from the stored pairwise contrasts does not work: they
    drop equal-valued columns and truncate to 12, so a missing cell is ambiguous between
    "equal" and "truncated", and only a median of 4 columns are present across all three
    pairs (fewer than 3 on 28.6% of rounds). Hence this, computed where `r_raw` is.

    Columns are chosen by the gate's own weight, restricted to those that actually vary
    across candidates — a column identical on all four cannot tell them apart, and
    printing it invites the writer to compare identical numbers.
    """
    import numpy as np
    targets = {t["property"]: t for t in (row.get("targets") or [])}
    n = int(res["cmask"].sum())
    ar = res["A_r"]
    gate_r = np.zeros_like(ar)
    for i in range(n):
        m = res["r_mask"][i]
        if m.sum() == 0:
            continue
        a = ar[i][m] / max(ar[i][m].sum(), 1e-12)
        gate_r[i][m] = a * m.sum()
    defined = res["r_mask"][:n].all(axis=0)
    out = []
    for j in np.flatnonzero(defined):
        vals = [float(res["r_raw"][i, j]) for i in range(n)]
        if max(vals) - min(vals) < 1e-9:
            continue                      # identical on every candidate: tells nothing
        w = float(gate_r[:n, j].max())
        sd = float(spec.scale.get(f"r::{spec.rule_names[j]}", 1.0)) or 1.0
        # SPREAD, not range. Ranking on (max-min) put a column at the top whenever ONE
        # candidate was an outlier, so the first table came back full of rows reading
        # "#1 +1 | #2 unchanged | #3 unchanged | #4 unchanged" -- true, and useless for
        # telling four options apart. The number of DISTINCT values leads, so a column
        # that separates three or four candidates outranks one that isolates a single
        # candidate, and the gate-weighted dispersion breaks ties.
        k = len({round(v / sd, 4) for v in vals})
        out.append((k, float(np.std(vals)) / sd * w, w, j, vals))
    out.sort(key=lambda t: (t[0], t[1]), reverse=True)
    out = [(a, w, j, vals) for a, _b, w, j, vals in out]
    rows = []
    for _, w, j, vals in out[:top_k]:
        name = spec.rule_names[j]
        hi = int(np.argmax(vals))
        rows.append({"feature": name,
                     "values": [pretty(name, v, targets) for v in vals],
                     "larger": hi,
                     "favours": (favours(name, vals[hi], min(vals)) and hi),
                     "gate": w, "tier": tier(w)})
    return rows


def evidence(row: dict, res: dict, spec: FeatureSpec, top_state: int = 6,
             top_rule: int = 4, top_contrast: int = 7,
             rival_thresh: float = 1.15, pick_override: int | None = None,
             rival_override: int | None = None,
             gate_floor: float | None = None) -> dict:
    """A and A_c -> the few facts the decision was actually made on.

    The gate is reported as a MULTIPLIER against a uniform gate, not as a share of
    attention. With 81 state and 150 rule features a softmax share is ~1% for
    everything and reads as noise; the multiplier is the number the model actually
    applies to the feature token (`alpha * n_tokens`), and it is what separates a
    feature the decision leaned on (measured up to 1.7x) from one it ignored (0.6x).

    A_c is used for the contrast ONLY when it is actually selective. Measured on this
    checkpoint the candidate self-attention sits at 0.24 against a uniform 0.25 — i.e.
    flat — so dressing it up as "the model compared against #k" would be inventing a
    finding. When the row is within `rival_thresh` of uniform the rival falls back to
    the nearest candidate by score (the decision's real margin) and the record says
    which rule was used.
    """
    targets = {t["property"]: t for t in (row.get("targets") or [])}
    n = int(res["cmask"].sum())
    ag, ar = res["A_g"], res["A_r"]

    gm = res["g_mask"]
    ag_n = ag[gm] / max(ag[gm].sum(), 1e-12)
    gate_g = np.zeros_like(ag)
    gate_g[gm] = ag_n * gm.sum()                       # 1.0 == a uniform gate
    order = np.argsort(-gate_g)[:top_state]
    state = [{"feature": spec.global_names[j],
              "text": pretty(spec.global_names[j], float(res["g_raw"][j]), targets),
              "gate": float(gate_g[j]), "tier": tier(float(gate_g[j]))}
             for j in order if gm[j]
             and gate_g[j] >= (AVERAGE_GATE if gate_floor is None else gate_floor)]

    gate_r = np.zeros_like(ar)
    cands = []
    for i in range(n):
        m = res["r_mask"][i]
        if m.sum() == 0:
            continue
        a = ar[i][m] / max(ar[i][m].sum(), 1e-12)
        gate_r[i][m] = a * m.sum()
        o = [j for j in np.argsort(-gate_r[i])[:top_rule] if m[j]]
        up = [j for j in o
              if gate_r[i, j] >= (AVERAGE_GATE if gate_floor is None else gate_floor)]
        o = up if len(up) >= 2 else o[:2]     # never leave a candidate with nothing
        cands.append({
            "index": i, "rule": row["candidates"][i].get("rule"),
            "q": float(res["q"][i]), "score": float(res["score"][i]),
            "label": float(row["candidates"][i].get("sat_ratio", -1)),
            "features": [{"feature": spec.rule_names[j],
                          "text": pretty(spec.rule_names[j], float(res["r_raw"][i, j]), targets),
                          "gate": float(gate_r[i, j]), "tier": tier(float(gate_r[i, j]))}
                         for j in o]})

    argmax = int(np.argmax(res["score"][:n]))
    # On the real SFT corpus the chain already committed a rule, and the job is to
    # explain THAT one — not the one this model would have taken. The attention is
    # still read from the committed row, so the evidence is "what the model looked at
    # for this candidate", and `argmax` is kept so the two can be compared.
    pick = argmax if pick_override is None else int(pick_override)
    A_c = res["A_c"][:n, :n]
    row_c = A_c[pick].copy()
    uniform = 1.0 / max(n, 1)
    # `rival_override` builds the contrast against a CALLER-CHOSEN candidate. Needed
    # because A_c's own choice is index-skewed: measured on 40,000 train rounds it elects
    # #1 on 34.3% of rounds against 20.9-23.4% for the others, since #1 is the tool's
    # top-ranked candidate and so the nearest neighbour in A_c's gated space. A corpus
    # built on it teaches "the option rejected is #1", and the trained student duly
    # under-picks #1. Balancing the rival index needs the contrast for an arbitrary pair,
    # not just A_c's.
    selective = float(np.abs(row_c - uniform).max()) > (rival_thresh - 1.0) * uniform
    if rival_override is not None and 0 <= int(rival_override) < n \
            and int(rival_override) != pick:
        rival = int(rival_override); how = "caller override"
    elif selective:
        r_ = row_c.copy(); r_[pick] = -1
        rival = int(np.argmax(r_)); how = "candidate-attention"
    else:
        d = np.abs(res["score"][:n] - res["score"][pick]); d[pick] = np.inf
        rival = int(np.argmin(d)); how = "nearest score (A_c is flat)"

    # what actually separates the two, weighted by how much the gate cares
    diffs = []
    mm = res["r_mask"][pick] & res["r_mask"][rival]
    for j in np.flatnonzero(mm):
        a, b = float(res["r_raw"][pick, j]), float(res["r_raw"][rival, j])
        if abs(a - b) < 1e-9:
            continue
        w = 0.5 * (gate_r[pick, j] + gate_r[rival, j])
        sd = float(spec.scale.get(f"r::{spec.rule_names[j]}", 1.0)) or 1.0
        diffs.append((abs(a - b) / sd * w, j, a, b))
    diffs.sort(reverse=True)
    # `favours` only exists for the columns with a known direction. For the rest the
    # ORDER is still stated ("larger"), because without it the LLM guesses and gets it
    # backwards — a draft read "#3 offers a lower molecular weight of 67.067 versus
    # 45.061 for #1" straight off two candidate blocks.
    contrast = [{"feature": spec.rule_names[j],
                 "picked": pretty(spec.rule_names[j], a, targets),
                 "rival": pretty(spec.rule_names[j], b, targets),
                 "favours": favours(spec.rule_names[j], a, b),
                 "larger": ("picked" if a > b else "rival"),
                 "gate": float(0.5 * (gate_r[pick, j] + gate_r[rival, j])),
                 "tier": tier(float(0.5 * (gate_r[pick, j] + gate_r[rival, j])))}
                for _, j, a, b in diffs[:top_contrast]]

    return {"state_attention_mass": float(ag.sum()),
            "product_attention_mass": float(res.get("A_p", np.zeros(0)).sum()),
            "rule_attention_mass": float(ar[:n].sum()),
            "gate_spread_state": float(gate_g[gm].max()) if gm.any() else 1.0,
            "gate_spread_rule": float(gate_r[pick].max()),
            "state": state, "candidates": cands, "pick": pick, "argmax": argmax,
            "pick_is_argmax": bool(pick == argmax),
            "rival": rival, "rival_rule": how,
            "self_weight": float(A_c[pick, pick]), "uniform": uniform,
            "A_c_row": [float(x) for x in row_c], "contrast": contrast}


# --------------------------------------------------------------------------- #
#  prompts
# --------------------------------------------------------------------------- #
SYSTEM = (
    "You write the assistant's turn that goes between a suggest_edits result and the "
    "edit_fragment call it leads to — the reason a medicinal chemist would give for "
    "committing this edit rather than the alternatives offered at the same step. "
    "3 sentences, 65 words or fewer IN TOTAL. First person, flowing prose, no lists, "
    "no headings, no markdown, no backticks. "
    "Use ONLY the numbers you are given; never invent a value or a chemical name.\n"
    "ORDER MATTERS. Reason first, conclude last: sentence 1 says what the molecule "
    "still needs, sentence 2 weighs the options against each other, and only the FINAL "
    "sentence names the candidate you take. Do not open with the answer — no "
    "'I select #2 because...'. The reader should be able to follow the argument to the "
    "choice rather than be handed the choice and then a justification.\n"
    "Two rules about comparisons. (1) Compare two candidates ONLY on a line that "
    "already puts both of their values side by side; never build a comparison out of "
    "numbers taken from two different candidate blocks — the direction is easy to get "
    "backwards. (2) A change in one property is not evidence about a different one: "
    "H-bond donors are not H-bond acceptors, TPSA is not logD. (3) The weighting "
    "numbers tell you WHICH facts to write about. They are not themselves facts about "
    "the molecule: never make the weighting the subject of a sentence, and never write "
    "that something was weighted, up-weighted, or treated as usual. Say the chemistry. "
    "A feature marked 'decisive here' is the one you may say the choice turns on. "
    "(4) Never mention the model, its attention, or its weighting. Write as the chemist "
    "making the call: say WHY the chemistry favours this edit. You may quote a success "
    "probability as a plain number, but not as something the model believes."
)


ORDINAL = ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th"]


def _action(row: dict) -> str:
    a = row.get("committed_args") or {}
    frm, to = (a.get("from_smiles") or ""), (a.get("to_smiles") or "")
    if frm.strip() in ("[*:1]", "*"):
        return f"attach {to}"
    if to.strip() in ("[*:1][H]", "[*:1]"):
        return f"remove {frm}"
    if frm and to:
        return f"replace {frm} with {to}"
    return "the committed edit"


def prompt_model(row: dict, ev: dict, n_state: int = 5, n_contrast: int = 9,
                 show_q: bool = False, closer: str | None = None) -> str:
    """A and A_c as three blocks, in the layout that measured best.

    The earlier layout printed each candidate's own top-gated features. Measured on
    `molbert-cand-norank-prod` that block was dead weight: the gate raises the SAME
    columns for every candidate (the committed candidate's top-4 overlapped its
    siblings' by 90%), and every one of its 272 lines cleared the "decisive here"
    threshold — a highlighter over the whole page. Worse, the closer then told the
    model to argue from those lines, which are by construction the ones that do NOT
    separate the options. So the per-candidate block is gone and the contrast block,
    which is where the separation actually lives, is longer.

    Settings were chosen by generating spans and scoring them against the candidate
    value table (`scripts/span_check.py`), 12 rounds x 8 samples per cell:

        layout                              false/span   usable spans
        per-candidate blocks (old)              --            --
        no blocks, 6 contrast lines            0.12          58%
        no blocks, 9 contrast lines            0.09          62%   <- this
        no blocks, 12 contrast lines           0.19          52%
        thresholds recalibrated per block      0.17          48%
        gate shown as a RANK, not a multiple   0.17          43%
        q printed next to each option          0.14          55%

    Two of those are worth stating rather than burying. **Recalibrating the tiers made
    it worse**, even though the raw distribution says the state block is 83% "decisive":
    once the per-candidate block is gone the original 1.25/1.60 thresholds land on a
    contrast block whose gates sit near 1.0, and that is where the differentiation is
    needed. **Ranking is the worst of the lot** — labelling a line "1st most weighted"
    puts ranking language in the model's mouth, and it comes back out as a superlative
    about the candidates that is not true of their values.

    ``usable span`` = discriminating AND free of any claim the value table contradicts.
    """
    pick, rival = ev["pick"], ev["rival"]
    L = [f"MOLECULE: {row['state_smiles']}",
         f"({len(ev['candidates'])} candidate edits were offered at this step)", ""]

    if n_state:
        L.append("WHAT THIS ROUND IS ABOUT")
        L.append("  (the state the model weighted most heavily. These facts hold "
                 "whichever option you take, so they say what the round needs — not "
                 "which option to take.)")
        for s_ in ev["state"][:n_state]:
            L.append(f"  - {s_['text']}   {gtag(s_['gate'])}")
        L.append("")

    L.append("THE OPTIONS")
    for c in ev["candidates"]:
        q = f"   q={c['q']:.2f}" if show_q else ""
        star = ("   <-- the one to take" if c["index"] == pick else
                "   <-- the one it was weighed against" if c["index"] == rival else "")
        L.append(f"  #{c['index'] + 1} {c['rule']}{q}{star}")
    L.append("")

    L.append(f"WHAT SEPARATES #{pick + 1} FROM #{rival + 1}")
    L.append(f"  (the model's candidate attention weighed #{pick + 1} against "
             f"#{rival + 1} more than against the others. These are the columns where "
             f"the two actually differ — a reason to prefer one over the other has to "
             f"come from here.)")
    for d in ev["contrast"][:n_contrast]:
        who = (f"favours #{pick + 1 if d['favours'] == 'picked' else rival + 1}"
               if d["favours"] else
               f"larger for #{pick + 1 if d['larger'] == 'picked' else rival + 1}")
        L.append(f"  - #{pick + 1}: {d['picked']}   |   #{rival + 1}: {d['rival']}"
                 f"   -> {who}   {gtag(d['gate'])}")
    if not ev["contrast"]:
        L.append("  - nothing: the two are identical on every column the model reads.")
    L.append("")

    if not ev.get("pick_is_argmax", True):
        L.append(f"NOTE: the search committed #{pick + 1}, but the model's own highest "
                 f"q is #{ev['argmax'] + 1}. Justify #{pick + 1} on the facts above and "
                 f"concede that #{ev['argmax'] + 1} scores higher rather than "
                 f"pretending #{pick + 1} leads on everything.")
        L.append("")

    L.append(f"The edit being made is: {_action(row)}.")
    L.append(closer if closer is not None else _simple_closer())
    return "\n".join(L)


DELIBERATION_CLOSER = """Write that turn: 3 sentences, 65 words or fewer IN TOTAL. First person, flowing prose.
No bullet points, no markdown, no backticks, no tool-call syntax, and do not think out
loud or correct yourself mid-sentence — write the finished reasoning only.

This turn is the thinking, not the decision. Use one sentence for each of the three
steps: what the molecule still needs, which of the numbers above decide it here, and how
the options differ on exactly those numbers. The tool call that follows records which
edit is made, so do not announce or endorse one. Do not say any option is better, worse,
preferred, superior, optimal or the right one, and do not write "I will", "I select",
"I choose", "I prefer" or "so I". State what each option does differently; the comparison
is the content. Say nothing you cannot check against the numbers above. Largest, highest,
lowest, most, maximum, minimal and only are claims about every value listed — use one
only if you have read them all and it holds. If the numbers leave the two close, say the
difference is small rather than making it sound decisive."""


def _simple_closer() -> str:
    """The closer this span is written under, chosen by measurement.

    The span used to end by announcing the edit ("I select #3 to attach the urea"). That
    is backwards for training data: the tool call after it already records the choice, so
    an announcing sentence spends a third of the budget on something the trajectory
    states anyway — and it teaches the order "pick, then justify". Forbidding it turns
    out to be worth far more than the sentence it costs, because the words go into the
    comparison instead. Over 12 rounds, scored against the candidate value table
    (`scripts/span_check.py`):

        closer                                   false/span  usable  commits  shape
        name the edit in the last sentence          0.12       60%     97%     82%
        + do not announce it                        0.28       74%      4%     90%
        + three named steps                         0.18       75%      2%     92%
        + ban evaluative words too                  0.12       80%      1%     82%
        + say so when the call is close             0.07       78%      0%     81%
        + one sentence per step (this)              0.10       78%      0%    100%

    Two of these are worth naming. **Banning evaluative language does the work of the
    anti-superlative rule as well** — "the best", "superior signal-to-noise" and "the
    preferred option" were most of the false superlatives, so removing the vocabulary
    removed the claims (10 superlative errors -> 3). And **the ban raises discrimination
    rather than lowering it** (72% -> 87%): with no sentence left to announce a winner,
    the budget goes to saying how the options differ, which is the thing that was
    missing.

    `usable` = discriminating AND free of any claim the value table contradicts.
    """
    return DELIBERATION_CLOSER


def prompt_model_a(row: dict, ev: dict) -> str:
    """A alone: which facts to show, and nothing about who to argue against.

    The paired arm to `prompt_model`. Same state block and same per-candidate feature
    lines, but no candidate self-attention: no named rival, no contrast block. What is
    left is "here are the facts this decision leaned on, for every candidate" — so the
    difference between the two texts prices A_c specifically rather than the whole
    evidence pipeline.
    """
    pick = ev["pick"]
    L = [f"MOLECULE: {row['state_smiles']}",
         f"(search depth {row.get('depth')}, {len(ev['candidates'])} candidate edits)", ""]
    L.append("WHAT THE MODEL LEANED ON IN THIS STATE")
    L.append("  (x1.00 = the model treated this feature as it usually does. A line with "
             "no word after the number was NOT raised: the fact is still true and you "
             "may use it, but it is not what the decision turned on.)")
    if ev["state"]:
        for s_ in ev["state"]:
            L.append(f"  - {s_['text']}   {gtag(s_['gate'])}")
    else:
        L.append("  - the gate did not single out any state feature here: the state did "
                 "not drive this decision, the candidates did.")
    L.append("")
    L.append("CANDIDATES  (q = the predicted probability that a search through this edit "
             "reaches the property box)")
    for c in ev["candidates"]:
        star = "   <-- the one to take" if c["index"] == pick else ""
        L.append(f"  #{c['index']+1} {c['rule']}   q={c['q']:.2f}{star}")
        for f in c["features"]:
            L.append(f"        {f['text']}   {gtag(f['gate'])}")
    L.append("")
    L.append(f"WRITE THAT TURN for taking #{pick+1}: 3 sentences, 65 words or fewer IN "
             f"TOTAL, first person, and name the candidate only in the LAST sentence. "
             f"Build it out of the facts marked 'decisive here' or 'important here'. "
             f"You are NOT told which single rival is the close one — weigh the field as "
             f"a whole.")
    return "\n".join(L)


def prompt_naive_ablation(row: dict) -> str:
    """The shipped NAIVE_REASONING=1 prompt, verbatim.

    Imported from `4_sftdata_gen/prompts.py` rather than re-implemented: the naive
    ablation hands the model the RAW tool JSON and nothing else — no candidate table,
    no landing-safety block, no spread verdict, no rank note, no fragment names — and
    a reconstruction that quietly added any of those would not be the baseline it
    claims to be.
    """
    import importlib
    pr = importlib.import_module("prompts")
    return pr.NAIVE_EDIT_REASONING_PROMPT.format(
        user_prompt=row.get("user_prompt") or "",
        mol_smiles=row["state_smiles"],
        raw_props=json.dumps(row.get("state_props") or {}, ensure_ascii=False),
        raw_candidates=json.dumps(row.get("raw_candidates") or [], ensure_ascii=False),
        from_smiles=(row.get("committed_args") or {}).get("from_smiles"),
        to_smiles=(row.get("committed_args") or {}).get("to_smiles"),
        anchors=(row.get("committed_args") or {}).get("anchors"))


def prompt_naive(row: dict, material: str, block: str, spread: str, pick: int) -> str:
    L = [f"MOLECULE: {row['state_smiles']}",
         f"(search depth {row.get('depth')})", "", "CANDIDATES", material]
    if block:
        L += ["", block]
    if spread:
        L += ["", "SPREAD", spread]
    L += ["", f"The candidate to select is #{pick+1}.",
          "WRITE THE REASON for taking it over the others, using only the numbers above."]
    return "\n".join(L)


# --------------------------------------------------------------------------- #
#  vLLM
# --------------------------------------------------------------------------- #
_RR = {"i": 0}


def generate(prompt: str, temperature: float = 0.3, max_tokens: int = 400) -> str:
    from openai import OpenAI
    url = VLLM_URLS[_RR["i"] % len(VLLM_URLS)]
    _RR["i"] += 1
    client = OpenAI(base_url=url, api_key="EMPTY")
    r = client.chat.completions.create(
        model=MODEL, temperature=temperature, max_tokens=max_tokens,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": prompt}],
        extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    txt = (r.choices[0].message.content or "").strip()
    return re.sub(r"<think>.*?</think>", "", txt, flags=re.S).strip()


_NUM = re.compile(r"-?\d+\.\d+|-?\d+")


def unsupported_numbers(text: str, prompt: str) -> list:
    """Numbers in the text that are not in the material it was given.

    Same failure the naive pipeline guards against; applied to both modes here so the
    two can be compared on hallucination rate rather than on impressions.
    """
    have = set()
    for x in _NUM.findall(prompt):
        try:
            have.add(round(float(x), 6))
        except ValueError:
            pass
    bad = []
    for x in _NUM.findall(text):
        try:
            v = round(float(x), 6)
        except ValueError:
            continue
        if abs(v) <= 10 and float(v).is_integer():
            continue                      # candidate numbers, counts, sentence digits
        if v not in have:
            bad.append(x)
    return bad


def load_states(args) -> list:
    """Eligible decision states, from either input.

    ``--states``     the exhaustive-search dump: every candidate carries a
                     ``sat_ratio``, so a pick can be scored. Filtered on
                     ``--min-spread`` because a flat set has nothing to explain.
    ``--toolchain``  the corpus stage 4 actually renders: one beam-search path, so
                     each state has the full candidate list but exactly ONE committed
                     rule and no labels. Nothing to filter on — every state where the
                     committed rule was found among the candidates is eligible.
    """
    rows, seen = [], 0
    rng = np.random.default_rng(max(args.sample_seed, 0))

    def offer(d):
        nonlocal seen
        seen += 1
        if seen <= args.offset:
            return False
        if args.sample_seed < 0:
            rows.append(d)
            return len(rows) >= args.limit
        if len(rows) < args.limit:
            rows.append(d)
        else:
            j = int(rng.integers(0, seen - args.offset))
            if j < args.limit:
                rows[j] = d
        return False

    if args.sftdata:
        import glob as _glob
        import importlib
        ts = importlib.import_module("5_rule_selection.toolchain_states")
        files = sorted(f for f in _glob.glob(os.path.join(args.sftdata, "*.jsonl"))
                       if not os.path.basename(f).startswith("_"))
        if not files:
            raise SystemExit(f"no jsonl under {args.sftdata}")
        stop = False
        for rec in ts.iter_records(files, limit=args.max_records):
            for d in ts.rows_from_sft_record(rec):
                if d.get("committed", -1) < 0 or \
                        len(d["candidates"]) < args.min_candidates or \
                        not d.get("corpus_text"):
                    continue
                if offer(d):
                    stop = True
                    break
            if stop:
                break
        print(f"# {len(rows)} states of {seen:,} eligible from {len(files)} rendered "
              f"file(s){', random sample' if args.sample_seed >= 0 else ''}")
        return rows

    if args.toolchain:
        import importlib
        ts = importlib.import_module("5_rule_selection.toolchain_states")
        files = ts.expand(args.toolchain)
        if not files:
            raise SystemExit(f"no tool-chain files under {args.toolchain}")
        stop = False
        for rec in ts.iter_records(files, limit=args.max_records):
            for d in ts.rows_from_record(rec):
                if d.get("committed", -1) < 0 or len(d["candidates"]) < args.min_candidates:
                    continue
                if offer(d):
                    stop = True
                    break
            if stop:
                break
        print(f"# {len(rows)} states of {seen:,} eligible from {len(files)} tool-chain "
              f"file(s){', random sample' if args.sample_seed >= 0 else ''}")
        return rows

    with open(args.states) as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            if d.get("label_spread", 0) < args.min_spread or \
                    len(d["candidates"]) < args.min_candidates:
                continue
            if offer(d):
                break
    print(f"# {len(rows)} states of {seen:,} eligible (label spread >= "
          f"{args.min_spread}{', random sample' if args.sample_seed >= 0 else ''})")
    return rows


def naive_material(row: dict, naive_mod, pick: int):
    """The shipped candidate tables, from whichever input we have.

    The tool chain records the suggest_edits response verbatim, so the per-property
    Δ **std** survives and `spread_verdict` has something to say. The tree dump kept
    only the mean, so there the spread block comes out empty — a real handicap of the
    tree-dump comparison, not of the naive method.
    """
    targets = {t["property"]: (t.get("min"), t.get("max"))
               for t in (row.get("targets") or [])}
    cands = []
    for c in row["candidates"]:
        if c.get("from_smiles") is not None:
            frm, to = c["from_smiles"], c["to_smiles"]
        else:
            frm, _, to = (c.get("rule") or "").partition("->")
            frm, to = frm.strip(), to.strip()
        d = {}
        for p_, v in (c.get("pred_delta") or {}).items():
            if isinstance(v, dict):
                d[p_] = {"avg": v.get("mean"), "std": v.get("std") or 0.0}
            else:
                d[p_] = {"avg": v, "std": 0.0}
        gap = c.get("predicted_gap")
        if gap is None:
            gap = c["features"].get("c_pred_gap", 0)
        cands.append({"from_smiles": frm, "to_smiles": to,
                      "anchors": c.get("anchors"),
                      "predicted_gap": round(float(gap), 4), "delta": d})
    material = naive_mod.format_candidate_options(
        cands, row.get("state_props"), targets,
        from_smiles=cands[pick]["from_smiles"], to_smiles=cands[pick]["to_smiles"])
    try:
        spread = naive_mod.spread_verdict(cands, row.get("state_props"), targets,
                                          cands[pick]["from_smiles"],
                                          cands[pick]["to_smiles"])
    except Exception:  # noqa: BLE001
        spread = ""
    return cands, material, spread or ""


def main() -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--states", help="states.jsonl from the exhaustive-search dump")
    src.add_argument("--toolchain",
                     help="directory or glob of toolchains_*chunk_*.jsonl — the REAL "
                          "SFT corpus (one beam path, one committed rule per node)")
    src.add_argument("--sftdata",
                     help="directory of RENDERED sft records (e.g. a "
                          "NAIVE_REASONING=1 corpus). The naive arm is then the "
                          "reasoning span the pipeline actually shipped, taken "
                          "verbatim — nothing is regenerated for it.")
    ap.add_argument("--max-records", type=int, default=0,
                    help="--toolchain: stop after this many chains (0 = all)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tensors", required=True)
    ap.add_argument("--ctx-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["naive", "model", "both", "all"], default="both",
                    help="'all' adds a third arm written from A ALONE (no candidate "
                         "self-attention), so the pair prices A_c on its own.")
    ap.add_argument("--pick", choices=["model", "committed"], default="model",
                    help="which candidate the text has to justify. 'committed' is the "
                         "only meaningful choice on --toolchain input: the chain already "
                         "chose, and the text explains that choice.")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--min-candidates", type=int, default=3)
    ap.add_argument("--sample-seed", type=int, default=-1,
                    help=">=0: reservoir-sample the eligible states instead of taking "
                         "the first --limit, which come from a handful of instances")
    ap.add_argument("--min-spread", type=float, default=0.15,
                    help="--states only: skip sets whose labels barely differ")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--naive-style", choices=["ablation", "tables"], default="ablation",
                    help="what the naive arm is when the state carries no shipped span. "
                         "'ablation' = the NAIVE_REASONING=1 prompt (raw tool JSON only, "
                         "which is what the corpus was written with); 'tables' = the "
                         "candidate tables from fragment_names.")
    ap.add_argument("--dump-evidence", default="",
                    help="also write the raw A / A_c matrices per state here (npz-ish "
                         "JSON), for the visualisation")
    args = ap.parse_args()
    if args.toolchain and args.pick == "model":
        print("# note: --toolchain input has a committed rule; --pick committed is "
              "usually what you want", file=sys.stderr)

    sc = Scorer(args.ckpt, args.tensors, args.ctx_dir, args.device)
    naive_mod = None
    if args.mode in ("naive", "both", "all"):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "4_sftdata_gen"))
        import importlib
        naive_mod = importlib.import_module("fragment_names")

    rows = load_states(args)
    n_new = sc.ensure_ctx_rows(rows)
    if n_new:
        print(f"# {n_new} molecule(s) outside the training context table — "
              f"fingerprinted on the fly")

    out = open(args.out, "w")
    eva = open(args.dump_evidence, "w") if args.dump_evidence else None
    for k, row in enumerate(rows):
        res = sc.run(row)
        forced = row.get("committed") if args.pick == "committed" else None
        ev = evidence(row, res, sc.spec, pick_override=forced)
        rec = {"id": row.get("id"), "source": row.get("source"), "depth": row.get("depth"),
               "state_smiles": row["state_smiles"],
               "labels": [c.get("sat_ratio") for c in row["candidates"]],
               "committed": row.get("committed"),
               "q": [c["q"] for c in ev["candidates"]], "model_pick": ev["pick"],
               "model_argmax": ev["argmax"], "evidence": ev}
        if args.mode == "all":
            p = prompt_model_a(row, ev)
            rec["model_a_prompt"] = p
            rec["model_a_text"] = generate(p, args.temperature)
            rec["model_a_unsupported"] = unsupported_numbers(rec["model_a_text"], p)
        if args.mode in ("model", "both", "all"):
            p = prompt_model(row, ev)
            rec["model_prompt"] = p
            rec["model_text"] = generate(p, args.temperature)
            rec["model_unsupported"] = unsupported_numbers(rec["model_text"], p)
        if args.mode in ("naive", "both", "all") and row.get("corpus_text"):
            # the shipped span, verbatim — no prompt, no generation, no drift
            rec["naive_pick"] = int(row["committed"])
            rec["naive_source"] = "corpus"
            rec["naive_text"] = row["corpus_text"]
            rec["naive_prompt"] = prompt_naive_ablation(row)
            rec["naive_unsupported"] = unsupported_numbers(
                rec["naive_text"], rec["naive_prompt"])
        elif args.mode in ("naive", "both", "all") and args.naive_style == "ablation":
            p = prompt_naive_ablation(row)
            rec["naive_pick"] = int(row["committed"])
            rec["naive_source"] = "regenerated-ablation"
            rec["naive_prompt"] = p
            rec["naive_text"] = generate(p, args.temperature, max_tokens=2048)
            rec["naive_unsupported"] = unsupported_numbers(rec["naive_text"], p)
        elif args.mode in ("naive", "both", "all"):
            if args.pick == "committed":
                npick = int(row["committed"])
            else:
                # the shipped ordering picks rank 1: the lowest predicted gap
                npick = int(np.argmin([
                    c.get("predicted_gap") if c.get("predicted_gap") is not None
                    else c["features"].get("c_pred_gap", 0) for c in row["candidates"]]))
            _c, material, spread = naive_material(row, naive_mod, npick)
            p = prompt_naive(row, material, "", spread, npick)
            rec["naive_pick"] = npick
            rec["naive_source"] = "tables"
            rec["naive_prompt"] = p
            rec["naive_text"] = generate(p, args.temperature)
            rec["naive_unsupported"] = unsupported_numbers(rec["naive_text"], p)
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out.flush()
        if eva is not None:
            eva.write(json.dumps({
                "id": row.get("id"), "depth": row.get("depth"),
                "state_smiles": row["state_smiles"],
                "targets": row.get("targets"), "state_props": row.get("state_props"),
                "committed": row.get("committed"),
                "guard_smarts": row.get("guard_smarts"),
                "candidates": [{"rule": c.get("rule"), "to_smiles": c.get("to_smiles"),
                                "from_smiles": c.get("from_smiles"),
                                "anchors": c.get("anchors"),
                                "predicted_gap": c.get("predicted_gap"),
                                "pred_delta": c.get("pred_delta")}
                               for c in row["candidates"]],
                "product_smiles": row.get("product_smiles"),
                "global_names": sc.spec.global_names, "rule_names": sc.spec.rule_names,
                "A_g": [float(x) for x in res["A_g"]],
                "A_r": [[float(x) for x in r_] for r_ in res["A_r"]],
                "A_c": [[float(x) for x in r_] for r_ in res["A_c"]],
                "g_raw": [float(x) for x in res["g_raw"]],
                "g_mask": [bool(x) for x in res["g_mask"]],
                "r_raw": [[float(x) for x in r_] for r_ in res["r_raw"]],
                "r_mask": [[bool(x) for x in r_] for r_ in res["r_mask"]],
                "q": [float(x) for x in res["q"]],
                "score": [float(x) for x in res["score"]],
                "cmask": [bool(x) for x in res["cmask"]]}, ensure_ascii=False) + "\n")
            eva.flush()
        print(f"  [{k+1}/{len(rows)}] {row.get('id')} depth {row.get('depth')} "
              f"pick#{ev['pick']+1} argmax#{ev['argmax']+1} "
              f"q={ev['candidates'][ev['pick']]['q']:.2f}", flush=True)
    out.close()
    if eva is not None:
        eva.close()
    print(f"# wrote {args.out}")


if __name__ == "__main__":
    main()
