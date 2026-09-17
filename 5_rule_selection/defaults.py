"""The shipped rule-selection setting, and the fallback the token limit forces.

**Default: `molbert-cand-norank-prod`.** From the 24-arm grid (2 contexts x 3
architectures x 2 losses x post-edit-molecule on/off, 15 epochs, scorer-matched at
~0.59 M), paired test-regret deltas:

    context   morgan -> MolBERT   mean -0.0196 over 12 pairs   SIGNIFICANT
    product   none   -> prod      mean -0.0088 over 12 pairs   SIGNIFICANT on MolBERT
    arch      gating -> flat      mean +0.0123 over  8 pairs   SIGNIFICANT
    arch      cand   -> nocand    mean +0.0006 over  8 pairs   noise
    loss      rank   -> norank    mean +0.0045 over 12 pairs   worse in 11/12

**The product axis is a MolBERT effect, not a leak.** Split by context, `none -> prod`
is -0.0172 regret on MolBERT (better in 6/6 pairs) and -0.0004 on morgan (WORSE in 2 of
6, i.e. nothing). That is the opposite of what the descriptor-leak worry predicts — a
folded 2048-bit fingerprint of the product adds nothing over the rule features, while
MolBERT's 768-d representation of it does. The morgan product input is the one with the
descriptors sliced off, and it is also the one that does not help.

**Caveat on this default, stated rather than buried.** `molbert-cand-norank-prod` is the
best VALUE loss of all 24 (0.1360). It is not the best decision: `molbert-cand-rank-prod`
is better on regret (0.2018 vs 0.2081), top1 (0.6727 vs 0.6652), skill (0.479 vs 0.462)
and pair accuracy (0.7208 vs 0.7137), and the value gap the other way is 0.0005 — a tie.
The 12-arm grid dropped the ranking term because that axis was noise there (+0.0024);
over 24 arms it is not (+0.0045, worse in 11/12 on regret and 12/12 on top1, and 6/6
among the prod arms). If what you care about is which rule a search picks rather than how
calibrated q_hat is, use `ALT_RANK` below — it is a one-line swap.

**The fallback.** MolBERT was pretrained on GuacaMol at 128 tokens and its featurizer
REFUSES anything longer — 3,901 of the 7.1 M molecules in the 50K dataset (0.055%),
and it grows with depth as the molecules do. A zero context vector is not a neutral
input when the gate is conditioned on it, so those molecules are routed to the
`morgan-cand-norank` checkpoint instead, which has no length limit. That is a
different model, not a different input: the two contexts have different widths (768 vs
2058) and each checkpoint's `ctx_query` / `ctx_value` are shaped for its own.

`DefaultScorer` does the routing per molecule and reports how often it fired.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

from .features import FeatureSpec
from .model import RuleSelector

ROOT = os.environ.get("RS_ROOT", "data/analysis/rule_selection/runs")
MOLBERT_HOME = os.environ.get("MOLBERT_HOME", "data/models/molbert")

def _cfg(loss: str, product: str = "prod", grid: str = "ablation_24") -> dict:
    """The primary/fallback pair for one (loss, product) corner of the grid.

    Both arms must share the corner: the fallback is a different MODEL, not a different
    input, and one that read the post-edit molecule while the other did not would make
    the routing a silent change of conditioning.
    """
    tag = f"-{product}" if product == "prod" else ""
    return {
        "primary": {
            "arm": f"molbert-cand-{loss}{tag}",
            "ckpt": f"{ROOT}/{grid}/molbert-cand-{loss}{tag}/best.pt",
            "tensors": f"{ROOT}/d5-50k/tensors_molbert",
            "ctx_dir": f"{ROOT}/d5-50k/ctx_molbert",
            "encoder": "molbert",
        },
        "fallback": {
            "arm": f"morgan-cand-{loss}{tag}",
            "ckpt": f"{ROOT}/{grid}/morgan-cand-{loss}{tag}/best.pt",
            "tensors": f"{ROOT}/d5-50k/tensors_morgan",
            "ctx_dir": f"{ROOT}/d5-50k/ctx_morgan",
            "encoder": "morgan",
        },
        "molbert_ckpt": f"{MOLBERT_HOME}/molbert_100epochs/checkpoints/last.ckpt",
        "molbert_repo": f"{MOLBERT_HOME}/MolBERT",
    }


DEFAULT = _cfg("norank")
# the decision-metric leader; see the caveat in the module docstring
ALT_RANK = _cfg("rank")
# the pre-product setting, for reproducing anything measured before the fourth axis
LEGACY_12 = _cfg("norank", product="none", grid="ablation_12")


class _Arm:
    """One checkpoint plus the encoder its context column was built with.

    The 7.1 M-row context TABLE is not loaded by default. Building the SMILES->row
    dict for it costs ~40 s and ~1.5 GB per arm, and inference on a handful of
    molecules does not need it — they are encoded on the spot. `use_table=True`
    brings it back when you are scoring molecules the dataset already covers.
    """

    def __init__(self, cfg: dict, device: torch.device, use_table: bool = False):
        blob = torch.load(cfg["ckpt"], map_location="cpu", weights_only=False)
        a = blob["args"]
        self.cfg, self.meta, self.args = cfg, blob["meta"], a
        self.spec = FeatureSpec.load(os.path.join(cfg["tensors"], "spec.json"))
        # the context width is in the checkpoint itself; no need to open the table
        self.dim = int(blob["model"]["ctx_query.weight"].shape[1])
        self.ctx = None
        self.row: dict = {}
        if use_table:
            with open(os.path.join(cfg["ctx_dir"], "index.json")) as fh:
                idx = json.load(fh)
            assert int(idx["dim"]) == self.dim
            self.ctx = np.load(os.path.join(cfg["ctx_dir"], "ctx.npy"), mmap_mode="r")
            self.row = {s: i for i, s in enumerate(idx["smiles"])}
        # a --product-ctx checkpoint also scores from each candidate's post-edit
        # molecule, embedded by THIS arm's encoder. prod_dims=0 in the args means "the
        # whole context vector", which is what MolBERT uses; morgan records 2048 there
        # so the descriptor tail is left out.
        self.prod_dims = (int(a.get("prod_dims") or 0) or self.dim) \
            if a.get("product_ctx") else 0
        self.n_prod = int(a.get("n_prod", 128))
        self.model = RuleSelector(
            self.meta["G"], self.meta["R"], self.dim,
            d_model=a["d_model"], d_key=a["d_key"], n_heads=a["heads"],
            mlp_hidden=a["mlp_hidden"], dropout=0.0,
            candidate_attention=not a.get("no_cand_attn", False),
            prod_dim=self.prod_dims, n_prod=self.n_prod).to(device).eval()
        self.model.load_state_dict(blob["model"])
        self.norm = self.spec.norm_vectors()
        self.extra: dict = {}
        self.dev = device

    def context(self, smi: str):
        j = self.row.get(smi, -1)
        if j >= 0 and self.ctx is not None:
            return np.array(self.ctx[j], dtype=np.float32)
        return self.extra.get(smi)

    def products(self, row: dict) -> np.ndarray | None:
        """[N, prod_dims] of this row's post-edit molecules, in THIS arm's encoder.

        A product the encoder cannot represent gets a zero row — which is not a
        fallback, it is what training saw: `ctx.npy` holds a zero vector for every
        molecule MolBERT's tokenizer refused, and `cctx.npy` points straight at it.
        """
        if not self.prod_dims:
            return None
        out = np.zeros((self.spec.max_candidates, self.prod_dims), np.float32)
        for i, c in enumerate((row.get("candidates") or [])[:self.spec.max_candidates]):
            v = self.context(c.get("smiles"))
            if v is not None:
                out[i] = np.asarray(v, np.float32)[:self.prod_dims]
        return out

    def run(self, row: dict, ctx: np.ndarray, prod: np.ndarray | None = None) -> dict:
        g, gm, r, rm, y, cm = self.spec.encode(row)
        gc, gsd, rc, rsd = self.norm
        if self.prod_dims and prod is None:
            prod = self.products(row)
        t = lambda x, d=torch.float32: torch.as_tensor(x, dtype=d, device=self.dev)[None]
        with torch.no_grad():
            out = self.model(t((g - gc) / gsd), t(gm, torch.bool),
                             t((r - rc) / rsd), t(rm, torch.bool),
                             t(cm, torch.bool), t(ctx),
                             None if prod is None else t(prod))
        return {"score": out["score"][0].cpu().numpy(),
                "q": out["q"][0].cpu().numpy(),
                "A_g": out["attn_global"][0].cpu().numpy(),
                "A_r": out["attn_rule"][0].cpu().numpy(),
                "A_p": out["attn_prod"][0].cpu().numpy(),
                "A_c": out["attn_cand"][0].cpu().numpy(),
                "g_raw": g, "g_mask": gm, "r_raw": r, "r_mask": rm,
                "cmask": cm, "y": y}


class DefaultScorer:
    """MolBERT primary, Morgan for whatever MolBERT's tokenizer refuses."""

    def __init__(self, cfg: dict | None = None, device: str = "cuda",
                 use_table: bool = False):
        cfg = cfg or DEFAULT
        self.cfg = cfg
        self.dev = torch.device(device if torch.cuda.is_available() else "cpu")
        self.primary = _Arm(cfg["primary"], self.dev, use_table)
        self.fallback = _Arm(cfg["fallback"], self.dev, use_table)
        self.n_fallback = 0
        self.n_total = 0
        self.n_prod_missing = 0
        self.n_prod_total = 0
        self.n_rows_no_products = 0
        self._routed: dict = {}
        self._molbert = None            # built once, not per prepare() call
        # Molecules the tokenizer refused. They are PERMANENTLY absent from `extra`, so
        # without this the self-healing path in `run` re-tokenises the same rejects for
        # every state that names them — 64 forked workers per call, and when a row's
        # misses are all rejects the batch comes back empty. Remembering the verdict
        # makes the retry a set lookup.
        self._refused: set = set()

    @property
    def reads_products(self) -> bool:
        return bool(self.primary.prod_dims or self.fallback.prod_dims)

    # -- context preparation ------------------------------------------- #
    def prepare_rows(self, rows: list) -> dict:
        """`prepare`, plus each row's post-edit molecules when the arms read them.

        A product is embedded into the arm that will SCORE ITS STATE, not into the arm
        its own SMILES would route to: the two contexts have different widths and the
        checkpoint is shaped for one of them. Routing stays a per-state decision, which
        is what `report()` counts.

        States and products are tokenised in ONE call. That is not tidiness — measured
        over 200 states, `tokenize_molbert` costs ~3 s of process-pool dispatch almost
        regardless of how many molecules it is handed (135 molecules: 3 s; 301: 3 s), so
        two calls made the product input look 2x more expensive than it is. One call
        over 996 molecules puts the marginal cost of the products where it belongs.
        """
        states = [r.get("state_smiles") for r in rows]
        if not self.reads_products:
            return self.prepare(states)
        prods = [c.get("smiles") for r in rows for c in (r.get("candidates") or [])]
        # A row builder that does not emit candidates[*].smiles would leave every
        # product at the zero vector — an input training used ONLY for a molecule the
        # tokenizer refused (0% of the 9.87 M products). That is silently wrong rather
        # than loudly broken, so count it and say so.
        blind = sum(1 for r in rows
                    if (r.get("candidates") or [])
                    and not any(c.get("smiles") for c in r["candidates"]))
        self.n_rows_no_products += blind
        if blind:
            print(f"# [warn] {blind:,}/{len(rows):,} rows carry no candidates[*].smiles, "
                  f"so {self.primary.cfg['arm']} scores them with a ZERO post-edit "
                  f"molecule. Whatever produced these rows needs to fill it in "
                  f"(toolchain_states.product_of does it).", file=sys.stderr, flush=True)
        route = self.prepare(states, also=prods)
        # each product goes to the arm that will score ITS state; a product the
        # tokenizer refused simply has no entry, and `_Arm.products` gives it the zero
        # row training used
        need_mo = set()
        for r in rows:
            which = route.get(r.get("state_smiles"), "morgan")
            arm = self.primary if which == "molbert" else self.fallback
            if which == "molbert" or not arm.prod_dims:
                continue
            for c in (r.get("candidates") or []):
                smi = c.get("smiles")
                if smi and arm.context(smi) is None:
                    need_mo.add(smi)
        if need_mo:
            self._embed_morgan(sorted(need_mo), self.fallback)
        return route

    def prepare(self, smiles: list, also: list | None = None) -> dict:
        """Embed every molecule with the arm that can represent it.

        -> {smiles: "molbert" | "morgan"} for the molecules in `smiles`. `also` is
        tokenised and embedded in the same pass but is NOT routed or reported — it is
        for the post-edit molecules, whose arm is decided by their state, not by
        themselves. Molecules already present in a context table are reused from it;
        the rest are encoded here.
        """
        import importlib
        ce = importlib.import_module("5_rule_selection.context_embed")
        smiles = [s for s in dict.fromkeys(smiles) if s]
        seen = set(smiles)
        extra = [s for s in dict.fromkeys(also or []) if s and s not in seen]
        allm = smiles + extra

        ids, length, valid = ce.tokenize_molbert(
            allm, self.cfg["molbert_repo"], workers=min(32, os.cpu_count() or 8))
        route = {s: ("molbert" if v else "morgan")
                 for s, v in zip(smiles, valid[:len(smiles)])}
        if extra:
            ok = int(np.asarray(valid[len(smiles):]).sum())
            self._refused.update(s_ for s_, v in zip(extra, valid[len(smiles):]) if not v)
            self.n_prod_total += len(extra)
            self.n_prod_missing += len(extra) - ok

        # one MolBERT forward for the states and the products together
        keep = [k for k, s in enumerate(allm)
                if valid[k] and (k >= len(smiles) or route[s] == "molbert")
                and self.primary.context(s) is None]
        if keep:
            self._embed_molbert([allm[k] for k in keep], ids[keep], self.primary)
        need_mo = [s for s in smiles
                   if route[s] == "morgan" and self.fallback.context(s) is None]
        if need_mo:
            self._embed_morgan(need_mo, self.fallback)
        self._routed.update(route)
        return route

    def _into_molbert(self, smiles: list, arm: "_Arm") -> None:
        """Tokenise then embed `smiles` into `arm`, for the self-healing path in `run`.

        Prefer `prepare_rows`, which does this in the same pass as the states.
        """
        import importlib
        ce = importlib.import_module("5_rule_selection.context_embed")
        ids, _length, valid = ce.tokenize_molbert(
            smiles, self.cfg["molbert_repo"], workers=min(32, os.cpu_count() or 8))
        keep = [k for k, v in enumerate(valid) if v]
        self._refused.update(smiles[k] for k, v in enumerate(valid) if not v)
        self.n_prod_total += len(smiles)
        self.n_prod_missing += len(smiles) - len(keep)
        if keep:
            self._embed_molbert([smiles[k] for k in keep], ids[keep], arm)

    def _build_molbert_once(self):
        if self._molbert is None:
            import importlib
            ce = importlib.import_module("5_rule_selection.context_embed")
            # bf16 is a GPU optimisation; on CPU several of these kernels fall back to
            # slow paths or are unimplemented, and 80 molecules do not need it
            dtype = torch.bfloat16 if self.dev.type == "cuda" else torch.float32
            self._molbert = ce._build_molbert(self.cfg["molbert_ckpt"], dtype, self.dev)
        return self._molbert

    def _embed_molbert(self, smiles: list, ids: np.ndarray, arm: "_Arm") -> None:
        # ~1.4 s to rebuild the encoder, so it is cached: with products this is called
        # once per prepare_rows AND once per self-healing run(), not once overall
        (word, ttype, ln, encoder), pos, hidden, *_ = self._build_molbert_once()
        with torch.no_grad():
            for i in range(0, len(smiles), 512):
                chunk = ids[i:i + 512]
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
                for s, v in zip(smiles[i:i + 512], emb):
                    arm.extra[s] = v.astype(np.float32)

    def _embed_morgan(self, smiles: list, arm: "_Arm") -> None:
        """Morgan, with the descriptor tail standardised against the TRAINING corpus.

        Not against this handful of molecules: the tail is 10 of 2058 columns and the
        checkpoint was fitted with the corpus statistics, so a local standardisation
        would shift them.
        """
        import importlib
        ce = importlib.import_module("5_rule_selection.context_embed")
        nb = 2048
        mu, sd = self._morgan_tail_stats(ce, nb)
        arr = ce._morgan_chunk(smiles)
        arr[:, nb:] = (arr[:, nb:] - mu) / sd
        for s, v in zip(smiles, arr):
            arm.extra[s] = v.astype(np.float32)

    def _morgan_tail_stats(self, ce, nb: int):
        """(mu, sd) of the 10 descriptor columns over the TRAINING corpus, cached.

        `encode_morgan` standardises that tail against whatever batch it is given, so
        a handful of molecules encoded now would land on a different scale from the
        ones the checkpoint was fitted on. The statistics are computed once from a
        20k sample of the corpus and written next to the context table.
        """
        cache = os.path.join(self.cfg["fallback"]["ctx_dir"], "morgan_tail_stats.json")
        if os.path.exists(cache):
            with open(cache) as fh:
                d = json.load(fh)
            return (np.array(d["mu"], np.float32)[None],
                    np.array(d["sd"], np.float32)[None])
        with open(os.path.join(self.cfg["fallback"]["ctx_dir"], "index.json")) as fh:
            smiles = json.load(fh)["smiles"]
        rng = np.random.default_rng(0)
        ref = [smiles[i] for i in rng.choice(len(smiles),
                                             size=min(20000, len(smiles)),
                                             replace=False)]
        tail = ce._morgan_chunk(ref)[:, nb:]
        mu, sd = tail.mean(0), tail.std(0) + 1e-6
        try:
            with open(cache, "w") as fh:
                json.dump({"mu": mu.tolist(), "sd": sd.tolist(), "n": len(ref)}, fh)
        except OSError:
            pass
        return mu[None].astype(np.float32), sd[None].astype(np.float32)

    # -- scoring --------------------------------------------------------- #
    def run(self, row: dict) -> dict:
        smi = row.get("state_smiles")
        which = self._routed.get(smi)
        if which is None:
            which = self.prepare([smi]).get(smi, "morgan")
        arm = self.primary if which == "molbert" else self.fallback
        ctx = arm.context(smi)
        if ctx is None:                     # encoder failed outright: neutral input
            ctx = np.zeros(arm.dim, np.float32)
        prod = None
        if arm.prod_dims:
            # self-healing: a caller that used prepare() instead of prepare_rows()
            # still gets the right answer, one state at a time
            miss = [c.get("smiles") for c in (row.get("candidates") or [])
                    if c.get("smiles") and arm.context(c.get("smiles")) is None
                    and c.get("smiles") not in self._refused]
            if miss:
                if which == "molbert":
                    self._into_molbert(sorted(set(miss)), arm)
                else:
                    self._embed_morgan(sorted(set(miss)), arm)
            prod = arm.products(row)
        self.n_total += 1
        self.n_fallback += int(which != "molbert")
        out = arm.run(row, ctx, prod)
        out["arm"] = arm.cfg["arm"]
        out["encoder"] = which
        out["spec"] = arm.spec
        return out

    def report(self) -> str:
        if not self.n_total:
            return "no states scored"
        out = (f"{self.n_total:,} states | {self.n_total - self.n_fallback:,} on "
               f"{self.primary.cfg['arm']} | {self.n_fallback:,} fell back to "
               f"{self.fallback.cfg['arm']} ({self.n_fallback/self.n_total:.2%})")
        if self.reads_products:
            if self.n_prod_total:
                out += (f" | {self.n_prod_total:,} post-edit molecules embedded, "
                        f"{self.n_prod_missing:,} refused by the tokenizer -> zero row "
                        f"({self.n_prod_missing/self.n_prod_total:.2%})")
            else:
                out += " | [warn] NO post-edit molecule was embedded"
            if self.n_rows_no_products:
                out += (f" | [warn] {self.n_rows_no_products:,} rows had no "
                        f"candidates[*].smiles and were scored on a zero product")
        return out
