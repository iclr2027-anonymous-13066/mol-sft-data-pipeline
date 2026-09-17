"""Train the rule-selection model. DDP over 8 GPUs, wandb logging.

    torchrun --nproc_per_node=8 -m 5_rule_selection.train --data <prepare_dataset out> ...

The model is small (a few M parameters over ~800 tokens per state), so this is
data-parallel for throughput, not for memory: every rank holds the whole model and a
shard of the states. The tensors are memmapped, so all ranks share one page cache.

Metrics are grouped so the run answers three separate questions:

  loss/*      is it optimising
  value/*     does q_hat mean what sat_ratio means (MSE / MAE / calibration)
  rank/*      inside one decision set, is the ORDER right (pairwise accuracy,
              Spearman, NDCG@1)
  decide/*    the number that matters for a search: pick argmax(q_hat) and report
              the regret against the best candidate. Logged next to two baselines —
              a random pick and the shipped `c_pred_gap` ranking — because
              branch_features already showed the shipped score is near-chance at the
              root, and an absolute number without those two says nothing.
  attn/*      where the gate is spending itself (global vs rule mass, entropies,
              off-diagonal mass in the candidate attention). With --product-ctx,
              `attn/prod_mass` is the share that went to the post-edit molecule's
              tokens rather than to the interpretable columns — uniform is
              N*n_prod/(G+N*(R+n_prod)); the run is only using the product if it
              climbs above that.

Everything in rank/* and decide/* is also logged per depth and per source, since the
sibling signal is known to strengthen with depth.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from .features import FeatureSpec
from .model import FlatScorer, RuleSelector, ranking_loss, value_loss

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:  # tqdm is not a hard dependency of this repo
    def _tqdm(*a, **k):  # noqa: D103
        class _Null:
            def update(self, *_a, **_k): pass
            def set_postfix_str(self, *_a, **_k): pass
            def close(self): pass
        return _Null()


# --------------------------------------------------------------------------- #
#  data
# --------------------------------------------------------------------------- #
class StateDataset(Dataset):
    """One decision state per item.

    `prod_dims > 0` additionally loads each candidate's POST-EDIT molecule from the same
    context matrix, via `cctx.npy`. It is sliced to the first `prod_dims` columns, which
    for morgan (2048) drops the 10 RDKit descriptors: those are the product's exact
    post-edit property values, and the label is a function of exactly those, so leaving
    them in hands the model the answer in a known column rather than the structure.
    MolBERT has no descriptor tail, so its `prod_dims` is the full width.

    The cost is I/O, not compute: an item goes from one context row to 1 + N of them
    (morgan, N=4: 8.2 KB -> 41 KB), read at random out of a 55 GB memmap. Raise
    --workers before blaming the GPU.
    """

    def __init__(self, root: str, split: int, ctx: np.ndarray, frac: float = 1.0,
                 seed: int = 0, prod_dims: int = 0):
        self.root = root
        m = lambda n: np.load(os.path.join(root, n), mmap_mode="r")
        self.g, self.gmask = m("g.npy"), m("gmask.npy")
        self.r, self.rmask = m("r.npy"), m("rmask.npy")
        self.y, self.cmask = m("y.npy"), m("cmask.npy")
        self.ctx_idx = np.asarray(m("ctx.npy"))
        self.prod_dims = int(prod_dims)
        self.cctx_idx = np.asarray(m("cctx.npy")) if self.prod_dims else None
        self.depth = np.asarray(m("depth.npy"))
        self.source = np.asarray(m("source.npy"))
        self.idx = np.flatnonzero(np.asarray(m("split.npy")) == split)
        if frac < 1.0:
            # subsample by INSTANCE, not by state: the question a learning curve has to
            # answer is "would labelling more instances help", and states from one tree
            # are not independent samples.
            g = np.asarray(m("group.npy"))[self.idx]
            keep = np.unique(g)
            rng = np.random.default_rng(seed)
            keep = rng.permutation(keep)[:max(1, int(len(keep) * frac))]
            self.idx = self.idx[np.isin(g, keep)]
        self.ctx = ctx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = int(self.idx[i])
        c = self.ctx_idx[j]
        item = {
            "g": torch.from_numpy(np.array(self.g[j], dtype=np.float32)),
            "gmask": torch.from_numpy(np.array(self.gmask[j])),
            "r": torch.from_numpy(np.array(self.r[j], dtype=np.float32)),
            "rmask": torch.from_numpy(np.array(self.rmask[j])),
            "y": torch.from_numpy(np.array(self.y[j], dtype=np.float32)),
            "cmask": torch.from_numpy(np.array(self.cmask[j])),
            "ctx": torch.from_numpy(np.array(self.ctx[c]) if c >= 0
                                    else np.zeros(self.ctx.shape[1], np.float32)),
            "depth": int(self.depth[j]), "source": int(self.source[j]),
        }
        if self.prod_dims:
            rows = self.cctx_idx[j]
            pd = self.prod_dims
            p = np.zeros((len(rows), pd), np.float32)
            for k, rj in enumerate(rows):
                if rj >= 0:                      # -1 = padded slot, or no embedding
                    p[k] = self.ctx[rj][:pd]
            item["p"] = torch.from_numpy(p)
        return item


def collate(items):
    keys = ["g", "gmask", "r", "rmask", "y", "cmask", "ctx"]
    if "p" in items[0]:
        keys.append("p")
    out = {k: torch.stack([it[k] for it in items]) for k in keys}
    out["depth"] = torch.tensor([it["depth"] for it in items])
    out["source"] = torch.tensor([it["source"] for it in items])
    return out


# --------------------------------------------------------------------------- #
#  metrics
# --------------------------------------------------------------------------- #
def _ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(x))
    return order.astype(np.float64)


def set_metrics(score: np.ndarray, y: np.ndarray, cmask: np.ndarray,
                base: np.ndarray | None = None):
    """Per decision set. Returns a dict of summed numerators and counts.

    Superseded by `set_rows` in the evaluation loop — kept as the readable reference
    the vectorised version is checked against (`scripts/check_set_rows.py`).
    """
    acc = {}
    add = lambda k, v, n=1: acc.__setitem__(k, (acc.get(k, (0.0, 0))[0] + v,
                                                acc.get(k, (0.0, 0))[1] + n))
    for b in range(len(y)):
        m = cmask[b] & (y[b] >= 0)
        if m.sum() < 2:
            continue
        yy, ss = y[b][m], score[b][m]
        best = float(yy.max())
        # decision: pick argmax and pay the regret
        add("decide/top1", float(yy[int(np.argmax(ss))] >= best - 1e-9))
        add("decide/regret", best - float(yy[int(np.argmax(ss))]))
        add("decide/regret_random", best - float(yy.mean()))
        add("decide/best", best)
        if base is not None:
            bb = base[b][m]
            add("decide/top1_baseline", float(yy[int(np.argmax(bb))] >= best - 1e-9))
            add("decide/regret_baseline", best - float(yy[int(np.argmax(bb))]))
        # order
        gap = yy[:, None] - yy[None, :]
        ds = ss[:, None] - ss[None, :]
        pos = gap > 1e-9
        if pos.any():
            add("rank/pair_acc", float((ds[pos] > 0).mean()))
            w = gap[pos]
            add("rank/pair_acc_w", float(((ds[pos] > 0) * w).sum() / w.sum()))
            if len(np.unique(yy)) > 1 and len(np.unique(ss)) > 1:
                ry, rs = _ranks(yy), _ranks(ss)
                num = ((ry - ry.mean()) * (rs - rs.mean())).sum()
                den = math.sqrt(((ry - ry.mean()) ** 2).sum() * ((rs - rs.mean()) ** 2).sum())
                if den > 0:
                    add("rank/spearman", float(num / den))
            # NDCG@1 with the label as gain
            add("rank/ndcg1", float(yy[int(np.argmax(ss))] / best) if best > 0 else 1.0)
    return acc


def set_rows(score: np.ndarray, y: np.ndarray, cmask: np.ndarray,
             base: np.ndarray | None = None) -> dict:
    """Per-DECISION-SET metric values, vectorised over the batch.

    Returns ``{name: array of length B}`` with NaN where the set does not contribute
    (fewer than two live candidates, no ordered pair, a degenerate Spearman). The
    caller sums the non-NaN entries and counts them, which is the same
    (numerator, count) pair `set_metrics` accumulated one set at a time.

    Why this exists: `set_metrics` was a python loop over every set, and evaluation ran
    it three times over the same states (once pooled, once per depth, once per source).
    Measured on the 120,757-state val split that was 18.3 s of a 19.3 s evaluation —
    the forward pass was 0.29 s. Every set here holds at most `N` = 4 candidates, so
    the whole batch fits in [B,N] and [B,N,N] arrays and the loop buys nothing.

    Exactness: every metric here reproduces `set_metrics` to float32 rounding
    (verified over 61,440 real decision sets) EXCEPT `rank/spearman`, which is
    deliberately different — see `_avg_ranks`.
    """
    B = y.shape[0]
    m = cmask & (y >= 0)
    n = m.sum(1)
    live = n >= 2
    nan = np.full(B, np.nan)
    out = {}
    if not live.any():
        return out

    NEG = -np.inf
    ym = np.where(m, y, NEG)
    sm = np.where(m, score, NEG)
    best = ym.max(1)
    pick = sm.argmax(1)
    ypick = y[np.arange(B), pick]

    def put(name, vals, where=live):
        col = nan.copy()
        col[where] = vals[where] if isinstance(vals, np.ndarray) else vals
        out[name] = col

    put("decide/top1", (ypick >= best - 1e-9).astype(np.float64))
    put("decide/regret", best - ypick)
    ymean = np.where(m, y, 0.0).sum(1) / np.maximum(n, 1)
    put("decide/regret_random", best - ymean)
    put("decide/best", best)
    if base is not None:
        bpick = np.where(m, base, NEG).argmax(1)
        ybpick = y[np.arange(B), bpick]
        put("decide/top1_baseline", (ybpick >= best - 1e-9).astype(np.float64))
        put("decide/regret_baseline", best - ybpick)

    # ---- pairwise, [B,N,N] ------------------------------------------------
    pair = m[:, :, None] & m[:, None, :]
    gap = y[:, :, None] - y[:, None, :]
    ds = score[:, :, None] - score[:, None, :]
    pos = pair & (gap > 1e-9)
    npos = pos.sum((1, 2))
    has_pair = live & (npos > 0)
    win = (ds > 0) & pos
    with np.errstate(invalid="ignore", divide="ignore"):
        acc_p = win.sum((1, 2)) / np.maximum(npos, 1)
        wsum = np.where(pos, gap, 0.0).sum((1, 2))
        acc_w = np.where(win, gap, 0.0).sum((1, 2)) / np.where(wsum > 0, wsum, 1.0)
    put("rank/pair_acc", acc_p, has_pair)
    put("rank/pair_acc_w", acc_w, has_pair)

    # ---- NDCG@1: gain of the picked candidate over the best ---------------
    with np.errstate(invalid="ignore", divide="ignore"):
        ndcg = np.where(best > 0, ypick / np.where(best > 0, best, 1.0), 1.0)
    put("rank/ndcg1", ndcg, has_pair)

    # ---- Spearman over the live candidates of each set --------------------
    yr = _avg_ranks(y, m)
    sr = _avg_ranks(score, m)
    cnt = n[:, None]
    ymu = np.where(m, yr, 0.0).sum(1, keepdims=True) / np.maximum(cnt, 1)
    smu = np.where(m, sr, 0.0).sum(1, keepdims=True) / np.maximum(cnt, 1)
    dy = np.where(m, yr - ymu, 0.0)
    dsr = np.where(m, sr - smu, 0.0)
    num = (dy * dsr).sum(1)
    den = np.sqrt((dy ** 2).sum(1) * (dsr ** 2).sum(1))
    uniq_y = _row_nunique(y, m) > 1
    uniq_s = _row_nunique(score, m) > 1
    ok = has_pair & uniq_y & uniq_s & (den > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = num / np.where(den > 0, den, 1.0)
    put("rank/spearman", rho, ok)
    return out


def _avg_ranks(x: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Tie-averaged ranks of the masked entries of each row: (#less) + (#equal-1)/2.

    This is a DELIBERATE change from the `_ranks` the per-set loop used. That one was
    `argsort(argsort(x))` — ordinal ranks whose tie-break is whatever numpy's sort
    happened to do, and numpy's default quicksort is not stable even on four elements:
    for `[0.667, 0.333, 0.0, 0.0]` it ranks the two tied zeros 1 and 0, i.e. backwards.
    So the old `rank/spearman` depended on numpy's sort internals, and ties are common
    here (sibling sets routinely hold several candidates at sat_ratio 0). Tie-averaged
    ranks are the textbook definition and are well defined; the other metrics are
    unaffected, but `rank/spearman` is NOT comparable to values logged before this.

    No sort is needed at N <= 4: the pairwise comparison matrix gives it directly.
    """
    live = m[:, None, :]
    less = ((x[:, None, :] < x[:, :, None]) & live).sum(2)
    eq = ((x[:, None, :] == x[:, :, None]) & live).sum(2)
    return (less + (eq - 1) / 2.0).astype(np.float64)


def _row_nunique(x: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Number of distinct values among the masked entries of each row."""
    eq = (x[:, None, :] == x[:, :, None]) & m[:, None, :] & m[:, :, None]
    # a value is counted once, by its first occurrence in the row
    first = eq & (np.arange(x.shape[1])[None, None, :] < np.arange(x.shape[1])[None, :, None])
    return (m & ~first.any(2)).sum(1)


def rows_to_acc(rows: dict, sel: np.ndarray | None = None) -> dict:
    """{name: per-set array} -> {name: (sum, count)} over `sel` (default all)."""
    acc = {}
    for k, v in rows.items():
        col = v if sel is None else v[sel]
        good = ~np.isnan(col)
        c = int(good.sum())
        if c:
            acc[k] = (float(col[good].sum()), c)
    return acc


def merge(dst: dict, src: dict) -> dict:
    for k, (v, n) in src.items():
        a, b = dst.get(k, (0.0, 0))
        dst[k] = (a + v, b + n)
    return dst


def reduce_dict(acc: dict, device) -> dict:
    """Sum (value, count) across ranks and return the means."""
    if not acc:
        return {}
    keys = sorted(acc)
    t = torch.tensor([[acc[k][0], acc[k][1]] for k in keys], dtype=torch.float64,
                     device=device)
    if dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return {k: (t[i, 0] / t[i, 1]).item() if t[i, 1] > 0 else float("nan")
            for i, k in enumerate(keys)}


def calibration(q: np.ndarray, y: np.ndarray, bins: int = 10):
    """|mean q_hat - mean y| per bin, weighted -> a single ECE-style number."""
    if len(y) == 0:
        return {}
    idx = np.clip((q * bins).astype(int), 0, bins - 1)
    tot, ece = 0, 0.0
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        ece += m.sum() * abs(q[m].mean() - y[m].mean())
        tot += m.sum()
    return {"value/ece": (ece / max(tot, 1), 1)}


# --------------------------------------------------------------------------- #
#  train
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="prepare_dataset.py output dir")
    ap.add_argument("--ctx-dir", default="", help="context_embed output (ctx.npy)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=256, help="per rank")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-frac", type=float, default=0.03,
                    help="linear ramp over this fraction of total steps, applied under "
                         "either schedule. 0 = start at --lr immediately.")
    ap.add_argument("--lr-schedule", choices=["constant", "cosine"], default="constant",
                    help="constant holds --lr for the whole run (after any warmup); "
                         "cosine decays it to 0 over --epochs. Constant is the default "
                         "because the cosine is defined over the EPOCH BUDGET, so a run "
                         "that early-stops never finishes its anneal — and two runs that "
                         "stop at different steps then did not see the same schedule, "
                         "which is not a difference you want inside an ablation.")
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--d-key", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--mlp-hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lambda-value", type=float, default=1.0)
    ap.add_argument("--lambda-rank", type=float, default=1.0,
                    help="0 turns the pairwise ranking loss OFF for TRAINING; it is "
                         "still computed on val/test as a metric, so the arms stay "
                         "comparable on the same number")
    ap.add_argument("--arch", choices=["gated", "flat"], default="gated",
                    help="gated = context-conditioned feature gating (+ candidate "
                         "self-attention unless --no-cand-attn). flat = concatenate "
                         "[molecule embedding ; global features ; this rule's features] "
                         "and push it through one shared MLP — no gating, no candidate "
                         "interaction. --no-cand-attn is meaningless for flat.")
    ap.add_argument("--target-params", type=int, default=600_000,
                    help="parameter budget every architecture is sized to. It matters "
                         "because the context width feeds straight into the parameter "
                         "count on BOTH architectures — the gated model projects the "
                         "context twice (ctx_query, ctx_value), so at ctx 768 it is "
                         "0.42 M against 0.67 M at ctx 2058, and the flat model takes "
                         "the context as most of its input row. Left unmatched, a "
                         "morgan-vs-molbert comparison would be a capacity comparison.")
    ap.add_argument("--match-params", action="store_true",
                    help="gated only: solve --mlp-hidden for --target-params instead of "
                         "using it as given, so every arm lands at the same size. The "
                         "product-embedding modules are EXCLUDED from the budget (as is "
                         "the flat arm's width solve), so a --product-ctx arm has the "
                         "same scorer geometry as its baseline and the comparison is "
                         "about the input, not the width. Total parameters therefore go "
                         "up; both numbers are printed and logged.")
    ap.add_argument("--flat-hidden", type=int, default=0,
                    help="flat only: fix the width instead of solving for it")
    ap.add_argument("--product-ctx", action="store_true",
                    help="feed each candidate's POST-EDIT molecule embedding, read from "
                         "cctx.npy (build it with scripts/add_product_ctx.py if the "
                         "tensors directory predates it). Projected to --n-prod scalars "
                         "by a 2-layer MLP, then: on the gated architectures each "
                         "scalar becomes its own feature token beside the R "
                         "interpretable ones, so the product is gated, pooled and "
                         "compared exactly like a feature; on flat it is concatenated "
                         "onto the per-candidate half of the row. The product columns "
                         "are reported as attn/prod_mass and excluded from A. NOTE this "
                         "is a large information addition, not a richer context: "
                         "sat_ratio is computed from the product's true properties, so "
                         "any product representation sits closer to the label than the "
                         "mmpdb-PREDICTED deltas the r_* columns carry. Read it as a "
                         "'with product structure' condition, not as a better encoder.")
    ap.add_argument("--prod-dims", type=int, default=0,
                    help="columns of the context vector the product is read from "
                         "(0 = all). Use 2048 for morgan: the last 10 are the standardised "
                         "RDKit descriptors, i.e. the product's exact post-edit MW / logP "
                         "/ TPSA / HBD / HBA / rotB / rings / heavy / QED, which is the "
                         "label's own arithmetic in a known column. Stripping them leaves "
                         "the fingerprint, which still carries much of it — folded ECFP4 "
                         "recovers HBD/HBA/rings well and TPSA/logP are near-additive over "
                         "fragments — so this removes the TRIVIAL oracle, not the "
                         "learnable one. MolBERT has no tail; leave it at 0.")
    ap.add_argument("--n-prod", type=int, default=128,
                    help="width the product embedding is projected to, and on the gated "
                         "architectures the number of tokens it becomes. As ONE token the "
                         "product would enter the pooled candidate vector at 1/(R+1) = "
                         "0.6% and the gate would have to find it from there; at "
                         "n_prod/(R+n_prod) the gate's job is suppression instead.")
    ap.add_argument("--no-cand-attn", action="store_true",
                    help="ablate the candidate-comparison module: each candidate's "
                         "pooled rule representation goes straight to the shared MLP")
    ap.add_argument("--select-metric", default="loss/val_value",
                    help="val metric that decides which epoch is 'before overfitting': "
                         "the checkpoint with the LOWEST value of it becomes best.pt, "
                         "and --patience counts against it. Defaults to the value loss "
                         "(the MSE term), which is a proper loss with a real overfitting "
                         "turn — decide/regret is a discrete argmax statistic and is "
                         "noisier epoch to epoch. Any key from the val dict works, e.g. "
                         "decide/regret or loss/val_rank; all of them are logged either "
                         "way.")
    ap.add_argument("--patience", type=int, default=0,
                    help="stop after this many epochs with no improvement in "
                         "--select-metric (0 = run every epoch)")
    ap.add_argument("--eval-test", action="store_true",
                    help="after training, reload best.pt and score the TEST split; "
                         "written to <out-dir>/metrics.json")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--train-frac", type=float, default=1.0,
                    help="fraction of TRAIN INSTANCES to keep (learning-curve sweeps)")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=0,
                    help="validate every N optimiser steps instead of once per epoch "
                         "(0 = per epoch). Every rank evaluates at the same step — the "
                         "metric reduction is a collective — which the DistributedSampler "
                         "guarantees by giving each rank the same number of batches.")
    ap.add_argument("--no-progress", action="store_true",
                    help="no tqdm bar (it is on by default on rank 0)")
    ap.add_argument("--baseline-feature", default="c_pred_gap",
                    help="rule column used as the shipped-objective baseline "
                         "(negated: lower predicted gap = better)")
    ap.add_argument("--wandb-project", default="molkit")
    ap.add_argument("--wandb-entity", default="",
                    help="wandb team; empty uses the logged-in default entity")
    ap.add_argument("--wandb-name", default="")
    ap.add_argument("--wandb-group", default="rule-selection",
                    help="wandb group, so the arms of one sweep land together")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    if world > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    meta = json.load(open(os.path.join(args.data, "meta.json")))
    spec = FeatureSpec.load(os.path.join(args.data, "spec.json"))
    ctx_dir = args.ctx_dir or meta.get("ctx_dir") or ""
    if not ctx_dir:
        raise SystemExit("no context embeddings: pass --ctx-dir")
    # memmap, NOT materialised: a morgan context over ~1.5M molecules is >12 GB and
    # every rank would otherwise hold its own copy. Items are copied one row at a time.
    ctx = np.load(os.path.join(ctx_dir, "ctx.npy"), mmap_mode="r")

    # the baseline column: the shipped objective's own ranking, negated
    try:
        base_col = spec.rule_names.index(args.baseline_feature)
    except ValueError:
        base_col = None

    prod_dims = 0
    if args.product_ctx:
        if not os.path.exists(os.path.join(args.data, "cctx.npy")):
            raise SystemExit(
                f"--product-ctx needs {args.data}/cctx.npy. This tensors directory was "
                f"built before it existed; run\n    python "
                f"5_rule_selection/scripts/add_product_ctx.py --tensors {args.data}")
        prod_dims = args.prod_dims or int(ctx.shape[1])
        if prod_dims > ctx.shape[1]:
            raise SystemExit(f"--prod-dims {prod_dims} > context width {ctx.shape[1]}")
        # a morgan product read at full width includes the standardised RDKit
        # descriptors, i.e. the post-edit MW/logP/TPSA/HBD/HBA/rotB/rings/heavy/QED —
        # 9 of the 14 properties sat_ratio is computed from, in known columns. Easy to
        # forget, and it silently turns the arm into an oracle, so say so.
        try:
            enc = json.load(open(os.path.join(ctx_dir, "index.json")))["encoder"]
        except Exception:
            enc = ""
        if rank == 0 and enc == "morgan" and prod_dims >= 2058:
            print("# [warn] --product-ctx on a morgan context at full width: the last "
                  "10 columns are the product's EXACT post-edit property values, so "
                  "the model is handed the label's own arithmetic. Pass --prod-dims "
                  "2048 for the fingerprint alone (what ablation_24way.sh does).",
                  flush=True)
    ds = lambda split, **kw: StateDataset(args.data, split, ctx, prod_dims=prod_dims, **kw)
    train_ds = ds(0, frac=args.train_frac, seed=args.seed)
    val_ds, test_ds = ds(1), ds(2)
    if rank == 0:
        print(f"# states: train {len(train_ds):,} (frac {args.train_frac}) "
              f"val {len(val_ds):,} test {len(test_ds):,} "
              f"| G={meta['G']} R={meta['R']} N={meta['N']} ctx={ctx.shape[1]}")
        if prod_dims:
            miss = meta.get("missing_product_context")
            print(f"# product context ON: dims [:{prod_dims}] of {ctx.shape[1]} -> "
                  f"{args.n_prod}"
                  + (f" ({args.n_prod} extra tokens/candidate, "
                     f"{meta['G'] + meta['N'] * (meta['R'] + args.n_prod):,} tokens/state "
                     f"vs {meta['G'] + meta['N'] * meta['R']:,})"
                     if args.arch != "flat" else " (concatenated)")
                  + (f" | {miss:,} products lack an embedding" if miss else ""))
        print(f"# checkpoint selection: lowest val {args.select_metric}"
              + (f", patience {args.patience} validations" if args.patience
                 else ", no early stop"))
        print(f"# lr: {args.lr:g} {args.lr_schedule}"
              + (f" after {args.warmup_frac:.0%} warmup" if args.warmup_frac > 0
                 else " (no warmup)"))

    # persistent_workers: without it the pool is torn down and respawned every epoch,
    # which on a 25-epoch run is pure waste. It does not change what is loaded — with a
    # DistributedSampler the indices are still produced fresh in the main process after
    # set_epoch, so shuffling per epoch is unaffected.
    mk = lambda ds, shuffle: DataLoader(
        ds, batch_size=args.batch_size, collate_fn=collate, num_workers=args.workers,
        pin_memory=True, drop_last=shuffle,
        sampler=DistributedSampler(ds, shuffle=shuffle) if world > 1 else None,
        shuffle=(shuffle and world == 1),
        persistent_workers=args.workers > 0,
        prefetch_factor=4 if args.workers > 0 else None)
    train_dl, val_dl = mk(train_ds, True), mk(val_ds, False)
    test_dl = mk(test_ds, False)

    if args.arch == "flat":
        if args.no_cand_attn and rank == 0:
            print("# note: --no-cand-attn has no meaning for --arch flat (ignored)")
        model = FlatScorer(meta["G"], meta["R"], ctx.shape[1],
                           hidden=args.flat_hidden,
                           target_params=args.target_params,
                           dropout=args.dropout,
                           prod_dim=prod_dims, n_prod=args.n_prod).to(device)
    else:
        mlp_hidden = args.mlp_hidden
        if args.match_params:
            def n_par_at(h):
                # prod_dim=0 on purpose: the budget sizes the SCORER, so a product arm
                # keeps the same mlp_hidden as the baseline it is compared against
                m = RuleSelector(meta["G"], meta["R"], ctx.shape[1],
                                 d_model=args.d_model, d_key=args.d_key,
                                 n_heads=args.heads, mlp_hidden=h, dropout=0.0,
                                 candidate_attention=not args.no_cand_attn)
                return sum(p.numel() for p in m.parameters())
            h = 32
            while h < 8192 and n_par_at(h + 32) <= args.target_params:
                h += 32
            mlp_hidden = h
            if rank == 0 and mlp_hidden != args.mlp_hidden:
                print(f"# --match-params: mlp_hidden {args.mlp_hidden} -> {mlp_hidden} "
                      f"to reach ~{args.target_params/1e6:.2f}M")
        model = RuleSelector(meta["G"], meta["R"], ctx.shape[1], d_model=args.d_model,
                             d_key=args.d_key, n_heads=args.heads,
                             mlp_hidden=mlp_hidden, dropout=args.dropout,
                             candidate_attention=not args.no_cand_attn,
                             prod_dim=prod_dims, n_prod=args.n_prod).to(device)
        args.mlp_hidden = mlp_hidden
    n_par = sum(p.numel() for p in model.parameters())
    n_prod_par = sum(p.numel() for n_, p in model.named_parameters()
                     if n_.startswith(("p_proj.", "p_tokens.")))
    if rank == 0:
        extra = (f", input {ctx.shape[1] + meta['G'] + meta['R'] + (args.n_prod if prod_dims else 0)}"
                 f" -> hidden {model.hidden}" if args.arch == "flat" else
                 (", no candidate attention" if args.no_cand_attn else ""))
        print(f"# arch {args.arch}: {n_par/1e6:.3f}M parameters{extra}")
        if prod_dims:
            # the SCORER GEOMETRY is the baseline's — mlp_hidden on gated, hidden on
            # flat — which is what makes the pair a comparison of inputs. The size is
            # only identical on the gated arms; flat's first layer is genuinely 128
            # columns wider, and those columns live in net, not in the product block.
            print(f"#   of which {n_prod_par/1e6:.3f}M is the product-embedding block, "
                  f"{(n_par - n_prod_par)/1e6:.3f}M the scorer"
                  + (f" (mlp_hidden {args.mlp_hidden}, unchanged by --product-ctx)"
                     if args.arch != "flat" else
                     f" (hidden {model.hidden}, unchanged; its first layer takes "
                     f"{args.n_prod} more columns)"))
    if world > 1:
        model = DDP(model, device_ids=[local_rank])

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    steps_per_epoch = len(train_dl)
    total_steps = max(1, steps_per_epoch * args.epochs)
    if rank == 0:
        every = args.eval_every or steps_per_epoch
        print(f"# {steps_per_epoch:,} steps/epoch x {args.epochs} epochs = "
              f"{total_steps:,} steps (batch {args.batch_size} x {world} ranks = "
              f"{args.batch_size * world:,} states/step); validating every "
              f"{every:,} steps ({total_steps // max(every, 1)} validations)")
    warm = max(1, int(total_steps * args.warmup_frac))

    def lr_at(step):
        if args.warmup_frac > 0 and step < warm:
            return step / warm
        if args.lr_schedule == "constant":
            return 1.0
        t = (step - warm) / max(1, total_steps - warm)
        return 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    use_wandb = (rank == 0) and not args.no_wandb
    if use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, entity=args.wandb_entity or None,
                   name=args.wandb_name or os.path.basename(args.out_dir.rstrip("/")),
                   group=args.wandb_group, job_type="train",
                   tags=["rule-selection", f"ctx{int(ctx.shape[1])}"],
                   config={**vars(args), **meta, "params": n_par,
                           "params_product": n_prod_par,
                           "params_scorer": n_par - n_prod_par,
                           "prod_dims": prod_dims,
                           "ctx_dim": int(ctx.shape[1])})
    os.makedirs(args.out_dir, exist_ok=True)

    def evaluate(dl=None):
        dl = val_dl if dl is None else dl
        model.eval()
        acc, byd, bys = {}, {}, {}
        qs, ys = [], []
        vl = rl = 0.0
        nb = 0
        with torch.no_grad():
            for batch in dl:
                b = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
                out = model(b["g"], b["gmask"], b["r"], b["rmask"], b["cmask"],
                            b["ctx"], b.get("p"))
                vl += value_loss(out["q"], b["y"], b["cmask"]).item()
                rl += ranking_loss(out["score"], b["y"], b["cmask"]).item()
                nb += 1
                s = out["score"].float().cpu().numpy()
                y = b["y"].float().cpu().numpy()
                cm = b["cmask"].cpu().numpy()
                q = out["q"].float().cpu().numpy()
                base = (-b["r"][:, :, base_col].float().cpu().numpy()
                        if base_col is not None else None)
                # every metric is a per-SET quantity, so slicing then computing and
                # computing then slicing give the same thing: compute once, bucket
                # three ways. The old code re-ran the per-set loop for the pooled
                # numbers, again per depth and again per source — 18.3 s of a 19.3 s
                # evaluation against 0.29 s for the forward pass.
                rows = set_rows(s, y, cm, base)
                merge(acc, rows_to_acc(rows))
                m = cm & (y >= 0)
                qs.append(q[m])
                ys.append(y[m])
                dep = batch["depth"].numpy()
                for d in np.unique(dep):
                    merge(byd.setdefault(int(d), {}), rows_to_acc(rows, dep == d))
                src = batch["source"].numpy()
                for sc in np.unique(src):
                    merge(bys.setdefault(int(sc), {}), rows_to_acc(rows, src == sc))
        q = np.concatenate(qs) if qs else np.zeros(0)
        y = np.concatenate(ys) if ys else np.zeros(0)
        flat = {"value/mse": (float(((q - y) ** 2).sum()), len(y)),
                "value/mae": (float(np.abs(q - y).sum()), len(y)),
                "value/mean_q": (float(q.sum()), len(y)),
                "value/mean_y": (float(y.sum()), len(y))}
        flat.update(calibration(q, y))
        merge(acc, flat)
        red = reduce_dict(acc, device)
        red["loss/val_value"] = vl / max(nb, 1)
        red["loss/val_rank"] = rl / max(nb, 1)
        for d, a in sorted(byd.items()):
            for k, v in reduce_dict(a, device).items():
                red[f"depth{d}/{k.split('/')[-1]}"] = v
        for sc, a in sorted(bys.items()):
            tag = "scaffold" if sc == 0 else "fg"
            for k, v in reduce_dict(a, device).items():
                red[f"{tag}/{k.split('/')[-1]}"] = v
        model.train()
        return red

    def validate(epoch, step, bar=None):
        """Evaluate, log, checkpoint, and say whether training should stop.

        Called at every validation point — end of epoch, or every --eval-every steps.
        `evaluate` reduces its metrics across ranks, so every rank must reach this
        together; the caller only ever invokes it on a step count that is identical on
        all ranks.
        """
        nonlocal best, best_epoch, best_step, bad
        te = time.time()
        red = evaluate()
        t_eval = time.time() - te
        stop = torch.zeros(1, device=device)
        if rank == 0:
            head = {k: round(v, 4) for k, v in red.items()
                    if k.split("/")[0] in ("loss", "value", "rank", "decide")}
            msg = f"[epoch {epoch} step {step}] {json.dumps(head, sort_keys=True)}"
            print(msg + f"  (eval {t_eval:.1f}s)", flush=True)
            if use_wandb:
                import wandb
                wandb.log({**red, "perf/eval_s": t_eval}, step=step)
            score = red.get(args.select_metric)
            if score is None:
                raise SystemExit(
                    f"--select-metric {args.select_metric!r} is not in the val metrics; "
                    f"available: {', '.join(sorted(red))}")
            ts = time.time()
            state = (model.module if world > 1 else model).state_dict()
            blob = {"model": state, "args": vars(args), "meta": meta,
                    "epoch": epoch, "step": step, "metrics": red}
            torch.save(blob, os.path.join(args.out_dir, "last.pt"))
            if score < best:
                best, best_epoch, best_step, bad = score, epoch, step, 0
                torch.save(blob, os.path.join(args.out_dir, "best.pt"))
                print(f"  new best {args.select_metric}={score:.4f} "
                      f"(decide/regret={red.get('decide/regret', float('nan')):.4f}) "
                      f"[save {time.time() - ts:.1f}s]", flush=True)
            else:
                bad += 1
                print(f"  no improvement ({bad}/{args.patience or '-'}) "
                      f"best={best:.4f} @ epoch {best_epoch} step {best_step}",
                      flush=True)
                if args.patience and bad >= args.patience:
                    stop += 1
            if bar is not None:
                bar.set_postfix_str(f"val {args.select_metric.split('/')[-1]}="
                                    f"{score:.4f} best={best:.4f}", refresh=True)
        # every rank has to leave the loop together or DDP deadlocks on the next step
        if world > 1:
            dist.broadcast(stop, src=0)
        return stop.item() > 0

    step = 0
    best = float("inf")
    best_epoch = best_step = -1
    bad = 0
    stopped = False
    t0 = time.time()
    for epoch in range(args.epochs):
        if stopped:
            break
        if world > 1:
            train_dl.sampler.set_epoch(epoch)
        bar = None
        if rank == 0 and not args.no_progress:
            bar = _tqdm(total=steps_per_epoch,
                        desc=f"epoch {epoch + 1}/{args.epochs}", unit="step",
                        dynamic_ncols=True, mininterval=1.0, leave=True)
        t_ep = time.time()
        for batch in train_dl:
            b = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
            out = model(b["g"], b["gmask"], b["r"], b["rmask"], b["cmask"],
                        b["ctx"], b.get("p"))
            lv = value_loss(out["q"], b["y"], b["cmask"])
            # lambda_rank == 0 skips the pair enumeration entirely rather than
            # multiplying it by zero: [B,N,N] pair tensors are not free, and a zeroed
            # term still walks the graph.
            lr_ = (ranking_loss(out["score"], b["y"], b["cmask"])
                   if args.lambda_rank != 0 else out["score"].sum() * 0.0)
            loss = args.lambda_value * lv + args.lambda_rank * lr_
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            step += 1
            if use_wandb and step % args.log_every == 0:
                ag = out["attn_global"].detach()
                ar = out["attn_rule"].detach()
                ent = lambda p: float(-(p.clamp_min(1e-12) * p.clamp_min(1e-12).log())
                                      .sum(-1).mean())
                ac = out["attn_cand"].detach()
                offdiag = float((ac.sum(-1) - ac.diagonal(dim1=1, dim2=2)).mean())
                import wandb
                wandb.log({"loss/train": loss.item(), "loss/train_value": lv.item(),
                           "loss/train_rank": lr_.item(),
                           "opt/lr": sched.get_last_lr()[0], "opt/grad_norm": float(gn),
                           "attn/global_mass": float(ag.sum(-1).mean()),
                           "attn/rule_mass": float(ar.sum((-1, -2)).mean()),
                           # what fraction of the gate went to the post-edit molecule
                           # instead of the interpretable columns. Uniform is
                           # N*n_prod/(G+N*(R+n_prod)); above that the decision is
                           # leaning on the product structure.
                           "attn/prod_mass": float(out["attn_prod"].detach()
                                                   .sum((-1, -2)).mean()),
                           "attn/entropy_global": ent(ag),
                           "attn/cand_offdiag": offdiag,
                           "perf/states_per_s": step * args.batch_size * world / (time.time() - t0),
                           "epoch": epoch}, step=step)
            if bar is not None:
                bar.update(1)
                if step % 20 == 0:
                    bar.set_postfix_str(f"loss {loss.item():.4f} "
                                        f"lr {sched.get_last_lr()[0]:.2e}"
                                        + ("" if best == float("inf")
                                           else f" best={best:.4f}"), refresh=False)
            if args.eval_every and step % args.eval_every == 0:
                if validate(epoch, step, bar):
                    stopped = True
                    break
        if bar is not None:
            bar.close()
        if rank == 0:
            print(f"# epoch {epoch} train {time.time() - t_ep:.1f}s "
                  f"({steps_per_epoch / max(time.time() - t_ep, 1e-9):.1f} step/s)",
                  flush=True)
        # per-epoch validation, unless a step schedule already covers it — but always
        # validate once at the very end so `best.pt` reflects the final weights.
        last = stopped or epoch == args.epochs - 1
        if not args.eval_every or last:
            if validate(epoch, step, None):
                stopped = True
        if stopped:
            if rank == 0:
                print(f"# early stop at epoch {epoch} step {step} "
                      f"(best {args.select_metric}={best:.4f} @ epoch {best_epoch} "
                      f"step {best_step})", flush=True)
            break

    # ---- final: reload the best checkpoint and score the held-out TEST split ----
    if args.eval_test:
        blob = torch.load(os.path.join(args.out_dir, "best.pt"), map_location="cpu",
                          weights_only=False)
        (model.module if world > 1 else model).load_state_dict(blob["model"])
        test = evaluate(test_dl)
        if rank == 0:
            print(f"[test @ epoch {blob['epoch']}] "
                  f"{json.dumps({k: round(v, 4) for k, v in test.items() if k.split('/')[0] in ('loss', 'value', 'rank', 'decide')}, sort_keys=True)}",
                  flush=True)
            blob["test_metrics"] = test
            torch.save(blob, os.path.join(args.out_dir, "best.pt"))
            with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
                json.dump({"run": args.wandb_name or os.path.basename(
                               args.out_dir.rstrip("/")),
                           "args": vars(args), "meta": meta, "params": n_par,
                           "params_product": n_prod_par,
                           "params_scorer": n_par - n_prod_par,
                           "best_epoch": blob["epoch"],
                           "best_step": blob.get("step"),
                           "steps_per_epoch": steps_per_epoch,
                           "epochs_run": epoch + 1,
                           "select_metric": args.select_metric,
                           "val": blob["metrics"], "test": test}, fh, indent=2)
            if use_wandb:
                import wandb
                wandb.log({f"test/{k}": v for k, v in test.items()}, step=step)
                wandb.summary.update({f"test/{k}": v for k, v in test.items()})
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
