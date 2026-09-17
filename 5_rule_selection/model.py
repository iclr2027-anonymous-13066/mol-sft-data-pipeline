"""The rule-selection model: score each candidate rule at one decision state.

Four stages, following the design note:

1. **Feature vectorisation.** Every interpretable scalar becomes a vector,
   `z_m = e_m + x_m * w_m`, with a per-feature identity embedding `e_m` and a
   per-feature value direction `w_m`. The same number means different things in
   different columns, so each column gets its own pair.
2. **Context-conditioned gating.** The molecular context `C` is the single query;
   the keys are all `G + N*R` feature tokens. One softmax says which features matter
   for THIS molecule. Features that are meaningless for the instance (the bound
   distance of an unconstrained property) are masked to -inf and get exactly zero.
3. **Candidate comparison.** The gated tokens pool into one vector per candidate,
   then self-attention over candidates makes each representation aware of its rivals
   — "is this rule better than the others I could pick", not "is this rule good".
   `candidate_attention=False` removes exactly this stage and nothing else: each
   candidate's pooled vector goes straight to the shared MLP, so the score is a
   function of that rule alone. It is the ablation arm — note that some sibling
   information still reaches it through the FEATURES (`set_prob_spread`,
   `c_prob_margin`, `c_is_unique_best`, `cand_similarity`), so this removes the
   learned comparison, not every trace of the set.
4. **Scoring.** A shared MLP over `[C~ ; u_G ; u_i]` gives one logit per candidate;
   `sigmoid` makes it a predicted success probability comparable to `sat_ratio`.

**The post-edit molecule** (optional, `prod_dim > 0`). Everything above scores a rule
from the state and the rule's own predicted deltas — never from the molecule the edit
would actually produce. Turning it on projects that product molecule to `n_prod`
scalars and gives EACH of them its own feature token, so the product enters exactly
where the interpretable features do: it is gated by the same softmax, pooled into the
same `u_i`, and compared by the same candidate attention. Per-dimension tokens rather
than one 128-d token, because as a single token the product would enter the pooled
mean at `1/(R+1)` — 0.6% of `u_i` at initialisation — and the gate would have to
discover it from there; at `n_prod/(R+n_prod)` the gate's job is suppression instead.

`A` (feature relevance) and `A_c` (candidate comparison) are returned so the
reasoning-evidence extraction can read them. The product columns are SPLIT OFF into
`attn_prod` rather than returned inside `A`: `spec.rule_names` indexes `A` one-to-one
and a latent dimension has no name to print. `reasoning.py` renormalises within the
mask it is given, so the gate multipliers it reports stay "among the interpretable
features" — and `attn_prod`'s mass is then the honest measure of how much the decision
leaned on the product structure instead.

Scale note: a single query over ~800 tokens gives each an attention weight around
1/800, which would shrink the pooled vectors to nothing. The gate is therefore
applied as `alpha * n_tokens` — the relative weighting the design asks for, at a
magnitude that keeps the pooled vector in the same range as its input.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

NEG = -1e9


class FeatureTokens(nn.Module):
    """x in R^{...,M} -> z in R^{...,M,d} with z = e_m + x_m * w_m."""

    def __init__(self, n_features: int, d_model: int):
        super().__init__()
        self.identity = nn.Parameter(torch.randn(n_features, d_model) * 0.02)
        self.direction = nn.Parameter(torch.randn(n_features, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.identity + x.unsqueeze(-1) * self.direction


class ProductEncoder(nn.Module):
    """The post-edit molecule of one candidate, projected to `n_prod` scalars.

    The input comes from the SAME cached encoder the state context does — for morgan
    the product molecule's 2048 fingerprint bits (the 10 RDKit descriptors are sliced
    off upstream: standardised or not, they ARE the exact post-edit property values,
    which is the label's own arithmetic handed over in a known column), for MolBERT the
    full 768.

    The closing LayerNorm is not cosmetic. Downstream each output scalar becomes its
    own feature token beside R columns that went through IQR normalisation and a
    +-value_clip clamp; an unnormalised projection would set the scale of the pooled
    candidate vector by itself.
    """

    def __init__(self, prod_dim: int, n_prod: int = 128, hidden: int = 0):
        super().__init__()
        self.n_prod = n_prod
        h = hidden or n_prod
        self.net = nn.Sequential(nn.Linear(prod_dim, h), nn.GELU(),
                                 nn.Linear(h, n_prod), nn.LayerNorm(n_prod))

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        return self.net(p)


class RuleSelector(nn.Module):
    def __init__(self, n_global: int, n_rule: int, ctx_dim: int, d_model: int = 128,
                 d_key: int = 64, n_heads: int = 4, mlp_hidden: int = 256,
                 dropout: float = 0.1, value_clip: float = 8.0,
                 candidate_attention: bool = True, prod_dim: int = 0,
                 n_prod: int = 128):
        super().__init__()
        self.n_global, self.n_rule = n_global, n_rule
        self.value_clip = value_clip
        self.candidate_attention = candidate_attention
        self.g_tokens = FeatureTokens(n_global, d_model)
        self.r_tokens = FeatureTokens(n_rule, d_model)
        # kept in their own modules rather than widening r_tokens: spec.rule_names
        # indexes r_tokens one-to-one, and `prod_dim == 0` must leave the state_dict
        # byte-identical to the arm without the product input
        self.n_prod = n_prod if prod_dim else 0
        self.p_proj = ProductEncoder(prod_dim, n_prod) if prod_dim else None
        self.p_tokens = FeatureTokens(n_prod, d_model) if prod_dim else None

        self.ctx_query = nn.Linear(ctx_dim, d_key)
        self.ctx_value = nn.Linear(ctx_dim, d_model)
        self.key = nn.Linear(d_model, d_key)
        self.scale = d_key ** -0.5

        self.norm_g = nn.LayerNorm(d_model)
        self.norm_c = nn.LayerNorm(d_model)
        # not created when ablated, so the parameter count and the state_dict both
        # reflect the arm that was actually trained
        self.cand_attn = (nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                                batch_first=True)
                          if candidate_attention else None)
        self.norm_after = nn.LayerNorm(d_model)
        self.score = nn.Sequential(
            nn.Linear(3 * d_model, mlp_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden // 2, 1))

    def forward(self, g, gmask, r, rmask, cmask, ctx, p=None):
        """g [B,G] · gmask [B,G] · r [B,N,R] · rmask [B,N,R] · cmask [B,N] · ctx [B,dc]
        · p [B,N,dp] the post-edit molecule of each candidate (only when prod_dim>0).

        Returns dict with `score` [B,N], `q` [B,N], `attn_global` [B,G],
        `attn_rule` [B,N,R], `attn_prod` [B,N,n_prod], `attn_cand` [B,N,N].
        """
        B, N, R = r.shape
        g = g.clamp(-self.value_clip, self.value_clip)
        r = r.clamp(-self.value_clip, self.value_clip)
        zg = self.g_tokens(g)                                  # [B,G,d]
        zr = self.r_tokens(r)                                  # [B,N,R,d]

        Rt = R
        if self.p_proj is not None:
            if p is None:
                raise ValueError("built with prod_dim > 0: forward() needs p [B,N,dp]")
            zp = self.p_tokens(self.p_proj(p).clamp(-self.value_clip, self.value_clip))
            zr = torch.cat([zr, zp], dim=2)                    # [B,N,R+n_prod,d]
            rmask = torch.cat([rmask, torch.ones(B, N, self.n_prod, dtype=torch.bool,
                                                 device=r.device)], dim=-1)
            Rt = R + self.n_prod

        q = self.ctx_query(ctx).unsqueeze(1)                   # [B,1,dk]
        kg = self.key(zg)                                      # [B,G,dk]
        kr = self.key(zr.reshape(B, N * Rt, -1))               # [B,N*Rt,dk]
        keys = torch.cat([kg, kr], dim=1)
        logits = (q @ keys.transpose(1, 2)).squeeze(1) * self.scale      # [B,G+N*Rt]

        # a padded candidate's tokens, and features this instance does not define,
        # take no attention mass at all
        rmask = rmask & cmask.unsqueeze(-1)
        mask = torch.cat([gmask, rmask.reshape(B, N * Rt)], dim=1)
        logits = logits.masked_fill(~mask, NEG)
        alpha = torch.softmax(logits, dim=-1)                           # [B,G+N*Rt]
        n_tok = mask.sum(-1, keepdim=True).clamp(min=1)
        gate = alpha * n_tok                                              # see docstring

        ag, ar = gate[:, :self.n_global], gate[:, self.n_global:].reshape(B, N, Rt)
        u_g = self.norm_g(((zg * ag.unsqueeze(-1)) * gmask.unsqueeze(-1)).sum(1)
                          / gmask.sum(1, keepdim=True).clamp(min=1))
        rm = rmask.unsqueeze(-1)
        u_i = self.norm_c(((zr * ar.unsqueeze(-1)) * rm).sum(2)
                          / rmask.sum(-1, keepdim=True).clamp(min=1))     # [B,N,d]

        pad = ~cmask
        if self.cand_attn is not None:
            attended, attn_c = self.cand_attn(u_i, u_i, u_i, key_padding_mask=pad,
                                              need_weights=True,
                                              average_attn_weights=True)
            u_hat = self.norm_after(u_i + attended)
        else:
            # each candidate sees only itself. The identity matrix is returned in
            # place of A_c so every consumer keeps working and the "rival" it names is
            # honestly nobody.
            u_hat = self.norm_after(u_i)
            attn_c = (torch.eye(N, device=u_i.device, dtype=u_i.dtype)
                      .expand(B, N, N) * cmask.unsqueeze(-1))

        c_proj = self.ctx_value(ctx)
        x = torch.cat([c_proj.unsqueeze(1).expand(-1, N, -1),
                       u_g.unsqueeze(1).expand(-1, N, -1), u_hat], dim=-1)
        score = self.score(x).squeeze(-1).masked_fill(pad, NEG)
        a_r = alpha[:, self.n_global:].reshape(B, N, Rt)
        # `u_hat` and `x` ride along for the injection study: `x` is EXACTLY what the
        # shared scoring MLP sees, so anything `q` knows about a candidate is already in
        # it, and `u_hat` is its only candidate-varying third (the other two thirds are
        # the context projection and the pooled state, both constant within a round).
        # Returning them adds no compute and does not touch the state_dict.
        return {"score": score, "q": torch.sigmoid(score.clamp(-30, 30)),
                "attn_global": alpha[:, :self.n_global],
                "attn_rule": a_r[..., :R],
                "attn_prod": a_r[..., R:],
                "attn_cand": attn_c,
                "u_hat": u_hat, "u_g": u_g, "c_proj": c_proj, "x": x}


class FlatScorer(nn.Module):
    """The plain baseline: concatenate and score. No gating, no candidate interaction.

    Per candidate the input is `[C ; g ; r_i]` — the molecule embedding and the state
    features, both shared across the candidates of a set, then that candidate's own rule
    features — and one shared MLP maps it to a logit. So the score of a rule is a
    function of the rule and the state alone; nothing in the architecture lets one
    candidate see another, and no feature is weighted by the molecule.

    Two details that are not optional:

    * **Masked features are zeroed.** `prepare_dataset` stores NORMALISED values, so a
      feature the instance does not define (the bound distance of an unconstrained
      property) sits at `(0 - center) / scale`, which is not zero and reads as a real
      measurement. `RuleSelector` never notices because the mask kills its attention;
      a flat MLP would be fed a fabricated number. Multiplying by the mask is the
      minimum fix. The mask BITS are not appended — that would be a different (and
      better-informed) model than the one this arm is meant to be.
    * **Padded candidates** get `-1e9`, as in `RuleSelector`, so argmax and the losses
      ignore them.

    `attn_global` / `attn_rule` come back as zeros and `attn_cand` as the identity:
    there is no attention here, and a zero in `attn/*` on wandb is how you confirm the
    architecture switch took effect.

    Width is solved for a parameter budget rather than fixed, because the input width
    depends on the context encoder (2058 for morgan, 768 for molbert) and the point of
    this arm is to compare architectures at equal capacity, not equal hidden size.

    With `prod_dim > 0` the projected post-edit molecule of the candidate is appended to
    the PER-CANDIDATE half of the row, `[C ; g ; r_i ; p~_i]` — it is a property of
    candidate i, not of the set, so it cannot ride in the shared block. There is no
    token axis here to tokenise it into, and there should not be: concatenation is what
    this arm is. The width is still solved on the BASELINE row, so the arm has the same
    hidden units as the one without the product — its first layer genuinely takes
    n_prod more columns, which is the honest price of the extra input rather than a
    narrower scorer.
    """

    def __init__(self, n_global: int, n_rule: int, ctx_dim: int, hidden: int = 0,
                 target_params: int = 600_000, dropout: float = 0.1,
                 value_clip: float = 8.0, prod_dim: int = 0, n_prod: int = 128):
        super().__init__()
        self.value_clip = value_clip
        self.n_prod = n_prod if prod_dim else 0
        self.p_proj = ProductEncoder(prod_dim, n_prod) if prod_dim else None
        d_base = ctx_dim + n_global + n_rule
        d_in = d_base + self.n_prod
        self.hidden = hidden or solve_hidden(d_base, target_params)
        h = self.hidden
        self.net = nn.Sequential(
            nn.Linear(d_in, h), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h, h // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h // 2, 1))

    def forward(self, g, gmask, r, rmask, cmask, ctx, p=None):
        B, N, R = r.shape
        g = (g * gmask).clamp(-self.value_clip, self.value_clip)
        rm = rmask & cmask.unsqueeze(-1)
        r = (r * rm).clamp(-self.value_clip, self.value_clip)
        shared = torch.cat([ctx, g], dim=-1).unsqueeze(1).expand(-1, N, -1)
        x = torch.cat([shared, r], dim=-1)
        if self.p_proj is not None:
            if p is None:
                raise ValueError("built with prod_dim > 0: forward() needs p [B,N,dp]")
            pt = self.p_proj(p).clamp(-self.value_clip, self.value_clip)
            x = torch.cat([x, pt * cmask.unsqueeze(-1)], dim=-1)
        pad = ~cmask
        score = self.net(x).squeeze(-1).masked_fill(pad, NEG)
        eye = torch.eye(N, device=r.device, dtype=r.dtype).expand(B, N, N)
        return {"score": score, "q": torch.sigmoid(score.clamp(-30, 30)),
                "attn_global": torch.zeros_like(g),
                "attn_rule": torch.zeros_like(r),
                "attn_prod": r.new_zeros(B, N, self.n_prod),
                "attn_cand": eye * cmask.unsqueeze(-1)}


def solve_hidden(d_in: int, target: int) -> int:
    """Widest `h` whose (d_in -> h -> h/2 -> 1) MLP stays under `target` parameters."""
    def n_par(h):
        return d_in * h + h + h * (h // 2) + (h // 2) + (h // 2) + 1
    h = 8
    while n_par(h + 8) <= target:
        h += 8
    return h


def value_loss(q_hat, y, cmask):
    """MSE on the candidates that exist."""
    m = cmask & (y >= 0)
    if m.sum() == 0:
        return q_hat.sum() * 0.0
    return ((q_hat[m] - y[m]) ** 2).mean()


def ranking_loss(score, y, cmask, margin_eps: float = 1e-6):
    """-mean over pairs q_i > q_j of |q_i - q_j| * log sigmoid(s_i - s_j).

    Weighted by the label gap, so a 0.65-vs-0 pair pushes harder than 0.65-vs-0.60.
    Sets where every candidate carries the same label contribute nothing — which is
    why the all-zero sets are dropped from the data in the first place.
    """
    m = cmask & (y >= 0)
    valid = m.unsqueeze(2) & m.unsqueeze(1)
    gap = y.unsqueeze(2) - y.unsqueeze(1)                     # [B,N,N] = q_i - q_j
    pos = valid & (gap > margin_eps)
    if pos.sum() == 0:
        return score.sum() * 0.0
    ds = score.unsqueeze(2) - score.unsqueeze(1)
    loss = -(gap[pos] * F.logsigmoid(ds[pos]))
    return loss.sum() / gap[pos].sum().clamp(min=1e-6)
