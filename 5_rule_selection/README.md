# 5_rule_selection — scoring the candidates `suggest_edits` offers

A small model that, at one edit step, scores each candidate rule on the shortlist
`suggest_edits` returns. `6_edit_reasoning/` uses two of its outputs when it renders
the assistant span: the feature gate (which functional group the network is attending
to) and the per-candidate score `q` (which decides the two observation lines).

The search itself is not duplicated here — it lives in `3_toolchain_gen/`, and this
folder drives it.

## The label

`3_toolchain_gen/branch_tree.py` expands the full edit tree per instance and stores,
per node, `n_leaf` / `n_sat_leaf` / `sat_ratio` / `sat_any` / `min_depth` alongside the
candidate features. `sat_ratio` is the fraction of paths through that node that end
inside the property box:

```
Q(s, a) = P(success | s, a, uniform policy over the expanded action set)
```

Two things it is **not**, both spelled out in `branch_tree.py`'s docstring:

1. Not `P(success | s, a, π_random)` over the true action set. The tree branches over
   `top_k` candidates while the real candidate count is far larger, so the label is
   conditional on the shortlist the `predicted_gap` ranking produced.
2. Not a binary "can this branch work" — `sat_any` is ~1 near the root, which is why
   the soft ratio exists.

`3_toolchain_gen/branch_features.py` also records the negative result that motivated
the model: on the root layer every one of ~40 hand-built features lands at within-round
AUC 0.475–0.53, and picking a branch by any of them — including an out-of-fold logistic
over all of them — stays near the trivial baseline. Read that docstring before trusting
a single feature.

## The model

`model.py`: every interpretable scalar becomes a vector, `z_m = e_m + x_m · w_m`, with
a per-feature identity embedding and a per-feature value direction — the same number
means different things in different columns. A molecular context vector gates those
features (**A**), and candidate self-attention (**A_c**) compares candidates against
each other. Both attentions are read back out in `6_edit_reasoning/`.

`context_embed.py` precomputes the context vector per unique SMILES; the encoder never
trains, so it is cached. Two encoders are available — Morgan fingerprints (no extra
dependency) and MolBERT. `defaults.py` records the 24-arm grid that picked the shipped
setting and the fallback the token limit forces.

## Training

```bash
# 1. label a tree dump
PYTHONPATH=. python 3_toolchain_gen/branch_tree.py --input <instances.jsonl> --out <dump>

# 2. fixed-shape memmapped tensors, FeatureSpec fitted on the train split only
PYTHONPATH=. python 5_rule_selection/prepare_dataset.py --states <dump>/states.jsonl --out <data>

# 3. train (DDP; the model is a few M parameters, so this is for throughput)
torchrun --nproc_per_node=8 -m 5_rule_selection.train --data <data> --out <run>
```

The checkpoint `6_edit_reasoning/` expects is a **gated** one — a flat-MLP ablation
checkpoint has no feature gating to reason from, and `reasoning.py` refuses it with
that message rather than rendering something unsupported.

## `toolchain_states.py`

The model trains on exhaustive-search trees, where every node carries all `top_k`
children and a `sat_ratio` per child. The corpus stage 4 actually renders is one
beam-search path: the full `suggest_edits` candidate list is there, but only one
candidate was committed and no counterfactual label exists for the others. This module
reads decision states out of those real chains so the trained model can be applied to
them.

## A note on the docstrings

These modules carry the measurements that decided each design choice, and some of them
name analysis scripts and ablation arms that are not part of this release. The code
paths are self-contained; only the cross-references are not.
