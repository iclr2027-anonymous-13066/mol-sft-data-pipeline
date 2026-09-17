# 6_edit_reasoning — the span between `suggest_edits` and `edit_fragment`

At every edit round the toolchain calls `suggest_edits`, gets a shortlist of candidate
rules back, and commits one of them with `edit_fragment`. This stage writes the
assistant turn that sits between those two calls: the text that enumerates the
candidates and states which one is taken.

The span is **rendered deterministically**. No language model is involved — the same
round always produces the same text, byte for byte.

## What it looks like

```
Out of range now: HBD (below).
Present in the molecule: aryl methyl sites for hydroxylation (1), benzene ring (1), thioether (1).
Each edit, at one sigma of its predicted shift:
  Edit 1 | holds MW, logP, logS, HBD, rings_total | adds aliphatic hydroxyl, new here | +2 carbon, +1 oxygen
  Edit 2 | holds MW, logP, logS, HBD, rings_total | adds aliphatic hydroxyl, new here | +3 carbon, +1 oxygen
  Edit 3 | holds MW, logP, HBD, rings_total | at risk logS | no change to any named group | +2 carbon, +1 nitrogen
  Edit 4 | holds MW, logP, logS, HBD, rings_total | adds carbonyl o and carboxylic acid, new here | +3 carbon, +2 oxygen
I think the thing to look at here is the aliphatic hydroxyl it adds: Edit 1 1, Edit 2 1, Edit 3 0, Edit 4 0.
The other side of it is the sp3 carbons it adds: Edit 1 1, Edit 2 1, Edit 3 1, Edit 4 0.
Weighing all of that together, I'll go with Edit 1.
Taking Edit 1: [*:1] -> [*:1]C(C)O.
```

The turn then carries the `edit_fragment` tool call for the named edit.

## The four buckets

Each candidate is placed against every constrained property by whether it survives
**one sigma** of its own predicted shift. `suggest_edits` returns `delta` as
`{avg, std}` per property, so the uncertainty is per property and the bucket is too:

| bucket | meaning |
|---|---|
| `holds` | inside the target range at ±1σ |
| `at risk` | the mean satisfies, one sigma escapes |
| `within reach` | the mean misses, one sigma reaches in |
| `misses` | outside at ±1σ |

The middle two are split by direction on purpose: one list would merge two opposite
meanings. `bucket_span.py` documents the measurements behind the choice of k = 1.0.

## Where each line comes from

| line | source | needs the rule-selection checkpoint |
|---|---|---|
| `Out of range now:` | the current molecule's measured properties | no |
| `Each edit, at one sigma…` | `suggest_edits`' own `delta {avg, std}` | no |
| `Weighing… / Taking Edit N:` | the committed candidate, already in the toolchain | no |
| `Present in the molecule:` | functional groups ranked by the checkpoint's gate | **yes** |
| the two observation lines | the columns picked by `qmax_gap` over the checkpoint's `q` | **yes** |

Without a trained checkpoint (see `5_rule_selection/`) the first three render normally
and the other two are omitted — `fg_line` returns `None` and the observation pass is
skipped, rather than printing something unsupported.

## Modules

| file | what it does |
|---|---|
| `recipe.py` | corpus, arm and work-directory definitions; `ARMS_RECIPE` picks one |
| `prepare.py` | builds the per-round records from the stage-4 toolchains |
| `base_dump.py` | one forward per round: every candidate's value on every live column |
| `gate_dump.py` | the checkpoint's whole gate vector per round |
| `fg_gates.py` | which functional group the gate is spending itself on, and its count |
| `n51_obs.py` | selects the two observation lines (`qmax_gap`, decorrelated) |
| `bucket_span.py` | the one-sigma bucket enumeration — the core renderer |
| `nat_span.py` | arm definitions, including the one used for the released corpus |
| `render_bucket.py` | writes the spans for a whole corpus, no language model |
| `assemble.py` | splices the spans into training records |
| `evidence.py` | the per-round evidence dump the gated variants read |

## Running it

```bash
PYTHONPATH=. python 6_edit_reasoning/prepare.py --procs 48
PYTHONPATH=. python 6_edit_reasoning/base_dump.py --splits train
PYTHONPATH=. python 6_edit_reasoning/fg_gates.py --splits train      # checkpoint
PYTHONPATH=. ARMS_RECIPE=main_fg python 6_edit_reasoning/n51_obs.py  # checkpoint
PYTHONPATH=. python 6_edit_reasoning/render_bucket.py --dst model
PYTHONPATH=. python 6_edit_reasoning/assemble.py --arms model --splits train val --procs 32
```

`base_dump` and `gate_dump` run on CPU at a few hundred rounds a second per worker;
they do not need the GPU the trainer is using.

## A note on the docstrings

These modules were written inside an ablation that compared several evidence blocks
against each other. The docstrings keep the measurements that decided each design
choice, and some of them name analysis scripts that are not part of this release.
The code paths themselves are self-contained; only the cross-references are not.
