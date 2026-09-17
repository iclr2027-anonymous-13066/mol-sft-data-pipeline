# How `_SCALE` was chosen (per-property normalization)

This document explains **how the values of `_SCALE`** (`search_plan._SCALE`) were set — the
per-property divisor that normalizes the multi-objective property gap during toolchain
generation. The values used to be hand-picked and therefore unjustified; they have been
replaced by a **reproducible basis derived from the data**.

---

## 1. What `_SCALE` does

The ref-free Δ-guided forward search (`--no-use-ref`) ranks candidate edits (moves)
**before measuring them**, predicting from the mmpdb Δ matrix how much each move would
shrink the gap to the target (`_SearchSpace.expand` / `norm_gap` in
[`search_plan.py`](search_plan.py)). The multi-objective gap is defined as

```
gap(v) = Σ_p  box_distance(v_p; [lo_p, hi_p]) / scale_p
```

Without `scale_p`, a property with a large unit (MW ~ hundreds) overwhelms one with a small
unit (BBBP ~ 0–1), and the search chases size and lipophilicity alone. `scale_p` therefore
defines **"one unit of meaningful progress in property p"**, which is what makes gap
reductions comparable across properties.

> Termination and success are decided from the **measured vector**, independently of
> `scale` (`_SearchSpace.all_satisfied`, `_gap <= 0`). `scale` only affects the **search
> ranking** of moves.

---

## 2. The basis: population standard deviation

Each `scale_p` is set to the **population standard deviation of that property over the
dataset's molecules**.

```
scale_p = std_population(p)
```

Reading: dividing the gap by this scale makes **one unit of gap equal one standard
deviation of that property's natural variation**, so all properties are summed in z-score
units. This is standard practice in multivariate feature scaling and, unlike hand-picked
values, it is computed once from the data, so **anyone can reproduce it** and it is
straightforward to justify.

### Data source

The `analyze_properties` responses already stored in the toolchain records are used, so no
separate tool server is needed. Values were collected over a sample of **40,000 molecules**
from `generation_2m_scaffold` and the std computed per property (effective sample size per
property n ~ 22k-24k, because not every property appears in every molecule's analyze call).

### Robust alternatives

For heavy-tailed properties (logS, MR, ...) one could use `1.4826*MAD` or `IQR/1.349`
instead of the std. Both are reported in the table below; they agree with the std in
direction and magnitude, so the std was adopted as is.

---

## 3. Resulting values (40,000 molecules)

| prop | n | std | MAD | IQR | **adopted scale** | previous (hand-picked) |
|---|---:|---:|---:|---:|---:|---:|
| MW | 22190 | 85.96 | 54.95 | 109.05 | **86.0** | 20.0 |
| logP | 22276 | 1.26 | 0.84 | 1.67 | **1.26** | 0.5 |
| HBD | 24072 | 0.85 | 1.00 | 1.00 | **0.85** | 1.0 |
| HBA | 24149 | 1.62 | 1.00 | 2.00 | **1.62** | 1.0 |
| TPSA | 22080 | 26.01 | 17.93 | 36.56 | **26.0** | 10.0 |
| rotB | 23949 | 1.93 | 1.00 | 3.00 | **1.93** | 1.0 |
| rings_total | 23962 | 1.08 | 1.00 | 1.00 | **1.08** | 1.0 |
| QED | 22182 | 0.149 | 0.104 | 0.222 | **0.15** | 0.05 |
| MR | 22172 | 23.93 | 15.16 | 30.35 | **24.0** | 6.0 |
| heavy_atoms | 24154 | 6.24 | 4.00 | 8.00 | **6.24** | 1.5 |
| formal_charge | 0 | — | — | — | **1.0** (no data, natural unit kept) | 1.0 |
| logD | 22093 | 1.36 | 0.88 | 1.78 | **1.36** | 0.5 |
| logS | 22325 | 1.56 | 1.10 | 2.21 | **1.56** | 0.5 |
| BBBP | 22243 | 0.125 | 0.048 | 0.124 | **0.125** | 0.1 |
| HIA | 0 | — | — | — | **0.1** (no data, previous value kept) | 0.1 |
| Mutag | 22118 | 0.188 | 0.118 | 0.254 | **0.19** | 0.1 |

**Main change**: the scale of the wide continuous properties (MW / MR / TPSA / heavy_atoms /
QED / logD / logS) grew 3-4x relative to the previous values. The old values underestimated
these properties relative to their std, which over-weighted them in the gap sum. The integer
count properties (HBD / HBA / rotB / rings_total) land naturally near 1 and barely move.

**Two exceptions**: `formal_charge` and `HIA` are not part of the analyze calls in this
dataset, so their sample is empty. `formal_charge` keeps the natural unit of an integer
count, `1.0`, and `HIA` keeps its previous value, `0.1`.

---

## 4. A/B check — the change is performance-neutral

To confirm the new scale does not hurt hit rate, an A/B was run with the ref-free search
(identical inputs, seed and budget; n=300; previous values vs. new std values).

| mode | previous scale | std scale | delta | verdict |
|---|---:|---:|---:|---|
| greedy | 35.3% | 34.3% | -1.0pp | no difference (88% overlap, 19 vs 16 disagreements, McNemar chi2 ~ 0.26) |
| beam (4x6) | 59.7% | 59.3% | -0.4pp | no difference (90% overlap, 15 vs 14 disagreements, McNemar chi2 ~ 0) |

- In both modes the previous values and the new std values are **statistically
  indistinguishable** (the disagreements are a symmetric reshuffle).
- The real lever on hit rate is the **search mode**, not the scale (greedy to beam is +24pp).
- **Conclusion**: this was adopted for a principled, reproducible basis rather than for a
  performance gain, and it costs nothing in performance.

> Why it is insensitive: greedy commits only the top-1 move, and the dominant move is
> usually the same under both scales; beam measures its candidates and then selects on the
> **measured gap**, so the scale only perturbs the ranking of what gets measured. The
> acceptance criterion is the measured vector, which does not change.

---

## 5. Reproducing the table

```bash
# recompute std/MAD/IQR per property and print the table (no tool server needed)
python 3_toolchain_gen/compute_scale_basis.py \
    --dir data/training_data/toolchain/generation_2m_scaffold \
    --n 40000
```

### Overriding for experiments (trying another scale without editing the source)

`search_plan._SCALE` can be overridden through the `TOOLCHAIN_SCALE_JSON` environment
variable (a JSON string or a path to a JSON file; only the properties named there are
overridden, the rest keep their defaults). When it is unset, the defaults above are used.

```bash
TOOLCHAIN_SCALE_JSON='{"MW":100,"logP":1.0}' python -m 3_toolchain_gen --no-use-ref ...
# or
TOOLCHAIN_SCALE_JSON=/path/to/scale.json python -m 3_toolchain_gen --no-use-ref ...
```

---

## 6. Keeping the mirror in sync

`_SCALE` lives in [`3_toolchain_gen/search_plan.py`](search_plan.py), where it guides the
search. The evaluation harness keeps its own mirror of the same table; if one side is
changed the other has to be updated to the same values, or the property-constraint
efficiency metric stops being consistent with the objective used at generation time.
