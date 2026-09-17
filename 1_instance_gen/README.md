# 1_instance_gen

Stage 1 of the SFT pipeline: build property-constrained generation-task instances from a molecule pool.

## Overview

This stage takes a property-tagged molecule pool and produces a JSONL file of **generation instances** — each instance describes a set of property constraints that a model must satisfy when proposing new molecules.

The key design is **ref-anchored constraints**: a reference molecule is sampled uniformly from the pool, and all property bounds are derived from that molecule's actual property values. This guarantees every instance is feasible (the ref molecule itself is always a valid hit) and that constraints are grounded in realistic chemical space.

Each instance contains both the property constraint specification and a **joint hit count/rate** — the number of pool molecules that simultaneously satisfy all constraints. This metadata is used downstream to filter out trivially easy or impossibly hard instances.

Stage 2 (`2_substructure_gen`) consumes the output JSONL from this stage as its primary input.

## Files

| File | Description |
|---|---|
| `build_generation_instances.py` | Main script: samples ref molecules, builds property constraints, computes joint hit statistics, writes JSONL |

## Usage

### Recommended: run via the shell script

```bash
bash run_scripts/pipeline/1_instance_gen.sh
```

The script resolves the `molkit` conda environment automatically (at `$CONDA_ROOT/envs/molkit`) and passes all default arguments.

Override defaults with environment variables:

```bash
POOL=data/pool/train_pool_props.parquet \
OUTPUT=data/training_data/instances/generation_200k.jsonl \
N=200000 Q_MIN=0.1 Q_MAX=0.5 SEED=0 \
  bash run_scripts/pipeline/1_instance_gen.sh
```

### Direct invocation

```bash
python 1_instance_gen/build_generation_instances.py \
    --pool data/pool/train_pool_props.parquet \
    --output data/training_data/instances/generation_200k.jsonl \
    --n 200000 \
    --q-min 0.1 \
    --q-max 0.5 \
    --seed 0
```

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--pool` | `data/pool/train_pool_props.parquet` | Input parquet with property columns + SMILES |
| `--output` | `data/training_data/instances/generation.jsonl` | Output JSONL path |
| `--n` | `10000` | Number of instances to generate |
| `--q-min` | `0.1` | Lower bound on the continuous-property quantile window size `q` |
| `--q-max` | `0.5` | Upper bound on the continuous-property quantile window size `q` |
| `--seed` | `0` | NumPy RNG seed for reproducibility |
| `--max-int` | `5` (all int props) | Max number of integer properties per instance |
| `--max-cont` | `9` (all cont props) | Max number of continuous properties per instance |
| `--int-offset` | `2` | Max offset from ref value for integer property bounds |
| `--hitrate-sample` | `0` (full pool) | If > 0, estimate hit-rate from a random sub-sample instead of the full pool (faster but approximate) |
| `--id-prefix` | `generation` | Prefix for instance IDs (e.g. `generation_0`, `generation_1`, ...) |

### Conda environment

Requires the `molkit` environment (`$CONDA_ROOT/envs/molkit`). Only `pyarrow` and `numpy` are needed; no RDKit or HTTP dependencies.

## Inputs / Outputs

### Input

A Parquet file with one row per molecule containing a SMILES column and all 14 property columns:

- **Integer properties** (5): `HBD`, `HBA`, `rotB`, `rings_total`, `heavy_atoms`
- **Continuous properties** (9): `MW`, `logP`, `logD`, `logS`, `TPSA`, `QED`, `BBBP`, `Mutag`, `MR`

Default path: `data/pool/train_pool_props.parquet`

### Output

A JSONL file where each line is one generation instance with the following schema:

```json
{
  "id": "generation_0",
  "task_type": "generation",
  "properties": [
    {"property": "MW",   "min": 310.0, "max": 450.1},
    {"property": "logP", "max": 3.5},
    {"property": "HBD",  "min": 1, "max": 3},
    {"property": "HBA",  "min": 4}
  ],
  "ref_smiles": "CCOc1ccc(...)cc1",
  "hit_count": 1482,
  "hit_rate": 0.00741,
  "pool_size": 200000
}
```

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique instance ID (`{id_prefix}_{i}`) |
| `task_type` | string | Always `"generation"` |
| `properties` | array | List of property constraints; each entry has `property` (name), and optionally `min` and/or `max` — a missing bound means no constraint in that direction |
| `ref_smiles` | string | SMILES of the reference molecule used to anchor the constraints |
| `hit_count` | int | Number of pool molecules satisfying all constraints simultaneously |
| `hit_rate` | float | `hit_count / pool_size` |
| `pool_size` | int | Total number of molecules in the pool |

Properties within each instance are ordered by `PROP_ORDER`: continuous properties first (`MW`, `logP`, `logD`, `logS`, `TPSA`, `QED`, `BBBP`, `Mutag`, `MR`), then integer properties (`HBD`, `HBA`, `rotB`, `rings_total`, `heavy_atoms`).

## Notes

**Sampling recipe in detail:**

1. A ref molecule is drawn uniformly from the pool.
2. A random number of integer properties `ki ~ U[1, max-int]` and continuous properties `kc ~ U[1, max-cont]` are selected.
3. `HIA` and `formal_charge` are excluded (near-constant distributions, low discriminative value).
4. **Integer constraints** — mode is chosen uniformly from `both` (two-sided), `single` (one-sided min or max), or `exact`. Bounds are offset from the ref value by a random integer in `[0, int-offset]`.
5. **Continuous constraints** — a quantile window of size `q ~ U[q-min, q-max]` is placed around the ref molecule's empirical percentile. A random straddle factor `alpha ~ U[0, 1]` controls how much of the window falls above vs. below the ref. If either window edge extends past the distribution boundary (quantile 0 or 1), that bound is dropped automatically, producing a one-sided constraint without explicit one-sided sampling. This keeps the pass-rate at most `q`.
6. Continuous bounds are rounded toward feasibility (min rounded down, max rounded up) to the property's defined decimal precision.
7. The joint hit count is computed over the full pool (or a sub-sample if `--hitrate-sample` is set). Because constraints are anchored to the ref, the ref molecule is always a valid hit, guaranteeing `hit_count >= 1`.

**Performance:** For large pools, the full joint hit computation is the bottleneck. Use `--hitrate-sample` (e.g. `50000`) for a significant speed-up at the cost of approximate hit counts.
