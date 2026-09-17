# 0_build_mmpdb

Stage 0 of the SFT pipeline — mine matched-molecular-pair (MMP) edit moves from a property-tagged molecule pool via mmpdb and export the move-set JSON files consumed by downstream toolchain generation.

---

## Overview

**What MMP move-mining is and why it matters.**
A matched molecular pair (MMP) is a pair of molecules that differ by exactly one fragment substitution at a single cut point. The key insight from the MMP literature is that substituent effects on additive physicochemical properties (logP, MW, TPSA, …) are largely scaffold-independent: if swapping fragment A → B raises logP by 0.8 in one context, it tends to do so in other contexts too. By mining MMPs from a large, property-labelled molecule pool, we obtain a data-driven catalogue of edit moves — each annotated with the mean/std property change it causes across all observed pairs — without needing to enumerate hypothetical edits by hand.

**Where this stage sits.**
This is the first stage of the SFT pipeline and runs before any instance generation:

```
0_build_mmpdb  →  1_instance_gen  →  2_substructure_gen  →  3_toolchain_gen  →  4_sftdata_gen
```

The JSON files it produces (`attach_library.json` and the per-cut swap files `single_cut.json` / `double_cut.json` / `triple_cut.json`) define the vocabulary of chemical edits available to the `edit_fragment` agent tool used throughout stages 3 and 4. `attach_library.json` doubles as the novelty source-universe of building blocks (stage 7).

---

## Files

### `build_mmp_moves.py`

The main script. Runs the full three-step pipeline:

1. **Sample** SMILES and property values from a Parquet pool, strip stereochemistry, deduplicate, and write `sample.smi` / `sample.props`.
2. **Build** the mmpdb database by invoking `mmpdb fragment` and `mmpdb index --properties` as subprocesses, producing `sample.fragments` and `sample.mmpdb` (an SQLite file).
3. **Extract** the move sets from the SQLite database and serialise them as JSON (and mirrored as flat CSV for inspection):
   - `attach_library.json` — single-attachment R-group fragments, ranked by the number of distinct scaffolds they decorate.
   - `single_cut.json` / `double_cut.json` / `triple_cut.json` — substituent swap rules A → B, one file per attachment count. Each row is a `(from → to, radius, context)` triple carrying the Δ-vector (mean, std) for every indexed physchem + ADMET property. `mmpdb index` computes environments at every radius 0..5 in one pass, so all radii are extracted together (radius 0 = context-free, higher = context-specific).

Key functions:
| Function | Purpose |
|---|---|
| `write_inputs()` | Sample parquet, strip stereo, dedup, write `.smi` / `.props` |
| `build_db()` | Shell out to `mmpdb fragment` + `mmpdb index` |
| `extract_attach_library()` | Query SQLite for single-attachment fragments, rank by context count |
| `extract_swap_rules()` | Query SQLite for swap rules across all radii 0..`max_radius` with per-context Δ-statistics; drop stereo-only swaps; return rows grouped by attachment count |
| `write_attach_csv()` / `write_swap_csv()` | Mirror the JSON outputs as flat CSVs |

### `export_mmpdb.py`

A standalone inspection utility. Given a `sample.mmpdb` SQLite file, dumps three human-readable CSVs next to it (or to `--out-dir`):

| Output | Content |
|---|---|
| `rules.csv` | One row per (rule × environment): from/to fragment, radius, #pairs, per-property avg/std/count |
| `compounds.csv` | One row per input molecule: ID, SMILES, all property values |
| `pairs.csv` | Up to 2,000 sampled matched pairs backing the rules |

This script is not part of the automated pipeline; it is run manually for debugging or analysis of the mmpdb database.

---

## Usage

### Standard run (recommended)

```bash
bash run_scripts/pipeline/0_build_mmpdb.sh
```

The script resolves Python automatically (prefers the `molkit` conda env) and passes all defaults. Override any setting with environment variables:

```bash
SRC=data/pool/train_pool_props.parquet \
OUT_DIR=data/mmp_moves \
N_SAMPLE=50000 \
NUM_JOBS=8 \
MIN_SUPPORT=10 \
MAX_RADIUS=5 \
SEED=0 \
  bash run_scripts/pipeline/0_build_mmpdb.sh
```

### Direct Python invocation

```bash
python 0_build_mmpdb/build_mmp_moves.py \
    --src      data/pool/train_pool_props.parquet \
    --out-dir  data/mmp_moves \
    --n-sample 1000000 \
    --num-jobs 8 \
    --shards   1 \
    --min-support 10 \
    --max-radius 5 \
    --seed     0 \
    --top-attach 100000 \
    --top-swap   500000
```

`--top-swap` caps the number of transforms (from → to), ranked by radius-0 support; each kept transform keeps its full radius ladder. Use `--skip-build` to skip fragmentation/indexing and re-extract from an existing `sample.mmpdb` (or `shard.*.mmpdb`) — useful after tuning `--min-support`, `--max-radius`, or `--top-swap` without re-running the expensive indexing step.

### Parallel build (`--shards N`)

`mmpdb index` is single-threaded and dominated (~80% of wall time) by the property-statistics step, which no `--num-jobs` flag touches. `--shards N` (N > 1) parallelises it:

```bash
python 0_build_mmpdb/build_mmp_moves.py \
    --src data/pool/train_pool_props.parquet \
    --out-dir data/mmp_moves \
    --n-sample 1000000 --num-jobs 48 --shards 16 --max-radius 5
```

It runs `mmpdb fragment` once, then `mmpdb fragdb_partition` to split the fragments into N files **by constant**, then `mmpdb index --properties` on every shard **concurrently**, and finally pools each shard's per-(rule, context) `(count, avg, std)` back together at extraction (pooled variance). Partitioning by constant is exact — a matched pair only forms within a shared constant, so no MMP crosses a shard boundary — so the pooled result is identical to the single-DB build (verified: support exact, Δ-avg exact, Δ-std to ddof precision). Measured ~5.3× at 8 shards (N=60k); a full 1M all-radii build drops from ~6.5 h to ~40–70 min with 16–32 shards. (`mmpdb merge` is **not** used because it discards properties by design.)

### Export / inspection

```bash
python 0_build_mmpdb/export_mmpdb.py \
    data/mmp_moves/sample.mmpdb \
    --out-dir data/mmp_moves \
    --max-rules 20000
```

### Prerequisites

- **mmpdb**: `pip install mmpdb`
- **RDKit**: must be present in the active environment
- **Recommended conda env**: `molkit` at `$CONDA_ROOT/envs/molkit` (the run script checks this location first)
- **pyarrow**: required by `build_mmp_moves.py` for reading the Parquet pool

---

## Inputs / Outputs

### Input

| Path | Description |
|---|---|
| `data/pool/train_pool_props.parquet` (default) | Property-tagged molecule pool. Must contain a `smiles` column plus all physchem and ADMET property columns listed in `--phys` / `--admet`. |

Default physchem properties indexed: `MW`, `logP`, `HBD`, `HBA`, `TPSA`, `rotB`, `rings_total`, `QED`, `MR`, `heavy_atoms`, `formal_charge`.

Default ADMET properties indexed: `logD`, `logS`, `BBBP`, `HIA`, `Mutag`.

### Intermediate (in `--out-dir`)

| File | Description |
|---|---|
| `sample.smi` | Tab-separated SMILES + compound ID for the sampled, stereo-stripped, deduplicated subset |
| `sample.props` | Tab-separated property values keyed by compound ID |
| `sample.fragments` | mmpdb fragment output |
| `sample.mmpdb` | mmpdb SQLite database (large; all rules and statistics) |

### Primary Outputs (consumed downstream)

| File | Schema | Used by |
|---|---|---|
| `attach_library.json` | `[{"fragment": "*C(=O)O", "num_heavies": int, "support": int, "contexts": int}, …]` sorted by `contexts` desc | building-block catalogue; novelty source-universe (`7_novelty_analysis`) |
| `single_cut.json` / `double_cut.json` / `triple_cut.json` | `[{"from": "[*:1]F", "to": "[*:1]Cl", "num_attachments": int, "radius": 0..5, "context": "[*:1](~*)", "context_smarts": "[#0;X1;…:1]", "support": int, "delta": {"logP": {"avg": 0.12, "std": 0.05}, …}}, …]` — grouped by transform, radii ascending, then support desc | `edit_fragment` tool in `3_toolchain_gen` and `4_sftdata_gen` |

### Inspection Outputs (optional, from `export_mmpdb.py`)

| File | Description |
|---|---|
| `attach_library.csv` | Flat version of `attach_library.json` |
| `single_cut.csv` / `double_cut.csv` / `triple_cut.csv` | Flat versions of the per-cut swap files with `radius`, `context`, and per-property `avg`/`std` columns |
| `rules.csv` | Full rule table from the SQLite DB |
| `compounds.csv` | Input compound table with property values |
| `pairs.csv` | Sample of matched pairs (up to 2,000) |

---

## Notes

**Stereochemistry stripping.**
All molecules have stereochemistry removed (`Chem.RemoveStereochemistry`) before being written to `sample.smi`. This is necessary because mmpdb's `index` step re-canonicalises fragment SMILES and, under RDKit >= 2026, hits a precondition violation in `Canon.cpp` when fragments carry directional (`/`, `\`) double bonds. Stripping stereo eliminates those bonds and prevents the crash. This is safe in practice because stereo-only swaps (where the from/to fragments differ only in stereochemistry) are discarded by `extract_swap_rules()` anyway.

**Deduplication after stereo-strip.**
Distinct stereoisomers of the same constitutional SMILES collapse to the same string after stereo removal. Keeping both as separate compounds would cause mmpdb to pair a molecule with its own copy, producing degenerate rules with near-zero Δ. `write_inputs()` deduplicates on the canonical, stereo-stripped SMILES to prevent this. `extract_swap_rules()` applies a secondary guard for any that slip through.

**Attachment-point labels are preserved in swap rules.**
Fragment SMILES in the swap files retain mmpdb's attachment-point labels (`[*:1]`, `[*:2]`, …). This is intentional: the `edit_fragment` tool uses these labels (via `anchors`) to correctly map each cut point when applying double-/triple-cut rules.

**Environment radius and context.**
`--max-radius 5` (default) extracts every radius 0..5. Radius 0 is a single context-free environment per cut count — maximal support, maximal transferability. Higher radii fan out into specific `context`s (recorded as mmpdb's pseudo-SMILES plus a matchable `context_smarts`): support drops because the matched pairs partition exactly across contexts, but the within-context Δ sharpens. Empirically the ADMET Δ-std roughly halves from radius 0 to 5 (context explains ~80% of the Δ variance), while near-additive physchem (logP, MR, MW) collapses to ~0 std by radius 2. A consumer can therefore trade support for precision by choosing the most specific context radius that matches its site, falling back to radius 0.

**No per-property `count`.**
Each `delta[prop]` stores only `{avg, std}`; the pair count is identical to the row's `support`, so the redundant per-property `count` is omitted.

**Re-running extraction only.**
If `--skip-build` is set, `build_mmp_moves.py` skips the `mmpdb fragment` + `mmpdb index` steps and reads from the existing `sample.mmpdb` in `--out-dir`. This allows fast iteration on `--min-support`, `--max-radius`, `--top-attach`, and `--top-swap` without re-fragmenting the molecule set.
