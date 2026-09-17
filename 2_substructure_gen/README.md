# 2_substructure_gen

Stage 2 of the SFT pipeline — reduce each instance's molecule to its Bemis-Murcko scaffold, describe it in natural language (vLLM), verify by RDKit, and attach evaluation SMARTS.

---

## Overview

This stage sits between stage 1 (instance generation) and stage 3 (toolchain generation):

```
1_instance_gen/  →  2_substructure_gen/  →  3_toolchain_gen/
  generation_200k.jsonl  →  generation_200k_scaffold/  →  toolchain data
      (instances with ref_smiles)   (+ scaffold fields)
```

Per-molecule pipeline:

```
SMILES
 → Murcko scaffold extraction
 → ring-system decomposition
 → common ring / heterocycle name mapping
 → linker & fusion / spiro / bridge / direct-link topology
 → substituent removal + attachment-point recording
 → template-based controlled NL generation
 → LLM sentence polishing (vLLM / Qwen3)
 → faithfulness validate/revise loop
 → SMARTS / RDKit substructure-check verification
```

The driver `augment_jsonl_with_scaffold.py` is the primary entry point for instance data (1:1, order-preserving, sharded JSONL output). `describe_scaffolds.py` is the bulk de-duplicating generator for standalone SMILES pools.

---

## Description prompt v8

`scaffold_prompts.py` is the production prompt and is fixed at **v8**. Earlier experimental
prompt variants have been removed. The pre-v8 production prompt is retained only as
`prompt_var/default.py` for restoration or A/B comparison; it is not imported by the generation
pipeline.

v8 treats each description as the input brief for a later SMARTS-generation task, rather than as
a report about an already-existing structure. Its main rules are:

- Start from a named ring, linker, junction, or heteroatom pattern; do not open with
  “This scaffold …” or an equivalent whole-object subject.
- State only information present in the authoritative FACTS block. Omitting a secondary detail is
  allowed; inventing or approximating one is not.
- Use the deterministic DRAFT as a wording reference, not as a sentence template to copy.
- Do not append a closing summary, heteroatom total, or overall aromaticity tally that merely
  repeats the named parts.
- Use a uniform plain register and no more than four sentences.

The FACTS block was changed with v8. The old
`- Overall: ...; scaffold heteroatoms: ...` line was replaced by
`- Aromaticity across the rings: ...`. The old heteroatom total counted ring systems but excluded
linkers, so it was not a reliable whole-scaffold total and also encouraged redundant summary
sentences. Generated text is normalised with Unicode NFKC plus explicit punctuation folding before
validation and storage, preventing model-specific dashes, quotes, or subscript characters from
leaking into the corpus.

The rationale, measurements, and output examples are in
[`description_results/report_default_vs_v8.html`](description_results/report_default_vs_v8.html).

---

## Files

| File | Role |
|------|------|
| `augment_jsonl_with_scaffold.py` | **Primary stage-2 driver.** Reads a JSONL of instances (one record per line), carries all original keys forward unchanged, appends scaffold fields, and writes sharded output (`<prefix>-00000.jsonl`, `-00001.jsonl`, …). Sharded writing is streaming, atomic per shard, and resumable — already-complete shards are skipped on restart. |
| `generate_descriptions.py` | Convenience entry point for a description-free instance JSONL. Produces one complete, order-preserving JSONL with the same scaffold/description schema as `generation_benchmark-00000.jsonl`, using the production stage-2 implementation. |
| `describe_scaffolds.py` | Bulk SMILES-pool generator. Accepts `.parquet` / `.jsonl` / `.txt` / `.smi` input, de-duplicates by SMILES, and appends to a single `scaffolds.jsonl` (resumable by `input_smiles`). Contains `VLLMPool` (shared-queue dynamic endpoint scheduling + per-server concurrency + dead-server pruning), `iter_input_smiles`, `analyze_one`, `build_item`, and `describe_one`. Imported by `augment_jsonl_with_scaffold.py`. |
| `scaffold_analyzer.py` | Pure RDKit. Murcko extraction, ring-system decomposition, comprehensive ring naming (curated SMILES dict → systematic monocycle namer → compositional fallback), linker classification, fused/spiro/bridged/directly-linked/linker-connected topology, attachment points. Entry point: `analyze_scaffold(smiles) -> dict`. |
| `scaffold_describer.py` | Logic layer: `format_facts` (authoritative FACTS block), `render_template` (deterministic draft), `normalize_text` (model-output typography normalisation), `validate_text` (polished text ↔ facts check), and `ScaffoldDescriber` (single source of prompts/validation). No model calls. |
| `scaffold_prompts.py` | The fixed production v8 prompt: controlled-NL sentence templates plus the LLM polish `SYSTEM` / `USER` / `REVISE` / `CONDENSE` prompts. |
| `prompt_var/default.py` | Byte-identical backup of the pre-v8 production prompt. Kept for restoration and A/B comparison only; not used by production. |
| `compare_descriptions.py` | Development harness for dry-run inspection and side-by-side model/prompt comparison. Uses v8 by default; accepts `--prompts prompt_var/default.py` for the pre-v8 baseline. |
| `DATASET_README.md` | Full schema and evaluation strategy for the output dataset (Korean). Copied into the output folder by both drivers. |
| `DATASET_README.en.md` | English version of the above. |

---

## Usage

### Standard run (stage 2 of the full pipeline)

Run from the repository root:

```bash
bash run_scripts/pipeline/2_substructure_gen.sh
```

The wrapper currently defaults to `generation_2m.jsonl`, 20,000-row shards, 30 concurrent requests
per server, and automatic analyzer process count. The example below assumes that Qwen is served on
port 20000 and gpt-oss on port 20200. Each endpoint is queried through `/v1/models`, and its actual
model name is detected independently:

```bash
# Qwen endpoint:   http://localhost:20000/v1
# gpt-oss endpoint: http://localhost:20200/v1
INPUT=/path/to/instances.jsonl \
OUT_DIR=/path/to/instances_scaffold \
SHARD_SIZE=20000 \
SMILES_KEY=ref_smiles \
CONCURRENCY=30 \
ANALYZER_PROCS=0 \
SERVERS="localhost:20000,localhost:20200" \
PYTHON=python \
  bash run_scripts/pipeline/2_substructure_gen.sh
```

Do not pass a forced `--model` when different endpoints serve different models. Automatic model
detection lets the pool send each request with the model ID advertised by the selected endpoint.
This is not strict round-robin scheduling: each endpoint has
`--concurrency-per-server` persistent workers consuming one shared queue, so a faster endpoint
naturally processes more records while a slower endpoint processes fewer. Qwen and gpt-oss also
receive their respective reasoning-control parameters automatically. The output currently does
not record which endpoint/model generated each individual row.

To use replicas of the same model on other hosts, add every host/port to `SERVERS`, for example:

```bash
SERVERS="host-a:20000,host-b:20000,host-c:20200,host-d:20200" \
  bash run_scripts/pipeline/2_substructure_gen.sh
```

The output is resumable. Re-running the same command skips complete shards and resumes from the
first missing or incomplete shard. Use `--overwrite` only with a direct invocation when a complete
restart is intended.

### Complete a description-free JSONL

Use `generate_descriptions.py` when the input contains the original instance fields and a SMILES
key, but no scaffold description yet. The command preserves every input key and the exact row
order, then appends the deterministic scaffold/evaluation fields and the v8 `description`. The
result has the same schema as `../..data/benchmark/generation_benchmark-00000.jsonl`.

Run from `2_substructure_gen`:

```bash
PY= # your_python_path (ex. envs/molkit/bin/python)

$PY generate_descriptions.py \
    --input ../..data/benchmark/source_without_descriptions.jsonl \
    --output ../..data/benchmark/source_with_descriptions.jsonl \
    --servers "localhost:20000" \
    --concurrency-per-server 30 \
    --analyzer-procs 0 \
    --validate-passes 3
```

In this example one model endpoint is expected at `localhost:20000` and advertises its model
through `/v1/models`.

The SMILES field is auto-detected from `ref_smiles`, `smiles`, or `smi`. You may still pass
`--smiles-key` explicitly; a mismatched or empty field now stops immediately instead of producing
rows with empty scaffold fields. For example, `sample_2.jsonl` uses `--smiles-key smiles`.

When the source has flat rows such as `sample_2.jsonl` (`smiles`, `MW`, `logP`, `HBD`, ...), add
`--prepare-generation-schema`. In the same command, each row is first converted to the benchmark
input keys `id`, `task_type`, `properties`, and `ref_smiles`, then passed directly into the existing
scaffold and v8-description pipeline. Scalar property values become exact constraints represented
as equal `min` and `max` bounds. Source-only columns such as `chembl_id` and `source_db` are omitted
so the final key structure matches `generation_benchmark-00000.jsonl`.

```bash
$PY generate_descriptions.py \
    --input ../..data/benchmark/sample_2.jsonl \
    --output ../..data/benchmark/sample_completed.jsonl \
    --prepare-generation-schema \
    --id-prefix generation \
    --servers "localhost:20000" \
    --concurrency-per-server 30 \
    --analyzer-procs 0 \
    --validate-passes 3
```

For a large flat-property Parquet input, use the same command with Parquet paths. Processing is
streamed in bounded batches and output is split into fixed-size Parquet files:

```bash
$PY generate_descriptions.py \
    --input /path/to/sample_2.parquet \
    --output /path/to/sample_completed.parquet \
    --prepare-generation-schema \
    --id-prefix generation \
    --servers "localhost:20000" \
    --concurrency-per-server 30 \
    --analyzer-procs 0 \
    --validate-passes 3 \
    --parquet-batch-size 1000 \
    --parquet-shard-size 100000
```

With `--parquet-shard-size 100000`, the command writes
`sample_completed-00000.parquet`, `sample_completed-00001.parquet`, and so on, with exactly
100,000 rows per file except the final remainder. Use `10000` for 10K-row files or `0` to write
one `sample_completed.parquet`. `--parquet-batch-size` controls working memory independently of
the output shard size. Existing output files are never replaced.

The input and output paths must differ, and the command refuses to replace an existing output
file. It creates only the requested JSONL and does not copy `README.md` or `README.en.md` into the
output directory. For large resumable production jobs, use the sharded
`augment_jsonl_with_scaffold.py` interface below instead; that dataset-oriented command does copy
the schema README files beside its output.

### Direct invocation — augment_jsonl_with_scaffold.py (instance files)

```bash
PY=python

# Sharded output (10K rows per file) — resumable
$PY 2_substructure_gen/augment_jsonl_with_scaffold.py \
    --input   data/training_data/instances/generation_200k.jsonl \
    --out-dir data/training_data/instances/generation_200k_scaffold \
    --shard-size 10000 \
    --smiles-key ref_smiles \
    --servers "localhost:20000,localhost:20200"

# Single output file
$PY 2_substructure_gen/augment_jsonl_with_scaffold.py \
    --input  data/training_data/instances/generation_benchmark.jsonl \
    --output data/training_data/instances/generation_benchmark_scaffold.jsonl \
    --smiles-key ref_smiles \
    --servers "localhost:20000,localhost:20200"
```

### Direct invocation — describe_scaffolds.py (SMILES pools)

```bash
PY=python

# From a large parquet pool (streamed; use --limit to slice)
$PY 2_substructure_gen/describe_scaffolds.py \
    --input data/develop/chembl_zinc_split/train_pool.parquet \
    --run-name scaffold_train_pool --limit 20000 \
    --servers "localhost:20000,localhost:20200"
# -> data/training_data/scaffold_train_pool/{scaffolds.jsonl, meta.json}

# Inspect deterministic analysis / draft for one molecule (no LLM)
$PY 2_substructure_gen/scaffold_describer.py "c1ccc(Nc2ncnc3ccccc23)cc1"
```

### Inspect or compare prompt v8

Run these commands from `2_substructure_gen`:

```bash
PY=python

# No model calls: inspect the v8 FACTS, deterministic draft, and stored description
$PY compare_descriptions.py \
    --input ../..data/benchmark/generation_benchmark-00000.jsonl \
    --n 3 --facts --dry-run

# Compare v8 output from two different models, one model per endpoint
$PY compare_descriptions.py \
    --input ../..data/benchmark/generation_benchmark-00000.jsonl \
    --sample 8 --seed 0 \
    --endpoint qwen=http://localhost:20000/v1 \
    --endpoint gptoss=http://localhost:20200/v1 \
    --out description_results/v8_check.jsonl \
    --html description_results/v8_check.html

# Full 2 x 2 comparison:
#   {Qwen, gpt-oss} x {production v8, retained pre-v8 prompt}
$PY compare_descriptions.py \
    --input ../..data/benchmark/generation_benchmark-00000.jsonl \
    --sample 60 --seed 0 --kind ring --max-sentences 4 \
    --endpoint qwen=http://localhost:20000/v1 \
    --endpoint gptoss=http://localhost:20200/v1 \
    --prompts prompt_var/default.py \
    --html description_results/default_vs_v8_check.html
```

With no `--prompts` argument, the harness uses the production `scaffold_prompts.py` (v8).
`--prompts` is repeatable. Each `--endpoint NAME=URL` adds a separate model comparison column. If
the same model has multiple replicas, put their URLs in one comma-separated endpoint argument,
such as `--endpoint qwen=http://host-a:20000/v1,http://host-b:20000/v1`; all URLs grouped under one
name must serve the same model. Unicode typography is normalised by default; pass `--no-normalize`
only when inspecting raw model-specific output.

### Prerequisites

- **Conda env:** `molkit` at `$CONDA_ROOT/envs/molkit` (rdkit, openai, tqdm; pyarrow for parquet input)
- **vLLM pool:** one or more OpenAI-compatible vLLM servers. Different models may run on different endpoints. The direct Python CLIs default to `localhost:8080-8087`; the wrapper has its own `SERVERS` default and should normally be overridden for the active hosts. Check every model endpoint before a run:

  ```bash
  curl -s http://localhost:20000/v1/models  # Qwen
  curl -s http://localhost:20200/v1/models  # gpt-oss
  ```

### Key env vars / CLI flags

| Name | Default | Description |
|------|---------|-------------|
| `INPUT` / `--input` | wrapper: `generation_2m.jsonl`; CLI: required | Input JSONL (for `augment_jsonl_with_scaffold.py`) |
| `OUT_DIR` / `--out-dir` | wrapper: `generation_2m_scaffold`; CLI: required unless `--output` is used | Output folder for sharded files |
| `SMILES_KEY` / `--smiles-key` | `ref_smiles` | Key holding the SMILES in each input record |
| `SERVERS` / `--servers` | wrapper-specific / `localhost:8080-8087` | Comma-separated `host:portspec` entries; endpoints may advertise different models |
| `--endpoint` | Qwen `:20000`, gpt-oss `:20200` in the comparison harness | Repeatable `NAME=URL` model endpoint for `compare_descriptions.py`; comma-join only replicas of the same model |
| `--host` | — | Alternative to `--servers`: host(s) × `--base-ports` |
| `--base-ports` | `8080-8087` | Port range when using `--host` |
| `CONCURRENCY` / `--concurrency-per-server` | wrapper: `30`; CLI: `5` | In-flight requests per vLLM server |
| `ANALYZER_PROCS` / `--analyzer-procs` | `0` | RDKit scaffold-analysis worker processes; `0` selects `min(32, cpu_count)` automatically |
| `SHARD_SIZE` / `--shard-size` | wrapper: `20000`; CLI: `10000` | Rows per output shard |
| `--validate-passes` | `3` | Maximum LLM revise attempts after the initial deterministic faithfulness check; `0` still checks and records violations but does not revise |
| `--reasoning-effort` | `low` | Model-family-specific reasoning control (`gpt-oss` / Qwen) |
| `--offset` / `--limit` | `0` / all | Slice the input (for distributed runs) |
| `--overwrite` | false | Delete existing output and restart from scratch |

`--analyzer-procs` controls only the CPU-side RDKit work that extracts and names scaffolds and
builds evaluation SMARTS before an LLM request is queued. It does not change the number of LLM
requests. `0` uses up to 32 processes automatically; choose a smaller value when running on a
memory-constrained machine.

`--validate-passes` controls correction, not initial generation. Every generated description is
checked once against the deterministic scaffold facts. When violations are found, the model may
be asked to revise the text up to this many times, and a revision is accepted only when it reduces
the violation count. `0` disables revise calls while retaining the initial check and recording any
remaining `description_violations`.

---

## Inputs / Outputs

### Input

A JSONL file from stage 1 (`1_instance_gen/`), one record per line. Each record must contain a SMILES string under the key specified by `--smiles-key` (default `ref_smiles`). All other keys are passed through unchanged.

Auto-detected input formats for `describe_scaffolds.py`: `.parquet` (streamed via pyarrow), `.jsonl` / `.json`, `.txt` / `.smi`.

### Output

**`augment_jsonl_with_scaffold.py` (sharded mode):**

A folder of fixed-size JSONL shards: `<prefix>-00000.jsonl`, `<prefix>-00001.jsonl`, …

Each shard is written atomically (`.tmp` → `rename`) and order is preserved. On restart, complete shards (line count matches expected) are skipped. `DATASET_README.md` and `DATASET_README.en.md` are also copied into the output folder.

**`describe_scaffolds.py`:**

`<out-dir>/<run-name>/scaffolds.jsonl` (append mode, resumable by `input_smiles`) + `meta.json` (run config and stats).

### Appended scaffold fields (per output record)

| Field | Description |
|-------|-------------|
| `scaffold_smiles` | Canonical SMILES of the Bemis-Murcko scaffold |
| `scaffold_smarts` | SMARTS of the scaffold for substructure matching |
| `dimension_smarts` | Dict of per-dimension projection SMARTS: `skeleton`, `element`, `aromaticity`, `bond`, `ring` — each isolates one property with others genericized |
| `eval_query` | Evaluation spec: `verdict_match_any` (list of full SMARTS — molecule passes if it matches ANY one), `dimension_sets` (per-dimension match-any sets), `n_members`, `n_free_slots` |
| `scaffold_verified` | Boolean — RDKit confirms the scaffold SMARTS matches the input molecule |
| `description` | Natural-language description of the scaffold. Ring scaffolds are LLM-polished; functional-group frameworks use the deterministic faithful draft directly |
| `description_violations` | List of faithfulness violations that survived the revise loop (present only when non-empty) |
| `has_scaffold` | Boolean — true when a scaffold was extracted. Ring molecules use the Murcko scaffold; ring-free molecules fall back to a **functional-group framework** (anchors = heteroatoms + unsaturated carbons, plus the linkers connecting them; terminal alkyl/halide side chains stripped). Only false when no functional-group anchor exists (e.g. a saturated hydrocarbon) |
| `scaffold_kind` | `ring` (Murcko) \| `functional_group` (acyclic FG framework) \| `none` |
| `functional_groups` | For FG frameworks, the SMARTS-verified functional groups inside the framework (e.g. `["carbamate", "quaternary ammonium"]`); usually empty for ring scaffolds |
| `n_ring_systems` | Count of ring systems |
| `n_rings_total` | Total ring count |
| `aromaticity` | Aromaticity classification string |
| `ring_systems` | List of ring system dicts: `name`, `n_rings`, `aromatic`, `internal_topology`, `ring_sizes`, `heteroatoms`, and (when present) `n_ring_carbonyls`, `carbonyl_groups`, `carbonyl_desc` |
| `connections` | List of linker dicts: `type`, `linker_pattern` (e.g. `-NH-C(=S)-NH-`), `functional_groups`, `linker_length`, `bond_span`, `linker_atoms`, `a_atom`, `b_atom`, `a_position`, `b_position` |
| `linker_attachment_points` | List of scaffold atoms where substituents were stripped |
| `topology_summary` | Human-readable topology string |
| `template_draft` | Deterministic controlled-NL draft before LLM polishing |
| `decompositions` | (composite cores) Component-ring SMARTS + `pair_fusions` — used to verify multi-ring cores by "all components present AND joined as stated" |
| `input_smiles` | The SMILES as read from the input (for resume/dedup keying in `describe_scaffolds.py`) |
| `analysis_error` | Present only when RDKit scaffold analysis failed |

---

## Notes

- The `eval_query.verdict_match_any` set always includes the true scaffold as a member, so the original molecule never fails its own query. Asserted attachment points stay pinned; only unasserted ones are varied over symmetry-distinct candidates (set capped at 64 to avoid blow-up).
- The `element` dimension set leaves the in-ring heteroatom arrangement free (a description gives element kind/count, not exact positions — e.g. thiazole → {1,3-thiazole, 1,2-isothiazole}); symmetry-equivalent arrangements are collapsed by graph canonicalization.
- Substituent positions are never stated in the description — substituents are stripped from the scaffold, so no evaluation SMARTS can verify them.
- Ring naming: a curated canonical-SMILES dictionary (~135 named rings/systems) is tried first; single rings fall back to a systematic namer (size + aromaticity + heteroatom locants); any remaining fused/spiro/bridged system is described compositionally from named constituent rings.
- To restore the pre-v8 wording rules, use the four prompt strings in `prompt_var/default.py`. Its prompts are preserved exactly, but a new run will still receive the current v8 FACTS format unless `format_facts()` is also deliberately reverted.
- For the full output field schema and evaluation strategy, see [DATASET_README.en.md](DATASET_README.en.md) (English) or [DATASET_README.md](DATASET_README.md) (Korean). Both files are also copied into each output folder automatically.
