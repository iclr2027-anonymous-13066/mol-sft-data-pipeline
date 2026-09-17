# run_scripts

Bash entry-point scripts for the SFT work, each reading its configuration entirely
from environment variables. Split by what a script is *for*:

| | |
|---|---|
| **`pipeline/`** | produces the training corpus — stages 0→4, plus the filters and repairs that rewrite it. Running these changes what the model is trained on. |
| **`analysis/`** | measures, inspects and evaluates. Running these changes nothing the model sees; they only write reports under `data/analysis/` or HTML. |


---

## Overview

These scripts are the recommended way to invoke the SFT data pipeline end-to-end.
They wrap the underlying Python modules with sensible defaults, explicit Python-binary
resolution, and clear progress output.

**Pipeline chain (data generation)**

```
0_build_mmpdb.sh          (precursor — build MMP move sets; suggest_edits reads these)
        |
1_instance_gen.sh         Stage 1 — sample property-constrained task instances
        |
2_substructure_gen.sh     Stage 2 — add scaffold descriptions via vLLM
        |
3_toolchain_gen.sh        Stage 3 — build ground-truth tool chains
        |
4_sftdata_gen.sh          Stage 4 — convert tool chains to SFT training data
        |
4b_filter_test_scaffold_leak.sh (optional) assert no benchmark-scaffold leak
        |
4c_make_satisfied_only.sh       (optional) write a satisfied-only copy for training

4d_qc_authored.sh               (check) QC the authored-edit rounds, if enabled
```

Every script defaults to the **2M** run: pool → `generation_2m.jsonl` →
`generation_2m_scaffold/` → `toolchain/generation_2m_scaffold/` →
`sftdata/generation_2m_scaffold/`. Point the env vars elsewhere for a smoke run.

`analysis/3_render_toolchains_html.sh` renders tool-chain chunks as a
self-contained HTML page; `4_sftdata_gen/view_constructive_sftdata.py` does the
same for generated SFT records. Neither is a required pipeline step.


---

## Scripts

### `pipeline/` — builds the training corpus

| Script | Stage | What it runs |
|---|---|---|
| `0_build_mmpdb.sh` | 0 | `0_build_mmpdb/build_mmp_moves.py` — mines matched molecular pairs from a property-tagged parquet pool; writes `attach_library.json` + per-cut `single_cut.json` / `double_cut.json` / `triple_cut.json`, each row carrying its radius 0–5 context and Δ-vector. `mmpdb` + RDKit; `benchmark` env |
| `1_instance_gen.sh` | 1 | `1_instance_gen/build_generation_instances.py` — samples ref molecules and emits property-constrained task instances as JSONL. pyarrow + numpy; no services |
| `2_substructure_gen.sh` | 2 | `2_substructure_gen/augment_jsonl_with_scaffold.py` — Murcko scaffold extraction, ring analysis, LLM description polish, RDKit verification. Needs a vLLM pool serving `Qwen/Qwen3.6-27B`; `SERVERS` is **not** probed, so keep it equal to what is running |
| `2b_functional_group_gen.sh` | 2b | The functional-group sibling of stage 2 — GPU-free, 61 fixed patterns → catalog-level descriptions. Separate constraint, separate output folder |
| `3_toolchain_gen.sh` | 3 | `python -m 3_toolchain_gen` — from the completed-scaffold seed, tunes properties with `suggest_edits` → `edit_fragment` rounds against the scaffold SMARTS and property box. Tools run **in-process** across `NUM_PROCS` workers; no servers needed |
| `3b_repair_suggest_responses.sh` | 3 fix | Recomputes the `suggest_edits` responses stored in FINISHED chain files. **Not** part of a fresh run — a one-off migration for chain sets built before `suggest_edits._measure` was fixed; a no-op audit otherwise |
| `34_fg_pipeline.sh` | 3+4 | One command end-to-end for the functional-group instance set (calls `3_toolchain_gen.sh` then `4_sftdata_gen.sh`) |
| `4_sftdata_gen.sh` | 4 | `python -m 4_sftdata_gen` — converts tool chains into chat-format SFT records: one record per chain — derivational seed round, one edit round per segment, terminal `<ANSWER>` for fully-satisfied chains only. Needs vLLM for `Qwen/Qwen3.6-27B` (dead URLs are probed out) |
| `4a_filter_authored_chains.sh` | 4 pre | "+authored" arm only — shrinks the stage-4 *input* to chains that will actually produce an authored round, before spending GPU on them |
| `4b_filter_test_scaffold_leak.sh` | 4 post | Removes test-set scaffold leakage from the stage-4 output |
| `4c_make_satisfied_only.sh` | 4 post | Builds the satisfied-only copy of the SFT data (ablation arm) plus its matching gen-eval instance subset |
| `4e_make_authored_delta.sh` | 4 post | Builds the additive delta for the "+authored" arm, re-keyed so it merges into the base without crossing the train/val split |


---

## Usage

Run scripts from the repository root. Each script resolves its Python binary
automatically (see Notes), but you can always override with `PYTHON=/path/to/python`.

### Recommended run order

**Step 0 — Build MMP move sets** (optional; only needed if you want data-driven
attach/swap rules for tool-chain generation)

```bash
SRC=data/pool/train_pool_props.parquet \
OUT_DIR=data/mmp_moves \
N_SAMPLE=50000 NUM_JOBS=8 \
  bash run_scripts/pipeline/0_build_mmpdb.sh
```

Key env-var overrides:

| Variable | Default | Description |
|---|---|---|
| `SRC` | `.../train_pool_props.parquet` | Property-tagged molecule parquet |
| `OUT_DIR` | `.../mmp_moves` | Output directory for move-set JSONs |
| `N_SAMPLE` | `1000000` | Molecules to sample for MMP mining |
| `NUM_JOBS` | `8` | Parallel mmpdb jobs |
| `MIN_SUPPORT` | `10` | Minimum pair count per rule/context |
| `MAX_RADIUS` | `5` | Include environment radii 0..MAX_RADIUS (mmpdb indexes all; 0 = context-free) |
| `SHARDS` | `1` | `>1` → parallel build (partition by constant + concurrent `index --properties`, pooled at extraction; ~5× faster). `1` = single-DB path |
| `SKIP_BUILD` | `0` | `1` → reuse existing `sample.mmpdb` / `shard.*.mmpdb` and only re-extract move sets (fast; for tuning `MIN_SUPPORT`/`MAX_RADIUS`) |

---

**Step 1 — Generate task instances**

The 2M run needs no overrides:

```bash
bash run_scripts/pipeline/1_instance_gen.sh
```

For a smoke run, shrink `N` and redirect `OUTPUT`:

```bash
OUTPUT=data/training_data/instances/generation_5k.jsonl \
N=5000 SEED=0 \
  bash run_scripts/pipeline/1_instance_gen.sh
```

Key env-var overrides:

| Variable | Default | Description |
|---|---|---|
| `POOL` | `.../instances/pool_split/train_pool_sfttrain.parquet` | Property-tagged pool. This is the **SFT-train half** of the held-out pool split; benchmark scaffolds are already removed from it (`train_pool_split_meta.json` rule_1), so do not point it back at the undivided `pool/train_pool_props.parquet` |
| `OUTPUT` | `.../instances/generation_2m.jsonl` | Output JSONL path |
| `N` | `2000000` | Number of instances to generate |
| `Q_MIN` / `Q_MAX` | `0.1` / `0.5` | Property quantile range for constraints |
| `ID_PREFIX` | `generation` | Prefix for instance IDs |
| `SEED` | `1` | Random seed |
| `WORKERS` / `CHUNK_SIZE` | `64` / `500` | Parallelism; reproducibility depends on `(SEED, CHUNK_SIZE, N)` only, not `WORKERS` |

Constraints are drawn from **14** properties. `HIA` and `formal_charge` are columns
in the pool but never become constraints (near-constant distributions), which is why
the 14 match what `analyze_properties` exposes.

---

**Step 2 — Add scaffold descriptions**

Start the vLLM pool and confirm it before running:
```bash
# one server per GPU (or per tensor-parallel group); repeat with a different
# --port for each, and list every resulting host:port in SERVERS
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3.6-27B --served-model-name Qwen/Qwen3.6-27B \
  --trust-remote-code --gpu-memory-utilization 0.9 --max-model-len 131072 \
  --enable-prefix-caching --host 0.0.0.0 --port 8080
curl -s http://localhost:8080/v1/models
```

```bash
bash run_scripts/pipeline/2_substructure_gen.sh
```

Key env-var overrides:

| Variable | Default | Description |
|---|---|---|
| `INPUT` | `.../generation_2m.jsonl` | JSONL produced by step 1 |
| `OUT_DIR` | `.../generation_2m_scaffold` | Output shard directory |
| `SHARD_SIZE` | `20000` | Rows per output shard (2M → 100 shards, which is also step 3's parallel work unit) |
| `SMILES_KEY` | `ref_smiles` | Field name holding the molecule SMILES |
| `CONCURRENCY` | `30` | Concurrent requests per vLLM server |
| `ANALYZER_PROCS` | `0` (auto) | CPU processes for the deterministic scaffold analysis |
| `SERVERS` | `localhost:8080,8082,8084,8086` | Comma-separated `host:port[-port]` pool spec. **Not reachability-probed** — a dead entry costs a timeout on every record, so keep it equal to what is running |

---

**Step 3 — Build tool chains**

By default the tools run in-process (`NUM_PROCS` worker processes), so no tool servers are needed. Only for the HTTP path (`NUM_PROCS=1 LOCAL_TOOLS=0`) must the chemistry tool servers be running first.

```bash
bash run_scripts/pipeline/3_toolchain_gen.sh
```

The trajectory is planned entirely from the description + properties by the Δ-guided
forward search; `ref_smiles` is never used. Work is assigned **per input file**, so
keep `NUM_PROCS` at or below the shard count (100 for the 2M run).

Key env-var overrides:

| Variable | Default | Description |
|---|---|---|
| `INPUT` | `.../generation_2m_scaffold` | Scaffold-augmented JSONL directory (step 2 output) |
| `OUTPUT_DIR` | `.../toolchain` | Output root; a subdir named after `INPUT` is created inside |
| `OUTPUT_NAME` | _(derived from `INPUT`)_ | Override the output subdir name |
| `CHUNK_SIZE` | `20000` | Records per final output chunk (2M → 100 chunks) |
| `PART_SIZE` | `1000` | Parallel work unit / incremental-write granularity |
| `LIMIT` | _(all)_ | Cap the number of instances processed |
| `SEARCH_MODE` | `beam` | `greedy` / `beam` / `hybrid` decorate strategy |
| `BEAM_WIDTH` / `BEAM_EXPAND` / `SUGGEST_TOP_K` | `4` / `4` / `4` | Keep these equal so the candidate count shown in the SFT data == the count the search explored |
| `NUM_PROCS` / `PER_PROC` | `48` / `4` | In-process workers (>1 runs tools locally, no servers). Throughput plateaus at 32–48; ≥96 thrashes ADMET startup. `NUM_PROCS=1 LOCAL_TOOLS=0` selects the HTTP path |
| `GPUS` | `0,1,2,3` | GPUs to round-robin the per-worker ADMET models over. The work is CPU-bound (GPU ~50% at best), so this only spreads model memory |
| `DIRECT_SCAFFOLD_SEED` | `1` | `0` selects the legacy 2-way-checkpoint format; it plans the same decorate steps |

Rough cost: hybrid ≈ 9.5 inst/s ≈ 2.4 days for 2M; greedy is 3–4× faster at a lower
satisfied rate; `beam` sits between them.

---

**Step 4 — Generate SFT training data**

Only the vLLM servers need to be up — the chemistry tool servers are NOT required (the pipeline reuses the pre-computed `expected_response` fields).

Run this **after step 3 has fully finished**, including its part→chunk merge: the
pipeline reads only `toolchains_*chunk_*.jsonl`, so a dir still holding
`toolchains_generation_part_*.jsonl` means step 3 is mid-run. The script
preflights this (and the vLLM fleet) and aborts instead of producing nothing.

```bash
bash run_scripts/pipeline/4_sftdata_gen.sh
```

**Regenerating over an existing output dir.** `--resume` skips a chunk whenever a
file of the same name already exists, and chunk names repeat across toolchain
regenerations — so output left over from a PREVIOUS toolchain generation would be
silently kept and mixed into training. The script aborts when it finds output
JSONL older than the current tool chains; move it aside first (mirroring the
`toolchain/…_v1` convention, which keeps the train configs' path valid):

```bash
mv data/training_data/sftdata/generation_2m_scaffold \
   data/training_data/sftdata/generation_2m_scaffold_v1
```

Key env-var overrides:

| Variable | Default | Description |
|---|---|---|
| `INPUT_ROOT` | `.../toolchain` | Tool-chain root directory (step 3 output) |
| `OUTPUT_ROOT` | `.../sftdata` | Output root for SFT records |
| `BASE_NAMES` | `generation_2m_scaffold` | Space-separated chain-set subdirs to process (same variable as `3b` / `4b` / `4c`) |
| `SUFFIXES` | `plain` | Variant subdirs applied on top of each `BASE_NAMES` entry |
| `KEEP_UNSATISFIED` | `1` | Keep chains that miss a constraint (they stop after the last tool response, with no `<ANSWER>`); `0` drops them |
| `AUTHORED_EDIT_FRACTION` | `0` (off) | Fraction of **eligible** decorate rounds rendered with an empty `suggest_edits` response, edit reasoning re-derived without a candidate list. See "Authored-edit branch" below |
| `AUTHORED_MAX_FAILING` | `2` | Max out-of-range properties for a round to be eligible for the above |
| `SHARDS` | `2` | Worker processes over the same dir, splitting chunks by index. 2 measured ~14% faster than 1; 3+ buys nothing on a 4-server fleet |
| `BATCH_SIZE` | `640` | Max concurrent inputs in flight (client-side) |
| `LLM_CONCURRENCY_PER_SERVER` | `40` | Max concurrent requests per vLLM server; total in-flight ≈ this × #`VLLM_URLS` |
| `NUM_GENERATIONS` | `1` | Number of alternative completions per instance |
| `VLLM_URLS` | `localhost:8080,8082,8084,8086` | Comma-separated vLLM endpoint URLs. Every URL is reachability-probed and the dead ones are dropped, so an over-long list only costs a startup warning |
| `VLLM_MODEL` | `Qwen/Qwen3.6-27B` | Model name served by vLLM |
| `SKIP_VLLM_CHECK` | `0` | `1` skips the fleet reachability preflight |
| `STALE_OK` | `0` | `1` keeps pre-existing output files and resumes anyway |
| `PYTHON_BIN` | `.../benchmark/bin/python` | Explicit Python binary path |

---

### Authored-edit branch (optional, off by default)

At inference `suggest_edits` returns an **empty list** on 64–80% of calls, and the
model collapses when it does: every training round pairs a non-empty candidate list
with *"Candidate #N is the top-ranked choice"*, so the trained program has no branch
for `[]`. It hallucinates a candidate and then emits malformed tool-call JSON (56–60%
of the collapses) or runs out of tokens mid-call (40–44%) — about 19% of val rollouts,
scored `no_action`.

Setting `AUTHORED_EDIT_FRACTION > 0` installs the missing branch. For a selected
round, the `suggest_edits` **call** is unchanged, its **response** renders as `[]`, and
the edit reasoning is re-derived from the property gap alone. The committed
`edit_fragment` and everything downstream stay ground truth.

Always send it to a **separate `OUTPUT_ROOT`** so the two corpora sit side by side:

```bash
OUTPUT_ROOT=data/training_data/sftdata_authored \
AUTHORED_EDIT_FRACTION=0.5 \
AUTHORED_MAX_FAILING=2 \
  bash run_scripts/pipeline/4_sftdata_gen.sh

```

`AUTHORED_EDIT_FRACTION` is a fraction of **eligible** rounds, not of all rounds.
Measured on chunk 0, 24% of decorate rounds pass the `AUTHORED_MAX_FAILING` filter,
so `0.5` renders ~12% of decorate rounds with an empty list. That ratio is the main
thing to tune — it deliberately does *not* match the 64–80% inference rate, because
flooding the corpus risks the opposite failure (the model authoring a rule even when
the tool *did* return candidates). `AUTHORED_MAX_FAILING=2` keeps the selection near
the real `[]` state profile: those states have 1 failing property 53% of the time and
2 33% of the time, and never 4 or more, whereas an average decorate round has 3.6.

**Caveat.** These are *not* real stuck states — the tool did return candidates and we
hid them. It teaches "when the list is empty, write a rule instead of collapsing"; it
does **not** teach finding a good rule at a genuinely stuck state. Only a planner-side
escape does that, and that needs a stage-3 rerun.

---

**Step 4b–4c — Optional post-processing**

Both default to `NAMES` = `generation_2m_scaffold` and are report-only until you opt
in with `APPLY=1` (4b) or by dropping `DRY_RUN` (4c).

```bash
# 4b — assert no benchmark-scaffold leak. For the 2M run this should report ~0:
#      the pool split already dropped every benchmark scaffold, so a non-zero
#      count means that assumption broke, not that the filter is doing its job.
bash run_scripts/pipeline/4b_filter_test_scaffold_leak.sh                  # report
APPLY=1 bash run_scripts/pipeline/4b_filter_test_scaffold_leak.sh          # rewrite in place

# 4c — write a satisfied-only COPY (…_satonly) for training; non-destructive
DRY_RUN=1 bash run_scripts/pipeline/4c_make_satisfied_only.sh              # count only
bash run_scripts/pipeline/4c_make_satisfied_only.sh                        # write the copy
```

**4b — `4b_filter_test_scaffold_leak.sh`** — drops SFT records whose scaffold also
appears in the eval set. Writes a leak-ID cache so a re-run is cheap.

| Variable | Default | Description |
|---|---|---|
| `NAMES` | `generation_2m_scaffold` | Space-separated SFT subdirs under `OUTPUT_ROOT` to scan. Pass several to clean a dir and its `_satonly` copy together |
| `OUTPUT_ROOT` | `.../sftdata` | SFT data root |
| `APPLY` | `0` | `0` reports only; `1` rewrites the shards in place |
| `TEST_FILE` | `.../benchmark_exact_subset100.jsonl` | Eval set to check against. A different file gets its own cache entry |
| `INSTANCES_DIR` | `.../instances/generation_2m_scaffold` | Instance records used to map SFT groups back to scaffolds |
| `MIN_HEAVY` / `MIN_RINGS` | `0` / `0` | Ignore scaffolds smaller than this. Raise to skip trivially-shared cores (e.g. a bare benzene) that are not real leaks |
| `PROCS` | `48` | Worker processes |
| `CACHE_DIR` | `$OUTPUT_ROOT/.testleak_cache` | Where the computed leak-ID set is cached |
| `FORCE_INFLIGHT` | `0` | `1` proceeds even while stage 4 is still writing |

```bash
# clean the mixed dir AND its satonly copy in one pass, ignoring tiny shared cores
NAMES="generation_2m_scaffold generation_2m_scaffold_satonly" \
MIN_HEAVY=10 APPLY=1 bash run_scripts/pipeline/4b_filter_test_scaffold_leak.sh

# check against a different eval set (gets its own cache entry)
TEST_FILE=data/training_data/instances/other_test.jsonl \
  bash run_scripts/pipeline/4b_filter_test_scaffold_leak.sh
```

**4c — `4c_make_satisfied_only.sh`** — writes `<name>_satonly` beside `<name>`, then
samples the train-set gen-eval instance subset the ablation configs point at. Never
modifies the source. Run it **after** 4b, which rewrites the source shards in place.

| Variable | Default | Description |
|---|---|---|
| `NAMES` | `generation_2m_scaffold` | Space-separated SFT subdirs to convert |
| `DST_SUFFIX` | `_satonly` | Suffix of the copy. Change it to keep two variants side by side |
| `DRY_RUN` | `0` | `1` counts what would be kept and writes nothing |
| `OVERWRITE` | `0` | `1` reconverts every chunk instead of only new/stale ones |
| `KEEP_UNSAT_SEED` | `1` | `1` keeps the SEED segment of unsatisfied chains (the SMARTS derivation, correct however the chain ended); `0` drops those chains entirely |
| `INCLUDE_UNFINISHED` | `0` | `1` also reads chunks with no `.done` marker — use only when stage 4 has been interrupted for good |
| `SKIP_TRAIN_SUBSET` | `0` | `1` does step 1 (the copy) and skips the instance-subset sampling |
| `TRAIN_SUBSET_N` | `100` | Instances to sample for `<name>_satonly-trainsubset<N>.jsonl` |
| `SUBSET_SEED` | `42` | Sampling seed. Keep fixed, or every rerun changes the eval set |
| `NUM_PROC` / `ARROW_NUM_PROC` | `16` / `48` | Conversion / Arrow-sidecar parallelism |
| `SKIP_ARROW` | `0` | `1` skips the `_records.arrow` sidecar (training still works, startup is slower) |

```bash
# see the keep rate before committing to the copy
DRY_RUN=1 bash run_scripts/pipeline/4c_make_satisfied_only.sh

# satisfied chains ONLY — no seed-segment salvage from the failures
KEEP_UNSAT_SEED=0 bash run_scripts/pipeline/4c_make_satisfied_only.sh

# convert the authored corpus, and take a 200-instance train subset
OUTPUT_ROOT=data/training_data/sftdata_authored \
TRAIN_SUBSET_N=200 bash run_scripts/pipeline/4c_make_satisfied_only.sh

# a settings change (e.g. KEEP_UNSAT_SEED) makes every existing chunk stale
KEEP_UNSAT_SEED=0 OVERWRITE=1 bash run_scripts/pipeline/4c_make_satisfied_only.sh
```

`3b_repair_suggest_responses.sh` is **not** part of a fresh run: it is a one-off
backfill that recomputes `suggest_edits` responses in chain sets generated before
`suggest_edits._measure` was fixed. On freshly generated 2M chains it rewrites
nothing, so run it only to migrate an older chain set (or as a no-op audit).

**Inspect the SFT records (optional)**

```bash
python 4_sftdata_gen/view_constructive_sftdata.py \
    --input data/training_data/sftdata/generation_2m_scaffold \
    --out /tmp/sftdata.html --limit 50
```

---

## Notes

**Python resolution.** Each script follows the same priority order: `PYTHON` env var
(if set) → known conda env path (`molkit` or `benchmark` depending on the
stage) → `python3` on `PATH`. Set `PYTHON=/path/to/python` to override for any script.

**Resumability.** Steps 2, 3, and 4 are all resumable — re-running after an
interruption skips already-completed shards or chunks and continues from where the
run stopped. Step 0 and step 1 produce a single file and will overwrite on re-run.

That same resume logic is a hazard when REGENERATING: chunk/shard names repeat across
runs, so output left from an earlier generation is silently kept and mixed in. Step 4
preflights this and aborts; steps 2 and 3 do not. Before regenerating over a path that
already holds output, move the old directories aside:

```bash
cd data/training_data
for d in instances/generation_2m_scaffold toolchain/generation_2m_scaffold \
         sftdata/generation_2m_scaffold; do
  [ -e "$d" ] && mv "$d" "${d}_v1"
done
mv instances/generation_2m.jsonl instances/generation_2m_v1.jsonl 2>/dev/null || true
```

**Tool servers.** Step 3 runs the chemistry tools **in-process** by default (no
servers needed); only the HTTP path (`NUM_PROCS=1 LOCAL_TOOLS=0`) needs them —
refer to the setup instructions in the repository root for how to start them. Step
4 does NOT need the tool servers (it reuses the pre-computed `expected_response`
fields). Steps 2 and 4 require the vLLM pool (scaffold description polish / reasoning
generation).

**Cross-references.**
- Stage implementations: `0_build_mmpdb/`, `1_instance_gen/`,
  `2_substructure_gen/`, `3_toolchain_gen/`, `4_sftdata_gen/`
