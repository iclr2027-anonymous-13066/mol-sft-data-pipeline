# 3_toolchain_gen

Stage 3 of the SFT pipeline: build ground-truth tool-call trajectories ("tool chains") for each scaffold-constrained generation instance.

---

## Overview

A **tool chain** is an ordered sequence of tool calls that a molecular generation agent would make to solve one instance from the scaffold-generation benchmark. Each record encodes: the user prompt (scaffold description + property ranges), the target molecule (`ref_smiles`), the full step-by-step trajectory, and verification results. These records are consumed by stage 4 (`4_sftdata_gen`) to produce the final fine-tuning dataset.

### What a tool chain looks like

With `--direct-scaffold-seed` (default) the chain **starts from the completed scaffold** and teaches only property tuning. It is a strict `checkpoint → (suggest_edits → edit → checkpoint)*` alternation, where every **checkpoint is a single 3-way parallel call** `match_substructure ∥ analyze_properties ∥ label_atom_indices`:

1. **Seed** — the seed is the completed scaffold (the Bemis–Murcko framework the description names). It is immediately followed by the seed checkpoint: `match_substructure` confirms the scaffold matches `scaffold_smarts`, `analyze_properties` records the baseline, and `label_atom_indices` grounds the atom indices the FIRST decorate edit will target.

2. **Decorate phase** — each property-tuning round is a `suggest_edits` call (ranks mmpdb-derived candidate edits toward the target property box, each already an `edit_fragment`-ready `{from_smiles, to_smiles, anchors}` triple) immediately followed by the `edit_fragment` that commits the chosen candidate (one matched-molecular-pair swap, pinned to a site/orientation via `anchors`). Each edit is followed by the SAME 3-way checkpoint on the edited molecule. The checkpoint's `label_atom_indices` companion grounds the NEXT edit — so the label that used to be a separate pre-edit step now rides on the previous checkpoint. This teaches a measure→suggest→edit→re-measure loop.

3. **Finalize / verify** — the LAST checkpoint's `match_substructure` + `analyze_properties` results drive the constraint evaluation, and the final molecule is emitted as the `<ANSWER>` in the user-facing trajectory.

`--no-direct-scaffold-seed` restores the legacy incremental assembly (2-way `match_substructure ∥ analyze_properties` checkpoints, a separate `label_atom_indices` before every edit). Note that with the current single edit primitive (`edit_fragment`) the old step-by-step scaffold-*building* path is no longer expressible, so legacy assembly only handles `edit_fragment` decorate edits and skips any instance whose plan still contains removed scaffold-building tools.

### How chains are built

The trajectory plan is computed offline by `search_plan.plan_search`, which decomposes `ref_smiles` into a hub-seeded, atom-map-tracked, phase-tagged edit sequence. In direct-scaffold mode the builder then takes the plan's `scaffold_smiles` as the seed and **keeps only the non-scaffold (decorate + finalize) edits** — the scaffold-building edits are dropped. This is safe because the planner chains `step[i]["result"]` byte-identically to `step[i+1]["arguments"]["mol_smiles"]` (both canonical) and `label_atom_indices` re-emits with `canonical=False`, so the label folded into checkpoint *i* grounds edit *i+1*'s `atom_index` exactly as a dedicated pre-edit label would. All tool calls in a single chain are dispatched concurrently via `asyncio.gather` (the plan is precomputed, so no call depends on another's response), then reassembled in order. The builder calls the **tool servers** over HTTP — the same FastAPI servers the agent uses at inference — so every `expected_response` in the stored chain is a real server response, not a mock.

After assembly, the builder evaluates the final `match_substructure` and `analyze_properties` responses against the instance's `scaffold_smarts` and property ranges, records per-property satisfaction (strict — measured value inside `[lo, hi]`), and writes the result to the output JSONL.

### Planning mode (`--use-ref` / `--no-use-ref`)

| Mode | How the plan is derived |
|---|---|
| `--use-ref` (default) | Plans from `ref_smiles` via `plan_reconstruction`. Guaranteed to terminate at the gold molecule. The chain still reads as a derivation from the description + properties at inference time. |
| `--no-use-ref` | Ref-free `suggest_edits`-guided planner (`search_plan.py`). From the completed scaffold seed (`scaffold_smiles`, which embeds `scaffold_smarts` by construction) it runs the exact two-tool loop the agent runs at inference: measure the real property vector with `analyze_properties`, call `suggest_edits` to rank the mmpdb-derived moves by predicted reduction in the normalised distance to the target box, then commit one candidate with `edit_fragment` (the matched-molecular-pair swap engine in `molkit.utils.molecule_edit_utils`) whose product still embeds the strict scaffold SMARTS (the *scaffold guard*), and re-measure. `--search-mode greedy` (default, `plan_search`) commits the single top-ranked guard-passing candidate each step (~1 measurement/step); `--search-mode beam` (`beam_search`) keeps `--beam-width` partial molecules, expands each into `--beam-expand` candidates, **measures them all and selects by the real gap** — higher hit rate, ~`width×expand`× the measurements; `--search-mode hybrid` runs greedy first and beam-rescues only the failures. The final molecule is what the search constructs, not the gold ref. `--search-max-steps` caps the decorate budget. |

When `--use-ref` planning finds no edit path (e.g. the seed is already the full scaffold), the builder falls back to an **anchor** strategy: it places the seed, skips the edit phase, and verifies the reference directly.

**Ref-free move set + scaffold guard.** The mmpdb-derived move set is owned by the `suggest_edits` tool (`molkit.utils.suggest_edits`), which keeps its own disk-backed move cache loaded once from `data/mmp_moves` (override with `MMP_MOVES_DIR`) — the per-cut swap files (`single_cut.json` / `double_cut.json` / `triple_cut.json`) built by stage 0, each row carrying a dense Δ-vector over the 16 indexed properties. `suggest_edits` ranks candidates by predicted normalised gap reduction and returns each as an `edit_fragment`-ready `{from_smiles, to_smiles, anchors}` triple; when `scaffold_smarts` is passed it only returns scaffold-preserving edits. But **termination and success are decided by the real measured vector** (via `analyze_properties`), so error in the predicted ADMET Δ never yields a false "satisfied". Every candidate product is additionally checked with `HasSubstructMatch(scaffold_smarts)` in the search and rejected if it breaks the framework; `edit_fragment` commits `products[0]` (what the tool returns), pinned by the candidate's `anchors` so that product is fully determined.

**ADMET measurement robustness.** The admet_ai backend intermittently returns `None` for all five ADMET outputs at once under GPU contention (physchem is unaffected; values are deterministic once they return). Both the forward search and the builder's verification/read-back `analyze_properties` calls retry on an all-`None` ADMET response (`search_plan.measure_with_retry`), so a validly-built molecule is not spuriously marked failed and stored `expected_response`s never carry a null ADMET.

### Inputs and outputs

**Input:** A JSONL file or directory of JSONL files produced by stage 2 (`2_substructure_gen`). Each record carries `ref_smiles`, `scaffold_smarts`, `properties`, `description`, `topology_summary`, `ring_systems`, and `connections`.

**Output:** A directory of JSONL chunk files (`toolchains_generation_chunk_NNNN.jsonl`) plus a `summary.json` with aggregate statistics (success rates, average tool-call counts, constructive vs. anchor strategy split). Each output record is a `GeneratorInput` Pydantic object serialised to JSON.

---

## Files

| File | Role |
|---|---|
| `__init__.py` | Package init; re-exports `ToolChainBuilder`, `ALL_TOOL_NAMES`, `main`, `parse_args`. |
| `__main__.py` | Entry point for `python -m 3_toolchain_gen`; calls `cli.main()`. |
| `cli.py` | **CLI entry point.** Defines and parses all CLI arguments, resolves the output directory, constructs a `ToolChainBuilder`, runs it, and writes `summary.json`. |
| `builder.py` | **Core builder.** `ToolChainBuilder` orchestrates trajectory planning (`_plan_trajectory`), phase-aware step assembly (`build`), concurrent HTTP dispatch, and constraint evaluation. Also contains I/O helpers: `_load_jsonl`, `_write_chunk`, `_scan_existing`, `dataset_name`. |
| `config.py` | Tool configuration: `ANALYSIS_TOOLS`, `EDIT_TOOLS`, `ALL_TOOL_NAMES` (the full tool set advertised to the model), `ADMET_PROPERTY_NAMES`. Also re-exports `TOOL_SERVER_PORTS`, `TOOL_SERVER_HOST`, and `TOOL_SERVER_TIMEOUT` (overridable via environment variables). |
| `constraints.py` | Pure property-constraint helpers: `extract_properties` (parses `{property, min?, max?}` dicts from an instance), `check_property` (strict satisfaction — `val` inside `[lo, hi]`). |
| `http_client.py` | Async and synchronous HTTP wrappers for the tool servers. `api_call_async` (aiohttp, used by the builder) and `api_call` (requests, for one-off calls). Includes retry logic, connection pooling, and a `close_async_session` teardown helper. Also exposes `api_call_batch_async` for batch endpoints. |
| `schema.py` | Pydantic data models: `ToolCall`, `ToolChainStep` (supports parallel tool calls), `GeneratorInput` (one tool-chain record), `ToolStep`, `TrainingExample` (with `to_messages()` / `to_training_format()` for OpenAI chat format), `VerificationResult`. |

---

## Usage

### Prerequisites

By default the tools run **in-process** (`--num-procs` worker processes, each loading its own admet_ai model), so **no tool servers need to be running** — the stored `expected_response`s are byte-identical to the HTTP path. To instead use the HTTP tool servers, run with `NUM_PROCS=1 LOCAL_TOOLS=0`; then all chemistry tool servers for the 5-tool set (`analyze_properties`, `match_substructure`, `label_atom_indices`, `edit_fragment`, `suggest_edits`) plus an ADMET backend must be running (`python -m molkit.tools.tool_server --all`, with
`ADMET_SERVER_URLS` pointing at your ADMET-AI deployment), and the builder will fail with connection errors if any required server is down.

The correct conda environment is resolved automatically by the run script (prefers `molkit`).

### Run the builder

```bash
bash run_scripts/pipeline/3_toolchain_gen.sh
```

Env-var overrides (all optional):

```bash
INPUT=/path/to/instances/generation_200k_scaffold \
OUTPUT_DIR=data/training_data/toolchain \
NUM_WORKERS=48 \
CHUNK_SIZE=1000 \
SEED_MODE=hub \
USE_REF=1 \
LIMIT=500 \
  bash run_scripts/pipeline/3_toolchain_gen.sh
```

The underlying Python command (run from the repo root):

```bash
python -m 3_toolchain_gen \
  --input  data/training_data/instances/generation_200k_scaffold \
  --output data/training_data/toolchain \
  --num-workers 48 \
  --seed-mode hub \
  --resume
```

Key CLI arguments:

| Argument | Default | Description |
|---|---|---|
| `--input` | (instances dir) | JSONL file or directory of JSONL files to process. |
| `--output` | (toolchain dir) | Base output directory. A subfolder named after the input is created inside it. |
| `--chunk-size` | 1000 | Records per output JSONL chunk file. |
| `--num-workers` | 20 | Concurrent async coroutines (set higher, e.g. 48, for large runs). |
| `--seed-mode` | `hub` | Seed for the reconstruction trajectory: `hub` (most-connected ring system), `ring_system` (smallest single ring), or `murcko` (full Bemis–Murcko scaffold). Ignored in direct-scaffold mode (the seed is always the completed scaffold). |
| `--direct-scaffold-seed` / `--no-direct-scaffold-seed` | `--direct-scaffold-seed` | Default: seed = completed scaffold, scaffold-building edits dropped, every checkpoint 3-way (`match ∥ analyze ∥ label`). `--no-…` restores the legacy incremental assembly (scaffold phase, 2-way checkpoints, separate pre-edit labels). |
| `--use-ref` / `--no-use-ref` | `--use-ref` | Plan from `ref_smiles` (default) or from description + properties alone. |
| `--analyze-each-decorate` | on | Emit an `analyze_properties` read-back after each intermediate decoration. |
| `--limit` | (none) | Process only the first N instances (useful for test runs). |
| `--resume` | off | Skip instances whose `instance_index` already appears in existing chunk files. |
| `-v` / `--verbose` | off | Enable DEBUG logging. |


## Inputs / Outputs

### Input

A `*_scaffold` JSONL file or directory from stage 2 (`2_substructure_gen`). Required fields per record:

- `ref_smiles` — gold reference molecule SMILES.
- `scaffold_smarts` — strict SMARTS the final molecule must match.
- `properties` — list of `{property, min?, max?}` target ranges.
- `description`, `topology_summary`, `ring_systems`, `connections` — natural-language and structured scaffold specification (written into the stored user prompt and metadata).
- `has_scaffold`, `description_violations` — failure flags; instances with either are skipped.

### Output

Written to `<output>/<dataset-name>/` (e.g. `.../toolchain/generation_200k_scaffold/`):

- `toolchains_generation_chunk_NNNN.jsonl` — chunk files (1000 records each by default). Each line is a JSON-serialised `GeneratorInput`:
  - `user_prompt` — the scaffold-description + property-range prompt.
  - `ground_truth_molecule` — the final molecule SMILES.
  - `tool_chain` — ordered list of `ToolChainStep` objects. Each has `tool_call` + `expected_response`, plus `parallel_tool_calls` (a list of companions run simultaneously) + `parallel_expected_responses` (index-aligned). A 3-way checkpoint is `tool_call = match_substructure` with `parallel_tool_calls = [analyze_properties, label_atom_indices]`; each decorate `edit_fragment` step is preceded by its own `suggest_edits` step. (Legacy chunks that stored a scalar `parallel_tool_call` still load — a `before` validator folds them into the list.)
  - `tool_set` — list of all tool names advertised for this task.
  - `metadata` — per-instance stats: `chain_strategy` (`constructive` / `anchor`), `direct_scaffold_seed`, `num_edit_steps`, `num_scaffold_steps` (0 in direct mode), `num_decorate_steps`, `num_tool_calls`, `all_constraints_satisfied` (== `all_constraints_strictly_satisfied`; both retained for downstream compatibility), per-property satisfaction, and any `tool_errors`.

- `summary.json` — aggregate statistics over all chunk files: total instances, success rate (near and strict), constructive vs. anchor count, average tool-call and edit-step counts, instances with tool errors.

---

## Notes

- **Resumability.** Pass `--resume` to skip already-processed instances when continuing an interrupted run. The builder scans existing chunk files for `instance_index` values and skips them; new records are appended to new chunks starting from the next chunk index.

- **Tool execution.** By default (`--num-procs > 1`, or `--local-tools`) the tools run in-process and no servers are contacted; the stored `expected_response`s are byte-identical to the HTTP path. In the HTTP path (`NUM_PROCS=1 LOCAL_TOOLS=0`) every `expected_response` is a live tool-server response — the builder mocks no calls — so start the servers (`python -m molkit.tools.tool_server --all`) and confirm they are healthy first; an unreachable server records an error in `metadata.tool_errors` and the instance may fail constraint verification.

- **`USE_REF` flag.** `--use-ref` plans from the gold molecule; `--no-use-ref` (the run-script default) runs the ref-free `suggest_edits`-guided forward search (`search_plan.py`): it constructs a molecule from the scaffold seed using mmpdb moves and only skips an instance when the seed itself is unusable, so it can produce training pairs whose answer differs from (or has no) gold ref. In the HTTP path it exercises the full 5-tool set — `analyze_properties` (measured every step), `suggest_edits`, and `edit_fragment` in addition to the verify tools.

- **Strict property satisfaction.** A property is satisfied only when its measured value lies inside the closed target interval `[lo, hi]` — there is no tolerance band. `all_constraints_satisfied` and `all_constraints_strictly_satisfied` are therefore identical (both kept for downstream compatibility).

- **Parallel steps.** Verification checkpoints (`match_substructure ∥ analyze_properties ∥ label_atom_indices`) are represented as a single `ToolChainStep` whose `parallel_tool_calls` list holds the companions run alongside the primary `tool_call`. Stage 4 (`4_sftdata_gen`) converts these into OpenAI-format assistant messages with N simultaneous tool calls followed by N tool responses.
