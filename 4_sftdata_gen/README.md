# 4_sftdata_gen

Convert constructive ground-truth tool-call trajectories (stage 3 output) into chat-format SFT training data.

---

## Overview

Stage 4 takes the toolchain JSONL files produced by `3_toolchain_gen` and turns each ground-truth tool chain into one or more fine-tuning examples in OpenAI chat format.

The stage-3 chains this stage consumes are **constructive substructure-generation** chains (`task_type = generation`, `chain_strategy = constructive`) in the **direct-scaffold-seed** format: the chain starts from the COMPLETE scaffold and tunes it by repeated editing rounds, verifying against a required substructure and numeric property targets. The tool vocabulary is:

| Tool | Role |
|---|---|
| `suggest_edits` | Ranks candidate MMP edits toward the target property box; called immediately BEFORE each edit to choose that edit's `edit_fragment` arguments |
| `edit_fragment` | The single matched-molecular-pair edit primitive (attach / swap / remove via `from_smiles → to_smiles` + `anchors`) that commits the chosen candidate |
| `match_substructure` ‖ `analyze_properties` ‖ `label_atom_indices` | The 3-way parallel **verification checkpoint** — confirms the required substructure is present, measures the target properties, AND labels the atoms (grounding the *next* edit's anchors) |

The core transformation is:

1. **Segmentation (one edit per segment)** — Each chain is split into:
   - a **seed segment** — the single 3-way seed checkpoint (`match ∥ analyze ∥ label`) run on the complete scaffold (NO edit);
   - one **edit segment per structural edit** — each `edit_fragment` call with its `suggest_edits` lead-in (the candidate ranking that informs it) and its trailing 3-way checkpoint (which also labels the atoms for the next edit);
   - a **terminal ANSWER segment** — emits `<ANSWER>` with no tool call, once the final round's own measurements satisfy every constraint (the answer is ALWAYS its own final segment; there is no `Verification:`-block-then-answer in the last edit segment).

   Segmentation scopes each LLM call to one build round. The segments of one chain are then **merged into a single conversation** (`schema.merge_segments`): the system and user turns are taken once and every segment contributes its assistant/tool turns in order, so one chain produces one training record.

2. **LLM reasoning generation** — For each tool call the `Generator` LLM writes natural-language reasoning that leads to that call. Reasoning prompts are chosen by step role **and build phase**:
   - **seed introduction** — in ONE reasoning turn, *derives* the **SMARTS pattern** encoding the whole required substructure (ring/linker inventory → the committed pattern) AND *writes a concrete scaffold SMILES* that contains it (the molecule the seed checkpoint then verifies/measures/labels). The committed SMARTS + scaffold SMILES are treated as authoritative (the informal NL description is not re-derived atom-by-atom).
   - **suggest_edits lead-in** — the substructure is already present, so the edit is a *property fix*: `_property_gap` feeds the reasoning the specific out-of-range properties (current value, target, and the direction each must move), and `SUGGEST_REASONING_PROMPT` frames the decision to call `suggest_edits` with those constraints rather than guess a substituent.
   - **edit_fragment commit** — `EDIT_FRAGMENT_REASONING_PROMPT` justifies committing one of the returned candidates (the `from_smiles → to_smiles` swap and its `anchors`) against the same property gap.
   - the **parallel verification checkpoint** (`CHECKPOINT_REASONING_PROMPT`).

3. **No carried state** — In one continuous conversation the model can see the earlier rounds directly, so nothing is summarised and carried between them. Whether a chain is finished is decided rule-based from the final round's own tool results (`constraint_state.is_fully_satisfied`), and the `Verification:` block printed before `<ANSWER>` is rebuilt the same way, so the model never has to re-emit un-computable numbers.

4. **System prompt** — a single line (`You are an expert molecular generation AI assistant.`). The tool-using protocol is taught by the examples themselves, not by system-prompt instructions. Kept in sync with the copy used by the training-time eval harness (not part of this repository).

5. **Optional augmentation / verification** — The `Augmentor` LLM can paraphrase tool names/descriptions and the system/user prompts; the `Verifier` LLM can score examples and trigger retries. Both are off by default.

The output is JSONL where each record is one chain: `messages` (system/user/assistant/tool turns), `tools` (JSON schemas), and `metadata` including `group_id`, `segments_merged`, `ends_with_answer`, and `seed_messages` (where the seed round ends, so the seed can be kept on its own).

---

## Files

### Pipeline modules

| File | Role |
|---|---|
| `__init__.py` | Package exports: `Augmentor`, `PipelineConfig`, `VLLMConfig`, `TrainingDataPipeline`, schema types |
| `__main__.py` | Entry point for `python -m 4_sftdata_gen`; calls `main()` |
| `main.py` | CLI argument parsing and pipeline construction; dispatches single-file or directory mode |
| `pipeline.py` | `TrainingDataPipeline` — orchestrates Generator → (Augmentor) → (Verifier); classifies inputs into satisfied / unsatisfied and (by default) **drops the unsatisfied ones** via `_balance_inputs` (toggle with `config.drop_unsatisfied_chains`); batch processing with ordered flush; resume logic; directory-mode chunk iteration |
| `generator.py` | `Generator` — builds `TrainingExample` segments; `generate_segments()` for the seed + per-edit segments + terminal ANSWER segment; per-step reasoning dispatch (seed intro / suggest_edits lead-in / edit_fragment commit / checkpoint / recovery); `generate()` single-example fallback |
| `segmenter.py` | `extract_rounds()` — walks the chain to build `Round` objects (SMILES, measured properties, substructure-match flag, labelled atoms, and `error` for a FAILED edit); `build_seed_segment()` / `build_segments()` — produce the `SeedSegment` + `Segment` list; `EDIT_TOOLS` constant; checkpoint + error detection |
| `constraint_state.py` | `is_fully_satisfied()` — determines the terminal answer (substructure match + all property targets), read off the round's own tool results; `build_constraint_check()` — the rule-based `Verification:` block printed before `<ANSWER>`; the value/range formatting helpers the reasoning prompts share |
| `schema.py` | Pydantic models: `ToolCall`, `ToolChainStep`, `GeneratorInput`, `ToolStep`, `TrainingExample`, `VerificationResult`; `TrainingExample.to_messages()` / `to_training_format()` produce the final JSONL record |
| `prompts.py` | Prompt templates: `SYSTEM_PROMPT_TEMPLATE` (the minimal one-liner), `SEED_INTRO_PROMPT` (derivational, derives the SMARTS), `SUGGEST_REASONING_PROMPT` (the `suggest_edits` lead-in) + `EDIT_FRAGMENT_REASONING_PROMPT` (committing a candidate) — the current decorate flow, `CHECKPOINT_REASONING_PROMPT`, `ANSWER_CONFIRM_PROMPT`, `REASONING_PROMPT`, `REFLECTION_PROMPT`, `TOOL_REASON_PROMPT`, plus augmentor/verifier prompts |
| `augmentor.py` | `Augmentor` — optional; paraphrases system/user prompts and tool names/descriptions via LLM; shuffles tool schema order |
| `verifier.py` | `Verifier` — optional; LLM scores each example and returns accept/reject + per-dimension scores + feedback |
| `llm_client.py` | `LLMClient` — async wrapper around one or more OpenAI-compatible (vLLM) endpoints; round-robin dispatch with per-server semaphores (40 concurrent/server) |
| `config.py` | `VLLMConfig` and `PipelineConfig` dataclasses; `PipelineConfig.from_json()` / `to_json()` for file-based config |

### tools/ — chemistry tool interface

| File | Role |
|---|---|
| `tools/registry.py` | `ToolRegistry` — loads schemas from `molkit.tools.TOOL_REGISTRY`; maps tool names to JSON schemas |
| `tools/executor.py` | `ToolExecutor` — returns the `expected_response` from the ground-truth chain (no network call); falls back to a registered implementation only if absent |

### HTML viewers

`view_constructive_sftdata.py` reads training JSONL and writes a self-contained HTML report for visual inspection (`--input` a JSONL file or directory, `--out` the HTML): one card per record — every turn, tool calls/responses and the final `<ANSWER>`, with molecules drawn inline and the required substructure highlighted. It requires only `rdkit` and the standard library.

---

## Usage

### Quick start

```bash
bash run_scripts/pipeline/4_sftdata_gen.sh
```

Key environment variables (all optional):

| Variable | Default | Description |
|---|---|---|
| `INPUT_ROOT` | `data/training_data/toolchain` | Root containing per-task subdirs with `toolchains_*chunk_*.jsonl` |
| `OUTPUT_ROOT` | `data/training_data/sftdata` | Root output directory (per-task subdirs created automatically) |
| `BATCH_SIZE` | `320` | Maximum concurrent examples in-flight |
| `NUM_GENERATIONS` | `1` | Independent generation passes per input |
| `VLLM_URLS` | `http://localhost:8080/v1,…,8086/v1` | Comma-separated vLLM server URLs |
| `VLLM_MODEL` | `Qwen/Qwen3.6-27B` | Model name served by the vLLM pool |
| `PYTHON_BIN` | `python` | Python interpreter |

### Full CLI reference

```bash
python -m 4_sftdata_gen \
    --input-dir   data/training_data/toolchain/generation_benchmark_scaffold \
    --output-dir  data/training_data/sftdata/generation_benchmark_scaffold \
    --generator-urls  http://localhost:8080/v1,http://localhost:8081/v1 \
    --generator-model Qwen/Qwen3.6-27B \
    --reflection-prob 0.3 \
    --batch-size      320 \
    --num-generations 1 \
    --resume
```

Single-file mode (one chunk):

```bash
python -m 4_sftdata_gen --input_path input_chunk_0000.jsonl -o output_chunk_0000.jsonl
```

#### Key flags

| Flag | Default | Description |
|---|---|---|
| `--no-segment` | off | Disable segmentation; emit one example per input (fallback path) |
| `--reflection-prob` | 0.3 | Probability of adding a reflection after a (solo) verification step |
| `--use-augmentor` | off | Enable augmentor step |
| `--use-verifier` | off | Enable verifier step |
| `--threshold` | 7.0 | Minimum verifier overall score to accept an example |
| `--max-retries` | 3 | Max generation retries per example when verifier is enabled |
| `--num-generations` | 1 | Independent generation passes per input (diverse reasoning, same tool calls) |
| `--resume` | off | Skip inputs whose `user_prompt` already appears in the output file; append new results |
| `--seed` | None | Random seed for reproducibility |
| `-c` / `--config` | None | Path to a JSON pipeline config file (overrides CLI LLM args) |

### Prerequisites

- **vLLM pool** serving `Qwen/Qwen3.6-27B` on ports 8080–8086.
- **Conda environment**: `molkit` (`$CONDA_ROOT/envs/molkit`). The `molkit` package must be importable (tool schemas are loaded from `molkit.tools.TOOL_REGISTRY`).
- **Tool servers are NOT required** — the `ToolExecutor` uses the pre-computed `expected_response` fields from the input JSONL.

---

## Inputs / Outputs

### Input

Directories of `toolchains_*chunk_*.jsonl` files produced by `3_toolchain_gen`. Each line is a JSON object matching `GeneratorInput`:

```json
{
  "user_prompt": "Generate ONE chemically valid, drug-like molecule that contains the substructure described below. …",
  "ground_truth_molecule": "O=C(O)c1cnc2c(c1)n(CCCn1cccn1)c(=O)n2CCc1ccccn1",
  "tool_chain": [
    {"tool_call": {"name": "match_substructure", "arguments": {"query": "[#6]1:…", "mol_smiles": "<scaffold>", "query_type": "smarts"}}, "expected_response": "{\"match\": true, …}",
     "parallel_tool_calls": [
       {"name": "analyze_properties", "arguments": {"mol_smiles": "<scaffold>", "property_names": ["MW", "logP", …]}},
       {"name": "label_atom_indices", "arguments": {"mol_smiles": "<scaffold>"}}],
     "parallel_expected_responses": ["{\"MW\": 380.1, …}", "[O:0]=…"]},
    {"tool_call": {"name": "suggest_edits", "arguments": {"mol_smiles": "<scaffold>", "constraints": {"logP": [0.0, 4.0], …}, "top_k": 4}}, "expected_response": "[{\"from_smiles\": \"[*:1][H]\", \"to_smiles\": \"[*:1]CC\", \"anchors\": {…}}, …]", "parallel_tool_calls": [], "parallel_expected_responses": []},
    {"tool_call": {"name": "edit_fragment", "arguments": {"mol_smiles": "<scaffold>", "from_smiles": "[*:1][H]", "to_smiles": "[*:1]CC", "anchors": {"1": 9}}}, "expected_response": "CCn1c(=O)…", "parallel_tool_calls": [], "parallel_expected_responses": []},
    {"tool_call": {"name": "match_substructure", "arguments": {"query": "[#6]1:…", "mol_smiles": "CCn1c(=O)…", "query_type": "smarts"}}, "expected_response": "{\"match\": true, …}",
     "parallel_tool_calls": [
       {"name": "analyze_properties", "arguments": {"mol_smiles": "CCn1c(=O)…", "property_names": ["MW", "logP", …]}},
       {"name": "label_atom_indices", "arguments": {"mol_smiles": "CCn1c(=O)…"}}],
     "parallel_expected_responses": ["{\"MW\": 392.16, …}", "[C:0]…"]}
  ],
  "tool_set": ["analyze_properties", "match_substructure", "label_atom_indices", "edit_fragment", "suggest_edits"],
  "metadata": {
    "task_type": "generation",
    "chain_strategy": "constructive",
    "direct_scaffold_seed": true,
    "seed_mode": "scaffold_direct",
    "smarts": "[#6]1:[#6]:[#7]:…",
    "description": "A central pyridine fused to imidazole core, bearing a cyclic urea carbonyl …",
    "seed_smiles": "<scaffold>",
    "predicted_molecule": "O=C(O)c1cnc2c(c1)n(CCCn1cccn1)c(=O)n2CCc1ccccn1",
    "target_properties": {"MW": [250.0, 800.0], "logP": [0.0, 4.0], "HBD": [null, 1.0], …},
    "num_scaffold_steps": 0,
    "edit_step_phases": ["decorate", "decorate", "decorate"],
    "substructure_match": true,
    "prop_constraint_results_strict": {"MW": true, "logP": true, …},
    "all_constraints_strictly_satisfied": true
  }
}
```

(Legacy chunks that stored a scalar `parallel_tool_call` / `parallel_expected_response` still load — a `before` validator folds them into the list form.)

### Output

Per-chunk JSONL files in the output directory, one record per training example. Each record:

```json
{
  "messages": [
    {"role": "system",    "content": "…"},
    {"role": "user",      "content": "Generate ONE … molecule that contains the substructure …"},
    // seed round: the FIRST assistant derives the SMARTS + writes the scaffold SMILES,
    //   then the 3-way seed checkpoint runs on it
    // each edit is a suggest_edits call, then the edit_fragment that commits a candidate:
    {"role": "assistant", "content": "…", "tool_calls": [{"function": {"name": "suggest_edits", …}}]},
    {"role": "tool",      "tool_call_id": "…", "content": "[{\"from_smiles\": …}]"},
    {"role": "assistant", "content": "…", "tool_calls": [{"function": {"name": "edit_fragment", …}}]},
    {"role": "tool",      "tool_call_id": "…", "content": "CCn1c(=O)…"},
    // the 3-way checkpoint = ONE assistant msg with 3 tool_calls, then 3 tool responses:
    {"role": "assistant", "content": "…", "tool_calls": [{"name": "match_substructure"}, {"name": "analyze_properties"}, {"name": "label_atom_indices"}]},
    {"role": "tool", …}, {"role": "tool", …}, {"role": "tool", …},
    // …one such block per edit round, in order…
    // terminal turn (no tool call): {"role": "assistant", "content": "…all targets satisfied…\n\n<ANSWER>SMILES</ANSWER>"}
  ],
  "tools": [ … ],
  "metadata": {
    "molecule": "O=C(O)c1cnc2c(c1)n(CCCn1cccn1)c(=O)n2CCc1ccccn1",
    "num_tool_calls": 12,
    "example_type": "normal",
    "generation_idx": 0,
    "group_id": "generation_9153__0000042",
    "segments_merged": 6,
    "ends_with_answer": true,          // false = the chain never reached <ANSWER>
    "seed_messages": 4                 // messages[:4] is the seed round on its own
  }
}
```

`group_id` ties the record back to its source input. `segments_merged` says how many build rounds went into the conversation, and `seed_messages` marks where the seed round ends. When `--num-generations > 1`, each pass is its own record and gets its own `group_id` suffix (`__gen0`, `__gen1`, …).

---

## Notes

- **Segmentation is on by default** — pass `--no-segment` to fall back to one example per input.
- **`is_final` decision** — A chain's last segment ends with `<ANSWER>` only when the final round both **matches the required substructure** and **meets every property target** (evaluated rule-based from the chain's own checkpoint data via `is_fully_satisfied`). Otherwise every segment is intermediate and the conversation just stops after the last tool response. Note that with `drop_unsatisfied_chains=True` (the default), chains whose final molecule is not fully satisfied are excluded upstream in `_balance_inputs`, so every generated chain reaches an `<ANSWER>`.
- **The SMARTS and starting core are derived, not asserted** — the seed segment's reasoning *derives* the SMARTS pattern encoding the whole substructure (ring/linker inventory) and identifies the hub ring system as the core SMILES to build from (`_hub_context` supplies the hub breakdown). The committed SMARTS + core are treated as ground truth; the reasoning is kept concise and decisive (backtracking / "wait, let me re-read…" rambling is truncated by `_truncate_meta`).
- **Answer is its own terminal segment** — the last edit segment ends after its checkpoint like any other; a separate final segment emits `<ANSWER>` with no tool call.
- **Tool execution** — The `ToolExecutor` always uses `expected_response` from the input JSONL; live tool servers are not contacted.
- **Throughput** — I/O-bound on the vLLM pool. Each `LLMClient` caps at 40 concurrent requests per server and round-robins across all URLs. `--batch-size` bounds the total concurrent examples.
- **Resume safety** — With `--resume`, already-processed inputs are identified by exact `user_prompt` match against the output file; new results are appended.
