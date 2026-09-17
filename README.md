# Molecular generation SFT data pipeline

Code that builds the supervised fine-tuning corpus for a tool-using molecular generation
agent: property-constrained task instances, natural-language structure constraints, and
verified tool-call trajectories that solve them.

Every stage is a standalone entry point that reads JSONL (or parquet) and writes JSONL.
Stages are numbered in dependency order.

```
0_build_mmpdb  →  1_instance_gen  →  2_substructure_gen   →  3_toolchain_gen  →  4_sftdata_gen
                                     2b_functional_group_gen                              │
                                       5_rule_selection  →  6_edit_reasoning  ────────────┘
```

| Stage | What it does | Output |
|---|---|---|
| `0_build_mmpdb` | Mines matched-molecular-pair transformations from the molecule pool and stores them as per-cut swap files, each rule carrying a dense Δ-vector over the indexed properties. This is the move set the `suggest_edits` tool ranks. | `mmp_moves/{single,double,triple}_cut.json` |
| `1_instance_gen` | Samples a reference molecule from the pool and derives property constraints from that molecule's own percentiles, so every instance is feasible by construction. Records the joint hit count so trivial and impossible instances can be filtered. | instance JSONL |
| `2_substructure_gen` | Reduces each instance's molecule to its Bemis-Murcko scaffold, describes the scaffold in natural language (deterministic draft → LLM polish → validate/revise loop), and attaches the evaluation SMARTS. Verified with RDKit substructure matching. | instances + scaffold fields |
| `2b_functional_group_gen` | The functional-group variant of the structure constraint. The 61 scorer-visible patterns are fixed, so their descriptions are generated once into `fg_catalog.json` and looked up per instance — the per-instance driver needs no GPU. | instances + FG fields |
| `3_toolchain_gen` | Decomposes the reference molecule into a hub-seeded, phase-tagged edit sequence and executes it against the real tools, so every stored tool response is a genuine response rather than a mock. | tool-chain JSONL |
| `4_sftdata_gen` | Assembles tool chains into SFT conversations, verifies each chain against the constraints it claims to satisfy, and applies the filtering passes (satisfied-only, leakage, infra errors). | SFT corpus JSONL |
| `5_rule_selection` | Scores the candidate rules `suggest_edits` offers at one edit step, labelled by the exhaustive tree's `sat_ratio`. Its feature gate and per-candidate score feed the next stage. | rule-selection checkpoint |
| `6_edit_reasoning` | Renders the assistant turn between the `suggest_edits` result and the `edit_fragment` call: a one-sigma bucket enumeration of the candidates ending in the committed edit. Deterministic — no language model. | SFT corpus with edit-reasoning spans |

## Installation

```bash
pip install -r requirements.txt
```

`0_build_mmpdb` additionally needs the `mmpdb` command-line tool on `PATH`.
`molkit/` is the shared molecular tooling (SMILES and scaffold handling, fragment
catalogs, the `suggest_edits` ranker, and the tool implementations); keep the repository
root on `PYTHONPATH` so `import molkit` resolves.

## The molecule pool

Stages 0 and 1 read a property-tagged molecule pool: a parquet file with a `smiles`
column plus the physicochemical and ADMET property columns. The pool is built following
the pool-construction procedure of **[MolDesignBench](https://huggingface.co/datasets/LG-AI-Research/MolDesignBench)** — public molecule databases
combined with its cleaning filters — and the property columns are computed for the
cleaned set. Set that up first, then point `POOL` (or `--pool`) at the resulting parquet.

## Running a stage

Each stage has its own README with arguments and worked examples. `run_scripts/pipeline/`
holds thin wrappers that pass the defaults:

```bash
bash run_scripts/pipeline/0_build_mmpdb.sh
bash run_scripts/pipeline/1_instance_gen.sh
bash run_scripts/pipeline/2_substructure_gen.sh
bash run_scripts/pipeline/3_toolchain_gen.sh
bash run_scripts/pipeline/4_sftdata_gen.sh
```

Every wrapper takes environment-variable overrides (`POOL`, `OUTPUT`, `SERVERS`, `PYTHON`,
…); see `run_scripts/README.md`. Stages 2, 2b and 4 call an OpenAI-compatible endpoint,
so a served model is required for those; `SERVERS` lists the endpoints.

## Evaluation data

`data/benchmark/` holds the held-out evaluation sets:

| File | Rows | Contents |
|---|---:|---|
| `testset_900.jsonl` | 900 | in-distribution test instances |
| `ood_300.jsonl` | 300 | out-of-distribution instances; the `ood_flag` field names the axis (`property_ood` / `structure_ood` / `property_structure_ood`, 100 each) |

Each record is a generation task and carries its exact user message in `question`, the
property constraints, the grading SMARTS, and a reference molecule that witnesses the
instance is satisfiable. See `data/benchmark/README.md` for the field list.

## Not included

Model training, agentic evaluation, and the construction of the evaluation sets above are
outside this repository; the evaluation sets are provided as data.

## License

MIT — see `LICENSE`.
