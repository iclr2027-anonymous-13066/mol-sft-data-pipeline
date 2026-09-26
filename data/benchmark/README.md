# Evaluation data

Held-out evaluation sets, one JSONL record per instance.

| File | Rows | Contents |
|---|---:|---|
| `testset_900.jsonl` | 900 | in-distribution (ID) test instances |
| `ood_300.jsonl` | 300 | scaffold-out-of-distribution (Scaffold-OOD) instances |

`testset_900.jsonl` (ID) follows the standard construction, with no explicit scaffold
exclusion: MolDesignBench property constraints paired with scaffold or functional-group
requirements, plus 300 instances newly built from the test molecule pool.

`ood_300.jsonl` (Scaffold-OOD) is scaffold-disjoint from the training instances. All 300
records are scaffold conditions whose property constraints include at least three of logD,
logS, BBBP and Mutag — the ADMET properties that shift most unpredictably under molecular
edits. The set has three property-window widths:

| Set | Property | Structure | Rows |
|---|---|---|---:|
| Scaffold-OOD | ADMET ≥ 3 (narrow window) | scaffold | 100 |
| Scaffold-OOD | ADMET ≥ 3 (medium window) | scaffold | 100 |
| Scaffold-OOD | ADMET ≥ 3 (wide window) | scaffold | 100 |

ADMET ≥ 3 means at least three of the four ADMET properties are constrained. The `ood_flag`
field records the construction arm each record came from; it maps to the windows as follows:


## The task

Each record is a **generation** task: produce one molecule that contains the described
substructure and whose computed properties all fall inside the stated ranges. `question`
is the exact user message; nothing else has to be assembled to run the benchmark.

## How the descriptions were written

The structure descriptions in these evaluation sets were produced separately from the
training-data pipeline in this repository.

Functional-group conditions reuse the functional-group descriptions of **[MolDesignBench](https://huggingface.co/datasets/LG-AI-Research/MolDesignBench)**,
so the same functional group is worded the same way in both benchmarks. Scaffold conditions
are free natural-language prose covering the ring systems, how they are connected, and their
heteroatom pattern.

## Fields

**Every record**

| Field | Meaning |
|---|---|
| `id` | instance identifier |
| `source` | `bench_generation` / `bench_optimization` — carried over from the public benchmark this instance was drawn from; `pipeline` — built by the pipeline in this repository |
| `task_type` | `generation` or `optimization`. Both are posed as generation tasks; the flag only selects how functional-group counts are graded (exactly, vs. as a minimum) |
| `condition_type` | `scaffold` or `fg` — whether the structure constraint is a scaffold or a functional group |
| `question` | the user message: structure description + property ranges + answer-format directive |
| `description` | the natural-language structure constraint |
| `properties` | the property constraints, as `{property, min, max}` |
| `eval_query` | grading target — the SMARTS that a submitted molecule must match (`verdict_match_any`) |
| `ref_smiles` | a molecule that satisfies every constraint. Present as the **feasibility witness**: it shows each instance has at least one solution. It is not the only valid answer |

**Scaffold instances** — `scaffold_smarts`, `scaffold_smiles`, `scaffold_kind`, plus the
structure analysis the description was written from: `ring_systems`, `connections`,
`aromaticity`, `n_ring_systems`, `n_rings_total`, `linker_attachment_points`,
`topology_summary`, `dimension_smarts` (the same substructure at several levels of
abstraction) and `template_draft` (the deterministic draft that preceded the final
description).

**Functional-group instances** (`testset_900.jsonl` only) — `fg_constraint`, `fg_smarts`,
`fg_summary`, `functional_groups`, `n_functional_groups`, the description variants
`description_name` / `description_definition` / `description_style`, and `answer`
(the reference molecule's measured property values).
