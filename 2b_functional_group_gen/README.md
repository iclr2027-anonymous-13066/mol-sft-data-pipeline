# 2b_functional_group_gen

Stage 2b of the SFT pipeline — the **functional-group** constraint counterpart of
`2_substructure_gen` (Murcko scaffolds). Separate constraint, separate output
folder; neither stage reads the other's fields.

```
1_instance_gen/ ─┬─→ 2_substructure_gen/  → generation_2m_scaffold/   (scaffold constraint)
                     └─→ 2b_functional_group_gen/ → generation_2m_fg/     (FG constraint)
```

---

## What it produces

For every input row, the original keys are carried forward unchanged and these are
appended (1:1, input order preserved):

| field | meaning |
|---|---|
| `smiles`, `input_smiles`, `parse_ok` | canonical / original SMILES, parse status |
| `has_fg`, `fg_source` | whether a constraint exists; `answer` or `derived` |
| `n_functional_groups`, `functional_groups` | **every** scorer-visible group in the molecule, with counts and SMARTS |
| `fg_constraint`, `n_fg_constraint` | the members that form the constraint: `{name, fr_key, count, match_mode, smarts}` |
| `fg_smarts` | the constraint's scoring SMARTS, one per member |
| `fg_smarts_joined` | the same members joined with `.` — see below |
| `eval_query` | authoritative scoring query (`count_all`; per member `op` ∈ `==`/`>=`, `count`, `uniquify`) |
| `fg_summary` | `"two benzene rings, one amide, and one nitrile"` |
| `seed_smiles`, `seed_source`, `seed_heavy_atoms`, `seed_verified` | starting molecule that already carries the whole constraint — see below |
| `description` | the natural-language brief (one of the two below, per `description_style`) |
| `description_definition` | spells the group out atom by atom |
| `description_name` | just names it: *"The molecule must contain a pyridine ring."* |
| `description_style` | which phrasing `description` holds (`definition` / `name`) |
| `description_violations` | present only when the text still contradicts the facts |
| `fg_verified` | does `ref_smiles` actually satisfy its own `eval_query`? |

`functional_groups` is the full honest analysis (the analogue of `ring_systems`);
`fg_constraint` is the subset that is actually scored (the analogue of `eval_query`'s
members). Keeping both means a different selection policy can be applied downstream
without recomputing anything.

### Do NOT join member SMARTS with `.`

It looks like "all of these" and it is not. A dot-joined SMARTS requires its parts to
map to **atom-disjoint** matches, so it rejects molecules where two required groups
legitimately share an atom:

```
CC(=O)N1CCN(C)CC1        (N-acetyl-N'-methylpiperazine)
  C(=O)-N        alone  -> 1
  N1CCNCC1       alone  -> 1
  C(=O)-N.N1CCNCC1      -> 0     both present, yet no match
```

The amide N *is* a piperazine N, so no disjoint mapping exists. Measured on real
two-group rows, the joined form disagrees with `eval_query` on **13.1%** of them. It
also fails on counting — the match count is the product of the parts.

So there is no single-string form of a multi-group constraint — but there is a
single-*call* form. `match_substructure` takes a **list** of queries, matches each on
its own, and reports `match` as the AND over them:

```json
{"query": ["c1ccccc1", "C=N-[N&X3]"], "mol_smiles": "...", "query_type": "smarts"}
{"match": true}
```

The response is the verdict alone — `match` is the only key, for a single query and
a list alike (the match count and the parsed-as kind are deliberately not reported).
`eval_query` likewise scores each member separately.

`suggest_edits(scaffold_smarts=...)` now accepts a **list** of SMARTS, all of which
must survive an edit — pass `fg_smarts` straight in. Measured on 60 real two-group
molecules:

| guard | products | keep BOTH groups |
|---|---|---|
| none | 168 | 159 (94.6%) |
| `fg_smarts` list | 168 | **168 (100%)** |

A one-group constraint needs nothing new: its single SMARTS was always a valid
`scaffold_smarts` — that parameter is just a SMARTS, with nothing scaffold-specific
about it.

---

## Two stages, and why only one of them can use a GPU

**Stage A — `build_fg_catalog.py` (once).** The scaffold pipeline calls an LLM per
molecule because every Murcko scaffold is different. A functional group is not:
there are exactly **61** scorer-visible patterns, fixed. Whatever the constraint is
on row 1 and on row 2,000,000, it is one of those 61. So the group *definitions* are
written once into `fg_catalog.json`, and that is the only step an LLM can help with.

That file is a build artifact, not source, so it lives with the generated training
data — `data/training_data/fg_catalog.json` — and is gitignored.
Readers resolve it as `$FG_CATALOG` → that path → a legacy in-repo copy if one is
still lying around (`fg_catalog.catalog_json_path()`).

The stored definition is deliberately **count-free**. The requirement clause
("exactly two of them" / "at least two of them") is generated per record by
`fg_describer.count_sentence(count, match_mode)` — it is the one clause the grader
compares against, so it is never model-generated and cannot drift from `eval_query`.
The model never sees a count.

**Stage B — `augment_jsonl_with_fg.py` (per row).** Detects groups, fixes the
constraint, looks the description up. Pure CPU, ~1200 rows/s on 8 processes; a 2M
run is minutes, not GPU-hours.

```bash
bash run_scripts/pipeline/2b_functional_group_gen.sh                 # both inputs
POLISH=1 SERVERS=localhost:8080 bash run_scripts/pipeline/2b_functional_group_gen.sh
BUILD_CATALOG=0 bash run_scripts/pipeline/2b_functional_group_gen.sh # reuse catalog
```

Without `POLISH=1` the description is the deterministic template, which is faithful
by construction. With it, the LLM output is kept **only if it passes validation** —
otherwise the draft is restored and the reason recorded in `polish_rejected`.

---

## The two input shapes

| input | constraint | SMILES key | rows kept |
|---|---|---|---|
| `molkit.jsonl` | authored: `answer.fragments` = `[{"hydroxylamine": 1}]` | `meta_info.ref_smiles` | 900 (`--keep-task-type generation --require-smiles`) |
| `generation_2m.jsonl` | none — derived from the molecule | `ref_smiles` | 2,000,000 |

`--fg-source auto` picks per record. The benchmark side is filtered to the
**generation** task, and to rows that have a `ref_smiles` — on molkit those
are exactly the 900 feasible generation instances (the 100 dropped ones are the
`infeasible: true` rows, which carry no reference molecule).

A derived constraint takes **1 or 2 groups sampled per row** (`--select random
--n-fg-min 1 --n-fg-max 2`) from the groups the reference molecule actually
contains. Sampling is keyed on the row itself, so a shard redone after an
interrupted run picks the same groups (`--seed` shifts every row at once).
`--select rarest` is available as an alternative, ordering by corpus document
frequency measured in Stage A — `benzene ring` occurs in 0.84 of the corpus and
constrains almost nothing, `thiocyanate` in 2e-05 and constrains a great deal.

Metabolic-site descriptors (`fr_aryl_methyl`, `fr_allylic_oxid`) are detected and
reported but never become a constraint: "generate a molecule with exactly one aryl
methyl site for hydroxylation" is not a chemistry brief. `--allow-descriptors`
overrides.

---

## Three things that are easy to get wrong

**1. The SMARTS must be the grader's.** Scoring SMARTS come from
`molkit.utils.fragments.fr_catalog()`, the source of truth shared with
`evaluate_benchmark.py`. Counting a pattern with `uniquify=True` reproduces
`fr_*(mol)` exactly — verified equal on 1500 corpus molecules across all 61
patterns. Do **not** substitute `scaffold_analyzer._FG_SMARTS_RAW`: it is curated
for scaffold-internal prose and disagrees on 12 of the 19 names the two share
(`imine` differs on 173/1500 molecules, `thioether` on 104, `sulfone` on 87).

**2. The name is not the definition.** An `fr_*` name is routinely narrower or wider
than it sounds, so descriptions are written from the pattern, never the label:

```
fr_N_O   "hydroxylamine"  [N!$(N=O)](-O)-C
         NH2OH -> 0 (the nitrogen must carry a carbon), (CH3)2N-OCH3 -> 2
fr_benzene "benzene ring" c1ccccc1        naphthalene -> 2, anthracene -> 3
fr_amide  "amide"         C(=O)-N         urea -> 2
fr_lactam "beta lactam"   N1C(=O)CC1      2-pyrrolidinone (5-ring) -> 0
```

`fg_catalog.py` measures this: every entry carries the match **count** on a panel of
named probe molecules, chosen for discrimination (near misses, not just members).
Those counts become FACTS the describer must respect, and `validate_text` fails any
description that names a probe the pattern scores 0 as an example — or denies one it
scores nonzero.

**3. `exact` vs `at least` is not stylistic — and this dataset chose `at least`.**
Every member is scored `>=` (`--match-mode min`, the default) and a derived member
requires **1** occurrence (`--require-count 1`): the group has to be present, and a
reference that happens to carry three copies does not make three a requirement.

The current benchmark scorer does something different
(`evaluate_benchmark.check_individual_fragments_with_dist`):

```python
if task_type == "generation": satisfied = (measured_count == count)   # EXACT
else:                         satisfied = (measured_count >= count)   # MINIMUM
```

**So scoring this dataset with that grader unchanged will fail every generation row
whose molecule carries a spare copy** — on molkit's authored constraints that
was 114/900. Either move its generation branch to `>=`, or rebuild with
`--match-mode auto` to reproduce the old behaviour. (Note the molkit question
text already says "features **at least one** hydroxylamine", which the `min` reading
matches and the current grader contradicts.)

`match_mode` is carried per member so the description says "exactly one" or "at least
two" accordingly — which is why that clause is generated, not stored.

**The requirement clause is omitted entirely when it carries no information.**
"at least one" is the default reading of any constraint, and the scaffold
descriptions this dataset sits beside carry no requirement sentence at all. So the
clause — and the counting rule sentence ("naphthalene counts as 2") with it — appears
only when `count > 1` or the mode is exact. Both datasets are currently all
`count = 1, min`, so neither clause appears anywhere in them.

### Two phrasings

```
name       : The molecule must contain a pyridine ring.
definition : A 6-membered ring made up of five aromatic carbons and one aromatic
             nitrogen is required.
```

Both are always written. The definition is what makes a brief verifiable; the name
style is how a chemist actually asks, and a model trained on definitions alone never
sees the plain request it will be given. `--description-style {mixed,definition,name}`
picks what `description` holds — `mixed` (default) chooses per row, deterministically,
so a resumed shard reproduces its own choice.

### Wording diversity

Because the constraint vocabulary is fixed at 61 patterns, every instance asking for
an amide would otherwise carry byte-identical text. Two cheap fixes, neither of which
pretends to add information:

- `build_fg_catalog.py --n-variants N` generates N wordings per group (61 × N calls,
  once) into `description_variants`. Variants are sampled at a rising temperature and
  kept **only if they pass validation**; variant 0 is always the deterministic draft,
  so every entry has a valid wording even if every sample fails.
- The name style rotates over `NAME_STYLE_OPENERS` ("The molecule must contain …",
  "Generate a molecule containing …", …) — pure templating, no model.

Each row draws its variant and opener from a hash of its own SMILES, so the choice
survives a resume. `verify_fg_dataset.py --seed` reproduces the same draw; a seed
mismatch shows up as a rebuild difference, which is the correct outcome.

### The seed molecule (`fg_seed.py`)

A Murcko scaffold **is** a molecule, so the scaffold pipeline can tell the model to
*transcribe* its committed SMARTS into SMILES — the two are one object written twice.
A functional group is not:

```
C(=O)-N               transcribed -> formamide, 3 heavy atoms
C(=O)-N + N1CCNCC1    transcribed -> two DISCONNECTED fragments, not a molecule
```

So the seed is **constructed**, not transcribed. Three strategies in order:

| `seed_source` | what it is |
|---|---|
| `carrier` | one known molecule already contains every required group |
| `joined` | two carriers bonded at a site where both groups survive |
| `none` | no construction found |

Carriers come from the probe panel (already chosen to hold each group in a small,
unambiguous molecule) plus a few larger join partners. Every candidate is accepted
**only after RDKit confirms the finished molecule still matches every member**. That
check is not a formality: bonding to ethanol's oxygen turns the hydroxyl into an
ether and destroys the group the seed exists to carry.

Measured over 20k rows → 593 distinct constraint signatures:

```
1-group  carrier  48
2-group  carrier  63
2-group  joined  482        -> 593/593 built, heavy atoms min 3 / median 10 / max 21
```

Seeds depend only on the constraint signature, so they are cached on it — at most
61 + C(61,2) constructions for the whole 2M corpus, 1.5 ms each cold.

Examples:

```
amide + piperazine        -> NC(=O)CC1CNCCN1
benzene ring + nitrile    -> N#CCc1ccccc1
beta lactam + thiophene   -> O=C1NCC1c1ccsc1
thiocyanate + barbiturate -> N#CSCC1C(=O)NC(=O)NC1=O
```

A median of 10 heavy atoms leaves the property box genuinely unmet, so the toolchain
still has real decoration work — a seed that already satisfied the box would make the
rest of the task vacuous.

### Scoring these instances

`agentic_eval.score_prediction` handles both dataset kinds: scaffold instances carry
`eval_query.dimension_sets`, FG instances carry `eval_query.members`, and the scorer
keys on which is present — so scaffold scoring is unchanged. For FG instances every
member must hold (AND), each on its own count and operator, and the per-member outcome
lands in the verdict's `fragment_detail`. `structure_success` / `property_success` /
`overall_success` keep their meaning, so existing training and eval metrics work
without modification.

`verify_fg_dataset.py` re-checks a produced dataset on all four points against the
live `fr_*` callables:

```bash
python verify_fg_dataset.py data/training_data/instances/benchmark_fg
```

---

## Files

| file | role |
|---|---|
| `fg_catalog.py` | the 61 entries: SMARTS, parsed composition, exclusions, probe counts |
| `fg_analyzer.py` | `analyze_fg` / constraint selection / `eval_query` / verification |
| `fg_prompts.py` | prompt content only (fg-v1) |
| `fg_describer.py` | FACTS block, deterministic draft, validation |
| `build_fg_catalog.py` | Stage A — writes `fg_catalog.json` to `/data` (+ optional vLLM polish) |
| `augment_jsonl_with_fg.py` | Stage B — the per-row driver |
| `verify_fg_dataset.py` | post-hoc check of a produced dataset against the grader |

Each module runs standalone for inspection:

```bash
python fg_catalog.py   fr_N_O fr_benzene      # SMARTS, composition, probe counts
python fg_describer.py fr_N_O fr_lactam       # FACTS, draft, validation
python fg_analyzer.py  "CC(=O)Nc1ccccc1"      # full per-molecule item
```
