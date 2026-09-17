# Scaffold-Description Dataset (`scaffolds.jsonl`)

> A guide for first-time readers. Each line (row) of `scaffolds.jsonl` corresponds to one molecule.
> Every row carries **(1) the molecule's scaffold**, **(2) a natural-language description of that
> scaffold**, and **(3) SMARTS artifacts for evaluating a generative model**.

---

## 0. At a glance

- **What**: each molecule is reduced to its **Bemis–Murcko scaffold** (the ring systems plus the
  linkers between them, with all substituents stripped) and that scaffold is described in fluent,
  chemist-style natural language.
- **Downstream use**: train a model **(scaffold description + property ranges) → generate one
  molecule**, then evaluate whether the generated molecule **actually contains the described
  scaffold** via SMARTS substructure matching.
- **The core difficulty**: a description is faithful but **lossy**, so it does not pin down a single
  scaffold. Evaluation must therefore accept "any one of the family of consistent scaffolds"
  (**match-any**). The SMARTS artifacts needed for this are precomputed in every row.

---

## 1. What is a scaffold?

The structure left after keeping only the **ring systems** and the **linkers that join them**, and
stripping the pendant substituents. E.g. the anilide `O=C(Nc1ccccc1)c1ccncc1` → scaffold
`O=C(Nc1ccccc1)c1ccncc1` (here there are no substituents, so molecule = scaffold). A double-bonded
**oxygen/sulfur (=O, =S)** on a ring or linker atom determines the framework's identity
(amide / urea / lactam …), so it is **preserved**.

---

## 2. Core idea — one description = MANY scaffolds

A chemist does not spell out every atomic position (that would be unnatural). So a description mixes
two kinds of fact:

- **Fixed** — what a chemist naturally pins down: the ring name (`pyridine`,
  `benzimidazol-2(3H)-one`), the linker identity (`amide`, `sulfonamide`), the ring locant a **linker**
  attaches to (`position 4 of the pyridine`), and aromatic-vs-saturated state.
- **Free** — what a chemist leaves vague: the exact linker-attachment carbon on a complex
  saturated/fused ring (`a ring carbon`), the exact fusion of a polycyclic given only by its component
  rings, and **the exact arrangement of heteroatoms in a ring** (e.g. "a 5-membered ring with one N
  and one S" does not say 1,3-thiazole vs 1,2-isothiazole). → the `element` dimension is built as a
  match-any set that leaves this heteroatom arrangement free (§3, §4).

> **Substituent positions are NOT described.** Substituents are stripped from the scaffold, so "where
> a substituent attaches" (a ring 2,5-position, a benzene ortho/meta/para pattern) is unverifiable and
> is excluded from both the description and the data. Only **which ring atom a linker attaches to** is
> kept — the linker is part of the scaffold, so it is verifiable.

> **Consequence**: many scaffolds are consistent with one description (vary the free parts and they
> all still fit). So requiring a match to the single "ground-truth" scaffold would **falsely reject**
> molecules that satisfy the description but realize a different valid choice (in this pool only ~11%
> of scaffolds are fully fixed → a single-SMARTS metric would wrongly fail most of them).
> → Evaluation must be **match-any** (pass if the molecule matches any one of several candidate SMARTS).

The central design of this dataset is: **encode exactly what the description states as FIXED into the
SMARTS, and keep what it leaves FREE free in the evaluation too** (several candidates / a relaxed query).

---

## 3. The five SMARTS dimensions

The scaffold is projected onto five "views", each keeping only one kind of information so that one
aspect can be checked in isolation (for partial credit / diagnostics).

| dimension | information kept | atom token | bond token |
|-----------|------------------|------------|------------|
| `skeleton` | connectivity (shape) only | `*` | `~` (any) |
| `element` | element (C/N/O/S) | `[#6]`,`[#7]`,`[#8]`,`[#16]` | `~` |
| `aromaticity` | aromatic vs saturated | `a` / `A` | `~` |
| `bond` | bond order | `*` | `-` `=` `#` `:` |
| `ring` | ring membership / size | `[r6]`,`[r5]`,`[R0]` | `~` |

**Example** — `dimension_smarts` for `O=C(Nc1ccccc1)c1ccncc1` (benzene–amide–pyridine):

```
skeleton   : *~*(~*~*1~*~*~*~*~*~1)~*1~*~*~*~*~*~1
element    : [#8]~[#6](~[#7]~[#6]1~[#6]~[#6]~[#6]~[#6]~[#6]~1)~[#6]1~[#6]~[#6]~[#7]~[#6]~[#6]~1
aromaticity: A~A(~A~a1~a~a~a~a~a~1)~a1~a~a~a~a~a~1
bond       : *=*(-*-*1:*:*:*:*:*:1)-*1:*:*:*:*:*:1
ring       : [R0]~[R0](~[R0]~[r6]1~[r6]~[r6]~[r6]~[r6]~[r6]~1)~[r6]1~[r6]~[r6]~[r6]~[r6]~[r6]~1
```

> **degree/H tokens are deliberately omitted** → the patterns still match a *substituted* molecule
> (the scaffold is a substructure).

### Property 1 — the dimensions are nested
All five SMARTS share the **same connectivity skeleton**; each property dimension is just "skeleton +
one label layer". Therefore **matching any property already implies matching skeleton**
(`element/aromaticity/bond/ring` ⟹ `skeleton`). So **skeleton is the weakest floor** and is redundant
as a standalone score — use it only as the alignment substrate / a coarse failure floor, and score
partial credit on the property dimensions. (Also, matching the five independently and ANDing them is
**unsound** — they could match at *different places* in the molecule. The authoritative verdict
requires all properties at the **same location**; see `verdict_match_any`.)

### Property 2 — the ambiguity shows up differently per dimension
The free part surfaces differently per dimension. Most notably, **the `element` dimension leaves the
exact arrangement of heteroatoms in a ring free** — a description usually states only the *kind and
count* of elements (e.g. "a 5-membered ring with one N and one S"). So `element` checks only
*composition + skeleton connectivity* and offers the heteroatom positions as several candidates
(match-any). The other four dimensions (skeleton/aromaticity/bond/ring) are element-agnostic, so they
are unaffected.

E.g. thiazole (`c1cscn1`) → `element` set = **2** (N/S as 1,3 = thiazole / 1,2 = isothiazole):
```
[#6]1~[#6]~[#16]~[#6]~[#7]~1     # 1,3 (the actual arrangement)
[#16]1~[#7]~[#6]~[#6]~[#6]~1     # 1,2
```
→ This is why per-dimension **match-any sets** differ in size (usually `element` is the largest).
Symmetry-equivalent arrangements are collapsed via graph canonicalization, and systems with very many
heteroatoms are pinned to the actual arrangement to avoid blow-up (set capped at 64). **The
authoritative `verdict_match_any` stays strict on the actual arrangement.**

---

## 4. SMARTS-generation strategy for evaluation

Each row's `eval_query` implements the "encode fixed, relax free" principle directly.

1. **Build family J (enumeration)** — **always keep the source scaffold as a member**, keep all FIXED
   facts, and re-attach only the FREE attachment points to symmetry-distinct candidate atoms. If
   symmetry leaves only one candidate, it is effectively fixed. (Why always keep the original:
   candidates are chosen by *bare ring-system* symmetry, so if substituents break that symmetry none of
   the variants may match the real molecule — dropping the original would leave a hole where "the
   correct molecule fails its own eval_query".)
   - all fixed/symmetric → **J = 1 → a single fixed SMARTS**.
   - some free → **J = several → a match-any set**.
   - additionally, the **`element` dimension set leaves the in-ring heteroatom arrangement free**
     (§3, Property 2); the other dimensions and `verdict_match_any` are unaffected (verdict stays
     strict on the actual arrangement).
2. **Authoritative verdict** = `verdict_match_any`: the **full SMARTS** of each J member (all
   properties on the same atoms). The molecule passes if it matches **any one**. (Fair: any valid
   free-choice realization passes / Sound: fixed positions stay strict — e.g. if "position 4" is
   fixed, a 3-substituted molecule is rejected.)
3. **Per-dimension diagnostics** = `dimension_sets`: J projected per dimension and deduplicated. Used
   for partial credit such as "got the elements right but the aromaticity wrong".
4. **Composite cores use decomposition** — an unnamed polycyclic (e.g. `two bridged cyclopentane
   rings`) has too many possible fused graphs to enumerate. So instead of pinning the exact fused
   graph, store the component-ring SMARTS + how they overlap (`decompositions`). At evaluation,
   `match_decomposition()` only checks **"all component rings present AND joined as stated"** → the
   exact fusion position stays free (match-any is implicit).

**Token choice (evaluation dimensions)**: only `skeleton·element·aromaticity·bond·ring` are used.
`hcount/degree/valence` (change under substitution → false rejects), `stereo` (dropped from the
description), and `charge` (neutral) are **not** used — see the token table in §3.

**Composite decomposition example** — `eval_query.decompositions[0]` for
`c1ccc(CN2CCC3CCNC[C@@H]3C2)cc1`:
```
name: "two fused piperidine rings"   topology: ortho-fused
components:
   [#7]1-[#6]-[#6]-[#6]-[#6]-[#6]-1        # a piperidine ring
   [#6]1-[#6]-[#6]-[#7]-[#6]-[#6]-1        # a piperidine ring
pair_fusions: [{i:0, j:1, relation:"fused", shared:2, shared_adjacent:true}]   # share 2 adjacent atoms
```

---

## 5. Record (JSONL) schema — every key

One line of `scaffolds.jsonl` = one JSON object with the keys below.

| key | type | meaning |
|-----|------|---------|
| `input_smiles` | str | the original input SMILES. **Key used for resume / dedup.** |
| `smiles` | str | RDKit canonical SMILES of the input molecule. |
| `parse_ok` | bool | whether RDKit parsed the input. |
| `has_scaffold` | bool | whether a scaffold was extracted. Ring molecules use the Murcko scaffold; ring-free molecules fall back to a functional-group framework, so this is true whenever a functional-group anchor exists. false only for e.g. saturated hydrocarbons with no anchor. |
| `scaffold_kind` | str | `ring` (Murcko) / `functional_group` (acyclic FG framework) / `none`. |
| `functional_groups` | list[str] | (FG framework only) functional groups confirmed by SMARTS inside the framework; usually empty for ring scaffolds. |
| `scaffold_smiles` | str | scaffold SMILES. ring: **Bemis–Murcko** (rings+linkers, substituents stripped; on-ring/linker =O/=S kept). functional_group: anchors (heteroatoms / unsaturated carbons) plus the linkers between them, terminal alkyl/halide side chains removed. |
| `scaffold_smarts` | str | SMARTS of the whole scaffold (basic substructure verification). |
| `dimension_smarts` | obj | the scaffold's **single projection** SMARTS: `{skeleton, element, aromaticity, bond, ring}` (§3). |
| `eval_query` | obj | the **evaluation spec** (fixed → single SMARTS, free → match-any). Sub-keys below. |
| `n_ring_systems` | int | number of ring systems. |
| `n_rings_total` | int | total number of rings (SSSR). |
| `aromaticity` | str | overall aromaticity: `all aromatic` / `all aliphatic (saturated)` / `mixed aromatic/aliphatic`. |
| `ring_systems` | list | per-ring-system info (table below). |
| `connections` | list | how ring systems are joined (linker / direct bond) (table below). |
| `linker_attachment_points` | list | attachment points on linkers (when present). |
| `topology_summary` | str | one-line summary of the overall topology. |
| `template_draft` | str | the **deterministic draft** before LLM polishing (template). |
| `description` | str | the **final natural-language description (the training target text)** — LLM-polished and faithfulness-checked. |
| `scaffold_verified` | bool | `scaffold_smarts` confirmed as a substructure of the input molecule. |
| `analysis_error` | str | (only on analysis failure) error message. |

### `eval_query` sub-keys
| key | type | meaning |
|-----|------|---------|
| `n_members` | int | size of family J. **1 = all fixed (single SMARTS), >1 = free choices exist (set)**. |
| `n_free_slots` | int | number of enumerated free attachment points. |
| `verdict_match_any` | list[str] | **authoritative** full-SMARTS list. The molecule contains the scaffold if it matches **any one**. |
| `dimension_sets` | obj | per-dimension match-any sets: `{skeleton:[...], element:[...], ...}`. For partial credit / diagnostics. |
| `decompositions` | list | decomposition spec for composite cores (table below). `[]` if none. |

### `eval_query.decompositions[]` sub-keys
| key | type | meaning |
|-----|------|---------|
| `system_id` | int | target ring-system id (index into `ring_systems`). |
| `name` | str | composite name (e.g. `two bridged cyclopentane rings`). |
| `topology` | str | internal topology: `ortho-fused`/`spiro`/`bridged`/`fused/spiro`. |
| `n_components` | int | number of component rings. |
| `components` | list[str] | SMARTS for each component ring. |
| `pair_fusions` | list | how ring pairs are joined: `{i, j, relation, shared(#shared atoms), shared_adjacent}`. |

### `ring_systems[]` sub-keys
| key | type | meaning |
|-----|------|---------|
| `name` | str | ring-system name (`pyridine`, `quinazoline`, `benzimidazol-2(3H)-one`, or a composite descriptive name). |
| `name_source` | str | `curated` (dictionary) / `curated-carbonyl` (lactam etc.) / `monocycle` (systematic) / `composite` (descriptive). |
| `n_rings` | int | number of rings in this system. |
| `aromatic` | bool | fully aromatic? |
| `aromaticity` | str | `aromatic` / `saturated / non-aromatic` / `partially aromatic ...`. |
| `internal_topology` | str | `single`/`ortho-fused`/`spiro`/`bridged`/`fused/spiro`. |
| `ring_sizes` | list[int] | ring sizes. |
| `heteroatoms` | obj | element → count (e.g. `{"N":1}`). |
| `n_ring_carbonyls`,`carbonyl_groups`,`carbonyl_desc` | (optional) | present when an exocyclic C=O/C=S sits on a ring atom (lactam / lactone / cyclic urea …). |

> **Substituent attachment points are NOT stored.** Substituents are stripped from the Bemis–Murcko
> scaffold, so "which ring position a substituent occupies" (a ring 2,5-position, a benzene
> ortho/meta/para pattern) is **unverifiable by any evaluation SMARTS**. Hence `attachment_points` /
> `n_attachment_points` / `substitution_relations` are omitted from the data and from the description;
> only the verifiable **linker-attachment position** (the linker is part of the scaffold) is kept, in
> `connections[].a_position/b_position`.

### `connections[]` sub-keys
| key | type | meaning |
|-----|------|---------|
| `a`,`b` | int | the two ring-system ids being joined. |
| `relation` | str | `linker-connected` / `directly linked` / (biaryl etc.). |
| `type` | str | linker name (e.g. `amide linker`, `thiourea linker`, `7-atom linker (amide + oxime ether)`). |
| `linker_pattern` | str | the **readable connectivity shape** (incl. pendant =O/=S and bond orders), e.g. `-NH-C(=O)-`, `-S(=O)2-NH-`. |
| `functional_groups` | list[str] | recognized groups (`amide`,`urea`,`oxime ether`…). |
| `linker_length` | int | number of linker backbone atoms. |
| `bond_span` | int | number of bonds between the two ring atoms. |
| `linker_atoms` | list | backbone atom sequence: `{index, element, nH, pendant:[{element,order}]}`. |
| `a_atom`,`b_atom` | int | the ring atom each end attaches to. |
| `a_position`,`b_position` | str | the attachment-position label (**fixed**: `position 4 of the pyridine` / **free**: `a ring carbon`, `three bonds from a ring nitrogen`). |

**Example (one connection + one ring_system)** — `O=C(Nc1ccccc1)c1ccncc1`:
```json
{"a":0,"b":1,"relation":"linker-connected","type":"amide linker",
 "linker_pattern":"-NH-C(=O)-","functional_groups":["amide"],"linker_length":2,"bond_span":3,
 "linker_atoms":[{"index":2,"element":"N","nH":1,"pendant":[]},
                 {"index":1,"element":"C","nH":0,"pendant":[{"element":"O","order":"="}]}],
 "a_atom":3,"b_atom":9,"a_position":"a ring carbon","b_position":"position 4 of the pyridine"}
{"name":"pyridine","name_source":"curated","n_rings":1,"aromatic":true,
 "aromaticity":"aromatic","internal_topology":"single","ring_sizes":[6],"heteroatoms":{"N":1}}
```
Here `b_position` is **fixed** ("position 4") → encoded as-is in evaluation; `a_position` is **free**
("a ring carbon") → but the benzene is symmetric so it collapses to one candidate. (Hence this
molecule has `eval_query.n_members == 1`.)

---

## 6. How to evaluate (summary)

For a generated molecule `mol`:
1. **Binary verdict (authoritative)**: pass if **any** SMARTS in `eval_query.verdict_match_any`
   satisfies `mol.HasSubstructMatch(SMARTS)`. (If composite cores are present, also confirm
   `eval_query.decompositions` via `match_decomposition(mol, decomp)`.)
2. **Partial credit (diagnostic)**: a dimension passes if `mol` matches any SMARTS in
   `eval_query.dimension_sets[dim]`. Note skeleton is the floor (subsumed by the others), so it is not
   recommended as a standalone score.

Helpers such as `match_decomposition` live in `2_substructure_gen/scaffold_analyzer.py`.

---

## 7. Generation pipeline / reproduction

```
SMILES → Murcko scaffold extraction → ring-system decomposition & naming → linker & fused/spiro/bridge
       → substituent removal + attachment recording → deterministic draft → vLLM (Qwen3) polishing
       → faithfulness check → SMARTS/RDKit substructure verification
       → dimension_smarts + eval_query (fixed/free/decomposition)
```
Code: `2_substructure_gen/` (`scaffold_analyzer.py` analysis/SMARTS, `scaffold_prompts.py` prompts,
`scaffold_describer.py` templates/validation, `describe_scaffolds.py` vLLM client + CLI).

Reproduce (example):
```bash
PY=python
$PY 2_substructure_gen/describe_scaffolds.py \
    --input data/develop/chembl_zinc_split/train_pool.parquet \
    --run-name scaffold_train_pool --limit 2000000
# -> <out-dir>/<run-name>/{scaffolds.jsonl, meta.json, README.md, README.en.md}
```

**Faithfulness principle**: descriptions and SMARTS state only what the scaffold *actually guarantees*.
Unprovable IUPAC locants are never invented (only positions a chemist would naturally state are fixed);
everything else is left free and handled by match-any at evaluation time.
