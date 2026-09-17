"""Prompt templates for the Generator, Augmentor, and Verifier LLMs.

This stage targets the **constructive substructure-generation** tool chain
produced by ``3_toolchain_gen`` (task_type ``generation``,
chain_strategy ``constructive``).  Each chain builds a molecule step by step
from a starting core/seed using structural editing tools — primarily
``label_atom_indices`` → ``attach_fragment`` rounds, with optional
``set_stereochemistry`` — and verifies the result with a parallel
``match_substructure`` + ``analyze_properties`` checkpoint.

The prompts below ask the Generator LLM to write the natural-language
reasoning that connects those ground-truth tool calls into a coherent
molecule generation trajectory.
"""

# ---------------------------------------------------------------------------
# System prompt embedded in every generated training example
# ---------------------------------------------------------------------------
# Minimal system prompt: the tool-using generation protocol (introduce a core, build
# it incrementally one edit per turn, emit <ANSWER> when done) is taught by the SFT
# examples
# themselves rather than by system-prompt instructions. Kept in sync with the copy
# in the training-time eval harness (not part of this repository).
SYSTEM_PROMPT_TEMPLATE = "You are an expert molecular generation AI assistant."

# ---------------------------------------------------------------------------
# Generator – seed introduction (first step: the seed 3-way checkpoint on the
# complete scaffold — match_substructure ∥ analyze_properties ∥
# label_atom_indices)
# ---------------------------------------------------------------------------
# DIRECT complete-scaffold seed: the chain starts from the FULL required scaffold
# (not a hub core). The seed-intro reasoning must, in ONE flow, (a) READ the
# substructure description and DERIVE the SMARTS pattern that encodes the whole
# required substructure, and (b) WRITE a concrete SMILES for a molecule that
# contains exactly that substructure — the starting scaffold it will build on.
# Then it verifies (match_substructure), measures (analyze_properties),
# and labels (label_atom_indices) that scaffold in one parallel checkpoint.
SEED_INTRO_PROMPT = """\
You are generating high-quality training data for a molecular generation AI assistant.

## Task
The assistant is STARTING a molecule generation task from scratch. It (a) reads the
described structure and writes the whole SMARTS substructure query in one step,
(b) transcribes that SMARTS directly into a concrete starting-scaffold SMILES, and
(c) verifies / measures / labels the scaffold in one parallel checkpoint.

## Instructions
- Write 3-5 short first-person sentences. DECISIVE FORWARD reasoning — no scratchpad,
  no "wait", "let me re-read", "actually", "hold on", "hmm".
- Sentence 1-2: describe the required substructure at a HIGH LEVEL — name the rings and
  how they connect (e.g. "a purine core linked through an alkyne to a phenyl, with an
  N-benzyl group"), enough to justify the pattern. Do NOT walk it atom-by-atom.
- Then write the FULL SMARTS from the "Committed SMARTS" block below directly, in ONE
  step, as the machine-checkable form the match_substructure query will use — character
  for character, not built up fragment by fragment.
- Then TRANSCRIBE that SMARTS straight into SMILES — read it off and write each atom as a
  concrete atom (aromatic ``[#6]:[#6]`` → ``c``, ``[#16](=[#8])(=[#8])`` → ``S(=O)(=O)``,
  ``[#7]`` → ``N``/``n``, etc.). Write the string from the "Starting-scaffold SMILES"
  block below EXACTLY as given and EXACTLY ONCE, in a single sentence that both reports
  the transcription and states it is the starting scaffold you will build on. NO EXAMPLE
  of that sentence is given on purpose — word it yourself and differently every time. In
  particular do not end it on "which I take as the starting scaffold", "which I will use
  as the starting scaffold", or any near-copy: that clause is worn out. Never write the
  SMILES a second time — a follow-up sentence repeating it is pure duplication. Do NOT
  mention canonicalization or a "canonical form" — the transcription IS the scaffold.
- The aromatic-hydrogen block below decides whether that read-off is mechanical. A SMARTS
  ``[#7]`` carries NO hydrogen count, so it matches both ``n`` and ``[nH]``. If the block
  names any atom, say so in that same transcription sentence — which ring N-H has to be
  written ``[nH]``, and that a bare ``n`` there would not kekulise — because that H is
  chemistry you supply, not something the pattern told you. If the block says "(none)",
  do not mention hydrogens or kekulisation at all.
- The committed SMARTS and SMILES are AUTHORITATIVE — commit to them; do not dispute the
  informal description or re-derive them differently. But do NOT borrow this prompt's
  vocabulary for them: never call the pattern "authoritative", "ground truth", or "the
  committed SMARTS", and refer to what was asked for as "the query" / "the description",
  never "the prompt". You are the one deciding on the pattern — write it that way.
- Close by stating the three checks you are about to run together: the SMARTS match, the
  property measurement, and the atom-index labelling. Word that closing sentence
  DIFFERENTLY every time — no fixed formula, and in particular do not end on "label its
  atom indices in a single parallel checkpoint" or any near-copy of it. Naming the
  checks in a different order, or leading with what you intend to learn from them, is
  better than a template.
- Do NOT enumerate specific property-target numbers here, and do NOT include the tool
  calls themselves.

## User Query
{user_prompt}

## Required Substructure (described in the query)
{substructure_description}

## Committed SMARTS (ground truth — commit to exactly this)
{substructure_smarts}

## Starting-scaffold SMILES = the SMARTS transcribed to SMILES (ground truth — use this EXACT string)
{seed_smiles}

## Aromatic hydrogens the pattern does NOT carry (RDKit-computed — AUTHORITATIVE)
{aromatic_h_note}

## Verified ring names (RDKit-computed — AUTHORITATIVE)
{ring_locant_note}

Reasoning (3-5 sentences; the full SMARTS, then that scaffold SMILES exactly once):"""


# ---------------------------------------------------------------------------
# Generator – seed introduction for a FUNCTIONAL-GROUP constraint (2b)
#
# Why this cannot reuse SEED_INTRO_PROMPT: that prompt tells the assistant to
# TRANSCRIBE the committed SMARTS into the starting SMILES, which is honest only
# because a Murcko scaffold IS a molecule — the SMARTS and the seed are one object
# written twice. A functional group is not. `C(=O)-N` transcribed gives formamide,
# and two required groups transcribe to two DISCONNECTED fragments, which is not a
# molecule at all. The seed here is CONSTRUCTED (fg_seed.py: a carrier that already
# holds every required group, or two carriers bonded where both survive), so the
# reasoning has to be "I pick a small molecule that carries these groups", never "I
# read the pattern off into SMILES".
#
# The other difference is the check: several required groups mean several SMARTS and
# therefore several `match_substructure` calls. They must NOT be joined with '.' — a
# dot-joined query demands atom-disjoint matches and so rejects a molecule whose
# amide N is also its piperazine N (measured: 13.1% of two-group constraints).
# ---------------------------------------------------------------------------
FG_SEED_INTRO_PROMPT = """\
You are generating high-quality training data for a molecular generation AI assistant.

## Task
The assistant is STARTING a molecule generation task from scratch. The query asks for a
molecule containing one or more FUNCTIONAL GROUPS. It (a) reads the required groups
and writes the SMARTS pattern for each, (b) CHOOSES a small starting molecule that
already carries all of them, and (c) verifies / measures / labels it in one parallel
checkpoint.

## Instructions
- Write 3-5 short first-person sentences. DECISIVE FORWARD reasoning — no scratchpad,
  no "wait", "let me re-read", "actually", "hold on", "hmm".
- Sentence 1-2: say what the required group(s) ARE as chemistry — the atoms, the bonds
  between them, and any exclusion the description states. Do NOT walk them atom-by-atom
  beyond that; one clause each is enough.
- Then write the SMARTS from the "Committed SMARTS" block below, character for
  character. If there is more than one, write each one — they are separate queries and
  each gets its own check. NEVER join them with a '.' into a single pattern.
- Then name the starting molecule from the "Starting molecule" block below EXACTLY as
  given and EXACTLY ONCE. This is a molecule you CHOOSE because it already contains
  the required group(s) and is small enough to build on — say it that way. Do NOT say
  you transcribed, read off, or converted the SMARTS into it: you did not, and that
  would be a false account of where it came from. Word this sentence differently every
  time; do not settle into a formula. Never write the SMILES a second time.
- Say nothing about how the molecule was built or where it came from beyond it being
  your chosen starting point. Do not mention carriers, joining, or construction.
- The aromatic-hydrogen block below decides whether the SMILES needs an explicit
  ``[nH]``. If it names any atom, say which ring N-H has to be written ``[nH]`` and
  that a bare ``n`` there would not kekulise. If it says "(none)", do not mention
  hydrogens or kekulisation at all.
- The committed SMARTS and starting molecule are AUTHORITATIVE — commit to them. But
  do NOT borrow this prompt's vocabulary: never call them "authoritative", "ground
  truth", or "committed", and refer to what was asked for as "the query" / "the
  description", never "the prompt".
- Close by stating the checks you are about to run together: the substructure match
  (one per required group), the property measurement, and the atom-index labelling.
  Word that closing sentence DIFFERENTLY every time — no fixed formula.
- Do NOT enumerate specific property-target numbers here, and do NOT include the tool
  calls themselves.

## User Query
{user_prompt}

## Required functional group(s) (described in the query)
{substructure_description}

## Committed SMARTS (ground truth — one per required group, commit to exactly these)
{fg_smarts_block}

## Starting molecule — already contains every required group (use this EXACT string)
{seed_smiles}

## Aromatic hydrogens the pattern does NOT carry (RDKit-computed — AUTHORITATIVE)
{aromatic_h_note}

## Verified ring names (RDKit-computed — AUTHORITATIVE)
{ring_locant_note}

Reasoning (3-5 sentences; each SMARTS, then that starting molecule exactly once):"""


# ---------------------------------------------------------------------------
# Generator – reasoning before a PROPERTY-DRIVEN decorate edit
# (`edit_fragment` — attach / swap / remove — used to tune properties)
# ---------------------------------------------------------------------------
DECORATE_EDIT_REASONING_PROMPT = """\
You are generating high-quality training data for a molecular generation AI assistant.

## Task
The required substructure is already present. Write reasoning for a structural
edit whose PURPOSE is to bring one or more out-of-range properties into their
target ranges. Ground the reasoning in the property gap and the direction each
property must move.

## User Query
{user_prompt}

## Property Status of the Current Molecule (before this edit)
{property_gap}

## Next Action
Tool: {tool_name}
Edit: {tool_arguments}
Current labelled molecule: {labeled_atoms}

## Instructions
- 2-4 short sentences.
- Note the required substructure is already in place, so this edit is a property
  adjustment (NOT scaffold building).
- STATE THE INTENT EXPLICITLY: name the specific out-of-range property/properties
  (current value + target) and make the DIRECTION unmistakable — which property you
  are RAISING and which you are LOWERING with this edit — then explain CHEMICALLY how
  the edit shifts them. For ``replace_fragment``, contrast the OLD group vs the NEW
  group (``from_smiles`` → ``to_smiles``) and why that swap moves the targeted property
  the right way (e.g. swapping an alkyl for a polar group LOWERS logP and RAISES TPSA).
  Use the EXACT direction the Property Status block gives (INCREASE / DECREASE) — do
  not invert it (for a negative-valued property like logS, "DECREASE" = MORE negative).
- Identify the SITE: for ``replace_fragment`` say which group is being swapped out and
  what replaces it; for ``form_bond`` name the two labelled atoms being joined (read
  ``atom_index_1`` / ``atom_index_2`` off the labelled molecule) and the ring/closure
  it forms.
- Phrase as an INTENTION. Do NOT state or predict a resulting SMILES, and do NOT
  include the tool call. Do NOT claim all targets are now met.

Reasoning:"""

# ---------------------------------------------------------------------------
# Generator – reasoning before a suggest_edits call (the edit's lead-in)
# ---------------------------------------------------------------------------
SUGGEST_REASONING_PROMPT = """\
You are generating high-quality training data for a molecular generation AI assistant.

## Task
The required substructure is already present, but some properties are still out of
range. Write reasoning for calling the ``suggest_edits`` tool, which ranks candidate
substituent edits (each a ready-to-use ``edit_fragment`` argument set) by how much
they move the molecule toward the target property box.

## User Query
{user_prompt}

## Property Status of the Current Molecule
{property_gap}

## Instructions
- EXACTLY 2 sentences, and the second one SHORT (12 words or fewer).
- Sentence 1 — the specific out-of-range property/properties: current value, target, and
  the DIRECTION each must move. Quote every value EXACTLY as the status block gives it,
  with all of its decimals ("48.150", not "48.2") — a shortened value no longer matches
  the measurement it refers to. This is where the substance goes; vary how you say it.
- Sentence 2 — just that you are calling ``suggest_edits`` for edits that move those
  properties. Do NOT explain why a tool beats a guess, and do NOT praise the tool: any
  clause like "leveraging the tool's ability to …", "data-driven candidate edits ranked
  by …", "rather than guessing / heuristic trial-and-error", "based on quantitative
  impact" is padding — it says nothing about THIS molecule and reads identically in
  every round. Twelve plain words are better than thirty decorative ones.
- Phrase as an INTENTION. Do NOT invent candidate edits or a resulting SMILES, and do
  NOT include the tool call itself.

Reasoning (2 sentences, the second one 12 words or fewer):"""

# ---------------------------------------------------------------------------
# Generator – reasoning before the edit_fragment that applies a chosen candidate
# ---------------------------------------------------------------------------
EDIT_FRAGMENT_REASONING_PROMPT = """\
You are generating high-quality training data for a molecular generation AI assistant.

## Task
``suggest_edits`` has just returned its ranked candidates. Write the reasoning that
CHOOSES one of them, in the order a chemist actually decides: first what the molecule
still needs, then which candidate looks likeliest to deliver it, and only then a
TENTATIVE commitment — the numbers are predictions, so the edit is an attempt whose
result will be measured, never a settled outcome. The required substructure is already
present, so this is a property adjustment, not scaffold building.

## Decide in This Order (the Landing Safety block below has already done the counting)
1. ELIMINATE. A candidate that breaks a constraint which is satisfied now, or that misses
   an exact-value target (heavy_atoms, HBA, rotB with equal bounds), is out — say which
   candidates that removes.
2. GAP CLOSED. Among what is left, the one that removes the largest share of the current
   box distance. The share, not the absolute distance: the distance does not compare
   across molecules of different size. Word it differently every time and never open two
   consecutive rounds the same way. When the share is 100% for more than one candidate it
   has decided nothing, so move on. A negative share means the edit makes the gap WORSE;
   name that as the cost.
3. SAFE LANDINGS. Constraints that end up inside the box with a full predicted std of
   room. Use this when the gap share ties, and to break the tie when two candidates close
   the same share by different routes. Read it against "IN after" in the same row: a
   candidate at 5/10 safe but 9/10 IN AFTER lands almost everything, just with thin
   margins — say that rather than "only five work".
4. OVERSHOOT. A candidate that carries an axis it was fixing PAST the far edge of its box
   is not a good move even when it closes 100% of the gap on paper. When the block reports
   an overshoot, that candidate has to be named and the overshoot said out loud.
5. SIZE. ΔMW, then heavy atoms. This is step 1 material when MW or heavy_atoms is itself a
   target; otherwise it is the tie-break when nothing above separates — take the smallest
   edit and say why, in your own words, differently each time.
6. RANK. The tool's own order (#1 is its first choice) is the last resort, and only
   because everything measured came out level.
7. You do NOT have to walk the whole order out loud, and a criterion that decides nothing
   is not worth words. Name the criterion that actually DECIDES, and at most one it had to
   pass through. If you do invoke a criterion the block calls a tie, say it is a tie —
   never assert a lead on one. The block also lists, in one line, every column that is
   IDENTICAL across the candidates: those decide nothing, so do not argue from them and do
   not narrate them either.

## This Round (COMPUTED — AUTHORITATIVE) — posture, never the choice
The block below is the same for every candidate, so nothing in it can pick one. It says
how far to trust the numbers that follow:
- the tightest CEILING is the constraint an aggressive edit breaks first — it says how
  much overshoot this round can afford;
- whether the LAST edit actually reduced the box distance. When it did not, this round is
  a correction and the sentence should say so;
- how far the LAST prediction landed from the measurement. When that line reports a MISS,
  the hedge in your last sentence has to be sized by it and must QUOTE the miss, not fall
  back on a generic "should". A Δ below is the same kind of estimate that was just shown
  to be off by that amount. When the line says the prediction was exact, say the tool is
  calibrated here and hedge lightly. Word it differently every time;
- the EDIT SITE, and whether it separates the candidates at all. It usually does not (on
  four candidate sets in five every candidate attaches at the same kind of atom), and the
  line says which case this is. Argue from the site ONLY when the line says it separates;
  otherwise name it just to say what the transformation IS.

## The block's column names are NOT English
Quote the block's NUMBERS, never its headers. "gap closed", "safe", "overshoot",
"+heavy", "ΔMW", "breaks", "rank" are column labels in a table — a sentence built out of
them reads as a spreadsheet, not as reasoning. Say what the number MEANS instead: "closes
93% of the distance", "lands eight of the twelve with a full std to spare", "carries QED
0.021 past its 0.890 ceiling", "adds three heavy atoms and 44 daltons", "breaks nothing",
"the tool's own first choice".

## How to Read the Candidate List Below
Each line gives the edit rule and, per constrained property, the current value, the
mmpdb-predicted change as ``Δ avg ± std`` (std = spread over the matched pairs: small
= a reliable, transferable shift; large = the shift varies by context, so a bounded
target can be overshot or missed), the resulting predicted value, and whether that
lands inside the target range. The ``[…-tightest of N]`` tag on each ± value is
COMPUTED — use it as given; never re-derive which spread is tighter yourself.

## Instructions
Write EXACTLY 3 sentences, 65 words or fewer IN TOTAL. Terse and concrete — every
sentence must carry a number or a name, no scene-setting, no restating the task.
Reasoning first, choice last: never open with the commitment.

1. Sentence 1 — the gap that DRIVES this choice: the property (at most two), its
   current value, its target, and the DIRECTION it must move. Use the EXACT direction
   the Property Status block gives (for a negative-valued property like logS,
   "DECREASE" = MORE negative). Do not list constraints that already pass.
2. Sentence 2 — the ONE criterion that decides, quoting the Landing Safety block: which
   candidates it eliminates, or the count of the one you take against the ones you pass
   over. Two criteria at the very most, and only when the first genuinely failed to
   separate them. When the block declares a criterion a tie, your
   sentence must say it does not separate them and move on; a spread comparison is worth
   words only when it is what actually decides between the survivors.
   NEVER compute a count, a share, or a spread comparison yourself — take every number
   from the Landing Safety block and the candidate list as written. Which ± is tighter is
   already decided
   in the Spread Comparison block and in the ``[…-tightest of N]`` tags — say only
   what they say. Do not write "±0.14 is tighter than ±0.05". When the block says the
   candidates TIE, say the spread does not discriminate and decide on the Δ instead;
   when the selected candidate is NOT the tightest, do not claim it is — say the
   larger Δ is worth the looser spread (or that the tighter option moves the wrong
   property), which is the real reason it was taken.
3. Sentence 3 — take the candidate as a TRIAL to be measured, and state what is STILL
   WRONG after it: name a constraint the edit leaves outside its box (with the value the
   table predicts), or a landing whose room is smaller than its own ±. If the Landing
   Safety block did not settle the decision, give the condition for undoing it instead:
   name the measurement that would send you back and the candidate you would take. Word
   that condition differently every time — there is no set phrase for it, and a sentence
   that reads as a formula teaches the formula. Never end on a settled result.

## Uncertainty is mandatory
Every Δ is an mmpdb PREDICTION averaged over matched pairs — never a measurement.
Never write an outcome as settled ("this raises MW to 409", "this brings logP into
range", "the result will be …"). Instead:
- hedge the prediction: "should", "is predicted to", "ought to land near", "only
  ~+0.4 on paper";
- make the spread the REASON for the hedge: "±0.62 is wide enough to overshoot the
  3.5 ceiling", "±0.05 leaves little room to miss";
- make the commitment provisional: "so apply … first and re-measure", "try …",
  "worth attempting before touching logP".
Model the shape "candidate A moves it further, but its ±X spread on <property> makes
that unreliable, so apply B first and check" — the choice is a next experiment.

- State the candidate's rank exactly as truthfully as the "Rank of this candidate" line
  does. Never call it the top / highest-ranked candidate unless that line says it is.
  When it is not rank 1, sentence 2 must say WHY it beats the higher-ranked ones for
  THIS property gap (spread, overshoot risk, wrong direction on another target, wrong
  heavy-atom count, …).
- Write flowing prose in your own words: never copy a field label from this prompt
  ("Rank of this candidate", "Edit site", "Group being attached", …) into a sentence.
- QUOTE EVERY NUMBER EXACTLY as it appears above — neither dropping decimals nor adding
  them. Write "48.150", never "48.2"; "176.059", never "176.1"; "+58.042", never "+58",
  because a shortened value no longer matches the measurement the next checkpoint will
  show. Equally, an integer count is written as the block writes it: HBD is "1" and
  heavy_atoms is "15", never "1.000" or "15.000" — padding a count with decimals invents
  a precision the property does not have. The same holds for the Landing Safety block's
  own cells: a share is written as it prints it ("+72%", "93%"), never "+72.000%".
- NAMING IS CONSTRAINED: name the removed/attached groups and the edit site ONLY with
  the names in the Verified Chemistry Facts block. Do NOT invent or guess a chemical
  name that is not listed there — if no name is given, refer to the fragment by its
  SMILES or by the ring/group names listed. Never rename a ring (a pyrrolidine is not
  a piperazine, a pyrrole is not a pyridine, a carboxamide is not a urea).
- When from_smiles is a bare attachment point, the edit substitutes a HYDROGEN — do not
  describe it as replacing a methyl or any other group.
- Describe the edit SITE only as the "Edit site" facts state (element, aromatic or not,
  and the ring system it is in). If the site is not in a ring, do NOT say the edit
  happens on a ring. The site is there so the transformation can be NAMED — it is not a
  reason for the choice, so do not argue from it (measured across candidate sets, the
  attachment site is identical for every candidate far more often than not, so "I chose
  it because the site is aromatic" is almost always a claim about nothing).
- Phrase as an INTENTION. Do NOT state or predict a resulting SMILES, do NOT include
  the tool call, and do NOT claim all targets are now met.

## User Query
{user_prompt}

## Property Status of the Current Molecule (before this edit)
{property_gap}

## Candidates suggest_edits Returned
{candidate_options}

## This Round (COMPUTED — AUTHORITATIVE) — same for every candidate
{round_context}

## Landing Safety (COMPUTED — AUTHORITATIVE, never recompute)
{landing_safety}

## Spread Comparison for the Out-of-range Properties (COMPUTED — AUTHORITATIVE)
{spread_verdict}

## The Candidate to Select
Action: {action}
edit_fragment args: from_smiles={from_smiles}, to_smiles={to_smiles}, anchors={anchors}
Rank of this candidate: {candidate_rank}
THIS IS CANDIDATE {picked_number}. Any candidate number your sentence names as
the one being applied must be that number — the other numbers may only appear as
the options you passed over or the fallback you would revert to.
Current labelled molecule: {labeled_atoms}

## Verified Chemistry Facts (RDKit-computed — AUTHORITATIVE)
Group being removed  (from_smiles): {from_desc}
Group being attached (to_smiles):   {to_desc}
Edit site:
{anchor_facts}

Reasoning (3 sentences, ≤65 words; quote every number with all its decimals; every Δ is
a prediction, so hedge it — "should", "is predicted to" — and take the candidate as a
trial to re-measure, never as a settled result):"""

# ---------------------------------------------------------------------------
# Generator – reasoning for an edit AUTHORED after an EMPTY suggest_edits
# ---------------------------------------------------------------------------
# Counterfactual counterpart of EDIT_FRAGMENT_REASONING_PROMPT: same round, but
# suggest_edits is rendered as `[]`, so there is no list to choose from and the
# assistant must derive the edit itself. Installs the missing `[]` branch — see
# PipelineConfig.authored_edit_fraction for why it exists and what it cannot fix.
#
# Unlike the C-only variant this template DOES hand the model the committed
# edit's mmpdb Δ. It is available (the planner measured it) and withholding it
# was what pushed the generator into inventing causal claims — measured: "adding
# lipophilic bulk raises Mutag", "lowering logP improves BBB penetration". Quote
# the Δ instead of theorising about mechanism.
EDIT_FRAGMENT_AUTHORED_PROMPT = """\
You are generating high-quality training data for a molecular generation AI assistant.

## Situation
The required substructure is already present, but some properties are still out of
range. The assistant called ``suggest_edits`` and **it returned an EMPTY list** — no
library move closes the remaining gap in one step without breaking the scaffold.

So there is no candidate to pick and no mmpdb Δ table to read. The assistant has only
what it can see: the constraint status it is already tracking, the labelled molecule,
and its own chemistry knowledge. Write the reasoning in which it decides the edit below
on that basis alone, and commits it as a trial to be measured.

## THE ONE RULE THAT MATTERS
Write only what the assistant could actually have worked out in this situation. Every
number it states must come from the Property Status block (values it already measured)
or from the Countable block (arithmetic on the fragment). Anything else it cannot know
yet — the point of the edit is to FIND OUT, and the checkpoint right after will measure
it. A sentence that asserts an unknowable number is worse than one that admits the
uncertainty.

## Instructions
Write 3 sentences, 70 words or fewer IN TOTAL. Terse and concrete. Reasoning first,
commitment last.
1. Sentence 1 — that suggest_edits returned nothing, and the gap that now drives the
   choice: the property (at most two), its current value, its target, and the DIRECTION
   it must move. Use the EXACT direction the Property Status block gives (for a
   negative-valued property like logS, "DECREASE" = MORE negative).
2. Sentence 2 — why this fragment:
   * if the blocking property is in the Countable block, DO THE ARITHMETIC — the
     fragment is worth N heavy atoms, or this much mass, so the value lands here. That
     is a real derivation and it is the strongest thing you can write.
   * if the blocking property is one the measurement has to settle, choose on what you
     CAN see — the change is small, it keeps the scaffold, it leaves the satisfied
     constraints alone — and put the open question in the hedge.
3. Sentence 3 — commit it, naming the transformation: which group leaves, which
   arrives, where it attaches, and that you re-measure.

## Voice
Write as the assistant thinking, not as someone following a rubric.
- NO markdown: no backticks, no asterisks. Plain prose. Tool and property names appear
  bare (suggest_edits, heavy_atoms), never in backticks.
- NEVER use this prompt's vocabulary about knowledge itself. Banned: "derivable",
  "not derivable", "non-derivable", "countable block", "predictable from the
  structure", "manually", "unknowable". They are how this prompt is organised, not how
  a chemist talks. Express the same thing through the hedge instead — "only the
  measurement will show whether …", "how far it moves X is what the checkpoint is
  for", "I cannot tell in advance whether …".
- Do not narrate the procedure ("so I probe X to INCREASE"). State the chemistry.

## The two shapes, as reasoning steps — NOT as sentences to copy
Do not lift wording from here. These describe what has to be established; the words
are yours.

When the blocking property is one the fragment fixes:
  1. the gap, as a number, and which way it must go
  2. the arithmetic — what the fragment is worth, and therefore where the value lands
  3. which satisfied constraints the fragment cannot disturb (no ring added, no donor
     added, whatever applies)
  4. the transformation and the measurement

When the blocking property is one only the measurement can settle:
  1. the gap, as a number, and which way it must go
  2. the grounds you actually have: this is the smallest change available, it keeps the
     scaffold, it does not touch what already passes
  3. the open question, in the hedge — how far it moves that property is not something
     you can settle here
  4. the transformation and the measurement

## Vary the voice — this round
{voice_hint}
Two rounds that open the same way are a defect: the phrasing is being copied instead of
reasoned. Never open with a formula you would reuse verbatim next round.

## Hard rules
- NEVER mention a candidate, a ranked list, a rank, "Candidate #N", "top-ranked", or a
  predicted_gap. There is no list. Any such reference is a factual error.
- NEVER state a numeric change for a property listed under "only the measurement can
  settle these". No "Δ +0.038 for BBBP", no "raises Mutag by 0.114", no invented value
  and no rounded one. Name the direction it must move and leave the size to the
  checkpoint.
- NEVER theorise a mechanism for those endpoints either. "Adding lipophilic bulk raises
  Mutag", "lowering logP improves BBB penetration" — these read as chemistry but they
  are guesses about a model output. Leave the question open in the hedge instead.
- QUOTE EVERY NUMBER EXACTLY as given below, with all its decimals — "48.150", never
  "48.2".
- NAMING IS CONSTRAINED: name the removed/attached groups and the edit site ONLY with
  the names in the Verified Chemistry Facts block. Do NOT invent or guess a chemical
  name that is not listed there — if the block says there is no curated name, refer to
  the fragment by its SMILES. Never rename a ring or a group.
- When from_smiles is a bare attachment point, the edit substitutes a HYDROGEN — do not
  describe it as replacing a methyl or any other group, and do not name a site the
  Edit-site facts do not name.
- The outcome is unmeasured: hedge it ("should", "ought to land near") and make the
  commitment provisional ("apply and re-measure").
- Do NOT state a resulting SMILES and do NOT include the tool call itself.

## User Query
{user_prompt}

## Property Status of the Current Molecule (before this edit)
{property_gap}

## Current labelled molecule
{labeled_atoms}

## The Edit To Commit
Action: {action}
edit_fragment args: from_smiles={from_smiles}, to_smiles={to_smiles}, anchors={anchors}

## Fixed by the fragment — arithmetic the assistant can do, so quote these
{countable_block}

## Only the measurement can settle these — direction only, NEVER a number
{direction_block}

## Already inside their box — the edit must not break these
{protect_block}

## Verified Chemistry Facts (RDKit-computed — AUTHORITATIVE)
Group being removed  (from_smiles): {from_desc}
Group being attached (to_smiles):   {to_desc}
Edit site:
{anchor_facts}

Reasoning (3 sentences, <=70 words; no candidate list exists; state a number ONLY if it
is in the Property Status or Countable block; for anything else say it must be measured):"""


# ---------------------------------------------------------------------------
# Generator – reasoning before the parallel verification checkpoint
# (match_substructure + analyze_properties)
# ---------------------------------------------------------------------------
CHECKPOINT_REASONING_PROMPT = """\
You are generating training data for a molecular generation AI assistant.

## Context
The molecule was just modified. The assistant now verifies it by running
``match_substructure`` (to confirm the required substructure is present) and
``analyze_properties`` (to measure the target properties) IN PARALLEL.

## User Query
{user_prompt}

## The Edit Just Applied
{edit_context}

## Properties That Were Out Of Range Before It (what this edit was aiming at)
{open_gaps}

## Next Action (parallel)
Tool 1: {tool_name_1}
Tool 2: {tool_name_2}

## Instructions
Write ONE short sentence (~12-22 words) saying you are re-checking the substructure and
re-measuring the properties of the molecule as it now stands. No example is given on
purpose: word it yourself, differently each time — nothing that reads as a formula, and
in particular not "measure the (updated) properties of the current molecule". Anchoring
it on what THIS edit was supposed to move ("…and see whether that carbonyl brought TPSA
up") is better than a generic check. Describe it as a progress check on the CURRENT
molecule — do NOT claim the molecule or scaffold is "finalized", "complete", or "done"
(more edits may still follow). Do NOT restate the full constraint list, do NOT quote a
property VALUE (nothing has been measured yet — that is what this call is for), and do
NOT include the tool calls themselves.

Reasoning (ONE sentence, ~12-22 words, worded differently every time):"""

# ---------------------------------------------------------------------------
# Generator – generic fallback reasoning (used for unrecognised solo steps)
# ---------------------------------------------------------------------------
REASONING_PROMPT = """\
You are generating high-quality training data for a molecular generation AI assistant.

## Task
Generate natural, expert-level reasoning that the assistant would produce
before calling a specific tool.

## User Query
{user_prompt}
{requirements_context}
## Conversation So Far
{conversation_history}

## Next Action
The assistant should call: {tool_name}
With arguments:
{tool_arguments}

## Instructions
- Be VERY concise: 1-2 short sentences.
- Explain WHY this tool is needed at this point in the build.
- Do NOT include the tool call itself, only the reasoning text.
- Do NOT restate the user's query verbatim.
{first_step_note}
Reasoning:"""

# ---------------------------------------------------------------------------
# Generator – reflection on a tool response
# ---------------------------------------------------------------------------
REFLECTION_PROMPT = """\
You are generating training data for a molecular generation AI assistant.

## Task
Generate a brief, insightful reflection on a tool response.

## User Query
{user_prompt}

## Conversation So Far
{conversation_history}

## Latest Tool Call
Tool: {tool_name}
Arguments: {tool_arguments}
Response: {tool_response}

## Instructions
Write a VERY brief reflection (1 short sentence, ~15 words) that notes the key
finding or the remaining structural gap vs. the target. No restating of the
query, no full enumeration of values.

Reflection:"""

# ---------------------------------------------------------------------------
# Generator – terminal ANSWER segment (confirms, then emits <ANSWER>). This is a
# SEPARATE final segment: the rule-based constraint check shows every target met,
# so the model confirms and finalises — no tool call, no re-derived Verification
# block.
# ---------------------------------------------------------------------------
ANSWER_CONFIRM_PROMPT = """\
You are generating training data for a molecular generation AI assistant.

## Task
The generation is COMPLETE. The constraint check below shows the current molecule
already CONTAINS the required substructure AND has every property target within
range. Write the assistant's brief closing reasoning that CONFIRMS this by reading
the check and states it is finalising the answer.

## User Query
{user_prompt}

## Constraint check (rule-based — all constraints satisfied)
{constraint_check}

## Final Molecule
{molecule}

## Instructions
- Write 1-2 short first-person sentences.
- Confirm, by reading the check above, that the required substructure is present
  AND all property targets are met, so no further edits are needed.
- Do NOT restate every numeric value; refer to the check's lines.
- Do NOT state HOW MANY targets there are ("all seven property targets…"): the count
  is easy to get wrong and adds nothing. Say "every property target" / "all of them".
  If you do name properties, spell each one EXACTLY as the check spells it
  (``rings_total``, not "rings"; ``heavy_atoms``, not "heavy atoms") and list them ALL
  or none — a partial list reads as if it were complete.
- Vary the wording — the whole sentence, not just its first word. Do NOT begin with
  "The constraint check confirms", and do NOT fall back on "every property target is
  within range" / "the substructure is present and every property target …": those exact
  phrasings are already worn out. Write it as your own reading of this molecule's state,
  and let the substructure you actually see (the ring system, the linkage) carry the
  sentence rather than a generic template.
- But name that substructure ONLY in the words the check's own description uses (its
  ring names, the linkage it describes). Never coin fusion nomenclature — a bracketed
  locant name like "thieno[2,3-c]pyridine" or "pyrazolo[1,5-a]pyrazine" asserts a
  specific fusion pattern and oxidation state you have not verified, and it is wrong
  more often than not. "the thiophene fused to the piperidine ring" says the same thing
  and stays true. Do not invent a name for a ring the description does not name.
- Do NOT include the <ANSWER> tag or the SMILES; only the confirming reasoning.

Reasoning (1-2 sentences, no count, no coined ring name):"""

# ---------------------------------------------------------------------------
# Generator – per-round tool selection reason (LLM-generated)
# ---------------------------------------------------------------------------
ROUND_REASONING_PROMPT = """\
You are generating high-quality training data for a molecular generation AI assistant.

## Task
One edit round of a molecule generation trajectory is being written. The required
substructure is already present, so this round is a property adjustment: the assistant
calls ``suggest_edits`` to rank candidate substituent edits, commits ONE of them with
``edit_fragment``, then re-verifies and re-measures the result. Write the FOUR separate
pieces of prose that round needs, in one pass.

## Output Format — exactly these four blocks, each tag on its own line
[SUGGEST]
<the reasoning written BEFORE the suggest_edits call>
[EDIT]
<the reasoning written BEFORE the edit_fragment call>
[CHECK]
<the reasoning written BEFORE the verify+measure checkpoint>
[NOTE]
<the one-line memory record of what this round did>

Nothing outside the blocks — no preamble, no commentary, no extra tags.

## These four are spoken at DIFFERENT points in the conversation
Each block is a separate assistant turn with tool calls and tool results in between,
so each must stand on its own. Never write one as a continuation of another ("Next,
…", "As noted above, …", "Then I will …"), and never repeat a sentence across blocks.
The information each block may use is fixed by WHEN it is spoken:

- [SUGGEST] is spoken BEFORE suggest_edits has returned anything. It therefore knows
  ONLY the property status — it does NOT know the candidates. Naming a candidate, a
  fragment, a Δ value, a ± spread or a rank in this block is a look-ahead leak and
  makes the trajectory dishonest. Write it as a request, not a preview.
- [EDIT] is spoken after the candidates came back; it may use all of them.
- [CHECK] is spoken after the edit was applied but BEFORE anything was re-measured. It
  must not contain a single property VALUE — no measurement exists yet.
- [NOTE] is the memory line for the finished round; past tense.

## [SUGGEST] — exactly 2 sentences, the second SHORT (12 words or fewer)
- Sentence 1 — the specific out-of-range property/properties: current value, target and
  the DIRECTION each must move. Quote every value EXACTLY as the status block gives it,
  all decimals ("48.150", not "48.2"). This is where the substance goes; vary how you
  say it.
- Sentence 2 — just that you are calling ``suggest_edits`` for edits that move those
  properties. Do NOT explain why a tool beats a guess and do NOT praise the tool: any
  clause like "leveraging the tool's ability to …", "data-driven candidate edits ranked
  by …", "rather than guessing / heuristic trial-and-error", "based on quantitative
  impact" is padding — it says nothing about THIS molecule and reads identically every
  round. Twelve plain words beat thirty decorative ones.
- Phrase as an INTENTION. Do NOT invent candidate edits or a resulting SMILES, and do
  NOT include the tool call itself.

## [EDIT] — exactly 3 sentences, 65 words or fewer IN TOTAL
Terse and concrete — every sentence carries a number or a name, no scene-setting, no
restating the task. Reasoning first, choice last: never open with the commitment.
Decide in this order, quoting the Landing Safety block (it has already done every count):
ELIMINATE what breaks a satisfied constraint or misses an exact-value target → largest
share of the gap CLOSED → most SAFE LANDINGS, read against IN AFTER in the same row →
no OVERSHOOT past the far edge → smallest edit (ΔMW, then heavy atoms) → the tool's own
RANK, last resort. Do not walk the whole order out loud: name the criterion that DECIDES
and at most one it passed through. If you invoke one the block calls a tie, say it is a tie
rather than claiming a lead. The block also lists in one line the columns that are
IDENTICAL for every candidate — never argue from one of those, and do not spend words
reporting them. Size is step-one material when MW or heavy_atoms is itself a target;
otherwise it only breaks a tie. Name the mechanism behind a Δ you already quoted (an added
carbon or halogen raises logP and MW; a new polar group raises TPSA and HBD/HBA), always
tied to the property that is out of range.
1. Sentence 1 — the gap that DRIVES this choice: the property (at most two), its current
   value, its target and the DIRECTION it must move. Use the EXACT direction the property
   status gives (for a negative-valued property like logS, "DECREASE" = MORE negative).
   Do not list constraints that already pass.
2. Sentence 2 — weigh the candidate you take against the one(s) you pass over, using both
   the predicted Δ and its ± spread. The spread must appear here and must do real work: a
   wide spread on a bounded or exact target means the landing point is unreliable and may
   overshoot; a tight spread is the safer bet; a candidate that fixes one gap while pushing
   another property out of range is a poor trade.
   NEVER compute a spread comparison yourself. Which ± is tighter is already decided in
   the Spread Comparison block and in the ``[…-tightest of N]`` tags — say only what they
   say. Do not write "±0.14 is tighter than ±0.05". When the block says the candidates
   TIE, say the spread does not discriminate and decide on the Δ instead; when the selected
   candidate is NOT the tightest, do not claim it is — say the larger Δ is worth the looser
   spread (or that the tighter option moves the wrong property), which is the real reason
   it was taken.
3. Sentence 3 — take the candidate as a TRIAL to be measured, and name the
   transformation: which group leaves, which arrives, and where it attaches.
- Uncertainty is mandatory. Every Δ is an mmpdb PREDICTION averaged over matched pairs —
  never a measurement. Never write an outcome as settled ("this raises MW to 409", "this
  brings logP into range", "the result will be …"). Instead hedge the prediction
  ("should", "is predicted to", "ought to land near", "only ~+0.4 on paper"); make the
  spread the REASON for the hedge ("±0.62 is wide enough to overshoot the 3.5 ceiling");
  and make the commitment provisional ("so apply … first and re-measure", "try …").
  Model the shape "candidate A moves it further, but its ±X spread on <property> makes
  that unreliable, so apply B first and check" — the choice is a next experiment.
- State the candidate's rank exactly as truthfully as the "Rank of this candidate" line
  does. Never call it the top / highest-ranked candidate unless that line says it is. When
  it is not rank 1, sentence 2 must say WHY it beats the higher-ranked ones for THIS gap.
- QUOTE EVERY NUMBER EXACTLY as given, with all of its decimals — write "48.150", never
  "48.2"; "176.059", never "176.1". A shortened value no longer matches the measurement
  the next checkpoint will show.
- Do NOT state or predict a resulting SMILES, do NOT include the tool call, and do NOT
  claim all targets are now met.

## [CHECK] — ONE sentence, ~12-22 words
Say you are re-checking the substructure and re-measuring the properties of the molecule
as it now stands. No example is given on purpose: word it yourself, differently every
time — nothing that reads as a formula, and in particular not "measure the (updated)
properties of the current molecule". Anchoring it on what THIS edit was supposed to move
("…and see whether that carbonyl brought TPSA up") beats a generic check. It is a progress
check on the CURRENT molecule — do NOT call the molecule or scaffold "finalized",
"complete" or "done" (more edits may follow). Do NOT restate the constraint list, do NOT
quote a property value, and do NOT include the tool calls.

## [NOTE] — ONE sentence, under 20 words
- Starts with a PAST-TENSE verb ("Added …", "Replaced …", "Swapped …"): this line is
  replayed in the next step's memory under "## Progress So Far", where it records what was
  already done. Never an instruction ("Attach …") or a gerund ("Attaching …").
- Explains which property goal this edit was building toward, but names a property ONLY if
  it appears in the out-of-range list below. A property already inside its range was not
  what this edit was for. Naming no property at all is fine — describe the structural
  change instead.
- Names the group(s) ONLY as the Verified Chemistry Facts name them. Does NOT repeat the
  tool name or arguments verbatim. Contains NO SMILES.

## Naming rule for [EDIT] and [NOTE]
Name the removed/attached groups and the edit site ONLY with the names in the Verified
Chemistry Facts block. Do NOT invent or guess a chemical name that is not listed there —
if no name is given, refer to the fragment by the ring/group names listed. Never rename a
ring (a pyrrolidine is not a piperazine, a pyrrole is not a pyridine, a carboxamide is not
a urea). When from_smiles is a bare attachment point the edit substitutes a HYDROGEN — do
not describe it as replacing a methyl. Describe the edit SITE only as the "Edit site" facts
state; if the site is not in a ring, do NOT say the edit happens on a ring. Never copy a
field label from this prompt ("Rank of this candidate", "Edit site", …) into prose.

## The block's column names are NOT English
Quote the block's NUMBERS, never its headers. "gap closed", "safe", "overshoot",
"+heavy", "ΔMW", "breaks", "rank" are column labels in a table — a sentence built out of
them reads as a spreadsheet, not as reasoning. Say what the number MEANS instead: "closes
93% of the distance", "lands eight of the twelve with a full std to spare", "carries QED
0.021 past its 0.890 ceiling", "adds three heavy atoms and 44 daltons", "breaks nothing",
"the tool's own first choice".

## How to Read the Candidate List Below
Each line gives the edit rule and, per constrained property, the current value, the
mmpdb-predicted change as ``Δ avg ± std`` (std = spread over the matched pairs: small = a
reliable, transferable shift; large = the shift varies by context, so a bounded target can
be overshot or missed), the resulting predicted value, and whether that lands inside the
target range. The ``[…-tightest of N]`` tag on each ± value is COMPUTED — use it as given;
never re-derive which spread is tighter yourself.

## User Query
{user_prompt}

## Property Status of the Current Molecule (before this edit)
{property_gap}

## Candidates suggest_edits Returned
{candidate_options}

## This Round (COMPUTED — AUTHORITATIVE) — same for every candidate
{round_context}

## Landing Safety (COMPUTED — AUTHORITATIVE, never recompute)
{landing_safety}

## Spread Comparison for the Out-of-range Properties (COMPUTED — AUTHORITATIVE)
{spread_verdict}

## The Candidate to Select
Action: {action}
edit_fragment args: from_smiles={from_smiles}, to_smiles={to_smiles}, anchors={anchors}
Rank of this candidate: {candidate_rank}
THIS IS CANDIDATE {picked_number}. Any candidate number your sentence names as
the one being applied must be that number — the other numbers may only appear as
the options you passed over or the fallback you would revert to.
Current labelled molecule: {labeled_atoms}

## Verified Chemistry Facts (RDKit-computed — AUTHORITATIVE)
Group being removed  (from_smiles): {from_desc}
Group being attached (to_smiles):   {to_desc}
Edit site:
{anchor_facts}

## Molecule State Before This Edit
{prev_state}

## Molecule State After This Edit (for [NOTE] only — [CHECK] may not use it)
{curr_state}

## Properties That Were OUT OF RANGE Before This Edit
{open_gaps}

## The Checkpoint That Follows The Edit
{checkpoint_tools}

## Length is a hard cap, not a target — count the words before you finish
[SUGGEST] 2 sentences, the 2nd ≤12 words.  [EDIT] exactly 3 sentences, ≤65 words IN
TOTAL — a 78-word answer is a FAILED answer, so cut adjectives and clauses until it
fits.  [CHECK] 1 sentence, 12-22 words.  [NOTE] 1 sentence, <20 words.

## No sentence may appear in two blocks
[SUGGEST] and [EDIT] both open on the property gap, and they are spoken one tool call
apart — so writing the same sentence in both is the most visible defect this round can
have. [EDIT] must say the gap in DIFFERENT words: lead with the size of the move needed,
or with the direction, or with the one property that actually decides the choice.
Likewise [CHECK] and [NOTE] must not reuse [EDIT]'s wording for the transformation.

## Every block is its own voice, not a template
Vary the openings across rounds. "I am calling suggest_edits to …", "Candidate #N is the
top-ranked option …", "The substructure remains intact and …" are worn out — if a block
would start that way, start it somewhere else.

Write the four blocks now, in the order [SUGGEST] [EDIT] [CHECK] [NOTE]. Quote every
number with all its decimals; [SUGGEST] may not mention any candidate; [CHECK] may not
contain a value; every Δ in [EDIT] is a prediction, so hedge it and take the candidate as
a trial to re-measure; [EDIT] is 3 sentences and ≤65 words:"""

TOOL_REASON_PROMPT = """\
You are generating training data for a molecular generation AI assistant.

## Task
Write a brief one-sentence reason explaining why this structural edit was made
at this step. This reason is stored in the assistant's working memory.

## User Query
{user_prompt}

## Edit Performed
{tool_name}({tool_args})

## Verified Chemistry Facts (RDKit-computed — AUTHORITATIVE)
{edit_facts}

## Molecule State Before Edit
{prev_state}

## Molecule State After Edit
{curr_state}

## Properties that were OUT OF RANGE before this edit (the only goals it can have had)
{open_gaps}

## Instructions
Write exactly ONE concise sentence (under 20 words) that:
- Starts with a PAST-TENSE verb ("Added …", "Replaced …", "Swapped …"): this line is
  replayed in the next step's memory under "## Progress So Far", where it records what
  was already done. Never write it as an instruction ("Attach …") or as a gerund
  ("Attaching …") — one convention, consistently.
- Explains which part of the required substructure (or which property goal) this
  edit was building toward
- Names a property ONLY if it appears in the out-of-range list above. A property that
  already sat inside its range was not what this edit was for, and saying "to increase
  MR and logP" when logP was already in range is simply false. Naming no property at
  all is fine — describe the structural change instead.
- Names the group(s) involved ONLY as the Verified Chemistry Facts block names them.
  Do NOT guess a chemical name that is not listed there (a pyrrolidine is not a
  piperazine, a pyrrole is not a pyridine, a carboxamide is not a urea, a bare
  attachment point is not a methyl). With no name given, say "group"/"substituent"
  or use the listed ring/functional-group names.
- Does NOT repeat the tool name or arguments verbatim
- Does NOT include SMILES strings
- Reads like a concise expert's internal note

Reason (ONE sentence, past tense, under 20 words):"""

# ===========================================================================
# Verifier prompts
# ===========================================================================

VERIFIER_SYSTEM_PROMPT = (
    "You are a quality assurance expert for AI training data. "
    "Your job is to evaluate training examples for molecular generation tool-calling "
    "AI assistants. You must respond with a JSON object containing your evaluation."
)

VERIFIER_PROMPT = """\
## Training Example to Evaluate

### User Query
{user_prompt}

### Conversation
{formatted_conversation}

### Predicted Molecule
{molecule_prediction}

## Evaluation Criteria
Rate each aspect from 0 to 10:

1. **Reasoning Quality**: Is the structural reasoning logical, chemistry-aware, and well-explained?
2. **Flow Coherence**: Does the conversation flow naturally between tool calls?
3. **Reflection Quality**: Are reflections (if present) meaningful? (Rate 7 if no reflections)
4. **Verification Quality**: Is the final verification (substructure + properties) sound?
5. **Overall Quality**: How good is this as training data for a molecular generation AI?

## Response Format
Respond with ONLY a JSON object:
{{
  "scores": {{
    "reasoning": <0-10>,
    "coherence": <0-10>,
    "reflection": <0-10>,
    "verification": <0-10>,
    "overall": <0-10>
  }},
  "accepted": <true if overall >= {threshold}>,
  "feedback": "<brief explanation>"
}}"""

# ===========================================================================
# Augmentor prompts
# ===========================================================================

AUGMENT_TOOLS_PROMPT = """\
You are a data augmentation assistant. Given the following chemistry tool \
definitions, generate **alternative function names** (valid Python snake_case) \
and **paraphrased descriptions** for each tool AND its parameters. \
The new names and descriptions must preserve the original semantics exactly.

## Original Tools
{tools_json}

## Instructions
- Each new function name must be a valid Python identifier in snake_case.
- Descriptions should convey the same meaning with different wording / sentence \
structure.
- Parameter descriptions should also be paraphrased (same meaning, different text).
- Do NOT change parameter names or types – only rename the function and rewrite \
descriptions.

Respond with ONLY a JSON object (no markdown fences):
{{
  "<original_function_name>": {{
    "new_name": "<alternative_snake_case_name>",
    "new_description": "<paraphrased tool description>",
    "param_descriptions": {{
      "<param_name>": "<paraphrased param description>",
      ...
    }}
  }},
  ...
}}"""

AUGMENT_SYSTEM_PROMPT = """\
Paraphrase the following system prompt for a molecular generation AI assistant. \
Keep the exact same meaning and all instructions, but use different wording \
and sentence structure. Preserve any formatting (e.g. newlines between paragraphs).

Do NOT add or remove any instructions — only rephrase.

Original:
{original_text}

Paraphrased version:"""

AUGMENT_USER_PROMPT = """\
Paraphrase the following user query about molecule generation. \
Keep the exact same meaning, requirements, and constraints. \
Any specific molecule names, SMILES strings, SMARTS patterns, numeric thresholds, \
or target names must be preserved exactly as-is — only rephrase the surrounding text.

Original:
{original_text}

Paraphrased version:"""


# ---------------------------------------------------------------------------
# NAIVE MODE (NAIVE_REASONING=1) — the ablation baseline.
#
# Everything the rest of this file computes for the model is withheld here. The
# prompt carries ONLY what the trained model will actually see at inference: the
# user query, the current molecule, the raw ``analyze_properties`` output, and the
# raw ``suggest_edits`` JSON. No landing table, no round context, no spread
# verdict, no rendered candidate comparison, no verified chemistry names, no
# decision order, no candidate index — and no guards or regeneration afterwards.
#
# It exists so the scaffolding can be priced. Every claim about what the computed
# blocks buy (ties reported as ties, ± comparisons that are not inverted, counts
# that match the table, a named candidate that is the one committed) is a claim
# about the difference between this prompt and the one above it, and that
# difference is only measurable if both can be generated from the same chains.
# ---------------------------------------------------------------------------

NAIVE_EDIT_REASONING_PROMPT = """You are a molecular generation AI assistant working through a
task. Below is the conversation so far and the tool call you are about to make. Write the
assistant's turn that goes between them — the reasoning that leads naturally from the tool
output you just received to the call you are about to make.

## The task you were given
{user_prompt}

## The molecule you are working on
{mol_smiles}

## What analyze_properties returned
{raw_props}

## What suggest_edits returned
{raw_candidates}

## The tool call you are about to make
edit_fragment(from_smiles={from_smiles}, to_smiles={to_smiles}, anchors={anchors})

Write that turn: 3 sentences, 65 words or fewer IN TOTAL. First person, flowing prose.
No bullet points, no markdown, no backticks, no tool-call syntax, and do not think out
loud or correct yourself mid-sentence — write the finished reasoning only."""
