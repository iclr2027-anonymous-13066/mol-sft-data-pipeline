"""Orchestrator – ties Generator, Augmentor, and Verifier together.

For each input file:
    1. Load inputs, classify via _balance_inputs into SATISFIED, CONFLICT,
       and UNSATISFIED groups.  All three groups are kept.
    2. For each kept input, Generator produces TrainingExample(s).
       UNSATISFIED inputs produce intermediate-only examples (accepted
       tool chain steps without a final ``<ANSWER>``).
    3. Accepted examples are streamed to a JSONL output file as they finish.
       A per-file tqdm progress bar tracks progress.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from pathlib import Path
from typing import Optional

from .augmentor import Augmentor
from .config import PipelineConfig
from .generator import Generator
from .schema import GeneratorInput, TrainingExample, merge_segments
from .tools.executor import ToolExecutor
from .tools.registry import ToolRegistry
from .verifier import Verifier

logger = logging.getLogger(__name__)


class TrainingDataPipeline:
    """End-to-end pipeline: load inputs → generate → augment → verify → save."""

    def __init__(
        self,
        config: PipelineConfig,
        tool_registry: Optional[ToolRegistry] = None,
    ) -> None:
        self.config = config
        self.registry = tool_registry or ToolRegistry.default()
        self.executor = ToolExecutor(self.registry)
        self.generator = Generator(config, self.registry, self.executor)
        self.augmentor = Augmentor(config)
        self.verifier = Verifier(config)
        self.generation_history: list[dict] = []

    # -- single example -----------------------------------------------------

    async def process_single(
        self, input_data: GeneratorInput
    ) -> list[TrainingExample]:
        """Generate training examples for one input. Returns 0, 1, or N examples.

        When ``config.num_generations > 1``, the generation is repeated
        independently multiple times so that the same ground-truth tool chain
        produces diverse LLM-generated reasoning text.  Each pass tags its
        examples with a ``generation_idx`` (0-based).
        """
        input_data.generation_history = self.generation_history[-10:]

        all_examples: list[TrainingExample] = []
        for gen_idx in range(self.config.num_generations):
            examples = await self._generate_once(input_data)
            # Guard: drop the entire generation pass if ANY example has an
            # empty molecule_prediction.  Conflict examples and intermediate
            # segments are exempt since they never emit <ANSWER> tags.  An
            # empty SMILES in a non-conflict, non-intermediate example means
            # a broken tool response slipped through — the resulting
            # <ANSWER></ANSWER> would be invalid training data.
            has_empty = any(
                not ex.molecule_prediction
                and not ex.is_intermediate_segment
                for ex in examples
            )
            if has_empty:
                logger.warning(
                    "Dropping generation pass %d — empty molecule_prediction: %s",
                    gen_idx, input_data.user_prompt[:80],
                )
                continue
            for ex in examples:
                ex.generation_idx = gen_idx
            all_examples.extend(examples)
        return all_examples

    async def _generate_once(
        self, input_data: GeneratorInput
    ) -> list[TrainingExample]:
        """Run a single generation pass and return the resulting examples.

        If segmentation is enabled (`config.segment_edit_boundaries`) and the
        input has ≥1 molecule-editing step, returns one example per build
        segment.  Otherwise returns at most a single example from the
        :meth:`Generator.generate` fallback path.
        """
        # Fast path: try segmentation first when enabled.
        if self.config.segment_edit_boundaries:
            try:
                segs = await self.generator.generate_segments(input_data)
            except Exception as e:
                logger.error("generate_segments failed: %s", e)
                segs = []
            if segs:
                return segs

        # Single-example fallback (no usable edit segments in the chain).
        max_attempts = 1 if not self.config.use_verifier else self.config.max_retries

        for attempt in range(1, max_attempts + 1):
            try:
                example = await self.generator.generate(input_data)
                if example is None:
                    return []

                if self.config.use_augmentor:
                    example = await self.augmentor.augment(example)

                if not self.config.use_verifier:
                    logger.info("Attempt %d | verifier disabled, accepting", attempt)
                    return [example]

                result = await self.verifier.verify(example)

                logger.info(
                    "Attempt %d | scores=%s | accepted=%s",
                    attempt, result.scores, result.accepted,
                )

                history_entry = {
                    "user_prompt": input_data.user_prompt,
                    "accepted": result.accepted,
                    "scores": result.scores,
                    "feedback": result.feedback,
                    "attempt": attempt,
                }
                self.generation_history.append(history_entry)

                if result.accepted:
                    return [example]

                input_data.generation_history.append(history_entry)

            except Exception as e:
                logger.error("Attempt %d failed with error: %s", attempt, e)
                if attempt == max_attempts:
                    raise

        logger.warning(
            "Failed to generate accepted example after %d attempts for: %s",
            max_attempts, input_data.user_prompt[:80],
        )
        return []

    # -- batch processing ---------------------------------------------------

    async def process_batch(
        self,
        inputs: list[GeneratorInput],
        write_callback=None,
        progress_desc: Optional[str] = None,
    ) -> list[TrainingExample]:
        """Process a list of inputs concurrently (bounded by batch_size).

        Outputs are flushed to *write_callback* in **input order** regardless
        of completion order, so every training example produced from the same
        source input is contiguous in the output stream.  The callback is
        invoked as ``write_callback(segments, group_id)`` with one chain's
        segments, which are merged into a single record.
        """
        import sys
        from tqdm import tqdm

        results: list[TrainingExample] = []
        semaphore = asyncio.Semaphore(self.config.batch_size)

        # Each input is written as soon as IT finishes. Writing in input order
        # instead (buffer completions, flush the contiguous prefix) starved the
        # GPUs: one slow input holds back every input behind it, so up to
        # ``batch_size`` completed inputs pile up and are then serialised in a
        # single blocking pass — measured as a 5-8 s freeze of the whole event
        # loop, repeating every 15-30 s, during which no request is dispatched
        # and all 16 vLLM servers drain to idle (in-flight < 100 for 13% of
        # samples, GPU below 30% for 15%). Nothing downstream needs global input
        # order; only a group's own segments must stay contiguous, which they do.
        drain_state = {"flushed_examples": 0}
        drain_lock = asyncio.Lock()

        def _group_id(inp: GeneratorInput, idx: int) -> str:
            meta = inp.metadata or {}
            task_id = meta.get("task_id")
            if isinstance(task_id, str) and task_id:
                # Append subtask + input index so inputs that share a task_id
                # across subtasks (e.g. editing add vs substitute) don't merge.
                subtask = meta.get("subtask") or ""
                suffix = f"__{subtask}" if subtask else ""
                return f"{task_id}{suffix}__{idx:07d}"
            return f"inst_{idx:07d}"

        def _flush_one(
            idx: int, inp: GeneratorInput, examples: list[TrainingExample],
        ) -> None:
            """Write one finished input, its segments merged into one record."""
            if write_callback is not None and examples:
                base_gid = _group_id(inp, idx)
                # Group by generation_idx: each pass is its own conversation and
                # gets its own group_id.
                gen_groups: dict[int, list] = {}
                for ex in examples:
                    gen_groups.setdefault(ex.generation_idx, []).append(ex)
                multi_gen = len(gen_groups) > 1
                for gen_idx in sorted(gen_groups.keys()):
                    gen_exs = gen_groups[gen_idx]
                    gid = f"{base_gid}__gen{gen_idx}" if multi_gen else base_gid
                    write_callback(gen_exs, gid)
            drain_state["flushed_examples"] += len(examples)

        async def _process(idx: int, inp: GeneratorInput) -> list[TrainingExample]:
            async with semaphore:
                try:
                    out = await self.process_single(inp)
                except Exception as e:
                    logger.error("Example failed (idx=%d): %s", idx, e)
                    out = []
            async with drain_lock:
                _flush_one(idx, inp, out or [])
                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix(
                        flushed=drain_state["flushed_examples"],
                        refresh=False,
                    )
            return out

        pbar = tqdm(
            total=len(inputs),
            desc=progress_desc or "progress",
            unit="inp",
            dynamic_ncols=True,
            file=sys.stderr,
            mininterval=0.2,
            leave=True,
        )
        tasks = [
            asyncio.create_task(_process(i, inp))
            for i, inp in enumerate(inputs)
        ]
        try:
            for out in await asyncio.gather(*tasks, return_exceptions=True):
                if isinstance(out, Exception):
                    continue
                if out:
                    results.extend(out)
        finally:
            pbar.close()

        return results

    # -- full run -----------------------------------------------------------

    async def run(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        resume: bool = False,
    ) -> list[TrainingExample]:
        """Load inputs, generate examples, verify, and save to JSONL.

        Each accepted example is written to *output_path* immediately so that
        progress is preserved even if the process is interrupted.

        If *resume* is True and *output_path* already contains results, inputs
        whose ``user_prompt`` is already present in the output file are skipped
        and new results are appended.
        """
        output_path = output_path or self.config.output_path

        # Resolve to absolute path so the save location is unambiguous
        output_path = str(Path(output_path).resolve())
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        inputs = self._load_inputs(input_path)
        logger.info("Loaded %d inputs from %s", len(inputs), input_path)
        # Classify into CONFLICT / SATISFIED (strict) / drop unsatisfied.
        # All kept examples follow either the conflict or satisfied path, so
        # the old fallback ref_smiles constraint check is no longer needed.
        inputs = self._balance_inputs(inputs)

        # ── Resume: skip already-processed inputs ──────────────────────────
        write_mode = "w"
        if resume and Path(output_path).exists():
            done_prompts = self._load_completed_prompts(output_path)
            if done_prompts:
                before = len(inputs)
                inputs = [inp for inp in inputs if inp.user_prompt not in done_prompts]
                logger.info(
                    "Resume: skipping %d already-completed inputs (%d remaining)",
                    before - len(inputs),
                    len(inputs),
                )
                write_mode = "a"
            if not inputs:
                logger.info("Resume: all inputs already processed, nothing to do.")
                return []

        logger.info("Writing accepted examples incrementally to %s", output_path)

        accepted_count = 0
        # Records are ~12 kB each and a chunk holds ~58k of them, so flushing
        # every record put ~58k write syscalls on the event loop. Buffer instead
        # and flush every FLUSH_EVERY records: a hard kill can then lose the tail
        # of the buffer, which costs nothing because an interrupted chunk is
        # re-generated from its last flushed record anyway (see run_dir).
        FLUSH_EVERY = 256
        with open(output_path, write_mode, buffering=1 << 22) as out_f:
            def _write(
                segments: list[TrainingExample],
                group_id: str,
            ) -> None:
                nonlocal accepted_count
                record = merge_segments(segments)
                if record is None:
                    return
                # Attach the instance id so downstream viewers and re-shuffling
                # utilities can trace a record back to its instance.
                record["metadata"]["group_id"] = group_id
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                accepted_count += 1
                if accepted_count % FLUSH_EVERY == 0:
                    out_f.flush()
                logger.debug(
                    "Saved record #%d (group=%s, %d segments) to %s",
                    accepted_count, group_id, len(segments), output_path,
                )

            examples = await self.process_batch(
                inputs,
                write_callback=_write,
                progress_desc=Path(input_path).name,
            )

        logger.info(
            "Generated %d / %d accepted examples → %s",
            len(examples),
            len(inputs),
            output_path,
        )
        return examples

    # -- directory run -------------------------------------------------------

    async def run_dir(
        self,
        input_dir: str,
        output_dir: str,
        resume: bool = False,
        shard: tuple[int, int] = (0, 1),
    ) -> int:
        """Process all toolchains_chunk_*.jsonl files in *input_dir* in order.

        Each chunk is processed independently and written to a matching file in
        *output_dir*.  Returns the total number of accepted examples across all
        chunks.

        If *resume* is True, chunks whose output file already exists are resumed
        from where they left off (already-processed inputs are skipped).
        """
        input_dir_path = Path(input_dir)
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        chunk_files = sorted(
            input_dir_path.glob("toolchains_*chunk_*.jsonl"),
            key=lambda p: [
                int(t) if t.isdigit() else t
                for t in re.split(r"(\d+)", p.stem)
            ],
        )
        if not chunk_files:
            logger.warning("No chunk files found in %s", input_dir)
            return 0

        # Chunk-level sharding: several worker processes over one directory, each
        # taking a disjoint set of chunks. A single process cannot keep the fleet
        # busy — its event loop needs ~45 ms of Python/RDKit per chain, ~71% of
        # one core over a 20k-chain chunk, and every burst of that stalls request
        # dispatch. Sharding gives each worker its own core and its own loop.
        shard_i, shard_n = shard
        if shard_n > 1:
            chunk_files = [p for k, p in enumerate(chunk_files) if k % shard_n == shard_i]
            logger.warning("Shard %d/%d: %d of the directory's chunks",
                           shard_i, shard_n, len(chunk_files))

        total = 0
        for chunk_path in chunk_files:
            chunk_name = chunk_path.name
            out_path = str(output_dir_path / chunk_name)
            marker = Path(out_path + ".done")
            # A chunk is skipped only when its marker says it FINISHED. A
            # non-empty output without a marker is a chunk that was interrupted
            # mid-way (a kill, a crash, a node reboot): skipping it on the old
            # "output exists" test silently dropped every input it had not
            # reached yet — up to a whole chunk of training data, with nothing in
            # the log to say so. Such a chunk is resumed input-by-input instead;
            # ``run`` skips the user_prompts already present in the file.
            if resume and marker.exists():
                logger.info("Resume: chunk %s already finished (%s), skipping",
                            chunk_name, marker.read_text().strip() or "?")
                continue
            if resume and Path(out_path).exists() and Path(out_path).stat().st_size > 0:
                logger.warning(
                    "Resume: chunk %s has output but no .done marker — reading it to "
                    "see which inputs it already covers, then finishing the rest "
                    "(a chunk that turns out to be complete just gets its marker)",
                    chunk_name)
            logger.info("Processing chunk %s → %s", chunk_path, out_path)
            examples = await self.run(str(chunk_path), out_path, resume=resume)
            total += len(examples)
            written = sum(1 for line in open(out_path) if line.strip()) \
                if Path(out_path).exists() else 0
            marker.write_text(f"{written} examples\n")
            logger.info("Chunk %s: %d examples accepted (%d in file)",
                        chunk_name, len(examples), written)

        return total

    # -- I/O ----------------------------------------------------------------

    @staticmethod
    def _load_completed_prompts(path: str) -> set[str]:
        """user_prompts already saved in *path*, after trimming a partial group.

        An interrupted run stops mid-input, so the file's last group can hold
        only some of its segments — a trajectory missing its terminal
        ``<ANSWER>`` segment. Its prompt would still count as "done" and the
        truncated instance would ship. So the file is first truncated back to the
        end of its last COMPLETE group (segment_index + 1 == segment_total), and
        only then are the done prompts read off it.

        Single pass: a finished chunk is ~700 MB and this runs for every chunk a
        resumed shard owns, so the file is parsed once — prompts are held back
        until their instance closes, instead of parsing the file a second time.
        """
        prompts: set[str] = set()
        pending: set[str] = set()   # prompts seen since the last group boundary
        last_complete_end = 0
        offset = 0
        try:
            with open(path, "rb") as f:
                for raw in f:
                    offset += len(raw)
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue
                    messages = record.get("messages") or []
                    # messages[1] is the user message
                    if len(messages) > 1 and messages[1].get("role") == "user":
                        pending.add(messages[1]["content"])
                    meta = record.get("metadata") or {}
                    if meta.get("segment_index", 0) + 1 >= meta.get("segment_total", 1):
                        prompts |= pending      # that instance is complete
                        pending.clear()
                        last_complete_end = offset
        except FileNotFoundError:
            return prompts

        if last_complete_end < offset:
            logger.warning(
                "Resume: %s ended mid-instance — trimming %d trailing byte(s) "
                "back to the last complete group",
                Path(path).name, offset - last_complete_end,
            )
            with open(path, "r+b") as f:
                f.truncate(last_complete_end)
        return prompts

    @staticmethod
    def _load_inputs(path: str) -> list[GeneratorInput]:
        inputs: list[GeneratorInput] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    inputs.append(GeneratorInput(**json.loads(line)))
        return inputs

    @staticmethod
    def _has_empty_suggest(inp: GeneratorInput) -> bool:
        """True when a ``suggest_edits`` step in the chain returned NO candidates.

        The ``edit_fragment`` that follows such a step has nothing it could have
        been chosen from, so any reasoning written for it would be invented — the
        whole chain is dropped rather than trained on (measured: ~0.2% of chains).
        """
        from .generator import _suggest_candidates

        for step in inp.tool_chain or []:
            calls = [step.tool_call] + list(step.parallel_tool_calls or [])
            responses = [step.expected_response] + list(
                step.parallel_expected_responses or [])
            for call, resp in zip(calls, responses):
                if call is not None and call.name == "suggest_edits":
                    if not _suggest_candidates(resp):
                        return True
        return False

    def _balance_inputs(
        self,
        inputs: list[GeneratorInput],
    ) -> list[GeneratorInput]:
        """Split inputs into satisfied / unsatisfied and (by default) drop the
        unsatisfied ones.

        Satisfaction is sourced from ``all_constraints_strictly_satisfied`` when
        present, falling back to ``all_constraints_satisfied``.  A satisfied
        chain reaches an ``<ANSWER>``; an unsatisfied one ends on an
        out-of-range checkpoint and would only yield truncated intermediate
        segments, so it is excluded unless ``config.drop_unsatisfied_chains`` is
        turned off.
        """
        satisfied: list[GeneratorInput] = []
        unsatisfied: list[GeneratorInput] = []
        no_candidates = 0

        for inp in inputs:
            # A chain with a candidate-less suggest_edits round is dropped
            # outright — before the satisfied/unsatisfied split, and before the
            # generator's non-segmented fallback path can render it anyway.
            if self._has_empty_suggest(inp):
                no_candidates += 1
                continue
            meta = inp.metadata or {}
            strict = meta.get("all_constraints_strictly_satisfied")
            loose = meta.get("all_constraints_satisfied")
            ok = bool(loose) if strict is None else bool(strict)
            (satisfied if ok else unsatisfied).append(inp)

        drop = getattr(self.config, "drop_unsatisfied_chains", True)
        combined = list(satisfied) if drop else satisfied + unsatisfied
        random.shuffle(combined)
        logger.warning(
            "Input breakdown — satisfied: %d, unsatisfied: %d (%s) | "
            "empty suggest_edits: %d (dropped) | kept total: %d",
            len(satisfied),
            len(unsatisfied),
            "dropped" if drop else "kept",
            no_candidates,
            len(combined),
        )
        return combined
