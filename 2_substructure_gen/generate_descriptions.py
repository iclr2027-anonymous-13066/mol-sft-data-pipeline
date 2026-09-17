#!/usr/bin/env python3
"""Complete description-free JSONL or Parquet rows with scaffold fields and v8 descriptions.

This is the explicit, user-facing entry point for turning records such as

    {"id": "...", "task_type": "...", "properties": {...}, "ref_smiles": "..."}

into the same 1:1, order-preserving schema as
``data/benchmark/generation_benchmark-00000.jsonl``. Every original key is retained. The
deterministic scaffold analysis/evaluation fields are appended, and ring-scaffold descriptions
are generated with production prompt v8 and its validate/revise loop.

The implementation delegates to ``augment_jsonl_with_scaffold.py`` so this convenience command
cannot drift from the production stage-2 pipeline.

Example:

    python generate_descriptions.py --input IN.jsonl --output OUT.jsonl \
        --smiles-key ref_smiles --servers "localhost:20000,localhost:20200"
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import json
import math
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm

from augment_jsonl_with_scaffold import build_argparser, main_async, polish
from describe_scaffolds import VLLMPool, analyze_one, build_base_urls


_PROPERTY_ORDER = (
    "MW", "logP", "logD", "logS", "TPSA", "QED", "BBBP", "Mutag", "MR",
    "HBD", "HBA", "rotB", "rings_total", "heavy_atoms",
)


def _resolve_smiles_key(path: str, requested: str | None) -> str:
    """Detect the SMILES field and reject a mismatched explicit key before generation."""
    first = None
    first_line = 0
    with open(path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if line.strip():
                first = json.loads(line)
                first_line = line_no
                break
    if first is None:
        raise ValueError(f"input JSONL is empty: {path}")
    if not isinstance(first, dict):
        raise ValueError(f"line {first_line} is not a JSON object")

    if requested:
        if not first.get(requested):
            raise ValueError(
                f"line {first_line} has no non-empty {requested!r}; "
                f"available keys: {', '.join(first.keys())}")
        return requested

    for candidate in ("ref_smiles", "smiles", "smi"):
        if first.get(candidate):
            return candidate
    raise ValueError(
        f"could not detect a SMILES field on line {first_line}; expected one of "
        f"'ref_smiles', 'smiles', or 'smi' (available: {', '.join(first.keys())})")


def _generation_record(row: dict, smiles_key: str, index: int, id_prefix: str) -> dict:
    """Convert one flat property row to the generation benchmark input schema."""
    smi = row.get(smiles_key)
    if not smi:
        raise ValueError(f"row {index} has no non-empty {smiles_key!r} value")

    properties = []
    for name in _PROPERTY_ORDER:
        value = row.get(name)
        if value is None or isinstance(value, bool):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            properties.append({"property": name, "min": number, "max": number})
    if not properties:
        raise ValueError(f"row {index} has no supported numeric property columns")

    return {
        "id": f"{id_prefix}_{index}",
        "task_type": "generation",
        "properties": properties,
        "ref_smiles": str(smi),
    }


def _prepare_generation_schema(path: str, smiles_key: str, id_prefix: str) -> str:
    """Convert flat SMILES/property rows to the generation benchmark input schema.

    A scalar property becomes an exact constraint with equal min/max bounds. Metadata columns
    outside the benchmark schema are intentionally omitted.
    """
    tmp = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".jsonl", prefix="substructure_instances_",
        delete=False)
    tmp_path = tmp.name
    try:
        with open(path, encoding="utf-8") as src, tmp:
            out_idx = 0
            for line_no, line in enumerate(src, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"line {line_no} is not a JSON object")
                record = _generation_record(row, smiles_key, out_idx, id_prefix)
                tmp.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_idx += 1
        if out_idx == 0:
            raise ValueError(f"input JSONL is empty: {path}")
        return tmp_path
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _resolve_parquet_smiles_key(path: str, requested: str | None) -> str:
    """Resolve a non-empty SMILES column from a Parquet input."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    names = pf.schema_arrow.names
    key = requested
    if key is None:
        key = next((name for name in ("ref_smiles", "smiles", "smi") if name in names), None)
    if not key or key not in names:
        wanted = repr(requested) if requested else "'ref_smiles', 'smiles', or 'smi'"
        raise ValueError(
            f"Parquet has no {wanted} column; available columns: {', '.join(names)}")
    for batch in pf.iter_batches(batch_size=256, columns=[key]):
        if any(value not in (None, "") for value in batch.column(0).to_pylist()):
            return key
    raise ValueError(f"Parquet column {key!r} contains no non-empty SMILES values")


def _finish_parquet(chunks: list[str], output: str, shard_size: int) -> list[str]:
    """Unify chunk schemas and atomically assemble one or more Parquet files."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not chunks:
        raise ValueError("input Parquet contains no rows")
    schema = pa.unify_schemas([pq.read_schema(path) for path in chunks])
    stem, _ = os.path.splitext(output)
    final_paths: list[str] = []
    temp_paths: list[str] = []
    writer = None
    rows_in_file = 0

    def _open_writer() -> None:
        nonlocal writer, rows_in_file
        if shard_size > 0:
            final = f"{stem}-{len(final_paths):05d}.parquet"
        else:
            final = output
        final_paths.append(final)
        temp_paths.append(final + ".tmp")
        writer = pq.ParquetWriter(temp_paths[-1], schema, compression="zstd")
        rows_in_file = 0

    try:
        for path in chunks:
            rows = pq.read_table(path).to_pylist()
            table = pa.Table.from_pylist(rows, schema=schema)
            offset = 0
            while offset < table.num_rows:
                if writer is None:
                    _open_writer()
                capacity = (shard_size - rows_in_file) if shard_size > 0 else table.num_rows
                take = min(capacity, table.num_rows - offset)
                writer.write_table(table.slice(offset, take))
                offset += take
                rows_in_file += take
                if shard_size > 0 and rows_in_file == shard_size:
                    writer.close()
                    writer = None
        if writer is not None:
            writer.close()
            writer = None
        for temp, final in zip(temp_paths, final_paths):
            os.replace(temp, final)
        return final_paths
    except Exception:
        if writer is not None:
            writer.close()
        for path in temp_paths:
            try:
                os.unlink(path)
            except OSError:
                pass
        raise


async def _main_parquet(args: argparse.Namespace) -> int:
    """Stream a Parquet input through the existing scaffold/polish pipeline."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(args.input)
    total = pf.metadata.num_rows
    base_urls = build_base_urls(args.servers, args.host, args.base_ports)
    pool = VLLMPool(
        base_urls=base_urls,
        per_server_concurrency=args.concurrency_per_server,
        timeout=args.timeout,
        max_retries=args.max_retries,
        reasoning_effort=args.reasoning_effort,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        model=args.model,
    )
    await pool.discover_models()
    dropped = pool.prune_dead()
    if dropped:
        print(f"# [warn] dropped {dropped} unreachable server(s).")
    if pool.n_servers == 0:
        print("error: no live vLLM server. Try --servers/--host.", file=sys.stderr)
        return 3
    print(f"# Live servers: {pool.n_servers}; "
          f"{args.concurrency_per_server} concurrent request(s) each")

    n_analyzers = args.analyzer_procs or min(32, os.cpu_count() or 8)
    executor = ProcessPoolExecutor(max_workers=n_analyzers)
    queue: asyncio.Queue = asyncio.Queue(
        maxsize=pool.n_servers * args.concurrency_per_server * 3)
    done_token = object()
    stats = {"rows": 0, "failed": 0, "violations": 0}
    pbar = tqdm(total=total, desc="Describe", unit="row", smoothing=0.02)

    async def consumer(server_idx: int) -> None:
        while True:
            payload = await queue.get()
            try:
                if payload is done_token:
                    return
                local_idx, record, an, item, results = payload
                item = await polish(pool, server_idx, an, item, args.validate_passes)
                out = dict(record)
                out.update(item)
                results[local_idx] = out
                stats["rows"] += 1
                stats["failed"] += int(bool(item.get("analysis_error")
                                            or item.get("describe_error")))
                stats["violations"] += int(bool(item.get("description_violations")))
                pbar.update(1)
            finally:
                queue.task_done()

    consumers = [
        asyncio.create_task(consumer(server_idx))
        for server_idx in range(pool.n_servers)
        for _ in range(args.concurrency_per_server)
    ]

    # Resume point. Rows before it are assumed already produced elsewhere, and ids continue from
    # it so a resumed run cannot collide with the ids the earlier run emitted.
    global_index = args.start_row
    to_skip = args.start_row
    chunk_paths: list[str] = []
    try:
        import contextlib
        if args.chunk_dir:
            os.makedirs(args.chunk_dir, exist_ok=True)
            chunk_ctx = contextlib.nullcontext(args.chunk_dir)
        else:
            chunk_ctx = tempfile.TemporaryDirectory(prefix="substructure_parquet_")
        with chunk_ctx as tmp_dir:
            for chunk_index, batch in enumerate(
                    pf.iter_batches(batch_size=args.parquet_batch_size)):
                if to_skip:
                    if to_skip >= batch.num_rows:
                        to_skip -= batch.num_rows
                        pbar.update(batch.num_rows)
                        continue
                    batch = batch.slice(to_skip)
                    to_skip = 0
                source_rows = batch.to_pylist()
                records = []
                for offset, row in enumerate(source_rows):
                    if args.prepare_generation_schema:
                        record = _generation_record(
                            row, args.smiles_key, global_index + offset, args.id_prefix)
                    else:
                        record = dict(row)
                    records.append(record)

                smiles = [
                    str(record.get("ref_smiles" if args.prepare_generation_schema
                                   else args.smiles_key) or "")
                    for record in records
                ]
                # Analysis is bounded to one Parquet batch, so memory use stays fixed.
                analysis = list(executor.map(analyze_one, smiles))
                results: list[dict | None] = [None] * len(records)
                for local_idx, (record, (an, item)) in enumerate(zip(records, analysis)):
                    await queue.put((local_idx, record, an, item, results))
                await queue.join()
                if any(row is None for row in results):
                    raise RuntimeError(f"Parquet batch {chunk_index} did not complete")

                chunk_path = os.path.join(tmp_dir, f"part-{global_index:09d}.parquet")
                pq.write_table(
                    pa.Table.from_pylist(results), chunk_path, compression="zstd")
                chunk_paths.append(chunk_path)
                global_index += len(records)

            for _ in consumers:
                await queue.put(done_token)
            await queue.join()
            await asyncio.gather(*consumers)
            output_paths = _finish_parquet(
                chunk_paths, args.output, args.parquet_shard_size)
    finally:
        for task in consumers:
            if not task.done():
                task.cancel()
        executor.shutdown(wait=False)
        pbar.close()

    destination = output_paths[0] if len(output_paths) == 1 else (
        f"{output_paths[0]} ... {output_paths[-1]} ({len(output_paths)} files)")
    print(f"# Wrote {stats['rows']} row(s) to {destination}; failed={stats['failed']}, "
          f"remaining_violations={stats['violations']}")
    return 0 if stats["failed"] == 0 else 1


def main() -> None:
    parser = build_argparser()
    parser.description = __doc__
    parser.add_argument(
        "--prepare-generation-schema", action="store_true",
        help="convert flat SMILES/property rows to id/task_type/properties/ref_smiles before "
             "running the existing scaffold + description pipeline")
    parser.add_argument(
        "--id-prefix", default="generation",
        help="ID prefix used with --prepare-generation-schema")
    parser.add_argument(
        "--parquet-batch-size", type=int, default=1000,
        help="rows held per Parquet processing/writing batch")
    parser.add_argument(
        "--start-row", type=int, default=0,
        help="skip this many input rows and start ids at the same number, so a resumed run "
             "continues where an earlier one stopped instead of re-emitting generation_0")
    parser.add_argument(
        "--chunk-dir", default=None,
        help="keep per-batch Parquet chunks here instead of a temp dir that is deleted on exit. "
             "Use it for long runs: if the job dies the completed chunks survive")
    parser.add_argument(
        "--parquet-shard-size", type=int, default=100000,
        help="rows per output Parquet file; 0 writes one file (default: 100000)")
    for action in parser._actions:
        if action.dest == "output":
            action.required = True
        elif action.dest in {"out_dir", "shard_size", "shard_prefix"}:
            action.help = argparse.SUPPRESS
        elif action.dest == "smiles_key":
            action.default = None
            action.help = (
                "SMILES field; omitted = auto-detect ref_smiles, smiles, or smi "
                "from the first record")
    args = parser.parse_args()

    if args.out_dir:
        parser.error("generate_descriptions.py writes one --output file; do not use --out-dir")
    if os.path.abspath(args.input) == os.path.abspath(args.output):
        parser.error("--output must differ from --input; the source file is never overwritten")
    input_ext = os.path.splitext(args.input)[1].lower()
    output_ext = os.path.splitext(args.output)[1].lower()
    is_parquet = input_ext in {".parquet", ".pq"}
    if is_parquet != (output_ext in {".parquet", ".pq"}):
        parser.error("input and output must both be Parquet or both be JSONL")
    if not is_parquet and input_ext not in {".jsonl", ".json"}:
        parser.error("--input must end in .jsonl, .json, .parquet, or .pq")
    if args.parquet_batch_size <= 0:
        parser.error("--parquet-batch-size must be greater than zero")
    if args.parquet_shard_size < 0:
        parser.error("--parquet-shard-size must be zero or greater")
    if is_parquet and args.parquet_shard_size > 0:
        if os.path.exists(args.output):
            parser.error(f"output path already exists: {args.output}")
        stem, _ = os.path.splitext(args.output)
        existing = sorted(glob.glob(stem + "-[0-9][0-9][0-9][0-9][0-9].parquet"))
        if existing:
            parser.error(f"output shard already exists: {existing[0]}")
    elif os.path.exists(args.output):
        parser.error(f"--output already exists: {args.output}")
    try:
        resolver = _resolve_parquet_smiles_key if is_parquet else _resolve_smiles_key
        args.smiles_key = resolver(args.input, args.smiles_key)
    except (OSError, ValueError, json.JSONDecodeError, ImportError) as exc:
        parser.error(str(exc))
    print(f"# SMILES key: {args.smiles_key}")

    if is_parquet:
        try:
            raise SystemExit(asyncio.run(_main_parquet(args)))
        except KeyboardInterrupt:
            print("\n# Interrupted; incomplete temporary Parquet data was removed.",
                  file=sys.stderr)
            raise SystemExit(130)

    prepared_input = None
    if args.prepare_generation_schema:
        try:
            prepared_input = _prepare_generation_schema(
                args.input, args.smiles_key, args.id_prefix)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            parser.error(str(exc))
        args.input = prepared_input
        args.smiles_key = "ref_smiles"
        print("# Input schema: generation benchmark "
              "(id, task_type, properties, ref_smiles)")

    # A single completed JSONL should not create README.md / README.en.md beside itself.
    # The production sharded driver retains that dataset-documentation behavior.
    args.copy_readmes = False

    try:
        rc = asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\n# Interrupted before the single output file was written.", file=sys.stderr)
        raise SystemExit(130)
    finally:
        if prepared_input:
            try:
                os.unlink(prepared_input)
            except OSError:
                pass
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
