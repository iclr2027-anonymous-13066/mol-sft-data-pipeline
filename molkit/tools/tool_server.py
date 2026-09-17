"""molkit/tools/tool_server.py  —  Per-tool FastAPI server

Each tool runs as its own HTTP process on its assigned port.

Usage (single tool)
-------------------
    python -m molkit.tools.tool_server --tool analyze_properties

    # Multi-process mode (bypasses the GIL). Only applied to tools listed in
    # ``MULTIWORKER_TOOLS`` (ignored elsewhere so we don't spawn N reloaders
    # of a rarely-used tool). Default N=1; override via --workers to scale
    # across cores when a tool's Python glue is GIL-bound.
    python -m molkit.tools.tool_server --tool analyze_properties --workers 8

Usage (all tools, one subprocess per tool)
------------------------------------------
    python -m molkit.tools.tool_server --all [--workers 1]

Endpoints
---------
    POST /call      { "inputs": {...}, "return_text": true }  → { "result": "..." }
    GET  /schema    → OpenAI function-calling schema dict
    GET  /health    → { "status": "ok", "tool": "<name>" }
"""

import argparse
import faulthandler
import logging
import multiprocessing
import os
import sys
from typing import Any

# Dump a Python traceback on fatal signals (SIGSEGV/SIGABRT/etc). Essential
# for triaging C-level crashes like "double free or corruption" that otherwise
# leave no stack trace.
faulthandler.enable()

logger = logging.getLogger(__name__)


# Tools that benefit from uvicorn multi-worker (CPU/RDKit-bound, no GPU
# models, OR isolation from occasional native crashes under load). Other
# tools are cheap / lazy-loaded so running 100 copies of them just wastes
# RAM for no throughput gain.
MULTIWORKER_TOOLS = {
    "analyze_properties",
    "match_substructure",
    "edit_fragment",
    "label_atom_indices",
    "suggest_edits",
}


# Env-var handshake used to pass the tool name to uvicorn worker processes
# that reimport this module via the "molkit.tools.tool_server:app" string.
_ENV_TOOL_NAME = "_MOLKIT_TOOL_SERVER_TOOL_INTERNAL"


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(tool_name: str):
    """Create a FastAPI app that serves a single named tool."""
    try:
        from contextlib import asynccontextmanager

        from fastapi import FastAPI, HTTPException, Request
        from fastapi.concurrency import run_in_threadpool
    except ImportError as exc:
        raise ImportError(
            "'fastapi' and 'pydantic' are required for the tool server. "
            "Install them with: pip install fastapi uvicorn[standard]"
        ) from exc

    from molkit.tools import TOOL_REGISTRY

    tool = TOOL_REGISTRY.get(tool_name)
    if tool is None:
        raise ValueError(
            f"Tool '{tool_name}' not found in TOOL_REGISTRY. "
            f"Available: {list(TOOL_REGISTRY)}"
        )

    @asynccontextmanager
    async def lifespan(_app):
        # FastAPI / Starlette run sync endpoints (and run_in_threadpool
        # calls) on AnyIO's default thread limiter, whose default capacity
        # is 40 tokens. When an async client fires hundreds of concurrent
        # requests they queue behind those 40 slots. Bump it so RDKit-heavy
        # tools — whose native calls release the GIL — can actually run in
        # parallel.
        try:
            from anyio import to_thread
            to_thread.current_default_thread_limiter().total_tokens = int(
                os.environ.get("TOOL_SERVER_THREADPOOL", "256")
            )
        except Exception as exc:
            logger.warning("Could not raise AnyIO threadpool: %s", exc)

        yield

    app = FastAPI(title=f"molkit — {tool_name}", lifespan=lifespan)

    # ── Profiling counters & log ────────────────────────────────────────
    # Three states per request:  queued → active (in threadpool) → done.
    # The JSONL log records all three durations so we can distinguish
    # "waiting for a threadpool slot" from "actual compute".
    import time as _time
    import json as _json

    _counters = {"queued": 0, "active": 0}  # per-worker-process
    _profile_dir = "/tmp/tool_profile"
    _do_profile = os.environ.get("TOOL_SERVER_TIMING", "0") != "0"
    if _do_profile:
        os.makedirs(_profile_dir, exist_ok=True)

    def _emit(record: dict) -> None:
        """Append one JSON line to the per-PID profile file.

        Opens/closes on every write so the file survives external
        deletion (e.g. ``rm /tmp/tool_profile/*.jsonl`` between runs).
        """
        if not _do_profile:
            return
        path = os.path.join(
            _profile_dir, f"{tool_name}_{os.getpid()}.jsonl"
        )
        try:
            with open(path, "a") as fp:
                fp.write(_json.dumps(record) + "\n")
        except Exception:
            pass

    @app.post("/call")
    async def call_tool(request: Request):
        t_arrive = _time.perf_counter()
        _counters["queued"] += 1
        try:
            body = await request.json()
        except Exception:
            _counters["queued"] -= 1
            raise HTTPException(status_code=400, detail="Request body must be valid JSON.")
        inputs      = body.get("inputs", {})
        return_text = body.get("return_text", True)

        # Transition: queued → active
        t_start = _time.perf_counter()
        _counters["queued"] -= 1
        _counters["active"] += 1
        try:
            result = await run_in_threadpool(tool, inputs, return_text=return_text)
        except Exception as exc:
            # Bad arguments are caught upstream by BaseTool and returned as an error
            # STRING, so reaching here means a genuine internal failure. Name the
            # tool and the exception type: the client only sees `detail`, and a bare
            # str(exc) is often empty or context-free.
            logger.exception("Tool '%s' failed on inputs %r", tool_name, inputs)
            raise HTTPException(
                status_code=500,
                detail=f"{tool_name} failed: {type(exc).__name__}: {exc}",
            )
        finally:
            t_done = _time.perf_counter()
            _counters["active"] -= 1
            if _do_profile:
                _emit({
                    "tool": tool_name, "path": "/call",
                    "queue_ms": round((t_start - t_arrive) * 1000, 1),
                    "compute_ms": round((t_done - t_start) * 1000, 1),
                    "total_ms": round((t_done - t_arrive) * 1000, 1),
                    "n_queued": _counters["queued"],
                    "n_active": _counters["active"],
                    "batch_size": 1,
                    "ts": round(_time.time(), 3),
                    "pid": os.getpid(),
                })
        return {"result": result}

    @app.post("/call_batch")
    async def call_tool_batch(request: Request):
        """Process multiple inputs in one HTTP round-trip.

        If the tool exposes a ``_batch()`` method (e.g. MolPropAnalyzer),
        all inputs are processed in a single call that batches ADMET
        internally — far more efficient than N individual calls.

        Otherwise, each input is dispatched to the threadpool concurrently
        so RDKit-heavy tools (which release the GIL) get true parallelism
        within a single worker process.
        """
        t_arrive = _time.perf_counter()
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Request body must be valid JSON.")
        inputs_list = body.get("inputs_list", [])
        return_text = body.get("return_text", True)
        if not isinstance(inputs_list, list):
            raise HTTPException(status_code=400, detail="'inputs_list' must be a list.")

        import asyncio as _aio

        n = len(inputs_list)

        # ── Native batch path ──────────────────────────────────────────
        # Tools with a _batch() method (e.g. MolPropAnalyzer) can process
        # all SMILES in one call: 1 external ADMET call instead of N.
        _has_batch = hasattr(tool, '_batch') and callable(getattr(tool, '_batch', None))
        _smiles_key = "mol_smiles"
        if _has_batch and n > 0 and all(_smiles_key in inp for inp in inputs_list):
            _counters["queued"] += 1
            _counters["active"] += 1
            try:
                smiles_list = [inp[_smiles_key] for inp in inputs_list]
                batch_result = await run_in_threadpool(tool._batch, smiles_list)
                if return_text:
                    _fmt = getattr(tool, '_format_text', None)
                    results = []
                    for smi in smiles_list:
                        entry = batch_result.get(smi)
                        if isinstance(entry, dict) and _fmt:
                            results.append(_fmt(smi, entry))
                        elif isinstance(entry, str):
                            results.append(entry)
                        else:
                            results.append(str(entry))
                else:
                    results = [batch_result.get(smi, {}) for smi in smiles_list]
            except Exception as exc:
                results = [f"Error: {exc}"] * n
            finally:
                t_done = _time.perf_counter()
                _counters["active"] -= 1
                _counters["queued"] -= 1
                if _do_profile:
                    _emit({
                        "tool": tool_name, "path": "/call_batch",
                        "batch_size": n, "native_batch": True,
                        "total_ms": round((t_done - t_arrive) * 1000, 1),
                        "n_queued": _counters["queued"],
                        "n_active": _counters["active"],
                        "ts": round(_time.time(), 3),
                        "pid": os.getpid(),
                    })
            return {"results": results}

        # ── Fallback: sequential loop in a single threadpool slot ─────
        # Running N items as N concurrent threadpool tasks causes severe
        # GIL contention (e.g. 200 tasks × ~1ms each balloons to 300ms+
        # per item due to thread scheduling overhead).  A simple sequential
        # loop in one thread eliminates that contention entirely.
        _counters["queued"] += 1
        _counters["active"] += 1

        def _sequential():
            out = []
            for inp in inputs_list:
                try:
                    out.append(tool(inp, return_text=return_text))
                except Exception as exc:
                    out.append(f"Error: {exc}")
            return out

        try:
            results = await run_in_threadpool(_sequential)
        finally:
            t_done = _time.perf_counter()
            _counters["active"] -= 1
            _counters["queued"] -= 1
            if _do_profile:
                _emit({
                    "tool": tool_name, "path": "/call_batch",
                    "batch_size": n,
                    "total_ms": round((t_done - t_arrive) * 1000, 1),
                    "n_queued": _counters["queued"],
                    "n_active": _counters["active"],
                    "ts": round(_time.time(), 3),
                    "pid": os.getpid(),
                })

        return {"results": results}

    @app.get("/schema")
    def get_schema():
        return tool.to_schema()

    @app.get("/health")
    def health():
        return {"status": "ok", "tool": tool_name}

    return app


# ---------------------------------------------------------------------------
# Module-level app loader for uvicorn's multi-worker mode.
#
# When uvicorn spawns workers with ``workers > 1`` it reimports the module
# via the ``"molkit.tools.tool_server:app"`` import string. The launcher
# below sets ``_ENV_TOOL_NAME`` before spawning, so each child knows which
# tool to expose. For single-worker mode ``app`` stays None and the
# launcher constructs it directly.
# ---------------------------------------------------------------------------
def _maybe_create_module_app():
    tool_name = os.environ.get(_ENV_TOOL_NAME)
    if not tool_name:
        return None
    return create_app(tool_name)


app = _maybe_create_module_app()


# ---------------------------------------------------------------------------
# Single-tool entry point (called in a subprocess for --all mode)
# ---------------------------------------------------------------------------

def _serve_one(tool_name: str, port: int, workers: int = 1) -> None:
    try:
        import uvicorn
    except ImportError:
        print(
            "ERROR: 'uvicorn' is required. "
            "Install with: pip install uvicorn[standard]",
            file=sys.stderr,
        )
        sys.exit(1)

    logging.basicConfig(level=logging.INFO)

    # Only the tools listed in MULTIWORKER_TOOLS benefit from many workers
    # (CPU-bound, no per-process state). For the rest, force workers=1 to
    # avoid spawning 100 copies of tools that are cheap to serve.
    effective_workers = workers if tool_name in MULTIWORKER_TOOLS else 1

    logger.info(
        "Starting tool server: %s on port %d (%d worker(s))",
        tool_name, port, effective_workers,
    )

    # Raise the TCP listen backlog so burst connections queue in the kernel
    # instead of receiving RST (errno 104).  With 8 client processes × 5
    # coroutines × ~4 concurrent batch calls, ~160 connections can arrive
    # simultaneously; when all 60 workers are busy the excess must wait in
    # the backlog.  Default uvicorn backlog (~2048) can overflow under
    # sustained bursts — 8192 gives comfortable headroom.
    backlog = int(os.environ.get("TOOL_SERVER_BACKLOG", "8192"))

    if effective_workers > 1:
        os.environ[_ENV_TOOL_NAME] = tool_name
        uvicorn.run(
            "molkit.tools.tool_server:app",
            host="0.0.0.0",
            port=port,
            workers=effective_workers,
            log_level="warning",
            backlog=backlog,
        )
    else:
        local_app = create_app(tool_name)
        uvicorn.run(
            local_app,
            host="0.0.0.0",
            port=port,
            log_level="warning",
            backlog=backlog,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Launch molkit tool server(s).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--tool", metavar="NAME",
        help="Name of the tool to serve (e.g. analyze_properties).",
    )
    group.add_argument(
        "--all", action="store_true",
        help="Launch all registered tools, one subprocess per tool.",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help=(
            "Number of uvicorn workers per tool process. Only tools listed "
            "in MULTIWORKER_TOOLS honour values > 1; all others force "
            "workers=1 to avoid wasteful multi-process hosting."
        ),
    )
    parser.add_argument(
        "--port", type=int, default=None,
        help="Port override (only valid with --tool).",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)

    from molkit.tools import TOOL_REGISTRY, TOOL_SERVER_PORTS

    if args.tool:
        port = args.port or TOOL_SERVER_PORTS.get(args.tool)
        if port is None:
            print(
                f"ERROR: No port assigned for '{args.tool}'. "
                "Use --port to specify one.",
                file=sys.stderr,
            )
            sys.exit(1)
        _serve_one(args.tool, port, workers=args.workers)

    else:  # --all
        if args.port is not None:
            print("WARNING: --port is ignored with --all.", file=sys.stderr)

        procs: list[multiprocessing.Process] = []
        for name, tool_inst in TOOL_REGISTRY.items():
            port = TOOL_SERVER_PORTS.get(name)
            if port is None:
                logger.warning("No port for '%s', skipping.", name)
                continue
            p = multiprocessing.Process(
                target=_serve_one,
                args=(name, port),
                kwargs={"workers": args.workers},
                daemon=True,
                name=f"tool-server-{name}",
            )
            p.start()
            procs.append(p)
            print(f"  Started {name} on port {port} (PID {p.pid})")

        print(f"\n{len(procs)} tool server(s) running. Press Ctrl+C to stop.")
        try:
            for p in procs:
                p.join()
        except KeyboardInterrupt:
            print("\nShutting down …")
            for p in procs:
                p.terminate()


if __name__ == "__main__":
    main()
