"""In-process (server-free) tool execution.

The tool servers (:mod:`molkit.tools.tool_server`) do nothing more than::

    result = tool(inputs, return_text=return_text)   # BaseTool.__call__
    return {"result": result}

so calling the registered tool instance directly, in this process, yields the
**byte-identical** structured payload the ``/call`` endpoint returns — the
builder stores ``json.dumps(result)`` either way, so a chain built in local mode
is indistinguishable from one built against the servers.

Why bother
----------
Running the tools locally removes the whole HTTP layer that dominates the
toolchain-build wall time: no per-call round-trip, no keep-alive/reset churn, no
``analyze_properties`` retry storm under GPU contention (the in-process
:class:`ADMETBatcher` coalesces concurrent requests and never returns the
transient all-``None`` ADMET the remote path guards against). The heavy
``admet_ai`` model is lazy-loaded once per process (~10-15 s) and then predicts
in ~0.1 s/molecule.

The fast tools (``match_substructure`` / ``label_atom_indices`` /
``attach_fragment`` / ``form_bond`` / ``set_stereochemistry`` …) are pure RDKit
and run inline; only ADMET is compute-heavy. Because ``admet_ai`` / RDKit release
the GIL during their heavy sections and the ADMET batcher merges concurrent
requests into one ``model.predict``, dispatching the synchronous call through a
thread pool (:func:`local_call_async`) lets the existing ``num_workers``
coroutines overlap exactly as they did over HTTP.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

_registry: Optional[dict] = None


def _get_registry() -> dict:
    """Lazily import and cache the tool registry (``name -> BaseTool``).

    Deferred so importing this module stays cheap and the (heavy) molkit tool
    stack is only pulled in when local mode is actually used.
    """
    global _registry
    if _registry is None:
        from molkit.tools import TOOL_REGISTRY
        _registry = TOOL_REGISTRY
    return _registry


def preload(with_admet: bool = True) -> None:
    """Warm the registry (and optionally the ADMET model) before the run.

    Loading ``admet_ai`` costs ~10-15 s; doing it once up front keeps the very
    first ``analyze_properties`` from paying that latency mid-batch.
    Best-effort: a missing/unloadable ADMET backend degrades gracefully (the
    server path behaves the same — ADMET simply comes back empty).
    """
    _get_registry()
    if with_admet:
        try:
            from molkit.utils.molmim import get_admet_model
            get_admet_model()
        except Exception as exc:  # pragma: no cover - optional backend
            logger.warning("ADMET model preload skipped: %s", exc)


def local_call(name: str, args: dict, *, return_text: bool = False) -> Any:
    """Execute tool *name* in-process, mirroring the server ``/call`` result.

    Matches :func:`http_client.api_call_async` semantics: an unknown tool and a
    raised exception both come back as an ``"Error: ..."`` string rather than
    propagating, so the builder records them in ``errors`` exactly as it does for
    a failed HTTP call.
    """
    tool = _get_registry().get(name)
    if tool is None:
        return f"Error: no port mapping for tool '{name}'"
    try:
        result = tool(args, return_text=return_text)
    except Exception as exc:  # pragma: no cover - parity with HTTP error path
        return f"Error: API call failed for '{name}': {exc}"
    if return_text and not isinstance(result, str):
        return str(result)
    return result


async def local_call_async(name: str, args: dict, *, return_text: bool = False) -> Any:
    """Async wrapper around :func:`local_call` (runs it in a worker thread).

    Keeps the builder's coroutine structure intact — the synchronous RDKit/ADMET
    work is offloaded so ``asyncio.gather`` over the per-instance step coroutines
    still overlaps (GIL-releasing ADMET + the request-coalescing batcher do the
    real concurrency).
    """
    return await asyncio.to_thread(local_call, name, args, return_text=return_text)


def local_call_batch(name: str, smiles_list: list, *, property_names=None) -> dict:
    """Featurise + predict a LIST of molecules in ONE pass -> ``{smiles: result}``.

    Routes to the tool's ``_batch`` (a single RDKit-descriptor loop + one batched
    ``analyze_properties`` prediction), which is ~3x cheaper per molecule
    than N separate ``local_call`` invocations: the ~50 ms fixed per-prediction
    overhead (dataframe build + model setup) is amortised across the batch, and the
    physchem cache is shared. Used by :func:`search_plan.beam_search` to measure a
    whole round's candidates at once. Falls back to per-item calls if the tool has
    no ``_batch``. Returns ``{}``-valued entries as raw error strings, mirroring the
    single-call error convention.
    """
    tool = _get_registry().get(name)
    if tool is None:
        return {s: f"Error: no port mapping for tool '{name}'" for s in smiles_list}
    batch = getattr(tool, "_batch", None)
    if batch is None:
        return {s: local_call(
            name,
            {"mol_smiles": s, **({"property_names": property_names} if property_names else {})},
            return_text=False) for s in smiles_list}
    try:
        return batch(smiles_list, property_names=property_names)
    except Exception as exc:  # pragma: no cover - parity with HTTP error path
        return {s: f"Error: batch call failed for '{name}': {exc}" for s in smiles_list}
