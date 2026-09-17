"""Async + thread-local HTTP helpers for the tool servers.

Tool servers (see :mod:`molkit.tools.tool_server`) accept:
    POST /call   body = {"inputs": {...}, "return_text": bool}  → {"result": ...}
    GET  /schema                                                → OpenAI schema dict

Two interfaces are exposed:

* ``api_call_async`` / ``get_async_session`` / ``close_async_session`` –
  asyncio-native POST via :mod:`aiohttp`.  Used by the toolchain builder so
  each instance can run in its own coroutine without blocking a thread per
  in-flight request.  There is no cross-worker batching – requests go out
  independently, so workers truly progress at their own pace.

* ``api_call`` / ``get_schema`` – synchronous helpers kept for one-off /
  initialisation use (e.g. schema fetch before the async loop starts).
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from typing import Any, Optional

import aiohttp
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import (
    TOOL_SERVER_HOST,
    TOOL_SERVER_PORTS,
    TOOL_SERVER_TIMEOUT,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Local (server-free) execution switch
# ---------------------------------------------------------------------------
# When enabled, api_call_async / api_call_batch_async dispatch to in-process tool
# instances (see :mod:`local_tools`) instead of POSTing to the tool servers. The
# structured result is byte-identical (the server does the same tool() call), so
# stored chains are indistinguishable — but the HTTP layer, its retries, and the
# ADMET-under-contention retry storm all disappear. Honoured from the env at
# import (LOCAL_TOOLS=1) and overridable via :func:`set_local_mode`.
_LOCAL_MODE: bool = os.environ.get("LOCAL_TOOLS", "0") not in ("0", "", "false", "False")


def set_local_mode(enabled: bool) -> None:
    """Enable/disable in-process tool execution (checked per call)."""
    global _LOCAL_MODE
    _LOCAL_MODE = bool(enabled)


def local_mode_enabled() -> bool:
    return _LOCAL_MODE

_thread_local = threading.local()
_CONNECT_RETRIES = 3
_RETRY_BACKOFF = 2.0  # seconds before first retry (scales linearly)

# ---------------------------------------------------------------------------
# Async session (one aiohttp.ClientSession per event loop)
# ---------------------------------------------------------------------------
_async_session: Optional[aiohttp.ClientSession] = None
_async_session_loop: Optional[asyncio.AbstractEventLoop] = None


def get_async_session() -> aiohttp.ClientSession:
    """Return the aiohttp session bound to the current running event loop.

    Connections are kept alive and reused across requests
    (``force_close=False``) to avoid per-request TCP handshake overhead.
    ``ServerDisconnectedError`` (stale keep-alive sockets closed by
    uvicorn) is handled by the retry loop in :func:`api_call_async`.
    ``limit_per_host=256`` keeps enough sockets available even when many
    workers fan out in parallel.
    """
    global _async_session, _async_session_loop
    loop = asyncio.get_running_loop()
    if _async_session is None or _async_session.closed or _async_session_loop is not loop:
        connector = aiohttp.TCPConnector(
            limit=0,
            limit_per_host=256,
            ttl_dns_cache=300,
            force_close=False,
            enable_cleanup_closed=True,
        )
        timeout = aiohttp.ClientTimeout(total=TOOL_SERVER_TIMEOUT)
        _async_session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        _async_session_loop = loop
    return _async_session


async def close_async_session() -> None:
    """Close the shared aiohttp session (safe to call multiple times)."""
    global _async_session, _async_session_loop
    if _async_session is not None and not _async_session.closed:
        await _async_session.close()
    _async_session = None
    _async_session_loop = None


async def api_call_async(name: str, args: dict, *, return_text: bool = True) -> Any:
    """Async POST to the tool server for *name* and return the result.

    Mirrors :func:`api_call` but uses aiohttp, so many concurrent coroutines
    can have requests in flight without one thread per call.  Connection
    errors are retried up to ``_CONNECT_RETRIES`` times with linear backoff.

    In local mode (:func:`set_local_mode` / ``LOCAL_TOOLS=1``) the call is served
    in-process — no HTTP, no retries — returning the identical structured result.
    """
    if _LOCAL_MODE:
        from .local_tools import local_call_async
        return await local_call_async(name, args, return_text=return_text)

    port = TOOL_SERVER_PORTS.get(name)
    if port is None:
        return f"Error: no port mapping for tool '{name}'"

    url = f"{TOOL_SERVER_HOST}:{port}/call"
    payload = {"inputs": args, "return_text": return_text}
    session = get_async_session()
    last_exc: Exception = RuntimeError("unreachable")

    for attempt in range(_CONNECT_RETRIES + 1):
        try:
            async with session.post(url, json=payload) as resp:
                resp.raise_for_status()
                data = await resp.json()
            result = data.get("result", "")
            if return_text:
                return str(result) if not isinstance(result, str) else result
            return result
        except asyncio.TimeoutError:
            return f"Error: tool server timeout for '{name}'"
        except aiohttp.ServerDisconnectedError as e:
            # Stale keep-alive socket – retry immediately with a fresh
            # connection, no backoff, no warning (expected under bursty load).
            last_exc = e
            if attempt < _CONNECT_RETRIES:
                continue
        except (aiohttp.ClientConnectionError, aiohttp.ClientOSError) as e:
            last_exc = e
            if attempt < _CONNECT_RETRIES:
                wait = _RETRY_BACKOFF * (attempt + 1)
                logger.warning(
                    "Tool server '%s' unreachable (attempt %d/%d), retrying in %.0fs: %s",
                    name, attempt + 1, _CONNECT_RETRIES + 1, wait, e,
                )
                await asyncio.sleep(wait)
        except aiohttp.ClientResponseError as e:
            return f"Error: API call failed for '{name}': {e}"
        except Exception as e:
            return f"Error: API call failed for '{name}': {e}"

    return f"Error: cannot connect to tool server for '{name}': {last_exc}"


async def api_call_batch_async(
    name: str,
    inputs_list: list[dict],
    *,
    return_text: bool = False,
) -> list:
    """Batch POST to ``/call_batch`` — sends N inputs in one HTTP request.

    Returns a list of results (one per input, same order).  Falls back to
    individual ``api_call_async`` calls if the batch endpoint is not
    available (HTTP 404/405).
    """
    if not inputs_list:
        return []
    if _LOCAL_MODE:
        from .local_tools import local_call_async
        return list(await asyncio.gather(*[
            local_call_async(name, inp, return_text=return_text)
            for inp in inputs_list
        ]))

    port = TOOL_SERVER_PORTS.get(name)
    if port is None:
        return [f"Error: no port mapping for tool '{name}'"] * len(inputs_list)

    url = f"{TOOL_SERVER_HOST}:{port}/call_batch"
    payload = {"inputs_list": inputs_list, "return_text": return_text}
    session = get_async_session()

    for attempt in range(_CONNECT_RETRIES + 1):
        try:
            async with session.post(url, json=payload) as resp:
                if resp.status in (404, 405):
                    # Server doesn't support batch — fall back to individual calls.
                    return list(await asyncio.gather(*[
                        api_call_async(name, inp, return_text=return_text)
                        for inp in inputs_list
                    ]))
                resp.raise_for_status()
                data = await resp.json()
            return data.get("results", [])
        except asyncio.TimeoutError:
            return [f"Error: tool server timeout for '{name}'"] * len(inputs_list)
        except aiohttp.ServerDisconnectedError as e:
            last_exc = e
            if attempt < _CONNECT_RETRIES:
                continue
        except (aiohttp.ClientConnectionError, aiohttp.ClientOSError) as e:
            last_exc = e
            if attempt < _CONNECT_RETRIES:
                wait = _RETRY_BACKOFF * (attempt + 1)
                logger.warning(
                    "Batch call '%s' unreachable (attempt %d/%d), retrying in %.0fs: %s",
                    name, attempt + 1, _CONNECT_RETRIES + 1, wait, e,
                )
                await asyncio.sleep(wait)
        except Exception as e:
            return [f"Error: batch call failed for '{name}': {e}"] * len(inputs_list)

    return [f"Error: cannot connect for '{name}'"] * len(inputs_list)


# ---------------------------------------------------------------------------
# Synchronous helpers (kept for init-time / external callers)
# ---------------------------------------------------------------------------


def _get_session() -> requests.Session:
    """Return a thread-local requests.Session, creating it on first access."""
    if not hasattr(_thread_local, "session"):
        session = requests.Session()
        retry = Retry(total=0, connect=0, read=0, backoff_factor=0)
        adapter = HTTPAdapter(
            max_retries=retry,
            pool_connections=len(TOOL_SERVER_PORTS),
            pool_maxsize=len(TOOL_SERVER_PORTS),
        )
        session.mount("http://", adapter)
        _thread_local.session = session
    return _thread_local.session


def api_call(name: str, args: dict, *, return_text: bool = True) -> Any:
    """Synchronous POST to the tool server (kept for non-async callers)."""
    port = TOOL_SERVER_PORTS.get(name)
    if port is None:
        return f"Error: no port mapping for tool '{name}'"

    url = f"{TOOL_SERVER_HOST}:{port}/call"
    payload = {"inputs": args, "return_text": return_text}
    session = _get_session()
    last_exc: Exception = RuntimeError("unreachable")

    for attempt in range(_CONNECT_RETRIES + 1):
        try:
            resp = session.post(url, json=payload, timeout=TOOL_SERVER_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            result = data.get("result", "")
            if return_text:
                return str(result) if not isinstance(result, str) else result
            return result
        except requests.exceptions.Timeout:
            return f"Error: tool server timeout for '{name}'"
        except requests.exceptions.ConnectionError as e:
            last_exc = e
            if attempt < _CONNECT_RETRIES:
                wait = _RETRY_BACKOFF * (attempt + 1)
                logger.warning(
                    "Tool server '%s' unreachable (attempt %d/%d), retrying in %.0fs: %s",
                    name, attempt + 1, _CONNECT_RETRIES + 1, wait, e,
                )
                if hasattr(_thread_local, "session"):
                    del _thread_local.session
                session = _get_session()
                time.sleep(wait)
        except Exception as e:
            return f"Error: API call failed for '{name}': {e}"

    return f"Error: cannot connect to tool server for '{name}': {last_exc}"


def get_schema(name: str) -> Optional[dict]:
    """Fetch a tool's JSON schema via GET /schema on its tool server."""
    port = TOOL_SERVER_PORTS.get(name)
    if port is None:
        return None

    url = f"{TOOL_SERVER_HOST}:{port}/schema"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.warning("Failed to fetch schema for '%s': %s", name, e)
        return None
