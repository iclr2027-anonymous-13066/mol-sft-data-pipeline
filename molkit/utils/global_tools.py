"""molkit/utils/global_tools.py  —  Tool instance registry with server/local fallback

Usage
-----
    from molkit.utils.global_tools import get_global_tools, get_tool_schemas

    tool_instances = get_global_tools()   # {name: callable}
    schemas        = get_tool_schemas()   # list[dict]  (OpenAI function-calling format)

How server fallback works
-------------------------
For each tool registered in TOOL_SERVER_PORTS:
  1. Health-check the tool server (GET /health).
  2. If healthy  → use HTTP caller  (isolates heavy models in a separate process).
  3. If not      → instantiate tool class locally (code mode).

Tools not in TOOL_SERVER_PORTS are always instantiated locally.
"""

from __future__ import annotations

import logging
import os

import requests

logger = logging.getLogger(__name__)

_GLOBAL_TOOL_INSTANCES: dict | None = None

# Ports confirmed healthy by the main process and stored as a comma-separated
# env var (TOOL_SERVERS_READY="9000,9001,…").  Worker processes inherit this
# and skip the per-worker health check entirely.
_ENV_READY_KEY = "TOOL_SERVERS_READY"


def _server_base_url() -> str:
    return os.environ.get("TOOL_SERVER_BASE_URL", "http://localhost").rstrip("/")


def _ready_ports() -> set[int]:
    """Return the set of ports already confirmed healthy by the main process."""
    raw = os.environ.get(_ENV_READY_KEY, "")
    if not raw:
        return set()
    result = set()
    for token in raw.split(","):
        token = token.strip()
        if token.isdigit():
            result.add(int(token))
    return result


def _check_server_health(port: int, timeout: float = 3.0) -> bool:
    """Return True if the tool server at *port* responds to GET /health."""
    try:
        resp = requests.get(f"{_server_base_url()}:{port}/health", timeout=timeout)
        return resp.status_code == 200
    except Exception:
        return False


def pre_check_servers(timeout: float = 10.0) -> None:
    """Check all tool server ports once and store results in an env var.

    Call this from the *main* process before spawning workers.  Workers
    inherit the env var and skip redundant health checks so that a busy
    server is not mis-identified as unavailable.
    """
    from molkit.tools import TOOL_SERVER_PORTS

    ready: list[int] = []
    for name, port in TOOL_SERVER_PORTS.items():
        if _check_server_health(port, timeout=timeout):
            ready.append(port)
            logger.info("Tool server '%s' (port %d) is healthy.", name, port)
        else:
            logger.warning("Tool server '%s' (port %d) not reachable; will use local fallback.", name, port)

    os.environ[_ENV_READY_KEY] = ",".join(str(p) for p in ready)
    logger.info("Pre-check complete. Ready ports: %s", ready)


def _make_http_caller(port: int):
    """Return a callable that POSTs inputs to the running tool server."""

    def caller(tool_input: dict, return_text: bool = True) -> str:
        payload = {"inputs": tool_input, "return_text": return_text}
        url = f"{_server_base_url()}:{port}/call"
        resp = requests.post(url, json=payload, timeout=300)
        if resp.status_code >= 400:
            # raise_for_status() throws away the body, leaving a bare "500 Server
            # Error: Internal Server Error for url: …". Surface the server's
            # `detail` (tool name + exception type + message) so the failure is
            # diagnosable from the caller's log / tool message.
            try:
                detail = resp.json().get("detail")
            except Exception:
                detail = (resp.text or "").strip()[:500]
            raise RuntimeError(
                f"Tool Server Error ({resp.status_code}) from {url}: "
                f"{detail or 'no detail'}"
            )
        return resp.json()["result"]

    return caller


def get_global_tools(force_reload: bool = False) -> dict:
    """Build tool callables with automatic server/local fallback.

    Returns
    -------
    dict[str, callable]
        Maps tool name → callable that accepts ``(inputs: dict, return_text: bool) → str``.
    """
    global _GLOBAL_TOOL_INSTANCES
    if _GLOBAL_TOOL_INSTANCES is not None and not force_reload:
        return _GLOBAL_TOOL_INSTANCES

    from molkit.tools import TOOL_REGISTRY, TOOL_SERVER_PORTS

    ready = _ready_ports()

    instances: dict = {}
    for name, tool_inst in TOOL_REGISTRY.items():
        port = TOOL_SERVER_PORTS.get(name)
        if port is not None and (port in ready or _check_server_health(port)):
            logger.info("Tool '%s' using server at port %d.", name, port)
            instances[name] = _make_http_caller(port)
        else:
            if port is not None:
                logger.debug("Tool '%s' server not available; using local instance.", name)
            instances[name] = tool_inst

    _GLOBAL_TOOL_INSTANCES = instances
    return instances


def get_tool_schemas() -> list[dict]:
    """Return OpenAI/LiteLLM tool schemas for all registered tools."""
    from molkit.tools import TOOL_REGISTRY
    return [tool.to_schema() for tool in TOOL_REGISTRY.values()]
