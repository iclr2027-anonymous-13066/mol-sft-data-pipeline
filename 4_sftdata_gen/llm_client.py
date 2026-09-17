"""Async LLM client wrapping one or more OpenAI-compatible vLLM servers.

When multiple ``base_urls`` are configured, requests are distributed across
them in round-robin order.  Because the pipeline runs entirely inside a single
asyncio event loop (single thread), the counter is updated without locks.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Optional

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    BadRequestError,
)

from .config import VLLMConfig

logger = logging.getLogger(__name__)

# Max concurrent in-flight requests PER vLLM server. Total concurrency across
# the fleet is PER_SERVER_CONCURRENCY × len(base_urls) (also bounded by the
# pipeline's batch_size = concurrent inputs). Override with
# LLM_CONCURRENCY_PER_SERVER.
#
# NOTE (measured 2026-07-06, 2× Qwen3.6-27B): raising this from 40 to 96 did NOT
# improve throughput — the servers were already saturated at 40/server (GPU
# ~80%, the practical ceiling for this short-generation workload). So the client
# cap is generally NOT the bottleneck; overall speed scales with the NUMBER of
# servers (round-robin/least-loaded fan-out), not this knob. Only raise it if you
# can measure the GPUs sitting idle (util well below ~80%) at the current value.
PER_SERVER_CONCURRENCY = int(os.environ.get("LLM_CONCURRENCY_PER_SERVER", "40"))

# How many DIFFERENT servers one request may be tried on before giving up, and how
# long a server that answered with a connection error / 5xx is taken out of the
# dispatch pool.
#
# Why this exists (measured 2026-07-27): a single engine died (vLLM
# EngineDeadError on :8080) and took ~78% of the whole run's inputs down with it.
# Without failover a request that landed on the dead port raised straight through
# ``generate_segments`` → the pipeline's fallback path runs with max_attempts=1
# when the verifier is off → the input was dropped and written as zero records.
# Worse, the dead port became an ATTRACTOR: connection-refused returns in
# milliseconds, so its reserved-slot counter fell back to 0 immediately and
# _next_index picked it as "least loaded" for nearly every request. 15 healthy
# servers sat idle. Cooling a failing endpoint out of the pool fixes both.
FAILOVER_ATTEMPTS = int(os.environ.get("LLM_FAILOVER_ATTEMPTS", "3"))
UNHEALTHY_COOLDOWN_S = float(os.environ.get("LLM_UNHEALTHY_COOLDOWN_S", "30"))


class LLMClient:
    """Thin async wrapper around one or more OpenAI-compatible (vLLM) endpoints.

    Requests are dispatched to the **least-loaded** server (fewest in-flight +
    queued requests), which keeps every GPU fed even with only a couple of
    servers — plain round-robin would send a request to the next server in turn
    even when its ``PER_SERVER_CONCURRENCY`` slots are full while another server
    is idle (head-of-line blocking → GPU underutilisation). Each server still
    has its own semaphore hard-capping concurrent in-flight requests.
    """

    def __init__(self, config: VLLMConfig) -> None:
        self.config = config
        self._clients: list[AsyncOpenAI] = [
            AsyncOpenAI(base_url=url, api_key=config.api_key)
            for url in config.base_urls
        ]
        self._semaphores: list[asyncio.Semaphore] = [
            asyncio.Semaphore(PER_SERVER_CONCURRENCY) for _ in self._clients
        ]
        # Reserved-slot counter per server (in-flight + queued), used to pick the
        # least-loaded endpoint. Single-threaded asyncio → no lock needed.
        self._load: list[int] = [0] * len(self._clients)
        # monotonic deadline until which a server stays out of the dispatch pool
        # (0.0 = healthy). Set when it answers with a connection error or a 5xx.
        self._down_until: list[float] = [0.0] * len(self._clients)
        if len(self._clients) > 1:
            logger.info(
                "LLMClient initialised with %d servers (cap %d concurrent/server, "
                "least-loaded dispatch): %s",
                len(self._clients),
                PER_SERVER_CONCURRENCY,
                config.base_urls,
            )

    def _next_index(self, exclude: frozenset[int] = frozenset()) -> int:
        # Pick the least-loaded HEALTHY server not already tried for this request
        # and reserve a slot on it immediately (synchronous, so concurrent callers
        # see the update and spread evenly). The reservation is released in
        # _complete's finally.
        now = time.monotonic()
        candidates = [
            i for i in range(len(self._load))
            if i not in exclude and self._down_until[i] <= now
        ]
        if not candidates:
            # Everything is either already tried or cooling down. Prefer the
            # untried servers whose cooldown expires soonest over failing here —
            # a fleet-wide blip must not turn into a dropped input.
            candidates = [i for i in range(len(self._load)) if i not in exclude]
            if not candidates:
                candidates = list(range(len(self._load)))
            candidates.sort(key=lambda i: self._down_until[i])
            candidates = candidates[:1]
        idx = min(candidates, key=lambda i: self._load[i])
        self._load[idx] += 1
        return idx

    def _mark_down(self, idx: int, exc: BaseException) -> None:
        """Take a server out of the dispatch pool for UNHEALTHY_COOLDOWN_S."""
        was_healthy = self._down_until[idx] <= time.monotonic()
        self._down_until[idx] = time.monotonic() + UNHEALTHY_COOLDOWN_S
        if was_healthy:
            logger.warning(
                "server %s marked unhealthy for %.0fs (%s: %s)",
                self.config.base_urls[idx], UNHEALTHY_COOLDOWN_S,
                type(exc).__name__, exc,
            )

    @staticmethod
    def _is_server_fault(exc: BaseException) -> bool:
        """True when *exc* says the SERVER is broken (so failover can help).

        A 400 is the request's own fault (e.g. context length exceeded): retrying
        it elsewhere only wastes work and would blacklist the whole fleet.
        """
        if isinstance(exc, BadRequestError):
            return False
        if isinstance(exc, (APIConnectionError, APITimeoutError)):
            return True
        return isinstance(exc, APIStatusError) and exc.status_code >= 500

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Send a single user prompt and return the assistant response text."""
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return await self._complete(messages, temperature, max_tokens)

    async def generate_with_messages(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Send an arbitrary message list and return the assistant response text."""
        return await self._complete(messages, temperature, max_tokens)

    async def _complete(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        # Turn off the reasoning/thinking channel for short concise-reasoning
        # generations (see VLLMConfig.enable_thinking).  Passed via
        # chat_template_kwargs, which vLLM forwards to the Qwen chat template.
        extra_body = None
        if not getattr(self.config, "enable_thinking", False):
            extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

        # Try up to FAILOVER_ATTEMPTS DIFFERENT servers: a dead engine must cost a
        # hop, not the input (see FAILOVER_ATTEMPTS).
        tried: set[int] = set()
        attempts = min(FAILOVER_ATTEMPTS, len(self._clients))
        response = None
        for attempt in range(1, attempts + 1):
            idx = self._next_index(exclude=frozenset(tried))
            tried.add(idx)
            try:
                async with self._semaphores[idx]:
                    response = await self._clients[idx].chat.completions.create(
                        model=self.config.model,
                        messages=messages,
                        temperature=temperature if temperature is not None else self.config.temperature,
                        max_tokens=max_tokens or self.config.max_tokens,
                        top_p=self.config.top_p,
                        extra_body=extra_body,
                    )
                if self._down_until[idx]:
                    self._down_until[idx] = 0.0
                    logger.warning("server %s is answering again",
                                   self.config.base_urls[idx])
                break
            except Exception as e:
                if not self._is_server_fault(e):
                    raise
                # A timeout means the server is BUSY, not broken — failover, but
                # leave it in the pool. A refused connection or a 5xx (engine
                # dead) does take it out.
                if not isinstance(e, APITimeoutError):
                    self._mark_down(idx, e)
                if attempt == attempts:
                    logger.error(
                        "all %d attempted server(s) failed (%s): %s",
                        attempt, type(e).__name__, e,
                    )
                    raise
                logger.debug(
                    "server %s failed (%s), failing over",
                    self.config.base_urls[idx], type(e).__name__,
                )
            finally:
                self._load[idx] -= 1  # release the reserved slot (see _next_index)
        msg = response.choices[0].message
        content = msg.content
        # Reasoning models (e.g. EXAONE-4.0) may return content=None and put
        # the visible answer in reasoning_content / reasoning instead.
        if not content:
            extra = getattr(msg, "model_extra", {}) or {}
            content = (
                extra.get("reasoning_content")
                or extra.get("reasoning")
                or getattr(msg, "reasoning_content", None)
                or getattr(msg, "reasoning", None)
                or ""
            )
        return content.strip() if content else ""
