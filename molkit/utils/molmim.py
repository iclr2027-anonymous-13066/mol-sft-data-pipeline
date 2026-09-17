"""molkit/utils/molmim.py — local ADMET/TDC property model host + scoring

Despite the module name (kept for import stability across the pipeline), this no
longer contains any MolMIM generation / CMA-ES optimisation code — only the
ADMET/TDC property model and the scoring helpers the tools use.

Provides
--------
COLUMN_MAP / TDC_PROP_MAP / SUPPORTED_CONSTRAINT_KEYS
    Property short-name → admet_ai column mapping (9 properties) and TDC oracle mapping (3 properties).

predict_admet(smiles_list)
    Run admet_ai on a SMILES list; returns a DataFrame indexed by SMILES.
    Routes through :class:`ADMETBatcher` so concurrent requests share a single
    ``model.predict`` call (or one round of remote ``/predict`` calls when
    ``ADMET_SERVER_URLS`` is set).

ADMETBatcher
    Process-wide batcher that merges concurrent requests into a single
    ``model.predict`` call.

predict_tdc(smiles_list, props)
    Call TDC Oracle Server for DRD2/GSK3B/JNK3; returns a DataFrame indexed by SMILES.

evaluate_smiles_batch(smiles_list, range_constraints)
    Score a batch of SMILES against multi-property range constraints.

Environment variables
---------------------
TDC_ORACLE_URL      Base URL of the TDC Oracle Server (for DRD2/GSK3B/JNK3).
                    Default: http://localhost:9020
ADMET_SERVER_URLS   Optional comma-separated list of remote ADMET servers
                    (each exposing POST /predict). When set, ADMETBatcher
                    distributes batch work across them and the local model
                    is not loaded.
"""

from __future__ import annotations

import logging
import math
import os
import queue
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import requests
from rdkit import Chem

logger = logging.getLogger(__name__)

# Prevent oversubscription when admet_ai / TDC fingerprinting spin up threads.
for _k, _v in {
    "OMP_NUM_THREADS":       "1",
    "MKL_NUM_THREADS":       "1",
    "NUMEXPR_NUM_THREADS":   "1",
    "OPENBLAS_NUM_THREADS":  "1",
    "CUDA_LAUNCH_BLOCKING":  "0",
    "TF_CPP_MIN_LOG_LEVEL":  "3",
}.items():
    os.environ.setdefault(_k, _v)

# ---------------------------------------------------------------------------
# 0) ADMET property mapping
# ---------------------------------------------------------------------------

COLUMN_MAP: Dict[str, str] = {
    "BBBP":  "BBB_Martins",
    "Mutag": "AMES",
    "HIA":   "HIA_Hou",
    "logS":  "Solubility_AqSolDB",
    "logD":  "Lipophilicity_AstraZeneca",
    "ampa":  "PAMPA_NCATS",
    "erg":   "hERG",
    "liver": "DILI",
    "carc":  "Carcinogens_Lagunin",
}

TDC_PROP_MAP: Dict[str, str] = {
    "DRD2":  "drd2",
    "GSK3B": "gsk3b",
    "JNK3":  "jnk3",
}

SUPPORTED_CONSTRAINT_KEYS: set = set(COLUMN_MAP.keys()) | set(TDC_PROP_MAP.keys())

# Canonical (user-facing) property name → parquet column name.
# Applied when reading the seed-pool / admet-tagged DB, which uses lowercase legacy names.
_DB_COLUMN_ALIAS: Dict[str, str] = {
    "DRD2":        "drd2",
    "GSK3B":       "gsk3b",
    "JNK3":        "jnk3",
    "pLogP":       "plogP",
    "rings_total": "rings",
}


# ---------------------------------------------------------------------------
# 1) ADMET-AI predictor singleton (no dynamic batcher — simple & subprocess-safe)
# ---------------------------------------------------------------------------

_ADMET_MODEL = None
_ADMET_LOCK  = threading.Lock()


def _patch_torch_load() -> None:
    """Patch torch.load to use weights_only=False (needed by admet_ai)."""
    try:
        import torch
        _orig = torch.load
        def _load_safe(*a, **kw):
            kw["weights_only"] = False
            return _orig(*a, **kw)
        torch.load = _load_safe
    except Exception:
        pass


def get_admet_model():
    """Return the (lazily loaded) ADMETModel singleton."""
    global _ADMET_MODEL
    if _ADMET_MODEL is not None:
        return _ADMET_MODEL
    with _ADMET_LOCK:
        if _ADMET_MODEL is None:
            _patch_torch_load()
            try:
                from admet_ai import ADMETModel  # type: ignore[import-untyped]
                # cache_molecules=False is REQUIRED for long unique-molecule runs.
                # With the default (True), ADMETModel enables chemprop's global
                # SMILES_TO_GRAPH / SMILES_TO_MOL caches (via set_cache_graph/
                # set_cache_mol) and its own SMILES_TO_MOL dict — all keyed by
                # SMILES with NO eviction. Every molecule we score is unique, so
                # the cache hit rate is 0% (zero speed benefit) while the retained
                # MolGraph/RDKit-mol objects grow the process RSS without bound
                # (~5 MB per 1000 molecules → a multi-GB/worker leak over a 2M run).
                # Disabling it removes the leak at no throughput cost.
                _ADMET_MODEL = ADMETModel(num_workers=0, fingerprint_multiprocessing_min=999999,
                                          cache_molecules=False)
            except Exception as exc:
                raise RuntimeError(f"admet_ai not available: {exc}") from exc
    return _ADMET_MODEL


# ---------------------------------------------------------------------------
# High-concurrency HTTP session factory
# ---------------------------------------------------------------------------

def _make_pooled_session(pool_size: int = 512) -> requests.Session:
    """Create a requests.Session with an enlarged connection pool.

    The default urllib3 pool_maxsize is 10 — far too small when hundreds of
    tool-server threads call the same host concurrently (ADMET backends, etc.).
    Hitting the cap causes "Connection pool is full"
    warnings and forces connection churn.
    """
    sess = requests.Session()
    from requests.adapters import HTTPAdapter
    adapter = HTTPAdapter(pool_connections=pool_size, pool_maxsize=pool_size)
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)
    return sess


# ---------------------------------------------------------------------------
# 1b) Dynamic ADMET batcher — merges concurrent requests into one model.predict()
# ---------------------------------------------------------------------------

# Shared executor used by ADMETBatcher._predict_distributed to fan batches
# out across N backend URLs. Re-used across all batcher workers so we don't
# create+join 30 threads per batch dispatch (which dominated CPU at scale).
# 32 is enough: max useful parallelism = number of ADMET backends (~80),
# but each HTTP call blocks only a few ms, so 32 threads saturates well.
# The old default of 256 × 20 uvicorn workers = 5,120 threads for one tool
# alone, causing significant RSS growth even before any requests arrive.
_ADMET_DISPATCH_EXECUTOR = ThreadPoolExecutor(
    max_workers=int(os.environ.get("ADMET_DISPATCH_WORKERS", "32")),
    thread_name_prefix="admet-dispatch",
)

# ---------------------------------------------------------------------------
# Backend health / failover knobs
# ---------------------------------------------------------------------------
# A backend that stops answering — hung worker, killed process, GPU OOM — used to
# poison every batch it touched: _call_backend retried the SAME dead URL three
# times, _predict_distributed then raised on any chunk error, and the caller turned
# the WHOLE batch into NA, including the chunks that did come back. One dead
# backend out of 48 was enough to silently void a scoring run.
#
# Failures are now routed around rather than raised. A backend that fails is
# benched for a cooldown, its SMILES are re-queued onto the survivors, and only
# SMILES that no live backend could serve raise. The cooldown doubles per
# consecutive failure (capped) and is cleared by the first success, so a fleet
# that is restarted is picked back up on its own — no process restart here.
_ADMET_COOLDOWN_S      = float(os.environ.get("ADMET_BACKEND_COOLDOWN_S", "60"))
_ADMET_COOLDOWN_MAX_S  = float(os.environ.get("ADMET_BACKEND_COOLDOWN_MAX_S", "600"))
# How many re-queue rounds a batch gets before its leftovers raise. Each round
# redistributes the failed SMILES across whatever is still live.
_ADMET_FAILOVER_ROUNDS = int(os.environ.get("ADMET_FAILOVER_ROUNDS", "3"))
# Split connect from read: an unreachable host should fail in seconds, while a
# large chunk on a healthy backend legitimately takes minutes.
_ADMET_CONNECT_TIMEOUT_S = float(os.environ.get("ADMET_CONNECT_TIMEOUT_S", "5"))
_ADMET_READ_TIMEOUT_S    = float(os.environ.get("ADMET_READ_TIMEOUT_S", "300"))

# Oversized-SMILES guard. A model that degenerates into "CCCC..." until it hits its
# token cap emits a string RDKit happily parses as a valid 16k-atom molecule; feeding
# that to admet_ai takes a worker from 0.8GB to 9GB of RSS and wedges it for an hour.
# With failover on top, one such string walks the fleet and kills a worker per retry.
# Nothing drug-like comes near this: real answers run tens of characters, and the
# benchmark's own molecules are well under 200 atoms. Skipped SMILES get no ADMET
# value, which scores as out-of-range — the right verdict for a 16k-atom alkane, and
# an honest one, since the model cannot predict on it either.
_ADMET_MAX_SMILES_LEN = int(os.environ.get("ADMET_MAX_SMILES_LEN", "1000"))


class ADMETBatcher:
    """Collects SMILES from concurrent requests and issues one ``model.predict`` per batch.

    Each caller ``.predict(smiles_list)`` is blocking; a background worker thread
    accumulates items during a short window (closing when item_gap_ms passes without
    a new arrival, or when abs_max_wait_ms is reached) and runs a single inference.

    When ``backend_urls`` is provided, batches are distributed across N remote
    ADMET servers (``/predict``) in parallel — no local model is loaded.
    """

    def __init__(
        self,
        max_batch_size:    int   = 512,
        item_gap_ms:       float = 10.0,
        abs_max_wait_ms:   float = 100.0,
        backend_urls:      Optional[List[str]] = None,
        num_workers:       int   = 8,
    ):
        self._max_batch_size  = max_batch_size
        self._item_gap_s      = item_gap_ms / 1000.0
        self._abs_max_wait_s  = abs_max_wait_ms / 1000.0
        self._queue: queue.Queue = queue.Queue()
        self._backend_urls: List[str] = [u.rstrip("/") for u in (backend_urls or [])]
        self._backend_sessions: List[requests.Session] = [
            _make_pooled_session() for _ in self._backend_urls
        ]
        # Backend health, shared by every worker thread and by the direct-dispatch
        # path in predict_admet (both run against this one singleton batcher).
        self._health_lock = threading.Lock()
        self._dead_until:  List[float] = [0.0] * len(self._backend_urls)
        self._down_streak: List[int]   = [0] * len(self._backend_urls)
        # N worker threads sharing the queue. Each blocks on HTTP to ADMET
        # backends, so a single consumer would bottleneck throughput.
        n = max(1, int(num_workers))
        self._threads: List[threading.Thread] = []
        for i in range(n):
            t = threading.Thread(
                target=self._worker, daemon=True, name=f"ADMETBatcher-{i}",
            )
            t.start()
            self._threads.append(t)

    def _call_backend(self, session: requests.Session, url: str, smiles_chunk: List[str],
                      attempts: int = 3) -> pd.DataFrame:
        """POST one chunk to one backend. *attempts* retries against THIS url.

        With a multi-backend pool the caller passes ``attempts=1``: re-sending to a
        backend that just timed out is strictly worse than handing the chunk to a
        live one, and each same-url retry costs another read timeout. The 3-attempt
        default is for the single-backend case, where there is nowhere to fail over.
        """
        last_exc: Exception = RuntimeError("unreachable")
        attempts = max(1, int(attempts))
        for attempt in range(attempts):
            try:
                resp = session.post(
                    f"{url}/predict", json={"smiles": smiles_chunk},
                    timeout=(_ADMET_CONNECT_TIMEOUT_S, _ADMET_READ_TIMEOUT_S),
                )
                resp.raise_for_status()
                data = resp.json()
                if "error" in data:
                    raise RuntimeError(f"Backend {url} returned error: {data['error']}")
                return pd.DataFrame.from_dict(data, orient="index")
            except (requests.ConnectionError, requests.Timeout, ConnectionResetError) as exc:
                last_exc = exc
                if attempt < attempts - 1:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise
            except Exception:
                raise
        raise last_exc

    # ---- backend health -------------------------------------------------

    def _mark_down(self, idx: int) -> float:
        """Bench backend *idx*; returns the cooldown in seconds."""
        with self._health_lock:
            self._down_streak[idx] += 1
            cool = min(_ADMET_COOLDOWN_S * (2 ** (self._down_streak[idx] - 1)),
                       _ADMET_COOLDOWN_MAX_S)
            self._dead_until[idx] = time.monotonic() + cool
        return cool

    def _mark_up(self, idx: int) -> None:
        """Clear any bench on backend *idx* after a successful call."""
        if self._down_streak[idx] or self._dead_until[idx]:
            with self._health_lock:
                self._down_streak[idx] = 0
                self._dead_until[idx] = 0.0

    def _live_indices(self) -> List[int]:
        now = time.monotonic()
        with self._health_lock:
            live = [i for i in range(len(self._backend_urls)) if self._dead_until[i] <= now]
        # Everything benched at once (fleet restarting, network blip): fall back to
        # the full list. The cooldown is a preference, not a ban — better to try a
        # benched backend than to fail the batch outright.
        return live or list(range(len(self._backend_urls)))

    def _try_backend(self, idx: int, chunk: List[str]) -> Optional[pd.DataFrame]:
        """Call one backend, benching it on failure. None = re-queue this chunk."""
        try:
            df = self._call_backend(
                self._backend_sessions[idx], self._backend_urls[idx], chunk, attempts=1,
            )
            self._mark_up(idx)
            return df if df is not None else pd.DataFrame()
        except Exception as exc:
            cool = self._mark_down(idx)
            logger.warning(
                "ADMET backend[%d] %s failed (%s); benched %.0fs, re-queueing %d SMILES",
                idx, self._backend_urls[idx], exc, cool, len(chunk),
            )
            return None

    def _predict_distributed(self, all_unique: List[str]) -> pd.DataFrame:
        n = len(self._backend_urls)
        # Shortcut: single backend → just call it. No chunking, no fanout, and
        # nowhere to fail over, so this one keeps the same-url retries.
        if n == 1:
            try:
                df = self._call_backend(
                    self._backend_sessions[0], self._backend_urls[0], all_unique,
                )
                self._mark_up(0)
                return df if df is not None else pd.DataFrame()
            except Exception as exc:
                self._mark_down(0)
                raise RuntimeError(f"ADMET predict failed: backend[0]={exc}")

        # Small batch → send the whole thing to ONE random backend. Rationale:
        #  - ADMET model batches SMILES well; one call with K items is faster
        #    than K calls with 1 item each (HTTP overhead dominates).
        #  - Consecutive backend ports live on the same GPU (e.g. 9290-9299 =
        #    GPU 1). If we split 4 unique SMILES across 4 consecutive
        #    backends we pin one GPU; random selection spreads concurrent
        #    dispatches across all GPUs.
        # MIN_CHUNK_SIZE protects backends from trivially small chunks even
        # when the batch is large enough to distribute.  A smaller value
        # spreads work across more GPUs at the cost of more HTTP requests;
        # 4 is a good balance for the toolchain builder where batches of
        # ~40 SMILES arrive frequently from concurrent instances.
        MIN_CHUNK_SIZE = int(os.environ.get("ADMET_MIN_CHUNK_SIZE", "2"))
        # Cap the fan-out per request (ADMET_MAX_CHUNKS_PER_REQUEST).  Without
        # it, k grows with the backend pool, so adding backends merely splits
        # each batch finer and the per-backend pile-up under concurrent load is
        # unchanged (measured: a backend's throughput collapses once it gets
        # more than ~1 concurrent call).  Bounding per-request fan-out lets
        # request-level concurrency — not intra-request splitting — consume the
        # extra backends: N concurrent batches then spread across the pool
        # instead of every batch hammering all of it.  A single low-concurrency
        # batch still parallelises up to this cap.
        MAX_CHUNKS = int(os.environ.get("ADMET_MAX_CHUNKS_PER_REQUEST", "8"))

        # Rounds of (split across live backends → dispatch → re-queue what failed).
        # A chunk whose backend dies is NOT lost and does NOT void the batch: it
        # goes back in `pending` and is redistributed over whatever answered.
        pending: List[str] = list(all_unique)
        collected: List[pd.DataFrame] = []
        last_errors: Dict[int, str] = {}
        rounds = 0

        while pending and rounds < _ADMET_FAILOVER_ROUNDS:
            rounds += 1
            live = self._live_indices()
            # Backend indices are SHUFFLED rather than consecutive so that a batch
            # of 10 chunks doesn't land on the same 1-2 GPUs (consecutive ports
            # share a GPU: 9290-9299=GPU0, 9300-9309=GPU1, etc.).
            if len(pending) <= MIN_CHUNK_SIZE or len(live) == 1:
                selected = [random.choice(live)]
            else:
                k = min(len(live), MAX_CHUNKS,
                        max(1, (len(pending) + MIN_CHUNK_SIZE - 1) // MIN_CHUNK_SIZE))
                indices = list(live)
                random.shuffle(indices)
                selected = indices[:k]

            k = len(selected)
            tasks: List[Tuple[int, List[str]]] = []
            for i, backend_idx in enumerate(selected):
                chunk = pending[i::k]
                if chunk:
                    tasks.append((backend_idx, chunk))

            requeue: List[str] = []
            merge_lock = threading.Lock()

            def _call(backend_idx: int, chunk: List[str]) -> None:
                df = self._try_backend(backend_idx, chunk)
                with merge_lock:
                    if df is None:
                        requeue.extend(chunk)
                        last_errors[backend_idx] = self._backend_urls[backend_idx]
                    else:
                        if not df.empty:
                            collected.append(df)

            # Use shared executor so we don't create+join threads per batch.
            futs = [
                _ADMET_DISPATCH_EXECUTOR.submit(_call, bi, ch) for bi, ch in tasks
            ]
            for f in futs:
                f.result()

            if not requeue:
                pending = []
                break
            # A backend that answered but returned no row for a SMILES is not a
            # failure to re-queue — only chunks whose call itself failed come back.
            pending = requeue
            if rounds < _ADMET_FAILOVER_ROUNDS:
                logger.warning("ADMET: re-queueing %d SMILES onto live backends "
                               "(round %d/%d)", len(pending), rounds + 1,
                               _ADMET_FAILOVER_ROUNDS)

        if pending:
            # Only now is it a real failure: every round found a dead backend for
            # these SMILES. Raise so the caller records NA rather than a wrong value.
            urls = ", ".join(sorted(set(last_errors.values())))
            raise RuntimeError(
                f"Distributed ADMET predict failed: {len(pending)} SMILES unserved "
                f"after {rounds} round(s); last failing backend(s): {urls or 'unknown'}"
            )

        dfs = [r for r in collected if r is not None and not r.empty]
        if not dfs:
            return pd.DataFrame()
        merged = pd.concat(dfs)
        if not merged.index.is_unique:
            merged = merged[~merged.index.duplicated(keep="first")].copy()
        return merged

    def predict(self, smiles_list: List[str]) -> pd.DataFrame:
        """Called from caller threads. Blocks until the batched prediction is ready."""
        event = threading.Event()
        result_holder: List[Optional[pd.DataFrame]] = [None]
        error_holder:  List[Optional[Exception]]   = [None]
        self._queue.put((smiles_list, event, result_holder, error_holder))
        event.wait()
        if error_holder[0] is not None:
            raise error_holder[0]
        return result_holder[0] if result_holder[0] is not None else pd.DataFrame()

    def _worker(self) -> None:
        while True:
            try:
                first = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            batch = [first]
            gap_deadline = time.monotonic() + self._item_gap_s
            abs_deadline = time.monotonic() + self._abs_max_wait_s
            while len(batch) < self._max_batch_size:
                remaining = min(gap_deadline - time.monotonic(), abs_deadline - time.monotonic())
                if remaining <= 0:
                    break
                try:
                    item = self._queue.get(timeout=remaining)
                    batch.append(item)
                    gap_deadline = time.monotonic() + self._item_gap_s
                except queue.Empty:
                    break

            request_smiles: List[List[str]] = []
            all_unique: List[str] = []
            seen_global: set = set()
            for smiles_list, _e, _r, _err in batch:
                req = []
                for s in smiles_list:
                    if isinstance(s, str):
                        s = s.strip()
                        if s:
                            req.append(s)
                            if s not in seen_global:
                                seen_global.add(s)
                                all_unique.append(s)
                request_smiles.append(req)

            pred: Optional[pd.DataFrame] = None
            exc:  Optional[Exception]   = None
            try:
                if all_unique:
                    if self._backend_urls:
                        pred = self._predict_distributed(all_unique)
                    else:
                        model = get_admet_model()
                        pred = model.predict(all_unique)
                    if pred is not None and not pred.empty and not pred.index.is_unique:
                        pred = pred[~pred.index.duplicated(keep="first")].copy()
                else:
                    pred = pd.DataFrame()
            except Exception as e:
                exc = e

            for i, (smiles_list, event, result_holder, error_holder) in enumerate(batch):
                if exc is not None:
                    error_holder[0] = exc
                    result_holder[0] = pd.DataFrame()
                elif pred is None or pred.empty:
                    result_holder[0] = pd.DataFrame()
                else:
                    req = request_smiles[i]
                    if not req:
                        result_holder[0] = pd.DataFrame()
                    else:
                        try:
                            valid_idx = [s for s in req if s in pred.index]
                            result_holder[0] = pred.loc[valid_idx] if valid_idx else pd.DataFrame()
                        except Exception:
                            result_holder[0] = pd.DataFrame()
                event.set()


_ADMET_BATCHER: Optional[ADMETBatcher] = None
_ADMET_BATCHER_LOCK = threading.Lock()


def get_admet_batcher() -> ADMETBatcher:
    """Return a process-wide ADMETBatcher singleton.

    Honours ``ADMET_SERVER_URLS`` (comma-separated) — when set, the batcher
    distributes work to remote backends instead of loading the model locally.
    """
    global _ADMET_BATCHER
    if _ADMET_BATCHER is None:
        with _ADMET_BATCHER_LOCK:
            if _ADMET_BATCHER is None:
                urls_env = os.environ.get("ADMET_SERVER_URLS", "").strip()
                backend_urls = [u.strip() for u in urls_env.split(",") if u.strip()] if urls_env else []
                if not backend_urls:
                    get_admet_model()  # load local model only when no remote backend
                _ADMET_BATCHER = ADMETBatcher(
                    max_batch_size  = 512,
                    item_gap_ms     = 20.0,
                    abs_max_wait_ms = 1500.0,
                    backend_urls    = backend_urls,
                )
    return _ADMET_BATCHER


# Batches at or above this size bypass the ADMETBatcher queue and call
# _predict_distributed directly.  This avoids the 20ms batching window
# delay and — more importantly — eliminates the per-worker-process
# isolation problem: each uvicorn worker has its own ADMETBatcher, so
# small batches from different workers can't be merged.  When the caller
# already provides a large batch (e.g. 40 SMILES from _batch_check_constraints),
# there's nothing to accumulate — distributing immediately is strictly better.
_DIRECT_DISPATCH_THRESHOLD = int(os.environ.get("ADMET_DIRECT_THRESHOLD", "8"))


def predict_admet(smiles_list: List[str]) -> pd.DataFrame:
    """Run admet_ai on *smiles_list*; returns a DataFrame indexed by SMILES.

    Large batches (≥ ``_DIRECT_DISPATCH_THRESHOLD``) are dispatched
    directly to ADMET backends without going through the batching queue,
    so they benefit from full multi-backend parallelism regardless of
    how many uvicorn workers are running.  Small batches still go through
    :class:`ADMETBatcher` to accumulate concurrent single-SMILES calls.
    """
    if not smiles_list:
        return pd.DataFrame()

    uniq: List[str] = []
    seen: set = set()
    for s in smiles_list:
        if not isinstance(s, str):
            continue
        s = s.strip()
        if s and s not in seen:
            seen.add(s)
            uniq.append(s)

    if not uniq:
        return pd.DataFrame()

    # Drop pathological strings BEFORE they reach a backend (see
    # _ADMET_MAX_SMILES_LEN). Length is checked rather than atom count so this
    # stays O(1) per SMILES and needs no RDKit parse on the hot path.
    if _ADMET_MAX_SMILES_LEN > 0:
        kept = [s for s in uniq if len(s) <= _ADMET_MAX_SMILES_LEN]
        if len(kept) != len(uniq):
            skipped = len(uniq) - len(kept)
            longest = max(len(s) for s in uniq)
            logger.warning(
                "ADMET: skipping %d oversized SMILES (longest %d chars > limit %d); "
                "they get no ADMET value rather than wedging a backend",
                skipped, longest, _ADMET_MAX_SMILES_LEN,
            )
            uniq = kept
        if not uniq:
            return pd.DataFrame()

    batcher = get_admet_batcher()

    if len(uniq) >= _DIRECT_DISPATCH_THRESHOLD and batcher._backend_urls:
        # Large batch: bypass batcher queue, distribute directly.
        pred = batcher._predict_distributed(uniq)
    else:
        # Small batch: go through batcher to accumulate concurrent calls.
        pred = batcher.predict(uniq)

    if pred is None or pred.empty:
        return pd.DataFrame()

    if not pred.index.is_unique:
        pred = pred[~pred.index.duplicated(keep="first")].copy()

    return pred


def predict_tdc(smiles_list: List[str], props: List[str]) -> pd.DataFrame:
    """Call TDC Oracle Server for *props* (DRD2/GSK3B/JNK3); returns DataFrame indexed by SMILES."""
    from molkit.utils.tdc_client import _call_oracle

    if not smiles_list or not props:
        return pd.DataFrame()

    rows: Dict[str, Dict[str, float]] = {s: {} for s in smiles_list}
    for prop in props:
        oracle_name = TDC_PROP_MAP[prop]
        try:
            scores = _call_oracle(oracle_name, smiles_list)
            if isinstance(scores, (int, float)):
                scores = [scores]
            for smi, score in zip(smiles_list, scores):
                rows[smi][prop] = float(score)
        except Exception as exc:
            logger.warning("TDC oracle '%s' failed: %s", oracle_name, exc)
            for smi in smiles_list:
                rows[smi][prop] = float("nan")

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "smiles"
    return df


# ---------------------------------------------------------------------------
# 2) Constraint normalisation + scoring
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RangeConstraint:
    lo: float
    hi: float

    def contains(self, x: float) -> bool:
        if pd.isna(x):
            return False
        return self.lo <= x <= self.hi

    def violation_distance(self, x: float) -> float:
        """0 if inside the range. Returns ∞ for NaN."""
        if pd.isna(x):
            return float("inf")
        if x < self.lo:
            return self.lo - x
        if x > self.hi:
            return x - self.hi
        return 0.0

    def width(self) -> float:
        w = self.hi - self.lo
        return w if w > 0 else 1.0


def normalize_constraints(range_dict: Dict[str, List[float]]) -> Dict[str, RangeConstraint]:
    if not isinstance(range_dict, dict) or not range_dict:
        raise ValueError("range_constraints must be a non-empty dict like {'QED':[0.5,1.0], ...}")

    out: Dict[str, RangeConstraint] = {}
    for k, v in range_dict.items():
        if k not in SUPPORTED_CONSTRAINT_KEYS:
            raise KeyError(
                f"Constraint key '{k}' not supported. "
                f"Supported: {sorted(SUPPORTED_CONSTRAINT_KEYS)}"
            )
        if not (isinstance(v, (list, tuple)) and len(v) == 2):
            raise ValueError(f"Constraint for '{k}' must be [lo, hi]. Got: {v}")
        lo, hi = float(v[0]), float(v[1])
        if lo > hi:
            lo, hi = hi, lo
        out[k] = RangeConstraint(lo=lo, hi=hi)
    return out


def score_row(
    row: pd.Series,
    constraints: Dict[str, RangeConstraint],
    weights: Optional[Dict[str, float]] = None,
    mode: str = "hinge_l2",
    *,
    normalize_by_range: bool = True,
) -> Tuple[float, Dict[str, Any]]:
    """Compute (score, detail) for a single SMILES row.

    score = +1 bonus if all OK, minus weighted penalty.
    """
    if weights is None:
        weights = {k: 1.0 for k in constraints}

    all_ok      = True
    penalties   : Dict[str, float] = {}
    oks         : Dict[str, bool]  = {}
    total_penalty = 0.0

    for prop, rc in constraints.items():
        x  = row.get(prop, math.nan)
        ok = rc.contains(float(x)) if not pd.isna(x) else False
        oks[prop] = ok
        if not ok:
            all_ok = False

        dist = rc.violation_distance(float(x)) if not pd.isna(x) else float("inf")
        if normalize_by_range and not math.isinf(dist):
            dist /= rc.width()

        w = float(weights.get(prop, 1.0))

        if mode == "hard":
            penalties[prop] = 0.0
            continue

        if math.isinf(dist):
            p = w * 1e6
        elif mode == "hinge_l2":
            p = w * (dist ** 2)
        elif mode == "hinge_l1":
            p = w * abs(dist)
        else:
            raise ValueError(f"Unknown score mode: {mode}")

        penalties[prop]  = p
        total_penalty   += p

    score = (-total_penalty + (1.0 if all_ok else 0.0)) if mode != "hard" else (1.0 if all_ok else 0.0)
    return score, {"all_ok": all_ok, "oks": oks, "penalties": penalties}


def _standardize_predictions(
    pred_df: pd.DataFrame,
    smiles_list: List[str],
    props: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """Re-index admet_ai output to use short property names (admet_ai props only)."""
    admet_props = [p for p in props if p in COLUMN_MAP]

    empty = pd.DataFrame({"smiles": []})
    for p in admet_props:
        empty[p] = []

    if not smiles_list or pred_df is None or pred_df.empty:
        return empty, []

    if not pred_df.index.is_unique:
        pred_df = pred_df[~pred_df.index.duplicated(keep="first")].copy()

    required_cols = [COLUMN_MAP[p] for p in admet_props]
    missing = [c for c in required_cols if c not in pred_df.columns]
    if missing:
        raise KeyError(
            f"Missing columns in admet_ai output: {missing}\n"
            f"Available: {list(pred_df.columns)}"
        )

    survived = set(pred_df.index.astype(str))
    kept     = list(dict.fromkeys(s for s in smiles_list if s in survived))
    if not kept:
        return empty, []

    pred_aligned = pred_df.loc[kept]
    if not pred_aligned.index.is_unique:
        pred_aligned = pred_aligned[~pred_aligned.index.duplicated(keep="first")].copy()
        pred_aligned = pred_aligned.loc[kept]

    out = pd.DataFrame({"smiles": kept})
    for p in admet_props:
        col = COLUMN_MAP[p]
        out[p] = pd.to_numeric(pred_aligned[col], errors="coerce").to_numpy()

    return out, kept


def evaluate_smiles_batch(
    smiles_list: List[str],
    range_constraints: Dict[str, List[float]],
    *,
    weights: Optional[Dict[str, float]] = None,
    score_mode: str = "hinge_l2",
    normalize_by_range: bool = True,
) -> pd.DataFrame:
    """Score *smiles_list* against range constraints; returns sorted DataFrame."""
    constraints = normalize_constraints(range_constraints)
    props       = list(constraints.keys())

    admet_props = [p for p in props if p in COLUMN_MAP]
    tdc_props   = [p for p in props if p in TDC_PROP_MAP]

    # ADMET and TDC are independent network calls to separate services; fire
    # them in parallel so each caller's generation pays max(ADMET, TDC) instead
    # of ADMET + TDC. With the two process-wide batchers (ADMETBatcher /
    # TDCBatcher), concurrency across callers is preserved.
    pred = None
    tdc_df: pd.DataFrame = pd.DataFrame()

    def _run_admet() -> None:
        nonlocal pred
        pred = predict_admet(smiles_list)

    def _run_tdc() -> None:
        nonlocal tdc_df
        # Pass the full smiles_list (not `kept` from ADMET) so the call can
        # start without waiting for the ADMET response. Invalid SMILES yield
        # NaN from the TDC server and are dropped when merged below.
        tdc_df = predict_tdc(smiles_list, tdc_props)

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="eval-par") as _ex:
        admet_fut = _ex.submit(_run_admet)
        tdc_fut   = _ex.submit(_run_tdc) if tdc_props else None
        admet_fut.result()
        if tdc_fut is not None:
            tdc_fut.result()

    std, kept = _standardize_predictions(pred, smiles_list, props)

    if std.empty and not tdc_props:
        std["score"]  = []
        std["all_ok"] = []
        return std

    # Merge TDC predictions when needed
    if tdc_props:
        if not tdc_df.empty:
            if std.empty:
                std = pd.DataFrame({"smiles": tdc_df.index.tolist()})
            for p in tdc_props:
                if p in tdc_df.columns:
                    std[p] = std["smiles"].map(tdc_df[p].to_dict()).astype(float)

    if std.empty:
        std["score"]  = []
        std["all_ok"] = []
        return std

    scores: List[float] = []
    ok_list: List[bool] = []

    for _, r in std.iterrows():
        s, d = score_row(r, constraints=constraints, weights=weights,
                         mode=score_mode, normalize_by_range=normalize_by_range)
        scores.append(float(s))
        ok_list.append(bool(d["all_ok"]))

    std["score"]  = scores
    std["all_ok"] = ok_list
    return std.sort_values(["all_ok", "score"], ascending=[False, False]).reset_index(drop=True)

