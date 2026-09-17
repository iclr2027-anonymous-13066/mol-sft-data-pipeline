#!/usr/bin/env python3
"""
describe_scaffolds.py
=====================
Takes a list of SMILES, analyses and names each molecule's **Murcko scaffold** deterministically
(scaffold_analyzer + scaffold_describer), has vLLM (Qwen3) polish the controlled
natural-language draft into fluent prose, then verifies it with an **RDKit substructure
check** before writing it out.

Pipeline, per molecule:
  SMILES -> Murcko scaffold -> ring-system decomposition -> ring / linker / topology
  analysis -> substituent removal + attachment points -> controlled NL draft -> (vLLM)
  polish -> deterministic validation (hallucination / style) -> revise ->
  SMARTS + RDKit substructure verification -> write

For large inputs (parquet files of millions of rows) it streams: the input is read in
chunks, analysed and generated, and each finished item is appended to the output JSONL as
a single line straight away. An interrupted run keeps its progress, and re-running skips
the SMILES already processed.

This file holds only the vLLM client (VLLMPool) and the CLI/orchestration. Prompts,
validation and rendering live in scaffold_describer (+ scaffold_prompts); the structure
analysis lives in scaffold_analyzer.

LLM backend: a pool of OpenAI-compatible vLLM servers, round-robined across several hosts
(default localhost:8080-8087 + remote-host:8080-8087, Qwen/Qwen3.6-27B). Servers that do
not answer are pruned from the pool during model detection, so work only spreads over the
live ones.

Input formats, detected from the extension:
  - .parquet    : streamed with pyarrow, using the 'smiles' column (or --smiles-key)
  - .txt / .smi : one SMILES per line; anything after whitespace is ignored, so
                  "SMILES name" is accepted
  - .jsonl      : one JSON object per line; read from 'smiles'/'ref_smiles'/'smi' or
                  --smiles-key
  - .json       : a list of strings, or a list of {smiles/ref_smiles} dicts
Examples
------
  # build training data from the first 5,000 molecules of train_pool.parquet, into a new folder
  python describe_scaffolds.py \
      --input data/develop/chembl_zinc_split/train_pool.parquet \
      --run-name scaffold_train_pool --limit 5000
  # default output: data/training_data/<run-name>/scaffolds.jsonl

Dependencies: openai>=1.0 (AsyncOpenAI), tqdm, rdkit, and pyarrow for parquet input
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from typing import Iterator, Optional

from openai import AsyncOpenAI
from tqdm import tqdm
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from scaffold_analyzer import analyze_scaffold
from scaffold_describer import ScaffoldDescriber, normalize_text, render_template, validate_text

DEFAULT_SERVERS = "localhost:8080-8087"
# DEFAULT_SERVERS = "localhost:8080-8087,remote-host:8080-8087"
DEFAULT_OUT_ROOT = "data/training_data"

_ABBR_RE = re.compile(r"(?:i\.e|e\.g|etc|vs|approx|cf|Fig|No|Dr|Mr|Ms|St)\.$", re.IGNORECASE)
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    parts = [p for p in _SENT_SPLIT_RE.split(text.strip()) if p]
    merged: list[str] = []
    for p in parts:
        if merged and _ABBR_RE.search(merged[-1]):
            merged[-1] += " " + p
        else:
            merged.append(p)
    return merged


# --------------------------------------------------------------------------- #
#  Port / server spec parsing
# --------------------------------------------------------------------------- #
def parse_ports(spec: str) -> list[int]:
    """'8080-8087' or '8080,8081,8084' -> a list of ports."""
    ports: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            ports.extend(range(int(lo), int(hi) + 1))
        else:
            ports.append(int(part))
    return ports


def build_base_urls(servers: Optional[str], host: Optional[str], base_ports: str) -> list[str]:
    """Build the list of OpenAI base_urls for the server pool.

    - With ``--host``, the pool is those hosts (comma-separated) x ``--base-ports``.
    - Otherwise it comes from ``--servers`` (comma-separated ``host:portspec``; a missing
      portspec falls back to base_ports).
    """
    urls: list[str] = []
    if host:
        ports = parse_ports(base_ports)
        for h in [x.strip() for x in host.split(",") if x.strip()]:
            urls += [f"http://{h}:{p}/v1" for p in ports]
        return urls
    for entry in (servers or "").split(","):
        entry = entry.strip()
        if not entry:
            continue
        if ":" in entry:
            h, pspec = entry.rsplit(":", 1)
            ports = parse_ports(pspec)
        else:
            h, ports = entry, parse_ports(base_ports)
        urls += [f"http://{h}:{p}/v1" for p in ports]
    return urls


# --------------------------------------------------------------------------- #
#  Async vLLM client: round-robin over servers, with a per-server concurrency limit
# --------------------------------------------------------------------------- #
class VLLMPool:
    """A pool of OpenAI-compatible vLLM servers. Prompts and validation are unified through
    ScaffoldDescriber."""

    def __init__(self, base_urls: list[str], per_server_concurrency: int = 5,
                 api_key: str = "EMPTY", timeout: float = 180.0, max_retries: int = 4,
                 reasoning_effort: Optional[str] = "low", temperature: float = 0.4,
                 max_tokens: int = 1024, max_sentences: int = 0,
                 model: Optional[str] = None) -> None:
        if not base_urls:
            raise ValueError("base_urls is empty")
        self.base_urls = base_urls
        self.per_server_concurrency = per_server_concurrency
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.reasoning_effort = reasoning_effort
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_sentences = max_sentences
        self._describer = ScaffoldDescriber()
        self._forced_model = model
        self._clients = [AsyncOpenAI(base_url=u, api_key=api_key, timeout=timeout, max_retries=0)
                         for u in base_urls]
        self._sems = [asyncio.Semaphore(per_server_concurrency) for _ in self._clients]
        self._models: list[Optional[str]] = [model for _ in self._clients]

    async def discover_models(self) -> None:
        if self._forced_model:
            return

        async def _one(i: int) -> None:
            last = None
            for attempt in range(self.max_retries):
                try:
                    listing = await self._clients[i].models.list()
                    if listing.data:
                        self._models[i] = listing.data[0].id
                        return
                    last = RuntimeError("empty model list")
                except Exception as exc:  # noqa: BLE001
                    last = exc
                if attempt + 1 < self.max_retries:
                    await asyncio.sleep(min(2.0 * (2 ** attempt), 15.0))
            print(f"  [warn] {self.base_urls[i]}: model discovery failed: {last}", file=sys.stderr)

        await asyncio.gather(*(_one(i) for i in range(len(self._clients))))

    def prune_dead(self) -> int:
        """Drop the servers whose model could not be detected (they did not answer). Returns how
        many were dropped."""
        if self._forced_model:  # a forced model means detection never ran, so nothing is dropped
            return 0
        live = [i for i, m in enumerate(self._models) if m]
        dropped = len(self._clients) - len(live)
        if dropped:
            self.base_urls = [self.base_urls[i] for i in live]
            self._clients = [self._clients[i] for i in live]
            self._sems = [self._sems[i] for i in live]
            self._models = [self._models[i] for i in live]
        return dropped

    @property
    def detected_models(self) -> list[Optional[str]]:
        return list(self._models)

    @property
    def n_servers(self) -> int:
        return len(self._clients)

    @staticmethod
    def _family(model: Optional[str]) -> str:
        m = (model or "").lower()
        if "gpt-oss" in m or "gpt_oss" in m:
            return "gpt-oss"
        if "qwen" in m:
            return "qwen"
        return "other"

    def _extra_body(self, model, think: Optional[bool] = None) -> Optional[dict]:
        eff = (self.reasoning_effort or "none").lower()
        fam = self._family(model)
        if fam == "gpt-oss":
            return {"reasoning_effort": eff} if eff != "none" else None
        if fam == "qwen":
            en = (eff in ("medium", "high")) if think is None else think
            return {"chat_template_kwargs": {"enable_thinking": en}}
        return None

    async def describe(self, server_idx: int, an: dict, draft: Optional[str] = None) -> str:
        """Have the LLM polish the draft from a scaffold analysis dict into fluent prose."""
        idx = server_idx % len(self._clients)
        model = self._models[idx] or self._forced_model
        if model is None:
            raise RuntimeError(f"no model for server {self.base_urls[idx]}")
        d = self._describer
        draft = draft if draft is not None else render_template(an)
        async with self._sems[idx]:
            text = await self._chat(
                idx, model,
                [{"role": "system", "content": d.system_prompt()},
                 {"role": "user", "content": d.user_prompt(an, draft=draft)}])
            if self.max_sentences and len(_split_sentences(text)) > self.max_sentences:
                try:
                    text = await self._chat(
                        idx, model,
                        [{"role": "system", "content": "You compress text. Output only the rewrite."},
                         {"role": "user", "content": d.condense_prompt(text, self.max_sentences)}],
                        think=False)
                except Exception:  # noqa: BLE001
                    pass
        if self.max_sentences:
            sents = _split_sentences(text)
            if len(sents) > self.max_sentences:
                text = " ".join(sents[: self.max_sentences])
        return text

    async def revise(self, server_idx: int, an: dict, draft: str,
                     violations: Optional[list] = None) -> str:
        idx = server_idx % len(self._clients)
        model = self._models[idx] or self._forced_model
        if model is None:
            raise RuntimeError(f"no model for server {self.base_urls[idx]}")
        d = self._describer
        async with self._sems[idx]:
            text = await self._chat(
                idx, model,
                [{"role": "system", "content": d.system_prompt()},
                 {"role": "user", "content": d.revise_prompt(an, draft, violations)}])
        if self.max_sentences:
            sents = _split_sentences(text)
            if len(sents) > self.max_sentences:
                text = " ".join(sents[: self.max_sentences])
        return text

    async def _chat(self, idx: int, model: str, messages: list[dict],
                    think: Optional[bool] = None) -> str:
        client = self._clients[idx]
        last = None
        for attempt in range(self.max_retries):
            eb = self._extra_body(model, think=(False if attempt > 0 else think))
            try:
                resp = await client.chat.completions.create(
                    model=model, messages=messages, temperature=self.temperature,
                    max_tokens=self.max_tokens * (1 + attempt), extra_body=eb)
                text = _extract_text(resp)
                if text:
                    return text
                last = RuntimeError("empty/rejected content")
            except Exception as exc:  # noqa: BLE001
                last = exc
            await asyncio.sleep(min(2.0 * (2 ** attempt), 30.0))
        raise RuntimeError(f"failed after {self.max_retries} attempts on {self.base_urls[idx]}: {last}")


def _extract_text(resp) -> str:
    if not resp.choices:
        return ""
    content = resp.choices[0].message.content or ""
    content = re.sub(r"<think>.*?</think>", " ", content, flags=re.DOTALL | re.IGNORECASE)
    if "<think>" in content.lower():
        content = re.split(r"</think>", content, flags=re.IGNORECASE)[-1].replace("<think>", " ")
    text = " ".join(content.split()).strip()
    if text and _looks_like_thinking(text):
        return ""
    return text


_THINK_MARKERS = ("thinking process", "let me think", "i need to", "first, i ",
                  "here's a thinking", "here is a thinking", "step 1:", "the user wants",
                  "the user provided", "we need to", "chain of thought", "my reasoning",
                  "let's analyze", "okay, let")


def _looks_like_thinking(t: str) -> bool:
    tl = t.lower()
    if any(p in tl for p in _THINK_MARKERS):
        return True
    if re.match(r"\s*(\*\*|#{1,6}\s|\d+\.\s|[-*]\s)", t):
        return True
    return False


# --------------------------------------------------------------------------- #
#  SMARTS / RDKit substructure verification
# --------------------------------------------------------------------------- #
def verify_scaffold(smiles: str, scaffold_smarts: str) -> dict:
    """Verify with RDKit that the extracted scaffold SMARTS really is a substructure of the
    original molecule."""
    out = {"smarts_parsed": False, "substructure_match": False}
    if not scaffold_smarts:
        return out
    mol = Chem.MolFromSmiles(smiles)
    q = Chem.MolFromSmarts(scaffold_smarts)
    if mol is None or q is None:
        return out
    out["smarts_parsed"] = True
    try:
        out["substructure_match"] = mol.HasSubstructMatch(q)
    except Exception:  # noqa: BLE001
        out["substructure_match"] = False
    return out


# --------------------------------------------------------------------------- #
#  Analysis -> result item
# --------------------------------------------------------------------------- #
def _ring_systems_summary(an: dict) -> list[dict]:
    out = []
    for rs in an["ring_systems"]:
        d = {"name": rs["name"], "name_source": rs["name_source"],
             "n_rings": rs["n_rings"], "aromatic": rs["aromatic"],
             "aromaticity": rs.get("aromaticity_desc", ""),
             "internal_topology": rs["internal_topology"],
             "ring_sizes": rs["ring_sizes"], "heteroatoms": rs["hetero_counts"]}
        # exocyclic carbonyl on a ring (lactam / lactone / -one / imide)
        if rs.get("n_ring_carbonyls"):
            d["n_ring_carbonyls"] = rs["n_ring_carbonyls"]
            d["carbonyl_groups"] = rs.get("carbonyl_groups", [])
            if rs.get("carbonyl_desc"):
                d["carbonyl_desc"] = rs["carbonyl_desc"]
        # NOTE: substituent attachment points (attachment_points / n_attachment_points) and
        # substitution patterns (substitution_relations, benzene ortho/meta/para) are no
        # longer stored. Substituents are stripped by the Murcko reduction, so no evaluation
        # SMARTS can check them — which is also why they are kept out of the description.
        # Where a linker attaches is still preserved, in connections[].a_position/b_position.
        out.append(d)
    return out


def _connections_summary(an: dict) -> list[dict]:
    out = []
    for c in an.get("connections", []):
        lk = c.get("linker") or {}
        d = {"a": c["a"], "b": c["b"], "relation": c["relation"], "type": c["type"],
             # linker shape (the pattern, including =O/=S branches) + functional group +
             # total length + atom sequence
             "linker_pattern": lk.get("pattern", ""),
             "functional_groups": lk.get("functional_groups", []),
             "linker_length": lk.get("length", 0), "bond_span": lk.get("bond_span"),
             "linker_atoms": lk.get("atom_sequence", []),
             # which atom/position of each ring system the two ends attach to
             "a_atom": c.get("a_atom"), "b_atom": c.get("b_atom"),
             "a_position": (c.get("a_position") or {}).get("position"),
             "b_position": (c.get("b_position") or {}).get("position")}
        out.append(d)
    return out


def build_item(smiles: str, an: dict) -> dict:
    """The result item holding the deterministic analysis and verification, before the LLM
    description."""
    verify = verify_scaffold(smiles, an.get("scaffold_smarts", "")) if an.get("has_scaffold") else \
        {"smarts_parsed": False, "substructure_match": False}
    return {
        "smiles": an.get("smiles", smiles),
        "input_smiles": smiles,
        "parse_ok": an.get("parse_ok", False),
        "has_scaffold": an.get("has_scaffold", False),
        # 'ring' (Murcko ring skeleton) | 'functional_group' (acyclic FG skeleton) | 'none'
        "scaffold_kind": an.get("scaffold_kind", "ring" if an.get("has_scaffold") else "none"),
        "functional_groups": an.get("functional_groups", []),
        "scaffold_smiles": an.get("scaffold_smiles", ""),
        "scaffold_smarts": an.get("scaffold_smarts", ""),
        # per-dimension projected SMARTS (skeleton/element/aromaticity/bond/ring), for
        # evaluation as a single projection
        "dimension_smarts": an.get("dimension_smarts", {}),
        # evaluation spec: fixed -> a single SMARTS; free -> a match-any set J
        # (the verdict plus the per-dimension sets)
        "eval_query": an.get("eval_query", {}),
        "n_ring_systems": an.get("n_ring_systems", 0),
        "n_rings_total": an.get("n_rings_total", 0),
        "aromaticity": an.get("aromaticity", ""),
        "ring_systems": _ring_systems_summary(an) if an.get("has_scaffold") else [],
        "connections": _connections_summary(an) if an.get("has_scaffold") else [],
        "linker_attachment_points": [
            {"scaffold_atom": p["scaffold_atom"], "element": p["element"],
             "n_substituents": p["n_substituents"], "position": p.get("position")}
            for p in an.get("linker_attachment_points", [])],
        "topology_summary": an.get("topology_summary", ""),
        "template_draft": render_template(an),
        "description": "",
        "scaffold_verified": bool(verify["substructure_match"]),
    }


# --------------------------------------------------------------------------- #
#  Input streaming, large parquet included: skip `offset` rows and yield at most `limit` SMILES.
# --------------------------------------------------------------------------- #
def _smiles_from_obj(obj, smiles_key: Optional[str]) -> Optional[str]:
    if isinstance(obj, str):
        return obj.split()[0] if obj.strip() else None
    if isinstance(obj, dict):
        if smiles_key and obj.get(smiles_key):
            return str(obj[smiles_key])
        for k in ("smiles", "ref_smiles", "smi", "SMILES", "canonical_smiles"):
            if obj.get(k):
                return str(obj[k])
    return None


def iter_input_smiles(path: str, smiles_key: Optional[str] = None,
                      offset: int = 0, limit: Optional[int] = None) -> Iterator[str]:
    """Stream SMILES lazily out of the input file (.parquet/.txt/.smi/.jsonl/.json)."""
    ext = os.path.splitext(path)[1].lower()
    yielded = 0
    i = -1

    def _emit(s):
        return s and s.strip()

    if ext == ".parquet":
        import pyarrow.parquet as pq
        col = smiles_key or "smiles"
        pf = pq.ParquetFile(path)
        if col not in pf.schema_arrow.names:
            raise ValueError(f"column {col!r} not in parquet; available e.g. "
                             f"{pf.schema_arrow.names[:8]} ... (use --smiles-key)")
        for batch in pf.iter_batches(batch_size=16384, columns=[col]):
            for s in batch.column(0).to_pylist():
                i += 1
                if i < offset or not _emit(s):
                    continue
                yield str(s).split()[0]
                yielded += 1
                if limit and yielded >= limit:
                    return
    elif ext == ".json":
        with open(path) as fh:
            data = json.load(fh)
        for o in (data if isinstance(data, list) else []):
            s = _smiles_from_obj(o, smiles_key)
            i += 1
            if i < offset or not _emit(s):
                continue
            yield s
            yielded += 1
            if limit and yielded >= limit:
                return
    elif ext == ".jsonl":
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                except json.JSONDecodeError:
                    continue
                s = _smiles_from_obj(o, smiles_key)
                i += 1
                if i < offset or not _emit(s):
                    continue
                yield s
                yielded += 1
                if limit and yielded >= limit:
                    return
    else:  # .txt / .smi / anything else
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                i += 1
                if i < offset:
                    continue
                yield line.split()[0]
                yielded += 1
                if limit and yielded >= limit:
                    return


def _load_done_keys(path: str) -> set:
    """Read the set of input_smiles already processed from an existing output JSONL (resume)."""
    done: set = set()
    if not os.path.exists(path):
        return done
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            k = o.get("input_smiles") or o.get("smiles")
            if k:
                done.add(k)
    return done


# --------------------------------------------------------------------------- #
#  Unit of work: one molecule. Analysis runs on a CPU executor, the LLM on its assigned server.
# --------------------------------------------------------------------------- #
def _error_item(smi: str, exc: Exception) -> dict:
    """Minimal record for a molecule whose analysis itself failed. It must still be written, so
    a re-run does not process it again."""
    return {"smiles": smi, "input_smiles": smi, "parse_ok": False, "has_scaffold": False,
            "scaffold_kind": "none", "functional_groups": [],
            "scaffold_smiles": "", "scaffold_smarts": "", "dimension_smarts": {},
            "eval_query": {},
            "n_ring_systems": 0,
            "n_rings_total": 0, "aromaticity": "", "ring_systems": [], "connections": [],
            "linker_attachment_points": [], "topology_summary": "", "template_draft": "",
            "description": "", "scaffold_verified": False, "analysis_error": str(exc)[:200]}


def analyze_one(smi: str) -> tuple:
    """Return (an, item). RDKit exceptions (e.g. MolToSmarts 'bad bond stereo') are contained
    as error records. This is synchronous CPU work, so the caller runs it on an executor and
    the event loop is never blocked."""
    try:
        an = analyze_scaffold(smi)
        return an, build_item(smi, an)
    except Exception as exc:  # noqa: BLE001
        return None, _error_item(smi, exc)


async def describe_one(pool: VLLMPool, server_idx: int, it: dict, an, validate_passes: int,
                       out_fh, write_lock: asyncio.Lock, pbar, stats: dict) -> None:
    """Polish, validate and repair one molecule on its assigned server_idx, then append it to
    the JSONL immediately."""
    try:
        if an is None or it.get("analysis_error"):
            stats["failed"] += 1
            pbar.write(f"  [analysis-fail] {it['input_smiles'][:60]!r}: "
                       f"{it.get('analysis_error', '')}")
            return   # not counted as a describe; the record is still written in finally
        if not it["has_scaffold"]:
            it["description"] = it["template_draft"]   # acyclic: the deterministic description
            stats["no_scaffold"] += 1
        elif it.get("scaffold_kind") == "functional_group":
            # Functional-group skeleton: bypass the ring-centred validate_text and keep the
            # deterministic template as it is.
            it["description"] = it["template_draft"]
            if it.get("scaffold_verified"):
                stats["verified"] += 1
        else:
            desc = normalize_text(
                await pool.describe(server_idx, an, draft=it["template_draft"]))
            viol = validate_text(desc, an)
            for _ in range(max(0, validate_passes)):
                if not viol:
                    break
                rev = normalize_text(await pool.revise(server_idx, an, desc, viol))
                rviol = validate_text(rev, an)
                if len(rviol) < len(viol):
                    desc, viol = rev, rviol
                else:
                    break
            it["description"] = desc
            if viol:
                it["description_violations"] = viol
                stats["with_violations"] += 1
            if it["scaffold_verified"]:
                stats["verified"] += 1
        stats["described"] += 1
    except Exception as exc:  # noqa: BLE001
        it["description"] = ""
        stats["failed"] += 1
        pbar.write(f"  [fail] {it['input_smiles'][:60]!r}: {exc}")
    finally:
        async with write_lock:
            out_fh.write(json.dumps(it, ensure_ascii=False) + "\n")
            out_fh.flush()
        pbar.update(1)


async def main_async(args: argparse.Namespace) -> int:
    # --- output location: create one new folder and write the JSONL inside it ---
    if args.output:
        output_path = args.output
        out_dir = os.path.dirname(os.path.abspath(output_path)) or "."
    else:
        run = args.run_name or ("scaffold_" + os.path.splitext(os.path.basename(args.input))[0])
        out_dir = os.path.join(args.out_dir, run)
        output_path = os.path.join(out_dir, "scaffolds.jsonl")
    os.makedirs(out_dir, exist_ok=True)
    meta_path = os.path.join(out_dir, "meta.json")
    # Ship the dataset README alongside the output, for someone seeing it for the first
    # time: it documents the schema and the evaluation strategy.
    _here = os.path.dirname(os.path.abspath(__file__))
    for _src_name, _dst_name in (("DATASET_README.md", "README.md"),):
        _src = os.path.join(_here, _src_name)
        try:
            if os.path.exists(_src):
                import shutil
                shutil.copyfile(_src, os.path.join(out_dir, _dst_name))
        except OSError:
            pass

    done = set() if args.overwrite else _load_done_keys(output_path)
    if args.overwrite and os.path.exists(output_path):
        os.remove(output_path)
    if done:
        print(f"# Resuming: {len(done)} item(s) already in {output_path} -> will skip.")

    # --- vLLM server pool ---
    base_urls = build_base_urls(args.servers, args.host, args.base_ports)
    print(f"# vLLM candidate servers ({len(base_urls)}): "
          f"{base_urls[0]} ... {base_urls[-1]} ({args.concurrency_per_server} concurrent each)")
    pool = VLLMPool(base_urls=base_urls, per_server_concurrency=args.concurrency_per_server,
                    timeout=args.timeout, max_retries=args.max_retries,
                    reasoning_effort=args.reasoning_effort, temperature=args.temperature,
                    max_tokens=args.max_tokens, max_sentences=args.max_sentences, model=args.model)
    await pool.discover_models()
    dropped = pool.prune_dead()
    if dropped:
        print(f"# [warn] dropped {dropped} unreachable server(s); using {pool.n_servers} live server(s).")
    if pool.n_servers == 0:
        print("error: no live vLLM server. Are they up? Try --servers/--host/--model.", file=sys.stderr)
        return 3
    uniq = sorted({m for m in pool.detected_models if m}) or [args.model]
    print(f"# Live servers: {pool.n_servers} (in-flight cap "
          f"{pool.n_servers * args.concurrency_per_server}); model(s): {uniq}")

    # --- streaming + dynamic work queue ---
    #   Each server gets exactly `concurrency_per_server` consumer coroutines, which keep
    #   pulling the next molecule off a shared queue. Because assignment is not static
    #   (molecule j -> server j % N), a fast local server takes more and a slow remote one
    #   takes less, and no server sits idle. It also removes the 2000-item chunk barrier:
    #   everything runs continuously, with the analysis overlapped on an executor.
    stats = {"described": 0, "failed": 0, "no_scaffold": 0, "verified": 0,
             "with_violations": 0, "skipped": 0}
    total_hint = f"{args.limit}" if args.limit else "all"
    print(f"# Input: {args.input} (offset={args.offset}, limit={total_hint}) -> {output_path}")
    print(f"# Dispatch: dynamic queue, {args.concurrency_per_server} persistent worker(s) per "
          f"server × {pool.n_servers} servers = {pool.n_servers * args.concurrency_per_server} "
          "always-busy slots.")
    write_lock = asyncio.Lock()
    pbar = tqdm(total=args.limit, desc="Scaffolds", unit="mol", smoothing=0.02)
    seen_this_run: set = set()
    loop = asyncio.get_event_loop()
    n_slots = pool.n_servers * args.concurrency_per_server
    queue: asyncio.Queue = asyncio.Queue(maxsize=n_slots * 3)
    _DONE = object()
    written = {"n": 0}

    def _write_meta() -> None:
        meta = {"input": args.input, "smiles_key": args.smiles_key, "output": output_path,
                "servers": pool.base_urls, "model": uniq, "offset": args.offset,
                "limit": args.limit, "concurrency_per_server": args.concurrency_per_server,
                "stats": stats}
        with open(meta_path, "w") as fh:
            json.dump(meta, fh, indent=2, ensure_ascii=False)

    out_fh = open(output_path, "a")

    async def consumer(server_idx: int) -> None:
        while True:
            payload = await queue.get()
            if payload is _DONE:
                return
            it, an = payload
            await describe_one(pool, server_idx, it, an, args.validate_passes,
                               out_fh, write_lock, pbar, stats)
            written["n"] += 1
            if written["n"] % 1000 == 0:
                _write_meta()

    # concurrency_per_server consumers per server, pinned to it, so each server runs exactly
    # that many at once.
    consumers = [asyncio.create_task(consumer(i))
                 for i in range(pool.n_servers) for _ in range(args.concurrency_per_server)]
    try:
        for smi in iter_input_smiles(args.input, args.smiles_key, args.offset, args.limit):
            if smi in done or smi in seen_this_run:
                stats["skipped"] += 1
                continue
            seen_this_run.add(smi)
            an, it = await loop.run_in_executor(None, analyze_one, smi)  # the CPU analysis goes to a thread
            await queue.put((it, an))                                    # a full queue throttles this naturally
        for _ in consumers:                                              # signal the consumers to stop
            await queue.put(_DONE)
        await asyncio.gather(*consumers)
    except (KeyboardInterrupt, asyncio.CancelledError):
        for c in consumers:
            c.cancel()
        out_fh.flush()
        _write_meta()
        pbar.close()
        print(f"\n# Interrupted -> progress saved to {output_path}. Re-run to resume.",
              file=sys.stderr)
        raise
    finally:
        out_fh.close()
        pbar.close()
    _write_meta()

    print(f"\n# Done. described={stats['described']} failed={stats['failed']} "
          f"no_scaffold={stats['no_scaffold']} verified={stats['verified']} "
          f"remaining_violations={stats['with_violations']} skipped(resume/dup)={stats['skipped']}")
    print(f"# Wrote: {output_path}\n# Meta : {meta_path}")
    # preview
    shown = 0
    if os.path.exists(output_path):
        with open(output_path) as fh:
            for line in fh:
                it = json.loads(line)
                if it.get("description"):
                    print(f"\n  SMILES   : {it['input_smiles']}")
                    print(f"  scaffold : {it['scaffold_smiles']}  (verified={it['scaffold_verified']})")
                    print(f"  desc     : {it['description']}")
                    shown += 1
                if shown >= 3:
                    break
    return 0 if stats["failed"] == 0 else 1


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Extract each molecule's Murcko scaffold, analyze/name it deterministically, "
                    "polish a controlled-NL description with vLLM (Qwen3), verify by RDKit "
                    "substructure check, and stream results to JSONL.")
    ap.add_argument("--input", required=True,
                    help="input (.parquet/.txt/.smi/.jsonl/.json).")
    ap.add_argument("--smiles-key", default=None,
                    help="the parquet column, or the SMILES key of a JSONL/JSON dict (default: smiles/ref_smiles/smi).")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_ROOT,
                    help=f"output root (default {DEFAULT_OUT_ROOT}); a new --run-name folder is created under it.")
    ap.add_argument("--run-name", default=None,
                    help="name of the output subfolder (default: 'scaffold_<input filename>').")
    ap.add_argument("--output", default=None,
                    help="set the output JSONL path directly; overrides --out-dir/--run-name.")
    ap.add_argument("--servers", default=DEFAULT_SERVERS,
                    help="vLLM servers as comma-separated 'host:portspec'. "
                         f"Default: {DEFAULT_SERVERS}")
    ap.add_argument("--host", default=None,
                    help="one host, or several comma-separated; replaces --servers with host x --base-ports.")
    ap.add_argument("--base-ports", default="8080-8087",
                    help="ports used for --host, and for --servers entries without a portspec (default 8080-8087)")
    ap.add_argument("--concurrency-per-server", type=int, default=5, help="concurrent requests per server (default 5)")
    ap.add_argument("--model", default=None, help="force the model name; otherwise it is auto-detected from /v1/models.")
    ap.add_argument("--reasoning-effort", default="low",
                    choices=["none", "low", "medium", "high"],
                    help="qwen3: none/low turns thinking off. This is a polish step, so low is recommended (default low)")
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--max-sentences", type=int, default=0, help="maximum sentences in the description (0 = no limit)")
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--max-retries", type=int, default=4)
    ap.add_argument("--chunk-size", type=int, default=2000,
                    help="(DEPRECATED, ignored) processing is now continuous through a dynamic "
                         "queue, with no chunk barrier. Use --concurrency-per-server for "
                         "concurrency.")
    ap.add_argument("--offset", type=int, default=0, help="skip the first N input rows (for sharding or resuming)")
    ap.add_argument("--limit", type=int, default=None, help="maximum molecules to process (default: all)")
    ap.add_argument("--validate-passes", type=int, default=3,
                    help="maximum repair attempts for validation violations (0 disables validation and repair; default 3)")
    ap.add_argument("--overwrite", action="store_true", default=False,
                    help="delete the existing output JSONL and start over (default: resume).")
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    try:
        raise SystemExit(asyncio.run(main_async(args)))
    except KeyboardInterrupt:
        print("\n# Interrupted. Progress saved; re-run the same command to resume.", file=sys.stderr)
        raise SystemExit(130)


if __name__ == "__main__":
    main()
