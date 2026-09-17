#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Augment a per-row JSONL with functional-group constraints (1:1, order-preserving).

The functional-group counterpart of ``augment_jsonl_with_scaffold.py``. For each
input record it:

  1. reads the SMILES from --smiles-key (dotted paths allowed: ``meta_info.ref_smiles``),
  2. detects every scorer-visible functional group in it (``fg_analyzer.build_item``),
  3. takes the FG constraint either from the record's own ``answer.fragments`` or,
     when it has none, derives one from the molecule,
  4. attaches the authoritative scoring query and the natural-language description,
  5. writes ``{**original_record, **fg_fields}`` to the output JSONL.

**No GPU, no vLLM.** Descriptions come from ``fg_catalog.json``, which
``build_fg_catalog.py`` produced once for all 61 patterns. There is nothing
molecule-specific to generate, so a 2M-row run is pure CPU.

Handles both input shapes:

  molkit.jsonl  answer.fragments = [{"hydroxylamine": 1}], SMILES at
                        meta_info.ref_smiles (null on infeasible rows) -> constraint
                        is taken verbatim; the molecule is used only to verify it.
  generation_2m.jsonl   no FG constraint, SMILES at ref_smiles -> constraint is
                        derived from the molecule (rarest group by default).

Output is either a single file (--output) or a folder of fixed-size shards (--out-dir
+ --shard-size, default 10000/shard): ``<prefix>-00000.jsonl``, ``-00001.jsonl``, ...
Sharded writing streams a shard at a time (so memory is O(shard), not O(corpus)),
is atomic per shard, and is resumable — already-complete shards are skipped.

Usage:
  PY=python
  # 2M training set (derived constraints):
  $PY 2b_functional_group_gen/augment_jsonl_with_fg.py \
      --input   data/training_data/instances/generation_2m.jsonl \
      --out-dir data/training_data/instances/generation_2m_fg \
      --shard-size 10000 --smiles-key ref_smiles
  # benchmark (authored constraints):
  $PY 2b_functional_group_gen/augment_jsonl_with_fg.py \
      --input   data/training_data/instances/molkit.jsonl \
      --out-dir data/training_data/instances/benchmark_fg \
      --smiles-key meta_info.ref_smiles --fragments-key answer.fragments
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Iterator, Optional

from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from fg_analyzer import build_item  # noqa: E402
from fg_catalog import catalog_json_path  # noqa: E402
from fg_describer import (  # noqa: E402
    FGDescriber, render_constraint_parts, render_name_style,
)

DEFAULT_CATALOG = catalog_json_path()


# --------------------------------------------------------------------------- #
#  Per-worker state (loaded once per process, not once per row)
# --------------------------------------------------------------------------- #
_CAT: Optional[dict] = None
_DESCR: Optional[dict] = None
_OPTS: dict = {}


def _init_worker(catalog_path: str, opts: dict) -> None:
    global _CAT, _DESCR, _OPTS
    _OPTS = opts
    try:
        with open(catalog_path) as fh:
            blob = json.load(fh)
        _CAT = blob.get("entries") or {}
    except OSError:
        _CAT = {}
    # {fr_key: definition}. Count-free by design — the requirement clause is generated
    # per row from (count, match_mode) so it always agrees with eval_query.
    # {fr_key: [wordings]}. With --n-variants>1 the catalog holds a paraphrase bank;
    # the row picks one, keyed on itself so a resumed shard reproduces its choice.
    _DESCR = {k: (e.get("description_variants")
                  or ([e["description"]] if e.get("description") else []))
              for k, e in _CAT.items()}


def _dotted(obj: Any, path: str) -> Any:
    """``meta_info.ref_smiles`` -> obj['meta_info']['ref_smiles'] (None if absent)."""
    cur = obj
    for part in (path or "").split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
        if cur is None:
            return None
    return cur


def process_row(line: str) -> Optional[str]:
    """One input line -> one output line, or None when the row is dropped."""
    rec = json.loads(line)
    smi = _dotted(rec, _OPTS["smiles_key"]) or ""
    frags = _dotted(rec, _OPTS["fragments_key"]) if _OPTS.get("fragments_key") else None
    task_type = rec.get("task_type") or _OPTS["default_task_type"]

    _, item = build_item(
        str(smi), task_type=task_type, fragments=frags,
        fg_source=_OPTS["fg_source"], select=_OPTS["select"],
        n_fg_min=_OPTS["n_fg_min"], n_fg_max=_OPTS["n_fg_max"],
        allow_descriptors=_OPTS["allow_descriptors"],
        match_mode=_OPTS["match_mode"], require_count=_OPTS["require_count"],
        seed=_OPTS["seed"])

    members = item.get("fg_constraint") or []
    # One draw per row decides both the definition wording and the name-style opener.
    seed_text = f'{_OPTS["seed"]}:{item.get("smiles", "")}'
    draw = int.from_bytes(hashlib.blake2b(seed_text.encode(), digest_size=8).digest(), "big")
    picked = {}
    for m in members:
        fr = m.get("fr_key")
        bank = (_DESCR or {}).get(fr) or []
        if bank:
            # Offset per member so a two-group row does not take variant 0 for both.
            picked[fr] = bank[(draw + len(picked)) % len(bank)]
    parts = render_constraint_parts(members, _CAT or {}, picked)
    definition = " ".join(t for _, t in parts).strip()
    name_style = render_name_style(members, opener=draw % 997)

    # Both phrasings are kept. The definition spells the group out atom by atom,
    # which is what makes the brief verifiable; the name style is how a chemist
    # actually asks, and a model trained on definitions alone never sees it.
    item["description_definition"] = definition
    item["description_name"] = name_style
    style = _OPTS["description_style"]
    if style == "mixed":
        h = hashlib.blake2b(f'{_OPTS["seed"]}:style:{item.get("smiles", "")}'.encode(),
                            digest_size=4).digest()
        style = "name" if int.from_bytes(h, "big") % 2 else "definition"
    item["description_style"] = style
    item["description"] = name_style if style == "name" else definition

    # Faithfulness check — each member's own sentence against its own facts.
    d = FGDescriber()
    viol: list[str] = []
    for m, text in parts:
        entry = (_CAT or {}).get(m.get("fr_key") or "")
        if not entry:
            continue
        viol += d.validate(text, entry, m.get("count", 1), m.get("match_mode", "exact"))
    if viol:
        item["description_violations"] = viol

    # A row with no scorer-visible group cannot state a constraint, and the scorer
    # would mark it structure_success=False forever — indistinguishable from a
    # genuine miss. Drop it here rather than ship an instance nothing can satisfy.
    if _OPTS["drop_empty_fg"] and not members:
        return None

    out = {k: v for k, v in rec.items()}
    # Property targets live at the top level in generation_2m but under `answer` in
    # molkit. Lift them so both datasets present the same shape — the eval
    # harness and the toolchain builder both read instance["properties"], and a
    # missing key silently yields an instance with NO property constraints at all.
    if not out.get("properties"):
        lifted = _dotted(rec, _OPTS["properties_key"]) if _OPTS.get("properties_key") else None
        if isinstance(lifted, list) and lifted:
            out["properties"] = lifted
    out.update(item)
    return json.dumps(out, ensure_ascii=False)


# --------------------------------------------------------------------------- #
#  Sharded, resumable, streaming IO
# --------------------------------------------------------------------------- #
def count_lines(path: str) -> int:
    n = 0
    with open(path, "rb") as fh:
        for _ in fh:
            n += 1
    return n


def make_filter(keep_task_type: str, require_smiles: bool, smiles_key: str):
    """Row predicate, or None when nothing is filtered.

    Returning None matters: with no filter the row count is a cheap line count, but
    with one every line must be parsed twice (once to count, once to process), which
    on a 2M file is minutes rather than seconds.
    """
    keep = {t.strip() for t in (keep_task_type or "").split(",") if t.strip()}
    if not keep and not require_smiles:
        return None

    def _keep(line: str) -> bool:
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            return False
        if keep and (rec.get("task_type") or "") not in keep:
            return False
        if require_smiles and not _dotted(rec, smiles_key):
            return False
        return True

    return _keep


def iter_kept(fh, keep):
    for line in fh:
        if not line.strip():
            continue
        if keep is None or keep(line):
            yield line


def count_kept(path: str, keep) -> int:
    if keep is None:
        return count_lines(path)
    with open(path) as fh:
        return sum(1 for _ in iter_kept(fh, keep))


def _manifest_path(out_dir: str, prefix: str) -> str:
    return os.path.join(out_dir, f"{prefix}-manifest.json")


def write_manifest(out_dir: str, prefix: str, shards: int, consumed: int,
                   shard_size: int) -> None:
    """Record how much INPUT the written shards consumed.

    Rows can be dropped, so shard k no longer corresponds to input rows
    [k*size, (k+1)*size) and the shard index alone cannot tell a resumed run where to
    restart. The manifest carries that offset; it is rewritten after every shard, so
    a kill costs at most one shard of rework.
    """
    tmp = _manifest_path(out_dir, prefix) + ".tmp"
    with open(tmp, "w") as fh:
        json.dump({"shards": shards, "input_consumed": consumed,
                   "shard_size": shard_size}, fh)
    os.replace(tmp, _manifest_path(out_dir, prefix))


def read_manifest(out_dir: str, prefix: str, shard_size: int) -> tuple[int, int]:
    """(shards_written, input_rows_consumed), validated against the files on disk.

    A manifest that disagrees with the actual shards (killed mid-write, or files
    removed by hand) is discarded rather than trusted — restarting from scratch is
    cheaper than shipping a dataset with a hole in it.
    """
    try:
        with open(_manifest_path(out_dir, prefix)) as fh:
            man = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return 0, 0
    shards = int(man.get("shards", 0))
    if int(man.get("shard_size", -1)) != shard_size:
        return 0, 0                     # different --shard-size: indices would not line up
    for k in range(shards):
        p = os.path.join(out_dir, f"{prefix}-{k:05d}.jsonl")
        if not os.path.exists(p):
            return 0, 0
        # Every shard but the last must be full; a short one means a truncated write.
        if k < shards - 1 and count_lines(p) != shard_size:
            return 0, 0
    return shards, int(man.get("input_consumed", 0))


def scan_complete_shards(out_dir: str, prefix: str, shard_size: int, n: int) -> int:
    """Number of leading shards already written in full (resume point).

    A shard counts as complete only when its line count equals what it should hold,
    so a run killed mid-write is redone rather than silently accepted.
    """
    nshards = (n + shard_size - 1) // shard_size
    k = 0
    while k < nshards:
        p = os.path.join(out_dir, f"{prefix}-{k:05d}.jsonl")
        if not os.path.exists(p):
            break
        expected = min(shard_size, n - k * shard_size)
        try:
            c = count_lines(p)
        except OSError:
            break
        if c != expected:
            break
        k += 1
    return k


def batched(it: Iterator[str], size: int) -> Iterator[list[str]]:
    buf: list[str] = []
    for x in it:
        buf.append(x)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="input JSONL (1 record per line).")
    ap.add_argument("--output", default=None, help="single output JSONL path.")
    ap.add_argument("--out-dir", default=None,
                    help="sharded output folder; results are split --shard-size per file "
                         "as `<prefix>-00000.jsonl`, ... Input order preserved, resumable.")
    ap.add_argument("--shard-size", type=int, default=10000)
    ap.add_argument("--shard-prefix", default=None, help="default: input filename stem.")
    ap.add_argument("--smiles-key", default="ref_smiles",
                    help="SMILES key; dotted paths allowed (meta_info.ref_smiles).")
    ap.add_argument("--fragments-key", default="answer.fragments",
                    help="authored FG constraint key; dotted paths allowed. "
                         "Absent/empty -> the constraint is derived from the molecule.")
    ap.add_argument("--fg-source", default="auto", choices=("auto", "answer", "derive"),
                    help="auto: use answer.fragments when present, else derive.")
    ap.add_argument("--n-fg-min", type=int, default=1,
                    help="fewest groups in a DERIVED constraint (default 1).")
    ap.add_argument("--n-fg-max", type=int, default=2,
                    help="most groups in a DERIVED constraint (default 2). With "
                         "--select random the count is sampled in [min, max] per row.")
    ap.add_argument("--select", default="random",
                    choices=("random", "rarest", "common", "all"),
                    help="which groups a derived constraint takes (default: random).")
    ap.add_argument("--match-mode", default="min", choices=("min", "exact", "auto"),
                    help="scoring operator: min (>=, default), exact (==), or auto "
                         "(exact for generation / min otherwise, i.e. the current "
                         "evaluate_benchmark.py behaviour).")
    ap.add_argument("--require-count", type=int, default=1,
                    help="occurrences each DERIVED member must have (default 1: the "
                         "group merely has to be present). Use 0 to require the "
                         "reference molecule's own count instead.")
    ap.add_argument("--description-style", default="mixed",
                    choices=("mixed", "definition", "name"),
                    help="which phrasing goes into `description`: definition (spells "
                         "the group out atom by atom), name (just names it), or mixed "
                         "(default; chosen per row, deterministically). Both are always "
                         "written to description_definition / description_name.")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for --select random. Selection is keyed per row, so it "
                         "is stable across resumes and shard boundaries.")
    ap.add_argument("--keep-task-type", default="",
                    help="comma-separated task_type allowlist; other rows are dropped. "
                         "e.g. 'generation'. Empty = keep all.")
    ap.add_argument("--require-smiles", action="store_true",
                    help="drop rows whose SMILES key is missing/empty. On molkit "
                         "this is exactly the infeasible rows.")
    ap.add_argument("--properties-key", default="answer.properties",
                    help="where to find property targets when the record has no "
                         "top-level 'properties' (dotted path). molkit keeps "
                         "them under answer; generation_2m has them at the top level.")
    ap.add_argument("--keep-empty-fg", action="store_true",
                    help="keep rows whose molecule has no scorer-visible functional "
                         "group. Dropped by default: they carry an empty constraint "
                         "that no molecule can be scored against.")
    ap.add_argument("--allow-descriptors", action="store_true",
                    help="let metabolic-site descriptors (aryl methyl / allylic oxidation "
                         "sites) become a derived constraint. Off by default: they are "
                         "liability annotations, not groups a task brief can ask for.")
    ap.add_argument("--default-task-type", default="generation",
                    help="task_type for records that do not carry one. Decides whether the "
                         "count is scored exactly (generation) or as a minimum.")
    ap.add_argument("--catalog", default=DEFAULT_CATALOG,
                    help="fg_catalog.json from build_fg_catalog.py.")
    ap.add_argument("--procs", type=int, default=0, help="worker processes (0 = auto).")
    ap.add_argument("--chunksize", type=int, default=200, help="rows per worker dispatch.")
    args = ap.parse_args()

    if bool(args.output) == bool(args.out_dir):
        sys.exit("error: give exactly one of --output (single file) or --out-dir (sharded).")
    if args.out_dir and args.shard_size <= 0:
        sys.exit("error: --shard-size must be > 0.")
    if not os.path.exists(args.catalog):
        sys.exit(f"error: catalog not found: {args.catalog}\n"
                 f"       run build_fg_catalog.py first.")

    keep = make_filter(args.keep_task_type, args.require_smiles, args.smiles_key)
    n_total = count_lines(args.input)
    n = count_kept(args.input, keep)
    filt = "" if keep is None else f" [kept {n}/{n_total} after filter]"
    print(f"# {n_total} record(s) in {args.input} (smiles key = {args.smiles_key!r}, "
          f"fragments key = {args.fragments_key!r}){filt}.")
    if n == 0:
        sys.exit("error: filter kept 0 rows.")

    sharded = bool(args.out_dir)
    shard_size = args.shard_size if sharded else n
    if sharded:
        out_dir = os.path.abspath(args.out_dir)
        prefix = args.shard_prefix or os.path.splitext(os.path.basename(args.input))[0]
        nshards = (n + shard_size - 1) // shard_size
    else:
        out_dir = os.path.dirname(os.path.abspath(args.output)) or "."
        prefix, nshards = None, 0
    os.makedirs(out_dir, exist_ok=True)

    next_shard, resume_off = read_manifest(out_dir, prefix, shard_size) if sharded else (0, 0)
    resume_off = min(resume_off, n)
    if resume_off:
        print(f"# Resuming: {next_shard} shard(s) written, "
              f"{resume_off} input row(s) already consumed.")

    opts = {
        "smiles_key": args.smiles_key,
        "fragments_key": args.fragments_key,
        "properties_key": args.properties_key,
        "fg_source": args.fg_source,
        "select": args.select,
        "n_fg_min": args.n_fg_min,
        "n_fg_max": args.n_fg_max,
        "allow_descriptors": args.allow_descriptors,
        "drop_empty_fg": not args.keep_empty_fg,
        "match_mode": args.match_mode,
        "require_count": (None if args.require_count == 0 else args.require_count),
        "seed": args.seed,
        "description_style": args.description_style,
        "default_task_type": args.default_task_type,
    }
    n_procs = args.procs or min(32, os.cpu_count() or 8)
    stats = {"rows": 0, "parse_fail": 0, "no_fg": 0, "from_answer": 0, "derived": 0,
             "verified": 0, "unverified": 0, "unresolved": 0, "violations": 0,
             "dropped_empty_fg": 0}

    pbar = tqdm(total=n - resume_off, desc="FG", unit="row", smoothing=0.02)
    single_fh = open(args.output, "w") if not sharded else None

    with open(args.input) as fin, \
            ProcessPoolExecutor(max_workers=n_procs, initializer=_init_worker,
                                initargs=(args.catalog, opts)) as ex:
        kept_rows = iter_kept(fin, keep)
        for _ in range(resume_off):                      # skip already-consumed input
            next(kept_rows, None)

        shard_idx = next_shard
        consumed = resume_off
        buf: list[str] = []                              # survivors awaiting a full shard

        def _flush(final: bool = False) -> None:
            """Write whole shards out of *buf*; on *final*, write the remainder too.

            Shards are sized by OUTPUT rows, because dropped rows make the 1:1
            input/output alignment false. The manifest records how much input each
            shard consumed so a resumed run starts at the right input offset.
            """
            nonlocal shard_idx, buf
            while len(buf) >= shard_size or (final and buf):
                take, buf = buf[:shard_size], buf[shard_size:]
                pth = os.path.join(out_dir, f"{prefix}-{shard_idx:05d}.jsonl")
                with open(pth + ".tmp", "w") as fh:
                    fh.write("\n".join(take) + "\n")
                os.replace(pth + ".tmp", pth)
                shard_idx += 1
                write_manifest(out_dir, prefix, shard_idx, consumed, shard_size)
                pbar.write(f"# shard {shard_idx - 1:05d} written ({len(take)} rows) "
                           f"-> {os.path.basename(pth)}")
                if not buf:
                    break

        for batch in batched(kept_rows, max(shard_size, args.chunksize * n_procs)):
            lines = [ln for ln in ex.map(process_row, batch, chunksize=args.chunksize)
                     if ln is not None]
            consumed += len(batch)
            stats["dropped_empty_fg"] += len(batch) - len(lines)
            for ln in lines:
                obj = json.loads(ln)
                stats["rows"] += 1
                if not obj.get("parse_ok"):
                    stats["parse_fail"] += 1
                if not obj.get("has_fg"):
                    stats["no_fg"] += 1
                stats["from_answer" if obj.get("fg_source") == "answer" else "derived"] += 1
                v = obj.get("fg_verified")
                stats["verified" if v else "unverified"] += 1
                if obj.get("unresolved_fg"):
                    stats["unresolved"] += 1
                if obj.get("description_violations"):
                    stats["violations"] += 1
            if sharded:
                buf.extend(lines)
                _flush()
            else:
                if lines:
                    single_fh.write("\n".join(lines) + "\n")
            pbar.update(len(batch))
        if sharded:
            _flush(final=True)

    if single_fh:
        single_fh.close()
    pbar.close()

    written = 0
    if sharded:
        # Count what is actually on disk: rows can be dropped, so the shard count is
        # an output-side fact, not `ceil(n / shard_size)`.
        paths = sorted(glob.glob(os.path.join(out_dir, f"{prefix}-[0-9]*.jsonl")))
        for p in paths:
            written += count_lines(p)
        dest = (f"{out_dir}/{prefix}-NNNNN.jsonl  "
                f"({len(paths)} shard(s), {shard_size}/shard)")
    else:
        written = count_lines(args.output)
        dest = args.output

    print(f"\n# Done. rows_in={n} rows_out={written} (this run: rows={stats['rows']} "
          f"from_answer={stats['from_answer']} derived={stats['derived']} "
          f"verified={stats['verified']} unverified={stats['unverified']} "
          f"parse_fail={stats['parse_fail']} no_fg={stats['no_fg']} "
          f"unresolved_name={stats['unresolved']} description_violations={stats['violations']} "
          f"dropped_empty_fg={stats['dropped_empty_fg']})")
    print(f"# Wrote: {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
