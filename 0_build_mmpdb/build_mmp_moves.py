#!/usr/bin/env python
"""Build a data-driven molecule-edit move set from a property-tagged pool via mmpdb.

Pipeline
--------
1. Sample SMILES + property values from a parquet pool (random across row groups).
2. Run ``mmpdb fragment`` + ``mmpdb index --properties`` to mine matched molecular
   pairs and their per-property change statistics. ``mmpdb index`` computes the
   environment fingerprint at EVERY radius 0..5 in a single pass, so all radii are
   available for free — no per-radius rebuild.
3. Extract the move sets and save them as JSON:
     (A) attach_library.json — single-attachment R-group fragments, ranked by how
         many distinct cores they decorate (also used as the novelty source-universe).
     (B) single_cut.json / double_cut.json / triple_cut.json — substituent swaps
         A->B for edit_fragment, one file per attachment count. Each row is a
         (from->to, radius, context) triple carrying the mean/std Δ for EVERY indexed
         property (physchem + ADMET) → multi-objective moves.

Radius / context
----------------
Each swap row carries its environment ``radius`` and ``context`` (the local
chemistry around the attachment point, as mmpdb's pseudo-SMILES + a matchable
SMARTS). Radius 0 is context-free — a single generic environment per cut count,
maximal support, maximal transferability. Higher radii fan out into specific
contexts: support drops (the pairs partition exactly across contexts) but the
within-context Δ sharpens. Empirically the ADMET Δ-std halves from r=0 to r=5
(context explains ~80% of the Δ variance; physchem is near-additive and collapses
to ~0 std by r=2). So a consumer can trade support for precision by picking the
most specific context radius that matches its site, falling back to radius 0.

Why mmpdb: an MMP transformation A->B is a substituent edit whose property change is
(for additive physchem) transferable across scaffolds. Each rule is property-agnostic
and stores a full Δ-vector, so one rule serves every property at once.

Usage
-----
    python 0_build_mmpdb/build_mmp_moves.py \
        --src data/pool/train_pool_props.parquet \
        --n-sample 50000 --out-dir data/mmp_moves

Requires mmpdb (``pip install mmpdb``) and RDKit in the active env.
NOTE: mmpdb's ``index`` crashes on molecules with directional (E/Z) double bonds
under RDKit >= 2026 (a fragment-canonicalisation precondition violation). We strip
stereochemistry in write_inputs() up front to avoid it; this is safe because
stereo-only swaps are discarded downstream anyway.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import multiprocessing as mp
import os
import re
import statistics
import subprocess
import sqlite3
import sys
import time
from collections import defaultdict

import pyarrow.parquet as pq
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

PHYS_DEFAULT = ["MW", "logP", "HBD", "HBA", "TPSA", "rotB", "rings_total",
                "QED", "MR", "heavy_atoms", "formal_charge"]
ADMET_DEFAULT = ["logD", "logS", "BBBP", "HIA", "Mutag"]
_DUM = re.compile(r"\[\d*\*(?::\d+)?\]")          # [*], [1*], [*:1] -> *


def _norm(smi: str) -> str:
    return _DUM.sub("*", smi)


def _dechiral(smi: str) -> str:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return smi
    Chem.RemoveStereochemistry(m)
    return Chem.MolToSmiles(m)


# ---------------------------------------------------------------------------
# 1. sample -> mmpdb input files
# ---------------------------------------------------------------------------
def write_inputs(src: str, n_sample: int, props: list[str], out_dir: str,
                 seed: int) -> tuple[str, str]:
    import random
    random.seed(seed)
    cols = ["smiles"] + props
    pf = pq.ParquetFile(src)
    ng = pf.num_row_groups
    groups = sorted(random.sample(range(ng), min(max(1, n_sample // 600), ng)))
    per = max(1, n_sample // max(1, len(groups)))
    rows = []
    for g in groups:
        d = pf.read_row_group(g, columns=cols).to_pydict()
        n = len(d["smiles"])
        for i in random.sample(range(n), min(per, n)):
            rows.append([d[c][i] for c in cols])
    random.shuffle(rows)
    rows = rows[:n_sample]

    smi_path = os.path.join(out_dir, "sample.smi")
    prop_path = os.path.join(out_dir, "sample.props")
    kept = 0
    seen: set[str] = set()
    with open(smi_path, "w") as sf, open(prop_path, "w") as pfh:
        pfh.write("ID\t" + "\t".join(props) + "\n")
        for idx, row in enumerate(rows):
            m = Chem.MolFromSmiles(row[0]) if row[0] else None
            if m is None:
                continue
            # Strip stereochemistry up front. mmpdb's `index` re-canonicalizes
            # fragment SMILES, and RDKit >= 2026 hits a precondition violation
            # ("neither end atom traversed", Canon.cpp) on fragments that carry
            # directional (/,\\) double bonds. Removing stereo eliminates those
            # bonds → no crash; harmless since we drop stereo-only swaps anyway.
            Chem.RemoveStereochemistry(m)
            can = Chem.MolToSmiles(m)
            # Deduplicate AFTER stereo-strip: distinct stereoisomers collapse to
            # the same constitutional SMILES. Keeping both as separate compounds
            # makes mmpdb pair a molecule with its own copy → "degenerate" rules
            # whose from/to fragments differ only in attachment labelling (Δ≈0).
            if can in seen:
                continue
            seen.add(can)
            cid = f"C{idx}"
            sf.write(f"{can}\t{cid}\n")
            vals = ["*" if v is None else f"{float(v):.4f}" for v in row[1:]]
            pfh.write(cid + "\t" + "\t".join(vals) + "\n")
            kept += 1
    print(f"[1/3] wrote {kept} unique molecules, stereo stripped + deduped (row-groups {groups})")
    return smi_path, prop_path


# ---------------------------------------------------------------------------
# 2. mmpdb fragment + index
# ---------------------------------------------------------------------------
def build_db(smi_path: str, prop_path: str, out_dir: str, num_jobs: int) -> str:
    frags = os.path.join(out_dir, "sample.fragments")
    db = os.path.join(out_dir, "sample.mmpdb")
    mmpdb = [sys.executable, "-m", "mmpdblib"]
    print("[2/3] mmpdb fragment ...")
    subprocess.run(mmpdb + ["fragment", smi_path, "--num-jobs", str(num_jobs),
                            "-o", frags], check=True)
    print("[2/3] mmpdb index ...")
    subprocess.run(mmpdb + ["index", frags, "--properties", prop_path, "-o", db],
                   check=True)
    return db


def build_db_sharded(smi_path: str, prop_path: str, out_dir: str, num_jobs: int,
                     n_shards: int, max_radius: int, max_parallel: int = 0) -> list[str]:
    """Parallel build: fragment once, partition by constant into ``n_shards`` fragdb
    files, then run ``index --properties`` on the shards, ``max_parallel`` at a time.

    Partitioning by constant is exact — a matched pair only forms between two
    fragmentations that share the same constant part, so ``fragdb_partition`` keeps
    every pair inside one shard and no MMP is lost. Each shard therefore computes
    its own per-(rule, environment) Δ-statistics independently and in parallel;
    ``extract_*_sharded()`` pools those per-shard (count, avg, std) back together
    (pooled variance) at extraction time. This keeps the property-statistics step —
    ~80% of the wall time of a single ``index --properties`` — OFF the critical
    path: it runs concurrently across shards instead of once serially. (``mmpdb
    merge`` cannot be used here because it drops properties by design.)

    ``n_shards`` controls partition granularity (more shards → the many small
    constants pack more evenly; a few ultra-common constants each still occupy
    one whole shard and set the wall-clock floor). ``max_parallel`` bounds how many
    ``index`` processes run at once (each is single-threaded + holds its shard in
    RAM), so keep it within cores AND memory. Default: min(n_shards, 32).

    Returns the list of shard mmpdb paths (``shard.NNNN.mmpdb`` in ``out_dir``).
    """
    mmpdb = [sys.executable, "-m", "mmpdblib"]
    frags = os.path.join(out_dir, "sample.fragments")
    print(f"[2/3] mmpdb fragment (num-jobs={num_jobs}) ...")
    subprocess.run(mmpdb + ["fragment", smi_path, "--num-jobs", str(num_jobs),
                            "-o", frags], check=True)

    # Clean any shard files left by a previous run so globs stay unambiguous.
    for stale in (glob.glob(os.path.join(out_dir, "shard.*.fragdb")) +
                  glob.glob(os.path.join(out_dir, "shard.*.mmpdb"))):
        os.remove(stale)
    tmpl = os.path.join(out_dir, "shard.{i:04}.fragdb")
    print(f"[2/3] mmpdb fragdb_partition -> {n_shards} shards ...")
    subprocess.run(mmpdb + ["fragdb_partition", frags, "-n", str(n_shards),
                            "--template", tmpl], check=True)
    parts = sorted(glob.glob(os.path.join(out_dir, "shard.*.fragdb")))
    if not parts:
        raise SystemExit("fragdb_partition produced no shards")

    cap = max_parallel if max_parallel and max_parallel > 0 else min(len(parts), 32)
    shard_dbs = [pf[:-len(".fragdb")] + ".mmpdb" for pf in parts]
    print(f"[2/3] mmpdb index --properties x{len(parts)} shards, {cap} at a time "
          f"(max-radius={max_radius}) ...")
    # Bounded pool: keep at most `cap` index processes alive at once.
    running: dict = {}          # Popen -> shard index
    pending = list(range(len(parts)))
    failed = done = 0
    while pending or running:
        while pending and len(running) < cap:
            i = pending.pop(0)
            running[subprocess.Popen(
                mmpdb + ["index", parts[i], "--properties", prop_path,
                         "--max-radius", str(max_radius), "-o", shard_dbs[i]])] = i
        time.sleep(0.5)
        for pr in [p for p in running if p.poll() is not None]:
            if pr.returncode != 0:
                failed += 1
            del running[pr]
            done += 1
            print(f"[2/3]   shard {done}/{len(parts)} done "
                  f"({len(running)} running, {len(pending)} queued)", flush=True)
    if failed:
        raise SystemExit(f"{failed}/{len(procs)} shard index jobs failed")
    return shard_dbs


# ---------------------------------------------------------------------------
# 3. extract move sets
# ---------------------------------------------------------------------------
def extract_attach_library(db: str, top_n: int) -> list[dict]:
    c = sqlite3.connect(db)
    rows = c.execute("""
        SELECT rs.smiles, rs.num_heavies, SUM(re.num_pairs), COUNT(DISTINCT r.id)
        FROM rule_smiles rs
        JOIN rule r ON r.to_smiles_id = rs.id
        JOIN rule_environment re ON re.rule_id = r.id AND re.radius = 0
        GROUP BY rs.id""").fetchall()
    lib = [{"fragment": _norm(s), "num_heavies": nh, "support": int(sup or 0), "contexts": ctx}
           for s, nh, sup, ctx in rows if s.count("*") == 1]
    lib.sort(key=lambda x: (-x["contexts"], -x["support"]))
    return lib[:top_n]


def extract_swap_rules(db: str, props: list[str], min_support: int,
                       max_radius: int, top_n: int) -> dict[int, list[dict]]:
    """Extract MMP swap rules across ALL environment radii 0..max_radius.

    Every matched pair of a transform contributes to its radius-0 environment, so
    we enumerate transforms by their radius-0 support (>= ``min_support``), then for
    each emit one row per (radius, context) whose support also clears ``min_support``.
    Radius 0 is the single context-free environment (max support); higher radii are
    its sub-contexts (support partitions exactly across them, Δ sharpens).

    Each row: ``{from, to, num_attachments, radius, context, context_smarts,
    support, delta}`` where ``delta[prop] = {avg, std}`` (no per-prop count — it is
    identical to ``support``). Returns ``{1: [...], 2: [...], 3: [...]}`` keyed by
    attachment count (single / double / triple cut).
    """
    c = sqlite3.connect(db)
    pid = {i: n for i, n in c.execute("SELECT id,name FROM property_name")}
    keep = ",".join(str(i) for i, n in pid.items() if n in props)

    # Enumerate transforms via their radius-0 environment (most general, one per
    # transform), ranked by support. Higher-radius rows are pulled per transform.
    base = c.execute("""
        SELECT re.id, re.rule_id, re.num_pairs, f.smiles, t.smiles
        FROM rule_environment re
        JOIN rule r ON r.id = re.rule_id
        JOIN rule_smiles f ON f.id = r.from_smiles_id
        JOIN rule_smiles t ON t.id = r.to_smiles_id
        WHERE re.radius = 0 AND re.num_pairs >= ?
        ORDER BY re.num_pairs DESC""", (min_support,)).fetchall()

    out: dict[int, list[dict]] = {1: [], 2: [], 3: []}
    n_kept = 0
    for r0_id, rule_id, _r0_pairs, fs, ts in base:
        if _dechiral(fs) == _dechiral(ts):
            continue  # drop stereo-only swaps
        n_att = fs.count("*")
        if n_att not in out:
            continue
        # Degenerate guard: a rule is meaningless if its matched pair links two
        # constitutionally identical molecules (Δ≈0) — they differ only in how the
        # attachment points were labelled. Dedup in write_inputs prevents these at
        # the source; this is a defensive secondary check on one radius-0 pair.
        pr = c.execute(
            "SELECT a.input_smiles,b.input_smiles FROM pair p "
            "JOIN compound a ON a.id=p.compound1_id JOIN compound b ON b.id=p.compound2_id "
            "WHERE p.rule_environment_id=? LIMIT 1", (r0_id,)).fetchone()
        if pr and _dechiral(pr[0]) == _dechiral(pr[1]):
            continue
        # Full radius ladder for this transform (indexed by rule_id → fast),
        # joined to its environment context (mmpdb pseudo-SMILES + matchable
        # SMARTS) and its per-property Δ, all in one query. Rows arrive grouped by
        # environment (radius, then support desc); we fold the property rows into
        # one row per environment.
        ladder = c.execute(f"""
            SELECT re.id, re.radius, re.num_pairs, ef.pseudosmiles, ef.smarts,
                   s.property_name_id, s.avg, s.std
            FROM rule_environment re
            JOIN environment_fingerprint ef ON ef.id = re.environment_fingerprint_id
            JOIN rule_environment_statistics s ON s.rule_environment_id = re.id
            WHERE re.rule_id = ? AND re.radius <= ? AND re.num_pairs >= ?
                  AND s.property_name_id IN ({keep})
            ORDER BY re.radius ASC, re.num_pairs DESC, re.id""",
            (rule_id, max_radius, min_support)).fetchall()
        cur_id = None
        row = None
        for reid, radius, rn, ctx, ctx_smarts, p_id, avg, std in ladder:
            if reid != cur_id:
                cur_id = reid
                # Keep attachment-point labels ([*:1],[*:2]) so edit_fragment can
                # map each point correctly — critical for double-cut rules.
                row = {"from": fs, "to": ts, "num_attachments": n_att,
                       "radius": radius, "context": ctx,
                       "context_smarts": ctx_smarts, "support": rn, "delta": {}}
                out[n_att].append(row)
            row["delta"][pid[p_id]] = {"avg": round(avg, 4), "std": round(std or 0, 4)}
        n_kept += 1
        if n_kept >= top_n:
            break
    return out


# ---------------------------------------------------------------------------
# 3b. sharded extraction — pool per-shard stats across constant-partitions
# ---------------------------------------------------------------------------
def _finalize_pooled(n: int, sum_x: float, sum_nx2: float, sum_ss: float
                     ) -> tuple[float, float]:
    """Pooled sample mean/std from per-shard accumulators, where each shard i
    contributed (count n_i, avg a_i, std s_i):
        sum_x   = Σ n_i·a_i
        sum_nx2 = Σ n_i·a_i²
        sum_ss  = Σ s_i²·(n_i-1)          (within-shard sum of squares, sample std)
    Exact reconstruction of the pooled sample variance over the union of pairs:
        Var = (SS_within + SS_between)/(N-1),  SS_between = Σ n_i(a_i-ā)² = sum_nx2 - N·ā².
    """
    if n <= 0:
        return 0.0, 0.0
    avg = sum_x / n
    if n == 1:
        return round(avg, 4), 0.0
    var = (sum_ss + (sum_nx2 - n * avg * avg)) / (n - 1)
    return round(avg, 4), round(math.sqrt(var) if var > 0 else 0.0, 4)


def extract_attach_library_sharded(shard_dbs: list[str], top_n: int) -> list[dict]:
    """Sharded twin of extract_attach_library: pool support + distinct-context count
    per single-attachment fragment across all constant-partitions (radius 0)."""
    support: dict[str, int] = defaultdict(int)
    froms: dict[str, set] = defaultdict(set)   # distinct source fragments = "contexts"
    heavies: dict[str, int] = {}
    for db in shard_dbs:
        c = sqlite3.connect(db)
        for to_smi, nh, from_smi, npairs in c.execute("""
                SELECT t.smiles, t.num_heavies, f.smiles, re.num_pairs
                FROM rule_environment re
                JOIN rule r ON r.id = re.rule_id
                JOIN rule_smiles t ON t.id = r.to_smiles_id
                JOIN rule_smiles f ON f.id = r.from_smiles_id
                WHERE re.radius = 0"""):
            if to_smi.count("*") != 1:
                continue
            support[to_smi] += npairs
            froms[to_smi].add(from_smi)
            heavies[to_smi] = nh
        c.close()
    lib = [{"fragment": _norm(s), "num_heavies": heavies[s],
            "support": int(support[s]), "contexts": len(froms[s])} for s in support]
    lib.sort(key=lambda x: (-x["contexts"], -x["support"]))
    return lib[:top_n]


# Worker state for the parallel sharded extraction (one set per worker process,
# opened once in the initializer so the per-transform tasks reuse it).
_W: dict = {}


def _swap_worker_init(shard_dbs, props, min_support, max_radius):
    conns = [sqlite3.connect(f"file:{db}?mode=ro", uri=True) for db in shard_dbs]
    _W["conns"] = conns
    _W["pid"] = [{i: n for i, n in c.execute("SELECT id,name FROM property_name")}
                 for c in conns]
    _W["keep"] = [",".join(str(i) for i, n in pm.items() if n in props) for pm in _W["pid"]]
    _W["sm2id"] = [dict(c.execute("SELECT smiles,id FROM rule_smiles")) for c in conns]
    _W["min_support"] = min_support
    _W["max_radius"] = max_radius


def _extract_one_transform(ft):
    """Pool one transform's (radius, context) Δ across all shards. Runs in a worker;
    returns (num_attachments, [row dicts]). Same pooling as the serial path."""
    fs, ts = ft
    n_att = fs.count("*")
    min_support, max_radius = _W["min_support"], _W["max_radius"]
    acc: dict = {}   # (radius, ctx_smarts) -> [support, pseudosmiles, {name:[N,Σx,Σnx²,ΣSS]}]
    for c, pm, keep, s2i in zip(_W["conns"], _W["pid"], _W["keep"], _W["sm2id"]):
        fid = s2i.get(fs); tid = s2i.get(ts)
        if fid is None or tid is None:
            continue
        rr = c.execute("SELECT id FROM rule WHERE from_smiles_id=? AND to_smiles_id=?",
                       (fid, tid)).fetchone()
        if not rr:
            continue
        seen = set()
        for reid, radius, npairs, ctx, ctx_sm, p_id, cnt, avg, std in c.execute(f"""
                SELECT re.id, re.radius, re.num_pairs, ef.pseudosmiles, ef.smarts,
                       s.property_name_id, s.count, s.avg, s.std
                FROM rule_environment re
                JOIN environment_fingerprint ef ON ef.id = re.environment_fingerprint_id
                JOIN rule_environment_statistics s ON s.rule_environment_id = re.id
                WHERE re.rule_id = ? AND re.radius <= ?
                      AND s.property_name_id IN ({keep})""", (rr[0], max_radius)):
            key = (radius, ctx_sm)
            e = acc.get(key)
            if e is None:
                e = acc[key] = [0, ctx, {}]
            if reid not in seen:
                seen.add(reid)
                e[0] += npairs
            name = pm.get(p_id)
            if name is None:
                continue
            pe = e[2].get(name)
            if pe is None:
                pe = e[2][name] = [0, 0.0, 0.0, 0.0]
            sd = std or 0.0
            pe[0] += cnt
            pe[1] += avg * cnt
            pe[2] += avg * avg * cnt
            pe[3] += sd * sd * (cnt - 1)
    rows = []
    for (radius, ctx_sm), (support, ctx, pacc) in sorted(
            acc.items(), key=lambda kv: (kv[0][0], -kv[1][0])):
        if support < min_support:
            continue
        delta = {}
        for name, (N, sx, snx2, sss) in pacc.items():
            avg, std = _finalize_pooled(N, sx, snx2, sss)
            delta[name] = {"avg": avg, "std": std}
        rows.append({"from": fs, "to": ts, "num_attachments": n_att,
                     "radius": radius, "context": ctx, "context_smarts": ctx_sm,
                     "support": support, "delta": delta})
    return n_att, rows


def extract_swap_rules_sharded(shard_dbs: list[str], props: list[str],
                               min_support: int, max_radius: int, top_n: int,
                               n_jobs: int = 16) -> dict[int, list[dict]]:
    """Sharded twin of extract_swap_rules, parallelised across transforms.

    Each shard indexed a disjoint set of constants, so a transform's pairs (and each
    (radius, context) bucket) may split across shards; we pool them EXACTLY: sum the
    support and combine per-property (count, avg, std) via pooled variance. The
    (radius, context_smarts) key is identical across shards (mmpdb's environment
    fingerprint is canonical). Same output as the serial path.

    The per-transform pooling is independent, so it is fanned out over ``n_jobs``
    worker processes — each opens its own read-only shard connections once — turning
    the ~O(transforms × shards) query load from serial hours into ~cores× less.
    """
    # 1. pooled radius-0 support per transform (one generic env per transform) →
    #    the transform set and its ranking. Cheap radius-0 scan in the main process.
    pooled0: dict[tuple, int] = defaultdict(int)
    for db in shard_dbs:
        c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        for npairs, fs, ts in c.execute("""
                SELECT re.num_pairs, f.smiles, t.smiles FROM rule_environment re
                JOIN rule r ON r.id = re.rule_id
                JOIN rule_smiles f ON f.id = r.from_smiles_id
                JOIN rule_smiles t ON t.id = r.to_smiles_id
                WHERE re.radius = 0"""):
            pooled0[(fs, ts)] += npairs
        c.close()
    cand = [(fs, ts) for (fs, ts), s in
            sorted(pooled0.items(), key=lambda kv: -kv[1])
            if s >= min_support and fs.count("*") in (1, 2, 3)
            and _dechiral(fs) != _dechiral(ts)][:top_n]
    n_jobs = max(1, min(n_jobs, len(cand)))
    print(f"    [shard-extract] {len(cand)} transforms (pooled r0 support>={min_support}), "
          f"pooling across {len(shard_dbs)} shards with {n_jobs} workers ...", flush=True)

    out: dict[int, list[dict]] = {1: [], 2: [], 3: []}
    if not cand:
        return out
    with mp.get_context("fork").Pool(
            n_jobs, initializer=_swap_worker_init,
            initargs=(shard_dbs, props, min_support, max_radius)) as pool:
        done = 0
        for n_att, rows in pool.imap(_extract_one_transform, cand, chunksize=64):
            out[n_att].extend(rows)
            done += 1
            if done % 5000 == 0:
                print(f"    [shard-extract] {done}/{len(cand)} transforms pooled", flush=True)
    return out


CUT_NAMES = {1: "single_cut", 2: "double_cut", 3: "triple_cut"}


def write_attach_csv(attach: list[dict], out_dir: str) -> str:
    attach_csv = os.path.join(out_dir, "attach_library.csv")
    with open(attach_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["fragment", "num_heavies", "support", "contexts"])
        for a in attach:
            w.writerow([a["fragment"], a["num_heavies"], a["support"], a["contexts"]])
    return attach_csv


def write_swap_csv(rules: list[dict], props: list[str], path: str) -> None:
    """Mirror one cut's swap rules as a flat CSV for easy inspection (no count
    column — it equals ``support``)."""
    cols = ["from", "to", "num_attachments", "radius", "context", "context_smarts",
            "support"]
    for p in props:
        cols += [f"{p}_avg", f"{p}_std"]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for s in rules:
            row = [s["from"], s["to"], s["num_attachments"], s["radius"],
                   s["context"], s["context_smarts"], s["support"]]
            for p in props:
                d = s["delta"].get(p)
                row += [d["avg"], d["std"]] if d else ["", ""]
            w.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="data/pool/train_pool_props.parquet")
    ap.add_argument("--out-dir", default="data/mmp_moves")
    ap.add_argument("--n-sample", type=int, default=1000000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-jobs", type=int, default=8,
                    help="parallel jobs for `mmpdb fragment`")
    ap.add_argument("--shards", type=int, default=1,
                    help="if >1, build via fragdb_partition + parallel `index "
                         "--properties` across this many constant-partitions, then "
                         "pool per-shard stats at extraction (parallelises the "
                         "property step; ~5x faster). 1 = single-DB path.")
    ap.add_argument("--max-parallel", type=int, default=0,
                    help="max concurrent shard index processes (each single-threaded "
                         "+ holds its shard in RAM). 0 = min(shards, 32).")
    ap.add_argument("--extract-jobs", type=int, default=16,
                    help="worker processes for the sharded swap extraction (each holds "
                         "all shard rule_smiles in RAM). Only used when --shards > 1.")
    ap.add_argument("--phys", nargs="+", default=PHYS_DEFAULT)
    ap.add_argument("--admet", nargs="+", default=ADMET_DEFAULT)
    ap.add_argument("--min-support", type=int, default=10)
    ap.add_argument("--max-radius", type=int, default=5,
                    help="include environment radii 0..MAX_RADIUS (mmpdb indexes up "
                         "to 5; 0 = context-free, higher = context-specific)")
    ap.add_argument("--top-attach", type=int, default=100000)
    ap.add_argument("--top-swap", type=int, default=500000,
                    help="max transforms (from->to) to keep, ranked by radius-0 "
                         "support; each keeps its full radius ladder")
    ap.add_argument("--skip-build", action="store_true",
                    help="reuse an existing sample.mmpdb in --out-dir")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    props = list(args.phys) + list(args.admet)
    sharded = args.shards > 1
    db = os.path.join(args.out_dir, "sample.mmpdb")
    shard_dbs: list[str] = []
    if not args.skip_build:
        smi, prop = write_inputs(args.src, args.n_sample, props, args.out_dir, args.seed)
        if sharded:
            shard_dbs = build_db_sharded(smi, prop, args.out_dir, args.num_jobs,
                                         args.shards, args.max_radius, args.max_parallel)
        else:
            db = build_db(smi, prop, args.out_dir, args.num_jobs)
    elif sharded:
        shard_dbs = sorted(glob.glob(os.path.join(args.out_dir, "shard.*.mmpdb")))
        if not shard_dbs:
            raise SystemExit("--skip-build --shards >1 but no shard.*.mmpdb in --out-dir")
        print(f"[3/3] reusing {len(shard_dbs)} existing shard DBs")

    print("[3/3] extracting move sets ..." + (f" (pooling {len(shard_dbs)} shards)"
                                              if sharded else ""))
    if sharded:
        attach = extract_attach_library_sharded(shard_dbs, args.top_attach)
    else:
        attach = extract_attach_library(db, args.top_attach)
    json.dump(attach, open(os.path.join(args.out_dir, "attach_library.json"), "w"), indent=1)
    write_attach_csv(attach, args.out_dir)
    print(f"  (A) attach_library.json/.csv : {len(attach)} fragments")
    if attach:
        print("      top attach:", [a["fragment"] for a in attach[:6]])

    if sharded:
        swaps = extract_swap_rules_sharded(shard_dbs, props, args.min_support,
                                           args.max_radius, args.top_swap,
                                           n_jobs=args.extract_jobs)
    else:
        swaps = extract_swap_rules(db, props, args.min_support, args.max_radius, args.top_swap)
    for n_att, name in CUT_NAMES.items():
        rules = swaps.get(n_att, [])
        json.dump(rules, open(os.path.join(args.out_dir, f"{name}.json"), "w"), indent=1)
        write_swap_csv(rules, props, os.path.join(args.out_dir, f"{name}.csv"))
        n_tf = len({(r["from"], r["to"]) for r in rules})
        print(f"  (B) {name}.json/.csv : {len(rules)} rows over {n_tf} transforms "
              f"(radii 0..{args.max_radius}, Δ for {len(props)} properties)")
        if rules:
            s = rules[0]
            dv = " ".join(f"{k}{v['avg']:+.2f}" for k, v in list(s["delta"].items())[:6])
            print(f"      example: {s['from']} >> {s['to']} (r={s['radius']}, n={s['support']})  {dv} ...")

    # ---------------------------------------------------------------------------
    # Build the suggest_edits fast-load cache so the served tool loads the move set
    # in <1s per worker instead of re-parsing this JSON on its first call.
    # ---------------------------------------------------------------------------
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    try:
        from molkit.utils.suggest_edits import build_cache
        for mc in (1, 2, 3):
            build_cache(args.out_dir, mc)
        print(f"  (C) suggest_edits cache : .suggest_cache_c1/2/3.pkl in {args.out_dir}")
    except Exception as e:  # noqa: BLE001
        print(f"  WARNING: could not build suggest_edits cache: {e}")


if __name__ == "__main__":
    main()
