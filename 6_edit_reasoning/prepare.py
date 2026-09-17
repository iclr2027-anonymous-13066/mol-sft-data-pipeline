# -*- coding: utf-8 -*-
"""Freeze the rounds every arm will train on, index-balanced by selection.

One CPU pass over the two corpora. For each record it decides the split from the seed
molecule's Murcko scaffold, checks every decision round has exactly
`Recipe.require_candidates` candidates, and takes the record only while every committed
index its rounds contribute still has quota left. It writes:

    <work>/records/<split>/<corpus>-<shard>.jsonl   the record, VERBATIM — this is the
                                                    base every arm assembles from
    <work>/rounds/<split>/<corpus>-<shard>.jsonl    one line per decision round: what
                                                    the prompt builders need
    <work>/manifest.json                            counts, the recipe, its digest

**The `suggest_edits` response is never touched.** Balance comes from choosing records,
not from reordering candidates: in the untouched corpus, position and `predicted_gap`
rank are nearly the same variable, so an index quota kills both shortcuts at once
(measured: "always #1" 25.0%, `argmin(gap)` ties-first 25.0%, ties-last 21.5%). A record
is accepted whole or not at all — dropping one round out of a trajectory would break the
conversation, and training is on records.

A record is skipped when it has no edit round, when any round has a candidate count
other than 4, when the committed edit is not among the candidates, when a constrained
property has no measured value, or when its index counts no longer fit the quota. The
shipped reasoning span is carried in the round table for reference but never used: the
`original` arm is not part of this experiment, because those spans name the candidate by
position ("#3", "the first suggested edit") in 100% of fg rounds and 25% of scaffold
rounds and are therefore about a ranking the arms are not allowed to lean on.

Usage::

    PYTHONPATH=. python 6_edit_reasoning/prepare.py --procs 48
    PYTHONPATH=. python 6_edit_reasoning/prepare.py --procs 8 --scan-files 2 \
        --train-rounds 400 --dry-run
"""
from __future__ import annotations

# What the rule-selection FeatureSpec can encode; a round with more candidates
# than this would be silently truncated by `encode`, so it is rejected here
# instead. Kept as a literal rather than read from spec.json: `prepare` is the
# CPU pass and must not need the checkpoint to run.
_MAX_CANDS = 4

import argparse
import glob
import json
from dataclasses import replace
import multiprocessing as mp
import os
import sys
from collections import Counter, defaultdict

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import importlib                                                    # noqa: E402


def _sel_strip(tool_content: str) -> str:
    """`assemble.sel_strip`, imported lazily so this module keeps no import-time cycle."""
    import importlib
    return importlib.import_module(
        "6_edit_reasoning.assemble").sel_strip(tool_content)

rp = importlib.import_module("6_edit_reasoning.recipe")


# --------------------------------------------------------------------------- #
def scaffold_key(smiles: str) -> str:
    """Murcko scaffold, or the molecule itself when it has no ring system.

    The split key has to be chemistry, not an id: instance ids are not comparable
    across instance files, and `guard_smarts` is useless for the fg corpus (61 fixed
    patterns, so a guard-disjoint split would hold out whole functional-group classes).
    """
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    try:
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return smiles or "?"
        s = MurckoScaffold.GetScaffoldForMol(m)
        out = Chem.MolToSmiles(s) if s is not None else ""
        return out or Chem.MolToSmiles(m)
    except Exception:                                               # noqa: BLE001
        return smiles or "?"


def edit_rounds(msgs: list) -> list:
    """`(call_i, tool_i, span_i)` for every suggest_edits -> edit_fragment round."""
    out = []
    for i, m in enumerate(msgs):
        names = [c["function"]["name"] for c in (m.get("tool_calls") or [])]
        if "suggest_edits" not in names or i + 2 >= len(msgs):
            continue
        if msgs[i + 1].get("role") != "tool":
            continue
        nxt = msgs[i + 2]
        if nxt.get("role") != "assistant":
            continue
        if [c["function"]["name"] for c in (nxt.get("tool_calls") or [])] \
                != ["edit_fragment"]:
            continue
        out.append((i, i + 1, i + 2))
    return out


def _cand_key(d: dict) -> tuple:
    return (d.get("from_smiles"), d.get("to_smiles"),
            json.dumps(d.get("anchors"), sort_keys=True))


def prepare_record(rec: dict, corpus: str, recipe) -> tuple:
    """(record, [round dicts]) or (None, reason) if the record is skipped."""
    msgs = rec["messages"]
    rounds = edit_rounds(msgs)
    if len(rounds) < recipe.min_rounds_per_record:
        return None, "no-edit-round"

    gid = (rec.get("metadata") or {}).get("group_id") or ""
    state_props: dict = {}
    out_rounds = []

    # `analyze_properties` results, so a round can report the molecule it edits FROM.
    #
    # An assistant turn batches several calls and the tool replies follow it ONE PER
    # CALL, in order — so the reply to the k-th call is `msgs[i+1+k]`, not `msgs[i+1]`.
    # Reading `msgs[i+1]` for every call put a `match_substructure` body where the
    # properties belonged and rejected the whole corpus as `incomplete-state-props`.
    for i, m in enumerate(msgs):
        calls = m.get("tool_calls") or []
        for k, c in enumerate(calls):
            if c["function"]["name"] != "analyze_properties":
                continue
            if i + 1 + k >= len(msgs) or msgs[i + 1 + k].get("role") != "tool":
                continue
            try:
                a = json.loads(c["function"]["arguments"])
                body = json.loads(_sel_strip(msgs[i + 1 + k]["content"]))
            except Exception:                                       # noqa: BLE001
                continue
            if isinstance(body, dict) and a.get("mol_smiles"):
                state_props.setdefault(a["mol_smiles"], {}).update(
                    {kk: vv for kk, vv in body.items() if not isinstance(vv, dict)})

    for depth, (ci, ti, si) in enumerate(rounds):
        try:
            cands = json.loads(_sel_strip(msgs[ti]["content"]))
            sargs = json.loads(msgs[ci]["tool_calls"][
                [c["function"]["name"] for c in msgs[ci]["tool_calls"]]
                .index("suggest_edits")]["function"]["arguments"])
            eargs = json.loads(msgs[si]["tool_calls"][0]["function"]["arguments"])
        except Exception:                                           # noqa: BLE001
            return None, "unparsable-round"
        # `require_candidates == 0` accepts any count the checkpoint can encode.
        # `FeatureSpec.encode` clamps with `n = min(len(cands), max_candidates)` and
        # masks the rest, and `bucket_span.render` numbers exactly the candidates it is
        # given, so a two-candidate round renders a two-row table and commits to
        # `Taking Edit 1` or `2`. An EMPTY list is still a reject -- there is no decision.
        rc = int(recipe.require_candidates)
        if not isinstance(cands, list) or not cands or len(cands) > _MAX_CANDS:
            return None, f"n_candidates!={rc or f'1..{_MAX_CANDS}'}"
        if rc and len(cands) != rc:
            return None, f"n_candidates!={rc}"
        want = _cand_key(eargs)
        committed = next((k for k, c in enumerate(cands)
                          if _cand_key(c) == want), -1)
        if committed < 0:
            return None, "committed-not-in-candidates"

        # NO permutation and no rewrite of msgs[ti]: the tool's response stands as
        # returned. `committed` is therefore the candidate's natural rank.
        mol = sargs.get("mol_smiles") or eargs.get("mol_smiles")
        targets = {k: list(v) for k, v in (sargs.get("constraints") or {}).items()}
        out_rounds.append({
            "corpus": corpus, "group_id": gid, "depth": depth,
            "msg_span": si, "msg_tool": ti,
            "smiles": mol, "targets": targets,
            "props": state_props.get(mol) or {},
            "candidates": cands, "committed": eargs, "picked": committed,
            "guard_smarts": sargs.get("scaffold_smarts"),
            "user_prompt": next((m.get("content") for m in msgs
                                 if m.get("role") == "user"), "") or "",
            "shipped_span": (msgs[si].get("content") or ""),
        })

    if any(not r["props"] or any(p not in r["props"] for p in r["targets"])
           for r in out_rounds):
        return None, "incomplete-state-props"
    return rec, out_rounds


# --------------------------------------------------------------------------- #
def _worker(job):
    path, corpus, recipe, out_dir, shard, budget = job
    import json as _json
    SPL = ("train", "val", "test")
    keep = {s: [] for s in SPL}
    rounds = {s: [] for s in SPL}
    # remaining rounds allowed per committed index, per split
    left = {s: rp.quota(budget[s], recipe) for s in SPL}
    skip = Counter()
    # Read once per shard, not per record: a 51,647-line file is 2 MB and the set is
    # consulted only for records that are not `normal`.
    _dropped_ids = rp.recovered_ids(corpus)
    seen = 0
    with open(path) as fh:
        for line in fh:
            seen += 1
            try:
                rec = _json.loads(line)
            except Exception:                                       # noqa: BLE001
                skip["bad-json"] += 1
                continue
            # A seed stub goes only when a WHOLE TRAJECTORY for the same instance now
            # exists -- see `recipe.DROP_RECOVERED`. `min_rounds_per_record=0` is on
            # for these recipes so a record that satisfies its box AT THE SEED is kept,
            # and that same setting would otherwise let every failed stub in behind it:
            # both have zero rounds, so `ends_with_answer` is what separates them and
            # the recovered-id list is what says which stubs have a replacement.
            if _dropped_ids and \
                    not (rec.get("metadata") or {}).get("ends_with_answer") \
                    and ((rec.get("metadata") or {}).get("group_id") or "") in _dropped_ids:
                skip["recovered-elsewhere"] += 1
                continue
            mol = (rec.get("metadata") or {}).get("molecule") or ""
            sk = scaffold_key(mol)
            sp = rp.split_of(sk, recipe)
            if not any(left[sp]):
                continue
            rec2, res = prepare_record(rec, corpus, recipe)
            if rec2 is None:
                skip[res] += 1
                continue
            counts = [0] * recipe.n_positions
            for r in res:
                counts[r["picked"]] += 1
            if not rp.fits(counts, left[sp]):
                skip["index-quota-full"] += 1
                continue
            for i, c in enumerate(counts):
                left[sp][i] -= c
            rec2.setdefault("metadata", {})["arms_split"] = sp
            rec2["metadata"]["arms_scaffold"] = sk
            keep[sp].append(rec2)
            rounds[sp].append(res)
            if not any(any(left[s]) for s in SPL):
                break
    written = {}
    for sp in ("train", "val", "test"):
        if not keep[sp]:
            continue
        rd = os.path.join(out_dir, "records", sp)
        rr = os.path.join(out_dir, "rounds", sp)
        os.makedirs(rd, exist_ok=True)
        os.makedirs(rr, exist_ok=True)
        with open(os.path.join(rd, f"{corpus}-{shard:05d}.jsonl"), "w") as fh:
            for r in keep[sp]:
                fh.write(_json.dumps(r, ensure_ascii=False) + "\n")
        with open(os.path.join(rr, f"{corpus}-{shard:05d}.jsonl"), "w") as fh:
            for rs_ in rounds[sp]:
                for r in rs_:
                    fh.write(_json.dumps(r, ensure_ascii=False) + "\n")
        written[sp] = (len(keep[sp]), sum(len(v) for v in rounds[sp]))
    return corpus, shard, seen, written, dict(skip)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--procs", type=int, default=48)
    ap.add_argument("--scan-files", type=int, default=0,
                    help="use only the first N chunk files per corpus (0 = all). The "
                         "budget usually stops the scan long before the files run out.")
    ap.add_argument("--train-rounds", type=int, default=None)
    ap.add_argument("--val-rounds", type=int, default=None)
    ap.add_argument("--test-rounds", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    kw = {}
    for k in ("train_rounds", "val_rounds", "test_rounds"):
        v = getattr(args, k)
        if v is not None:
            kw[k] = v
    # `rp.Recipe(**kw)` builds from the DATACLASS DEFAULTS, not from the selected
    # recipe, so a `--train-rounds` on a full-corpus run would silently put back the
    # index quota and both corpora. Overrides are for the smoke test; a real run selects
    # its recipe with ARMS_RECIPE and passes none.
    recipe = replace(rp.DEFAULT, **kw) if kw else rp.DEFAULT
    out_dir = args.out or rp.work_dir(recipe)
    print(f"# recipe {recipe.version}\n# out {out_dir}", flush=True)
    if args.dry_run:
        print(recipe.to_json())
        return

    os.makedirs(out_dir, exist_ok=True)
    jobs = []
    for corpus in recipe.corpora:
        files = sorted(f for f in glob.glob(rp.CORPORA[corpus] + "/*.jsonl")
                       if not os.path.basename(f).startswith("_"))
        if args.scan_files:
            files = files[:args.scan_files]
        # The budget is per corpus, so each shard gets an equal slice of it — and so
        # does the per-index quota, which is what keeps the workers independent while
        # the totals still come out flat.
        n = max(len(files), 1)
        budget = {"train": -(-recipe.train_rounds // n),
                  "val": -(-recipe.val_rounds // n),
                  "test": -(-recipe.test_rounds // n)}
        for k, f in enumerate(files):
            jobs.append((f, corpus, recipe, out_dir, k, budget))
        print(f"# {corpus}: {len(files)} files, per-shard budget {budget}", flush=True)

    tot = defaultdict(lambda: [0, 0])
    skips = Counter()
    seen = 0
    with mp.Pool(args.procs) as pool:
        for corpus, shard, s, written, skip in pool.imap_unordered(_worker, jobs):
            seen += s
            skips.update(skip)
            for sp, (nr, nd) in written.items():
                tot[(corpus, sp)][0] += nr
                tot[(corpus, sp)][1] += nd
            done = sum(v[1] for k, v in tot.items() if k[1] == "train")
            print(f"  {corpus} shard {shard:>5}  records {s:>6}  "
                  f"train rounds so far {done:>7,}", flush=True)

    man = {"recipe": json.loads(recipe.to_json()), "version": recipe.version,
           "records_scanned": seen,
           "counts": {f"{c}/{s}": {"records": v[0], "rounds": v[1]}
                      for (c, s), v in sorted(tot.items())},
           "skipped": dict(skips.most_common())}
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(man, fh, indent=1)
    print("\n" + json.dumps(man["counts"], indent=1))
    print("\nskipped:", json.dumps(man["skipped"], indent=1))
    print(f"\n# wrote {out_dir}/manifest.json")


if __name__ == "__main__":
    main()
