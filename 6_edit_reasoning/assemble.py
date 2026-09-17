# -*- coding: utf-8 -*-
"""Write the per-arm training corpora: the prepared record with its spans swapped.

Input is the permuted records from `prepare.py` and the spans from `generate.py`.
Output is one directory per arm, in the training record format the trainer reads,
so training needs a config and no new loader.

    <DATA_ROOT>/<recipe>/<arm>/train/toolchains_arms_<shard>.jsonl
    <DATA_ROOT>/<recipe>/<arm>/val/...
    <DATA_ROOT>/<recipe>/<arm>/manifest.json

**Every arm gets the SAME records.** A round whose span failed to generate in ANY arm
disqualifies its record from EVERY arm — otherwise an arm whose generation was flakier
trains on a different (and possibly easier) subset, and the accuracy difference prices
the flakiness. The intersection is computed across the arms named in `--arms`, so
adding an arm later means re-assembling all of them.

`noreason` is not generated: it is the same records with the span set to empty, and it
is the floor the whole experiment rests on. Without it, "reasoning helps" is not a
claim this design can make.

The shipped-corpus arm is deliberately absent. Its spans name the candidate by POSITION
("#3", "the first suggested edit") in 100% of fg rounds and 25% of scaffold rounds, and
`prepare.py` moved the positions — the text would be false about the data it sits in.
`naive` is the same generator re-run on the permuted order and is the honest stand-in.

Usage::

    PYTHONPATH=. python 6_edit_reasoning/assemble.py \
        --arms naive allfeat model noreason --splits train val --procs 32
"""
from __future__ import annotations

import argparse
import glob
import importlib
import json
import multiprocessing as mp
import os
import sys
from collections import Counter

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

rp = importlib.import_module("6_edit_reasoning.recipe")
prep = importlib.import_module("6_edit_reasoning.prepare")

GENERATED = ("naive", "allfeat", "model")


def _spans_of(work: str, arm: str, split: str, base: str) -> dict:
    p = f"{work}/spans/{arm}/{split}/{base}"
    out = {}
    if not os.path.exists(p):
        return out
    with open(p) as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except Exception:                                       # noqa: BLE001
                continue
            if d.get("text"):
                out[(d["group_id"], d["depth"])] = d["text"]
    return out


# `<|sel|>` is registered on the tokenizer at training time and lands on id 151669 --
# the FIRST embedding row with no token mapped to it (`len(tok)` 151669 against
# `config.vocab_size` 151936). So it needs no `resize_token_embeddings`, which would
# change the checkpoint's shape and every consumer of it, and it carries no pretrained
# meaning the way a repurposed `<|object_ref_start|>` would.
#
# ONE TOKEN, NOT FOUR. The embedding at these positions is overwritten with
# `proj(u_hat[i])` before the forward, so the marker is a POSITION and nothing else --
# a per-candidate id would buy nothing. The index is carried by the `Edit i:` text,
# which is also the vocabulary the rendered table uses one turn later.
SEL_TPL = "Edit {i}: <|sel|>"


SEL_SPLIT = "\nEdit 1: "


def sel_strip(tool_content: str) -> str:
    """The `suggest_edits` result without the `Edit i: <|sel|>` block.

    EVERY READER OF THAT MESSAGE HAS TO GO THROUGH THIS. The block turns a message that
    used to be pure JSON into JSON-plus-text, and three places parse it with a bare
    `json.loads`: `evaluate.test_items` (the candidates the eval scores against) and two
    in `prepare.edit_rounds` (which finds the rounds at all). Each one raises
    `JSONDecodeError: Extra data` on a corpus carrying the block -- loudly, which is the
    good case, but it means the arm cannot be prepared or evaluated until they all strip.

    Splitting on the first `\nEdit 1: ` is safe because the JSON array is one line: the
    writers emit it with `json.dumps` and no indent, so no `\n` occurs inside it.
    """
    if "<|sel|>" not in tool_content:
        return tool_content
    return tool_content.split(SEL_SPLIT, 1)[0]


def _sel_block(tool_content: str) -> str:
    """The placeholder block appended to a `suggest_edits` result, one line per edit.

    WHY IT SITS AFTER THE WHOLE ARRAY AND NOT INSIDE IT. `u_hat` is the selector's
    per-candidate state AFTER its candidate-attention, so `u_hat[0]` already depends on
    candidates 1..3. A placeholder written between two candidates would hand the reader
    a vector about edits it has not been shown yet, which causal attention has no way to
    use. After the closing bracket every candidate has been read, so the vectors
    describe only what is already on the page.

    WHY THE INDEX IS SPELLED OUT. `Edit 3: <sel_3>` binds the vector to the row the
    rendered table calls `Edit 3`, so the correspondence survives `prepare.py` having
    permuted the candidate order, and a round with fewer than four candidates simply
    writes fewer lines.
    """
    try:
        n = len(json.loads(sel_strip(tool_content)))
    except Exception:                                               # noqa: BLE001
        return ""
    if not n:
        return ""
    return "\n" + "\n".join(SEL_TPL.format(i=i + 1) for i in range(n))


def _worker(job):
    work, out_root, arms_, split, rec_path, strict, sel, suffix = job
    base = os.path.basename(rec_path)
    # The canonical three are ALWAYS loaded, whatever was asked for, because their
    # intersection is what fixes the record set -- and that set has to stay byte
    # identical to the one every arm already trained on. A newly added arm (a new span
    # style under its own spans/<name>) is loaded on top and must COVER that set; it
    # cannot enlarge it. Getting this wrong is silent: an arm assembled without the
    # intersection simply trains on more records than the arm it is compared against.
    extra = [a for a in arms_ if a != "noreason" and a not in GENERATED]
    spans = {a: _spans_of(work, a, split, base) for a in list(GENERATED) + extra}
    # INTERSECT OVER THE ARMS THAT ACTUALLY HAVE SPANS, and an arm with none is skipped
    # rather than zeroing the set. The canonical three fix the record set for the STUDY,
    # where all three exist; a standalone corpus (ARMS_RECIPE=main_*, one arm, no naive
    # or allfeat ever rendered) has none of them, and the old form turned that into
    # `set() & keys` -- an empty coverage that dropped every record. `--no-strict` was
    # the workaround, and it removed the gate this function needs for a different
    # reason: a round can legitimately have NO span. `base_dump` refuses to score a
    # round with fewer than two live candidates, so a one-candidate round has no q, no
    # base_cols row and no rendered span -- 1,423 of main_fg's 1,498,121. Without the
    # gate, the first such record reached `spans[a][(gid, d)]` and raised KeyError,
    # which is how the fg assemble died after every earlier stage had succeeded.
    #
    # A record is taken whole or not at all, so one uncovered round drops its record and
    # `dropped-record` counts it. That is the same conclusion `base_dump` already came
    # to: a single candidate is not a selection, and there is nothing for this arm's
    # span to decide.
    covered = None
    for a in list(GENERATED) + extra:
        if not strict:
            break
        s = spans.get(a) or {}
        if not s:
            continue
        covered = set(s) if covered is None else (covered & set(s))
    counts = Counter()
    fhs = {}
    try:
        for a in arms_:
            d = os.path.join(out_root, a + suffix, split)
            os.makedirs(d, exist_ok=True)
            fhs[a] = open(os.path.join(d, f"toolchains_arms_{base}"), "w")
        with open(rec_path) as fh:
            for line in fh:
                rec = json.loads(line)
                gid = (rec.get("metadata") or {}).get("group_id") or ""
                msgs = rec["messages"]
                rounds = prep.edit_rounds(msgs)
                keys = [(gid, d) for d in range(len(rounds))]
                if covered is not None and any(k not in covered for k in keys):
                    counts["dropped-record"] += 1
                    continue
                counts["records"] += 1
                counts["rounds"] += len(rounds)
                if sel:
                    # Appended ONCE per record, before any arm writes it: the block is a
                    # property of the PROMPT, not of the reasoning style, so every arm
                    # assembled in this call sees the identical tool turn.
                    for _d, (_ci, _ti, _si) in enumerate(rounds):
                        c = msgs[_ti].get("content") or ""
                        if "<|sel|>" not in c:
                            msgs[_ti]["content"] = c + _sel_block(c)
                for a in arms_:
                    for d, (_ci, _ti, si) in enumerate(rounds):
                        if a == "noreason":
                            msgs[si]["content"] = ""
                        else:
                            msgs[si]["content"] = spans[a][(gid, d)]
                    rec["metadata"]["arms_arm"] = a
                    fhs[a].write(json.dumps(rec, ensure_ascii=False) + "\n")
    finally:
        for f in fhs.values():
            f.close()
    return base, dict(counts)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--arms", nargs="+",
                    default=["naive", "allfeat", "model", "noreason"])
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--procs", type=int, default=32)
    ap.add_argument("--no-strict", action="store_true",
                    help="do NOT intersect coverage across arms. Only for a single-arm "
                         "smoke test — it makes the arms incomparable.")
    ap.add_argument("--sel-tokens", action="store_true",
                    help="append `Edit i: <sel_i>` to every suggest_edits result -- the "
                         "placeholders a soft-token arm substitutes `u_hat` into")
    ap.add_argument("--suffix", default="",
                    help="write to <arm><suffix>/ instead of <arm>/, so an arm can "
                         "reuse another's spans with a different PROMPT")
    ap.add_argument("--arrow", action="store_true",
                    help="also build the _records.arrow sidecar the loader mmaps")
    args = ap.parse_args(argv)

    work = args.work or rp.work_dir(rp.DEFAULT)
    out_root = args.out or f"{rp.DATA_ROOT}/{rp.DEFAULT.version}"
    print(f"# work {work}\n# out  {out_root}\n# arms {args.arms}", flush=True)

    hashes = {}
    for a in args.arms:
        p = f"{work}/spans/{a}/prompt_hash.json"
        if os.path.exists(p):
            hashes[a] = json.load(open(p))

    totals = {}
    for split in args.splits:
        recs = sorted(glob.glob(f"{work}/records/{split}/*.jsonl"))
        jobs = [(work, out_root, args.arms, split, p, not args.no_strict,
                 args.sel_tokens, args.suffix) for p in recs]
        tot = Counter()
        with mp.Pool(args.procs) as pool:
            for base, c in pool.imap_unordered(_worker, jobs):
                tot.update(c)
        totals[split] = dict(tot)
        print(f"# {split}: {tot['records']:,} records / {tot['rounds']:,} rounds "
              f"kept, {tot['dropped-record']:,} records dropped for missing spans",
              flush=True)

    man = {"recipe": rp.DEFAULT.version, "arms": args.arms,
           "prompt_hashes": {a: h.get("prompt_hash") for a, h in hashes.items()},
           "gen": {a: {k: h.get(k) for k in ("model", "temperature")}
                   for a, h in hashes.items()},
           "counts": totals, "strict_intersection": not args.no_strict,
           "work_dir": work}
    for a in args.arms:
        os.makedirs(os.path.join(out_root, a + args.suffix), exist_ok=True)
        with open(os.path.join(out_root, a + args.suffix, "manifest.json"), "w") as fh:
            json.dump({**man, "arm": a + args.suffix, "spans_from": a,
                       "sel_tokens": args.sel_tokens,
                       "prompt": hashes.get(a)}, fh, indent=1)
    print("\n" + json.dumps({k: v for k, v in man.items() if k != "gen"}, indent=1))

    if args.arrow:
        import subprocess
        for a in args.arms:
            for split in args.splits:
                d = os.path.join(out_root, a + args.suffix, split)
                if not glob.glob(d + "/*.jsonl"):
                    continue
                subprocess.run([sys.executable,
                                os.path.join(_ROOT, "SFT", "5_train_sft", "scripts",
                                             "jsonl_to_arrow.py"), d,
                                "--num-proc", str(args.procs)], check=False)


if __name__ == "__main__":
    main()
