"""Decision states read out of the REAL SFT tool chains.

The rule-selection model was trained on exhaustive-search trees, where every node
carries all `top_k` children and a `sat_ratio` label per child. The corpus stage 4
actually renders is different: it is one BEAM-SEARCH PATH, so a node has the full
`suggest_edits` candidate list but only ONE of them was committed and no
counterfactual label exists for the others.

That is still exactly the input the reasoning text is written for — "why this rule
and not the other three" — so this module rebuilds the same row schema
`scripts/build_dataset.py` emits, straight from `toolchains_*chunk_*.jsonl`:

    {"id", "depth", "state_smiles", "state_props", "targets",
     "global": {st_*/set_*/h_*}, "candidates": [{"rule", "features", "pred_delta"}],
     "committed": <index of the rule the chain took>, ...}

with three differences forced by the data:

* **no `sat_ratio`** — `candidates[i]["sat_ratio"]` is absent. Nothing here can score
  a pick; the label is only ever "the search took this one".
* **`pred_delta` carries the std** — the tree dump kept only the mean, so the naive
  baseline lost its spread arguments. The tool chain records `{"avg","std"}` per
  property, which is what `fragment_names` wants.
* **`depth_left`** is measured against a fixed budget (`TREE_DEPTH`), not against the
  chain's own length: the chain's length is the future, and no feature may see it.

The features themselves are computed by the SAME `branch_features.candidate_features`
the training data used, so the encoder sees columns it recognises.
"""
from __future__ import annotations

import glob
import json
from typing import Optional
import os

# The depth budget the training trees were built with: st_depth_left is
# `tree_depth - depth` there, and the encoder's normalisation was fitted on that.
TREE_DEPTH = 5

_CHECKPOINT_TOOLS = ("analyze_properties",)


def _calls(entry: dict):
    """The tool calls of one chain entry: the main one plus its parallel siblings."""
    yield entry.get("tool_call") or {}, entry.get("expected_response")
    for c, r in zip(entry.get("parallel_tool_calls") or [],
                    entry.get("parallel_expected_responses") or []):
        yield c, r


def _as_props(resp) -> dict:
    if isinstance(resp, dict):
        return resp
    try:
        d = json.loads(resp)
        return d if isinstance(d, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def product_of(cur: str, cand: dict):
    """The molecule this candidate's edit would produce, or None.

    `candidates[*].smiles` is not decoration: a `--product-ctx` checkpoint reads it, and
    without it every candidate silently gets the zero vector that training only ever
    used for a molecule MolBERT refused. `search_plan._apply` makes the same
    `replace_fragment` call, pinned by the candidate's own anchors so the product is
    determined; the guard and visited-set filters it ALSO applies are not applied here.
    branch_tree drops a candidate whose product fails the guard, so a training row never
    held one — but these row builders keep the FULL suggest_edits response, and dropping
    would change the candidate set, `committed`, and the sibling features. Embedding the
    product the edit would actually make is the honest alternative, and the candidate's
    guard relationship is already in `site_on_guard` / `site_frac_guard`.
    """
    from rdkit import Chem
    from molkit.utils.molecule_edit_utils import replace_fragment
    try:
        anchors = {int(k): int(v) for k, v in (cand.get("anchors") or {}).items()}
        prods = replace_fragment(cur, cand.get("from_smiles"), cand.get("to_smiles"),
                                 anchors=anchors)
        if not prods:
            return None
        m = Chem.MolFromSmiles(prods[0])
        return None if m is None else Chem.MolToSmiles(m)
    except Exception:
        return None


def _as_cands(resp) -> list:
    if isinstance(resp, list):
        return resp
    try:
        d = json.loads(resp)
        return d if isinstance(d, list) else []
    except Exception:  # noqa: BLE001
        return []


def _match_committed(cands: list, args: dict) -> int:
    """Index of the candidate the chain actually committed, or -1.

    Matched on (from, to, anchors) first; the anchors are dropped on the second pass
    because a rule can be offered once and applied at a site the record spells
    slightly differently.
    """
    key = (args.get("from_smiles"), args.get("to_smiles"))
    anc = args.get("anchors")
    for i, c in enumerate(cands):
        if (c.get("from_smiles"), c.get("to_smiles")) == key and c.get("anchors") == anc:
            return i
    for i, c in enumerate(cands):
        if (c.get("from_smiles"), c.get("to_smiles")) == key:
            return i
    return -1


def build_row(cur: str, cur_props: dict, targets: dict, cands: list, *,
              guard_smarts=None, depth: int = 0, prev_delta=None, prev_props=None,
              prev_rule=None, max_cut: int = 3, tree_depth: int = TREE_DEPTH) -> dict:
    """One decision state -> the row the RuleSelector scores.

    Extracted from :func:`rows_from_record` so a LIVE search can build the same row it
    would have built from a recorded chain. Both callers share this code, so a feature
    added here cannot drift between the dataset and inference.

    ``cur`` / ``cur_props`` are the current molecule and its MEASURED properties,
    ``targets`` the property box as ``{name: [lo, hi]}``, ``cands`` the raw
    ``suggest_edits`` candidate list. The ``prev_*`` arguments carry the last committed
    edit (what it promised, the properties before it, and its rule), which is what the
    ``h_*`` history features read; leave them None at the seed.

    The returned dict has no ``committed`` key — that is the label, and a live search
    does not have one.
    """
    import importlib

    from molkit.utils.suggest_edits import _SCALE
    bf = importlib.import_module("3_toolchain_gen.branch_features")
    bt = importlib.import_module("3_toolchain_gen.branch_tree")

    node = {"smiles": cur, "props": cur_props, "depth": depth,
            "pred_delta": prev_delta, "parent_props": prev_props, "rule": prev_rule}
    state_cache: dict = {}
    here = []
    for rank, cand in enumerate(cands):
        feat = bf.candidate_features(cur, cur_props, targets, cand, rank,
                                     max(tree_depth - depth, 1), _SCALE,
                                     state_cache, max_cut=max_cut,
                                     guard_smarts=guard_smarts)
        here.append((None, feat, cand))
    bt._add_sibling_features(here, node)
    bt._add_history_features(here, node, targets, list(targets))

    # the same global / per-candidate split build_dataset.py makes
    splits = []
    for _p, feat, _c in here:
        g, c = {}, {}
        for k, v in feat.items():
            if k in ("rule", "_delta"):
                continue
            (g if k.startswith(("st_", "set_", "h_")) else c)[k] = v
        splits.append((g, c))
    glob_ = dict(splits[0][0])
    demoted = {k for k in glob_ if any(g.get(k) != glob_[k] for g, _ in splits[1:])}
    for k in demoted:
        glob_.pop(k, None)

    cand_rows = []
    for (_p, feat, cand), (gk, cf) in zip(here, splits):
        cf = dict(cf)
        for k in demoted:
            cf[k] = gk.get(k)
        delta = {p: {"mean": (v or {}).get("avg"), "std": (v or {}).get("std")}
                 for p, v in (cand.get("delta") or {}).items()}
        cand_rows.append({
            "rule": feat.get("rule"),
            "smiles": product_of(cur, cand),
            "from_smiles": cand.get("from_smiles"),
            "to_smiles": cand.get("to_smiles"),
            "anchors": cand.get("anchors"),
            "predicted_gap": cand.get("predicted_gap"),
            "pred_delta": delta, "features": cf})

    return {"depth": depth, "state_smiles": cur, "state_props": cur_props,
            "targets": [{"property": p, "min": v[0], "max": v[1]}
                        for p, v in targets.items()],
            "n_candidates": len(cand_rows), "global": glob_, "candidates": cand_rows}


def rows_from_record(rec: dict, max_cut: int = 3, tree_depth: int = TREE_DEPTH) -> list:
    """Every (suggest_edits -> edit_fragment) pair of one chain, as a decision state.

    One forward pass over the chain: the committed rule of round k is known before
    round k+1 is built, which is what the `h_*` history features need (what the last
    edit promised, and how far the measurement then landed from it).
    """
    import importlib

    from molkit.utils.suggest_edits import _SCALE
    bf = importlib.import_module("3_toolchain_gen.branch_features")
    bt = importlib.import_module("3_toolchain_gen.branch_tree")

    meta = rec.get("metadata") or {}
    chain = rec.get("tool_chain") or []
    props_by_smiles: dict = {}
    rows: list = []

    pending = None          # the row built at the last suggest_edits, awaiting its edit
    prev_props = None       # measured properties BEFORE the last committed edit
    prev_delta = None       # what that edit promised
    prev_rule = None
    step = 0

    for e in chain:
        for call, resp in _calls(e):
            if call.get("name") == "analyze_properties":
                smi = (call.get("arguments") or {}).get("mol_smiles")
                if smi:
                    props_by_smiles[smi] = _as_props(resp)

        name = (e.get("tool_call") or {}).get("name")

        if name == "suggest_edits":
            args = e.get("tool_call", {}).get("arguments") or {}
            cur = args.get("mol_smiles")
            constraints = args.get("constraints") or {}
            guard = args.get("scaffold_smarts")
            cands = _as_cands(e.get("expected_response"))
            cur_props = props_by_smiles.get(cur) or {}
            if not cur or not cands or not cur_props or not constraints:
                pending = None
                continue
            targets = {p: list(v) for p, v in constraints.items()}
            pending = build_row(cur, cur_props, targets, cands, guard_smarts=guard,
                                depth=step, prev_delta=prev_delta,
                                prev_props=prev_props, prev_rule=prev_rule,
                                max_cut=max_cut, tree_depth=tree_depth)
            pending.update({"source": "toolchain", "id": meta.get("task_id"),
                            "index": meta.get("instance_index"),
                            "chain_succeeded": bool(meta.get("all_constraints_satisfied")),
                            "n_edit_steps": meta.get("num_edit_steps")})
            continue

        if name == "edit_fragment" and pending is not None:
            eargs = e.get("tool_call", {}).get("arguments") or {}
            raw = [{"from_smiles": c["from_smiles"], "to_smiles": c["to_smiles"],
                    "anchors": c["anchors"]} for c in pending["candidates"]]
            j = _match_committed(raw, eargs)
            pending["committed"] = j
            pending["committed_rule"] = (
                f"{eargs.get('from_smiles')}->{eargs.get('to_smiles')}")
            pending["product_smiles"] = e.get("expected_response")
            rows.append(pending)
            if j >= 0:
                pd = pending["candidates"][j].get("pred_delta") or {}
                prev_delta = {k: (v or {}).get("mean") for k, v in pd.items()}
                prev_rule = pending["candidates"][j].get("rule")
            else:
                prev_delta, prev_rule = None, None
            prev_props = pending["state_props"]
            pending = None
            step += 1

    return rows


def iter_records(paths: list, limit: int = 0):
    n = 0
    for path in paths:
        with open(path) as fh:
            for line in fh:
                if not line.strip():
                    continue
                yield json.loads(line)
                n += 1
                if limit and n >= limit:
                    return


def expand(pattern: str) -> list:
    if os.path.isdir(pattern):
        return sorted(glob.glob(os.path.join(pattern, "toolchains_*chunk_*.jsonl")))
    return sorted(glob.glob(pattern))


# --------------------------------------------------------------------------- #
#  the rendered SFT corpus — where the shipped naive reasoning actually lives
# --------------------------------------------------------------------------- #
def _tool_responses(msgs: list, i: int) -> dict:
    """call-id -> response content, for the tool messages that answer message *i*."""
    calls = msgs[i].get("tool_calls") or []
    by_id, k = {}, 0
    for m in msgs[i + 1:]:
        if m.get("role") != "tool":
            break
        cid = m.get("tool_call_id")
        if cid is None and k < len(calls):
            cid = calls[k].get("id")
        by_id[cid] = m.get("content")
        k += 1
    return by_id


class SuggestRowBuilder:
    """The decision state behind ONE `suggest_edits`, built incrementally.

    `rows_from_sft_record` replays a finished record through this; the agentic rollout
    drives the same object live, so the row a rollout scores is the row training was
    built from -- by construction, not by a re-implementation that can drift.

    Feed it in conversation order: `analyze(smi, props)` for every measured molecule,
    `suggest(args, cands)` when the tool returns a candidate list (it returns the row),
    and `commit(args)` when `edit_fragment` picks one, which advances the depth and
    carries the history features (`prev_delta`, `prev_props`, `prev_rule`) into the
    next round.
    """

    def __init__(self, max_cut: int = 3, tree_depth: int = TREE_DEPTH,
                 group_id: str = "", user_prompt: str = ""):
        self.max_cut = max_cut
        self.tree_depth = tree_depth
        self.group_id = group_id
        self.user_prompt = user_prompt
        self.props_by_smiles: dict = {}
        self.step = 0
        self.prev_props = self.prev_delta = self.prev_rule = None
        self.pending = None

    def analyze(self, smi, body) -> None:
        if smi:
            self.props_by_smiles[smi] = _as_props(body)

    def suggest(self, args: dict, body) -> Optional[dict]:
        """-> the row, or None when the state is not scoreable (no props, no cands)."""
        import importlib
        from molkit.utils.suggest_edits import _SCALE
        bf = importlib.import_module("3_toolchain_gen.branch_features")
        bt = importlib.import_module("3_toolchain_gen.branch_tree")

        cur = args.get("mol_smiles")
        cands = _as_cands(body)
        cur_props = self.props_by_smiles.get(cur) or {}
        constraints = args.get("constraints") or {}
        if not cur or not cands or not cur_props or not constraints:
            self.pending = None
            return None
        targets = {p: list(v) for p, v in constraints.items()}
        node = {"smiles": cur, "props": cur_props, "depth": self.step,
                "pred_delta": self.prev_delta, "parent_props": self.prev_props,
                "rule": self.prev_rule}
        state_cache: dict = {}
        here = []
        for rank, cand in enumerate(cands):
            feat = bf.candidate_features(cur, cur_props, targets, cand, rank,
                                         max(self.tree_depth - self.step, 1), _SCALE,
                                         state_cache, max_cut=self.max_cut,
                                         guard_smarts=args.get("scaffold_smarts"))
            here.append((None, feat, cand))
        bt._add_sibling_features(here, node)
        bt._add_history_features(here, node, targets, list(targets))

        splits = []
        for _p, feat, _c in here:
            g, c = {}, {}
            for k, v in feat.items():
                if k in ("rule", "_delta"):
                    continue
                (g if k.startswith(("st_", "set_", "h_")) else c)[k] = v
            splits.append((g, c))
        glob_ = dict(splits[0][0])
        for k in [k for k in glob_
                  if any(g.get(k) != glob_[k] for g, _ in splits[1:])]:
            glob_.pop(k, None)

        cand_rows = []
        for (_p, feat, cand), (_gk, cf) in zip(here, splits):
            cand_rows.append({
                "rule": feat.get("rule"),
                "smiles": product_of(cur, cand),
                "from_smiles": cand.get("from_smiles"),
                "to_smiles": cand.get("to_smiles"),
                "anchors": cand.get("anchors"),
                "predicted_gap": cand.get("predicted_gap"),
                "pred_delta": {p: {"mean": (v or {}).get("avg"),
                                   "std": (v or {}).get("std")}
                               for p, v in (cand.get("delta") or {}).items()},
                "features": dict(cf)})

        self.pending = {"source": "sftdata", "id": self.group_id,
                        "user_prompt": self.user_prompt,
                        "depth": self.step, "state_smiles": cur,
                        "state_props": cur_props,
                        "targets": [{"property": p, "min": v[0], "max": v[1]}
                                    for p, v in targets.items()],
                        "n_candidates": len(cand_rows),
                        "global": glob_, "candidates": cand_rows,
                        "raw_candidates": cands,
                        "guard_smarts": args.get("scaffold_smarts")}
        return self.pending

    def commit(self, args: dict, body=None, corpus_text: str = "") -> Optional[dict]:
        """-> the completed row, or None when nothing was pending."""
        if self.pending is None:
            return None
        raw = [{"from_smiles": c["from_smiles"], "to_smiles": c["to_smiles"],
                "anchors": c["anchors"]} for c in self.pending["candidates"]]
        j = _match_committed(raw, args)
        row = self.pending
        row["committed"] = j
        row["committed_args"] = args
        row["committed_rule"] = f"{args.get('from_smiles')}->{args.get('to_smiles')}"
        row["product_smiles"] = body
        row["corpus_text"] = (corpus_text or "").strip()
        if j >= 0:
            pd = row["candidates"][j].get("pred_delta") or {}
            self.prev_delta = {k: (v or {}).get("mean") for k, v in pd.items()}
            self.prev_rule = row["candidates"][j].get("rule")
        else:
            self.prev_delta = self.prev_rule = None
        self.prev_props = row["state_props"]
        self.pending = None
        self.step += 1
        return row


def rows_from_sft_record(rec: dict, max_cut: int = 3,
                         tree_depth: int = TREE_DEPTH) -> list:
    """Decision states from a RENDERED sft record, carrying its own reasoning span.

    This is the corpus `4_sftdata_gen` writes, so the assistant turn that precedes the
    `edit_fragment` call is the reasoning the pipeline actually shipped — under
    `NAIVE_REASONING=1` it is the naive ablation, otherwise the computed-block one.
    Taking it verbatim removes any question of whether a re-implementation of that
    prompt is faithful.

    `fullctx` records carry the whole trajectory, so the tool responses (measured
    properties, the suggest_edits list) are all present and nothing has to be parsed
    out of a memory block.
    """
    msgs = rec.get("messages") or []
    meta = rec.get("metadata") or {}
    user_prompt = next((m.get("content") or "" for m in msgs
                        if m.get("role") == "user"), "")
    b = SuggestRowBuilder(max_cut=max_cut, tree_depth=tree_depth,
                          group_id=meta.get("group_id"), user_prompt=user_prompt)
    rows: list = []

    for i, m in enumerate(msgs):
        calls = m.get("tool_calls") or []
        if not calls:
            continue
        resp = _tool_responses(msgs, i)
        for tc in calls:
            fn = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"])
            except Exception:  # noqa: BLE001
                continue
            body = resp.get(tc.get("id"))
            if fn == "analyze_properties":
                b.analyze(args.get("mol_smiles"), body)
            elif fn == "suggest_edits":
                b.suggest(args, body)
            elif fn == "edit_fragment":
                row = b.commit(args, body, m.get("content") or "")
                if row is not None:
                    rows.append(row)
    return rows
