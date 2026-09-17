"""The interpretable feature vectors the rule-selection model consumes.

Two vectors per decision:

* **global** `g in R^G` — the state: one measured value per property, whether that
  property carries a lower / upper bound, the signed distance to each bound, plus the
  aggregate state columns branch_features already emits (`st_*`, `set_*`, `h_*`).
* **rule** `r_i in R^R` per candidate — the move's predicted per-property delta
  (mean and std), the functional groups it creates and destroys, its structure, and
  the edit site, from the `c_*` / `r_*` / `site_*` / `fg_*` / `ctx_*` / `cand_*`
  columns.

Two things are built here rather than read from the dump, because branch_features
aggregates them away:

* per-property columns (`st_val__<p>`, `st_has_lo__<p>`, `st_has_hi__<p>`,
  `st_lo_z__<p>`, `st_hi_z__<p>`, `r_dmean__<p>`, `r_dstd__<p>`) come from the
  `props` / `targets` / `pred_delta` fields branch_tree now dumps;
* per-functional-group columns (`r_dfg__<name>`) come from the `fg_created` /
  `fg_destroyed` name lists.

**Masking.** A feature that is meaningless for this instance — the bound distance of
a property the instance does not constrain, a predicted delta for a property nobody
asked about — is marked in the mask so the context gating can send its attention
weight to zero. The mask is part of the data, not a model detail.

Normalisation is robust (median / IQR) and fitted on the training split only;
indicator columns (0/1) are left alone. The fitted spec is written next to the
tensors so evaluation and inference encode identically.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np

# The 14 properties instances are actually built from (5 integer + 9 continuous):
# 1_instance_gen/build_generation_instances.py INT_PROPS + CONT_PROPS.
PROPS = ["HBD", "HBA", "rotB", "rings_total", "heavy_atoms",
         "MW", "logP", "logD", "logS", "TPSA", "QED", "BBBP", "Mutag", "MR"]

# Columns in the dump that are name LISTS, not scalars: expanded per functional group.
LIST_COLS = ("fg_created", "fg_destroyed")
# Never features: identifiers, the label itself, bookkeeping.
DROP_COLS = {"rule", "smiles", "sat_ratio", "sat_any", "min_depth", "pred_delta",
             "features", "_delta"}


def fg_names(fallback: list | None = None) -> list:
    """The 61-pattern catalog the benchmark scorer uses, in its own order."""
    try:
        import importlib
        bf = importlib.import_module("3_toolchain_gen.branch_features")
        pats = bf._fg_patterns()
        names = list(pats.keys()) if isinstance(pats, dict) else [p[0] for p in pats]
        if names:
            return names
    except Exception:
        pass
    return fallback or []


@dataclass
class FeatureSpec:
    """Column names, which of them are conditional, and the fitted scaling."""
    global_names: list = field(default_factory=list)
    rule_names: list = field(default_factory=list)
    # column -> property it depends on; masked out when the instance omits it
    global_cond: dict = field(default_factory=dict)
    rule_cond: dict = field(default_factory=dict)
    fg: list = field(default_factory=list)
    props: list = field(default_factory=lambda: list(PROPS))
    center: dict = field(default_factory=dict)
    scale: dict = field(default_factory=dict)
    max_candidates: int = 4

    # -- construction ----------------------------------------------------
    @staticmethod
    def from_rows(rows: list, fg: list | None = None) -> "FeatureSpec":
        fg = fg if fg is not None else fg_names()
        spec = FeatureSpec(fg=list(fg))
        gnames, rnames = [], []
        gcond, rcond = {}, {}
        for p in spec.props:
            for stem in ("st_val", "st_has_lo", "st_has_hi", "st_lo_z", "st_hi_z"):
                name = f"{stem}__{p}"
                gnames.append(name)
                if stem != "st_val":
                    gcond[name] = p
            for stem in ("r_dmean", "r_dstd"):
                name = f"{stem}__{p}"
                rnames.append(name)
                rcond[name] = p
        for name in spec.fg:
            rnames.append(f"r_dfg__{name}")
        # the scalar columns the dump already carries
        gseen, rseen = set(gnames), set(rnames)
        for row in rows:
            for k, v in (row.get("global") or {}).items():
                if k not in gseen and _is_scalar(v):
                    gseen.add(k)
                    gnames.append(k)
            for c in row.get("candidates") or []:
                for k, v in (c.get("features") or {}).items():
                    if k in LIST_COLS or k in DROP_COLS or k in rseen:
                        continue
                    if _is_scalar(v):
                        rseen.add(k)
                        rnames.append(k)
        spec.global_names, spec.rule_names = gnames, rnames
        spec.global_cond, spec.rule_cond = gcond, rcond
        spec.max_candidates = max((len(r.get("candidates") or []) for r in rows),
                                  default=4)
        return spec

    # -- encoding --------------------------------------------------------
    def encode(self, row: dict):
        """-> (g, g_mask, R, r_mask, y, cand_mask) as float32 / bool arrays."""
        props = row.get("state_props") or {}
        targets = {t["property"]: t for t in (row.get("targets") or [])}
        g = np.zeros(len(self.global_names), dtype=np.float32)
        gm = np.ones(len(self.global_names), dtype=bool)
        gsrc = row.get("global") or {}
        for j, name in enumerate(self.global_names):
            stem, _, p = name.partition("__")
            if p and p in self.props:
                t = targets.get(p)
                v = props.get(p)
                lo = t.get("min") if t else None
                hi = t.get("max") if t else None
                sc = self.scale.get(f"prop::{p}", 1.0) or 1.0
                if stem == "st_val":
                    g[j] = 0.0 if v is None else float(v)
                    gm[j] = v is not None
                elif stem == "st_has_lo":
                    g[j] = 1.0 if lo is not None else 0.0
                    gm[j] = t is not None
                elif stem == "st_has_hi":
                    g[j] = 1.0 if hi is not None else 0.0
                    gm[j] = t is not None
                elif stem == "st_lo_z":
                    ok = v is not None and lo is not None
                    g[j] = (float(v) - float(lo)) / sc if ok else 0.0
                    gm[j] = ok
                elif stem == "st_hi_z":
                    ok = v is not None and hi is not None
                    g[j] = (float(hi) - float(v)) / sc if ok else 0.0
                    gm[j] = ok
            else:
                g[j] = _num(gsrc.get(name))
                gm[j] = name in gsrc

        cands = row.get("candidates") or []
        n = min(len(cands), self.max_candidates)
        R = np.zeros((self.max_candidates, len(self.rule_names)), dtype=np.float32)
        rm = np.zeros((self.max_candidates, len(self.rule_names)), dtype=bool)
        y = np.full(self.max_candidates, -1.0, dtype=np.float32)
        cm = np.zeros(self.max_candidates, dtype=bool)
        fgi = {name: k for k, name in enumerate(self.fg)}
        for i in range(n):
            c = cands[i]
            cm[i] = True
            y[i] = float(c.get("sat_ratio") or 0.0)
            feats = c.get("features") or {}
            delta = c.get("pred_delta") or {}
            created = _as_names(feats.get("fg_created"))
            destroyed = _as_names(feats.get("fg_destroyed"))
            dfg = np.zeros(len(self.fg), dtype=np.float32)
            for nm in created:
                if nm in fgi:
                    dfg[fgi[nm]] += 1.0
            for nm in destroyed:
                if nm in fgi:
                    dfg[fgi[nm]] -= 1.0
            for j, name in enumerate(self.rule_names):
                stem, _, p = name.partition("__")
                if stem in ("r_dmean", "r_dstd") and p in self.props:
                    d = delta.get(p) if isinstance(delta, dict) else None
                    if isinstance(d, dict):
                        v = d.get("mean" if stem == "r_dmean" else "std")
                    else:
                        v = d if stem == "r_dmean" else None
                    sc = self.scale.get(f"prop::{p}", 1.0) or 1.0
                    R[i, j] = 0.0 if v is None else float(v) / sc
                    # a delta for a property nobody constrains is noise here
                    rm[i, j] = v is not None and p in targets
                elif stem == "r_dfg":
                    k = fgi.get(name.split("__", 1)[1])
                    R[i, j] = dfg[k] if k is not None else 0.0
                    rm[i, j] = True
                else:
                    R[i, j] = _num(feats.get(name))
                    rm[i, j] = name in feats
        return g, gm, R, rm, y, cm

    # -- fitting ---------------------------------------------------------
    def fit(self, rows: list) -> "FeatureSpec":
        """Robust center/scale per column, plus a per-property scale for the z-columns.

        The property scale is the population IQR of the measured value, so `st_lo_z`
        is 'how many typical steps below the bound' rather than raw units — a logP
        unit and a TPSA unit are not comparable otherwise.
        """
        pv = {p: [] for p in self.props}
        for row in rows:
            for p, v in (row.get("state_props") or {}).items():
                if p in pv and v is not None:
                    pv[p].append(float(v))
        for p, vals in pv.items():
            if len(vals) > 20:
                q1, q3 = np.percentile(vals, [25, 75])
                self.scale[f"prop::{p}"] = float(max(q3 - q1, 1e-3))
            else:
                self.scale[f"prop::{p}"] = 1.0

        enc = [self.encode(r) for r in rows]          # once per row, not three times
        gs = np.stack([e[0] for e in enc])
        rs = np.concatenate([e[2][e[5]] for e in enc])
        del enc
        for names, mat, tag in ((self.global_names, gs, "g"), (self.rule_names, rs, "r")):
            for j, name in enumerate(names):
                col = mat[:, j]
                uniq = np.unique(col[np.isfinite(col)])
                if len(uniq) <= 2 and set(np.round(uniq, 6)).issubset({0.0, 1.0, -1.0}):
                    self.center[f"{tag}::{name}"], self.scale[f"{tag}::{name}"] = 0.0, 1.0
                    continue
                med = float(np.median(col))
                q1, q3 = np.percentile(col, [25, 75])
                self.center[f"{tag}::{name}"] = med
                self.scale[f"{tag}::{name}"] = float(max(q3 - q1, 1e-3))
        return self

    def norm_vectors(self):
        gc = np.array([self.center.get(f"g::{n}", 0.0) for n in self.global_names], np.float32)
        gsd = np.array([self.scale.get(f"g::{n}", 1.0) for n in self.global_names], np.float32)
        rc = np.array([self.center.get(f"r::{n}", 0.0) for n in self.rule_names], np.float32)
        rsd = np.array([self.scale.get(f"r::{n}", 1.0) for n in self.rule_names], np.float32)
        return gc, gsd, rc, rsd

    def cond_index(self):
        """Per column, the property index it depends on (-1 = unconditional)."""
        pi = {p: i for i, p in enumerate(self.props)}
        g = np.array([pi.get(self.global_cond.get(n, ""), -1) for n in self.global_names], np.int64)
        r = np.array([pi.get(self.rule_cond.get(n, ""), -1) for n in self.rule_names], np.int64)
        return g, r

    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump({"global_names": self.global_names, "rule_names": self.rule_names,
                       "global_cond": self.global_cond, "rule_cond": self.rule_cond,
                       "fg": self.fg, "props": self.props, "center": self.center,
                       "scale": self.scale, "max_candidates": self.max_candidates},
                      fh)

    @staticmethod
    def load(path: str) -> "FeatureSpec":
        with open(path) as fh:
            d = json.load(fh)
        return FeatureSpec(**d)


def _is_scalar(v) -> bool:
    return isinstance(v, (int, float, bool)) and not isinstance(v, bool) or isinstance(v, bool)


def _num(v) -> float:
    if v is None or isinstance(v, (list, dict, str)):
        return 0.0
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    return f if math.isfinite(f) else 0.0


def _as_names(v) -> list:
    if not v:
        return []
    if isinstance(v, str):
        return [v]
    out = []
    for x in v:
        out.append(x if isinstance(x, str) else (x.get("name") if isinstance(x, dict) else None))
    return [x for x in out if x]
