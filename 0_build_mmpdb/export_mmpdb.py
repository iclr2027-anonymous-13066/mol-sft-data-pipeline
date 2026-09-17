#!/usr/bin/env python
"""Export an mmpdb SQLite database to human-readable CSV/JSON.

Writes, next to the .mmpdb (or to --out-dir):
  rules.csv      — one row per (transformation rule x environment): from/to fragment,
                   radius, #pairs, and per-property avg/std/count of the change.
  compounds.csv  — one row per input molecule: id, SMILES, and its property values.
  pairs.csv      — a sample of the matched molecular pairs backing the rules.

Usage:
  python 0_build_mmpdb/export_mmpdb.py /data/.../mmp_moves/sample.mmpdb [--out-dir DIR] [--max-rules N]
"""
from __future__ import annotations
import argparse, csv, os, re, sqlite3

_DUM = re.compile(r"\[\d*\*(?::\d+)?\]")
def _norm(s): return _DUM.sub("*", s)


def export(db_path: str, out_dir: str, max_rules: int) -> None:
    c = sqlite3.connect(db_path)
    props = [n for _, n in sorted(c.execute("SELECT id,name FROM property_name"))]
    pid = {n: i for i, n in c.execute("SELECT id,name FROM property_name")}
    os.makedirs(out_dir, exist_ok=True)

    # ---- rules.csv (wide: per-property avg/std/count) --------------------
    rule_envs = c.execute(
        "SELECT re.id, re.radius, re.num_pairs, f.smiles, t.smiles "
        "FROM rule_environment re "
        "JOIN rule r ON r.id=re.rule_id "
        "JOIN rule_smiles f ON f.id=r.from_smiles_id "
        "JOIN rule_smiles t ON t.id=r.to_smiles_id "
        "ORDER BY re.num_pairs DESC LIMIT ?", (max_rules,)).fetchall()
    cols = ["from_fragment", "to_fragment", "radius", "num_pairs"]
    for p in props:
        cols += [f"{p}_avg", f"{p}_std", f"{p}_count"]
    rules_path = os.path.join(out_dir, "rules.csv")
    with open(rules_path, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(cols)
        for reid, radius, npairs, fs, ts in rule_envs:
            st = {p_id: (a, s, cnt) for p_id, a, s, cnt in c.execute(
                "SELECT property_name_id,avg,std,count FROM rule_environment_statistics "
                "WHERE rule_environment_id=?", (reid,))}
            row = [_norm(fs), _norm(ts), radius, npairs]
            for p in props:
                a, s, cnt = st.get(pid[p], (None, None, ""))
                row += [round(a, 4) if a is not None else "",
                        round(s, 4) if s is not None else "", cnt]
            w.writerow(row)

    # ---- compounds.csv ---------------------------------------------------
    comp_path = os.path.join(out_dir, "compounds.csv")
    cprops = c.execute(
        "SELECT cp.compound_id, pn.name, cp.value FROM compound_property cp "
        "JOIN property_name pn ON pn.id=cp.property_name_id").fetchall()
    pmap: dict = {}
    for cid, name, val in cprops:
        pmap.setdefault(cid, {})[name] = val
    with open(comp_path, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["id", "smiles"] + props)
        for cid, public_id, in_smiles in c.execute(
                "SELECT id, public_id, input_smiles FROM compound"):
            w.writerow([public_id, in_smiles] + [pmap.get(cid, {}).get(p, "") for p in props])

    # ---- pairs.csv (sample) ----------------------------------------------
    pairs_path = os.path.join(out_dir, "pairs.csv")
    with open(pairs_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["from_fragment", "to_fragment", "radius", "constant_core",
                    "compound1", "compound2"])
        for fs, ts, radius, const, c1, c2 in c.execute(
                "SELECT f.smiles,t.smiles,re.radius,cs.smiles,a.public_id,b.public_id "
                "FROM pair p "
                "JOIN rule_environment re ON re.id=p.rule_environment_id "
                "JOIN rule r ON r.id=re.rule_id "
                "JOIN rule_smiles f ON f.id=r.from_smiles_id "
                "JOIN rule_smiles t ON t.id=r.to_smiles_id "
                "JOIN constant_smiles cs ON cs.id=p.constant_id "
                "JOIN compound a ON a.id=p.compound1_id "
                "JOIN compound b ON b.id=p.compound2_id LIMIT 2000"):
            w.writerow([_norm(fs), _norm(ts), radius, _norm(const), c1, c2])

    print(f"wrote:\n  {rules_path}\n  {comp_path}\n  {pairs_path}")
    print(f"properties: {props}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--max-rules", type=int, default=20000)
    a = ap.parse_args()
    export(a.db, a.out_dir or os.path.dirname(os.path.abspath(a.db)), a.max_rules)
