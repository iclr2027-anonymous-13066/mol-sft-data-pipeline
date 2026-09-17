#!/usr/bin/env python3
"""Drop SFT records whose source instance shares a Murcko scaffold with the TEST set.

The test set is the 100-instance exact benchmark subset
(`benchmark_exact_subset100.jsonl`), which scores a model on hitting a
target scaffold. A training chain built from an instance with the SAME
`scaffold_smiles` is test leakage: the model can memorise the exact scaffold it
will be asked to reproduce. This finds every leaking instance id and, with
--apply, rewrites the sftdata files dropping ALL records of the affected chains
(a chain = one `metadata.group_id`, whose prefix is the instance id).

BUT many raw hits are trivial scaffolds — benzene / pyridine / cyclohexane are
not memorisable leakage, they are chemistry. Use --min-scaffold-heavy /
--min-scaffold-rings to keep those; the report prints the drop cost at several
thresholds so the trade-off is visible before --apply.

Matching is plain string equality on `scaffold_smiles`; both sides are written by
the same pipeline as RDKit-canonical SMILES (verified: 0/300 sampled train
scaffolds change under re-canonicalisation).

Normally driven by run_scripts/pipeline/4b_filter_test_scaffold_leak.sh. Direct use:

Report only (default):
    python 4_sftdata_gen/scripts/filter_test_scaffold_leak.py \
        --dir data/training_data/sftdata/generation_2m_scaffold \
        --leak-ids /some/cache.json
Apply (rewrite in place, atomic per file; only chunks with a `.done` marker):
    ... --min-scaffold-heavy 10 --apply
The --leak-ids cache stores {instance_id: scaffold_smiles} for one (test file,
instances dir) pair, so re-running with a different threshold reuses it and
skips the 13G instance scan.
"""
import argparse
import glob
import json
import os
import re
import shutil
from collections import Counter
from multiprocessing import Pool

DEFAULT_TEST = (
    "data/training_data/instances/benchmark_exact_subset100.jsonl"
)
DEFAULT_INSTANCES = "data/training_data/instances/generation_2m_scaffold"

# metadata sits at the end of every record; a regex beats json.loads on 32G.
GID_RE = re.compile(rb'"group_id":\s*"([^"]+?)__')


def _load_test(path):
    """-> (set of test scaffold_smiles, set of test target smiles, n rows)."""
    scaffolds, smiles, n = set(), set(), 0
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            n += 1
            d = json.loads(line)
            if d.get("scaffold_smiles"):
                scaffolds.add(d["scaffold_smiles"])
            if d.get("smiles"):
                smiles.add(d["smiles"])
    return scaffolds, smiles, n


def _scaffold_size(scaffolds):
    """-> {scaffold_smiles: (heavy_atoms, n_rings)} (RDKit, only ~700 mols)."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    out = {}
    for s in scaffolds:
        m = Chem.MolFromSmiles(s)
        out[s] = (m.GetNumHeavyAtoms(), m.GetRingInfo().NumRings()) if m else (0, 0)
    return out


_TEST_SCAF = _TEST_SMI = None


def _init(scaffolds, smiles):
    global _TEST_SCAF, _TEST_SMI
    _TEST_SCAF, _TEST_SMI = scaffolds, smiles


def _scan_instances(path):
    """-> (n rows, {id: scaffold_smiles}, [ids whose target molecule is in test])."""
    hits, mol_hits, n = {}, [], 0
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            n += 1
            try:
                d = json.loads(line)
            except json.JSONDecodeError:  # torn tail of a file still being written
                continue
            if d.get("scaffold_smiles") in _TEST_SCAF:
                hits[d["id"]] = d["scaffold_smiles"]
            if d.get("smiles") in _TEST_SMI:
                mol_hits.append(d["id"])
    return n, hits, mol_hits


def _scan_sft(args):
    """-> (file, n records, n leaking records, {leaking group_ids})."""
    path, leak_ids = args
    total = leak = 0
    gids = set()
    with open(path, "rb") as fh:
        for line in fh:
            if not line.strip():
                continue
            total += 1
            m = GID_RE.search(line)
            if not m:
                continue
            gid = m.group(1).decode()
            if gid in leak_ids:
                leak += 1
                gids.add(gid)
    return path, total, leak, gids


def _rewrite(args):
    """Rewrite one file without the leaking records. -> (file, kept, dropped)."""
    path, leak_ids = args
    tmp = path + ".tmp"
    kept = dropped = 0
    with open(path, "rb") as src, open(tmp, "wb") as dst:
        for line in src:
            if not line.strip():
                continue
            m = GID_RE.search(line)
            if m and m.group(1).decode() in leak_ids:
                dropped += 1
                continue
            dst.write(line)
            kept += 1
    shutil.move(tmp, path)
    return path, kept, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="sftdata subdir holding toolchains_*.jsonl")
    ap.add_argument("--test-file", default=DEFAULT_TEST)
    ap.add_argument("--instances-dir", default=DEFAULT_INSTANCES)
    ap.add_argument("--leak-ids", help="cache of {instance_id: scaffold_smiles}; reused if it exists")
    ap.add_argument("--procs", type=int, default=32)
    ap.add_argument("--min-scaffold-heavy", type=int, default=0,
                    help="only treat a shared scaffold as leakage if it has >= N heavy atoms "
                         "(0 = drop every shared scaffold, incl. benzene)")
    ap.add_argument("--min-scaffold-rings", type=int, default=0,
                    help="likewise, on the scaffold's ring count")
    ap.add_argument("--apply", action="store_true", help="rewrite files in place")
    ap.add_argument("--force-inflight", action="store_true",
                    help="with --apply, also rewrite chunks lacking a .done marker")
    args = ap.parse_args()

    scaffolds, smiles, n_test = _load_test(args.test_file)
    print(f"[test] {n_test} rows | {len(scaffolds)} unique scaffolds | {len(smiles)} unique smiles")

    if args.leak_ids and os.path.exists(args.leak_ids):
        with open(args.leak_ids) as fh:
            id_scaf = json.load(fh)
        print(f"[instances] loaded {len(id_scaf)} scaffold-leaking ids from {args.leak_ids}")
    else:
        files = sorted(glob.glob(os.path.join(args.instances_dir, "*.jsonl")))
        print(f"[instances] scanning {len(files)} shards in {args.instances_dir} ...")
        id_scaf, n_inst, n_mol = {}, 0, 0
        with Pool(args.procs, initializer=_init, initargs=(scaffolds, smiles)) as pool:
            for n, hits, mol_hits in pool.imap_unordered(_scan_instances, files):
                n_inst += n
                n_mol += len(mol_hits)
                id_scaf.update(hits)
        print(f"[instances] {n_inst} rows | {len(id_scaf)} share a test scaffold "
              f"({100.0 * len(id_scaf) / max(n_inst, 1):.3f}%) | {n_mol} share the exact test molecule")
        if args.leak_ids:
            with open(args.leak_ids, "w") as fh:
                json.dump(id_scaf, fh)
            print(f"[instances] wrote {args.leak_ids}")

    size = _scaffold_size(set(id_scaf.values()))
    per_scaf = Counter(id_scaf.values())
    total_hits = sum(per_scaf.values())

    # No shared scaffold at all is the EXPECTED outcome when the instance pool was
    # split with the benchmark scaffolds already removed (pool_split rule_1). Say
    # so and skip the breakdowns — every percentage below divides by total_hits.
    if not total_hits:
        print("\n[scaffolds] none — no instance shares a scaffold with the test set.")
        print("[thresholds] nothing to drop at any --min-scaffold-heavy.")
    else:
        print("\n[scaffolds] top 15 shared scaffolds by instance count:")
        for s, c in per_scaf.most_common(15):
            h, r = size[s]
            print(f"    {c:7d} ({100.0 * c / total_hits:5.1f}%)  heavy={h:2d} rings={r}  {s}")

        print("\n[thresholds] instances dropped at each --min-scaffold-heavy:")
        for t in (0, 7, 10, 12, 14, 16, 20):
            n = sum(c for s, c in per_scaf.items() if size[s][0] >= t)
            ns = sum(1 for s in per_scaf if size[s][0] >= t)
            print(f"    >= {t:2d} heavy atoms: {n:7d} instances ({100.0 * n / total_hits:5.1f}% of hits) "
                  f"| {ns}/{len(per_scaf)} scaffolds")

    leak_ids = {i for i, s in id_scaf.items()
                if size[s][0] >= args.min_scaffold_heavy and size[s][1] >= args.min_scaffold_rings}
    print(f"\n[filter] min_heavy={args.min_scaffold_heavy} min_rings={args.min_scaffold_rings} "
          f"-> dropping chains of {len(leak_ids)} instance ids")

    sft_files = sorted(glob.glob(os.path.join(args.dir, "*.jsonl")))
    print(f"[sftdata] scanning {len(sft_files)} files in {args.dir} ...")
    total = leak = 0
    per_file, all_gids = {}, set()
    with Pool(args.procs) as pool:
        for path, n, k, gids in pool.imap_unordered(_scan_sft, [(f, leak_ids) for f in sft_files]):
            total += n
            leak += k
            per_file[path] = (n, k)
            all_gids |= gids
    print(f"[sftdata] {total} records | {leak} leaking ({100.0 * leak / max(total, 1):.3f}%) "
          f"across {len(all_gids)} chains / {len({g.split('__')[0] for g in all_gids})} instances")

    if not args.apply:
        print("\n(report only — rerun with --apply to rewrite the files)")
        return

    targets = [f for f in sft_files if per_file[f][1]]
    if not args.force_inflight:
        for f in [f for f in targets if not os.path.exists(f + ".done")]:
            print(f"[apply] SKIP (no .done marker, still being written): {os.path.basename(f)}")
        targets = [f for f in targets if os.path.exists(f + ".done")]
    print(f"[apply] rewriting {len(targets)} files ...")
    kept_t = dropped_t = 0
    with Pool(args.procs) as pool:
        for path, kept, dropped in pool.imap_unordered(_rewrite, [(f, leak_ids) for f in targets]):
            kept_t += kept
            dropped_t += dropped
    print(f"[apply] done — kept {kept_t}, dropped {dropped_t}")


if __name__ == "__main__":
    main()
