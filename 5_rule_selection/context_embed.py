"""Molecular context vectors `C` for the rule-selection model, precomputed per SMILES.

The model only uses `C` as the QUERY that gates the interpretable features, so it is
computed once per unique molecule and cached — the same state molecule shows up in
many decisions, and the encoder never trains.

Two encoders:

``morgan``  (default, no extra dependencies) Morgan fingerprint (radius 2, 2048 bits)
            concatenated with a short descriptor block. Boring but honest, and it
            makes the whole pipeline runnable with nothing but RDKit.

``molbert`` BenevolentAI MolBERT, 12 layers / hidden 768 / vocab 42.
            **The published package cannot be installed here**: setup.py pins
            torch==1.4.0, pytorch-lightning==0.8.4, transformers==3.5.1 (python 3.7
            era) against this node's torch 2.11 / transformers 5.x. What DOES work,
            verified, is using the repo for its tokenizer only and loading the
            checkpoint weights into a locally built encoder:

              git clone --depth 1 https://github.com/BenevolentAI/MolBERT.git
              curl -L -o molbert.zip https://ndownloader.figshare.com/files/25611290
              unzip molbert.zip          # -> molbert_100epochs/checkpoints/last.ckpt

            `molbert.utils.featurizer.molfeaturizer` imports cleanly under modern
            rdkit/numpy (it only needs those), so tokenisation is the original, not a
            re-implementation. The encoder here is HF's `BertEncoder` plus MolBERT's
            own non-learnt sinusoidal position embedding (`SuperPositionalBertEmbeddings`
            in their models/base.py) — a plain `BertModel` would silently substitute a
            LEARNT position embedding the checkpoint does not contain, so this part
            has to be rebuilt rather than loaded.

Usage:
    python -m 5_rule_selection.context_embed --states .../states.jsonl \
        --out-dir .../ctx_morgan --encoder morgan
    MOLBERT_REPO=/path/to/MolBERT python -m 5_rule_selection.context_embed \
        --states .../states.jsonl --out-dir .../ctx_molbert --encoder molbert \
        --ckpt /path/to/molbert_100epochs/checkpoints/last.ckpt --batch 256
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np


def unique_smiles(states_paths) -> list:
    """Every molecule a decision stands on, in first-seen order.

    Takes one path or several. Several is what a combined corpus needs — training rows
    from one states file and held-out rows from another must share ONE context matrix,
    because `ctx.npy` stores a row index into it and there is only one such matrix per
    tensors directory.
    """
    if isinstance(states_paths, str):
        states_paths = [states_paths]
    seen, out = set(), []
    k = 0
    for path in states_paths:
        with open(path) as fh:
            for line in fh:
                if not line.strip():
                    continue
                k += 1
                if k % 100000 == 0:
                    print(f"  scan {k:,} states, {len(out):,} molecules", flush=True)
                d = json.loads(line)
                for s in ([d.get("state_smiles")]
                          + [c.get("smiles") for c in d.get("candidates") or []]):
                    if s and s not in seen:
                        seen.add(s)
                        out.append(s)
    return out


# --------------------------------------------------------------------------- #
#  morgan
# --------------------------------------------------------------------------- #
_MORGAN_CFG = {"n_bits": 2048, "radius": 2}


def _morgan_chunk(smiles: list) -> np.ndarray:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    n_bits, radius = _MORGAN_CFG["n_bits"], _MORGAN_CFG["radius"]
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    desc = [Descriptors.MolWt, Descriptors.MolLogP, Descriptors.TPSA,
            Descriptors.NumHDonors, Descriptors.NumHAcceptors,
            Descriptors.NumRotatableBonds, Descriptors.RingCount,
            Descriptors.FractionCSP3, Descriptors.HeavyAtomCount, Descriptors.qed]
    out = np.zeros((len(smiles), n_bits + len(desc)), dtype=np.float32)
    for i, smi in enumerate(smiles):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        out[i, :n_bits] = gen.GetFingerprintAsNumPy(m)
        for j, fn in enumerate(desc):
            try:
                out[i, n_bits + j] = float(fn(m))
            except Exception:
                pass
    return out


def encode_morgan(smiles: list, n_bits: int = 2048, radius: int = 2,
                  workers: int = 0, chunk: int = 5000) -> np.ndarray:
    """Fingerprints over a process pool. A depth-5 dataset has >1M unique molecules and
    one core spends the better part of an hour on that."""
    import multiprocessing as mp
    _MORGAN_CFG.update(n_bits=n_bits, radius=radius)
    workers = workers or min(48, (os.cpu_count() or 8))
    chunks = [smiles[i:i + chunk] for i in range(0, len(smiles), chunk)]
    parts = []
    with mp.get_context("fork").Pool(workers) as pool:
        for k, arr in enumerate(pool.imap(_morgan_chunk, chunks), 1):
            parts.append(arr)
            if k % 40 == 0:
                print(f"  morgan {min(k * chunk, len(smiles)):,}/{len(smiles):,}", flush=True)
    out = np.concatenate(parts) if parts else np.zeros((0, n_bits + 10), np.float32)
    del parts
    # descriptors on a wildly different scale from bits: standardise the tail
    tail = out[:, n_bits:]
    mu, sd = tail.mean(0, keepdims=True), tail.std(0, keepdims=True) + 1e-6
    out[:, n_bits:] = (tail - mu) / sd
    return out


# --------------------------------------------------------------------------- #
#  molbert
# --------------------------------------------------------------------------- #
class _SinusoidalPositions:
    """MolBERT's non-learnt position embedding, from the checkpoint's own inv_freq."""

    def __init__(self, inv_freq):
        self.inv_freq = inv_freq

    def __call__(self, length: int, device, dtype):
        pos = np.arange(length, dtype=np.float64)
        import torch
        pos_seq = torch.tensor(pos, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        sin_inp = torch.ger(pos_seq, self.inv_freq)
        emb = np.concatenate([np.sin(sin_inp.cpu().numpy()), np.cos(sin_inp.cpu().numpy())], -1)
        return torch.tensor(emb, device=device, dtype=dtype).unsqueeze(0)


# --------------------------------------------------------------------------- #
#  molbert
# --------------------------------------------------------------------------- #
_MB = {"repo": "", "max_len": 128}
_FEAT = None            # set by _tok_init, in a pool worker OR in this process


def _tok_init():
    global _FEAT
    import logging
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    logging.disable(logging.WARNING)
    if _MB["repo"] not in sys.path:
        sys.path.insert(0, _MB["repo"])
    from molbert.utils.featurizer.molfeaturizer import SmilesIndexFeaturizer
    _FEAT = SmilesIndexFeaturizer.bert_smiles_index_featurizer(_MB["max_len"])


def _tok_chunk(smis):
    ids, valid = _FEAT.transform(smis)
    ids = np.asarray(ids, dtype=np.uint8)          # vocab is 42, pad_idx 0
    return ids, np.asarray(valid, dtype=bool)


def tokenize_molbert(smiles: list, repo: str, max_len: int = 128, workers: int = 0,
                     chunk: int = 20000):
    """SMILES -> (ids [N,max_len] uint8, length [N] int16, valid [N] bool).

    MolBERT's own tokenizer, run in a process pool. It is pure rdkit + python, so it
    parallelises perfectly and is NOT the bottleneck once taken out of the GPU loop:
    248k molecules/s on 96 processes, i.e. 29 s for 7.1 M. Inside the GPU loop (one
    batch tokenised, one batch forwarded, serially) the same work costs ~46 min of
    GPU idle time.

    **A single chunk is tokenised in-process, with no pool at all.** Every worker runs
    `_tok_init`, which imports MolBERT's featurizer, and that import is what a fork
    costs — so a caller handing over five molecules paid ~1.5 s to spawn 32 processes
    of which 31 had nothing to do. That is the shape of every INFERENCE call (a policy
    loop preparing one state plus its four candidate products per round), and it made
    the context preparation 86% of that loop's wall clock. Measured on 5 molecules:
    1.52 s over 32 workers, 0.43 s over 1, 0.02 s inline. The bulk path — millions of
    molecules, many chunks — is untouched.
    """
    import multiprocessing as mp
    _MB.update(repo=repo, max_len=max_len)
    workers = workers or min(96, (os.cpu_count() or 8))
    chunks = [smiles[i:i + chunk] for i in range(0, len(smiles), chunk)]
    if len(chunks) <= 1:
        t0 = time.time()
        ids = np.zeros((len(smiles), max_len), dtype=np.uint8)
        valid = np.zeros(len(smiles), dtype=bool)
        if smiles:
            if _FEAT is None:
                _tok_init()                    # once per process, then cached
            a, v = _tok_chunk(smiles)
            ids[:len(v)], valid[:len(v)] = a, v
        length = (ids != 0).sum(1).astype(np.int16)
        n_ok = int(valid.sum())
        if len(smiles) >= 2000:                # stay quiet on the per-round calls
            print(f"# tokenised {len(smiles):,} in {time.time()-t0:.0f}s (inline); "
                  f"{int((~valid).sum()):,} rejected", flush=True)
        return ids, length, valid
    ids = np.zeros((len(smiles), max_len), dtype=np.uint8)
    valid = np.zeros(len(smiles), dtype=bool)
    t0 = time.time()
    at = 0
    with mp.get_context("fork").Pool(workers, initializer=_tok_init) as pool:
        for k, (a, v) in enumerate(pool.imap(_tok_chunk, chunks), 1):
            ids[at:at + len(v)] = a
            valid[at:at + len(v)] = v
            at += len(v)
            if k % 80 == 0:
                print(f"  tokenise {at:,}/{len(smiles):,}  {time.time()-t0:.0f}s",
                      flush=True)
    length = (ids != 0).sum(1).astype(np.int16)
    # `valid` can be ALL FALSE — a caller handing over a batch of molecules the
    # featurizer refuses outright (the deeper products of a depth-5 tree run long, and
    # MolBERT was pretrained on GuacaMol at 128 tokens). Reducing over the empty
    # selection raises, so the length summary is only printed when there is one.
    n_ok = int(valid.sum())
    tail = (f"length mean {length[valid].mean():.1f} max {int(length[valid].max())}"
            if n_ok else "no molecule survived the tokenizer")
    print(f"# tokenised {len(smiles):,} in {time.time()-t0:.0f}s "
          f"({len(smiles)/max(time.time()-t0,1e-9):,.0f} mol/s, {workers} procs); "
          f"{int((~valid).sum()):,} rejected ({(~valid).mean():.4%}); " + tail)
    return ids, length, valid


def _build_molbert(ckpt: str, dtype, device):
    """The encoder, rebuilt locally: HF BertEncoder + MolBERT's sinusoidal positions."""
    import torch
    from torch import nn
    from transformers import BertConfig
    from transformers.models.bert.modeling_bert import BertEncoder

    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    hp = ck.get("hyper_parameters") or {}
    max_len = int(hp.get("max_seq_length", 128))
    vocab = int(sd["model.bert.embeddings.word_embeddings.weight"].shape[0])
    hidden = int(sd["model.bert.embeddings.word_embeddings.weight"].shape[1])
    n_layers = len({k.split(".")[4] for k in sd
                    if k.startswith("model.bert.encoder.layer.")})
    cfg = BertConfig(vocab_size=vocab, hidden_size=hidden, num_hidden_layers=n_layers,
                     num_attention_heads=12, intermediate_size=4 * hidden,
                     max_position_embeddings=512, type_vocab_size=2)
    # SDPA over the explicit attention matmuls: measured 29.0k vs 19.4k molecules/s
    # at bf16, seq 128. The old code asked for "eager".
    cfg._attn_implementation = "sdpa"

    pref = "model.bert."
    word = nn.Embedding(vocab, hidden, padding_idx=0)
    ttype = nn.Embedding(cfg.type_vocab_size, hidden)
    ln = nn.LayerNorm(hidden, eps=cfg.layer_norm_eps)
    encoder = BertEncoder(cfg)
    word.weight.data.copy_(sd[pref + "embeddings.word_embeddings.weight"])
    ttype.weight.data.copy_(sd[pref + "embeddings.token_type_embeddings.weight"])
    ln.weight.data.copy_(sd[pref + "embeddings.LayerNorm.weight"])
    ln.bias.data.copy_(sd[pref + "embeddings.LayerNorm.bias"])
    enc_sd = {k[len(pref + "encoder."):]: v for k, v in sd.items()
              if k.startswith(pref + "encoder.")}
    missing, unexpected = encoder.load_state_dict(enc_sd, strict=False)
    if missing:
        raise SystemExit(f"refusing to embed with partially loaded weights: "
                         f"{len(missing)} missing, e.g. {list(missing)[:5]}")
    pos = _SinusoidalPositions(sd[pref + "embeddings.position_embeddings.inv_freq"])
    mods = [m.to(device, dtype).eval() for m in (word, ttype, ln, encoder)]
    return mods, pos, hidden, max_len, vocab, n_layers, len(enc_sd), len(unexpected)


def _embed_shard(gpu: int, shm_name: str, n_rows: int, ids_path: str, order_path: str,
                 batches: list, ckpt: str, pooling: str, dtype_name: str,
                 hidden: int, tag: str) -> None:
    """One GPU's share of the length-sorted batches, scattered into SHARED MEMORY.

    Not into the output memmap. The batches are length-sorted, so their row indices are
    spread across the whole 21.9 GiB file and `out[rows] = emb` becomes 4,096 random
    3 kB writes per batch. Measured on GPFS that was 4.7 s per batch against 0.06 s of
    GPU work — 850 molecules/s with the GPU at 0% utilisation, i.e. the padding win
    from sorting was paid back forty times over in random I/O. Scattering into RAM
    costs microseconds and the parent writes the array out in one sequential pass.
    """
    from multiprocessing import shared_memory

    import torch
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    dev = torch.device("cuda:0")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
             "fp32": torch.float32}[dtype_name]
    (word, ttype, ln, encoder), pos, *_ = _build_molbert(ckpt, dtype, dev)
    ids_all = np.load(ids_path)                     # 911 MB, held in RAM
    order = np.load(order_path)
    shm = shared_memory.SharedMemory(name=shm_name)
    out = np.ndarray((n_rows, hidden), dtype=np.float32, buffer=shm.buf)
    done = 0
    t0 = time.time()
    with torch.no_grad():
        for bi, (lo, hi) in enumerate(batches):
            rows = np.asarray(order[lo:hi])
            chunk = np.asarray(ids_all[rows])
            # trim to this batch's longest sequence — the list is length-sorted, so a
            # batch is nearly uniform and the padding waste (56% at a fixed 128)
            # essentially disappears
            width = max(int((chunk != 0).sum(1).max()), 1)
            t = torch.from_numpy(chunk[:, :width].astype(np.int64)).to(dev)
            attn = (t != 0)
            h = word(t) + pos(width, dev, word.weight.dtype) + ttype(torch.zeros_like(t))
            h = ln(h)
            ext = (~attn)[:, None, None, :].to(h.dtype) * torch.finfo(h.dtype).min
            h = encoder(h, attention_mask=ext).last_hidden_state
            if pooling == "cls":
                emb = h[:, 0]
            else:
                m = attn.unsqueeze(-1).to(h.dtype)
                emb = (h * m).sum(1) / m.sum(1).clamp(min=1)
            out[rows] = emb.float().cpu().numpy()
            done += len(rows)
            if bi % 200 == 0:
                print(f"  [{tag}] {done:,} mols  {time.time()-t0:.0f}s "
                      f"({done/max(time.time()-t0,1e-9):,.0f} mol/s)", flush=True)
    print(f"  [{tag}] done {done:,} in {time.time()-t0:.0f}s "
          f"({done/max(time.time()-t0,1e-9):,.0f} mol/s)", flush=True)
    shm.close()


def encode_molbert(smiles: list, ckpt: str, repo: str, out_dir: str,
                   batch: int = 4096, pooling: str = "mean", gpus=(0,),
                   dtype_name: str = "bf16", workers: int = 0) -> tuple:
    """MolBERT mean-pooled embeddings for `smiles`, written to <out_dir>/ctx.npy.

    Four optimisations over the straightforward loop, each measured on this data
    (one GPU, molecules/s):

        fp32 + eager + fixed seq 128            2,253      <- the original
        fp32 + sdpa                             2,371
        bf16 + sdpa                            28,979
        bf16 + sdpa + length-trimmed (~56)     ~66,000
        ... x 2 GPUs                          ~130,000

    plus tokenisation moved out of the GPU loop into a process pool (29 s for 7.1 M
    instead of ~46 min of serial work blocking the device). 7.1 M molecules go from
    ~1 h 40 m on one GPU to a couple of minutes.

    A SMILES the featurizer rejects — in practice one longer than 128 tokens, since
    MolBERT was pretrained on GuacaMol at that length and simply has no
    representation for it — keeps a ZERO row. The count is returned so the caller can
    exclude those molecules from the dataset rather than train on a fake context.
    """
    import torch
    import torch.multiprocessing as tmp

    ids, length, valid = tokenize_molbert(smiles, repo, workers=workers)
    # probe the checkpoint on CPU just for its shapes / weight-load verdict
    _, _, hidden, max_len, vocab, n_layers, n_given, n_unexp = _build_molbert(
        ckpt, torch.float32, torch.device("cpu"))
    print(f"# molbert: vocab {vocab} hidden {hidden} layers {n_layers} max_len "
          f"{max_len} | encoder weights {n_given} given, {n_unexp} unexpected")

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ctx.npy")
    ids_path = os.path.join(out_dir, "_ids.npy")
    order_path = os.path.join(out_dir, "_order.npy")
    np.save(ids_path, ids)
    # length-sorted so each batch is nearly uniform; rejected rows are dropped from the
    # work list entirely (their output row stays zero)
    keep = np.flatnonzero(valid)
    order = keep[np.argsort(length[keep], kind="stable")]
    np.save(order_path, order)

    edges = list(range(0, len(order), batch)) + [len(order)]
    all_batches = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
    gpus = list(gpus) or [0]
    shards = [all_batches[i::len(gpus)] for i in range(len(gpus))]
    print(f"# {len(order):,} molecules to embed in {len(all_batches):,} batches of "
          f"{batch} over {len(gpus)} GPU(s) [{','.join(map(str, gpus))}], "
          f"{dtype_name}, length-sorted", flush=True)

    from multiprocessing import shared_memory
    nbytes = len(smiles) * hidden * 4
    shm = shared_memory.SharedMemory(create=True, size=nbytes)
    buf = np.ndarray((len(smiles), hidden), dtype=np.float32, buffer=shm.buf)
    buf[:] = 0.0                                   # rejected rows stay zero
    print(f"# staging {nbytes/2**30:.1f} GiB in shared memory ({shm.name})", flush=True)
    t0 = time.time()
    try:
        if len(gpus) == 1:
            _embed_shard(gpus[0], shm.name, len(smiles), ids_path, order_path,
                         shards[0], ckpt, pooling, dtype_name, hidden, f"gpu{gpus[0]}")
        else:
            procs = []
            mpctx = tmp.get_context("spawn")
            for g, sh in zip(gpus, shards):
                pr = mpctx.Process(target=_embed_shard,
                                   args=(g, shm.name, len(smiles), ids_path,
                                         order_path, sh, ckpt, pooling, dtype_name,
                                         hidden, f"gpu{g}"))
                pr.start()
                procs.append(pr)
            for pr in procs:
                pr.join()
                if pr.exitcode != 0:
                    raise SystemExit(f"a shard failed (exit {pr.exitcode})")
        wall = time.time() - t0
        print(f"# embedded {len(order):,} molecules in {wall:.0f}s "
              f"({len(order)/max(wall,1e-9):,.0f} mol/s over {len(gpus)} GPU(s))",
              flush=True)
        t1 = time.time()
        mm = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32,
                                       shape=(len(smiles), hidden))
        CH = 250_000
        for i in range(0, len(smiles), CH):
            mm[i:i + CH] = buf[i:i + CH]
        mm.flush()
        del mm
        print(f"# wrote {nbytes/2**30:.1f} GiB sequentially in {time.time()-t1:.0f}s",
              flush=True)
    finally:
        del buf
        shm.close()
        shm.unlink()
    for f in (ids_path, order_path):
        if os.path.exists(f):
            os.remove(f)
    X = np.load(out_path, mmap_mode="r")
    rejected = [smiles[i] for i in np.flatnonzero(~valid)]
    return X, rejected


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", action="append", required=True,
                    help="states.jsonl (repeatable). All of them share ONE matrix.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--encoder", choices=["morgan", "molbert"], default="morgan")
    ap.add_argument("--ckpt", default=os.environ.get("MOLBERT_CKPT", ""))
    ap.add_argument("--repo", default=os.environ.get("MOLBERT_REPO", ""))
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--gpus", default="0", help="molbert: comma-separated GPU ids")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16",
                    help="molbert forward precision; the output is always stored fp32")
    ap.add_argument("--smiles-from", default="",
                    help="reuse the molecule list (and ORDER) of an existing "
                         "index.json instead of rescanning --states. Required when two "
                         "encoders have to be row-comparable.")
    ap.add_argument("--pooling", choices=["mean", "cls"], default="mean")
    ap.add_argument("--workers", type=int, default=0, help="morgan process pool (0=auto)")
    ap.add_argument("--device", default="cuda", help="cuda | cpu (599 mols is fine on cpu)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.smiles_from:
        with open(args.smiles_from) as fh:
            smiles = json.load(fh)["smiles"]
        print(f"# {len(smiles):,} molecules, order taken from {args.smiles_from}",
              flush=True)
    else:
        smiles = unique_smiles(args.states)
        print(f"# {len(smiles):,} unique molecules from {args.states}", flush=True)

    rejected = []
    if args.encoder == "morgan":
        X = encode_morgan(smiles, workers=args.workers)
        np.save(os.path.join(args.out_dir, "ctx.npy"), X)
    else:
        if not args.ckpt or not args.repo:
            raise SystemExit("--encoder molbert needs --ckpt and --repo "
                             "(MOLBERT_CKPT / MOLBERT_REPO)")
        gpus = [int(x) for x in str(args.gpus).split(",") if x.strip() != ""]
        X, rejected = encode_molbert(smiles, args.ckpt, args.repo, args.out_dir,
                                     batch=args.batch, pooling=args.pooling,
                                     gpus=gpus, dtype_name=args.dtype,
                                     workers=args.workers)
    zero = int((np.abs(np.asarray(X[:])).sum(1) == 0).sum())
    with open(os.path.join(args.out_dir, "index.json"), "w") as fh:
        json.dump({"encoder": args.encoder, "dim": int(X.shape[1]),
                   "n_zero_vectors": zero, "smiles": smiles}, fh)
    # The molecules the encoder cannot represent, so the dataset build can drop the
    # decision states that stand on them instead of feeding the model a zero vector.
    if rejected:
        rp = os.path.join(args.out_dir, "rejected_smiles.txt")
        with open(rp, "w") as fh:
            fh.write("\n".join(rejected) + "\n")
        print(f"# {len(rejected):,} molecules ({len(rejected)/len(smiles):.4%}) rejected "
              f"-> {rp}  (pass it to prepare_dataset.py --exclude-smiles)")
    if zero:
        print(f"# {zero:,} molecules ({zero/len(smiles):.4%}) have a zero context vector")
    print(f"# wrote {tuple(X.shape)} -> {args.out_dir}/ctx.npy")


if __name__ == "__main__":
    main()
