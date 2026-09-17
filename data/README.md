# `data/`

Only `benchmark/` is versioned. The other directories below are created by the pipeline
and are ignored by git, which is why a fresh clone shows a single folder here.

| Directory | Versioned | Contents |
|---|---|---|
| `benchmark/` | yes | the held-out evaluation sets (see `benchmark/README.md`) |
| `pool/` | no | the property-tagged molecule pool, `train_pool_props.parquet` |
| `mmp_moves/` | no | the mmpdb-derived move set built by stage 0, read by the `suggest_edits` tool (override with `MMP_MOVES_DIR`) |
| `develop/` | no | intermediates from building the pool |
| `training_data/` | no | stage 1–4 outputs: `instances/`, `toolchain/`, `sftdata/`, and `fg_catalog.json` |
| `analysis/` | no | analysis outputs |

Every stage takes the location as an argument (`--pool`, `--output`, `--out-dir`, …) or an
environment variable (`POOL`, `OUTPUT`, `SRC`, …), so nothing forces this layout; it is
just what the defaults assume. See the root `README.md` for how the pool is built.
