# molkit/tools

RDKit/CPU molecular tools and the FastAPI server that exposes each one over HTTP.

---

## Overview

Every tool in this package is a `BaseTool` subclass (`base.py`). Each subclass defines:

- `name` — snake_case identifier used in LLM function-calling schemas
- `description` — natural language description passed to the LLM
- `parameters` — JSON Schema for the tool's inputs
- `execute(**kwargs)` — core logic; returns `str | dict`

The tool registry (`__init__.py`) maps each tool name to a fixed port. When the tool stack is running, each tool is served as its own OS process (one `uvicorn` server per tool) via `tool_server.py`. RDKit's C++ layer releases the GIL, so multiple worker processes handle concurrent requests without contention.

### HTTP API

Every tool server exposes these endpoints:

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/call` | Call the tool once |
| `POST` | `/call_batch` | Call the tool on a list of inputs |
| `GET` | `/schema` | Return the OpenAI function-calling schema |
| `GET` | `/health` | Liveness check — returns `{"status": "ok", "tool": "<name>"}` |

**`/call` request body:**
```json
{
  "inputs": { "mol_smiles": "CCO" },
  "return_text": true
}
```
- `inputs` — dict matching the tool's `parameters` schema
- `return_text` — `true` (default): return a human-readable string suitable for an LLM tool result; `false`: return the raw dict/value (for evaluation code)

**`/call` response:**
```json
{ "result": "Atom-indexed SMILES: [CH3:0][CH2:1][OH:2]" }
```

For tools that implement `_batch()` (currently `analyze_properties`), `/call_batch` routes all SMILES through a single native batch call — far more efficient than N individual `/call` requests. Other tools process items sequentially in one threadpool slot.

---

## Tool catalog

Source of truth: `TOOL_SERVER_PORTS` in `__init__.py`. The port column is the default
(`MOLKIT_TOOL_PORT_BASE` = 10000); setting that variable shifts all five together.

| Tool name | Port | Class | What it does |
|-----------|------|-------|--------------|
| `analyze_properties` | 10000 | `MolPropAnalyzer` | Physicochemical properties (MW, logP, HBD, HBA, TPSA, rotatable bonds, ring count, QED, MR, heavy-atom count, formal charge) via RDKit, plus ADMET (logD, logS, BBBP, Mutag, HIA) via `admet_ai` when an ADMET backend is reachable. Optional `property_names` restricts output. Supports `_batch()`. |
| `match_substructure` | 10001 | `SubstructureMatch` | Tests whether a SMARTS or SMILES substructure query occurs in a molecule. `query_type` (`auto`/`smarts`/`smiles`) controls parsing; `auto` tries SMILES first, then SMARTS. |
| `label_atom_indices` | 10002 | `AtomIndexLabeler` | Returns a mapped SMILES with each atom annotated as `[sym:index]` (e.g. `[CH3:0][CH2:1][OH:2]`). Use this to find the atom indices for `edit_fragment`'s `anchors`. |
| `edit_fragment` | 10003 | `EditFragment` | Attach, swap, or remove a substituent via a matched-molecular-pair transform `from_smiles -> to_smiles`. Both sides mark attachment point(s) with a mapped dummy `[*:1]` (`[*:2]`/`[*:3]` for double/triple cuts). **attach** = `from='[*:1]'`; **swap** = `from='[*:1]A', to='[*:1]B'`; **remove** = `to='[*:1][H]'`. Pin the site (and, for multi-cut, the orientation) with `anchors = {label: atom_index}`. |
| `suggest_edits` | 10004 | `SuggestEdits` | Given a molecule and a target property box `constraints={prop: [lo, hi]}`, rank the mmpdb move set (built by `0_build_mmpdb`, via `molkit/utils/suggest_edits.py`) by predicted reduction in normalised (z-score) distance to the box, and return up to `top_k` candidates as `{from_smiles, to_smiles, anchors, predicted_gap, delta}` — each directly usable as `edit_fragment` args. `delta` carries the predicted `{avg, std}` change for exactly the properties named in `constraints`, in that order, including the ones the edit leaves unchanged (Δ 0). Reads the move set from `$MMP_MOVES_DIR`. |

### `edit_fragment` in one glance

```
attach   from='[*:1]'      to='[*:1]C(=O)O'   anchors={"1":0}      c1ccccc1  -> O=C(O)c1ccccc1
swap     from='[*:1]Cl'    to='[*:1]OC'                            Clc1ccccc1 -> COc1ccccc1
remove   from='[*:1]C'     to='[*:1][H]'                           Cc1ccccc1  -> c1ccccc1
```

`anchors` labels are the `[*:N]` numbers; atom indices come from `label_atom_indices`. Omit `anchors` to apply at every matching site (each distinct product returned).

---

## Server

### Starting

Every tool is a FastAPI app; `tool_server.py` launches them. One process per tool:

```bash
python -m molkit.tools.tool_server --all
```

or a single tool:

```bash
python -m molkit.tools.tool_server --tool analyze_properties
python -m molkit.tools.tool_server --tool analyze_properties --port 10000 --workers 4
```

Ports come from `TOOL_SERVER_PORTS`, which is anchored at `MOLKIT_TOOL_PORT_BASE`
(default `10000`, so the five tools listen on `10000`-`10004` in the order of the table
above). Server and client both read that table, so moving the base moves them together —
useful when another process already holds those ports:

```bash
MOLKIT_TOOL_PORT_BASE=9000 python -m molkit.tools.tool_server --all
```

`--workers` above 1 is honoured only for the tools listed in `MULTIWORKER_TOOLS`
(CPU/RDKit-bound, no GPU); everything else is forced to a single worker.

### Health check

```bash
curl http://localhost:10000/health   # → {"status": "ok", "tool": "analyze_properties"}
```

### ADMET backends (`analyze_properties`)

`analyze_properties` reaches ADMET-AI backends for logD/logS/BBBP/Mutag/HIA via
`utils/molmim.py` `predict_admet`. Backend URLs come from the `ADMET_SERVER_URLS`
environment variable (comma-separated `http://host:port` entries), so point it at whatever
ADMET-AI deployment you have:

```bash
ADMET_SERVER_URLS=http://localhost:8500,http://localhost:8501 \
  python -m molkit.tools.tool_server --tool analyze_properties
```

If no ADMET backend is reachable, RDKit-only properties are returned and the ADMET fields
are omitted (graceful degradation), so the pipeline still runs without one.
