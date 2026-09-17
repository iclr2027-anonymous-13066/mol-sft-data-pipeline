"""molkit/tools/analysis.py  —  Analysis category

Tools
-----
MolPropAnalyzer    : physicochemical + ADMET properties from SMILES (analyze_properties)
SubstructureMatch  : whether a SMARTS/SMILES substructure occurs in a molecule
"""

from __future__ import annotations

import logging
import math
import os
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors, rdMolDescriptors

from molkit.tools.base import BaseTool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fragment name mapping  (RDKit fr_* <-> human-readable common name)
#
# Single source of truth: molkit.utils.fragments, which DERIVES the names from
# RDKit's own fragment descriptions using the same policy the benchmark scorer
# uses. Re-exported here under the historical private aliases. Do not hand-write
# a table in this file again -- it drifts from the grader.
# ---------------------------------------------------------------------------
from molkit.utils.fragments import (  # noqa: E402
    COMMON_NAME_TO_FR as _COMMON_NAME_TO_FR,
    FR_COMMON_NAMES as _FR_COMMON_NAMES,
)


# ---------------------------------------------------------------------------
# SubstructureMatch
# ---------------------------------------------------------------------------

class SubstructureMatch(BaseTool):
    """Test whether a SMARTS/SMILES substructure query occurs in a molecule."""

    name = "match_substructure"
    description = (
        "Check whether a substructure query is present in a molecule. The query "
        "may be a SMARTS pattern or a SMILES fragment; the molecule is given as "
        "SMILES. Returns whether the substructure occurs in the molecule. "
        "Pass a LIST of queries to check several at once — "
        "each is matched on its own and 'match' is true only if ALL of them are "
        "present, which is what you want when a task requires more than one group."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": ["string", "array"],
                "items": {"type": "string"},
                "description": (
                    "Substructure pattern to search for, as SMARTS or SMILES "
                    "(e.g. 'c1ccccc1', 'C(=O)O', '[#7]', '[CX3](=O)[OX2H1]'). "
                    "Give a LIST to require several patterns at once; each is matched "
                    "separately, so do NOT join them with '.' (a dot-joined query "
                    "demands atom-disjoint matches and would reject a molecule whose "
                    "two required groups share an atom)."
                ),
            },
            "mol_smiles": {
                "type": "string",
                "description": "SMILES string of the full molecule to search within.",
            },
            "query_type": {
                "type": "string",
                "enum": ["auto", "smarts", "smiles"],
                "description": (
                    "How to interpret 'query'. 'auto' (default) parses it as SMILES "
                    "first, then falls back to SMARTS. Use 'smarts' or 'smiles' to "
                    "force a single interpretation."
                ),
            },
        },
        "required": ["query", "mol_smiles"],
    }

    examples = [
        {
            "input": {"query": "c1ccccc1", "mol_smiles": "c1ccc(O)cc1"},
            "output": "The substructure 'c1ccccc1' is present in c1ccc(O)cc1.",
        },
        {
            "input": {"query": "[CX3](=O)[OX2H1]", "mol_smiles": "CC(=O)O"},
            "output": "The substructure '[CX3](=O)[OX2H1]' is present in CC(=O)O.",
        },
        {
            "input": {"query": "C(=O)O", "mol_smiles": "CCO"},
            "output": "The substructure 'C(=O)O' is not present in CCO.",
        },
        {
            "input": {"query": ["c1ccccc1", "C(=O)-N"], "mol_smiles": "CC(=O)Nc1ccccc1"},
            "output": "All 2 substructures are present in CC(=O)Nc1ccccc1.",
        },
    ]

    @staticmethod
    def _parse_query(query: str, query_type: str):
        """Return (query_mol, interpretation) — query_mol is None if unparseable."""
        if query_type == "smarts":
            return Chem.MolFromSmarts(query), "SMARTS"
        if query_type == "smiles":
            return Chem.MolFromSmiles(query), "SMILES"
        # auto: SMILES first (sanitized → correct aromaticity/implicit-H matching),
        # then fall back to SMARTS for query-only syntax like '[#7]' or '[CX3]'.
        mol = Chem.MolFromSmiles(query)
        if mol is not None:
            return mol, "SMILES"
        return Chem.MolFromSmarts(query), "SMARTS"

    def _analyse(self, query, mol_smiles: str, query_type: str = "auto") -> str | list:
        """Validate the inputs and run every query.

        Returns an error STRING, or the internal per-query detail list
        (``query`` / ``match`` / ``num_matches`` / ``query_interpreted_as``).
        That detail never leaves the tool: the observable result is the match
        verdict alone (see :meth:`execute`). The count and the parsed-as kind are
        deliberately withheld — they are not what the tool was asked, and putting
        a number in the observation invites the model to reason about it.
        """
        # A list means "all of these must be present". Each pattern is matched on its
        # own — never concatenated — because a dot-joined query requires the parts to
        # map to ATOM-DISJOINT matches and would miss a molecule whose amide nitrogen
        # is also its piperazine nitrogen.
        if isinstance(query, str):
            queries = [query]
        elif isinstance(query, (list, tuple)):
            queries = list(query)
        else:
            return "Input Argument Error: 'query' must be a string or a list of strings."
        if not queries:
            return "Input Argument Error: 'query' must not be empty."
        for q in queries:
            if not q or not isinstance(q, str):
                return "Input Argument Error: every 'query' entry must be a non-empty string."
        if not mol_smiles or not isinstance(mol_smiles, str):
            return "Input Argument Error: 'mol_smiles' must be a non-empty string."
        if query_type not in ("auto", "smarts", "smiles"):
            return "Input Argument Error: 'query_type' must be one of 'auto', 'smarts', 'smiles'."

        mol = Chem.MolFromSmiles(mol_smiles)
        if mol is None:
            return f"SMILES Syntax Error: Invalid SMILES '{mol_smiles}'"

        per_query = []
        for q in queries:
            try:
                query_mol, used_type = self._parse_query(q, query_type)
            except Exception as e:
                return f"Execution Error: Failed to parse query '{q}'. {e}"
            if query_mol is None:
                kind = {"smarts": "SMARTS", "smiles": "SMILES"}.get(
                    query_type, "SMARTS or SMILES")
                return f"Query Syntax Error: Could not parse query '{q}' as {kind}."
            try:
                matches = mol.GetSubstructMatches(query_mol, uniquify=True)
            except Exception as e:
                return f"Execution Error: Substructure search failed. {e}"
            per_query.append({
                "query": q,
                "match": len(matches) > 0,
                "num_matches": len(matches),
                "query_interpreted_as": used_type,
            })

        return per_query

    def execute(self, query, mol_smiles: str, query_type: str = "auto") -> str | dict:
        per_query = self._analyse(query, mol_smiles, query_type)
        if isinstance(per_query, str):
            return per_query
        # One key, single or list query alike. ALL must be present: a query
        # requiring two groups is not satisfied by one.
        return {"match": all(r["match"] for r in per_query)}

    def __call__(self, inputs: dict, return_text: bool = True) -> str | dict:
        per_query = self._analyse(**inputs)
        if isinstance(per_query, str):
            return per_query
        matched = all(r["match"] for r in per_query)
        if not return_text:
            return {"match": matched}
        s = inputs.get("mol_smiles", "")
        if len(per_query) == 1:
            q = per_query[0]["query"]
            if matched:
                return f"The substructure '{q}' is present in {s}."
            return f"The substructure '{q}' is not present in {s}."
        if matched:
            return f"All {len(per_query)} substructures are present in {s}."
        # Name the missing ones: "not all present" on its own is not actionable.
        missing = [r["query"] for r in per_query if not r["match"]]
        return (f"Not all substructures are present in {s}. "
                f"Missing: {', '.join(repr(m) for m in missing)}.")

# ---------------------------------------------------------------------------
# MolPropAnalyzer
# ---------------------------------------------------------------------------


# Thread-safe LRU cache for _rdkit_properties.  The toolchain builder
# queries the same molecule from multiple code paths (constraint check,
# scoring, verification) so cache hits save ~5-10 ms of redundant RDKit
# computation per call.  Keyed on canonical SMILES.
_RDKIT_PROP_CACHE_SIZE = int(os.environ.get("RDKIT_PROP_CACHE_SIZE", "4096"))

# Dedicated executor for overlapping the (I/O + GPU-bound) ADMET prediction
# with the (CPU-bound) RDKit descriptor computation inside ``_batch``.  The
# two are independent given the SMILES list, so running them concurrently
# turns each batch's wall time from ``rdkit + admet`` into ``max(rdkit, admet)``
# and shortens how long every request holds its server threadpool slot —
# directly improving concurrent throughput.  Threads here only block on the
# ADMET round-trip (fan-out across GPU backends is delegated to molmim's own
# dispatch executor), so a generous pool is cheap.  Sized to match the tool
# server threadpool so every concurrent batch can keep an ADMET call in flight.
# Must be a SEPARATE pool from molmim's dispatch executor to avoid deadlock
# (the overlap thread submits to that pool and blocks on its result).
_ADMET_OVERLAP_EXECUTOR = ThreadPoolExecutor(
    max_workers=int(os.environ.get("ADMET_OVERLAP_WORKERS", "256")),
    thread_name_prefix="admet-overlap",
)


def _is_nan(v) -> bool:
    try:
        return math.isnan(float(v)) or math.isinf(float(v))
    except (TypeError, ValueError):
        return True


def _safe_round(v, ndigits: int = 3):
    """Round a numeric value; return None if NaN/Inf (JSON-safe)."""
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return None
        return round(f, ndigits)
    except (TypeError, ValueError):
        return None
_rdkit_prop_cache: dict[str, dict] = {}
_rdkit_prop_cache_lock = threading.Lock()


def _rdkit_properties(mol: Chem.Mol) -> dict:
    """Compute RDKit physicochemical properties for a sanitized molecule.

    Results are cached by canonical SMILES (thread-safe, bounded LRU).
    """
    from rdkit.Chem import QED

    smiles = Chem.MolToSmiles(mol)

    # Fast cache lookup (no lock needed for reads on CPython dict).
    cached = _rdkit_prop_cache.get(smiles)
    if cached is not None:
        return dict(cached)  # return a copy to prevent mutation

    ring_info   = mol.GetRingInfo()
    rings_total = ring_info.NumRings()
    heavy_atoms = mol.GetNumHeavyAtoms()
    # formal_charge = sum(a.GetFormalCharge() for a in mol.GetAtoms())

    props = {
        "MW":           round(rdMolDescriptors.CalcExactMolWt(mol), 3),
        "logP":         round(Descriptors.MolLogP(mol), 3),
        "HBD":          rdMolDescriptors.CalcNumHBD(mol),
        "HBA":          rdMolDescriptors.CalcNumHBA(mol),
        "TPSA":         round(rdMolDescriptors.CalcTPSA(mol), 3),
        "rotB":         rdMolDescriptors.CalcNumRotatableBonds(mol),
        "rings_total":  rings_total,
        "QED":          round(QED.qed(mol), 3),
        "MR":           round(Descriptors.MolMR(mol), 3),
        "heavy_atoms":  heavy_atoms,
        # "formal_charge": formal_charge,
    }

    with _rdkit_prop_cache_lock:
        if len(_rdkit_prop_cache) >= _RDKIT_PROP_CACHE_SIZE:
            # Evict oldest ~25% to amortise eviction cost.
            evict_n = _RDKIT_PROP_CACHE_SIZE // 4
            for _ in range(evict_n):
                _rdkit_prop_cache.pop(next(iter(_rdkit_prop_cache)), None)
        _rdkit_prop_cache[smiles] = dict(props)

    return props


class MolPropAnalyzer(BaseTool):
    """Compute physicochemical + ADMET properties for a molecule from its SMILES.

    RDKit properties (always available):
        MW, logP, HBD, HBA, TPSA, rotB, rings_total, QED, MR,
        heavy_atoms, formal_charge

    ADMET properties (predicted via admet_ai; included when requested or when
    no subset is given, computed only if an ADMET backend is reachable):
        logD, logS, BBBP, Mutag, HIA

    By default every property is returned; pass ``property_names`` to
    restrict the output to a chosen subset.
    """

    # Canonical RDKit property names, in display order.
    RDKIT_PROPERTY_NAMES = [
        "MW", "logP", "HBD", "HBA", "TPSA", "rotB",
        "rings_total", "QED", "MR", "heavy_atoms", # "formal_charge",
    ]
    # ADMET property names (predicted by admet_ai), in display order.
    ADMET_PROPERTY_NAMES = [
        "logD", "logS", "BBBP", "Mutag", # "HIA"
    ]

    # All property names, in display order.  Also the allowed values for the
    # ``property_names`` argument.
    PROPERTY_NAMES = RDKIT_PROPERTY_NAMES + ADMET_PROPERTY_NAMES

    name = "analyze_properties"
    description = (
        "Compute physicochemical and ADMET properties for a molecule given its "
        "SMILES. Available properties: " + ", ".join(PROPERTY_NAMES) + ". "
        "The ADMET properties (" + ", ".join(ADMET_PROPERTY_NAMES) + ") are "
        "predicted by admet_ai and require an ADMET backend to be running. "
        "By default all are returned; pass 'property_names' to return only the "
        "specified subset."
    )
    parameters = {
        "type": "object",
        "properties": {
            "mol_smiles": {
                "type": "string",
                "description": "SMILES string of the molecule to analyze.",
            },
            "property_names": {
                "type": "array",
                "items": {"type": "string", "enum": PROPERTY_NAMES},
                "description": (
                    "Optional subset of properties to return. When omitted, all "
                    "properties are returned. Valid names: "
                    + ", ".join(PROPERTY_NAMES) + "."
                ),
            },
        },
        "required": ["mol_smiles"],
    }

    # examples = [
    #     {
    #         "input": {"mol_smiles": "c1ccc(cc1)Oc1cccc(c1)c1nc2c([nH]1)cccc2"},
    #         "output": '{"MW": 286.111, "logP": 5.022, "HBD": 1, "HBA": 2, "TPSA": 37.91, "rotB": 3, "rings_total": 4, "QED": 0.574, "MR": 88.046, "heavy_atoms": 22, "formal_charge": 0, "logD": 4.504, "logS": -6.584, "BBBP": 0.709, "Mutag": 0.74, "HIA": 1.0}',
    #     },
    #     {
    #         "input": {"mol_smiles": "CCO", "property_names": ["MW", "logP", "QED"]},
    #         "output": '{"MW": 46.042, "logP": -0.001, "QED": 0.407}',
    #     },
    #     {
    #         "input": {"mol_smiles": "CCO", "property_names": ["logD", "logS", "BBBP", "Mutag", "HIA"]},
    #         "output": '{"logD": -0.025, "logS": 1.263, "BBBP": 0.972, "Mutag": 0.044, "HIA": 0.997}',
    #     },
    # ]
    examples = [
        {
            "input": {"mol_smiles": "c1ccc(cc1)Oc1cccc(c1)c1nc2c([nH]1)cccc2"},
            "output": '{"MW": 286.111, "logP": 5.022, "HBD": 1, "HBA": 2, "TPSA": 37.91, "rotB": 3, "rings_total": 4, "QED": 0.574, "MR": 88.046, "heavy_atoms": 22, "logD": 4.504, "logS": -6.584, "BBBP": 0.709, "Mutag": 0.74}',
        },
        {
            "input": {"mol_smiles": "CCO", "property_names": ["MW", "logP", "QED"]},
            "output": '{"MW": 46.042, "logP": -0.001, "QED": 0.407}',
        },
        {
            "input": {"mol_smiles": "CCO", "property_names": ["logD", "logS", "BBBP", "Mutag"]},
            "output": '{"logD": -0.025, "logS": 1.263, "BBBP": 0.972, "Mutag": 0.044}',
        },
    ]

    @classmethod
    def _validate_property_names(cls, property_names):
        """Return an error string if *property_names* is invalid, else None."""
        if property_names is None:
            return None
        if (not isinstance(property_names, list)
                or not all(isinstance(p, str) for p in property_names)):
            return "Input Argument Error: 'property_names' must be a list of strings."
        unknown = [p for p in property_names if p not in cls.PROPERTY_NAMES]
        if unknown:
            return (
                f"Input Argument Error: unknown property name(s) {unknown}. "
                f"Valid names: {cls.PROPERTY_NAMES}."
            )
        return None

    @classmethod
    def _select(cls, props: dict, property_names) -> dict:
        """Return only the requested properties (in request order); all if None."""
        if not property_names:
            return props
        return {p: props[p] for p in property_names if p in props}

    @classmethod
    def _wants_admet(cls, property_names) -> bool:
        """True if ADMET prediction is needed for this request.

        ADMET is computed when no subset is given (return everything) or when
        the requested subset includes at least one ADMET property.
        """
        if not property_names:
            return True
        return any(p in cls.ADMET_PROPERTY_NAMES for p in property_names)

    @classmethod
    def _admet_properties(cls, smiles_list: list[str]) -> dict[str, dict]:
        """Batch ADMET prediction → {smiles: {logD, logS, BBBP, Mutag}}.

        Routes through :func:`molkit.utils.molmim.predict_admet` so this call
        shares the process-wide ADMETBatcher (one ``model.predict`` per batching
        window) and, when ``ADMET_SERVER_URLS`` is set, distributes work across
        the local ADMET GPU servers.  Returns ``{}`` (graceful degradation) if no
        ADMET backend is reachable.
        """
        if not smiles_list:
            return {}
        try:
            from molkit.utils.molmim import COLUMN_MAP, predict_admet
            df = predict_admet(smiles_list)
            if df is None or df.empty:
                return {}
            result: dict[str, dict] = {}
            for smi in smiles_list:
                if smi not in df.index:
                    continue
                row = df.loc[smi]
                result[smi] = {
                    name: _safe_round(row.get(COLUMN_MAP[name]))
                    for name in cls.ADMET_PROPERTY_NAMES
                }
            return result
        except Exception as e:
            logger.warning("ADMET batch prediction failed: %s", e)
            return {}

    def execute(self, mol_smiles: str, property_names=None) -> str | dict:
        if not mol_smiles or not isinstance(mol_smiles, str):
            return "Input Argument Error: 'mol_smiles' must be a non-empty string."
        err = self._validate_property_names(property_names)
        if err:
            return err
        mol = Chem.MolFromSmiles(mol_smiles)
        if mol is None:
            return f"SMILES Syntax Error: Invalid SMILES '{mol_smiles}'"
        try:
            props = _rdkit_properties(mol)
        except Exception as e:
            return f"Execution Error: Property calculation failed. {e}"

        if self._wants_admet(property_names):
            admet = self._admet_properties([mol_smiles])
            if mol_smiles in admet:
                props.update(admet[mol_smiles])

        return self._select(props, property_names)

    def _batch(self, smiles_list: list[str], property_names=None) -> dict[str, dict | str]:
        """Process a list of SMILES; returns {smiles: props_dict | error_str}.

        Computes RDKit descriptors (cached by canonical SMILES).  When
        ``property_names`` is given, each result dict is restricted to the
        requested properties; the full set is still cached for reuse.
        """
        results: dict[str, dict | str] = {}
        valid: list[str] = []          # parsed & uncached → need RDKit descriptors
        valid_for_admet: list[str] = []  # all parsed SMILES (cache-hits included)
        mols: dict[str, Chem.Mol] = {}

        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi) if smi else None
            if mol is None:
                results[smi] = f"SMILES Syntax Error: Invalid SMILES '{smi}'"
            else:
                valid_for_admet.append(smi)
                # Check cache first.
                cached = _rdkit_prop_cache.get(Chem.MolToSmiles(mol))
                if cached is not None:
                    results[smi] = dict(cached)
                else:
                    valid.append(smi)
                    mols[smi] = mol

        # Kick off ADMET (I/O + GPU-bound) in the background NOW, so it overlaps
        # with the CPU-bound RDKit descriptor loop below.  ADMET only needs the
        # validated SMILES strings — it is independent of the RDKit results — so
        # the two run concurrently and the batch's wall time becomes
        # max(rdkit, admet) instead of their sum.  See _ADMET_OVERLAP_EXECUTOR.
        admet_future = None
        if valid_for_admet and self._wants_admet(property_names):
            admet_future = _ADMET_OVERLAP_EXECUTOR.submit(
                self._admet_properties, valid_for_admet,
            )

        # RDKit descriptors (pure CPU, fast).
        from rdkit.Chem import QED
        for smi in valid:
            mol = mols[smi]
            ring_info = mol.GetRingInfo()
            results[smi] = {
                "MW":           round(rdMolDescriptors.CalcExactMolWt(mol), 3),
                "logP":         round(Descriptors.MolLogP(mol), 3),
                "HBD":          rdMolDescriptors.CalcNumHBD(mol),
                "HBA":          rdMolDescriptors.CalcNumHBA(mol),
                "TPSA":         round(rdMolDescriptors.CalcTPSA(mol), 3),
                "rotB":         rdMolDescriptors.CalcNumRotatableBonds(mol),
                "rings_total":  ring_info.NumRings(),
                "QED":          round(QED.qed(mol), 3),
                "MR":           round(Descriptors.MolMR(mol), 3),
                "heavy_atoms":  mol.GetNumHeavyAtoms(),
                # "formal_charge": sum(a.GetFormalCharge() for a in mol.GetAtoms()),
            }

        # Cache the full property dicts (before any property_names filtering).
        for smi in valid:
            csmi = Chem.MolToSmiles(mols[smi])
            with _rdkit_prop_cache_lock:
                if len(_rdkit_prop_cache) >= _RDKIT_PROP_CACHE_SIZE:
                    evict_n = _RDKIT_PROP_CACHE_SIZE // 4
                    for _ in range(evict_n):
                        _rdkit_prop_cache.pop(next(iter(_rdkit_prop_cache)), None)
                _rdkit_prop_cache[csmi] = dict(results[smi])

        # Join the overlapped ADMET prediction and merge into the result dicts
        # (both freshly-computed and cache-hit, since the RDKit cache holds
        # RDKit-only dicts).  ADMET values are not written back to the RDKit
        # cache.  _admet_properties already degrades gracefully to {} on error.
        if admet_future is not None:
            try:
                admet = admet_future.result()
            except Exception as e:  # pragma: no cover - defensive
                logger.warning("ADMET batch prediction failed: %s", e)
                admet = {}
            for smi, vals in admet.items():
                if isinstance(results.get(smi), dict):
                    results[smi].update(vals)

        # Restrict each result to the requested properties.
        if property_names:
            for smi, entry in results.items():
                if isinstance(entry, dict):
                    results[smi] = self._select(entry, property_names)

        return results

    @classmethod
    def _format_text(cls, smi: str, props: dict) -> str:
        """Render whichever properties are present as a single sentence."""
        labels = {
            "MW":            "a molecular weight of {} Da",
            "logP":          "a logP of {}",
            "HBD":           "{} hydrogen-bond donors",
            "HBA":           "{} hydrogen-bond acceptors",
            "TPSA":          "a topological polar surface area (TPSA) of {} Å²",
            "rotB":          "{} rotatable bonds",
            "rings_total":   "{} total rings",
            "QED":           "a quantitative estimate of drug-likeness (QED) of {}",
            "MR":            "a molar refractivity (MR) of {}",
            "heavy_atoms":   "{} heavy atoms",
            # "formal_charge": "a formal charge of {}",
            "logD":          "a predicted logD of {}",
            "logS":          "a predicted aqueous solubility (logS) of {}",
            "BBBP":          "a predicted blood-brain barrier penetration (BBBP) probability of {}",
            "Mutag":         "a predicted mutagenicity (AMES) probability of {}",
            # "HIA":           "a predicted human intestinal absorption (HIA) probability of {}",
        }
        parts = [
            labels[k].format(props[k])
            for k in cls.PROPERTY_NAMES
            if k in props and props[k] is not None
        ]
        if not parts:
            return f"No properties were computed for the molecule {smi}."
        if len(parts) == 1:
            body = parts[0]
        elif len(parts) == 2:
            body = f"{parts[0]} and {parts[1]}"
        else:
            body = ", ".join(parts[:-1]) + ", and " + parts[-1]
        return f"The molecule {smi} has {body}."

    def __call__(self, inputs: dict, return_text: bool = True) -> str | dict:
        # Batch mode — used by evaluate_benchmark.py
        if "mol_smiles_list" in inputs:
            smiles_list = inputs["mol_smiles_list"]
            if not isinstance(smiles_list, list):
                return "Input Argument Error: 'mol_smiles_list' must be a list."
            property_names = inputs.get("property_names")
            err = self._validate_property_names(property_names)
            if err:
                return err
            batch = self._batch(smiles_list, property_names)
            if return_text:
                lines = []
                for smi in smiles_list:
                    entry = batch.get(smi)
                    if isinstance(entry, dict):
                        lines.append(self._format_text(smi, entry))
                    else:
                        lines.append(f"The molecule {smi} is invalid and cannot be analyzed.")
                return "\n".join(lines)
            return batch

        # Single-molecule mode
        result = self.execute(**inputs)
        if isinstance(result, str):
            return result
        if return_text:
            return self._format_text(inputs.get("mol_smiles", ""), result)
        return result
