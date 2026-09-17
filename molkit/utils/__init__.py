"""molkit/utils  —  Shared utility helpers."""

from molkit.utils.errors import (
    MolKitError,
    MolKitValueError,
    MolKitSMILESError,
    MolKitToolError,
    MolKitAPIError,
    MolKitConfigError,
    MolKitNotImplementedError,
)
from molkit.utils.smiles import (
    is_smiles,
    is_multiple_smiles,
    split_smiles,
    largest_mol,
    tanimoto,
    calc_rotatable_bonds,
    calc_ring_counts,
    calc_heavy_atoms_and_charge,
    calc_fsp3,
    calc_hetero_atom_count,
    calc_topological_diameter,
    calc_molar_refractivity,
)
from molkit.utils.canonicalize import (
    canonicalize_molecule_smiles,
    get_molecule_id,
)
from molkit.utils.fragments import (
    fr_catalog,
    build_fragment_catalog,
    analyze_fragments,
    results_to_text,
    list_available_fragments,
    FR_COMMON_NAMES,
    COMMON_NAME_TO_FR,
    FR_DISPLAY_NAME_OVERRIDES,
    FR_PRETTY_DISPLAY_NAMES,
)
from molkit.utils.parquet_search import search_partitioned_parquet
from molkit.utils.global_tools import get_global_tools, get_tool_schemas
from molkit.utils.molmim import (
    COLUMN_MAP,
    SUPPORTED_CONSTRAINT_KEYS,
    predict_admet,
    evaluate_smiles_batch,
)

__all__ = [
    # errors
    "MolKitError",
    "MolKitValueError",
    "MolKitSMILESError",
    "MolKitToolError",
    "MolKitAPIError",
    "MolKitConfigError",
    "MolKitNotImplementedError",
    # smiles
    "is_smiles",
    "is_multiple_smiles",
    "split_smiles",
    "largest_mol",
    "tanimoto",
    "calc_rotatable_bonds",
    "calc_ring_counts",
    "calc_heavy_atoms_and_charge",
    "calc_fsp3",
    "calc_hetero_atom_count",
    "calc_topological_diameter",
    "calc_molar_refractivity",
    # canonicalize
    "canonicalize_molecule_smiles",
    "get_molecule_id",
    # fragments
    "fr_catalog",
    "build_fragment_catalog",
    "analyze_fragments",
    "results_to_text",
    "list_available_fragments",
    "FR_COMMON_NAMES",
    "COMMON_NAME_TO_FR",
    "FR_DISPLAY_NAME_OVERRIDES",
    "FR_PRETTY_DISPLAY_NAMES",
    # parquet
    "search_partitioned_parquet",
    # global tools
    "get_global_tools",
    "get_tool_schemas",
    # admet / property model (molkit.utils.molmim)
    "COLUMN_MAP",
    "SUPPORTED_CONSTRAINT_KEYS",
    "predict_admet",
    "evaluate_smiles_batch",
]
