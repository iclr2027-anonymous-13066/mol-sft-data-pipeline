"""molkit/utils/errors.py  —  Exception hierarchy for molkit."""

from __future__ import annotations


class MolKitError(Exception):
    """Base exception for all molkit errors."""


class MolKitValueError(MolKitError, ValueError):
    """Raised when an argument has an invalid value."""


class MolKitSMILESError(MolKitValueError):
    """Raised when a SMILES string is invalid or cannot be parsed."""


class MolKitToolError(MolKitError):
    """Raised when a tool call fails at runtime."""


class MolKitAPIError(MolKitError):
    """Raised when an external API call fails (PubChem, Wikipedia, etc.)."""


class MolKitConfigError(MolKitError):
    """Raised when required configuration or environment variables are missing."""


class MolKitNotImplementedError(MolKitError, NotImplementedError):
    """Raised when a tool or feature is not yet implemented."""
