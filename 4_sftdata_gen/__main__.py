"""Entry point for ``python -m 4_sftdata_gen`` (run from project root).

Runs the training-data generation pipeline. Tool-chain construction has been
moved to the sibling package ``3_toolchain_gen`` and should be invoked
separately via ``python -m 3_toolchain_gen``.
"""

from .main import main

main()
