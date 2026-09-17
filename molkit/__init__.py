"""molkit — molecular tooling shared by the data-generation stages.

Two subpackages are used by the pipeline:

* ``molkit.utils``  — SMILES/scaffold handling, fragment catalogs, and the
  mmpdb-derived ``suggest_edits`` move ranker.
* ``molkit.tools``  — the tool implementations and their schemas
  (``TOOL_REGISTRY``), served over HTTP by ``molkit.tools.tool_server`` or
  called in-process by the toolchain builder.
"""
