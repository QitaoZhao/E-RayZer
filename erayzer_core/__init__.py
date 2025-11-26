"""Localized E-RayZer core modules used for inference."""
from __future__ import annotations

import importlib
import sys

# Expose the vendored packages under their original import names so the
# checkpoint/config can keep referencing `model.*` and `utils.*`.
if "erayzer_core.model" not in sys.modules:
    importlib.import_module("erayzer_core.model")
if "erayzer_core.utils" not in sys.modules:
    importlib.import_module("erayzer_core.utils")

sys.modules.setdefault("model", sys.modules["erayzer_core.model"])
sys.modules.setdefault("utils", sys.modules["erayzer_core.utils"])
