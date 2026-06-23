"""Stage package: makes the repo root importable for every stage subprocess.

Each stage is run by the orchestrator (``src/stages/run.py``) as its own process via ``mlflow.run`` with the
working directory set to the repo root, e.g. ``python src/stages/setup.py --key value``. Because Python puts
the *script's* directory (``src/stages``) on ``sys.path[0]``, ``from __init__ import root_path`` resolves to
this file; appending ``os.getcwd()`` (the repo root) then makes ``import src...`` work as well.
"""
import os
import sys

sys.path.append(os.getcwd())

from src import root_path, console_handler  # noqa: F401  (re-exported for stages that `from __init__ import ...`)
