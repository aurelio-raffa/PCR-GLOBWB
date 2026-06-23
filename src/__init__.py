"""MLflow pipeline harness for PCR-GLOBWB (vendored & trimmed from github.com/aurelio-raffa/plumber).

This package provides a thin, reproducible orchestration layer on top of the existing PCR-GLOBWB
command-line tools so that a whole run -- output-directory setup, (optional) LDD basin/tile computation,
INI instantiation and the model launch -- can be tracked as a single MLflow experiment and submitted as
one LSF job. See PIPELINE.md at the repo root for the full picture.
"""
import os
import sys
import logging

# the repo root is the directory that contains this `src/` package; add it to sys.path so stage
# subprocesses (run from `src/stages/`) can `import src...` and `from __init__ import root_path`.
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)
logging.basicConfig(filename=os.path.join(root_path, 'output.log'), level=logging.INFO)

# console handler shared by the orchestrator and stages so their logs also reach stderr
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")
console_handler.setFormatter(formatter)
