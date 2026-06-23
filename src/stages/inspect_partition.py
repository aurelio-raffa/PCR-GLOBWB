"""Stage (optional / standalone) -- inspect a clone partition: tile summary + colour-coded image.

Wraps ``src.utils.partition_summary.inspect_partition``. Run it as a pipeline stage (e.g. right after
``compute_basins``) to record the summary + image for the partition you just built, or standalone to inspect
a third party's partition:

    python src/stages/inspect_partition.py --partition foo.npz --output_image foo.png
    python src/stages/inspect_partition.py --landmask_dir their_clone_maps --extents their_extents.csv
"""
from __init__ import root_path  # noqa: F401  -- runs src/stages/__init__.py so `import src...` resolves
from fire import Fire

from src.utils.partition_summary import inspect_partition


if __name__ == '__main__':
    Fire(inspect_partition)
