"""Stage 2 (optional) -- compute LDD basins/tiles and the per-tile clone & landmask maps.

Thin wrapper over the two existing root tools:

  1. ``compute_ldd_basins.py``     -- LDD map -> tile extents CSV + partition NPZ (+ optional image/dendrogram)
  2. ``create_tile_clone_maps.py`` -- partition/extents + global clone map -> ``clonemap_%s.map`` (and, from
                                      the partition, ``landmask_%s.map``) under ``clone_maps_dir``

The ``%s`` clone/landmask maps it produces are exactly what the parallel runner expects in the INI's
``cloneMap``/``landmask`` fields. Make this stage optional by setting ``skip: true`` on it in the pipeline
YAML (e.g. when the tiles already exist from a previous run).

All paths are resolved against the repo root (the stages run with the repo root as the working directory).
"""
import os

from fire import Fire

from __init__ import root_path
from src.utils.shell import run_command, python_tool


def _opt(flag, value):
    """Return ``[flag, value]`` when ``value`` is set, else ``[]`` (for argparse-optional flags)."""
    if value is None or value == '':
        return []
    return [flag, str(value)]


def compute_basins(
        ldd_map: str,
        n_tiles,
        output_extents: str,
        output_partition: str,
        global_clone: str,
        clone_maps_dir: str,
        ub_cells=None,
        lb_cells=None,
        snap_cellsize=None,
        output_image: str = None,
        output_dendrogram: str = None,
        make_landmask: bool = True,
        landmask_pattern: str = None,
) -> None:
    """Compute the domain decomposition and write the per-tile clone/landmask maps.

    Args:
        ldd_map: Path to the PCRaster LDD map (``--ldd``).
        n_tiles: Target number of tiles after aggregation (``--n_tiles``).
        output_extents: Output tile-extents CSV (``--output_extents``).
        output_partition: Output partition ``.npz`` (``--output_partition``); needed for landmasks.
        global_clone: Global PCRaster clone map fed to ``create_tile_clone_maps.py`` (``--global_clone``).
        clone_maps_dir: Output directory for the per-tile ``clonemap_%s.map`` / ``landmask_%s.map``.
        ub_cells, lb_cells, snap_cellsize: Optional split/filter/snap controls forwarded to
            ``compute_ldd_basins.py`` when set.
        output_image, output_dendrogram: Optional diagnostic outputs forwarded when set.
        make_landmask: If True (default), build per-tile landmasks from the partition; otherwise only clone
            maps are written (from the extents CSV).
        landmask_pattern: Optional override for the landmask filename pattern (must contain ``%s``).
    """
    # 1) domain decomposition -------------------------------------------------------------------------------
    basins_cmd = python_tool(
        'compute_ldd_basins.py',
        '--ldd', ldd_map,
        '--n_tiles', str(n_tiles),
        '--output_extents', output_extents,
        '--output_partition', output_partition,
        *_opt('--ub_cells', ub_cells),
        *_opt('--lb_cells', lb_cells),
        *_opt('--snap_cellsize', snap_cellsize),
        *_opt('--output_image', output_image),
        *_opt('--output_dendrogram', output_dendrogram),
    )
    run_command(basins_cmd, cwd=root_path)

    # 2) per-tile clone (+landmask) maps -------------------------------------------------------------------
    os.makedirs(os.path.join(root_path, clone_maps_dir), exist_ok=True)
    if make_landmask:
        # the partition NPZ carries the tile_map needed to cut per-tile landmasks
        clone_cmd = python_tool(
            'create_tile_clone_maps.py',
            '--global_clone', global_clone,
            '--output_dir', clone_maps_dir,
            '--partition', output_partition,
            *_opt('--landmask_pattern', landmask_pattern),
        )
    else:
        clone_cmd = python_tool(
            'create_tile_clone_maps.py',
            '--global_clone', global_clone,
            '--output_dir', clone_maps_dir,
            '--extents', output_extents,
        )
    run_command(clone_cmd, cwd=root_path)


if __name__ == '__main__':
    Fire(compute_basins)
