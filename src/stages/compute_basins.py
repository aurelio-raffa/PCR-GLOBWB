"""Stage 2 (optional) -- compute LDD basins/tiles and the per-tile clone & landmask maps.

Calls the implementations in ``src/utils`` directly (no subprocess):

  1. ``src.utils.ldd_basins.compute_ldd_basins``     -- LDD map -> tile extents CSV + partition NPZ
  2. ``src.utils.tile_clone_maps.create_tile_clone_maps`` -- extents (+partition) + global clone map ->
                                                          ``clonemap_%s.map`` (and ``landmask_%s.map``)

The ``%s`` maps it produces are what the parallel runner expects in the INI's ``cloneMap``/``landmask`` fields.
Make this stage optional with ``skip: true`` in the pipeline YAML (e.g. when the tiles already exist).
"""
from __init__ import root_path  # noqa: F401  -- runs src/stages/__init__.py so `import src...` resolves
from fire import Fire

from src.utils.ldd_basins import compute_ldd_basins
from src.utils.tile_clone_maps import create_tile_clone_maps


def _int_or_none(value):
    return int(value) if value not in (None, '') else None


def compute_basins(
        ldd_map: str,
        n_tiles,
        output_extents: str,
        output_partition: str,
        global_clone: str,
        clone_maps_dir: str,
        ub_cells=None,
        lb_cells=None,
        snap_cellsize=0.5,
        output_image: str = None,
        output_dendrogram: str = None,
        make_landmask: bool = True,
        clone_pattern: str = 'clonemap_%s.map',
        landmask_pattern: str = 'landmask_%s.map',
) -> None:
    """Compute the domain decomposition and write the per-tile clone/landmask maps.

    Args mirror the two underlying tools (see their docstrings in ``src/utils``). ``make_landmask`` toggles
    whether per-tile landmasks are cut from the partition NPZ (recommended for parallel runs).
    """
    # 1) LDD decomposition -> tile extents CSV + partition NPZ
    compute_ldd_basins(
        ldd=ldd_map,
        n_tiles=int(n_tiles),
        output_extents=output_extents,
        output_partition=output_partition,
        ub_cells=_int_or_none(ub_cells),
        lb_cells=_int_or_none(lb_cells),
        snap_cellsize=float(snap_cellsize) if snap_cellsize not in (None, '') else 0.5,
        output_image=output_image or None,
        output_dendrogram=output_dendrogram or None,
    )

    # 2) per-tile clone (+landmask) maps. The extents CSV is always required to write any maps; the partition
    #    NPZ is additionally required to cut per-tile landmasks (passing only --partition would otherwise make
    #    the tool print a blank template and exit without writing anything).
    create_tile_clone_maps(
        global_clone=global_clone,
        output_dir=clone_maps_dir,
        extents=output_extents,
        filename_pattern=clone_pattern,
        partition=output_partition if make_landmask else None,
        landmask_pattern=landmask_pattern,
    )


if __name__ == '__main__':
    Fire(compute_basins)
