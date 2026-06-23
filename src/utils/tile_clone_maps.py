"""
Tile Clone Map and Landmask Generator for PCR-GLOBWB Parallel Execution

Generates per-tile PCRaster clone maps and landmasks for use with the
parallel runner infrastructure.

Background
----------
The parallel glue runner substitutes each clone code into the cloneMap and
landmask paths at runtime:

    cloneMap = /inputs/cloneMaps/global_parallelization/clonemap_%s.map
             -> /inputs/cloneMaps/global_parallelization/clonemap_M01.map

    landmask = /inputs/cloneMaps/global_parallelization/landmask_%s.map
             -> /inputs/cloneMaps/global_parallelization/landmask_M01.map

Clone maps are boolean PCRaster maps covering the tile's bounding box with
all cells set to TRUE.  PCRaster requires a rectangle for setclone(), so the
bounding box is unavoidable.

Landmasks are boolean PCRaster maps of the same bounding box, but TRUE only
for cells that belong to that tile's partition.  Without a correct landmask
the model simulates every cell in the bounding box, wasting compute on cells
that belong to neighbouring tiles.

Generating landmasks requires the partition NPZ produced by compute_ldd_basins:

    python compute_ldd_basins.py ... --output_partition partition.npz

Usage
-----
Clone maps only (from extents CSV):
    python create_tile_clone_maps.py \\
        --global_clone clone_global_05min.map \\
        --output_dir /inputs/cloneMaps/global_parallelization \\
        --extents tile_extents.csv

Clone maps + landmasks (from partition NPZ):
    python create_tile_clone_maps.py \\
        --global_clone clone_global_05min.map \\
        --output_dir /inputs/cloneMaps/global_parallelization \\
        --partition partition.npz \\
        --landmask_pattern 'landmask_%s.map'

Print blank extents template:
    python create_tile_clone_maps.py \\
        --global_clone clone_global_05min.map \\
        --output_dir /inputs/cloneMaps/global_parallelization
"""

import argparse
import csv
import os
import sys
from types import SimpleNamespace

import numpy as np

# ---------------------------------------------------------------------------
# Constants matching the global 05min PCRaster clone map
# ---------------------------------------------------------------------------

CELL_SIZE = 1.0 / 12.0  # 5 arcminutes expressed in decimal degrees
GLOBAL_XMIN = -180.0
GLOBAL_YMAX = 90.0


# ---------------------------------------------------------------------------
# PCRaster helpers
# ---------------------------------------------------------------------------

def _require_pcraster():
    try:
        import pcraster as pcr
        return pcr
    except ImportError:
        sys.exit(
            "PCRaster Python bindings not found.\n"
            "Activate the conda environment that includes the model dependencies "
            "and re-run this script."
        )


def read_global_clone_cellsize(path: str) -> float:
    """Read cell size from a PCRaster CSF header without importing pcraster."""
    import struct
    with open(path, 'rb') as f:
        raw = f.read(128)
    return struct.unpack_from('<d', raw, 108)[0]


def grid_aligned_nw_corner(xmin: float, ymax: float,
                           cellsize: float) -> tuple[float, float]:
    """
    Snap a tile's north-west corner onto the global grid.

    The clone must be a pixel-exact window of the global map: gdalwarpPCR and
    isSameClone align the source to the clone via its xUL/yUL/cellsize, so any
    sub-cell offset between the tile grid and the global grid lets the regridder
    drop cells to missing value (see known_issues.txt, 28 July 2014 — clone
    corners must coincide with the global grid).  Anchoring west/north to the
    global origin by an integer number of cells guarantees that coincidence,
    instead of accumulating float drift from ymin + nrows * cellsize.
    """
    west = GLOBAL_XMIN + round((xmin - GLOBAL_XMIN) / cellsize) * cellsize
    north = GLOBAL_YMAX - round((GLOBAL_YMAX - ymax) / cellsize) * cellsize
    return west, north


def write_clone_map(pcr, output_path: str, xmin: float, ymin: float,
                    xmax: float, ymax: float, cellsize: float) -> None:
    """Write a PCRaster boolean clone map (all TRUE) for [xmin,xmax] x [ymin,ymax]."""
    ncols = int(round((xmax - xmin) / cellsize))
    nrows = int(round((ymax - ymin) / cellsize))
    if ncols <= 0 or nrows <= 0:
        sys.exit(
            f"Invalid extent for {output_path}: "
            f"({xmin},{ymin})-({xmax},{ymax}) gives {ncols}x{nrows} cells"
        )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    west, north = grid_aligned_nw_corner(xmin, ymax, cellsize)
    # setclone(nrRows, nrCols, cellSize, westernBoundary, northernBoundary)
    pcr.setclone(nrows, ncols, cellsize, west, north)
    pcr.report(pcr.spatial(pcr.boolean(1)), output_path)


def write_landmask(pcr, output_path: str,
                   xmin: float, ymin: float, xmax: float, ymax: float,
                   cellsize: float,
                   tile_map: np.ndarray,
                   tile_idx: int,
                   global_xmin: float, global_ymax: float) -> int:
    """
    Write a per-tile PCRaster landmask (TRUE where tile_map == tile_idx).

    tile_map    : global (nrows_global, ncols_global) int16 array from the
                  partition NPZ; value = tile index (0-based), -1 ocean,
                  -2 filtered.
    tile_idx    : 0-based index of this tile in the partition.
    global_xmin / global_ymax : reference corner of tile_map.

    Returns the number of active (TRUE) cells written.
    """
    ncols = int(round((xmax - xmin) / cellsize))
    nrows = int(round((ymax - ymin) / cellsize))
    if ncols <= 0 or nrows <= 0:
        sys.exit(f"Degenerate extent for landmask {output_path}")

    # Row/col offsets of this tile within the global tile_map
    nrows_global, ncols_global = tile_map.shape
    col0 = int(round((xmin - global_xmin) / cellsize))
    row0 = int(round((global_ymax - ymax) / cellsize))
    col1 = col0 + ncols
    row1 = row0 + nrows

    # Clamp to global grid bounds (should be exact, but guard against fp drift)
    col0 = max(0, col0);
    col1 = min(ncols_global, col1)
    row0 = max(0, row0);
    row1 = min(nrows_global, row1)

    tile_slice = tile_map[row0:row1, col0:col1]

    active = (tile_slice == tile_idx).astype(np.uint8)
    n_active = int(active.sum())

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Set clone to this tile's extent (NW corner anchored to the global grid so
    # the landmask, the clone, and every regridded input share one grid), then
    # report the boolean mask
    west, north = grid_aligned_nw_corner(xmin, ymax, cellsize)
    pcr.setclone(nrows, ncols, cellsize, west, north)
    # pcr2numpy / numpy2pcr path: build boolean map from numpy
    active_pcr = pcr.numpy2pcr(pcr.Boolean, active.astype(float), -1)
    pcr.report(active_pcr, output_path)

    return n_active


# ---------------------------------------------------------------------------
# Extents CSV helpers
# ---------------------------------------------------------------------------

def print_template_csv() -> None:
    lines = [
        "# Tile extents for PCR-GLOBWB 05min parallelization",
        "#",
        "# Columns (decimal degrees, WGS84 / EPSG:4326):",
        "#   code  : clone code, e.g. M01",
        "#   xmin  : western  boundary (longitude)",
        "#   ymin  : southern boundary (latitude)",
        "#   xmax  : eastern  boundary (longitude)",
        "#   ymax  : northern boundary (latitude)",
        "#",
        "# Generate extents automatically with:",
        "#   python compute_ldd_basins.py --ldd <ldd.map> --n_tiles N "
        "--output_extents <file.csv> --output_partition <file.npz>",
        "#",
        "code,xmin,ymin,xmax,ymax",
    ]
    print("\n".join(lines))


def load_extents(csv_path: str) -> dict:
    extents = {}
    with open(csv_path) as f:
        reader = csv.DictReader(row for row in f if not row.lstrip().startswith('#'))
        for row in reader:
            code = row['code'].strip()
            try:
                extents[code] = (
                    float(row['xmin']), float(row['ymin']),
                    float(row['xmax']), float(row['ymax']),
                )
            except (ValueError, KeyError) as exc:
                sys.exit(f"Cannot parse extents for {code}: {exc}")
    return extents


def validate_extents(extents: dict, cellsize: float) -> None:
    eps = cellsize * 1e-4
    for code, (xmin, ymin, xmax, ymax) in extents.items():
        for name, val in (('xmin', xmin), ('ymin', ymin),
                          ('xmax', xmax), ('ymax', ymax)):
            remainder = (val - GLOBAL_XMIN) % cellsize
            if remainder > eps and abs(remainder - cellsize) > eps:
                print(f"WARNING: {code} {name}={val} not on 5-arcmin grid "
                      f"(remainder={remainder:.6f})")
        if xmin >= xmax or ymin >= ymax:
            sys.exit(f"{code}: degenerate extent")


# ---------------------------------------------------------------------------
# Partition NPZ helpers
# ---------------------------------------------------------------------------

def load_partition(npz_path: str) -> dict:
    """Load the partition NPZ written by compute_ldd_basins --output_partition."""
    d = np.load(npz_path, allow_pickle=True)
    return {
        'tile_map': d['tile_map'],  # (nrows, ncols) int16
        'codes': list(d['codes']),  # ['M01', 'M02', ...]
        'xmin': d['xmin'],
        'ymin': d['ymin'],
        'xmax': d['xmax'],
        'ymax': d['ymax'],
        'cell_size': float(d['cell_size']),
        'global_xmin': float(d['global_xmin']),
        'global_ymax': float(d['global_ymax']),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--global_clone', required=True,
        help='Path to the global 05min PCRaster clone map.',
    )
    parser.add_argument(
        '--output_dir', required=True,
        help='Directory where per-tile maps will be written.',
    )
    parser.add_argument(
        '--extents', default=None,
        help='CSV file with columns code,xmin,ymin,xmax,ymax.  '
             'Required for clone-map generation; if omitted a blank template '
             'is printed and the script exits.',
    )
    parser.add_argument(
        '--filename_pattern', default='clonemap_%s.map',
        help='Clone map filename pattern (must contain %%s for the clone code). '
             'Default: clonemap_%%s.map',
    )
    parser.add_argument(
        '--partition', default=None, metavar='NPZ',
        help='Partition NPZ file produced by compute_ldd_basins '
             '--output_partition.  When provided, per-tile landmask maps are '
             'generated alongside the clone maps.',
    )
    parser.add_argument(
        '--landmask_pattern', default='landmask_%s.map',
        help='Landmask filename pattern (must contain %%s for the clone code). '
             'Only used when --partition is given.  '
             'Default: landmask_%%s.map',
    )
    return parser


def create_tile_clone_maps(global_clone, output_dir, extents=None,
                           filename_pattern='clonemap_%s.map', partition=None,
                           landmask_pattern='landmask_%s.map') -> None:
    """Per-tile clone (+landmask) map generation (importable form of the original ``main``).

    Parameters mirror the CLI flags; the original body is preserved verbatim by binding them into an ``args``
    namespace. Driven by the root shim ``create_tile_clone_maps.py`` (argparse) and the pipeline stage
    ``src/stages/compute_basins.py`` (Fire). As in the original, ``extents`` (the CSV) is required to write
    any maps; ``partition`` is additionally required to cut per-tile landmasks.
    """
    args = SimpleNamespace(
        global_clone=global_clone, output_dir=output_dir, extents=extents,
        filename_pattern=filename_pattern, partition=partition, landmask_pattern=landmask_pattern,
    )

    if '%s' not in args.filename_pattern:
        sys.exit("--filename_pattern must contain %s (e.g. 'clonemap_%s.map')")

    if not os.path.isfile(args.global_clone):
        sys.exit(f"Global clone map not found: {args.global_clone}")

    # No extents file -> print template and exit
    if args.extents is None:
        print_template_csv()
        print("\n# Template printed.  Fill in extents and re-run with --extents <file>.",
              file=sys.stderr)
        sys.exit(0)

    cellsize = read_global_clone_cellsize(args.global_clone)
    if abs(cellsize - CELL_SIZE) > CELL_SIZE * 1e-4:
        print(f"WARNING: global clone cell size {cellsize:.10f} differs from "
              f"expected {CELL_SIZE:.10f}. Proceeding with actual value.",
              file=sys.stderr)

    extents = load_extents(args.extents)
    validate_extents(extents, cellsize)

    # Load partition NPZ if provided
    partition = None
    if args.partition is not None:
        if '%s' not in args.landmask_pattern:
            sys.exit("--landmask_pattern must contain %s (e.g. 'landmask_%s.map')")
        if not os.path.isfile(args.partition):
            sys.exit(f"Partition NPZ not found: {args.partition}")
        partition = load_partition(args.partition)
        # Validate that all extents codes are present in the partition
        missing = set(extents) - set(partition['codes'])
        if missing:
            sys.exit(f"Codes in extents CSV not found in partition NPZ: {missing}")

    pcr = _require_pcraster()
    os.makedirs(args.output_dir, exist_ok=True)

    n_clones = len(extents)
    n_landmask = 0

    print(f"Generating {n_clones} clone map(s)"
          + (" + landmask(s)" if partition else "")
          + f" in {args.output_dir} ...")

    for code in sorted(extents.keys()):
        xmin, ymin, xmax, ymax = extents[code]
        ncols_tile = int(round((xmax - xmin) / cellsize))
        nrows_tile = int(round((ymax - ymin) / cellsize))

        clone_path = os.path.join(args.output_dir, args.filename_pattern % code)
        write_clone_map(pcr, clone_path, xmin, ymin, xmax, ymax, cellsize)

        suffix = f"  {ncols_tile:4d}x{nrows_tile:4d}  ->  {os.path.basename(clone_path)}"

        if partition is not None:
            tile_idx = partition['codes'].index(code)
            lm_path = os.path.join(args.output_dir, args.landmask_pattern % code)
            n_active = write_landmask(
                pcr, lm_path,
                xmin, ymin, xmax, ymax, cellsize,
                partition['tile_map'], tile_idx,
                partition['global_xmin'], partition['global_ymax'],
            )
            if n_active == 0:
                sys.exit(
                    f"ERROR: landmask for {code} (tile_idx={tile_idx}) has 0 active "
                    f"cells. The extents CSV and the partition NPZ are most likely "
                    f"from different compute_ldd_basins runs (code present but its "
                    f"cells were filtered/reindexed). Regenerate both from a single "
                    f"run. An all-FALSE landmask makes PCR-GLOBWB simulate an empty "
                    f"year and then crash with ZeroDivisionError on totalCellArea."
                )
            fill_pct = 100.0 * n_active / (ncols_tile * nrows_tile)
            suffix += f"  +  {os.path.basename(lm_path)}  ({n_active:,} active, {fill_pct:.1f}% fill)"
            n_landmask += 1

        print(f"  {code}: lon [{xmin:8.3f}, {xmax:8.3f}]  lat [{ymin:7.3f}, {ymax:7.3f}]"
              + suffix)

    print(f"\nDone.  {n_clones} clone map(s)"
          + (f", {n_landmask} landmask(s)" if n_landmask else "")
          + " written.")


# The CLI lives in the root-level shim ``create_tile_clone_maps.py`` (argparse) and the pipeline stage
# ``src/stages/compute_basins.py`` (Fire); this module is import-only.
