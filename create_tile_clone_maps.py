"""
Tile Clone Map Generator for PCR-GLOBWB Parallel Execution

Generates per-tile PCRaster clone maps (e.g. clonemap_M01.map ... clonemap_M53.map)
from the global 05min clone map, for use with the parallel runner infrastructure.

Background
----------
The parallel glue runner substitutes each clone code into the cloneMap and landmask
paths at runtime:

    cloneMap = /inputs/cloneMaps/global_parallelization/clonemap_%s.map
             -> /inputs/cloneMaps/global_parallelization/clonemap_M01.map  (tile M01)

Each generated file is a PCRaster boolean clone map covering one tile's bounding box
at 5-arcmin resolution, with all cells set to active (TRUE).  The landmask (also
required per tile) can be produced by the same script via --also_landmask once you
have a global landmask available.

Tile Extents
------------
The M01-M53 bounding boxes are NOT defined in this repository.  They are part of the
PCR-GLOBWB input data package distributed by Utrecht University:

    general/cloneMaps/global_parallelization/mask_M<n>.map

If you do not have those files yet, run this script WITHOUT --extents to print a
blank template CSV.  Fill in the extents (obtainable from Utrecht or from existing
mask_M*.map headers via `gdalinfo`) and re-run with --extents <file>.

All extents must snap to the 5-arcmin grid (i.e. be exact multiples of 1/12 degree
away from -180 / -90).

Usage
-----
Print blank template:
    python create_tile_clone_maps.py \\
        --global_clone clone_landmask_maps/clone_landmask_examples/clone_global_05min.map \\
        --output_dir /inputs/cloneMaps/global_parallelization

Generate maps from filled-in CSV:
    python create_tile_clone_maps.py \\
        --global_clone clone_landmask_maps/clone_landmask_examples/clone_global_05min.map \\
        --output_dir /inputs/cloneMaps/global_parallelization \\
        --extents tile_extents.csv

Use a custom filename pattern (must match cloneMap in your INI):
    python create_tile_clone_maps.py ... --filename_pattern 'mask_%s.map'
"""

import argparse
import csv
import os
import sys

import numpy as np


# ---------------------------------------------------------------------------
# Constants matching the global 05min PCRaster clone map
# ---------------------------------------------------------------------------

CELL_SIZE   = 1.0 / 12.0   # 5 arcminutes expressed in decimal degrees
GLOBAL_XMIN = -180.0
GLOBAL_YMAX =   90.0


# ---------------------------------------------------------------------------
# I/O helpers
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
    # CSF v2 header layout (all little-endian):
    #   offset 0:   signature (64 bytes)
    #   offset 108: cell size X (double, 8 bytes)
    import struct
    with open(path, 'rb') as f:
        raw = f.read(128)
    cell_size_x = struct.unpack_from('<d', raw, 108)[0]
    return cell_size_x


def write_clone_map(pcr, output_path: str, xmin: float, ymin: float,
                    xmax: float, ymax: float, cellsize: float) -> None:
    """Write a PCRaster boolean clone map covering [xmin,xmax] x [ymin,ymax]."""
    ncols = int(round((xmax - xmin) / cellsize))
    nrows = int(round((ymax - ymin) / cellsize))
    ymax_map = ymin + nrows * cellsize   # recompute to avoid fp drift

    if ncols <= 0 or nrows <= 0:
        sys.exit(
            f"Invalid extent for {output_path}: "
            f"({xmin},{ymin})-({xmax},{ymax}) gives {ncols}x{nrows} cells"
        )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # setclone(nrRows, nrCols, cellSize, westernBoundary, northernBoundary)
    pcr.setclone(nrows, ncols, cellsize, xmin, ymax_map)
    clone_map = pcr.spatial(pcr.boolean(1))   # all cells TRUE
    pcr.report(clone_map, output_path)


# ---------------------------------------------------------------------------
# Extents CSV helpers
# ---------------------------------------------------------------------------

def print_template_csv() -> None:
    """Print a blank extents CSV template header to stdout and exit."""
    lines = [
        "# Tile extents for PCR-GLOBWB 05min parallelization",
        "#",
        "# Columns (decimal degrees, WGS84 / EPSG:4326):",
        "#   code  : clone code, e.g. M01  (any unique string without commas)",
        "#   xmin  : western  boundary (longitude)",
        "#   ymin  : southern boundary (latitude)",
        "#   xmax  : eastern  boundary (longitude)",
        "#   ymax  : northern boundary (latitude)",
        "#",
        "# All values must lie on the 5-arcmin grid:",
        "#   (value - (-180)) must be a multiple of 1/12 degree",
        "#",
        "# To generate extents automatically from the LDD map, use:",
        "#   python compute_ldd_basins.py --ldd <ldd.map> --n_tiles N --output_extents <file.csv>",
        "#",
        "code,xmin,ymin,xmax,ymax",
        "# M01,<xmin>,<ymin>,<xmax>,<ymax>",
        "# M02,<xmin>,<ymin>,<xmax>,<ymax>",
        "# ...",
        "# Save this file, fill in the extents, then re-run with --extents <file>",
    ]
    print("\n".join(lines))


def load_extents(csv_path: str) -> dict:
    """Load tile extents from a CSV file with columns: code,xmin,ymin,xmax,ymax."""
    extents = {}
    with open(csv_path) as f:
        reader = csv.DictReader(row for row in f if not row.lstrip().startswith('#'))
        for row in reader:
            code = row['code'].strip()
            try:
                extents[code] = (
                    float(row['xmin']),
                    float(row['ymin']),
                    float(row['xmax']),
                    float(row['ymax']),
                )
            except (ValueError, KeyError) as exc:
                sys.exit(f"Cannot parse extents for {code}: {exc}")
    return extents


def validate_extents(extents: dict, cellsize: float) -> None:
    """Raise SystemExit if any extent is off-grid or degenerate."""
    eps = cellsize * 1e-4
    for code, (xmin, ymin, xmax, ymax) in extents.items():
        for name, val in (('xmin', xmin), ('ymin', ymin), ('xmax', xmax), ('ymax', ymax)):
            remainder = (val - GLOBAL_XMIN) % cellsize
            if remainder > eps and abs(remainder - cellsize) > eps:
                print(
                    f"WARNING: {code} {name}={val} does not lie on the "
                    f"5-arcmin grid (remainder={remainder:.6f})"
                )
        if xmin >= xmax or ymin >= ymax:
            sys.exit(f"{code}: degenerate extent xmin={xmin} xmax={xmax} ymin={ymin} ymax={ymax}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--global_clone',
        required=True,
        help='Path to the global 05min PCRaster clone map '
             '(clone_landmask_maps/clone_landmask_examples/clone_global_05min.map)',
    )
    parser.add_argument(
        '--output_dir',
        required=True,
        help='Directory where the per-tile clone maps will be written',
    )
    parser.add_argument(
        '--extents',
        default=None,
        help='CSV file with columns code,xmin,ymin,xmax,ymax.  '
             'If omitted, a blank template CSV is printed to stdout and the '
             'script exits without generating any files.',
    )
    parser.add_argument(
        '--filename_pattern',
        default='clonemap_%s.map',
        help='Output filename pattern. Must contain %%s for the clone code.  '
             'Must match the cloneMap path pattern in your INI template.  '
             'Default: clonemap_%%s.map',
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if '%s' not in args.filename_pattern:
        sys.exit("--filename_pattern must contain %s (e.g. 'clonemap_%s.map')")

    if not os.path.isfile(args.global_clone):
        sys.exit(f"Global clone map not found: {args.global_clone}")

    # If no extents file provided, print template and exit
    if args.extents is None:
        print_template_csv()
        print(
            "\n# Template printed.  Fill in the extents and re-run with --extents <file>.",
            file=sys.stderr,
        )
        sys.exit(0)

    cellsize = read_global_clone_cellsize(args.global_clone)
    if abs(cellsize - CELL_SIZE) > CELL_SIZE * 1e-4:
        print(
            f"WARNING: global clone map cell size {cellsize:.10f} differs from "
            f"expected {CELL_SIZE:.10f}. Proceeding with actual cell size.",
            file=sys.stderr,
        )

    extents = load_extents(args.extents)
    validate_extents(extents, cellsize)

    pcr = _require_pcraster()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Generating {len(extents)} clone maps in {args.output_dir} ...")
    for code in sorted(extents.keys()):
        xmin, ymin, xmax, ymax = extents[code]
        filename = args.filename_pattern % code
        output_path = os.path.join(args.output_dir, filename)

        ncols = int(round((xmax - xmin) / cellsize))
        nrows = int(round((ymax - ymin) / cellsize))

        write_clone_map(pcr, output_path, xmin, ymin, xmax, ymax, cellsize)
        print(f"  {code}: lon [{xmin:8.3f}, {xmax:8.3f}]  "
              f"lat [{ymin:7.3f}, {ymax:7.3f}]  "
              f"{ncols:4d}x{nrows:4d}  ->  {filename}")

    print(f"\nDone. {len(extents)} files written.")


if __name__ == '__main__':
    main()
