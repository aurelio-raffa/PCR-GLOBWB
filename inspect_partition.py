#!/usr/bin/env python3
"""Root CLI shim: inspect a clone partition -- validate the inputs, compute statistics, and plot the partition.

The implementation lives in ``src/utils/partition_summary.py`` (shared with the pipeline stage
``src/stages/inspect_partition.py``). Three input modes:

    # a directory of per-tile clone/landmask maps (.map and/or .nc); validates the convention, derives each
    # bounding box from the file header, reconstructs the partition, and reports stats + an image:
    python inspect_partition.py --maps_dir clone_landmask_maps/20260508_partition \
        --clone_pattern 'clone_%s.map' --output_summary partition_summary.txt --output_image partition.png
    #   (add --landmask_pattern 'landmask_%s.map' when per-tile landmasks are present)

    # a partition NPZ produced by compute_ldd_basins:
    python inspect_partition.py --partition partition.npz --output_image partition.png

    # an extents CSV only (bounding-box-only summary):
    python inspect_partition.py --extents tile_extents.csv
"""
import argparse

from src.utils.partition_summary import inspect_partition, CELL_SIZE


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # directory mode
    p.add_argument('--maps_dir', default=None,
                   help='Directory of per-tile clone/landmask maps (.map and/or .nc)')
    p.add_argument('--clone_pattern', default='clone_%s.map',
                   help='Clone-map filename pattern (must contain %%s). Default: clone_%%s.map')
    p.add_argument('--landmask_pattern', default=None,
                   help='Landmask filename pattern (must contain %%s); preferred over the clone map when present')
    p.add_argument('--clone_areas', default='auto',
                   help='Which codes: "auto" (glob the pattern), or a CSV like "M01,M02". Default: auto')
    p.add_argument('--cell_size', type=float, default=CELL_SIZE,
                   help='Expected cell size in degrees (default: 5 arcmin = 1/12)')
    p.add_argument('--no_validate', action='store_true', help='Skip the grid/cellsize validation checks')
    # NPZ / extents modes
    p.add_argument('--partition', default=None, help='Partition NPZ from compute_ldd_basins')
    p.add_argument('--extents', default=None, help='Extents CSV (bounding-box-only summary)')
    # outputs
    p.add_argument('--output_summary', default=None, help='Write the text summary/validation report here')
    p.add_argument('--output_image', default=None, help='Write the colour-coded partition PNG here')
    p.add_argument('--label', default=None, help='Title/label for the summary and image')
    p.add_argument('--no_annotate', action='store_true',
                   help='Do not annotate the image with per-tile cell counts')
    return p


if __name__ == '__main__':
    args = build_parser().parse_args()
    inspect_partition(
        partition=args.partition, extents=args.extents, maps_dir=args.maps_dir,
        clone_pattern=args.clone_pattern, landmask_pattern=args.landmask_pattern,
        clone_areas=args.clone_areas, cell_size=args.cell_size, validate=not args.no_validate,
        output_summary=args.output_summary, output_image=args.output_image,
        label=args.label, annotate=not args.no_annotate,
    )
