#!/usr/bin/env python3
"""Root CLI shim: inspect a clone partition (tile summary + image), including third-party partitions.

The implementation lives in ``src/utils/partition_summary.py`` (shared with the pipeline stage
``src/stages/inspect_partition.py``).

    # a partition NPZ produced by compute_ldd_basins
    python inspect_partition.py --partition partition.npz --output_image partition.png

    # a third party's clone partition given as per-tile landmask maps + an extents CSV
    python inspect_partition.py --landmask_dir their_clone_maps --extents their_extents.csv \
        --output_summary their_summary.txt --output_image their_partition.png

    # an extents CSV only (bounding-box-only summary)
    python inspect_partition.py --extents tile_extents.csv
"""
import argparse

from src.utils.partition_summary import inspect_partition


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--partition', default=None, help='Partition NPZ from compute_ldd_basins')
    p.add_argument('--extents', default=None, help='Extents CSV (bbox-only, or paired with --landmask_dir)')
    p.add_argument('--landmask_dir', default=None,
                   help='Directory of third-party per-tile landmask maps (with --extents)')
    p.add_argument('--landmask_pattern', default='landmask_%s.map',
                   help='Landmask filename pattern (must contain %%s). Default: landmask_%%s.map')
    p.add_argument('--output_summary', default=None, help='Write the text tile summary to this path')
    p.add_argument('--output_image', default=None, help='Write the colour-coded partition PNG to this path')
    p.add_argument('--label', default=None, help='Title/label for the summary and image')
    p.add_argument('--no_annotate', action='store_true',
                   help='Do not annotate the image with per-tile cell counts')
    return p


if __name__ == '__main__':
    args = build_parser().parse_args()
    inspect_partition(
        partition=args.partition, extents=args.extents, landmask_dir=args.landmask_dir,
        landmask_pattern=args.landmask_pattern, output_summary=args.output_summary,
        output_image=args.output_image, label=args.label, annotate=not args.no_annotate,
    )
