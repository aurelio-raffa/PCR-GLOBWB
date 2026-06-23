#!/usr/bin/env python3
"""Root CLI shim for per-tile clone & landmask map generation.

The implementation moved to ``src/utils/tile_clone_maps.py`` so the pipeline stage
(``src/stages/compute_basins.py``, Fire-wrapped) and this command-line tool share a single implementation.
This shim only re-exposes the original ``python create_tile_clone_maps.py ...`` argparse interface.

    python create_tile_clone_maps.py --global_clone clone_global_05min.map \
        --output_dir /inputs/cloneMaps --extents tile_extents.csv [--partition partition.npz]
"""
from src.utils.tile_clone_maps import build_parser, create_tile_clone_maps

if __name__ == '__main__':
    create_tile_clone_maps(**vars(build_parser().parse_args()))
