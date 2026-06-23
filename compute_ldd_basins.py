#!/usr/bin/env python3
"""Root CLI shim for LDD-based domain decomposition.

The implementation moved to ``src/utils/ldd_basins.py`` so the pipeline stage
(``src/stages/compute_basins.py``, Fire-wrapped) and this command-line tool share a single implementation.
This shim only re-exposes the original ``python compute_ldd_basins.py ...`` argparse interface.

    python compute_ldd_basins.py --ldd <ldd.map> --n_tiles 53 --ub_cells 100000 \
        --lb_cells 500 --output_extents tile_extents.csv --output_partition partition.npz
"""
from src.utils.ldd_basins import build_parser, compute_ldd_basins

if __name__ == '__main__':
    compute_ldd_basins(**vars(build_parser().parse_args()))
