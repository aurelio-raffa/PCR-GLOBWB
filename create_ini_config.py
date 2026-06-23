#!/usr/bin/env python3
"""Root CLI shim for rendering a PCR-GLOBWB INI from a template.

The implementation moved to ``src/utils/ini_config.py`` so the pipeline stage
(``src/stages/create_ini.py``, Fire-wrapped) and this command-line tool share a single implementation.
This shim only re-exposes the original ``python create_ini_config.py ...`` argparse interface (which writes a
timestamped file in the current directory; the pipeline stage passes an explicit ``output_path`` instead).

    python create_ini_config.py --name run01 --base_ini config/05min_parallel.ini \
        --outputDir /scratch/out --cloneMap '.../clonemap_%s.map' --inputDir /data/inputs
"""
from src.utils.ini_config import parser, create_ini_config

if __name__ == '__main__':
    create_ini_config(**vars(parser.parse_args()))
