"""Stage (final, optional) -- plot a PCR-GLOBWB netCDF output variable.

Wraps ``src.utils.plot_output.plot_output``. As a pipeline stage it runs after ``run_model`` to render a
figure (map / animation / time series) of a chosen output variable; it can also be run standalone on any
PCR-GLOBWB netCDF file:

    python src/stages/plot_output.py --nc_path .../netcdf/discharge_monthAvg_output.nc --variable discharge
"""
from __init__ import root_path  # noqa: F401  -- runs src/stages/__init__.py so `import src...` resolves
from fire import Fire

from src.utils.plot_output import plot_output


if __name__ == '__main__':
    Fire(plot_output)
