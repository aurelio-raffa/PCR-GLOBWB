#!/usr/bin/env python3
"""Root CLI shim: plot a PCR-GLOBWB netCDF output variable as a map, animation or time series.

The implementation lives in ``src/utils/plot_output.py`` (shared with the pipeline stage
``src/stages/plot_output.py``). This shim re-exposes the original command-line interface.

    python plot_simulation_output.py outputs/.../netcdf/discharge_monthAvg_output.nc
    python plot_simulation_output.py FILE.nc --date 2006-07-15
    python plot_simulation_output.py FILE.nc --bbox 5 20 40 50         # lonmin lonmax latmin latmax
    python plot_simulation_output.py FILE.nc --loc 12.5 45.4           # lon lat -> time series
"""
import argparse
import sys

from src.utils.plot_output import plot_output


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Plot a PCR-GLOBWB netCDF output variable as a map, "
                    "animation or time series.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("nc_path", help="path to the netCDF output file")
    p.add_argument("--var", default=None,
                   help="variable name (auto-detected if omitted)")
    p.add_argument("--bbox", type=float, nargs=4, default=None,
                   metavar=("LONMIN", "LONMAX", "LATMIN", "LATMAX"),
                   help="lon/lat bounding box (default: crop to data extent)")
    p.add_argument("--date", default=None,
                   help="plot one map for this date (nearest step is used)")
    p.add_argument("--loc", type=float, nargs=2, default=None,
                   metavar=("LON", "LAT"),
                   help="plot a time series at the nearest cell to this point")
    p.add_argument("--max-frames", type=int, default=None,
                   help="max gif frames (default: min(n_years, 20))")
    p.add_argument("--cmap", default="viridis", help="matplotlib colormap")
    p.add_argument("--log", action="store_true", help="logarithmic color scale")
    p.add_argument("--vmin", type=float, default=None, help="color scale minimum")
    p.add_argument("--vmax", type=float, default=None, help="color scale maximum")
    p.add_argument("--no-robust", action="store_true",
                   help="use full data min/max instead of 2-98 percentiles")
    p.add_argument("--no-coastlines", action="store_true",
                   help="never use cartopy even if it is installed")
    p.add_argument("--outdir", default="__auto__",
                   help="where to write the figures (default: next to the input file)")
    p.add_argument("--show", action="store_true",
                   help="display the figures interactively as well")
    p.add_argument("--fps", type=int, default=4, help="animation frames per second")
    p.add_argument("--dpi", type=int, default=110, help="output resolution")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        plot_output(
            args.nc_path, variable=args.var, bbox=args.bbox, date=args.date,
            location=args.loc, max_frames=args.max_frames, cmap=args.cmap,
            log=args.log, vmin=args.vmin, vmax=args.vmax, robust=not args.no_robust,
            coastlines=not args.no_coastlines, outdir=args.outdir, show=args.show,
            fps=args.fps, dpi=args.dpi,
        )
    except Exception as exc:
        sys.stderr.write("[plot] error: %s\n" % exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
