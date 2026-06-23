"""PCR-GLOBWB output plotter (importable implementation).

Plot a variable from a PCR-GLOBWB netCDF output file as a geographic heat map, an animation, or a
single-cell time series. ``plot_output`` is the importable entry point used by the pipeline stage
(``src/stages/plot_output.py``) and the root CLI shim (``plot_simulation_output.py``).

The low-level matplotlib helpers (headless backend, colour limits, norm, transparent-NaN colormap, save)
come from ``src.utils.plotting`` so they are shared with the partition-inspection plotting (no duplication).

Dependencies
------------
Hard:  numpy, netCDF4, matplotlib (numpy + netCDF4 are already in the PCR-GLOBWB environment).
Optional, used only if importable (never required):
       * cartopy -> draws coastlines / borders; otherwise a plain lon/lat frame is used.
       * Pillow  -> writes the animated gif; otherwise individual PNG frames are written instead.
"""
import datetime
import os
import sys
import warnings

import numpy as np
import netCDF4 as nc

from .plotting import pyplot, color_limits, make_norm, transparent_nan_cmap, save_figure

# Optional coastlines. Detected once, never required.
try:
    import cartopy.crs as ccrs            # noqa: F401
    import cartopy.feature as cfeature    # noqa: F401
    HAS_CARTOPY = True
except Exception:
    HAS_CARTOPY = False

MV = 1.0e20            # PCR-GLOBWB missing value (see model/virtualOS.py)
_BIG = 1.0e19          # anything with |value| >= this is treated as missing


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _first(ds, names):
    """First variable name in *names* that exists in dataset *ds*."""
    for n in names:
        if n in ds.variables:
            return n
    return None


def _to_nan(arr):
    """Masked / sentinel values -> NaN, returned as float64 ndarray."""
    arr = np.ma.filled(np.ma.asarray(arr).astype("float64"), np.nan)
    arr[~np.isfinite(arr)] = np.nan
    arr[np.abs(arr) >= _BIG] = np.nan
    return arr


def _to_pydate(d):
    """cftime or datetime object -> plain ``datetime.datetime``."""
    return datetime.datetime(
        int(d.year), int(d.month), int(d.day),
        int(getattr(d, "hour", 0)), int(getattr(d, "minute", 0)),
        int(getattr(d, "second", 0)),
    )


def _parse_date(s):
    """Flexible date parsing -> ``datetime.datetime``."""
    if isinstance(s, (datetime.datetime, datetime.date)):
        return datetime.datetime(s.year, s.month, s.day)
    try:
        from dateutil import parser as dparser   # python-dateutil is in the env
        return dparser.parse(str(s))
    except Exception:
        return datetime.datetime.fromisoformat(str(s))


# ---------------------------------------------------------------------------
# reading the netCDF file
# ---------------------------------------------------------------------------
class NcField(object):
    """Lightweight container describing a (time, lat, lon) variable."""

    def __init__(self, ds, varname):
        self.ds = ds
        self.latname = _first(ds, ["lat", "latitude", "y", "Y"])
        self.lonname = _first(ds, ["lon", "longitude", "x", "X"])
        self.timename = _first(ds, ["time", "t", "Time"])
        if self.latname is None or self.lonname is None:
            raise ValueError("could not find latitude / longitude variables "
                             "in the file (looked for lat/latitude, lon/longitude)")

        self.varname = varname or self._detect_var()
        if self.varname not in ds.variables:
            raise ValueError("variable %r not found. Available data variables: %s"
                             % (self.varname, ", ".join(self.data_vars())))
        self.var = ds.variables[self.varname]
        self.dims = list(self.var.dimensions)
        self.has_time = self.timename in self.dims

        self.lat = np.asarray(ds.variables[self.latname][:], dtype="float64")
        self.lon = np.asarray(ds.variables[self.lonname][:], dtype="float64")

        self.units = getattr(self.var, "units", "")
        self.long_name = getattr(self.var, "long_name",
                                 getattr(self.var, "standard_name", self.varname))

        if self.has_time:
            tv = ds.variables[self.timename]
            self.tnum = np.atleast_1d(np.asarray(tv[:], dtype="float64"))
            self.tunits = getattr(tv, "units", "days since 1901-01-01")
            self.tcalendar = getattr(tv, "calendar", "standard")
            raw = np.atleast_1d(nc.num2date(self.tnum, self.tunits, self.tcalendar))
            self.dates = np.array([_to_pydate(d) for d in raw])
        else:
            self.tnum = None
            self.dates = None

    # -- variable discovery -------------------------------------------------
    def data_vars(self):
        coords = {self.latname, self.lonname, self.timename}
        return [v for v in self.ds.variables
                if v not in coords and self.ds.variables[v].ndim >= 2]

    def _detect_var(self):
        cands = self.data_vars()
        if not cands:
            raise ValueError("no 2-D/3-D data variable found in the file")
        # prefer a 3-D (time, lat, lon) variable
        three_d = [v for v in cands if self.ds.variables[v].ndim == 3]
        chosen = (three_d or cands)[0]
        if len(cands) > 1:
            sys.stderr.write("[plot] multiple data variables %s; using %r "
                             "(use variable= to override)\n" % (cands, chosen))
        return chosen

    # -- axis bookkeeping ---------------------------------------------------
    def _order(self):
        """Return the moveaxis order that yields (time, lat, lon) / (lat, lon)."""
        wants = ([self.timename] if self.has_time else []) + [self.latname, self.lonname]
        return [self.dims.index(w) for w in wants]

    # -- data access --------------------------------------------------------
    def cube(self):
        """Full field as float64 with NaNs, shape (time, lat, lon) or (lat, lon)."""
        data = _to_nan(self.var[:])
        return np.transpose(data, self._order())

    def slice_at(self, idx):
        """A single 2-D (lat, lon) field at time index *idx*."""
        if not self.has_time:
            return self._latlon(_to_nan(self.var[:]))
        # index the time axis wherever it sits
        taxis = self.dims.index(self.timename)
        sl = [slice(None)] * len(self.dims)
        sl[taxis] = int(idx)
        field = _to_nan(self.var[tuple(sl)])         # 2-D, axes are the non-time dims
        remaining = [d for d in self.dims if d != self.timename]
        return self._latlon(field, remaining)

    def series_at(self, i, j):
        """Time series (1-D) at lat index *i*, lon index *j*."""
        taxis = self.dims.index(self.timename)
        sl = [0] * len(self.dims)
        sl[taxis] = slice(None)
        sl[self.dims.index(self.latname)] = int(i)
        sl[self.dims.index(self.lonname)] = int(j)
        return _to_nan(self.var[tuple(sl)])

    def _latlon(self, field2d, dims=None):
        """Transpose a 2-D field to (lat, lon)."""
        dims = dims if dims is not None else [d for d in self.dims if d != self.timename]
        order = [dims.index(self.latname), dims.index(self.lonname)]
        return np.transpose(field2d, order)


# ---------------------------------------------------------------------------
# cropping
# ---------------------------------------------------------------------------
def _crop_indices(lat, lon, mask2d, bbox):
    """Return (row0, row1, col0, col1) slices.

    If *bbox* is given it is used directly; otherwise the extent is taken from
    the rows/columns of *mask2d* (True where data is available).
    """
    if bbox is not None:
        lonmin, lonmax, latmin, latmax = bbox
        rows = np.where((lat >= latmin) & (lat <= latmax))[0]
        cols = np.where((lon >= lonmin) & (lon <= lonmax))[0]
        if rows.size == 0 or cols.size == 0:
            raise ValueError("the requested bbox does not overlap the grid")
    else:
        if not mask2d.any():
            raise ValueError("the variable contains no valid data to plot")
        rows = np.where(mask2d.any(axis=1))[0]
        cols = np.where(mask2d.any(axis=0))[0]
    return rows.min(), rows.max() + 1, cols.min(), cols.max() + 1


# ---------------------------------------------------------------------------
# plotting primitives
# ---------------------------------------------------------------------------
def _setup_axes(fig, lon, lat, coastlines):
    """Create a map axis; use cartopy if available, else a plain lon/lat frame."""
    use_cp = coastlines and HAS_CARTOPY
    if use_cp:
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)
        ax.set_extent([float(lon.min()), float(lon.max()),
                       float(lat.min()), float(lat.max())], crs=ccrs.PlateCarree())
        gl = ax.gridlines(draw_labels=True, linewidth=0.2, alpha=0.4)
        gl.top_labels = False
        gl.right_labels = False
        return ax, ccrs.PlateCarree()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(float(lon.min()), float(lon.max()))
    ax.set_ylim(float(lat.min()), float(lat.max()))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal")
    return ax, None


def _draw(ax, transform, lon, lat, field, cmap, norm):
    kw = dict(cmap=cmap, norm=norm, shading="nearest")
    if transform is not None:
        kw["transform"] = transform
    return ax.pcolormesh(lon, lat, np.ma.masked_invalid(field), **kw)


def _map_figure(plt, lon, lat, field, *, title, cbar_label, cmap, vmin, vmax,
                log, coastlines, dpi):
    fig = plt.figure(figsize=(9, 6.5))
    ax, transform = _setup_axes(fig, lon, lat, coastlines)
    norm = make_norm(plt, vmin, vmax, log)
    mesh = _draw(ax, transform, lon, lat, field, transparent_nan_cmap(plt, cmap), norm)
    cb = fig.colorbar(mesh, ax=ax, shrink=0.75, pad=0.03)
    cb.set_label(cbar_label)
    ax.set_title(title)
    return fig


# ---------------------------------------------------------------------------
# the three output modes
# ---------------------------------------------------------------------------
def plot_maps(f, lon, lat, cube, *, cmap, vmin, vmax, robust, log, coastlines,
              base, outdir, show, dpi):
    """Mean + std maps for the whole record."""
    plt = pyplot(headless=not show)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(cube, axis=0)
        std = np.nanstd(cube, axis=0)

    lo, hi = color_limits(mean, vmin, vmax, robust, log)
    fig = _map_figure(plt, lon, lat, mean, title="%s - temporal mean" % f.long_name,
                      cbar_label=_label(f), cmap=cmap, vmin=lo, vmax=hi, log=log,
                      coastlines=coastlines, dpi=dpi)
    save_figure(plt, fig, _path(outdir, base, "mean"), dpi=dpi, show=show)

    slo, shi = color_limits(std, None, None, robust, False)
    fig = _map_figure(plt, lon, lat, std, title="%s - temporal std. dev." % f.long_name,
                      cbar_label=_label(f), cmap=cmap, vmin=slo, vmax=shi, log=False,
                      coastlines=coastlines, dpi=dpi)
    save_figure(plt, fig, _path(outdir, base, "std"), dpi=dpi, show=show)


def plot_single(f, lon, lat, field, when, *, cmap, vmin, vmax, robust, log,
                coastlines, base, outdir, show, dpi):
    """One map for a single time step."""
    plt = pyplot(headless=not show)
    lo, hi = color_limits(field, vmin, vmax, robust, log)
    label = when.strftime("%Y-%m-%d") if when is not None else ""
    fig = _map_figure(plt, lon, lat, field,
                      title="%s - %s" % (f.long_name, label),
                      cbar_label=_label(f), cmap=cmap, vmin=lo, vmax=hi, log=log,
                      coastlines=coastlines, dpi=dpi)
    tag = "date_%s" % label if label else "snapshot"
    save_figure(plt, fig, _path(outdir, base, tag), dpi=dpi, show=show)


def plot_animation(f, lon, lat, cube, idxs, *, cmap, vmin, vmax, robust, log,
                   coastlines, base, outdir, show, dpi, fps):
    """Animated gif over the selected time indices."""
    plt = pyplot(headless=not show)
    from matplotlib.animation import FuncAnimation
    lo, hi = color_limits(cube, vmin, vmax, robust, log)      # shared scale
    norm = make_norm(plt, lo, hi, log)

    fig = plt.figure(figsize=(9, 6.5))
    ax, transform = _setup_axes(fig, lon, lat, coastlines)
    mesh = _draw(ax, transform, lon, lat, cube[idxs[0]], transparent_nan_cmap(plt, cmap), norm)
    cb = fig.colorbar(mesh, ax=ax, shrink=0.75, pad=0.03)
    cb.set_label(_label(f))
    title = ax.set_title("")

    def _label_for(k):
        if f.dates is not None:
            return "%s - %s" % (f.long_name, f.dates[idxs[k]].strftime("%Y-%m-%d"))
        return "%s - frame %d" % (f.long_name, k)

    def update(k):
        mesh.set_array(np.ma.masked_invalid(cube[idxs[k]]).ravel())
        title.set_text(_label_for(k))
        return mesh, title

    update(0)
    ani = FuncAnimation(fig, update, frames=len(idxs), blit=False)

    path = _path(outdir, base, "anim", ext="gif")
    wrote = False
    try:
        from matplotlib.animation import PillowWriter
        ani.save(path, writer=PillowWriter(fps=fps), dpi=dpi)
        sys.stderr.write("[plot] wrote %s (%d frames)\n" % (path, len(idxs)))
        wrote = True
    except Exception as exc:
        sys.stderr.write("[plot] gif writer unavailable (%s); writing PNG frames "
                         "instead\n" % exc)
        framedir = os.path.splitext(path)[0] + "_frames"
        os.makedirs(framedir, exist_ok=True)
        for k in range(len(idxs)):
            update(k)
            fig.savefig(os.path.join(framedir, "frame_%03d.png" % k),
                        dpi=dpi, bbox_inches="tight")
        sys.stderr.write("[plot] wrote %d frames to %s\n" % (len(idxs), framedir))
    if show and wrote:
        plt.show()
    else:
        plt.close(fig)


def plot_timeseries(f, lonq, latq, *, base, outdir, show, dpi, log):
    """Time series at the grid cell nearest to (lonq, latq)."""
    if not f.has_time:
        raise ValueError("file has no time dimension; cannot plot a time series")
    plt = pyplot(headless=not show)
    i = int(np.argmin(np.abs(f.lat - latq)))
    j = int(np.argmin(np.abs(f.lon - lonq)))
    series = f.series_at(i, j)
    clat, clon = float(f.lat[i]), float(f.lon[j])

    fig = plt.figure(figsize=(11, 4.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(f.dates, series, marker=".", ms=3, lw=1.0)
    if log:
        ax.set_yscale("log")
    ax.set_xlabel("date")
    ax.set_ylabel(_label(f))
    ax.set_title("%s at lat=%.4f, lon=%.4f (nearest cell to %.4f, %.4f)"
                 % (f.long_name, clat, clon, latq, lonq))
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    tag = "timeseries_lat%.3f_lon%.3f" % (clat, clon)
    save_figure(plt, fig, _path(outdir, base, tag), dpi=dpi, show=show)
    return f.dates, series


# ---------------------------------------------------------------------------
# naming
# ---------------------------------------------------------------------------
def _label(f):
    return "%s [%s]" % (f.varname, f.units) if f.units else f.varname


def _path(outdir, base, tag, ext="png"):
    if outdir is None:
        return None
    return os.path.join(outdir, "%s_%s.%s" % (base, tag, ext))


def _frame_indices(f, max_frames):
    """Evenly spaced, unique time indices for the animation."""
    n_t = len(f.tnum)
    if max_frames is None:
        years = sorted(set(int(d.year) for d in f.dates))
        max_frames = min(len(years), 20)
    max_frames = max(1, min(int(max_frames), n_t))
    return np.unique(np.linspace(0, n_t - 1, max_frames).round().astype(int))


# ---------------------------------------------------------------------------
# orchestration (importable entry point)
# ---------------------------------------------------------------------------
def plot_output(nc_path, variable=None, bbox=None, date=None, location=None,
                max_frames=None, cmap="viridis", log=False, vmin=None, vmax=None,
                robust=True, coastlines=True, outdir="__auto__", show=False,
                fps=4, dpi=110):
    """Plot a PCR-GLOBWB netCDF variable.  See module docstring for behaviour.

    Returns the :class:`NcField` describing the variable (handy for inspection
    in a notebook).  ``outdir="__auto__"`` saves next to the input file; pass
    ``outdir=None`` to skip saving (e.g. when ``show=True`` in a notebook).
    """
    ds = nc.Dataset(nc_path)
    try:
        f = NcField(ds, variable)

        if outdir == "__auto__":
            outdir = os.path.dirname(os.path.abspath(nc_path))
        if outdir:
            os.makedirs(outdir, exist_ok=True)
        base = "%s_%s" % (f.varname, os.path.splitext(os.path.basename(nc_path))[0])

        # --- mode 3: single-cell time series --------------------------------
        if location is not None:
            lonq, latq = float(location[0]), float(location[1])
            plot_timeseries(f, lonq, latq, base=base, outdir=outdir, show=show,
                            dpi=dpi, log=log)
            return f

        # --- mode 2b: single date ------------------------------------------
        if date is not None:
            if not f.has_time:
                raise ValueError("file has no time dimension; date= is not applicable")
            target = nc.date2num(_parse_date(date), f.tunits, f.tcalendar)
            idx = int(np.argmin(np.abs(f.tnum - target)))
            field = f.slice_at(idx)
            r0, r1, c0, c1 = _crop_indices(f.lat, f.lon, np.isfinite(field), bbox)
            plot_single(f, f.lon[c0:c1], f.lat[r0:r1], field[r0:r1, c0:c1],
                        f.dates[idx], cmap=cmap, vmin=vmin, vmax=vmax, robust=robust,
                        log=log, coastlines=coastlines, base=base, outdir=outdir,
                        show=show, dpi=dpi)
            return f

        # --- load the cube once for the map / animation modes ---------------
        cube = f.cube()
        if cube.ndim == 2:                              # file without a time axis
            r0, r1, c0, c1 = _crop_indices(f.lat, f.lon, np.isfinite(cube), bbox)
            plot_single(f, f.lon[c0:c1], f.lat[r0:r1], cube[r0:r1, c0:c1], None,
                        cmap=cmap, vmin=vmin, vmax=vmax, robust=robust, log=log,
                        coastlines=coastlines, base=base, outdir=outdir, show=show,
                        dpi=dpi)
            return f

        valid = np.isfinite(cube).any(axis=0)
        r0, r1, c0, c1 = _crop_indices(f.lat, f.lon, valid, bbox)
        lonc, latc = f.lon[c0:c1], f.lat[r0:r1]
        cube = cube[:, r0:r1, c0:c1]

        # --- mode 2a: mean + std + animation -------------------------------
        plot_maps(f, lonc, latc, cube, cmap=cmap, vmin=vmin, vmax=vmax,
                  robust=robust, log=log, coastlines=coastlines, base=base,
                  outdir=outdir, show=show, dpi=dpi)

        if cube.shape[0] > 1:
            idxs = _frame_indices(f, max_frames)
            plot_animation(f, lonc, latc, cube, idxs, cmap=cmap, vmin=vmin,
                           vmax=vmax, robust=robust, log=log, coastlines=coastlines,
                           base=base, outdir=outdir, show=show, dpi=dpi, fps=fps)
        return f
    finally:
        ds.close()
