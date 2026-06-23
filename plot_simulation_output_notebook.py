"""
Quick, self-contained notebook version of the PCR-GLOBWB output plotter.

Paste the ``quicklook`` function below into a Jupyter cell (it needs only
numpy, netCDF4 and matplotlib).  It targets PCR-GLOBWB output -- a time/lat/lon
variable with ``lat`` descending and missing data flagged with 1e20 -- and, like
`python plot_simulation_output.py`, copes with any dimension order.  The full
script additionally adds cartopy coastlines (if installed) and writes the
figures / gif to disk.

Examples
--------
    f = quicklook("outputs/.../netcdf/discharge_monthAvg_output.nc")          # mean + std + animation
    quicklook("FILE.nc", date="2006-07-15")                                   # one date (nearest step)
    quicklook("FILE.nc", bbox=(5, 20, 40, 50))                                # lon/lat box
    t, y = quicklook("FILE.nc", loc=(12.5, 45.4))                             # time series, returns arrays
    quicklook("FILE.nc", log=True, max_frames=12)                             # log scale, fewer frames
"""

import datetime

import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
from matplotlib import animation, colors


def quicklook(path, var=None, bbox=None, date=None, loc=None, max_frames=None,
              cmap="viridis", log=False, robust=True, fps=4):
    """Plot a PCR-GLOBWB netCDF variable inline.

    Modes (checked in this order):
      * ``loc=(lon, lat)``  -> time series at the nearest cell (returns dates, values);
      * ``date="YYYY-MM-DD"`` -> one map for the nearest time step;
      * otherwise            -> temporal mean map, std map and an inline animation
                                (default frames = ``min(n_years, 20)``).

    ``bbox=(lonmin, lonmax, latmin, latmax)`` restricts the area; without it the
    map is cropped to where the variable actually has data.  Returns the loaded
    cube/coords (or the series) so you can inspect the numbers directly.
    """
    ds = nc.Dataset(path)
    try:
        # --- locate coordinates and the data variable --------------------
        name = lambda opts: next((n for n in opts if n in ds.variables), None)
        latn, lonn, tn = name(["lat", "latitude", "y"]), name(["lon", "longitude", "x"]), name(["time", "t"])
        lat = np.asarray(ds.variables[latn][:], float)
        lon = np.asarray(ds.variables[lonn][:], float)
        if var is None:
            cands = [v for v in ds.variables if v not in {latn, lonn, tn}
                     and ds.variables[v].ndim >= 2]
            var = ([v for v in cands if ds.variables[v].ndim == 3] or cands)[0]
        V = ds.variables[var]
        dims = list(V.dimensions)
        has_t = tn in dims
        yax, xax = dims.index(latn), dims.index(lonn)
        tax = dims.index(tn) if has_t else None
        units = getattr(V, "units", "")
        long_name = getattr(V, "long_name", var)
        label = "%s [%s]" % (var, units) if units else var

        def clean(a):                       # masked / 1e20 sentinels -> NaN
            a = np.ma.filled(np.ma.asarray(a).astype(float), np.nan)
            a[~np.isfinite(a)] = np.nan
            a[np.abs(a) >= 1e19] = np.nan
            return a

        # reads that are robust to the dimension order of the variable
        def map_at(k=None):                 # one (lat, lon) map; k = time index or None
            sl = [slice(None)] * V.ndim
            if has_t and k is not None:
                sl[tax] = int(k)
            a = clean(V[tuple(sl)])
            keep = [d for d in dims if not (has_t and k is not None and d == tn)]
            return np.transpose(a, (keep.index(latn), keep.index(lonn)))

        def cube3d():                       # (time, lat, lon), or (1, lat, lon) if no time
            a = clean(V[:])
            return (np.transpose(a, (tax, yax, xax)) if has_t
                    else np.transpose(a, (yax, xax))[None])

        def series_at(i, j):                # time series at lat index i, lon index j
            sl = [0] * V.ndim
            sl[tax], sl[yax], sl[xax] = slice(None), int(i), int(j)
            return clean(V[tuple(sl)])

        if has_t:
            tv = ds.variables[tn]
            tnum = np.atleast_1d(np.asarray(tv[:], float))
            tu = getattr(tv, "units", "days since 1901-01-01")
            tc = getattr(tv, "calendar", "standard")
            dates = [datetime.datetime(d.year, d.month, d.day)
                     for d in np.atleast_1d(nc.num2date(tnum, tu, tc))]

        # --- mode: time series at nearest cell ---------------------------
        if loc is not None:
            if not has_t:
                raise ValueError("file has no time axis; cannot plot a series")
            i = int(np.argmin(np.abs(lat - loc[1])))
            j = int(np.argmin(np.abs(lon - loc[0])))
            series = series_at(i, j)
            fig, ax = plt.subplots(figsize=(11, 4))
            ax.plot(dates, series, ".-", ms=3, lw=1)
            if log:
                ax.set_yscale("log")
            ax.set(xlabel="date", ylabel=label,
                   title="%s at lat=%.3f, lon=%.3f (nearest to %.3f, %.3f)"
                         % (long_name, lat[i], lon[j], loc[1], loc[0]))
            ax.grid(alpha=.3)
            fig.autofmt_xdate()
            plt.show()
            return np.array(dates), series

        # --- helper to crop a (..., lat, lon) array ----------------------
        def crop(arr, valid2d):
            if bbox is not None:
                rows = np.where((lat >= bbox[2]) & (lat <= bbox[3]))[0]
                cols = np.where((lon >= bbox[0]) & (lon <= bbox[1]))[0]
            else:
                rows = np.where(valid2d.any(1))[0]
                cols = np.where(valid2d.any(0))[0]
            if rows.size == 0 or cols.size == 0:
                raise ValueError("selection contains no data")
            r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
            return arr[..., r0:r1, c0:c1], lat[r0:r1], lon[c0:c1]

        def limits(a):                       # robust 2-98 pct color limits
            x = a[np.isfinite(a)]
            x = x[x > 0] if log else x
            if x.size == 0:
                return 0., 1.
            lo, hi = (np.percentile(x, [2, 98]) if robust else (x.min(), x.max()))
            return (float(lo), float(hi)) if hi > lo else (float(x.min()), float(x.max() + 1e-9))

        def norm(lo, hi):
            return colors.LogNorm(max(lo, 1e-12), hi) if log else colors.Normalize(lo, hi)

        cm = plt.get_cmap(cmap).copy()
        cm.set_bad("none")

        def heatmap(field2d, la, lo_, title, lo, hi):
            fig, ax = plt.subplots(figsize=(9, 6.5))
            m = ax.pcolormesh(lo_, la, np.ma.masked_invalid(field2d), cmap=cm,
                              norm=norm(lo, hi), shading="nearest")
            ax.set(xlabel="Longitude", ylabel="Latitude", title=title)
            ax.set_aspect("equal")
            fig.colorbar(m, ax=ax, shrink=.75, pad=.03, label=label)
            return fig

        # --- mode: single date -------------------------------------------
        if date is not None:
            if not has_t:
                raise ValueError("file has no time axis; 'date' is not applicable")
            if isinstance(date, datetime.date):
                when = date
            else:
                try:                                 # flexible parsing (e.g. "15 Jul 2006")
                    from dateutil import parser as _dp
                    when = _dp.parse(str(date))
                except Exception:
                    when = datetime.datetime.fromisoformat(str(date))
            k = int(np.argmin(np.abs(tnum - nc.date2num(when, tu, tc))))
            field = map_at(k)
            field, la, lo_ = crop(field, np.isfinite(field))
            heatmap(field, la, lo_, "%s - %s" % (long_name, dates[k].strftime("%Y-%m-%d")),
                    *limits(field))
            plt.show()
            return field, la, lo_

        # --- mode: mean + std + animation --------------------------------
        cube = cube3d()
        cube, la, lo_ = crop(cube, np.isfinite(cube).any(0))
        with np.errstate(invalid="ignore"):
            mean, std = np.nanmean(cube, 0), np.nanstd(cube, 0)
        heatmap(mean, la, lo_, "%s - temporal mean" % long_name, *limits(mean))
        heatmap(std, la, lo_, "%s - temporal std. dev." % long_name, *limits(std))
        plt.show()

        result = {"cube": cube, "lat": la, "lon": lo_, "mean": mean, "std": std,
                  "dates": (np.array(dates) if has_t else None)}
        if not has_t or cube.shape[0] < 2:
            return result

        # animation over evenly spaced frames
        if max_frames is None:
            max_frames = min(len({d.year for d in dates}), 20)
        idx = np.unique(np.linspace(0, cube.shape[0] - 1,
                                    max(1, min(max_frames, cube.shape[0]))).round().astype(int))
        lo, hi = limits(cube)
        fig, ax = plt.subplots(figsize=(9, 6.5))
        m = ax.pcolormesh(lo_, la, np.ma.masked_invalid(cube[idx[0]]), cmap=cm,
                          norm=norm(lo, hi), shading="nearest")
        ax.set(xlabel="Longitude", ylabel="Latitude")
        ax.set_aspect("equal")
        fig.colorbar(m, ax=ax, shrink=.75, pad=.03, label=label)
        ttl = ax.set_title("")

        def update(n):
            m.set_array(np.ma.masked_invalid(cube[idx[n]]).ravel())
            ttl.set_text("%s - %s" % (long_name, dates[idx[n]].strftime("%Y-%m-%d")))
            return m, ttl

        ani = animation.FuncAnimation(fig, update, frames=len(idx), blit=False)
        plt.close(fig)
        result["anim"] = ani
        try:
            from IPython.display import HTML, display
            display(HTML(ani.to_jshtml(fps=fps)))   # inline player, needs no extra deps
        except Exception:
            pass
        return result          # dict: data + figures' arrays + ['anim'] handle
    finally:
        ds.close()
