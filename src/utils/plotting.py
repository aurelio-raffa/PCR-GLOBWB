"""Shared matplotlib helpers for the plotting stages.

These are the low-level utilities used by *both* new plotting stages -- the partition inspector
(``src/utils/partition_summary.py`` / ``ldd_basins.py``) and the simulation-output plotter
(``src/utils/plot_output.py``) -- so the matplotlib setup lives in one place rather than being duplicated.

matplotlib is imported lazily (via :func:`pyplot`) so that:
  * a CLI / pipeline stage can switch to the headless ``Agg`` backend before pyplot is first imported
    (works over SSH with no X display), while a notebook keeps its inline backend;
  * importing these utility modules never requires matplotlib until something is actually plotted.
"""


def pyplot(headless: bool = True):
    """Return ``matplotlib.pyplot``, optionally forcing the non-interactive ``Agg`` backend first."""
    import matplotlib
    if headless:
        try:
            matplotlib.use('Agg', force=True)
        except Exception:
            pass
    import matplotlib.pyplot as plt
    return plt


def get_cmap(name, lut=None):
    """Return colormap ``name`` (optionally resampled to ``lut`` levels), across matplotlib versions.

    ``matplotlib.cm.get_cmap`` / ``pyplot.cm.get_cmap`` were removed in matplotlib 3.11; the colormap registry
    (``matplotlib.colormaps``, available since 3.6) is the supported replacement, with a legacy fallback.
    """
    import matplotlib
    registry = getattr(matplotlib, 'colormaps', None)
    if registry is not None:                                   # matplotlib >= 3.6
        cmap = registry[name]
        return cmap.resampled(lut) if lut else cmap
    import matplotlib.cm as mcm                                 # matplotlib < 3.6 (legacy)
    return mcm.get_cmap(name, lut)


def transparent_nan_cmap(plt, name):
    """A copy of colormap ``name`` whose 'bad' (NaN/masked) colour is transparent."""
    import copy
    cm = copy.copy(get_cmap(name))
    cm.set_bad(color='none')        # NaN cells transparent -> the axes background shows through
    return cm


def color_limits(data, vmin=None, vmax=None, robust=True, log=False):
    """Resolve ``(vmin, vmax)`` from explicit values or robust 2-98 percentiles of the finite data."""
    import numpy as np
    finite = data[np.isfinite(data)]
    if log:
        finite = finite[finite > 0]
    if finite.size == 0:
        return (0.0, 1.0)
    lo = vmin if vmin is not None else (np.nanpercentile(finite, 2) if robust else finite.min())
    hi = vmax if vmax is not None else (np.nanpercentile(finite, 98) if robust else finite.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(finite.min()), float(max(finite.max(), finite.min() + 1e-12))
    if log and lo <= 0:
        lo = float(finite[finite > 0].min()) if (finite > 0).any() else 1e-6
    return float(lo), float(hi)


def make_norm(plt, vmin, vmax, log=False):
    """A matplotlib ``Normalize`` (or ``LogNorm`` when ``log``) for the given limits."""
    import matplotlib.colors as mcolors
    if log:
        return mcolors.LogNorm(vmin=max(vmin, 1e-12), vmax=vmax)
    return mcolors.Normalize(vmin=vmin, vmax=vmax)


def save_figure(plt, fig, path, dpi=150, show=False, announce=True):
    """Save ``fig`` to ``path`` (tight bbox) and close it; optionally also ``show`` and announce on stderr."""
    import sys
    if path is not None:
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        if announce:
            print(f'[plot] wrote {path}', file=sys.stderr)
    if show:
        plt.show()
    else:
        plt.close(fig)
