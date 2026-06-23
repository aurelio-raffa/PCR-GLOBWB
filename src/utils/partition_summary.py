"""Inspect a PCR-GLOBWB clone partition: print/save a tile summary and a colour-coded partition image.

This is the tile *summary + plotting* extracted from ``compute_ldd_basins`` into a reusable, standalone form,
so it can be run on **any** clone partition -- including a third party's -- rather than only as a side effect
of computing one. It reuses the existing ``print_summary`` and ``save_partition_image`` from
``src.utils.ldd_basins`` (no duplication), and the shared matplotlib helpers in ``src.utils.plotting``.

Input modes (priority order):
  * ``partition``              -- a partition NPZ from ``compute_ldd_basins`` (full: cell counts + image).
  * ``landmask_dir`` + ``extents`` -- reconstruct the global tile map from a third party's per-tile landmask
                                  maps (needs pcraster) -> full inspection.
  * ``extents``                -- an extents CSV alone -> bounding-box-only summary (no cell counts / image).
"""
import io
import os
import contextlib

import numpy as np


def _basins_from_tile_map(tile_map, codes, xmin, ymin, xmax, ymax, cell_size):
    """Build the ``basins`` dict list that ``print_summary`` / ``save_partition_image`` expect."""
    valid = tile_map[tile_map >= 0].ravel()
    counts = np.bincount(valid, minlength=len(codes)) if valid.size else np.zeros(len(codes), dtype=int)
    basins = []
    for i, code in enumerate(codes):
        x0, y0, x1, y1 = float(xmin[i]), float(ymin[i]), float(xmax[i]), float(ymax[i])
        ncols = int(round((x1 - x0) / cell_size))
        nrows = int(round((y1 - y0) / cell_size))
        bbox_cells = max(ncols * nrows, 0)
        n_cells = int(counts[i]) if i < len(counts) else 0
        fill = 100.0 * n_cells / bbox_cells if bbox_cells else 0.0
        basins.append(dict(code=str(code), n_cells=n_cells, bbox_cells=bbox_cells, fill_pct=fill,
                           xmin=x0, ymin=y0, xmax=x1, ymax=y1, root=i))
    basins.sort(key=lambda b: b['n_cells'], reverse=True)
    return basins


def _tile_map_from_landmasks(landmask_dir, extents, landmask_pattern='landmask_%s.map'):
    """Reconstruct a global tile map from a directory of per-tile PCRaster landmask maps.

    Each landmask is read on its own (grid-aligned) tile clone and stamped, at the tile's global row/col
    offset, into a full-globe ``tile_map`` (value = tile index, -1 elsewhere). Reuses the 05min grid
    constants and corner-snapping from ``tile_clone_maps``.
    """
    from src.utils.tile_clone_maps import _require_pcraster, grid_aligned_nw_corner, \
        CELL_SIZE, GLOBAL_XMIN, GLOBAL_YMAX
    pcr = _require_pcraster()

    cellsize = CELL_SIZE
    global_ymin, global_xmax = -90.0, 180.0
    nrows_g = int(round((GLOBAL_YMAX - global_ymin) / cellsize))
    ncols_g = int(round((global_xmax - GLOBAL_XMIN) / cellsize))
    tile_map = np.full((nrows_g, ncols_g), -1, dtype=np.int16)

    codes = list(extents)
    xs0, ys0, xs1, ys1 = [], [], [], []
    for i, code in enumerate(codes):
        x0, y0, x1, y1 = extents[code]
        ncols_t = int(round((x1 - x0) / cellsize))
        nrows_t = int(round((y1 - y0) / cellsize))
        west, north = grid_aligned_nw_corner(x0, y1, cellsize)
        pcr.setclone(nrows_t, ncols_t, cellsize, west, north)
        active = np.asarray(pcr.pcr2numpy(pcr.readmap(os.path.join(landmask_dir, landmask_pattern % code)), 0))
        col0 = int(round((x0 - GLOBAL_XMIN) / cellsize))
        row0 = int(round((GLOBAL_YMAX - y1) / cellsize))
        rr, cc = np.where(active > 0)
        gr, gc = row0 + rr, col0 + cc
        ok = (gr >= 0) & (gr < nrows_g) & (gc >= 0) & (gc < ncols_g)
        tile_map[gr[ok], gc[ok]] = i
        xs0.append(x0); ys0.append(y0); xs1.append(x1); ys1.append(y1)
    return (tile_map, codes,
            np.array(xs0), np.array(ys0), np.array(xs1), np.array(ys1), cellsize)


def _capture(func, *args, **kwargs) -> str:
    """Run ``func`` capturing everything it prints to stdout, and return it as a string."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        func(*args, **kwargs)
    return buffer.getvalue()


def _emit(text: str, output_summary=None) -> None:
    print(text, end='' if text.endswith('\n') else '\n')
    if output_summary:
        with open(output_summary, 'w') as handle:
            handle.write(text)
        print(f'Tile summary written: {output_summary}')


def _print_extent_summary(extents, label=None, output_summary=None) -> None:
    """Bounding-box-only summary (when no cell counts are available)."""
    lines = ['=' * 60]
    if label:
        lines.append(f'  {label}')
    lines.append(f'  Tiles: {len(extents)}  (bounding boxes only — pass a partition NPZ or '
                 f'a landmask dir for cell counts and an image)')
    lines.append(f"  {'Code':<6} {'xmin':>9} {'ymin':>8} {'xmax':>9} {'ymax':>8}")
    for code, (x0, y0, x1, y1) in extents.items():
        lines.append(f"  {code:<6} {x0:9.3f} {y0:8.3f} {x1:9.3f} {y1:8.3f}")
    lines.append('=' * 60)
    _emit('\n'.join(lines) + '\n', output_summary)


def inspect_partition(partition=None, extents=None, landmask_dir=None,
                      landmask_pattern='landmask_%s.map', output_summary=None, output_image=None,
                      label=None, annotate=True) -> None:
    """Inspect a clone partition. See the module docstring for the input modes."""
    from src.utils.tile_clone_maps import load_partition, load_extents
    from src.utils.ldd_basins import print_summary, save_partition_image

    if partition is not None:
        p = load_partition(partition)
        tile_map, codes = p['tile_map'], p['codes']
        basins = _basins_from_tile_map(tile_map, codes, p['xmin'], p['ymin'], p['xmax'], p['ymax'],
                                       p['cell_size'])
    elif landmask_dir is not None:
        if extents is None:
            raise SystemExit('inspect_partition: --landmask_dir requires --extents (the tile bounding boxes).')
        ext = load_extents(extents)
        tile_map, codes, xs0, ys0, xs1, ys1, cs = _tile_map_from_landmasks(landmask_dir, ext, landmask_pattern)
        basins = _basins_from_tile_map(tile_map, codes, xs0, ys0, xs1, ys1, cs)
    elif extents is not None:
        _print_extent_summary(load_extents(extents), label=label, output_summary=output_summary)
        return
    else:
        raise SystemExit('inspect_partition: provide --partition NPZ, or --landmask_dir + --extents, '
                         'or --extents (bbox-only).')

    _emit(_capture(print_summary, basins, label=label or 'Partition summary'), output_summary)

    if output_image:
        save_partition_image(tile_map, np.arange(len(codes), dtype=np.int32), output_image,
                             annotate=bool(annotate), basins=basins)
