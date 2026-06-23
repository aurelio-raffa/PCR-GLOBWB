"""Inspect a PCR-GLOBWB clone partition: validate the inputs, compute tile statistics, and plot the partition.

This is the tile *summary + plotting* extracted from ``compute_ldd_basins`` into a reusable, standalone form,
so it can be run on **any** clone partition -- including a third party's directory of clone/landmask maps --
rather than only as a side effect of computing one. It reuses ``print_summary`` and ``save_partition_image``
from ``src.utils.ldd_basins`` (no duplication) and the shared matplotlib helpers in ``src.utils.plotting``.

Input modes (priority order):
  * ``partition``  -- a partition NPZ from ``compute_ldd_basins`` (full: cell counts + image).
  * ``maps_dir``   -- a directory of per-tile PCRaster ``.map`` and/or NetCDF ``.nc`` clone/landmask files.
                      Each file's bounding box, grid and cell values are read from its own header (no extents
                      CSV needed), the files are VALIDATED against the expected convention, the global tile
                      map is reconstructed, and statistics + an image are produced. ``.map`` files are read by
                      parsing the PCRaster CSF header directly (no pcraster dependency).
  * ``extents``    -- an extents CSV alone -> bounding-box-only summary (no cell counts / image).

Convention notes (what the validator checks):
  * a *clone* map is an all-TRUE boolean map defining a tile's rectangular window;
  * a *landmask* is a boolean map that is TRUE only on the cells that actually belong to the tile;
  * both are boolean (PCRaster valueScale 224), on a common cell size, grid-aligned to the global origin.
  Where the per-tile windows overlap (clones do; landmasks should not), the overlap is reported.
"""
import io
import os
import re
import glob
import struct
import contextlib

import numpy as np

# global 5-arcmin reference grid (shared with tile_clone_maps); the south/east edges are implied
from src.utils.tile_clone_maps import CELL_SIZE, GLOBAL_XMIN, GLOBAL_YMAX
GLOBAL_YMIN, GLOBAL_XMAX = -90.0, 180.0

_CSF_SIGNATURE = b'RUU CROSS SYSTEM MAP FORMAT'
_VS_BOOLEAN = 224                      # PCRaster valueScale for a boolean map


# --------------------------------------------------------------------------------------------------------------
# reading a single per-tile map (PCRaster .map via CSF header, or NetCDF .nc)
# --------------------------------------------------------------------------------------------------------------
def _read_csf_boolean(path: str) -> dict:
    """Read a boolean PCRaster (CSF v2) map by parsing its header directly (no pcraster needed).

    Returns a dict with: ``active`` (bool 2-D, TRUE cells), ``rows``/``cols``, ``xUL``/``yUL``, ``cellsize``,
    ``value_scale``, ``n_mv``/``n_false`` and ``signature_ok``. Raises ValueError if the file is too small.
    """
    with open(path, 'rb') as handle:
        raw = handle.read()
    signature_ok = raw[:len(_CSF_SIGNATURE)] == _CSF_SIGNATURE
    value_scale = struct.unpack_from('<H', raw, 64)[0]
    x_ul = struct.unpack_from('<d', raw, 84)[0]
    y_ul = struct.unpack_from('<d', raw, 92)[0]
    rows = struct.unpack_from('<I', raw, 100)[0]
    cols = struct.unpack_from('<I', raw, 104)[0]
    cellsize = struct.unpack_from('<d', raw, 108)[0]

    n = int(rows) * int(cols)
    # boolean maps store one UINT1 byte per cell; the data block is the file's last rows*cols bytes
    # (CSF header padding -> data offset is normally 256, but deriving it from the size is robust).
    data_offset = len(raw) - n
    if data_offset < 0:
        raise ValueError(f'{path}: file too small for {rows}x{cols} boolean cells')
    data = np.frombuffer(raw, dtype=np.uint8, count=n, offset=data_offset).reshape(rows, cols)
    return dict(active=(data == 1), rows=int(rows), cols=int(cols), xUL=float(x_ul), yUL=float(y_ul),
                cellsize=float(cellsize), value_scale=int(value_scale), signature_ok=signature_ok,
                data_offset=int(data_offset), n_mv=int((data == 255).sum()), n_false=int((data == 0).sum()))


def _read_nc_clone(path: str) -> dict:
    """Read a per-tile NetCDF clone/landmask: derive the bbox/grid from lat/lon and the TRUE cells from data."""
    import netCDF4 as nc
    ds = nc.Dataset(path)
    try:
        latn = next((c for c in ('lat', 'latitude', 'y', 'Y') if c in ds.variables), None)
        lonn = next((c for c in ('lon', 'longitude', 'x', 'X') if c in ds.variables), None)
        if latn is None or lonn is None:
            raise ValueError(f'{path}: no lat/lon coordinate variables')
        lat = np.asarray(ds.variables[latn][:], dtype='float64')
        lon = np.asarray(ds.variables[lonn][:], dtype='float64')
        cellsize = abs(float(lon[1] - lon[0])) if lon.size > 1 else float('nan')
        coords = {latn, lonn, 'time', 'Time', 't'}
        dvars = [v for v in ds.variables if v not in coords and ds.variables[v].ndim >= 2]
        if not dvars:
            raise ValueError(f'{path}: no 2-D data variable')
        data = np.ma.filled(np.ma.asarray(ds.variables[dvars[0]][:]).astype('float64'), np.nan)
        if lat.size > 1 and lat[0] < lat[-1]:        # ensure north-up (lat descending) like the CSF maps
            data = data[::-1]
        rows, cols = data.shape
        active = np.isfinite(data) & (data != 0)
        x_ul = float(lon.min()) - cellsize / 2.0
        y_ul = float(lat.max()) + cellsize / 2.0
        return dict(active=active, rows=int(rows), cols=int(cols), xUL=x_ul, yUL=y_ul, cellsize=cellsize,
                    value_scale=None, signature_ok=True, data_offset=None,
                    n_mv=int(np.isnan(data).sum()), n_false=int((data == 0).sum()), data_var=dvars[0])
    finally:
        ds.close()


def _read_tile_map_file(path: str) -> dict:
    """Read a per-tile map dispatching on extension (.map -> CSF boolean, .nc -> NetCDF)."""
    if path.endswith('.nc'):
        return _read_nc_clone(path)
    return _read_csf_boolean(path)


# --------------------------------------------------------------------------------------------------------------
# enumeration + validation
# --------------------------------------------------------------------------------------------------------------
def _enumerate_codes(maps_dir: str, pattern: str):
    """Return the tile codes found in ``maps_dir`` for a ``%s`` filename ``pattern`` (natural-sorted)."""
    left, _, right = pattern.partition('%s')
    regex = re.compile('^' + re.escape(left) + '(.+?)' + re.escape(right) + '$')
    codes = []
    for name in os.listdir(maps_dir):
        m = regex.match(name)
        if m:
            codes.append(m.group(1))

    def _key(code):
        digits = re.findall(r'\d+', code)
        return (int(digits[0]) if digits else 0, code)
    return sorted(set(codes), key=_key)


def _resolve_codes(maps_dir, pattern, clone_areas):
    """Resolve the requested clone codes: ``auto`` globs the pattern; a CSV is split; else a single token."""
    if clone_areas is None or str(clone_areas).strip().lower() == 'auto':
        return _enumerate_codes(maps_dir, pattern)
    return [token.strip() for token in str(clone_areas).split(',') if token.strip()]


def _aligned(value, origin, cellsize, tol=1e-4):
    """True if ``value`` sits on the global grid (an integer number of cells from ``origin``)."""
    remainder = abs((value - origin) / cellsize)
    remainder -= round(remainder)
    return abs(remainder) <= tol


def _validate_and_load(maps_dir, codes, clone_pattern, landmask_pattern, expected_cellsize):
    """Load + validate every tile's map. Returns (entries, report_lines).

    For each code the landmask file is preferred (if a landmask pattern is given and the file exists), else the
    clone file. A sibling ``.nc`` for a chosen ``.map`` (or vice versa) is cross-checked for matching dims/bbox.
    """
    entries, report = [], []
    n_clone, n_landmask, n_warn, n_fail, n_xchecked, n_xmatch = 0, 0, 0, 0, 0, 0

    def _path(pattern, code):
        return os.path.join(maps_dir, pattern % code) if pattern else None

    for code in codes:
        lm = _path(landmask_pattern, code) if landmask_pattern else None
        cl = _path(clone_pattern, code) if clone_pattern else None
        source = lm if (lm and os.path.isfile(lm)) else (cl if (cl and os.path.isfile(cl)) else None)
        if source is None:
            report.append(f'  {code}: FAIL  no file matching '
                          f'{landmask_pattern or clone_pattern!r} in {maps_dir}')
            n_fail += 1
            continue
        try:
            info = _read_tile_map_file(source)
        except Exception as error:
            report.append(f'  {code}: FAIL  {os.path.basename(source)} unreadable ({error})')
            n_fail += 1
            continue

        msgs = []
        # boolean check (CSF only)
        if source.endswith('.map') and info['value_scale'] not in (None, _VS_BOOLEAN):
            msgs.append(f'valueScale={info["value_scale"]} (expected boolean {_VS_BOOLEAN})')
        if source.endswith('.map') and not info['signature_ok']:
            msgs.append('not a CSF "RUU CROSS SYSTEM MAP FORMAT" file')
        # grid checks
        if expected_cellsize and abs(info['cellsize'] - expected_cellsize) > expected_cellsize * 1e-3:
            msgs.append(f'cellsize {info["cellsize"]:.8f} != expected {expected_cellsize:.8f}')
        if not (_aligned(info['xUL'], GLOBAL_XMIN, info['cellsize'])
                and _aligned(info['yUL'], GLOBAL_YMAX, info['cellsize'])):
            msgs.append('NW corner not aligned to the global grid')

        n_active = int(info['active'].sum())
        n_total = info['rows'] * info['cols']
        kind = 'clone (all-TRUE bbox)' if n_active == n_total else \
               ('landmask (partial)' if 0 < n_active < n_total else 'empty')
        if kind.startswith('clone'):
            n_clone += 1
        elif kind.startswith('landmask'):
            n_landmask += 1

        # cross-check a sibling file in the other format, when present
        sibling = (source[:-4] + '.nc') if source.endswith('.map') else (source[:-3] + '.map')
        if os.path.isfile(sibling):
            try:
                sib = _read_tile_map_file(sibling)
                n_xchecked += 1
                if (sib['rows'], sib['cols']) == (info['rows'], info['cols']) \
                        and abs(sib['xUL'] - info['xUL']) <= info['cellsize'] \
                        and abs(sib['yUL'] - info['yUL']) <= info['cellsize']:
                    n_xmatch += 1
                else:
                    msgs.append(f'sibling {os.path.basename(sibling)} dims/bbox differ')
            except Exception as error:
                msgs.append(f'sibling {os.path.basename(sibling)} unreadable ({error})')

        info.update(code=code, source=source, n_active=n_active, n_total=n_total, kind=kind)
        entries.append(info)
        if msgs:
            n_warn += 1
            report.append(f'  {code}: WARN  {os.path.basename(source)}  [{kind}]  ' + '; '.join(msgs))

    header = [
        '=' * 90,
        f'  VALIDATION — {len(codes)} requested, {len(entries)} loaded, {n_fail} failed',
        f'    kinds        : {n_clone} clone (all-TRUE bbox), {n_landmask} landmask (partial)',
        f'    cross-check  : {n_xmatch}/{n_xchecked} sibling .map/.nc pairs agree on dims/bbox'
        if n_xchecked else '    cross-check  : (no sibling files found)',
        f'    warnings     : {n_warn}',
    ]
    report = header + (report if report else ['    all files conform to the expected convention.']) + ['=' * 90]
    return entries, report


# --------------------------------------------------------------------------------------------------------------
# partition assembly + basin dicts
# --------------------------------------------------------------------------------------------------------------
def _build_tile_map(entries, cellsize):
    """Stamp every tile's active cells into a global tile map; also return per-cell coverage counts."""
    nrows_g = int(round((GLOBAL_YMAX - GLOBAL_YMIN) / cellsize))
    ncols_g = int(round((GLOBAL_XMAX - GLOBAL_XMIN) / cellsize))
    tile_map = np.full((nrows_g, ncols_g), -1, dtype=np.int16)
    coverage = np.zeros((nrows_g, ncols_g), dtype=np.int16)
    for i, e in enumerate(entries):
        xmin, ymax = e['xUL'], e['yUL']
        col0 = int(round((xmin - GLOBAL_XMIN) / cellsize))
        row0 = int(round((GLOBAL_YMAX - ymax) / cellsize))
        rr, cc = np.where(e['active'])
        gr, gc = row0 + rr, col0 + cc
        ok = (gr >= 0) & (gr < nrows_g) & (gc >= 0) & (gc < ncols_g)
        tile_map[gr[ok], gc[ok]] = i
        coverage[gr[ok], gc[ok]] += 1
    return tile_map, coverage


def _basins_from_entries(entries, cellsize):
    """Build the ``basins`` dict list (own per-tile cell counts; ``print_summary`` sorts descending)."""
    basins = []
    for i, e in enumerate(entries):
        xmin, ymax = e['xUL'], e['yUL']
        xmax = xmin + e['cols'] * cellsize
        ymin = ymax - e['rows'] * cellsize
        bbox_cells = e['rows'] * e['cols']
        fill = 100.0 * e['n_active'] / bbox_cells if bbox_cells else 0.0
        basins.append(dict(code=e['code'], n_cells=e['n_active'], bbox_cells=bbox_cells, fill_pct=fill,
                           xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, root=i))
    basins.sort(key=lambda b: b['n_cells'], reverse=True)
    return basins


def _basins_from_tile_map(tile_map, codes, xmin, ymin, xmax, ymax, cell_size):
    """Build ``basins`` from a partition NPZ's tile_map + per-tile extents arrays."""
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


# --------------------------------------------------------------------------------------------------------------
# output helpers
# --------------------------------------------------------------------------------------------------------------
def _capture(func, *args, **kwargs) -> str:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        func(*args, **kwargs)
    return buffer.getvalue()


def _emit(text: str, output_summary=None, append=False) -> None:
    print(text, end='' if text.endswith('\n') else '\n')
    if output_summary:
        with open(output_summary, 'a' if append else 'w') as handle:
            handle.write(text if text.endswith('\n') else text + '\n')


def _coverage_report(coverage, cellsize) -> str:
    union = int((coverage >= 1).sum())
    overlap = int((coverage >= 2).sum())
    max_cov = int(coverage.max()) if coverage.size else 0
    pct = 100.0 * overlap / union if union else 0.0
    return ('  COVERAGE — union {union:,} cells covered by >=1 tile; {overlap:,} cells ({pct:.2f}%) '
            'covered by >=2 tiles (max overlap depth {max_cov}).\n'
            '    Overlaps are expected for clone bounding boxes and should be ~0 for true landmasks.\n'
            ).format(union=union, overlap=overlap, pct=pct, max_cov=max_cov)


def _print_extent_summary(extents, label=None, output_summary=None) -> None:
    lines = ['=' * 60]
    if label:
        lines.append(f'  {label}')
    lines.append(f'  Tiles: {len(extents)}  (bounding boxes only — pass a partition NPZ or a maps_dir '
                 f'for cell counts and an image)')
    lines.append(f"  {'Code':<6} {'xmin':>9} {'ymin':>8} {'xmax':>9} {'ymax':>8}")
    for code, (x0, y0, x1, y1) in extents.items():
        lines.append(f"  {code:<6} {x0:9.3f} {y0:8.3f} {x1:9.3f} {y1:8.3f}")
    lines.append('=' * 60)
    _emit('\n'.join(lines) + '\n', output_summary)


# --------------------------------------------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------------------------------------------
def inspect_partition(partition=None, extents=None, maps_dir=None,
                      clone_pattern='clone_%s.map', landmask_pattern=None, clone_areas='auto',
                      cell_size=CELL_SIZE, validate=True, output_summary=None, output_image=None,
                      label=None, annotate=True) -> None:
    """Inspect a clone partition (validate inputs, compute statistics, plot). See the module docstring."""
    from src.utils.ldd_basins import print_summary, save_partition_image

    # ---- mode A: directory of per-tile .map/.nc files (validate + reconstruct from headers) ----
    if maps_dir is not None:
        codes = _resolve_codes(maps_dir, landmask_pattern or clone_pattern, clone_areas)
        if not codes:
            raise SystemExit(f'inspect_partition: no tiles found in {maps_dir} for pattern '
                             f'{landmask_pattern or clone_pattern!r} (clone_areas={clone_areas!r}).')
        entries, report = _validate_and_load(
            maps_dir, codes, clone_pattern, landmask_pattern, cell_size if validate else None)
        if not entries:
            _emit('\n'.join(report), output_summary)
            raise SystemExit('inspect_partition: no tiles could be loaded; see the validation report above.')

        cellsize = float(np.median([e['cellsize'] for e in entries]))
        tile_map, coverage = _build_tile_map(entries, cellsize)
        basins = _basins_from_entries(entries, cellsize)
        loaded_codes = [e['code'] for e in entries]

        _emit('\n'.join(report) + '\n', output_summary)
        _emit(_coverage_report(coverage, cellsize), output_summary, append=True)
        _emit(_capture(print_summary, basins, label=label or f'Partition: {maps_dir}'),
              output_summary, append=True)
        if output_image:
            save_partition_image(tile_map, np.arange(len(loaded_codes), dtype=np.int32), output_image,
                                 annotate=bool(annotate), basins=basins)
        return

    # ---- mode B: a partition NPZ from compute_ldd_basins ----
    if partition is not None:
        from src.utils.tile_clone_maps import load_partition
        p = load_partition(partition)
        tile_map, codes = p['tile_map'], p['codes']
        basins = _basins_from_tile_map(tile_map, codes, p['xmin'], p['ymin'], p['xmax'], p['ymax'],
                                       p['cell_size'])
        _emit(_capture(print_summary, basins, label=label or 'Partition summary'), output_summary)
        if output_image:
            save_partition_image(tile_map, np.arange(len(codes), dtype=np.int32), output_image,
                                 annotate=bool(annotate), basins=basins)
        return

    # ---- mode C: an extents CSV alone (bounding-box-only summary) ----
    if extents is not None:
        from src.utils.tile_clone_maps import load_extents
        _print_extent_summary(load_extents(extents), label=label, output_summary=output_summary)
        return

    raise SystemExit('inspect_partition: provide --maps_dir DIR, or --partition NPZ, or --extents CSV.')
