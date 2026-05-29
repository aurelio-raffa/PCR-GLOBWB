"""
LDD-Based Domain Decomposition for PCR-GLOBWB Parallel Execution

Loads the global 05min LDD (Local Drain Direction) map, builds the directed
flow graph (one edge per non-ocean cell pointing to its downstream neighbour),
computes weakly connected components (equivalent to undirected connected
components), and optionally aggregates the resulting drainage basins into a
user-specified number of balanced subdomains.

Conceptual equivalence with the NetworkX formulation:
  - Each non-ocean cell is a node (LDD != 255).
  - A directed edge (u -> v) is added for every cell u whose LDD value points
    to neighbour v; if v is ocean or out-of-bounds the edge is omitted.
  - Weakly connected components of the directed graph == connected components
    of the undirected version.
  scipy.sparse.csgraph is used instead of NetworkX because the graph has
  ~9.3 M nodes; NetworkX would require several GB of memory for that size.

Aggregation
-----------
The raw connected-component count (~52K for the global 05min LDD) is much
larger than a practical tile count.  The --n_tiles flag triggers a hierarchical
greedy merge that reduces the number of subdomains to N:

  At every step, merge the adjacent pair (A, B) whose combined cell count is
  smallest.  This keeps the merge cost minimal and tends to produce the most
  balanced partition because the smallest components are always consumed first.
  Subdomains that are individually larger than the target (total / N) cannot be
  split and remain as oversize tiles; the algorithm accepts this gracefully.

The final extents CSV is consumable by create_tile_clone_maps.py.

Usage
-----
Report raw basin structure (no aggregation):
    python compute_ldd_basins.py \\
        --ldd misc/ldd_aqueduct_version_2021-09-16/lddsound_05min_version_20210330.map

Aggregate to 71 subdomains and write extents:
    python compute_ldd_basins.py \\
        --ldd misc/ldd_aqueduct_version_2021-09-16/lddsound_05min_version_20210330.map \\
        --n_tiles 71 \\
        --output_extents tile_extents_71.csv

Generate clone maps from the extents:
    python create_tile_clone_maps.py \\
        --global_clone clone_landmask_maps/clone_landmask_examples/clone_global_05min.map \\
        --output_dir /inputs/cloneMaps/global_parallelization \\
        --extents tile_extents_71.csv
"""

import argparse
import heapq
import os
import struct
import sys

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


# ---------------------------------------------------------------------------
# PCRaster LDD constants
# ---------------------------------------------------------------------------

LDD_MV  = 255  # missing value (ocean / undefined)
LDD_PIT = 5    # sink cell: no outgoing edge

# PCRaster row 0 is the NORTHERN edge, so "south" = row+1.
#   7 8 9
#   4 5 6
#   1 2 3
LDD_OFFSETS = {
    1: (+1, -1),   # SW
    2: (+1,  0),   # S
    3: (+1, +1),   # SE
    4: ( 0, -1),   # W
    6: ( 0, +1),   # E
    7: (-1, -1),   # NW
    8: (-1,  0),   # N
    9: (-1, +1),   # NE
}

CELL_SIZE   = 1.0 / 12.0   # 5 arcminutes in decimal degrees
GLOBAL_XMIN = -180.0
GLOBAL_YMAX =   90.0


# ---------------------------------------------------------------------------
# Phase 1 — Load and flow-graph connected components
# ---------------------------------------------------------------------------

def load_ldd(path: str) -> tuple[np.ndarray, int, int]:
    """Read a PCRaster uint8 LDD map directly from the CSF binary."""
    with open(path, 'rb') as f:
        raw = f.read()
    nrows = struct.unpack_from('<I', raw, 100)[0]
    ncols = struct.unpack_from('<I', raw, 104)[0]
    ldd   = np.frombuffer(raw[256: 256 + nrows * ncols], dtype=np.uint8
                          ).reshape(nrows, ncols)
    return ldd, nrows, ncols


def build_flow_edges(ldd: np.ndarray, nrows: int, ncols: int,
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Return (src, dst) flat-index arrays for every valid land-to-land flow edge."""
    src_list, dst_list = [], []
    for ldd_val, (dr, dc) in LDD_OFFSETS.items():
        r_src, c_src = np.where(ldd == ldd_val)
        if r_src.size == 0:
            continue
        r_dst, c_dst = r_src + dr, c_src + dc
        in_bounds = ((r_dst >= 0) & (r_dst < nrows) &
                     (c_dst >= 0) & (c_dst < ncols))
        r_src, c_src = r_src[in_bounds], c_src[in_bounds]
        r_dst, c_dst = r_dst[in_bounds], c_dst[in_bounds]
        land_dst = ldd[r_dst, c_dst] != LDD_MV
        src_list.append(r_src[land_dst] * ncols + c_src[land_dst])
        dst_list.append(r_dst[land_dst] * ncols + c_dst[land_dst])
    return (np.concatenate(src_list).astype(np.int32),
            np.concatenate(dst_list).astype(np.int32))


def compute_flow_components(ldd: np.ndarray, nrows: int, ncols: int,
                             ) -> np.ndarray:
    """
    Build the sparse directed flow graph and return flat scipy component labels.
    Ocean cells end up as isolated nodes (each their own component); they are
    distinguished from land-cell components in later phases by checking LDD_MV.
    """
    n_cells = nrows * ncols
    src, dst = build_flow_edges(ldd, nrows, ncols)
    graph    = csr_matrix((np.ones(len(src), dtype=np.int8), (src, dst)),
                          shape=(n_cells, n_cells))
    _, labels = connected_components(graph, directed=True, connection='weak',
                                     return_labels=True)
    return labels   # shape (n_cells,), dtype int32


# ---------------------------------------------------------------------------
# Phase 2 — Compact labels and 8-connectivity adjacency
# ---------------------------------------------------------------------------

def make_compact_labels(labels: np.ndarray, ldd: np.ndarray,
                         nrows: int, ncols: int,
                         ) -> tuple[np.ndarray, int, np.ndarray]:
    """
    Remap scipy labels to a compact [0, n_land_comp) integer range, ignoring
    ocean-only components.

    Returns
    -------
    compact_2d  : (nrows, ncols) int32 array; -1 for ocean cells
    n_comp      : number of land-cell components
    comp_sizes  : (n_comp,) int64 array of cell counts
    """
    land_flat     = np.where(ldd.ravel() != LDD_MV)[0]
    land_labels   = labels[land_flat]
    unique_lbls, inv = np.unique(land_labels, return_inverse=True)
    n_comp        = len(unique_lbls)

    compact = np.full(nrows * ncols, -1, dtype=np.int32)
    compact[land_flat] = inv.astype(np.int32)

    sizes = np.bincount(inv, minlength=n_comp).astype(np.int64)
    return compact.reshape(nrows, ncols), n_comp, sizes


def build_adjacency(compact_2d: np.ndarray, n_comp: int,
                    ) -> tuple[list[set], np.ndarray, np.ndarray]:
    """
    Build 8-connectivity adjacency between compact component IDs.

    Returns
    -------
    adj    : list of sets; adj[i] = set of component IDs adjacent to i
    pair_a : (n_pairs,) int32  unique adjacent pairs, a < b
    pair_b : (n_pairs,) int32
    """
    all_a, all_b = [], []
    for a_sl, b_sl in [
        (compact_2d[:, :-1],    compact_2d[:, 1:]),     # right
        (compact_2d[:-1, :],    compact_2d[1:, :]),     # down
        (compact_2d[:-1, :-1],  compact_2d[1:, 1:]),    # down-right diagonal
        (compact_2d[:-1, 1:],   compact_2d[1:, :-1]),   # down-left diagonal
    ]:
        mask = (a_sl >= 0) & (b_sl >= 0) & (a_sl != b_sl)
        all_a.append(a_sl[mask].astype(np.int64))
        all_b.append(b_sl[mask].astype(np.int64))

    all_a = np.concatenate(all_a)
    all_b = np.concatenate(all_b)

    # Normalise direction and deduplicate via integer encoding
    mn = np.minimum(all_a, all_b)
    mx = np.maximum(all_a, all_b)
    encoded = mn * n_comp + mx
    unique_enc = np.unique(encoded)
    pair_a = (unique_enc // n_comp).astype(np.int32)
    pair_b = (unique_enc  % n_comp).astype(np.int32)

    adj: list[set] = [set() for _ in range(n_comp)]
    for a, b in zip(pair_a.tolist(), pair_b.tolist()):
        adj[a].add(b)
        adj[b].add(a)

    return adj, pair_a, pair_b


# ---------------------------------------------------------------------------
# Phase 3 — Hierarchical greedy aggregation
# ---------------------------------------------------------------------------

def aggregate_components(n_comp: int, sizes: np.ndarray,
                          adj: list[set], n_target: int,
                          ) -> np.ndarray:
    """
    Reduce n_comp components to n_target by repeatedly merging the adjacent
    pair with the smallest combined cell count.

    Invariant maintained throughout:
      adj[r] contains only current root IDs (roots satisfy parent[r] == r).

    Parameters
    ----------
    n_comp   : initial number of components
    sizes    : (n_comp,) int64 cell counts (mutated in-place for roots)
    adj      : list of sets of root IDs (mutated in-place)
    n_target : desired final component count

    Returns
    -------
    parent : (n_comp,) int32 array mapping every original component to its
             final super-component root
    """
    if n_comp <= n_target:
        return np.arange(n_comp, dtype=np.int32)

    parent = np.arange(n_comp, dtype=np.int32)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]   # path compression
            x = parent[x]
        return x

    # Build initial heap — (combined_size, a, b) with a < b for uniqueness
    heap: list[tuple[int, int, int]] = []
    for a in range(n_comp):
        for b in adj[a]:
            if a < b:
                heapq.heappush(heap, (int(sizes[a]) + int(sizes[b]), a, b))

    n_remaining = n_comp
    n_merges    = n_comp - n_target

    for step in range(n_merges):
        if not heap:
            print(f"WARNING: heap exhausted after {step} merges "
                  f"({n_remaining} components remain, target {n_target}). "
                  f"The graph may have disconnected regions.",
                  file=sys.stderr)
            break

        # Pop until we find a valid (still-separate, still-adjacent) pair
        while heap:
            cost, a, b = heapq.heappop(heap)
            ra, rb = find(a), find(b)
            if ra != rb and rb in adj[ra]:
                break
        else:
            break

        # --- Merge rb into ra (ra keeps its ID as the root) ---
        parent[rb] = ra
        sizes[ra] += sizes[rb]

        # Update adjacency: ra inherits all of rb's neighbours
        for nb in adj[rb]:
            if nb != ra:
                adj[nb].discard(rb)
                adj[nb].add(ra)
                adj[ra].add(nb)
        adj[ra].discard(rb)
        adj[rb] = set()   # rb is no longer a root

        # Push new candidate merges for ra onto the heap
        for nb in adj[ra]:
            heapq.heappush(heap, (int(sizes[ra]) + int(sizes[nb]), ra, nb))

        n_remaining -= 1

        if (step + 1) % 5000 == 0:
            print(f"  ... {step + 1}/{n_merges} merges done "
                  f"({n_remaining} components remaining)", flush=True)

    return parent


# ---------------------------------------------------------------------------
# Phase 4 — Compute extents and assign M-codes
# ---------------------------------------------------------------------------

def compute_extents(compact_2d: np.ndarray, parent: np.ndarray,
                     sizes: np.ndarray, nrows: int, ncols: int,
                     ) -> list[dict]:
    """
    For each final super-component (find(x) == x), compute the geographic
    bounding box of its land cells and return a list sorted by cell count desc.
    """
    n_comp = len(parent)

    # Build a remap: original compact ID -> final root
    roots = np.array([_find_root(i, parent) for i in range(n_comp)], dtype=np.int32)

    # Remap compact_2d to final root IDs
    # compact_2d has values -1 (ocean) or 0..n_comp-1
    final_2d = np.where(compact_2d >= 0, roots[compact_2d], -1)

    unique_roots = np.unique(roots)   # active root IDs only
    print(f"  {len(unique_roots):,} super-components after aggregation", flush=True)

    # Sort land cells by final root for efficient slicing
    flat_land = np.where(final_2d.ravel() >= 0)[0]
    land_roots = final_2d.ravel()[flat_land]
    sort_idx   = np.argsort(land_roots, kind='stable')
    sorted_roots = land_roots[sort_idx]
    sorted_rows  = (flat_land[sort_idx] // ncols).astype(np.int32)
    sorted_cols  = (flat_land[sort_idx]  % ncols).astype(np.int32)

    boundaries = np.searchsorted(sorted_roots, unique_roots)
    boundaries = np.append(boundaries, len(sorted_roots))

    basins = []
    for i, root in enumerate(unique_roots.tolist()):
        s, e = int(boundaries[i]), int(boundaries[i + 1])
        r    = sorted_rows[s:e]
        c    = sorted_cols[s:e]
        basins.append({
            'root':    root,
            'n_cells': int(sizes[root]),
            'xmin':    GLOBAL_XMIN +  c.min()      * CELL_SIZE,
            'xmax':    GLOBAL_XMIN + (c.max() + 1) * CELL_SIZE,
            'ymax':    GLOBAL_YMAX -  r.min()       * CELL_SIZE,
            'ymin':    GLOBAL_YMAX - (r.max() + 1)  * CELL_SIZE,
        })

    basins.sort(key=lambda b: b['n_cells'], reverse=True)
    return basins


def _find_root(x: int, parent: np.ndarray) -> int:
    """Non-mutating find (used during result extraction only)."""
    while parent[x] != x:
        x = parent[x]
    return x


def assign_codes(basins: list[dict]) -> list[dict]:
    """Assign M01, M02, ... codes in descending cell-count order."""
    for i, b in enumerate(basins):
        b['code'] = f"M{i + 1:02d}"
    return basins


# ---------------------------------------------------------------------------
# Reporting and output
# ---------------------------------------------------------------------------

def print_summary(basins: list[dict], label: str = '') -> None:
    total = sum(b['n_cells'] for b in basins)
    n     = len(basins)
    avg   = total / n if n else 0
    mx    = basins[0]['n_cells'] if basins else 0
    mn    = basins[-1]['n_cells'] if basins else 0

    print(f"\n{'='*65}")
    if label:
        print(f"  {label}")
    print(f"  Subdomains : {n}")
    print(f"  Total land : {total:,} cells")
    print(f"  Target/tile: {avg:,.0f} cells  (total / N)")
    print(f"  Largest    : {mx:,} cells  ({100*mx/total:.1f}%)")
    print(f"  Smallest   : {mn:,} cells  ({100*mn/total:.1f}%)")
    print(f"  Imbalance  : {mx/avg:.2f}x  (largest / average)")
    print(f"\n  {'Code':<8} {'Cells':>10}  {'xmin':>8} {'ymin':>7}"
          f" {'xmax':>8} {'ymax':>7}  {'%land':>6}")
    print(f"  {'-'*8} {'-'*10}  {'-'*8} {'-'*7} {'-'*8} {'-'*7}  {'-'*6}")
    for b in basins:
        print(f"  {b['code']:<8} {b['n_cells']:>10,}  "
              f"{b['xmin']:8.3f} {b['ymin']:7.3f} "
              f"{b['xmax']:8.3f} {b['ymax']:7.3f}  "
              f"{100*b['n_cells']/total:6.2f}%")
    print(f"{'='*65}\n")


def write_extents_csv(basins: list[dict], path: str, ldd_path: str) -> None:
    with open(path, 'w') as f:
        f.write("# Tile extents derived from LDD weakly connected components\n")
        f.write(f"# Source LDD: {ldd_path}\n")
        f.write("# Codes assigned in descending cell-count order (M01 = largest)\n")
        f.write("code,xmin,ymin,xmax,ymax\n")
        for b in basins:
            f.write(f"{b['code']},{b['xmin']:.10f},{b['ymin']:.10f},"
                    f"{b['xmax']:.10f},{b['ymax']:.10f}\n")
    print(f"Extents CSV written: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        '--ldd', required=True,
        help='Path to the global 05min PCRaster LDD map',
    )
    p.add_argument(
        '--n_tiles', type=int, default=None,
        help='Target number of subdomains after hierarchical aggregation. '
             'If omitted, only the raw connected-component summary is printed '
             'and no merging is performed.',
    )
    p.add_argument(
        '--output_extents', default=None,
        help='Output extents CSV (requires --n_tiles). '
             'Compatible with create_tile_clone_maps.py.',
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.output_extents and not args.n_tiles:
        sys.exit('--output_extents requires --n_tiles')

    if not os.path.isfile(args.ldd):
        sys.exit(f'LDD map not found: {args.ldd}')

    # ------------------------------------------------------------------
    # Phase 1: connected components
    # ------------------------------------------------------------------
    print(f'Loading LDD: {args.ldd}', flush=True)
    ldd, nrows, ncols = load_ldd(args.ldd)
    n_land = int((ldd != LDD_MV).sum())
    print(f'  Grid: {nrows} x {ncols},  land cells: {n_land:,}', flush=True)

    print('Building flow graph and computing connected components ...', flush=True)
    labels = compute_flow_components(ldd, nrows, ncols)

    compact_2d, n_comp, sizes = make_compact_labels(labels, ldd, nrows, ncols)
    print(f'  {n_comp:,} land-cell components found', flush=True)

    if args.n_tiles is None:
        # Report-only mode: show raw basins (top 30) without aggregation
        parent = np.arange(n_comp, dtype=np.int32)
        basins = compute_extents(compact_2d, parent, sizes, nrows, ncols)
        basins = assign_codes(basins)
        print_summary(basins[:30],
                      label=f'Top 30 of {n_comp:,} raw components (no aggregation)')
        print('Run with --n_tiles N to aggregate into N subdomains.')
        return

    # ------------------------------------------------------------------
    # Phase 2: adjacency
    # ------------------------------------------------------------------
    print('Building 8-connectivity adjacency graph ...', flush=True)
    adj, pair_a, pair_b = build_adjacency(compact_2d, n_comp)
    n_edges = len(pair_a)
    print(f'  {n_edges:,} unique adjacent component pairs', flush=True)

    # ------------------------------------------------------------------
    # Phase 3: aggregation
    # ------------------------------------------------------------------
    print(f'Aggregating {n_comp:,} components -> {args.n_tiles} subdomains ...', flush=True)
    parent = aggregate_components(n_comp, sizes, adj, args.n_tiles)

    # ------------------------------------------------------------------
    # Phase 4: extents + output
    # ------------------------------------------------------------------
    print('Computing geographic extents ...', flush=True)
    basins = compute_extents(compact_2d, parent, sizes, nrows, ncols)
    basins = assign_codes(basins)
    print_summary(basins, label=f'Final {len(basins)} subdomains')

    if args.output_extents:
        write_extents_csv(basins, args.output_extents, args.ldd)


if __name__ == '__main__':
    main()
