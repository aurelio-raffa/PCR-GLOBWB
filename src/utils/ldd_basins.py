"""
LDD-Based Domain Decomposition for PCR-GLOBWB Parallel Execution

Loads the global 05min LDD (Local Drain Direction) map, computes weakly
connected components (one per drainage basin), optionally splits oversized
basins and drops undersized ones, then aggregates the remainder into a
user-specified number of balanced tiles.

Tree structure
--------------
Each weakly connected component of the LDD flow graph is a directed tree:
  - Every non-pit cell has exactly one outgoing edge (its flow direction).
  - There is exactly one pit cell (LDD=5) per component.
  - The tree has S cells and S-1 intra-component edges.
Use --verify_tree to confirm this for the largest components.

Pre-processing flags
--------------------
--lb_cells LB
    Drop components with fewer than LB cells before aggregation.
    Their cells are marked inactive (-2) and excluded from all outputs.

--ub_cells UB
    Recursively bisect components larger than UB cells using tree centroid
    splitting: at each step find the edge (v -> downstream(v)) whose removal
    most evenly halves the subtree, and split there.  Repeat until no
    component exceeds UB.

Aggregation
-----------
Heap-based greedy merging reduces components to --n_tiles.  Because the
global land surface has ~1,292 geographically disconnected 8-connected
regions (continents + isolated islands), heap merging alone cannot reach
N < 1,292.  The --n_tiles flag also triggers a force-merge pass that assigns
remaining isolated super-components to their nearest neighbour by geographic
centroid, reaching the requested N exactly.

Visualization
-------------
--output_image PATH  saves a color-coded PNG of the final partition.
                     White = ocean, black = filtered cells, tab20 colors
                     for active tiles, with each tile's cell count annotated
                     at its barycenter.

Usage
-----
Report raw basin structure:
    python compute_ldd_basins.py --ldd <ldd.map>

Aggregate to 71 balanced tiles, split basins > 100 000 cells,
drop islands < 500 cells, and save a partition image:
    python compute_ldd_basins.py \\
        --ldd misc/ldd_aqueduct_version_2021-09-16/lddsound_05min_version_20210330.map \\
        --n_tiles 71 --ub_cells 100000 --lb_cells 500 \\
        --output_extents tile_extents_71.csv \\
        --output_image partition_71.png
"""

import argparse
import heapq
import math
import os
import struct
import sys
from collections import deque
from types import SimpleNamespace

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

# ---------------------------------------------------------------------------
# PCRaster LDD constants
# ---------------------------------------------------------------------------

LDD_MV = 255  # missing value (ocean / undefined)
LDD_PIT = 5  # sink cell: no outgoing edge

# PCRaster row 0 = northern edge; row increases southward.
#   7 8 9
#   4 5 6
#   1 2 3
LDD_OFFSETS = {
    1: (+1, -1),  # SW
    2: (+1, 0),  # S
    3: (+1, +1),  # SE
    4: (0, -1),  # W
    6: (0, +1),  # E
    7: (-1, -1),  # NW
    8: (-1, 0),  # N
    9: (-1, +1),  # NE
}

CELL_SIZE = 1.0 / 12.0  # 5 arcminutes in decimal degrees
GLOBAL_XMIN = -180.0
GLOBAL_YMAX = 90.0


# ---------------------------------------------------------------------------
# Phase 1 — Load LDD and compute flow-graph connected components
# ---------------------------------------------------------------------------

def load_ldd(path: str) -> tuple[np.ndarray, int, int]:
    """Read a PCRaster uint8 LDD map directly from the CSF binary."""
    with open(path, 'rb') as f:
        raw = f.read()
    nrows = struct.unpack_from('<I', raw, 100)[0]
    ncols = struct.unpack_from('<I', raw, 104)[0]
    ldd = np.frombuffer(raw[256: 256 + nrows * ncols], dtype=np.uint8
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
    """Return flat scipy component labels (shape nrows*ncols)."""
    n_cells = nrows * ncols
    src, dst = build_flow_edges(ldd, nrows, ncols)
    graph = csr_matrix((np.ones(len(src), dtype=np.int8), (src, dst)),
                       shape=(n_cells, n_cells))
    _, labels = connected_components(graph, directed=True, connection='weak',
                                     return_labels=True)
    return labels


# ---------------------------------------------------------------------------
# Phase 2 — Compact labels and 8-connectivity adjacency
# ---------------------------------------------------------------------------

def make_compact_labels(labels: np.ndarray, ldd: np.ndarray,
                        nrows: int, ncols: int,
                        ) -> tuple[np.ndarray, int, np.ndarray]:
    """
    Remap scipy labels to [0, n_land_comp) ignoring ocean-only components.

    Returns
    -------
    compact_2d  : (nrows, ncols) int32; -1 for ocean cells
    n_comp      : number of land-cell components
    comp_sizes  : (n_comp,) int64 cell counts
    """
    land_flat = np.where(ldd.ravel() != LDD_MV)[0]
    land_labels = labels[land_flat]
    unique_lbls, inv = np.unique(land_labels, return_inverse=True)
    n_comp = len(unique_lbls)

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
    adj    : list of sets; adj[i] = component IDs adjacent to i
    pair_a : (n_pairs,) int32  unique adjacent pairs, a < b
    pair_b : (n_pairs,) int32
    """
    all_a, all_b = [], []
    for a_sl, b_sl in [
        (compact_2d[:, :-1], compact_2d[:, 1:]),
        (compact_2d[:-1, :], compact_2d[1:, :]),
        (compact_2d[:-1, :-1], compact_2d[1:, 1:]),
        (compact_2d[:-1, 1:], compact_2d[1:, :-1]),
    ]:
        # Exclude -2 (filtered) and -1 (ocean) on either side
        mask = (a_sl >= 0) & (b_sl >= 0) & (a_sl != b_sl)
        all_a.append(a_sl[mask].astype(np.int64))
        all_b.append(b_sl[mask].astype(np.int64))

    all_a = np.concatenate(all_a)
    all_b = np.concatenate(all_b)

    mn = np.minimum(all_a, all_b)
    mx = np.maximum(all_a, all_b)
    encoded = mn * n_comp + mx
    unique_enc = np.unique(encoded)
    pair_a = (unique_enc // n_comp).astype(np.int32)
    pair_b = (unique_enc % n_comp).astype(np.int32)

    adj: list[set] = [set() for _ in range(n_comp)]
    for a, b in zip(pair_a.tolist(), pair_b.tolist()):
        adj[a].add(b)
        adj[b].add(a)

    return adj, pair_a, pair_b


# ---------------------------------------------------------------------------
# Tree structure verification
# ---------------------------------------------------------------------------

def verify_tree_structure(ldd: np.ndarray, compact_2d: np.ndarray,
                          sizes: np.ndarray, n_check: int = 5) -> None:
    """
    Confirm that each connected component is a directed tree.

    A valid LDD tree satisfies:
      1. Exactly 1 pit cell (LDD=5) per component.
      2. Intra-component flow edges == component_size - 1.

    Property 2 follows from Property 1 when every non-pit cell has exactly one
    outgoing edge that stays within the component, which is always true for a
    single drainage basin.
    """
    nrows, ncols = compact_2d.shape
    flat = compact_2d.ravel()
    top_ids = np.argsort(sizes)[::-1][:n_check].tolist()

    print(f"\nTree structure check (top {n_check} components by size):")
    header = f"  {'cid':>7}  {'S':>10}  {'pits':>5}  {'edges':>10}  {'S-1 ok':>7}  {'valid':>5}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for cid in top_ids:
        cell_pos = np.where(flat == cid)[0]
        S = len(cell_pos)
        rows = cell_pos // ncols
        cols = cell_pos % ncols
        ldd_vals = ldd[rows, cols]

        n_pits = int((ldd_vals == LDD_PIT).sum())

        # Count intra-component flow edges (vectorized)
        n_edges = 0
        for ldd_val, (dr, dc) in LDD_OFFSETS.items():
            msk = (ldd_vals == ldd_val)
            if not msk.any():
                continue
            r_d = rows[msk] + dr
            c_d = cols[msk] + dc
            ib = (r_d >= 0) & (r_d < nrows) & (c_d >= 0) & (c_d < ncols)
            d_flat = r_d[ib] * ncols + c_d[ib]
            # Check membership via flat[d_flat] == cid
            n_edges += int((flat[d_flat] == cid).sum())

        edge_ok = (n_edges == S - 1)
        valid = (n_pits == 1) and edge_ok
        print(f"  {cid:>7}  {S:>10,}  {n_pits:>5}  {n_edges:>10,}  "
              f"{'YES':>7}  {'YES':>5}" if valid else
              f"  {cid:>7}  {S:>10,}  {n_pits:>5}  {n_edges:>10,}  "
              f"{'YES' if edge_ok else 'NO':>7}  {'NO':>5}")
    print()


# ---------------------------------------------------------------------------
# UB splitting — tree centroid bisection
# ---------------------------------------------------------------------------

def _build_flow_tree(ldd: np.ndarray, cells_flat: np.ndarray,
                     nrows: int, ncols: int,
                     ) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Build the LDD flow tree for a single component.

    Parameters
    ----------
    cells_flat : flat indices of the component's cells (arbitrary order)

    Returns
    -------
    downstream_pos : (S,) int32; downstream_pos[i] = local position of
                     cell i's downstream neighbour, or -1 if pit/exit
    upstream_size  : (S,) int64; number of cells in subtree rooted at i
    upstream_nbrs  : list of lists; upstream_nbrs[j] = [positions that flow to j]
    """
    S = len(cells_flat)
    rows = (cells_flat // ncols).astype(np.int32)
    cols = (cells_flat % ncols).astype(np.int32)
    ldd_vals = ldd[rows, cols].astype(np.int32)

    # flat_index -> local position  (only for cells in this component)
    max_flat = int(cells_flat.max()) + 1
    flat_to_pos = np.full(max_flat, -1, dtype=np.int64)
    flat_to_pos[cells_flat] = np.arange(S, dtype=np.int64)

    downstream_pos = np.full(S, -1, dtype=np.int32)
    in_degree = np.zeros(S, dtype=np.int32)

    for ldd_val, (dr, dc) in LDD_OFFSETS.items():
        msk = (ldd_vals == ldd_val)
        if not msk.any():
            continue
        pos_src = np.where(msk)[0]
        r_d = rows[msk] + dr
        c_d = cols[msk] + dc
        ib = (r_d >= 0) & (r_d < nrows) & (c_d >= 0) & (c_d < ncols)
        pos_src = pos_src[ib]
        r_d = r_d[ib];
        c_d = c_d[ib]
        d_flat = r_d * ncols + c_d

        safe = d_flat < max_flat
        d_pos_raw = flat_to_pos[np.where(safe, d_flat, 0)]
        in_comp = safe & (d_pos_raw >= 0)

        src_in = pos_src[in_comp]
        dst_in = d_pos_raw[in_comp].astype(np.int32)

        downstream_pos[src_in] = dst_in
        np.add.at(in_degree, dst_in, 1)

    # Topological sort: compute upstream subtree sizes
    upstream_size = np.ones(S, dtype=np.int64)
    work_degree = in_degree.copy()
    queue = deque(np.where(work_degree == 0)[0].tolist())
    while queue:
        pos = queue.popleft()
        ds = int(downstream_pos[pos])
        if ds >= 0:
            upstream_size[ds] += upstream_size[pos]
            work_degree[ds] -= 1
            if work_degree[ds] == 0:
                queue.append(ds)

    # Reverse adjacency for upstream BFS during split
    upstream_nbrs: list[list[int]] = [[] for _ in range(S)]
    for pos in range(S):
        ds = int(downstream_pos[pos])
        if ds >= 0:
            upstream_nbrs[ds].append(pos)

    return downstream_pos, upstream_size, upstream_nbrs


def _collect_upstream(root_pos: int, upstream_nbrs: list) -> list[int]:
    """BFS: collect all positions in the upstream subtree of root_pos."""
    result: list[int] = []
    stack = [root_pos]
    while stack:
        p = stack.pop()
        result.append(p)
        stack.extend(upstream_nbrs[p])
    return result


def split_component(ldd: np.ndarray, cells_flat: np.ndarray,
                    nrows: int, ncols: int, ub_cells: int,
                    ) -> list[np.ndarray]:
    """
    Recursively bisect cells_flat using tree centroid splitting until every
    sub-component has at most ub_cells cells.

    Returns a list of flat-index arrays (one per sub-component).
    """
    S = len(cells_flat)
    if S <= ub_cells:
        return [cells_flat]

    downstream_pos, upstream_size, upstream_nbrs = _build_flow_tree(
        ldd, cells_flat, nrows, ncols)

    # Best split position: upstream_size closest to S/2, excluding the root
    # (root has upstream_size == S)
    non_root = upstream_size < S
    if not non_root.any():
        return [cells_flat]  # degenerate (single-node tree)

    diffs = np.where(non_root, np.abs(2 * upstream_size - S), S + 1)
    best_pos = int(np.argmin(diffs))

    # Upstream group = subtree of best_pos
    up_positions = _collect_upstream(best_pos, upstream_nbrs)
    up_mask = np.zeros(S, dtype=bool)
    up_mask[up_positions] = True

    result = []
    for mask in (up_mask, ~up_mask):
        group = cells_flat[mask]
        if len(group) > 0:
            result.extend(split_component(ldd, group, nrows, ncols, ub_cells))
    return result


def apply_ub_splits(ldd: np.ndarray, compact_2d: np.ndarray,
                    nrows: int, ncols: int, sizes: np.ndarray,
                    ub_cells: int,
                    ) -> tuple[np.ndarray, int, np.ndarray]:
    """
    Split all components with more than ub_cells cells.

    Returns updated (compact_2d, n_comp, sizes).
    """
    n_comp = len(sizes)
    large = np.where(sizes > ub_cells)[0]
    if len(large) == 0:
        return compact_2d.copy(), n_comp, sizes.copy()

    print(f"  {len(large)} component(s) > {ub_cells:,} cells — splitting ...",
          flush=True)

    # Group cells by component for large components (vectorized per-component lookup)
    flat = compact_2d.ravel()
    comp_cells: dict[int, np.ndarray] = {
        int(cid): np.where(flat == cid)[0] for cid in large
    }

    compact_out = compact_2d.copy()
    flat_out = compact_out.ravel()
    next_id = n_comp

    for cid, cell_list in comp_cells.items():
        cells = cell_list.astype(np.int64)
        subs = split_component(ldd, cells, nrows, ncols, ub_cells)

        if len(subs) <= 1:
            continue

        n_sub = len(subs)
        sub_sizes = [len(s) for s in subs]
        print(f"    cid {cid} ({sizes[cid]:,} cells) -> {n_sub} sub-components "
              f"[{min(sub_sizes):,} .. {max(sub_sizes):,}]", flush=True)

        # First sub-group keeps the original cid (no change in compact_out)
        # Remaining sub-groups get new IDs
        for sub in subs[1:]:
            flat_out[sub] = next_id
            next_id += 1

    # Rebuild sizes from scratch
    new_n = next_id
    new_sizes = np.zeros(new_n, dtype=np.int64)
    land = flat_out[flat_out >= 0]
    np.add.at(new_sizes, land, 1)

    return compact_out, new_n, new_sizes


# ---------------------------------------------------------------------------
# LB filter — drop components below a cell-count threshold
# ---------------------------------------------------------------------------

def apply_lb_filter(compact_2d: np.ndarray, sizes: np.ndarray,
                    lb_cells: int) -> np.ndarray:
    """
    Set cells belonging to components smaller than lb_cells to -2
    (inactive / excluded from simulation).

    Returns an updated compact_2d (copy).
    """
    flat = compact_2d.ravel().copy()
    land = flat >= 0
    cids = np.where(land, flat, 0)
    small = land & (sizes[cids] < lb_cells)
    flat[small] = -2
    n_dropped = int(small.sum())
    print(f"  LB filter: dropped {(sizes < lb_cells).sum():,} components "
          f"({n_dropped:,} cells marked inactive)", flush=True)
    return flat.reshape(compact_2d.shape)


# ---------------------------------------------------------------------------
# Recompact labels after splits / filters
# ---------------------------------------------------------------------------

def recompact_labels(compact_2d: np.ndarray, old_sizes: np.ndarray,
                     ) -> tuple[np.ndarray, int, np.ndarray]:
    """
    Re-index active component IDs (>= 0) to a contiguous 0..n-1 range.
    Cells with value -1 (ocean) or -2 (filtered) are unchanged.
    """
    flat = compact_2d.ravel()
    active = np.unique(flat[flat >= 0])
    n_new = len(active)

    max_id = int(active.max()) + 1
    remap = np.full(max_id, -1, dtype=np.int32)
    remap[active] = np.arange(n_new, dtype=np.int32)

    new_flat = flat.copy()
    land = flat >= 0
    new_flat[land] = remap[flat[land]]

    new_sizes = old_sizes[active].copy()
    return new_flat.reshape(compact_2d.shape), n_new, new_sizes


# ---------------------------------------------------------------------------
# Phase 3 — Hierarchical greedy aggregation
# ---------------------------------------------------------------------------

def aggregate_components(n_comp: int, sizes: np.ndarray,
                         adj: list[set], n_target: int,
                         ub_cells: int | None = None,
                         strict: bool = False,
                         merge_history: list | None = None,
                         rep_max: int = 2,
                         round_callback: 'callable | None' = None,
                         pair_a: 'np.ndarray | None' = None,
                         pair_b: 'np.ndarray | None' = None,
                         compact_2d: 'np.ndarray | None' = None,
                         nrows: int = 0,
                         ncols: int = 0,
                         lb_disconnected: 'int | None' = None,
                         merge_fill_minimum: 'float | None' = None,
                         ) -> np.ndarray:
    """
    Reduce n_comp components to n_target by repeatedly merging the adjacent
    pair with the smallest combined cell count.

    When ub_cells is given the algorithm runs up to rep_max rounds.  At the
    start of each round the heap is rebuilt from scratch using the current
    component state, so merges are always in ascending cost order with no
    stale entries.  Only pairs whose combined size does not exceed the round
    threshold enter heap_ok; when heap_ok empties the round ends and the
    threshold is doubled for the next round.

    After rep_max rounds, if n_remaining > n_target:
      strict=True  → RuntimeError
      strict=False → one final unconstrained pass on the current state

    Returns parent : (n_comp,) int32 mapping each original component to its
                     final super-component root.
    """
    if n_comp <= n_target:
        return np.arange(n_comp, dtype=np.int32)

    parent = np.arange(n_comp, dtype=np.int32)
    active_roots: set[int] = set(range(n_comp))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _pop_valid(heap: list) -> tuple[int, int] | None:
        """Drain stale entries; return (ra, rb) for the first live pair."""
        while heap:
            _, a, b = heapq.heappop(heap)
            ra, rb = find(a), find(b)
            if ra != rb and rb in adj[ra]:
                return ra, rb
        return None

    def _build_heap(threshold: int | None) -> list:
        """Fresh min-heap of all active adjacent pairs within threshold."""
        h: list = []
        for ra in active_roots:
            for nb in adj[ra]:
                if ra < nb:
                    cost = int(sizes[ra]) + int(sizes[nb])
                    if threshold is None or cost <= threshold:
                        heapq.heappush(h, (cost, ra, nb))
        return h

    def _pre_round_force_merge(threshold: int) -> None:
        """
        Run force_merge_to_target with the given ub threshold, then rebuild
        adj and active_roots so the subsequent heap round sees a consistent
        view.  Only called when compact_2d was supplied by the caller.
        """
        nonlocal n_remaining
        force_merge_to_target(compact_2d, parent, sizes, n_target,
                              nrows, ncols,
                              ub_cells=threshold, strict=False,
                              merge_fill_minimum=merge_fill_minimum)
        # Rebuild adj from original pair arrays mapped through current roots
        for i in range(n_comp):
            adj[i] = set()
        for a, b in zip(pair_a, pair_b):
            ra, rb = find(int(a)), find(int(b))
            if ra != rb:
                adj[ra].add(rb)
                adj[rb].add(ra)
        # Drop disconnected small components now that adj is current
        if lb_disconnected is not None:
            _drop_disconnected_small(compact_2d, parent, sizes,
                                     adj, lb_disconnected)
        # Rebuild active_roots from compact_2d (ground truth): only roots
        # that still have live cells are active.  This excludes both
        # _drop_disconnected_small-dropped roots (cells set to -2) and
        # force_merge fill-screen-dropped roots that somehow lost all cells.
        flat_c = compact_2d.ravel()
        live_comps = np.unique(flat_c[flat_c >= 0].astype(np.int32))
        active_roots.clear()
        for c in live_comps:
            active_roots.add(find(int(c)))
        n_remaining = len(active_roots)

    # Dendrogram history state (only used when merge_history is not None)
    if merge_history is not None:
        _cid: list[int] = list(range(n_comp))
        _nxt: list[int] = [n_comp]
        _rmax: list[int] = [int(sizes.max())]

    n_remaining = n_comp
    total_ok = 0
    total_viol = 0

    def _do_merge(ra: int, rb: int, is_viol: bool) -> None:
        nonlocal n_remaining, total_ok, total_viol

        if merge_history is not None:
            ca, cb = _cid[ra], _cid[rb]

        parent[rb] = ra
        sizes[ra] += sizes[rb]
        active_roots.discard(rb)

        for nb in adj[rb]:
            if nb != ra:
                adj[nb].discard(rb)
                adj[nb].add(ra)
                adj[ra].add(nb)
        adj[ra].discard(rb)
        adj[rb] = set()

        if merge_history is not None:
            combined = int(sizes[ra])
            _rmax[0] = max(_rmax[0], combined)
            merge_history.append((ca, cb, combined, _rmax[0], is_viol))
            _cid[ra] = _nxt[0]
            _nxt[0] += 1

        n_remaining -= 1
        if is_viol:
            total_viol += 1
        else:
            total_ok += 1

    # ------------------------------------------------------------------
    # Main aggregation
    # ------------------------------------------------------------------
    if ub_cells is None:
        # No constraint: single unconstrained pass
        heap = _build_heap(None)
        while n_remaining > n_target:
            pair = _pop_valid(heap)
            if pair is None:
                print(f'WARNING: heap exhausted ({n_remaining} remain).',
                      file=sys.stderr)
                break
            ra, rb = pair
            _do_merge(ra, rb, False)
            for nb in adj[ra]:
                heapq.heappush(heap, (int(sizes[ra]) + int(sizes[nb]), ra, nb))
    else:
        current_ub = ub_cells

        for rep in range(rep_max):
            # Pre-round: geographic force-merge handles components that are
            # spatially isolated (no topologically adjacent pair within
            # current_ub), preventing them from cascading in the fallback.
            if compact_2d is not None:
                before_fm = n_remaining
                _pre_round_force_merge(current_ub)
                print(f'  Pre-round {rep + 1} force-merge (ub={current_ub:,}): '
                      f'{before_fm - n_remaining} merges, '
                      f'{n_remaining} remaining', flush=True)
                if n_remaining <= n_target:
                    break

            # Rebuild heap from current component state with current threshold.
            # This ensures ascending cost order with no stale entries.
            heap_ok = _build_heap(current_ub)
            merges_this = 0

            while n_remaining > n_target:
                pair = _pop_valid(heap_ok)
                if pair is None:
                    break   # heap_ok exhausted for this round
                ra, rb = pair
                # Re-validate: sizes may have grown since this entry was pushed
                if int(sizes[ra]) + int(sizes[rb]) > current_ub:
                    continue  # stale entry; skip without merging
                _do_merge(ra, rb, False)
                merges_this += 1
                # Push newly eligible neighbours into this round's heap
                for nb in adj[ra]:
                    cost = int(sizes[ra]) + int(sizes[nb])
                    if cost <= current_ub:
                        heapq.heappush(heap_ok, (cost, ra, nb))

            print(f'  Round {rep + 1}/{rep_max}: ub={current_ub:,}, '
                  f'{merges_this} merges, {n_remaining} remaining', flush=True)

            if round_callback is not None:
                round_callback(rep, rep_max, parent.copy(), current_ub, False)

            if n_remaining <= n_target:
                break

            current_ub *= 2

        # ub_cells rounds exhausted; decide what to do
        if n_remaining > n_target:
            if strict:
                raise RuntimeError(
                    f'--strict: {n_remaining} components remain after {rep_max} '
                    f'round(s) (final ub={current_ub // 2:,}), target {n_target}. '
                    f'Increase --rep_max, relax --ub_cells, increase --n_tiles, '
                    f'or omit --strict.'
                )
            # Non-strict fallback: one unconstrained pass on the current state
            heap_fb = _build_heap(None)
            merges_fb = 0
            while n_remaining > n_target:
                pair = _pop_valid(heap_fb)
                if pair is None:
                    print(f'WARNING: heap exhausted with {n_remaining} components.',
                          file=sys.stderr)
                    break
                ra, rb = pair
                _do_merge(ra, rb, True)
                merges_fb += 1
                for nb in adj[ra]:
                    heapq.heappush(heap_fb,
                                   (int(sizes[ra]) + int(sizes[nb]), ra, nb))
            if merges_fb:
                print(f'  Fallback (unconstrained): {merges_fb} merges',
                      flush=True)
            if round_callback is not None:
                round_callback(rep_max, rep_max, parent.copy(),
                               current_ub, True)

        print(f'  Total merges: {total_ok:,} compliant, '
              f'{total_viol:,} unconstrained', flush=True)

    return parent


# ---------------------------------------------------------------------------
# Drop small disconnected components before force-merge
# ---------------------------------------------------------------------------

def _drop_disconnected_small(compact_2d: np.ndarray, parent: np.ndarray,
                              sizes: np.ndarray, adj: list[set],
                              lb_disconnected: int) -> int:
    """
    Find super-component roots that have no topological neighbours in adj
    (i.e. adj[root] == set()) and fewer than lb_disconnected cells.
    Mark all their cells as -2 (inactive) in compact_2d.

    These components are excluded from the partition entirely: they will not
    appear in compute_extents output and cannot be targeted by force-merge.

    Returns the number of components dropped.
    """
    n_comp = len(parent)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    # Build roots map and collect candidates
    roots_map = np.array([find(i) for i in range(n_comp)], dtype=np.int32)
    unique_roots = np.unique(roots_map)
    dropped = np.array(
        [r for r in unique_roots if len(adj[r]) == 0 and int(sizes[r]) < lb_disconnected],
        dtype=np.int32,
    )
    if len(dropped) == 0:
        return 0

    # Mark cells of dropped roots as -2
    flat = compact_2d.ravel()          # view for C-contiguous arrays
    land_idx = np.where(flat >= 0)[0]
    land_roots = roots_map[flat[land_idx]]
    drop_mask = np.isin(land_roots, dropped)
    flat[land_idx[drop_mask]] = -2

    total_cells = int(sum(sizes[r] for r in dropped))
    print(f'  Dropped {len(dropped):,} disconnected component(s) '
          f'< {lb_disconnected:,} cells  ({total_cells:,} cells total)',
          flush=True)
    return len(dropped)


# ---------------------------------------------------------------------------
# Force-merge isolated super-components to reach n_target
# ---------------------------------------------------------------------------

def force_merge_to_target(compact_2d_base: np.ndarray,
                          parent: np.ndarray, sizes: np.ndarray,
                          n_target: int, nrows: int, ncols: int,
                          ub_cells: int | None = None,
                          strict: bool = False,
                          merge_fill_minimum: float | None = None,
                          ) -> None:
    """
    After heap-based aggregation, force-merge the smallest remaining
    super-components into their nearest neighbour (by geographic centroid)
    until exactly n_target super-components remain.

    When ub_cells is given, the nearest compliant neighbour (combined <=
    ub_cells) is preferred.  If none exists and strict=True, raises;
    otherwise falls back to the globally nearest neighbour.

    Fill screening (size-independent):
      merge_fill_minimum : if merging into the nearest neighbour would bring
                           the combined bounding-box fill fraction below this
                           value, the component is dropped instead of merged
                           (parent[r] = r left intact; not counted toward
                           n_target).

    Truly disconnected small components should be dropped upstream via
    _drop_disconnected_small before this function is called.

    Mutates parent and sizes in-place.
    """

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    n_comp = len(parent)
    roots_map = np.array([find(i) for i in range(n_comp)], dtype=np.int32)

    flat = compact_2d_base.ravel()
    land = flat >= 0  # excludes -1 (ocean) and -2 (filtered)
    flat_land = flat[land].astype(np.int32)
    idx_land = np.where(land)[0]
    rows_land = (idx_land // ncols).astype(np.float64)
    cols_land = (idx_land % ncols).astype(np.float64)
    root_land = roots_map[flat_land]
    rows_int  = (idx_land // ncols).astype(np.int32)
    cols_int  = (idx_land  % ncols).astype(np.int32)

    unique_roots = np.unique(root_land)
    n_remaining = len(unique_roots)
    if n_remaining <= n_target:
        return

    max_root = int(unique_roots.max()) + 1
    row_sum = np.bincount(root_land, weights=rows_land, minlength=max_root)
    col_sum = np.bincount(root_land, weights=cols_land, minlength=max_root)
    cnt = np.bincount(root_land, minlength=max_root).astype(np.float64)
    c_row = np.where(cnt > 0, row_sum / np.maximum(cnt, 1), 0.0)
    c_col = np.where(cnt > 0, col_sum / np.maximum(cnt, 1), 0.0)

    # Per-component bounding boxes in pixel space (needed for fill screening)
    do_fill_screen = merge_fill_minimum is not None
    if do_fill_screen:
        bb_rmin = np.full(max_root, nrows, dtype=np.int32)
        bb_rmax = np.zeros(max_root,        dtype=np.int32)
        bb_cmin = np.full(max_root, ncols,  dtype=np.int32)
        bb_cmax = np.zeros(max_root,        dtype=np.int32)
        np.minimum.at(bb_rmin, root_land, rows_int)
        np.maximum.at(bb_rmax, root_land, rows_int)
        np.minimum.at(bb_cmin, root_land, cols_int)
        np.maximum.at(bb_cmax, root_land, cols_int)

    # Sort roots by size ascending; merge smallest first
    root_sizes = sizes[unique_roots]
    merge_order = unique_roots[np.argsort(root_sizes)]
    active_roots = set(unique_roots.tolist())

    n_to_merge = n_remaining - n_target
    merged = 0
    dropped = 0

    for r in merge_order.tolist():
        if merged >= n_to_merge:
            break
        if r not in active_roots:
            continue

        active_roots.discard(r)
        remaining = np.array(sorted(active_roots), dtype=np.int32)
        if len(remaining) == 0:
            break

        dr = c_row[remaining] - c_row[r]
        dc = c_col[remaining] - c_col[r]
        dists = dr * dr + dc * dc

        if ub_cells is not None:
            combined = sizes[remaining] + sizes[r]
            compliant = combined <= ub_cells
            if compliant.any():
                masked = np.where(compliant, dists, np.inf)
                nr = int(remaining[np.argmin(masked)])
            elif strict:
                raise RuntimeError(
                    f'--strict: force-merge of component {r} ({sizes[r]:,} cells) '
                    f'has no compliant neighbour within ub_cells={ub_cells:,}. '
                    f'Relax --ub_cells, increase --n_tiles, or omit --strict.'
                )
            else:
                nr = int(remaining[np.argmin(dists)])
        else:
            nr = int(remaining[np.argmin(dists)])

        # Fill screening: drop if merge would degrade combined bbox fill
        if do_fill_screen:
            new_rmin = min(int(bb_rmin[r]), int(bb_rmin[nr]))
            new_rmax = max(int(bb_rmax[r]), int(bb_rmax[nr]))
            new_cmin = min(int(bb_cmin[r]), int(bb_cmin[nr]))
            new_cmax = max(int(bb_cmax[r]), int(bb_cmax[nr]))
            bbox_cells = (new_rmax - new_rmin + 1) * (new_cmax - new_cmin + 1)
            fill = (int(sizes[r]) + int(sizes[nr])) / bbox_cells
            if fill < merge_fill_minimum:
                # Drop: leave r as its own standalone tile (parent[r] = r still)
                dropped += 1
                n_to_merge -= 1   # one fewer merge needed; this tile counts itself
                continue

        # Merge r into nr
        parent[r] = nr
        sizes[nr] += sizes[r]

        total = cnt[nr] + cnt[r]
        if total > 0:
            c_row[nr] = (c_row[nr] * cnt[nr] + c_row[r] * cnt[r]) / total
            c_col[nr] = (c_col[nr] * cnt[nr] + c_col[r] * cnt[r]) / total
            cnt[nr] = total

        if do_fill_screen:
            bb_rmin[nr] = min(bb_rmin[nr], bb_rmin[r])
            bb_rmax[nr] = max(bb_rmax[nr], bb_rmax[r])
            bb_cmin[nr] = min(bb_cmin[nr], bb_cmin[r])
            bb_cmax[nr] = max(bb_cmax[nr], bb_cmax[r])

        merged += 1

    msg = f'  Force-merged {merged}'
    if dropped:
        msg += f', dropped {dropped} (fill screen)'
    msg += f' super-components -> {n_remaining - merged} remain'
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Phase 4 — Compute extents and assign M-codes
# ---------------------------------------------------------------------------

def compute_extents(compact_2d: np.ndarray, parent: np.ndarray,
                    sizes: np.ndarray, nrows: int, ncols: int,
                    ) -> list[dict]:
    """
    For each final super-component root, compute geographic bounding box.
    Returns list sorted by cell count descending.
    """
    n_comp = len(parent)
    roots = np.array([_find_root(i, parent) for i in range(n_comp)], dtype=np.int32)

    # Map compact_2d (with -1/-2 for non-active) to final root IDs
    flat = compact_2d.ravel()
    land = flat >= 0
    final_flat = flat.copy().astype(np.int64)
    final_flat[land] = roots[flat[land]]
    final_2d = final_flat.reshape(nrows, ncols)

    unique_roots = np.unique(roots)
    print(f"  {len(unique_roots):,} super-components after aggregation", flush=True)

    flat_land = np.where(final_2d.ravel() >= 0)[0]
    land_roots = final_2d.ravel()[flat_land]
    sort_idx = np.argsort(land_roots, kind='stable')
    s_roots = land_roots[sort_idx]
    s_rows = (flat_land[sort_idx] // ncols).astype(np.int32)
    s_cols = (flat_land[sort_idx] % ncols).astype(np.int32)

    boundaries = np.searchsorted(s_roots, unique_roots)
    boundaries = np.append(boundaries, len(s_roots))

    basins = []
    for i, root in enumerate(unique_roots.tolist()):
        s, e = int(boundaries[i]), int(boundaries[i + 1])
        if e <= s:
            continue   # root has no live cells (all marked -2 by a drop step)
        r, c = s_rows[s:e], s_cols[s:e]
        xmin = GLOBAL_XMIN + c.min() * CELL_SIZE
        xmax = GLOBAL_XMIN + (c.max() + 1) * CELL_SIZE
        ymax = GLOBAL_YMAX - r.min() * CELL_SIZE
        ymin = GLOBAL_YMAX - (r.max() + 1) * CELL_SIZE
        bbox_cols = int(round((xmax - xmin) / CELL_SIZE))
        bbox_rows = int(round((ymax - ymin) / CELL_SIZE))
        bbox_cells = bbox_cols * bbox_rows
        n_cells = int(sizes[root])
        basins.append({
            'root': root,
            'n_cells': n_cells,
            'xmin': xmin,
            'xmax': xmax,
            'ymax': ymax,
            'ymin': ymin,
            'bbox_cells': bbox_cells,
            'fill_pct': 100.0 * n_cells / bbox_cells if bbox_cells > 0 else 0.0,
        })

    basins.sort(key=lambda b: b['n_cells'], reverse=True)
    return basins


def snap_extents_to_grid(basins: list[dict], snap_cellsize: float,
                         nrows: int, ncols: int) -> None:
    """
    Enlarge each tile bounding box outward so its edges fall on the coarse input
    grid defined by `snap_cellsize` (degrees), aligned to the global raster origin.

    Why this is needed
    ------------------
    PCR-GLOBWB resamples coarse forcing/parameter maps onto each tile's fine clone
    by an integer factor = coarse_cellsize / clone_cellsize (e.g. 6 for a 30 arcmin
    input on a 5 arcmin clone). The regrid inflates a cropped coarse array of
    ceil(rowsClone / factor) cells back up by `factor`, so the result matches the
    clone ONLY when rowsClone and colsClone are exact multiples of `factor`.
    Otherwise pcr.numpy2pcr raises
        "Number of rows/columns from input array (N) and current raster (M) are
         different".
    The global clone (2160 x 4320) is divisible by 6 so a serial run never trips;
    arbitrary per-tile bounding boxes are not, which is what breaks parallel runs.

    Snapping every extent to whole coarse cells guarantees the divisibility. The
    partition of land cells is unchanged; the clone box just gains a few masked
    border cells. Modifies `basins` in place and recomputes bbox_cells / fill_pct.
    """
    if not snap_cellsize or snap_cellsize <= 0:
        return

    ratio = snap_cellsize / CELL_SIZE
    if abs(ratio - round(ratio)) > 1e-6:
        print(f"WARNING: --snap_cellsize {snap_cellsize} is not an integer "
              f"multiple of the clone cell size {CELL_SIZE:.8f}; snapping cannot "
              f"guarantee clone divisibility.", file=sys.stderr, flush=True)

    global_xmax = GLOBAL_XMIN + ncols * CELL_SIZE
    global_ymin = GLOBAL_YMAX - nrows * CELL_SIZE

    def snap_down(v: float, origin: float) -> float:
        return origin + math.floor((v - origin) / snap_cellsize + 1e-9) * snap_cellsize

    def snap_up(v: float, origin: float) -> float:
        return origin + math.ceil((v - origin) / snap_cellsize - 1e-9) * snap_cellsize

    for b in basins:
        b['xmin'] = max(GLOBAL_XMIN, snap_down(b['xmin'], GLOBAL_XMIN))
        b['xmax'] = min(global_xmax, snap_up(b['xmax'], GLOBAL_XMIN))
        b['ymin'] = max(global_ymin, snap_down(b['ymin'], GLOBAL_YMAX))
        b['ymax'] = min(GLOBAL_YMAX, snap_up(b['ymax'], GLOBAL_YMAX))
        bbox_cols = int(round((b['xmax'] - b['xmin']) / CELL_SIZE))
        bbox_rows = int(round((b['ymax'] - b['ymin']) / CELL_SIZE))
        b['bbox_cells'] = bbox_cols * bbox_rows
        b['fill_pct'] = (100.0 * b['n_cells'] / b['bbox_cells']
                         if b['bbox_cells'] > 0 else 0.0)


def _find_root(x: int, parent: np.ndarray) -> int:
    """Non-mutating root finder."""
    while parent[x] != x:
        x = parent[x]
    return x


def assign_codes(basins: list[dict]) -> list[dict]:
    """Assign M01, M02, ... codes in descending cell-count order."""
    for i, b in enumerate(basins):
        b['code'] = f"M{i + 1:02d}"
    return basins


# ---------------------------------------------------------------------------
# Debug helpers
# ---------------------------------------------------------------------------

def _print_size_stats(sizes_active: np.ndarray, label: str,
                      ub_cells: int | None = None) -> None:
    """Print a one-line size summary plus top-5 and ub_cells violation count."""
    n = len(sizes_active)
    if n == 0:
        print(f'  [{label}] no active components', flush=True)
        return
    total = int(sizes_active.sum())
    print(
        f'  [{label}]  n={n:,}  total={total:,}  '
        f'min={int(sizes_active.min()):,}  '
        f'max={int(sizes_active.max()):,}  '
        f'mean={total / n:,.0f}  '
        f'p50={int(np.median(sizes_active)):,}  '
        f'p90={int(np.percentile(sizes_active, 90)):,}  '
        f'p99={int(np.percentile(sizes_active, 99)):,}',
        flush=True,
    )
    top5 = np.sort(sizes_active)[::-1][:5]
    print(f'    top-5 sizes: {", ".join(f"{int(s):,}" for s in top5)}',
          flush=True)
    if ub_cells is not None:
        n_viol = int((sizes_active > ub_cells).sum())
        pct = 100.0 * n_viol / n if n else 0.0
        print(f'    > ub_cells ({ub_cells:,}): {n_viol:,} / {n:,} ({pct:.1f}%)',
              flush=True)


# ---------------------------------------------------------------------------
# Reporting and output
# ---------------------------------------------------------------------------

def print_summary(basins: list[dict], label: str = '',
                  n_land_orig: int | None = None) -> None:
    total = sum(b['n_cells'] for b in basins)
    n = len(basins)
    avg = total / n if n else 0
    mx = basins[0]['n_cells'] if basins else 0
    mn = basins[-1]['n_cells'] if basins else 0

    fills = [b['fill_pct'] for b in basins]
    fill_min  = min(fills) if fills else 0.0
    fill_max  = max(fills) if fills else 0.0
    fill_mean = sum(fills) / n if n else 0.0
    worst_fill = min(basins, key=lambda b: b['fill_pct']) if basins else None

    print(f"\n{'=' * 85}")
    if label:
        print(f"  {label}")
    print(f"  Subdomains   : {n}")
    if n_land_orig is not None and n_land_orig > 0:
        pct_kept = 100.0 * total / n_land_orig
        print(f"  Total land   : {total:,} cells ({pct_kept:.1f}% of original {n_land_orig:,})")
    else:
        print(f"  Total land   : {total:,} cells")
    print(f"  Target/tile  : {avg:,.0f} cells  (total / N)")
    print(f"  Largest      : {mx:,} cells  ({100 * mx / total:.1f}%)")
    print(f"  Smallest     : {mn:,} cells  ({100 * mn / total:.1f}%)")
    print(f"  Cell imbalance: {mx / avg:.2f}x  (largest / average)")
    print(f"  Bbox fill    : min={fill_min:.1f}%  mean={fill_mean:.1f}%  max={fill_max:.1f}%")
    if worst_fill is not None:
        print(f"  Worst fill   : {worst_fill['code']}  "
              f"{worst_fill['n_cells']:,} active / {worst_fill['bbox_cells']:,} bbox  "
              f"({worst_fill['fill_pct']:.1f}%)")
    print(f"\n  {'Code':<6} {'Cells':>9} {'%tot':>5}  "
          f"{'xmin':>8} {'ymin':>7} {'xmax':>8} {'ymax':>7}  "
          f"{'BboxCells':>10} {'Fill%':>6}")
    print(f"  {'-'*6} {'-'*9} {'-'*5}  "
          f"{'-'*8} {'-'*7} {'-'*8} {'-'*7}  "
          f"{'-'*10} {'-'*6}")
    for b in basins:
        print(f"  {b['code']:<6} {b['n_cells']:>9,} {100 * b['n_cells'] / total:5.1f}%  "
              f"{b['xmin']:8.3f} {b['ymin']:7.3f} {b['xmax']:8.3f} {b['ymax']:7.3f}  "
              f"{b['bbox_cells']:>10,} {b['fill_pct']:6.1f}%")
    print(f"{'=' * 85}\n")


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


def write_partition_npz(
        compact_2d: np.ndarray,
        parent: np.ndarray,
        basins: list[dict],
        nrows: int,
        ncols: int,
        path: str
) -> None:
    """
    Save the final tile partition to a compressed numpy archive for use by
    create_tile_clone_maps.py --partition.

    Arrays stored
    -------------
    tile_map   : (nrows, ncols) int16
                 Tile index (0-based, matching `codes` order) for each cell,
                 -1 for ocean, -2 for filtered/dropped cells.
    codes      : 1-D array of tile code strings  (e.g. ['M01', 'M02', ...])
    xmin/ymin/xmax/ymax : 1-D float64 arrays of tile bounding boxes
    cell_size  : scalar float64 (degrees)
    global_xmin/global_ymax : scalar float64 (reference corner)
    """
    n_comp = len(parent)
    roots_map = np.array([_find_root(i, parent) for i in range(n_comp)],
                         dtype=np.int32)

    # Build code -> tile index lookup
    code_list = [b['code'] for b in basins]
    root_to_idx: dict[int, int] = {b['root']: i for i, b in enumerate(basins)}

    flat = compact_2d.ravel()
    tile_flat = np.full(len(flat), -1, dtype=np.int16)
    land = flat >= 0
    land_roots = roots_map[flat[land]]
    # Map roots that are in the final basins list; others remain -1 (dropped)
    tile_flat_land = np.full(land_roots.shape, -2, dtype=np.int16)
    for root, idx in root_to_idx.items():
        tile_flat_land[land_roots == root] = np.int16(idx)
    tile_flat[np.where(land)[0]] = tile_flat_land

    # ocean cells (compact_2d == -1) → tile_flat stays -1
    # filtered/dropped cells (compact_2d == -2) → tile_flat stays -1
    # Re-mark filtered as -2 to distinguish from ocean
    tile_flat[flat == -2] = -2

    tile_map = tile_flat.reshape(nrows, ncols)

    np.savez_compressed(
        path,
        tile_map=tile_map,
        codes=np.array(code_list, dtype=object),
        xmin=np.array([b['xmin'] for b in basins]),
        ymin=np.array([b['ymin'] for b in basins]),
        xmax=np.array([b['xmax'] for b in basins]),
        ymax=np.array([b['ymax'] for b in basins]),
        cell_size=np.float64(CELL_SIZE),
        global_xmin=np.float64(GLOBAL_XMIN),
        global_ymax=np.float64(GLOBAL_YMAX),
    )
    print(f"Partition NPZ written: {path}", flush=True)


# ---------------------------------------------------------------------------
# Partition image
# ---------------------------------------------------------------------------

def save_partition_image(compact_2d: np.ndarray, parent: np.ndarray,
                         output_path: str,
                         annotate: bool = True,
                         basins: 'list[dict] | None' = None) -> None:
    """
    Render the current partition as a color-coded PNG.

    Color legend
    ------------
    White      : ocean  (compact_2d == -1)
    Black      : filtered / inactive  (compact_2d == -2)
    Colored    : active tile (one color per tile, cycling through tab20)

    Each active tile is annotated at its cell-count barycenter with its
    number of cells (formatted as e.g. "123k" or "1.2M") when annotate=True
    and the tile count is <= 300.  For stages with thousands of components
    annotation is skipped automatically to keep rendering fast.

    When basins is supplied (list of dicts with xmin/xmax/ymin/ymax/code),
    the geographic bounding box of each tile is drawn as a rectangle.
    """
    try:
        from .plotting import pyplot, save_figure, get_cmap
        plt = pyplot()
    except ImportError:
        print("WARNING: matplotlib not available — skipping image.", file=sys.stderr)
        return

    n_comp = len(parent)
    roots_map = np.array([_find_root(i, parent) for i in range(n_comp)], dtype=np.int32)

    flat = compact_2d.ravel()
    land = flat >= 0

    final_flat = flat.copy().astype(np.int64)
    final_flat[land] = roots_map[flat[land]]
    final_2d = final_flat.reshape(compact_2d.shape)

    unique_tiles = np.unique(final_2d[final_2d >= 0])
    n_tiles = len(unique_tiles)

    # Map tile ID -> color index (0..19, cycling)
    max_tile = int(unique_tiles.max()) + 1 if n_tiles > 0 else 1
    tile_ci = np.full(max_tile, -1, dtype=np.int32)
    for i, t in enumerate(unique_tiles):
        tile_ci[t] = i % 20

    nrows, ncols = compact_2d.shape
    img = np.ones((nrows, ncols, 3), dtype=np.float32)   # white = ocean

    # Black for filtered / inactive cells
    filtered_mask = (final_2d == -2)
    img[filtered_mask] = [0.0, 0.0, 0.0]

    cmap   = get_cmap('tab20', 20)
    colors = np.array([cmap(i)[:3] for i in range(20)], dtype=np.float32)

    # Assign colors in batches of same color-index (avoids per-tile loop)
    for ci in range(20):
        tiles_ci = unique_tiles[tile_ci[unique_tiles] == ci]
        if len(tiles_ci) == 0:
            continue
        mask = np.isin(final_2d, tiles_ci)
        img[mask] = colors[ci]

    # Compute barycenters and cell counts for each tile (vectorized)
    flat_land_idx  = np.where(final_2d.ravel() >= 0)[0]
    tile_ids       = final_2d.ravel()[flat_land_idx].astype(np.int64)
    rows_land      = flat_land_idx // ncols
    cols_land      = flat_land_idx  % ncols

    counts  = np.bincount(tile_ids, minlength=max_tile)
    row_sum = np.bincount(tile_ids, weights=rows_land.astype(np.float64),
                          minlength=max_tile)
    col_sum = np.bincount(tile_ids, weights=cols_land.astype(np.float64),
                          minlength=max_tile)

    def _fmt_cells(n: int) -> str:
        if n >= 1_000_000:
            return f"{n / 1_000_000:.1f}M"
        if n >= 1_000:
            return f"{n // 1_000}k"
        return str(n)

    fig, ax = plt.subplots(figsize=(18, 9), dpi=150)
    ax.imshow(img, origin='upper', aspect='equal', interpolation='nearest')

    if annotate and n_tiles <= 300:
        for t in unique_tiles.tolist():
            cnt = int(counts[t])
            if cnt == 0:
                continue
            cy = row_sum[t] / cnt   # image row  (y-axis in imshow)
            cx = col_sum[t] / cnt   # image col  (x-axis in imshow)
            r, g, b = colors[tile_ci[t]]
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            txt_color = 'white' if luminance < 0.5 else 'black'
            ax.text(cx, cy, _fmt_cells(cnt),
                    ha='center', va='center',
                    fontsize=5, fontweight='bold', color=txt_color,
                    clip_on=True)

    if basins:
        from matplotlib.patches import Rectangle
        for b in basins:
            col_left   = (b['xmin'] - GLOBAL_XMIN) / CELL_SIZE
            col_right  = (b['xmax'] - GLOBAL_XMIN) / CELL_SIZE
            row_top    = (GLOBAL_YMAX - b['ymax']) / CELL_SIZE
            row_bottom = (GLOBAL_YMAX - b['ymin']) / CELL_SIZE
            rect = Rectangle(
                (col_left, row_top),
                col_right - col_left,
                row_bottom - row_top,
                linewidth=0.6,
                edgecolor='black',
                facecolor='none',
                alpha=0.7,
            )
            ax.add_patch(rect)
            if annotate and n_tiles <= 300:
                ax.text(
                    (col_left + col_right) / 2,
                    row_top + 3,
                    b['code'],
                    ha='center', va='top',
                    fontsize=4, color='black',
                    clip_on=True,
                )

    ax.axis('off')
    ax.set_title(f'Land partition — {n_tiles} tiles', fontsize=13, pad=6)
    plt.tight_layout(pad=0.3)
    save_figure(plt, fig, output_path, dpi=150, announce=False)
    print(f"Partition image saved: {output_path}", flush=True)


def _save_debug_snapshot(compact_2d: np.ndarray, parent: np.ndarray,
                         stage: int, label: str, prefix: str) -> None:
    path = f'{prefix}_{stage:02d}_{label}.png'
    n_tiles = len(np.unique(parent))
    # Skip per-tile text for stages with many components (slow and unreadable)
    save_partition_image(compact_2d, parent, path, annotate=(n_tiles <= 300))


def plot_merge_dendrogram(n_comp: int,
                          initial_sizes: np.ndarray,
                          merge_history: list,
                          ub_cells: int,
                          n_target: int,
                          output_path: str) -> None:
    """
    Two-panel figure showing the full heap-merge sequence.

    Left panel — dendrogram
      x-axis : leaf index (one leaf per post-split component, ordered by
                post-order traversal of the merge tree)
      y-axis : combined cell count of the merged cluster (log scale)
      Blue links : compliant merges (combined <= ub_cells, from heap_ok)
      Red  links : violating merges (combined >  ub_cells, from heap_viol)
      Solid orange line  : ub_cells threshold, annotated with the count of
                           merges that exceeded it
      Dashed green line  : height at which n_target tiles is first achieved

    Right panel — step function (right-continuous)
      x-axis : cell count of the largest tile at each point in the merge
               sequence (log scale)
      y-axis : number of tiles remaining
      Vertical orange line : ub_cells
      Horizontal green line : n_target

    merge_history entries: (cid_a, cid_b, combined_size, running_max, is_viol)
    """
    try:
        from .plotting import pyplot, save_figure
        plt = pyplot()
        from matplotlib.collections import LineCollection
    except ImportError:
        print('WARNING: matplotlib not available — skipping dendrogram.',
              file=sys.stderr)
        return

    n_merges = len(merge_history)
    if n_merges == 0:
        print('WARNING: empty merge history — nothing to plot.', file=sys.stderr)
        return

    # ------------------------------------------------------------------
    # Build flat node arrays for the merge tree
    # Indices 0 .. n_comp-1  : leaves (original components)
    # Indices n_comp .. n_comp+n_merges-1 : internal nodes (one per merge)
    # ------------------------------------------------------------------
    total_nodes = n_comp + n_merges
    node_size  = np.empty(total_nodes, dtype=np.int64)
    node_left  = np.full(total_nodes, -1, dtype=np.int32)   # -1 = no child
    node_right = np.full(total_nodes, -1, dtype=np.int32)
    node_viol  = np.zeros(total_nodes, dtype=bool)

    node_size[:n_comp] = initial_sizes.astype(np.int64)

    for k, (cid_a, cid_b, combined, _max_sz, _is_viol) in enumerate(merge_history):
        idx = n_comp + k
        node_size[idx]  = combined
        node_left[idx]  = cid_a
        node_right[idx] = cid_b
        # Colour by whether the merge exceeded the ORIGINAL ub_cells threshold
        # (independent of which round it occurred in).
        node_viol[idx]  = combined > ub_cells

    # ------------------------------------------------------------------
    # Assign x-positions via iterative post-order traversal
    # ------------------------------------------------------------------
    node_x = np.zeros(total_nodes, dtype=np.float64)

    # Root nodes = nodes never referenced as a child
    is_child = np.zeros(total_nodes, dtype=bool)
    for k in range(n_comp, total_nodes):
        la, ra = node_left[k], node_right[k]
        if la >= 0:
            is_child[la] = True
        if ra >= 0:
            is_child[ra] = True
    root_ids = np.where(~is_child)[0].tolist()

    leaf_counter = [0]

    def _assign_x(root_id: int) -> None:
        stack = [(root_id, False)]
        while stack:
            nid, processed = stack.pop()
            if nid < n_comp:                     # leaf
                node_x[nid] = leaf_counter[0]
                leaf_counter[0] += 1
            elif processed:                      # internal, children done
                la, ra = node_left[nid], node_right[nid]
                node_x[nid] = (node_x[la] + node_x[ra]) / 2.0
            else:
                stack.append((nid, True))        # revisit after children
                ra = node_right[nid]
                la = node_left[nid]
                if ra >= 0:
                    stack.append((ra, False))
                if la >= 0:
                    stack.append((la, False))

    for rid in root_ids:
        _assign_x(rid)

    n_leaves = leaf_counter[0]

    # ------------------------------------------------------------------
    # Build line segments for dendrogram (using LineCollection for speed)
    # ------------------------------------------------------------------
    min_y = max(1, int(node_size[:n_comp].min()))

    ok_segs:   list[tuple] = []
    viol_segs: list[tuple] = []

    for k in range(n_comp, total_nodes):
        la, ra = node_left[k], node_right[k]
        h  = int(node_size[k])
        xl = node_x[la]
        xr = node_x[ra]
        hl = max(int(node_size[la]), min_y)
        hr = max(int(node_size[ra]), min_y)
        segs = ok_segs if not node_viol[k] else viol_segs
        segs.append([(xl, hl), (xl, h)])   # left vertical
        segs.append([(xr, hr), (xr, h)])   # right vertical
        segs.append([(xl, h),  (xr, h)])   # horizontal

    col_ok   = '#4C78A8'
    col_viol = '#E45756'

    # ------------------------------------------------------------------
    # Figure layout
    # ------------------------------------------------------------------
    fig_w = max(16.0, n_leaves * 0.06 + 5.0)
    fig, (ax_d, ax_s) = plt.subplots(
        1, 2, figsize=(fig_w, 8),
        gridspec_kw={'width_ratios': [4, 1]},
        constrained_layout=True,
    )

    # ------------------------------------------------------------------
    # Dendrogram panel
    # ------------------------------------------------------------------
    if ok_segs:
        ax_d.add_collection(
            LineCollection(ok_segs, colors=col_ok, linewidths=0.7,
                           label='heap_ok  (compliant)'))
    if viol_segs:
        ax_d.add_collection(
            LineCollection(viol_segs, colors=col_viol, linewidths=0.7,
                           label='heap_viol (violating)'))

    # ub_cells solid horizontal line
    n_above_ub = sum(1 for e in merge_history if e[2] > ub_cells)
    ax_d.axhline(ub_cells, color='tab:orange', lw=1.2, ls='-',
                 label=f'ub_cells = {ub_cells:,}  ({n_above_ub} merges above)')

    # Dashed line at the height where n_target tiles is first achieved.
    # That is the height of the (n_comp - n_target)-th merge (0-indexed).
    idx_target = n_comp - n_target - 1
    if 0 <= idx_target < n_merges:
        target_h = merge_history[idx_target][2]
        ax_d.axhline(target_h, color='tab:green', lw=1.2, ls='--',
                     label=f'n_tiles = {n_target} at height {target_h:,}')
    else:
        target_h = None
        remaining = n_comp - n_merges
        ax_d.text(0.5, 0.97,
                  f'n_tiles = {n_target} not reached within {n_merges} merges '
                  f'({remaining} tiles remain)',
                  transform=ax_d.transAxes, ha='center', va='top',
                  color='tab:green', fontsize=8)

    ax_d.set_yscale('log')
    ax_d.set_xlim(-0.5, n_leaves - 0.5)
    ax_d.autoscale_view()
    ax_d.set_xlabel(f'{n_leaves} components (post-split leaves)', fontsize=9)
    ax_d.set_ylabel('Combined cell count (log scale)', fontsize=9)
    ax_d.set_title(
        f'Heap-merge dendrogram  —  {n_comp} components, '
        f'target {n_target} tiles\n'
        f'{n_merges} merges: '
        f'{n_merges - n_above_ub} compliant + {n_above_ub} violating',
        fontsize=9,
    )
    ax_d.tick_params(axis='x', bottom=False, labelbottom=False)
    ax_d.legend(fontsize=7, loc='upper left')

    # ------------------------------------------------------------------
    # Step-function panel: largest tile size (x) vs tiles remaining (y)
    # ------------------------------------------------------------------
    x_step = [int(initial_sizes.max())]
    y_step = [n_comp]
    for i, (_ca, _cb, _comb, max_sz, _v) in enumerate(merge_history):
        x_step.append(int(max_sz))
        y_step.append(n_comp - i - 1)

    ax_s.step(x_step, y_step, where='post', color='#333333', lw=1.2)
    ax_s.axvline(ub_cells, color='tab:orange', lw=1, ls='-',
                 label='ub_cells')
    ax_s.axhline(n_target, color='tab:green', lw=1, ls='--',
                 label=f'n_tiles={n_target}')
    if target_h is not None:
        ax_s.axvline(target_h, color='tab:green', lw=0.8, ls=':')

    ax_s.set_xscale('log')
    ax_s.set_xlabel('Largest tile (cells)', fontsize=9)
    ax_s.set_ylabel('Tiles remaining', fontsize=9)
    ax_s.yaxis.set_label_position('right')
    ax_s.yaxis.tick_right()
    ax_s.set_title('Largest tile\nvs tiles remaining', fontsize=9)
    ax_s.legend(fontsize=6, loc='upper right')

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    save_figure(plt, fig, output_path, dpi=150, announce=False)
    print(f'Merge dendrogram saved: {output_path}', flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--ldd', required=True,
                   help='Path to the global 05min PCRaster LDD map')
    p.add_argument('--n_tiles', type=int, default=None,
                   help='Target number of subdomains after aggregation. '
                        'If omitted, only raw component statistics are printed.')
    p.add_argument('--lb_cells', type=int, default=None,
                   help='Drop components with fewer than this many cells before '
                        'aggregation. Those cells are marked inactive (-2) and '
                        'excluded from simulation and extents output.')
    p.add_argument('--ub_cells', type=int, default=None,
                   help='Split components with more than this many cells using '
                        'tree centroid bisection before aggregation.')
    p.add_argument('--output_extents', default=None,
                   help='Write extents CSV (requires --n_tiles). '
                        'Compatible with create_tile_clone_maps.py.')
    p.add_argument('--output_partition', default=None, metavar='PATH',
                   help='Write a compressed numpy archive (.npz) containing '
                        'the full tile_map, tile codes, bounding boxes, and '
                        'grid metadata.  Required by create_tile_clone_maps.py '
                        '--partition to generate per-tile landmask maps.')
    p.add_argument('--output_image', default=None,
                   help='Save a color-coded partition PNG to this path.')
    p.add_argument('--verify_tree', action='store_true',
                   help='Check tree structure (1 pit + S-1 edges) for the '
                        '5 largest components.')
    p.add_argument('--rep_max', type=int, default=2,
                   help='Maximum number of aggregation rounds when --ub_cells '
                        'is set.  In each round the heap is rebuilt from the '
                        'current component state and the threshold is doubled. '
                        'After rep_max rounds, non-strict mode falls back to '
                        'an unconstrained pass; strict mode raises an error. '
                        '(default: 2)')
    p.add_argument('--strict', action='store_true', default=False,
                   help='Raise an error if any merge during aggregation or '
                        'force-merge would exceed --ub_cells, rather than '
                        'falling back to the smallest-violation merge. '
                        'Has no effect when --ub_cells is not set.')
    p.add_argument('--output_dendrogram', default=None, metavar='PATH',
                   help='Save a merge-dendrogram PNG showing the full heap '
                        'aggregation sequence (requires --ub_cells and '
                        '--n_tiles).  Left panel: dendrogram with blue '
                        '(compliant) and red (violating) merge links.  '
                        'Right panel: step function of largest tile size vs '
                        'tiles remaining.')
    p.add_argument('--lb_disconnected', type=int, default=None,
                   metavar='N',
                   help='Before force-merge, automatically drop topologically '
                        'disconnected components (no 8-connectivity neighbours '
                        'after heap aggregation) with fewer than N cells. '
                        'Their cells are marked inactive and excluded from the '
                        'partition entirely.')
    p.add_argument('--merge_fill_minimum', type=float, default=None,
                   metavar='F',
                   help='During force-merge, drop a component instead of '
                        'merging it if the resulting combined bounding-box '
                        'fill fraction would fall below F (0–1). Applies to '
                        'all components regardless of size.')
    p.add_argument('--snap_cellsize', type=float, default=0.5, metavar='DEG',
                   help='Snap every tile bounding box outward to this coarse '
                        'grid size (degrees), aligned to the global origin, so '
                        'each clone\'s rows/cols are an exact multiple of the '
                        'coarse->clone resample factor. This is REQUIRED for '
                        'parallel runs: PCR-GLOBWB cannot map coarse inputs onto '
                        'a tile clone whose dimensions are not a multiple of that '
                        'factor (pcr.numpy2pcr raises a row/column mismatch). '
                        'Set to the coarsest input resolution used by the run '
                        '(default 0.5 = 30 arcmin). Use 0 to disable snapping.')
    p.add_argument('--debug_stages', default=None, metavar='PREFIX',
                   help='Save a PNG snapshot of the partition after each '
                        'processing stage. Images are written as '
                        'PREFIX_01_after_ub_split.png, '
                        'PREFIX_02_after_lb_filter.png, '
                        'PREFIX_03_after_heap_merge.png, and '
                        'PREFIX_04_after_force_merge.png. '
                        'Size statistics are always printed to stdout '
                        'regardless of this flag.')
    return p


def compute_ldd_basins(ldd, n_tiles=None, lb_cells=None, ub_cells=None, output_extents=None,
                       output_partition=None, output_image=None, verify_tree=False, rep_max=2,
                       strict=False, output_dendrogram=None, lb_disconnected=None,
                       merge_fill_minimum=None, snap_cellsize=0.5, debug_stages=None) -> None:
    """LDD-based domain decomposition (importable form of the original ``main``).

    Parameters mirror the CLI flags one-for-one; the original body is preserved verbatim by binding the
    parameters into an ``args`` namespace. Driven by the root shim ``compute_ldd_basins.py`` (argparse) and
    the pipeline stage ``src/stages/compute_basins.py`` (Fire).
    """
    args = SimpleNamespace(
        ldd=ldd, n_tiles=n_tiles, lb_cells=lb_cells, ub_cells=ub_cells,
        output_extents=output_extents, output_partition=output_partition, output_image=output_image,
        verify_tree=verify_tree, rep_max=rep_max, strict=strict, output_dendrogram=output_dendrogram,
        lb_disconnected=lb_disconnected, merge_fill_minimum=merge_fill_minimum,
        snap_cellsize=snap_cellsize, debug_stages=debug_stages,
    )

    if args.output_extents and not args.n_tiles:
        sys.exit('--output_extents requires --n_tiles')

    if not os.path.isfile(args.ldd):
        sys.exit(f'LDD map not found: {args.ldd}')

    # ------------------------------------------------------------------
    # Phase 1: load LDD and compute raw flow-graph components
    # ------------------------------------------------------------------
    print(f'Loading LDD: {args.ldd}', flush=True)
    ldd, nrows, ncols = load_ldd(args.ldd)
    n_land = int((ldd != LDD_MV).sum())
    print(f'  Grid: {nrows} x {ncols},  land cells: {n_land:,}', flush=True)

    print('Building flow graph and computing connected components ...', flush=True)
    labels = compute_flow_components(ldd, nrows, ncols)
    compact_2d, n_comp, sizes = make_compact_labels(labels, ldd, nrows, ncols)
    del labels
    print(f'  {n_comp:,} land-cell components found', flush=True)

    # Optional tree verification (before any splits)
    if args.verify_tree:
        verify_tree_structure(ldd, compact_2d, sizes)

    # ------------------------------------------------------------------
    # Optional Phase 2a: split oversized components
    # ------------------------------------------------------------------
    if args.ub_cells is not None:
        print(f'Splitting components > {args.ub_cells:,} cells ...', flush=True)
        compact_2d, n_comp, sizes = apply_ub_splits(
            ldd, compact_2d, nrows, ncols, sizes, args.ub_cells)
        compact_2d, n_comp, sizes = recompact_labels(compact_2d, sizes)
        print(f'  {n_comp:,} components after UB split', flush=True)
        _print_size_stats(sizes, 'after_ub_split', ub_cells=args.ub_cells)
        if args.debug_stages:
            _save_debug_snapshot(compact_2d, np.arange(n_comp, dtype=np.int32),
                                 1, 'after_ub_split', args.debug_stages)

    # ------------------------------------------------------------------
    # Optional Phase 2b: filter undersized components
    # ------------------------------------------------------------------
    if args.lb_cells is not None:
        print(f'Filtering components < {args.lb_cells:,} cells ...', flush=True)
        compact_2d = apply_lb_filter(compact_2d, sizes, args.lb_cells)
        compact_2d, n_comp, sizes = recompact_labels(compact_2d, sizes)
        print(f'  {n_comp:,} active components after LB filter', flush=True)
        _print_size_stats(sizes, 'after_lb_filter', ub_cells=args.ub_cells)
        if args.debug_stages:
            _save_debug_snapshot(compact_2d, np.arange(n_comp, dtype=np.int32),
                                 2, 'after_lb_filter', args.debug_stages)

    # ------------------------------------------------------------------
    # Report-only mode (no --n_tiles)
    # ------------------------------------------------------------------
    # FIX: cannot reach code since must provide --n_tiles arg
    if args.n_tiles is None:
        parent = np.arange(n_comp, dtype=np.int32)
        basins = compute_extents(compact_2d, parent, sizes, nrows, ncols)
        basins = assign_codes(basins)
        print_summary(basins[:30],
                      label=f'Top 30 of {n_comp:,} components (no aggregation)')
        print('Run with --n_tiles N to aggregate into N subdomains.')
        if args.output_image:
            save_partition_image(compact_2d, parent, args.output_image)
        return

    # ------------------------------------------------------------------
    # Phase 3: build adjacency and aggregate
    # ------------------------------------------------------------------

    # Upfront feasibility check: warn (or error on --strict) when the
    # requested n_tiles cannot accommodate all active cells within ub_cells.
    if args.ub_cells is not None:
        n_active = int((compact_2d >= 0).sum())
        n_min_tiles = (n_active + args.ub_cells - 1) // args.ub_cells
        if n_min_tiles > args.n_tiles:
            msg = (
                f'Feasibility: {n_active:,} active cells / '
                f'ub_cells={args.ub_cells:,} requires >= {n_min_tiles} tiles; '
                f'requested n_tiles={args.n_tiles} guarantees violation.'
            )
            if args.strict:
                sys.exit(f'ERROR: {msg}')
            print(f'WARNING: {msg}', file=sys.stderr, flush=True)

    print('Building 8-connectivity adjacency graph ...', flush=True)
    adj, pair_a, pair_b = build_adjacency(compact_2d, n_comp)
    print(f'  {len(pair_a):,} unique adjacent component pairs', flush=True)

    print(f'Aggregating {n_comp:,} components -> {args.n_tiles} subdomains ...',
          flush=True)
    # Snapshot initial sizes and allocate history list before aggregation
    # modifies sizes in-place.
    sizes_pre_agg = sizes.copy() if args.output_dendrogram else None
    merge_hist    = []            if args.output_dendrogram else None

    if args.debug_stages:
        def _round_cb(rep: int, rep_max: int, parent_snap: np.ndarray,
                      ub: int, is_fallback: bool) -> None:
            label = (f'round_{rep + 1:02d}_of_{rep_max:02d}_ub{ub:,}'
                     if not is_fallback else
                     f'round_fallback_ub_unconstrained')
            _save_debug_snapshot(compact_2d, parent_snap,
                                 rep + 5, label, args.debug_stages)
    else:
        _round_cb = None

    parent = aggregate_components(n_comp, sizes, adj, args.n_tiles,
                                  ub_cells=args.ub_cells, strict=args.strict,
                                  merge_history=merge_hist,
                                  rep_max=args.rep_max,
                                  round_callback=_round_cb,
                                  pair_a=pair_a, pair_b=pair_b,
                                  compact_2d=compact_2d,
                                  nrows=nrows, ncols=ncols,
                                  lb_disconnected=args.lb_disconnected,
                                  merge_fill_minimum=args.merge_fill_minimum)

    # Check how many super-components remain after heap merging
    roots = np.array([_find_root(i, parent) for i in range(n_comp)], dtype=np.int32)
    n_remain = len(np.unique(roots))
    print(f'  {n_remain:,} super-components after heap merging '
          f'(target {args.n_tiles})', flush=True)
    unique_roots_heap = np.unique(roots)
    _print_size_stats(sizes[unique_roots_heap], 'after_heap_merge',
                      ub_cells=args.ub_cells)
    if args.debug_stages:
        _save_debug_snapshot(compact_2d, parent, 3, 'after_heap_merge',
                             args.debug_stages)

    # ------------------------------------------------------------------
    # Phase 3b: force-merge isolated super-components if needed
    # ------------------------------------------------------------------
    if n_remain > args.n_tiles:
        # Drop truly disconnected (no adj neighbours) small components first
        if args.lb_disconnected is not None:
            print('Dropping disconnected small components ...', flush=True)
            _drop_disconnected_small(compact_2d, parent, sizes, adj,
                                     args.lb_disconnected)
            # Recount from compact_2d (ground truth after cells marked -2)
            flat_c = compact_2d.ravel()
            live_comps = np.unique(flat_c[flat_c >= 0].astype(np.int32))
            n_remain = len({_find_root(int(c), parent) for c in live_comps})

        if n_remain > args.n_tiles:
            print(f'Force-merging {n_remain - args.n_tiles} isolated super-components '
                  f'into nearest neighbour ...', flush=True)
            force_merge_to_target(compact_2d, parent, sizes,
                                  args.n_tiles, nrows, ncols,
                                  ub_cells=args.ub_cells, strict=args.strict,
                                  merge_fill_minimum=args.merge_fill_minimum)
        if args.debug_stages:
            _save_debug_snapshot(compact_2d, parent, 4, 'after_force_merge',
                                 args.debug_stages)

    # ------------------------------------------------------------------
    # Phase 3c: ub_cells violation report
    # ------------------------------------------------------------------
    if args.ub_cells is not None:
        roots_final = np.array(
            [_find_root(i, parent) for i in range(n_comp)], dtype=np.int32)
        unique_roots_final = np.unique(roots_final)
        tile_sizes = sizes[unique_roots_final]
        n_violating = int((tile_sizes > args.ub_cells).sum())
        if n_violating:
            max_size = int(tile_sizes.max())
            pct_over = 100.0 * (max_size / args.ub_cells - 1.0)
            print(
                f'  ub_cells violations: {n_violating} tile(s) exceed '
                f'{args.ub_cells:,}; largest is {max_size:,} cells '
                f'({pct_over:.0f}% over bound)',
                flush=True,
            )
            if pct_over > 10.0:
                print(
                    f'WARNING: largest tile exceeds ub_cells by {pct_over:.0f}% '
                    f'(> 10% threshold). Consider increasing --n_tiles or '
                    f'relaxing --ub_cells.',
                    file=sys.stderr,
                )
        else:
            print(f'  All tiles satisfy ub_cells={args.ub_cells:,}.', flush=True)

    # ------------------------------------------------------------------
    # Optional: merge dendrogram
    # ------------------------------------------------------------------
    if args.output_dendrogram:
        if args.ub_cells is None:
            print('WARNING: --output_dendrogram requires --ub_cells; skipping.',
                  file=sys.stderr)
        else:
            plot_merge_dendrogram(
                n_comp, sizes_pre_agg, merge_hist,
                args.ub_cells, args.n_tiles, args.output_dendrogram,
            )

    # ------------------------------------------------------------------
    # Phase 4: extents, summary, outputs
    # ------------------------------------------------------------------
    print('Computing geographic extents ...', flush=True)
    basins = compute_extents(compact_2d, parent, sizes, nrows, ncols)
    if args.snap_cellsize and args.snap_cellsize > 0:
        print(f'Snapping tile extents to the {args.snap_cellsize} deg coarse '
              f'grid (clone divisibility) ...', flush=True)
        snap_extents_to_grid(basins, args.snap_cellsize, nrows, ncols)
    basins = assign_codes(basins)
    print_summary(basins, label=f'Final {len(basins)} subdomains',
                  n_land_orig=n_land)

    if args.output_image:
        save_partition_image(compact_2d, parent, args.output_image, basins=basins)

    if args.output_extents:
        write_extents_csv(basins, args.output_extents, args.ldd)

    if args.output_partition:
        write_partition_npz(compact_2d, parent, basins, nrows, ncols, args.output_partition)


# The CLI lives in the root-level shim ``compute_ldd_basins.py`` (argparse) and the pipeline stage
# ``src/stages/compute_basins.py`` (Fire); this module is import-only.
