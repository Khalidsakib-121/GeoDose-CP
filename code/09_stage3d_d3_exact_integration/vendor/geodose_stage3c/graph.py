from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Dict

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path


@dataclass
class GraphCache:
    n_nodes: int
    adjacency: list[set[int]]
    matrix: csr_matrix
    distances_by_target: Dict[int, np.ndarray]
    edge_hash: str


def _canonical_edge_frame(edges: pd.DataFrame) -> pd.DataFrame:
    if len(edges) == 0:
        return pd.DataFrame(columns=["a", "b"])
    a = np.minimum(edges["source_node"].to_numpy(int), edges["target_node"].to_numpy(int))
    b = np.maximum(edges["source_node"].to_numpy(int), edges["target_node"].to_numpy(int))
    frame = pd.DataFrame({"a": a, "b": b})
    return frame[frame["a"] != frame["b"]].drop_duplicates().sort_values(["a", "b"], kind="mergesort")


def build_graph_cache(units: pd.DataFrame, edges: pd.DataFrame, target_nodes: np.ndarray) -> GraphCache:
    n = int(units["node_index"].max()) + 1
    canonical = _canonical_edge_frame(edges)
    src = canonical["a"].to_numpy(int)
    dst = canonical["b"].to_numpy(int)
    data = np.ones(len(src) * 2, dtype=float)
    rows = np.r_[src, dst]
    cols = np.r_[dst, src]
    matrix = csr_matrix((data, (rows, cols)), shape=(n, n))
    targets = np.asarray(target_nodes, dtype=int)
    if len(targets):
        dist = shortest_path(matrix, directed=False, unweighted=True, indices=targets)
        distances = {int(target): np.asarray(dist[i], dtype=float) for i, target in enumerate(targets)}
    else:
        distances = {}
    adjacency = [set() for _ in range(n)]
    for a, b in zip(src, dst):
        adjacency[int(a)].add(int(b))
        adjacency[int(b)].add(int(a))
    edge_text = "|".join(f"{int(a)}-{int(b)}" for a, b in zip(src, dst))
    return GraphCache(n, adjacency, matrix, distances, hashlib.sha256(edge_text.encode("utf-8")).hexdigest())


def geographic_distance_logweights(
    target_xy: np.ndarray,
    calibration_xy: np.ndarray,
    bandwidth: float = 0.25,
) -> np.ndarray:
    """Frozen ABL7 Euclidean-distance heuristic, not final M3."""
    target = np.asarray(target_xy, dtype=float).reshape(1, 2)
    calibration = np.asarray(calibration_xy, dtype=float)
    if calibration.ndim != 2 or calibration.shape[1] != 2:
        raise ValueError("calibration_xy must have shape (n,2)")
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("bandwidth must be positive and finite")
    distance_sq = np.sum((calibration - target) ** 2, axis=1)
    # Keep the calibration kernel on the same absolute scale as the target
    # pseudo-weight k(0)=1 (log-weight 0).  Calibration-only normalization
    # would change the target/calibration mass ratio and therefore the interval.
    return -distance_sq / (2.0 * float(bandwidth) ** 2)


def deterministic_graph_safe_set(cache: GraphCache, target_node: int, calibration_nodes: np.ndarray) -> np.ndarray:
    target_node = int(target_node)
    forbidden = set(cache.adjacency[target_node]) | {target_node}
    candidates = [int(v) for v in sorted(np.asarray(calibration_nodes, dtype=int)) if int(v) not in forbidden]
    selected: list[int] = []
    selected_set: set[int] = set()
    for node in candidates:
        if not (cache.adjacency[node] & selected_set):
            selected.append(node)
            selected_set.add(node)
    return np.asarray(selected, dtype=int)


def verify_independent_set(cache: GraphCache, target_node: int, selected: np.ndarray) -> bool:
    selected_set = set(map(int, selected))
    if int(target_node) in selected_set or cache.adjacency[int(target_node)] & selected_set:
        return False
    return all(not (cache.adjacency[node] & (selected_set - {node})) for node in selected_set)


def morans_i(values_by_node: pd.Series, edges: pd.DataFrame) -> float:
    """Unstandardized binary-adjacency Moran's I diagnostic."""
    if len(values_by_node) < 3:
        return float("nan")
    values = values_by_node.dropna().astype(float)
    node_set = set(map(int, values.index))
    canonical = _canonical_edge_frame(edges)
    canonical = canonical[canonical["a"].isin(node_set) & canonical["b"].isin(node_set)]
    if len(canonical) == 0:
        return float("nan")
    centered = values - float(values.mean())
    denominator = float(np.sum(centered.to_numpy() ** 2))
    if denominator <= 0:
        return float("nan")
    numerator = 0.0
    for row in canonical.itertuples(index=False):
        numerator += 2.0 * float(centered.loc[int(row.a)] * centered.loc[int(row.b)])
    weight_sum = 2.0 * len(canonical)
    return float((len(values) / weight_sum) * (numerator / denominator))


def graph_semivariogram(
    values_by_node: pd.Series,
    cache: GraphCache,
    max_distance: int = 4,
) -> pd.DataFrame:
    values = values_by_node.dropna().astype(float)
    nodes = np.asarray(sorted(map(int, values.index)), dtype=int)
    rows: list[dict[str, float | int | str]] = []
    if len(nodes) < 2:
        return pd.DataFrame(columns=["distance_bin", "pair_count", "semivariance"])
    distances = shortest_path(cache.matrix, directed=False, unweighted=True, indices=nodes)
    vals = values.loc[nodes].to_numpy(float)
    accum: dict[str, list[float]] = {str(d): [] for d in range(1, max_distance + 1)}
    accum[f">={max_distance + 1}"] = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            d = distances[i, nodes[j]]
            if not np.isfinite(d) or d < 1:
                continue
            label = str(int(d)) if d <= max_distance else f">={max_distance + 1}"
            accum[label].append(0.5 * float((vals[i] - vals[j]) ** 2))
    for label, values_list in accum.items():
        rows.append(
            {
                "distance_bin": label,
                "pair_count": len(values_list),
                "semivariance": float(np.mean(values_list)) if values_list else float("nan"),
            }
        )
    return pd.DataFrame(rows)
