from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from .io import D3M4Error


@dataclass(frozen=True)
class LocalGraphLaw:
    n_nodes: int
    rho: float
    scale: float
    transform_power: float
    block_nodes: tuple[int, ...]
    block_positions: dict[int, int]
    local_edges: tuple[tuple[int, int, float], ...]
    degrees: np.ndarray
    outside_residual: np.ndarray
    residual_law: str


def induced_block_connected(block_nodes: list[int] | tuple[int, ...], edges: pd.DataFrame) -> bool:
    nodes = set(map(int, block_nodes))
    if not nodes:
        return False
    adjacency = {n: set() for n in nodes}
    for row in edges.itertuples(index=False):
        a, b = int(row.source_node), int(row.target_node)
        if a in nodes and b in nodes:
            adjacency[a].add(b)
            adjacency[b].add(a)
    seen = set()
    stack = [next(iter(nodes))]
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        stack.extend(adjacency[n] - seen)
    return seen == nodes


def build_local_graph_law(
    case_units: pd.DataFrame,
    case_edges: pd.DataFrame,
    block_nodes: list[int] | tuple[int, ...],
    rho: float,
    residual_law: str,
    scale: float = 0.35,
) -> LocalGraphLaw:
    block = tuple(map(int, block_nodes))
    if len(block) < 2 or len(set(block)) != len(block):
        raise D3M4Error("M4_R01_INVALID_BLOCK")
    if not induced_block_connected(block, case_edges):
        raise D3M4Error("M4_R02_BLOCK_DISCONNECTED")
    units = case_units.sort_values("node_index").reset_index(drop=True)
    n_nodes = int(units["node_index"].max()) + 1
    if not np.array_equal(units["node_index"].to_numpy(int), np.arange(n_nodes)):
        raise D3M4Error("Node indices are not contiguous")
    degree = units["graph_degree_true"].to_numpy(float)
    outside = units["shared_spatial_residual"].to_numpy(float)
    if np.any(degree < 0) or not np.isfinite(outside).all():
        raise D3M4Error("Invalid graph degree or residual truth")
    bset = set(block)
    local: list[tuple[int, int, float]] = []
    for row in case_edges.itertuples(index=False):
        a, b = int(row.source_node), int(row.target_node)
        if a == b:
            continue
        if a in bset or b in bset:
            if degree[a] <= 0 or degree[b] <= 0:
                raise D3M4Error("Positive edge touches zero-degree node")
            sij = 1.0 / math.sqrt(float(degree[a] * degree[b]))
            local.append((a, b, sij))
    if residual_law == "iid_continuous":
        power = 1.0
        if abs(float(rho)) > 1e-14:
            raise D3M4Error("IID residual law has nonzero spatial rho")
    elif residual_law == "gaussian_gmrf":
        power = 1.0
    elif residual_law == "transformed_gmrf_power_1_5":
        power = 1.5
    else:
        raise D3M4Error(f"Unsupported oracle residual law: {residual_law}")
    return LocalGraphLaw(
        n_nodes=n_nodes,
        rho=float(rho),
        scale=float(scale),
        transform_power=float(power),
        block_nodes=block,
        block_positions={n: i for i, n in enumerate(block)},
        local_edges=tuple(local),
        degrees=degree,
        outside_residual=outside,
        residual_law=str(residual_law),
    )


def inverse_power_transform(e: np.ndarray, power: float) -> np.ndarray:
    e = np.asarray(e, dtype=float)
    if power <= 0:
        raise D3M4Error("Residual transform power must be positive")
    if power == 1.0:
        return e.copy()
    return np.sign(e) * np.abs(e) ** (1.0 / power)


def _value_at(node: int, block_values: np.ndarray, law: LocalGraphLaw, transformed_outside: np.ndarray) -> float:
    pos = law.block_positions.get(int(node))
    if pos is not None:
        return float(block_values[pos])
    return float(transformed_outside[int(node)])


def local_log_factor(block_residual: np.ndarray, law: LocalGraphLaw) -> tuple[float, bool]:
    e_b = np.asarray(block_residual, dtype=float)
    if e_b.shape != (len(law.block_nodes),) or not np.isfinite(e_b).all():
        raise D3M4Error("Invalid M3/M4 block residual vector")
    z_b = inverse_power_transform(e_b, law.transform_power)
    z_outside = inverse_power_transform(law.outside_residual, law.transform_power)
    inv_var = 1.0 / (law.scale * law.scale)
    logv = -0.5 * inv_var * float(np.dot(z_b, z_b))
    if law.rho != 0.0:
        for a, b, sij in law.local_edges:
            za = _value_at(a, z_b, law, z_outside)
            zb = _value_at(b, z_b, law, z_outside)
            logv += law.rho * sij * za * zb * inv_var
    return float(logv), False


def full_log_factor(block_residual: np.ndarray, law: LocalGraphLaw, all_edges: pd.DataFrame) -> tuple[float, bool]:
    e = law.outside_residual.copy()
    for pos, node in enumerate(law.block_nodes):
        e[node] = float(block_residual[pos])
    z = inverse_power_transform(e, law.transform_power)
    inv_var = 1.0 / (law.scale * law.scale)
    logv = -0.5 * inv_var * float(np.dot(z, z))
    if law.rho != 0.0:
        for row in all_edges.itertuples(index=False):
            a, b = int(row.source_node), int(row.target_node)
            if law.degrees[a] <= 0 or law.degrees[b] <= 0:
                continue
            sij = 1.0 / math.sqrt(float(law.degrees[a] * law.degrees[b]))
            logv += law.rho * sij * float(z[a] * z[b]) * inv_var
    return float(logv), False
