from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import eigsh

from .io import require


def build_adjacency(n_nodes: int, edges: pd.DataFrame) -> sparse.csr_matrix:
    source = edges["source_node"].to_numpy(dtype=int)
    target = edges["target_node"].to_numpy(dtype=int)
    require(np.all(source >= 0) and np.all(target >= 0), "Negative graph endpoint")
    require(np.all(source < n_nodes) and np.all(target < n_nodes), "Graph endpoint outside node range")
    require(not np.any(source == target), "Self-loop in graph")
    rows = np.concatenate([source, target])
    cols = np.concatenate([target, source])
    data = np.ones(len(rows), dtype=float)
    W = sparse.coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()
    W.sum_duplicates()
    W.data[:] = 1.0
    require((W != W.T).nnz == 0, "Graph adjacency is not symmetric")
    return W


def normalized_precision(n_nodes: int, edges: pd.DataFrame, rho: float, scale: float) -> sparse.csr_matrix:
    require(abs(rho) < 0.999, "rho outside proper registered range")
    require(scale > 0.0, "Residual scale is nonpositive")
    W = build_adjacency(n_nodes, edges)
    degree = np.asarray(W.sum(axis=1)).reshape(-1)
    inv_sqrt = np.zeros_like(degree)
    positive = degree > 0.0
    inv_sqrt[positive] = 1.0 / np.sqrt(degree[positive])
    D_inv = sparse.diags(inv_sqrt, format="csr")
    S = D_inv @ W @ D_inv
    Q = (sparse.eye(n_nodes, format="csr") - float(rho) * S) / (float(scale) ** 2)
    Q = 0.5 * (Q + Q.T)
    return Q.tocsr()


def minimum_precision_eigenvalue(Q: sparse.csr_matrix) -> float:
    n = Q.shape[0]
    if n <= 3:
        return float(np.min(np.linalg.eigvalsh(Q.toarray())))
    v0 = np.linspace(1.0, 2.0, n, dtype=float)
    v0 /= np.linalg.norm(v0)
    value = eigsh(Q, k=1, which="SA", return_eigenvectors=False, tol=1e-12, v0=v0)[0]
    return float(value)


def block_is_connected(block_nodes: np.ndarray, edges: pd.DataFrame) -> bool:
    block = set(int(x) for x in np.asarray(block_nodes).tolist())
    if not block:
        return False
    adjacency = {node: set() for node in block}
    for row in edges.itertuples(index=False):
        source, target = int(row.source_node), int(row.target_node)
        if source in block and target in block:
            adjacency[source].add(target)
            adjacency[target].add(source)
    start = next(iter(block))
    seen = {start}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for neighbor in adjacency[node]:
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    return seen == block


def precision_blocks(Q: sparse.csr_matrix, block_nodes: np.ndarray, boundary_nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    block = np.asarray(block_nodes, dtype=int)
    boundary = np.asarray(boundary_nodes, dtype=int)
    Q_BB = Q[block][:, block].toarray()
    Q_BD = Q[block][:, boundary].toarray() if len(boundary) else np.zeros((len(block), 0), dtype=float)
    require(np.allclose(Q_BB, Q_BB.T, atol=1e-12, rtol=0.0), "Q_BB not symmetric")
    return Q_BB, Q_BD


def gaussian_local_log_factor(residual_B: np.ndarray, Q_BB: np.ndarray, Q_BD: np.ndarray, residual_D: np.ndarray) -> np.ndarray:
    residual_B = np.asarray(residual_B, dtype=float)
    one_dim = residual_B.ndim == 1
    if one_dim:
        residual_B = residual_B[None, :]
    first = -0.5 * np.einsum("pi,ij,pj->p", residual_B, Q_BB, residual_B)
    if Q_BD.shape[1] > 0:
        second = -((residual_B @ Q_BD) @ np.asarray(residual_D, dtype=float))
    else:
        second = np.zeros(len(residual_B), dtype=float)
    result = first + second
    return result[0] if one_dim else result


def gaussian_full_log_factor(residual: np.ndarray, Q: sparse.csr_matrix) -> float:
    residual = np.asarray(residual, dtype=float)
    return float(-0.5 * residual @ (Q @ residual))
