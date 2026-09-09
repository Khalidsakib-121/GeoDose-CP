from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_VERSION = "1.3.1"
EXPECTED_BLOCKS = 27042
EXPECTED_EDGES = 102893
EXPECTED_FOLDS = 5
EXPECTED_BLOCK_YEARS = 81126
EXPECTED_MINES = 5
EXPECTED_YEARS = [2023, 2024, 2025]
EXPECTED_ALL_YEAR_ELIGIBLE = 23610
EXPECTED_STAGE2B_ARCHIVE_SHA256 = "329a21f3b94d88f8f0b0b3327dbe250eb1df0911f73a6309c6c80c8417bd73a8"
ROOT_SEED = 20260805
PRIMARY_DOSE_GRID = np.array([0.00, 0.10, 0.25, 0.50, 0.75, 0.90, 1.00], dtype=float)
PRIMARY_BANDWIDTH = 0.10


class Stage3AError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Stage3AError(message)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalized_zip_map(zf: zipfile.ZipFile) -> dict[str, str]:
    return {name.replace("\\", "/"): name for name in zf.namelist()}


def read_stage2b_csv(zf: zipfile.ZipFile, names: dict[str, str], logical_name: str) -> pd.DataFrame:
    raw = zf.read(names[logical_name])
    if logical_name.endswith(".gz"):
        return pd.read_csv(io.BytesIO(raw), compression="gzip")
    return pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")


def read_stage2b_json(zf: zipfile.ZipFile, names: dict[str, str], logical_name: str) -> dict[str, Any]:
    return json.loads(zf.read(names[logical_name]).decode("utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    mapped = series.astype(str).str.strip().str.lower().map({"true": True, "false": False})
    require(mapped.notna().all(), f"Could not parse boolean field {series.name}")
    return mapped.astype(bool)


def softmax3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    m = np.maximum(np.maximum(a, b), c)
    ea, eb, ec = np.exp(a - m), np.exp(b - m), np.exp(c - m)
    den = ea + eb + ec
    return np.column_stack([ea / den, eb / den, ec / den])


def expit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def beta_log_pdf(x: np.ndarray, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)
    lgamma = np.vectorize(math.lgamma)
    return (
        (alpha - 1.0) * np.log(x)
        + (beta - 1.0) * np.log1p(-x)
        - (lgamma(alpha) + lgamma(beta) - lgamma(alpha + beta))
    )


def build_queen_graph(n_rows: int, n_cols: int) -> tuple[pd.DataFrame, np.ndarray]:
    edges: list[tuple[int, int]] = []
    W = np.zeros((n_rows * n_cols, n_rows * n_cols), dtype=float)
    directions = [(0, 1), (1, -1), (1, 0), (1, 1)]

    def idx(r: int, c: int) -> int:
        return r * n_cols + c

    for r in range(n_rows):
        for c in range(n_cols):
            i = idx(r, c)
            for dr, dc in directions:
                rr, cc = r + dr, c + dc
                if 0 <= rr < n_rows and 0 <= cc < n_cols:
                    j = idx(rr, cc)
                    edges.append((i, j))
                    W[i, j] = 1.0
                    W[j, i] = 1.0
    return pd.DataFrame(edges, columns=["source_node", "target_node"]), W


def gmrf_field(
    W: np.ndarray,
    rho: float,
    rng: np.random.Generator,
    scale: float = 1.0,
) -> tuple[np.ndarray, float]:
    """Draw exactly from N(0, Q^{-1}) for Q=(I-rho*S)/scale^2.

    The realization is not centered or standardized after drawing. Any
    realization-dependent normalization would destroy the stated Gaussian
    Markov law and introduce global dependence.
    """
    require(scale > 0, "GMRF scale must be positive")
    degree = W.sum(axis=1)
    inv_sqrt = np.zeros_like(degree)
    inv_sqrt[degree > 0] = 1.0 / np.sqrt(degree[degree > 0])
    S = inv_sqrt[:, None] * W * inv_sqrt[None, :]
    Q = (np.eye(W.shape[0]) - rho * S) / (scale * scale)
    eig_min = float(np.linalg.eigvalsh(Q)[0])
    require(eig_min > 1e-8, f"GMRF precision not positive definite: min eigenvalue {eig_min}")
    L = np.linalg.cholesky(Q)
    field = np.linalg.solve(L.T, rng.standard_normal(W.shape[0]))
    return field, eig_min


def beta_density_integral(alpha: float, beta: float, nodes: int = 512) -> float:
    """Numerically integrate a Beta density on (0,1) by Gauss-Legendre quadrature."""
    x_raw, w_raw = np.polynomial.legendre.leggauss(nodes)
    x = 0.5 * (x_raw + 1.0)
    w = 0.5 * w_raw
    density = np.exp(beta_log_pdf(x, np.full_like(x, alpha), np.full_like(x, beta)))
    return float(np.sum(w * density))


def mixed_treatment_normalization_error(preview: pd.DataFrame, representative_units: int = 25) -> float:
    indices = np.linspace(0, len(preview) - 1, representative_units, dtype=int)
    errors: list[float] = []
    for idx in indices:
        row = preview.iloc[idx]
        integral = beta_density_integral(float(row.beta_alpha), float(row.beta_beta))
        mass = float(row.pi_atom_0 + row.pi_atom_1 + row.pi_interior * integral)
        errors.append(abs(mass - 1.0))
    return float(max(errors))


def localized_kernel_moments(center: float, bandwidth: float, nodes: int = 512) -> dict[str, float]:
    require(0.0 < center < 1.0, "Interior localized target center must lie in (0,1)")
    require(bandwidth > 0.0, "Localized intervention bandwidth must be positive")
    x_raw, w_raw = np.polynomial.legendre.leggauss(nodes)
    a = 0.5 * (x_raw + 1.0)
    w = 0.5 * w_raw
    unnormalized = np.exp(-0.5 * ((a - center) / bandwidth) ** 2)
    normalizer = float(np.sum(w * unnormalized))
    q_weights = w * unnormalized / normalizer
    return {
        "E_A": float(np.sum(q_weights * a)),
        "E_A2": float(np.sum(q_weights * a * a)),
        "E_sin_2piA": float(np.sum(q_weights * np.sin(2.0 * np.pi * a))),
        "normalizer": normalizer,
    }


def generate_preview(
    seed: int = ROOT_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    n_rows = n_cols = 25
    edges, W = build_queen_graph(n_rows, n_cols)
    n = n_rows * n_cols
    rows = np.repeat(np.arange(n_rows), n_cols)
    cols = np.tile(np.arange(n_cols), n_rows)
    xcoord = (cols - (n_cols - 1) / 2) / ((n_cols - 1) / 2)
    ycoord = (rows - (n_rows - 1) / 2) / ((n_rows - 1) / 2)

    # Proper Gaussian Markov draws: no sample centering or sample-SD normalization.
    x1, eig1 = gmrf_field(W, 0.45, rng, scale=0.80)
    x2, eig2 = gmrf_field(W, 0.25, rng, scale=0.80)
    x3 = rng.normal(size=n)
    x4 = rng.uniform(-1, 1, size=n)

    logits0 = -1.55 + 0.55 * x1 - 0.35 * x3 + 0.15 * ycoord
    logits1 = -1.75 - 0.45 * x1 + 0.55 * x2 + 0.25 * x4
    logitsI = np.zeros(n)
    probs = softmax3(logits0, logits1, logitsI)
    u = rng.random(n)
    cat = np.where(u < probs[:, 0], 0, np.where(u < probs[:, 0] + probs[:, 1], 1, 2))

    interior_location = expit(-0.05 + 0.85 * x1 - 0.55 * x2 + 0.35 * x3 + 0.25 * xcoord)
    kappa = 8.0
    # Lower bound 1.0 keeps the density finite and numerically auditable near endpoints.
    aa = np.clip(interior_location * kappa, 1.0, None)
    bb = np.clip((1.0 - interior_location) * kappa, 1.0, None)
    beta_mean = aa / (aa + bb)

    treatment = np.empty(n, float)
    treatment[cat == 0] = 0.0
    treatment[cat == 1] = 1.0
    interior = cat == 2
    treatment[interior] = rng.beta(aa[interior], bb[interior])

    g = np.empty(n, float)
    g[cat == 0] = probs[cat == 0, 0]
    g[cat == 1] = probs[cat == 1, 1]
    g[interior] = probs[interior, 2] * np.exp(
        beta_log_pdf(treatment[interior], aa[interior], bb[interior])
    )

    residual, eigR = gmrf_field(W, 0.60, rng, scale=0.35)
    baseline = 0.25 + 0.65 * x1 - 0.40 * x2 + 0.25 * (x3**2 - 1) + 0.20 * np.sin(np.pi * xcoord)

    def mean_at(a: np.ndarray) -> np.ndarray:
        return baseline + 0.55 * a - 0.35 * a * a + 0.25 * np.sin(2 * np.pi * a) + 0.20 * a * x1

    y_mean = mean_at(treatment)
    y_true = y_mean + residual
    ue_proxy = np.clip(12 + 4 * np.abs(x2) + 3 * (x4 > 0) + rng.normal(0, 1.5, n), 0, 40)

    preview = pd.DataFrame(
        {
            "unit_id": [f"SIM_R{r:02d}_C{c:02d}" for r, c in zip(rows, cols)],
            "grid_row": rows,
            "grid_col": cols,
            "x_coord": xcoord,
            "y_coord": ycoord,
            "X_spatial_1": x1,
            "X_spatial_2": x2,
            "X_nonlinear_1": x3,
            "X_nonlinear_2": x4,
            "treatment_category": np.where(cat == 0, "atom_0", np.where(cat == 1, "atom_1", "interior")),
            "A": treatment,
            "pi_atom_0": probs[:, 0],
            "pi_atom_1": probs[:, 1],
            "pi_interior": probs[:, 2],
            "beta_alpha": aa,
            "beta_beta": bb,
            "beta_mean": beta_mean,
            "oracle_g_mixed_at_A": g,
            "conditional_mean_at_A": y_mean,
            "shared_spatial_residual": residual,
            "Y_true_at_A": y_true,
            "Y_observed_at_A": y_true,
            "ue_proxy": ue_proxy,
            "graph_degree": W.sum(axis=1).astype(int),
        }
    )

    hard_truth_parts: list[pd.DataFrame] = []
    localized_truth_parts: list[pd.DataFrame] = []
    for a0 in PRIMARY_DOSE_GRID:
        a_vec = np.full(n, a0)
        ma = mean_at(a_vec)
        hard_truth_parts.append(
            pd.DataFrame(
                {
                    "unit_id": preview["unit_id"],
                    "dose": a0,
                    "conditional_mean": ma,
                    "Y_true": ma + residual,
                    "truth_role": "hard_dose_secondary_truth",
                }
            )
        )

        if a0 in (0.0, 1.0):
            localized_mean = ma
            intervention_type = "audited_endpoint_atom"
            bandwidth = np.nan
            e_a = a0
        else:
            moments = localized_kernel_moments(float(a0), PRIMARY_BANDWIDTH)
            localized_mean = (
                baseline
                + 0.55 * moments["E_A"]
                - 0.35 * moments["E_A2"]
                + 0.25 * moments["E_sin_2piA"]
                + 0.20 * moments["E_A"] * x1
            )
            intervention_type = "truncated_gaussian_kernel_on_open_unit_interval"
            bandwidth = PRIMARY_BANDWIDTH
            e_a = moments["E_A"]
        localized_truth_parts.append(
            pd.DataFrame(
                {
                    "unit_id": preview["unit_id"],
                    "target_dose": a0,
                    "bandwidth": bandwidth,
                    "intervention_type": intervention_type,
                    "expected_intervention_dose": e_a,
                    "localized_conditional_mean": localized_mean,
                    "localized_Y_true_mean": localized_mean + residual,
                    "target_object": "random_localized_stochastic_potential_outcome",
                    "artifact_scope": "analytic_conditional_mean_functional_only",
                    "coverage_truth_status": "realized_target_draws_deferred_to_stage3b",
                    "truth_role": "primary_localized_stochastic_target_mean_functional",
                }
            )
        )

    hard_truth = pd.concat(hard_truth_parts, ignore_index=True)
    localized_truth = pd.concat(localized_truth_parts, ignore_index=True)
    normalization_error = mixed_treatment_normalization_error(preview)

    diagnostics = {
        "seed": seed,
        "grid_rows": n_rows,
        "grid_cols": n_cols,
        "unit_count": n,
        "queen_edge_count": int(len(edges)),
        "atom_0_count": int((cat == 0).sum()),
        "atom_1_count": int((cat == 1).sum()),
        "interior_count": int(interior.sum()),
        "precision_min_eigenvalues": {"X1": eig1, "X2": eig2, "residual": eigR},
        "gmrf_draw_law": "exact N(0,Q^-1) draw from Q=(I-rho*S)/scale^2",
        "realization_dependent_centering_or_scaling": False,
        "treatment_min": float(treatment.min()),
        "treatment_max": float(treatment.max()),
        "oracle_density_min": float(g.min()),
        "oracle_density_max": float(g.max()),
        "mixed_treatment_normalization_max_abs_error": normalization_error,
        "dose_grid": PRIMARY_DOSE_GRID.tolist(),
        "primary_estimand": "localized stochastic potential outcome",
        "primary_target_object": "random Y_i(A_star)",
        "localized_truth_artifact_scope": "analytic conditional mean functional only",
        "realized_target_draws_required_before_coverage": True,
        "primary_interior_kernel": "truncated Gaussian kernel on (0,1)",
        "primary_bandwidth": PRIMARY_BANDWIDTH,
        "hard_dose_truth_role": "secondary generator diagnostic and sensitivity truth",
        "dense_inverse_formed": False,
        "preview_note": (
            "Dense Cholesky is used only for the 625-unit validation preview; no dense precision inverse is formed. "
            "The preview residual is not standardized after drawing, preserving the stated Gaussian Markov law."
        ),
    }
    return preview, hard_truth, localized_truth, edges, diagnostics
