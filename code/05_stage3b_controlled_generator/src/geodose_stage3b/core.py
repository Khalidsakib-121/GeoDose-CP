from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import platform
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

SCRIPT_VERSION = "1.2.0-stage3b-freeze"
EXPECTED_STAGE3A_SHA256 = "9a46e21f4f488447f0829f079c8f19a345e6ac1748bc6907e20917bbb096d406"
EXPECTED_STAGE3A_VERSION = "1.3.1"
PRIMARY_DOSES = (0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0)
PRIMARY_BANDWIDTH = 0.10
VALIDATION_REPLICATION = 1


class Stage3BError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Stage3BError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_int(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "little")


def derived_seed(base_seed: int, label: str) -> int:
    ss = np.random.SeedSequence([int(base_seed), stable_int(label)])
    return int(ss.generate_state(1, dtype=np.uint32)[0])


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n", float_format="%.17g")


def write_csv_gz(path: Path, frame: pd.DataFrame) -> None:
    text = frame.to_csv(index=False, lineterminator="\n", float_format="%.17g")
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            gz.write(text.encode("utf-8"))


def read_csv_gz_bytes(raw: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(raw), compression="gzip")


def read_stage3a_archive(path: Path) -> Dict[str, Any]:
    require(path.exists(), f"Missing frozen Stage 3A archive: {path}")
    actual = sha256_file(path)
    require(actual == EXPECTED_STAGE3A_SHA256, f"Stage 3A archive SHA-256 mismatch: {actual}")
    with zipfile.ZipFile(path) as zf:
        require(zf.testzip() is None, "Stage 3A archive CRC failure")
        names = {n.replace("\\", "/"): n for n in zf.namelist()}
        prefix = "outputs_stage3a/"
        required = [
            "STAGE3A_VERIFICATION.json",
            "stage3a_manifest.json",
            "stage3_registry.yaml",
            "scenario_registry.csv",
            "seed_registry.csv",
            "target_registry.csv",
            "residual_law_registry.csv",
            "target_draw_contract.json",
            "target_population_contract.json",
            "method_registry.csv",
            "mine_role_registry.csv",
            "additional_stress_test_registry.csv",
            "feature_registry.csv",
            "factor_registry.csv",
            "ablation_registry.csv",
        ]
        for logical in required:
            require(prefix + logical in names, f"Missing Stage 3A artifact: {logical}")
        verification = json.loads(zf.read(names[prefix + "STAGE3A_VERIFICATION.json"]).decode("utf-8"))
        manifest = json.loads(zf.read(names[prefix + "stage3a_manifest.json"]).decode("utf-8"))
        require(verification.get("status") == "verified_complete", "Stage 3A is not verified_complete")
        require(str(verification.get("script_version")) == EXPECTED_STAGE3A_VERSION, "Unexpected Stage 3A version")
        require(verification.get("production_experiments_run") is False, "Stage 3A unexpectedly ran production experiments")
        frames: Dict[str, pd.DataFrame] = {}
        for logical in [
            "scenario_registry.csv",
            "seed_registry.csv",
            "target_registry.csv",
            "residual_law_registry.csv",
            "method_registry.csv",
            "mine_role_registry.csv",
            "additional_stress_test_registry.csv",
            "feature_registry.csv",
            "factor_registry.csv",
            "ablation_registry.csv",
        ]:
            frames[logical] = pd.read_csv(io.BytesIO(zf.read(names[prefix + logical])), encoding="utf-8-sig")
        return {
            "sha256": actual,
            "verification": verification,
            "manifest": manifest,
            "frames": frames,
            "target_draw_contract": json.loads(zf.read(names[prefix + "target_draw_contract.json"]).decode("utf-8")),
            "target_population_contract": json.loads(zf.read(names[prefix + "target_population_contract.json"]).decode("utf-8")),
            "stage3_registry_yaml": zf.read(names[prefix + "stage3_registry.yaml"]).decode("utf-8"),
        }



def validate_stage3a_contract(stage3a: Dict[str, Any], cases_frame: pd.DataFrame) -> Dict[str, Any]:
    scenarios = stage3a["frames"]["scenario_registry.csv"].copy()
    targets = stage3a["frames"]["target_registry.csv"].copy()
    residuals = stage3a["frames"]["residual_law_registry.csv"].copy()
    methods = stage3a["frames"]["method_registry.csv"].copy()
    stresses = stage3a["frames"]["additional_stress_test_registry.csv"].copy()
    features = stage3a["frames"]["feature_registry.csv"].copy()
    factors = stage3a["frames"]["factor_registry.csv"].copy()
    ablations = stage3a["frames"]["ablation_registry.csv"].copy()

    require(set(scenarios["scenario_id"]) == {f"S{i}" for i in range(1, 11)}, "Stage 3A scenario registry mismatch")
    require(set(cases_frame["scenario_id"]).issuperset(set(scenarios["scenario_id"])), "Stage 3B case registry omits a frozen scenario")
    require(set(stresses["stress_id"]) == {"ST1", "ST2", "ST3"}, "Stage 3A stress registry mismatch")
    require({"ST1", "ST2", "ST3"}.issubset(set(cases_frame["scenario_id"])), "Stage 3B omits a frozen stress test")

    registered_doses = tuple(float(x) for x in sorted(targets["target_dose"].unique()))
    require(registered_doses == PRIMARY_DOSES, "Stage 3B target grid differs from Stage 3A")
    interior_bw = targets.loc[~targets["endpoint_atom"].astype(bool), "bandwidth_primary"].astype(float)
    require(np.allclose(interior_bw, PRIMARY_BANDWIDTH, atol=0, rtol=0), "Stage 3B primary bandwidth differs from Stage 3A")
    require((targets["estimand_type"] == "localized_stochastic_potential_outcome").all(), "Unexpected Stage 3A estimand")
    require(stage3a["target_draw_contract"]["primary_target_object"] == "random localized stochastic potential outcome Y_i(A_star)", "Stage 3A target-draw contract mismatch")
    require("uniform over theorem-eligible final-test slots" in stage3a["target_population_contract"]["controlled_target_population"], "Stage 3A controlled target population mismatch")

    require(set(methods["method_id"]) == {f"M{i}" for i in range(1, 7)}, "Stage 3A method registry mismatch")
    require((methods["implementation_status"] == "registered_not_implemented").all(), "Stage 3B input unexpectedly marks a method implemented")
    require({"iid_continuous", "proper_gaussian_gmrf", "monotone_transformed_gmrf", "misspecified_spatial_reference"}.issubset(set(residuals["residual_law_name"])), "Stage 3A residual-law registry mismatch")
    require({"iid_continuous", "gaussian_gmrf", "transformed_gmrf_power_1_5"}.issubset(set(cases_frame["residual_law"])), "Stage 3B residual-law cases incomplete")

    expected_rho = {
        "S1": {0.0}, "S2": {0.6}, "S3": {0.0}, "S4": {0.0, 0.2, 0.4, 0.6, 0.8},
        "S5": {0.4}, "S6": {0.6}, "S7": {0.0}, "S8": {0.6}, "S9": {0.4}, "S10": {0.4},
    }
    for scenario_id, levels in expected_rho.items():
        actual = np.sort(cases_frame.loc[cases_frame.scenario_id == scenario_id, "spatial_rho"].astype(float).unique())
        expected = np.sort(np.asarray(list(levels), dtype=float))
        require(actual.shape == expected.shape and np.allclose(actual, expected, atol=1e-12, rtol=0), f"Stage 3B rho mapping mismatch for {scenario_id}: {actual.tolist()}")
    s9_lambda = np.sort(cases_frame.loc[cases_frame.scenario_id == "S9", "measurement_lambda"].astype(float).unique())
    require(np.allclose(s9_lambda, np.asarray([0.0, 0.5, 1.0]), atol=1e-12, rtol=0), "S9 treatment-correlation levels mismatch")
    required_measurement_modes = {"random", "treatment_correlated", "substrate_bias", "combined"}
    require(required_measurement_modes.issubset(set(cases_frame.loc[cases_frame.scenario_id == "S9", "measurement_error"])), "S9 measurement mechanisms incomplete")
    f09 = factors.loc[factors.factor_id == "F09", "registered_levels"]
    require(len(f09) == 1 and all(level in str(f09.iloc[0]) for level in ["none", "random", "treatment_correlated", "substrate_bias"]), "Stage 3A F09 factor contract mismatch")
    require("ABL2" in set(ablations.ablation_id), "Stage 3A target-design-transport ablation missing")
    design_shift_cases = cases_frame.loc[cases_frame.target_design_mode != "identity"]
    require(len(design_shift_cases) >= 1 and (design_shift_cases.target_design_delta.astype(float) > 0).all(), "No non-identity target-design transport validation case")
    require(set(cases_frame.loc[cases_frame.scenario_id == "S6", "fitted_graph"]) == {"rook", "queen_omit50"}, "S6 graph variants mismatch")
    require(np.allclose(np.sort(cases_frame.loc[cases_frame.scenario_id == "S4", "spatial_rho"].astype(float).unique()), np.asarray([0.0, 0.2, 0.4, 0.6, 0.8]), atol=1e-12, rtol=0), "S4 hero rho grid mismatch")
    require(set(cases_frame.loc[cases_frame.scenario_id.isin(["S1", "S2", "S3", "S7"]), "covariate_law"]) == {"iid"}, "Reduction cases must use IID covariates")
    require(set(cases_frame.loc[cases_frame.scenario_id.isin(["S1", "S3", "S7"]), "residual_law"]) == {"iid_continuous"}, "Reduction cases must use IID residuals")

    feature_rows = features.set_index("feature_name")
    require(bool(feature_rows.loc["ue_proxy", "allowed_in_measurement_error_mechanism"]), "UE proxy measurement role mismatch")
    require(not bool(feature_rows.loc["ue_proxy", "allowed_in_fitted_treatment_model"]), "UE proxy treatment leakage guard mismatch")

    return {
        "status": "aligned",
        "stage3a_archive_sha256": stage3a["sha256"],
        "scenario_ids": sorted(scenarios["scenario_id"].tolist()),
        "stress_ids": sorted(stresses["stress_id"].tolist()),
        "method_ids": sorted(methods["method_id"].tolist()),
        "target_doses": list(registered_doses),
        "primary_bandwidth": PRIMARY_BANDWIDTH,
        "reduction_case_covariate_law": "iid",
        "reduction_case_residual_law": "iid_continuous",
        "target_design_ratio_feature": "X_nonlinear_1",
        "target_design_shift_case_ids": sorted(cases_frame.loc[cases_frame.target_design_mode != "identity", "case_id"].tolist()),
        "measurement_error_modes": sorted(cases_frame.loc[cases_frame.scenario_id == "S9", "measurement_error"].unique().tolist()),
        "methods_implemented": False,
    }

def expit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    e = np.exp(x[~positive])
    out[~positive] = e / (1.0 + e)
    return out


def softmax3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    m = np.maximum(np.maximum(a, b), c)
    ea, eb, ec = np.exp(a - m), np.exp(b - m), np.exp(c - m)
    den = ea + eb + ec
    return np.column_stack([ea / den, eb / den, ec / den])


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


def normal_cdf(x: np.ndarray | float) -> np.ndarray | float:
    arr = np.asarray(x, dtype=float)
    vals = 0.5 * (1.0 + np.vectorize(math.erf)(arr / math.sqrt(2.0)))
    if np.ndim(x) == 0:
        return float(vals)
    return vals


def truncated_normal_density(a: np.ndarray, center: float, bandwidth: float) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    z = normal_cdf((1.0 - center) / bandwidth) - normal_cdf((0.0 - center) / bandwidth)
    require(float(z) > 0.0, "Invalid truncated-normal normalizer")
    density = np.exp(-0.5 * ((a - center) / bandwidth) ** 2) / (bandwidth * math.sqrt(2.0 * math.pi) * float(z))
    density[(a <= 0.0) | (a >= 1.0)] = 0.0
    return density


def truncated_normal_integral(center: float, bandwidth: float, nodes: int = 512) -> float:
    x_raw, w_raw = np.polynomial.legendre.leggauss(nodes)
    a = 0.5 * (x_raw + 1.0)
    w = 0.5 * w_raw
    return float(np.sum(w * truncated_normal_density(a, center, bandwidth)))


def draw_truncated_normal(
    rng: np.random.Generator,
    center: float,
    bandwidth: float,
    size: int,
) -> np.ndarray:
    require(size >= 0, "Size must be nonnegative")
    result = np.empty(size, dtype=float)
    filled = 0
    while filled < size:
        batch = max(64, (size - filled) * 2)
        proposal = rng.normal(center, bandwidth, batch)
        accepted = proposal[(proposal > 0.0) & (proposal < 1.0)]
        take = min(len(accepted), size - filled)
        if take:
            result[filled : filled + take] = accepted[:take]
            filled += take
    return result


def role_independence_cuts(n_cols: int, sample_design: str) -> Tuple[int, ...]:
    """Return column boundaries removed from the DGP graph.

    Nuisance training and support audit are independent external components.
    Calibration and test remain in one inference graph so spatial dependence is
    still present across the conformal calibration/target boundary.
    """
    if n_cols == 25 and sample_design == "small_calibration":
        return (13, 18)
    if n_cols == 25 and sample_design == "maup_aligned":
        return (9, 13)
    if n_cols == 25:
        return (9, 14)
    if n_cols == 13:
        return (5, 7)
    raise Stage3BError(f"Unsupported grid width/design: {n_cols}/{sample_design}")


def build_grid_graph(
    n_rows: int,
    n_cols: int,
    queen: bool = True,
    cut_after_cols: Sequence[int] = (),
) -> Tuple[pd.DataFrame, np.ndarray]:
    directions = [(0, 1), (1, 0)]
    if queen:
        directions.extend([(1, -1), (1, 1)])
    cuts = set(int(x) for x in cut_after_cols)
    edges: List[Tuple[int, int]] = []
    W = np.zeros((n_rows * n_cols, n_rows * n_cols), dtype=float)
    for r in range(n_rows):
        for c in range(n_cols):
            i = r * n_cols + c
            for dr, dc in directions:
                rr, cc = r + dr, c + dc
                if not (0 <= rr < n_rows and 0 <= cc < n_cols):
                    continue
                lo_c, hi_c = sorted((c, cc))
                if hi_c == lo_c + 1 and lo_c in cuts:
                    continue
                j = rr * n_cols + cc
                lo, hi = (i, j) if i < j else (j, i)
                edges.append((lo, hi))
                W[lo, hi] = 1.0
                W[hi, lo] = 1.0
    frame = pd.DataFrame(sorted(set(edges)), columns=["source_node", "target_node"])
    return frame, W


@dataclass
class GraphBasis:
    n_rows: int
    n_cols: int
    sample_design: str
    cut_after_cols: Tuple[int, ...]
    edges_queen: pd.DataFrame
    W_queen: np.ndarray
    edges_rook: pd.DataFrame
    W_rook: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray

    @classmethod
    def create(cls, n_rows: int, n_cols: int, sample_design: str) -> "GraphBasis":
        cuts = role_independence_cuts(n_cols, sample_design)
        q_edges, Wq = build_grid_graph(n_rows, n_cols, queen=True, cut_after_cols=cuts)
        r_edges, Wr = build_grid_graph(n_rows, n_cols, queen=False, cut_after_cols=cuts)
        degree = Wq.sum(axis=1)
        inv_sqrt = np.zeros_like(degree)
        inv_sqrt[degree > 0] = 1.0 / np.sqrt(degree[degree > 0])
        S = inv_sqrt[:, None] * Wq * inv_sqrt[None, :]
        vals, vecs = np.linalg.eigh(S)
        return cls(n_rows, n_cols, sample_design, cuts, q_edges, Wq, r_edges, Wr, vals, vecs)

    @property
    def n(self) -> int:
        return self.n_rows * self.n_cols

    def gmrf_draw_from_innovation(self, rho: float, scale: float, innovation: np.ndarray) -> Tuple[np.ndarray, float]:
        require(abs(rho) < 0.999, "rho must remain within the proper spectral range")
        innovation = np.asarray(innovation, dtype=float)
        require(innovation.shape == (self.n,), "GMRF innovation has the wrong shape")
        denom = 1.0 - rho * self.eigenvalues
        require(np.min(denom) > 1e-10, "GMRF precision is not positive definite")
        draw = scale * (self.eigenvectors @ (innovation / np.sqrt(denom)))
        min_precision_eigenvalue = float(np.min(denom) / (scale * scale))
        return draw, min_precision_eigenvalue

    def gmrf_draw(self, rho: float, scale: float, rng: np.random.Generator) -> Tuple[np.ndarray, float]:
        return self.gmrf_draw_from_innovation(rho, scale, rng.standard_normal(self.n))

@dataclass(frozen=True)
class CaseConfig:
    case_id: str
    scenario_id: str
    case_name: str
    case_type: str
    common_random_group: str
    n_rows: int
    n_cols: int
    spatial_rho: float
    target_shift: str
    overlap: str
    fitted_graph: str
    measurement_error: str
    measurement_lambda: float
    confounding: str
    outcome_form: str
    endpoint_structure: str
    residual_law: str
    covariate_law: str
    sample_design: str
    primary_target_mode: str
    primary_target_dose: Optional[float]
    primary_bandwidth: Optional[float]
    temporal_dependence: bool = False
    support_scale: str = "90m_primary"
    measurement_substrate_lambda: float = 0.0
    target_design_mode: str = "identity"
    target_design_delta: float = 0.0


def build_case_registry() -> pd.DataFrame:
    """Return the complete pre-method Stage 3B validation matrix.

    The matrix implements every generator-level Stage 3A factor needed before
    M1--M6: reduction cases, graph and residual-law stresses, isolated EO
    measurement-error components, a non-identity target-design transport case,
    support sensitivity, endpoint atoms, temporal dependence, and the C2
    negative control.
    """
    cases: List[CaseConfig] = []
    add = cases.append

    def c(
        case_id: str, scenario_id: str, case_name: str, case_type: str, common_random_group: str,
        n_rows: int, n_cols: int, spatial_rho: float, target_shift: str, overlap: str,
        fitted_graph: str, measurement_error: str, measurement_lambda: float, confounding: str,
        outcome_form: str, endpoint_structure: str, residual_law: str, covariate_law: str,
        sample_design: str, primary_target_mode: str, primary_target_dose: Optional[float],
        primary_bandwidth: Optional[float], *, temporal_dependence: bool = False,
        support_scale: str = "90m_primary", measurement_substrate_lambda: float = 0.0,
        target_design_mode: str = "identity", target_design_delta: float = 0.0,
    ) -> None:
        add(CaseConfig(
            case_id, scenario_id, case_name, case_type, common_random_group, n_rows, n_cols,
            spatial_rho, target_shift, overlap, fitted_graph, measurement_error, measurement_lambda,
            confounding, outcome_form, endpoint_structure, residual_law, covariate_law, sample_design,
            primary_target_mode, primary_target_dose, primary_bandwidth, temporal_dependence,
            support_scale, measurement_substrate_lambda, target_design_mode, target_design_delta,
        ))

    c("S1_BASE", "S1", "weak shift and negligible dependence", "scenario", "S1_BASE", 25, 25, 0.0, "weak", "good", "true_queen", "none", 0.0, "observed_moderate", "nonlinear", "mixed_atoms_0_1", "iid_continuous", "iid", "primary", "localized", 0.50, 0.10)
    c("S2_SPATIAL", "S2", "spatial dependence only", "scenario", "S2_SPATIAL", 25, 25, 0.6, "none", "good", "true_queen", "none", 0.0, "observed_moderate", "nonlinear", "mixed_atoms_0_1", "gaussian_gmrf", "iid", "primary", "observational", None, None)
    c("S3_SHIFT", "S3", "treatment shift only", "scenario", "S3_SHIFT", 25, 25, 0.0, "strong", "moderate", "true_queen", "none", 0.0, "observed_moderate", "nonlinear", "mixed_atoms_0_1", "iid_continuous", "iid", "primary", "localized", 0.90, 0.10)
    for rho in (0.0, 0.2, 0.4, 0.6, 0.8):
        c(f"S4_RHO{int(rho*100):03d}", "S4", f"combined shift rho={rho:.1f}", "scenario_factor", "S4_COMMON", 25, 25, rho, "strong", "moderate", "true_queen", "none", 0.0, "observed_moderate", "nonlinear", "mixed_atoms_0_1", "gaussian_gmrf", "spatial_gmrf", "primary", "localized", 0.90, 0.10)
    c("S4_RHO060_NONGAUSSIAN", "S4", "non-Gaussian graph-law validation", "residual_law_validation", "S4_COMMON", 25, 25, 0.6, "strong", "moderate", "true_queen", "none", 0.0, "observed_moderate", "nonlinear", "mixed_atoms_0_1", "transformed_gmrf_power_1_5", "spatial_gmrf", "primary", "localized", 0.90, 0.10)
    c("S4_RHO060_DESIGN_SHIFT", "S4", "known non-identity target-design shift", "target_design_transport_validation", "S4_COMMON", 25, 25, 0.6, "strong", "moderate", "true_queen", "none", 0.0, "observed_moderate", "nonlinear", "mixed_atoms_0_1", "gaussian_gmrf", "spatial_gmrf", "primary", "localized", 0.90, 0.10, target_design_mode="normal_location_shift", target_design_delta=0.75)
    c("S5_POOR_OVERLAP", "S5", "poor overlap and concentrated endpoint target", "scenario", "S5_POOR_OVERLAP", 25, 25, 0.4, "tail", "poor", "true_queen", "none", 0.0, "observed_strong", "nonlinear", "mixed_atoms_0_1", "gaussian_gmrf", "spatial_gmrf", "primary", "localized", 1.0, None)
    c("S6_ROOK", "S6", "fitted rook graph", "graph_misspecification", "S6_COMMON", 25, 25, 0.6, "moderate", "moderate", "rook", "none", 0.0, "observed_moderate", "nonlinear", "mixed_atoms_0_1", "gaussian_gmrf", "spatial_gmrf", "primary", "localized", 0.75, 0.10)
    c("S6_OMIT50", "S6", "fitted graph with 50 percent omitted edges", "graph_misspecification", "S6_COMMON", 25, 25, 0.6, "moderate", "moderate", "queen_omit50", "none", 0.0, "observed_moderate", "nonlinear", "mixed_atoms_0_1", "gaussian_gmrf", "spatial_gmrf", "primary", "localized", 0.75, 0.10)
    c("S7_EXCHANGEABLE", "S7", "benign exchangeable observational-law target", "scenario", "S7_EXCHANGEABLE", 25, 25, 0.0, "none", "good", "true_queen", "none", 0.0, "observed_none", "linear", "mixed_atoms_0_1", "iid_continuous", "iid", "primary", "observational", None, None)
    c("S8_SEVERE_TAIL", "S8", "severe tail weight concentration", "scenario_failure", "S8_SEVERE_TAIL", 25, 25, 0.6, "outside_near_boundary", "severe", "true_queen", "none", 0.0, "observed_strong", "nonlinear", "mixed_atoms_0_1", "gaussian_gmrf", "spatial_gmrf", "primary", "localized", 0.98, 0.10)
    c("S8_SMALL_CAL", "S8", "too little calibration information", "scenario_failure", "S8_SMALL_CAL", 25, 25, 0.6, "moderate", "moderate", "true_queen", "none", 0.0, "observed_moderate", "nonlinear", "mixed_atoms_0_1", "gaussian_gmrf", "spatial_gmrf", "small_calibration", "localized", 0.75, 0.10)

    # Isolate each proposal-registered EO error component before also supplying
    # one combined fixture. All variants share the same latent/measurement draws.
    c("S9_RANDOM", "S9", "heteroscedastic random EO error", "measurement_error_factor", "S9_COMMON", 25, 25, 0.4, "moderate", "moderate", "true_queen", "random", 0.0, "observed_moderate", "nonlinear", "mixed_atoms_0_1", "gaussian_gmrf", "spatial_gmrf", "primary", "localized", 0.75, 0.10)
    c("S9_TREATMENT05", "S9", "random plus treatment-correlated EO error lambda=0.5", "measurement_error_factor", "S9_COMMON", 25, 25, 0.4, "moderate", "moderate", "true_queen", "treatment_correlated", 0.5, "observed_moderate", "nonlinear", "mixed_atoms_0_1", "gaussian_gmrf", "spatial_gmrf", "primary", "localized", 0.75, 0.10)
    c("S9_TREATMENT10", "S9", "random plus treatment-correlated EO error lambda=1.0", "measurement_error_factor", "S9_COMMON", 25, 25, 0.4, "moderate", "moderate", "true_queen", "treatment_correlated", 1.0, "observed_moderate", "nonlinear", "mixed_atoms_0_1", "gaussian_gmrf", "spatial_gmrf", "primary", "localized", 0.75, 0.10)
    c("S9_SUBSTRATE", "S9", "random plus substrate-specific EO bias", "measurement_error_factor", "S9_COMMON", 25, 25, 0.4, "moderate", "moderate", "true_queen", "substrate_bias", 0.0, "observed_moderate", "nonlinear", "mixed_atoms_0_1", "gaussian_gmrf", "spatial_gmrf", "primary", "localized", 0.75, 0.10, measurement_substrate_lambda=0.10)
    c("S9_COMBINED05", "S9", "random plus treatment and substrate EO error", "measurement_error_factor", "S9_COMMON", 25, 25, 0.4, "moderate", "moderate", "true_queen", "combined", 0.5, "observed_moderate", "nonlinear", "mixed_atoms_0_1", "gaussian_gmrf", "spatial_gmrf", "primary", "localized", 0.75, 0.10, measurement_substrate_lambda=0.10)

    c("S10_90M", "S10", "primary 90 m support", "support_factor", "S10_90M", 25, 25, 0.4, "moderate", "moderate", "true_queen", "none", 0.0, "observed_moderate", "nonlinear", "mixed_atoms_0_1", "gaussian_gmrf", "spatial_gmrf", "maup_aligned", "localized", 0.75, 0.10, support_scale="90m_primary")
    c("S10_180M", "S10", "anchored 180 m support-level rerun", "support_factor", "S10_180M", 13, 13, 0.4, "moderate", "moderate", "true_queen", "none", 0.0, "observed_moderate", "nonlinear", "mixed_atoms_0_1", "gaussian_gmrf", "spatial_gmrf", "primary", "localized", 0.75, 0.10, support_scale="180m_anchored_rerun")
    c("ST1_MIXED_ATOMS", "ST1", "strong mixed endpoint atoms", "additional_stress", "ST1_COMMON", 25, 25, 0.4, "moderate", "moderate", "true_queen", "none", 0.0, "observed_moderate", "nonlinear", "mixed_atoms_strong", "gaussian_gmrf", "spatial_gmrf", "primary", "localized", 0.75, 0.10)
    c("ST1_INTERIOR_ONLY", "ST1", "interior-only reference treatment", "additional_stress", "ST1_COMMON", 25, 25, 0.4, "moderate", "moderate", "true_queen", "none", 0.0, "observed_moderate", "nonlinear", "interior_only", "gaussian_gmrf", "spatial_gmrf", "primary", "localized", 0.75, 0.10)
    c("ST2_TEMPORAL", "ST2", "repeated mine-year temporal dependence", "additional_stress", "ST2_TEMPORAL", 25, 25, 0.4, "moderate", "moderate", "true_queen", "none", 0.0, "observed_moderate", "nonlinear", "mixed_atoms_0_1", "gaussian_gmrf", "spatial_gmrf", "primary", "localized", 0.75, 0.10, temporal_dependence=True)
    c("ST3_HIDDEN_C2", "ST3", "unobserved spatial confounding negative control", "additional_stress", "ST3_HIDDEN_C2", 25, 25, 0.4, "moderate", "moderate", "true_queen", "none", 0.0, "hidden_C2_negative_control", "nonlinear", "mixed_atoms_0_1", "gaussian_gmrf", "spatial_gmrf", "primary", "localized", 0.75, 0.10)
    return pd.DataFrame([case.__dict__ for case in cases])

def frame_to_cases(frame: pd.DataFrame) -> List[CaseConfig]:
    result: List[CaseConfig] = []
    for row in frame.to_dict(orient="records"):
        for key in ("primary_target_dose", "primary_bandwidth"):
            if pd.isna(row[key]):
                row[key] = None
        row["temporal_dependence"] = bool(row["temporal_dependence"])
        result.append(CaseConfig(**row))
    return result


def role_assignment(n_rows: int, n_cols: int, sample_design: str) -> np.ndarray:
    cols = np.tile(np.arange(n_cols), n_rows)
    roles = np.full(n_rows * n_cols, "nuisance_training", dtype=object)
    if sample_design == "small_calibration" and n_cols == 25:
        roles[(cols >= 14) & (cols <= 18)] = "support_audit"
        roles[cols == 19] = "calibration"
        roles[cols >= 20] = "test_target"
        return roles
    if sample_design == "maup_aligned" and n_cols == 25:
        roles[(cols >= 10) & (cols <= 13)] = "support_audit"
        roles[(cols >= 14) & (cols <= 17)] = "calibration"
        roles[cols >= 18] = "test_target"
        return roles
    if n_cols == 25:
        roles[(cols >= 10) & (cols <= 14)] = "support_audit"
        roles[(cols >= 15) & (cols <= 19)] = "calibration"
        roles[cols >= 20] = "test_target"
    elif n_cols == 13:
        roles[(cols >= 6) & (cols <= 7)] = "support_audit"
        roles[(cols >= 8) & (cols <= 9)] = "calibration"
        roles[cols >= 10] = "test_target"
    else:
        raise Stage3BError(f"Unsupported grid width: {n_cols}")
    return roles


def treatment_parameters(
    case: CaseConfig,
    x1: np.ndarray,
    x2: np.ndarray,
    x3: np.ndarray,
    x4: np.ndarray,
    xcoord: np.ndarray,
    ycoord: np.ndarray,
    hidden_u: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(x1)
    if case.confounding == "observed_none":
        eta = np.full(n, 0.0)
        logits0 = np.full(n, -1.55)
        logits1 = np.full(n, -1.70)
    else:
        strength = 1.0 if case.confounding in {"observed_strong", "hidden_C2_negative_control"} else 0.65
        coord_term = 0.20 * xcoord if case.covariate_law == "spatial_gmrf" else 0.0
        eta = -0.10 + strength * (0.80 * x1 - 0.50 * x2 + 0.35 * x3 + coord_term)
        ycoord_term = 0.10 * ycoord if case.covariate_law == "spatial_gmrf" else 0.0
        logits0 = -1.60 + strength * (0.45 * x1 - 0.30 * x3 + ycoord_term)
        logits1 = -1.80 + strength * (-0.35 * x1 + 0.50 * x2 + 0.20 * x4)
        if case.confounding == "hidden_C2_negative_control":
            eta = eta + 1.00 * hidden_u
            logits0 = logits0 - 0.55 * hidden_u
            logits1 = logits1 + 0.55 * hidden_u

    if case.overlap == "good":
        kappa = 6.0
    elif case.overlap == "moderate":
        kappa = 10.0
    elif case.overlap == "poor":
        eta = eta - 1.40
        logits1 = logits1 - 2.0
        logits0 = logits0 + 0.6
        kappa = 24.0
    elif case.overlap == "severe":
        eta = eta - 2.20
        logits1 = logits1 - 3.5
        logits0 = logits0 + 1.0
        kappa = 45.0
    else:
        raise Stage3BError(f"Unknown overlap level: {case.overlap}")

    if case.endpoint_structure == "interior_only":
        probs = np.column_stack([np.zeros(n), np.zeros(n), np.ones(n)])
    else:
        if case.endpoint_structure == "mixed_atoms_strong":
            logits0 = logits0 + 0.75
            logits1 = logits1 + 0.55
        probs = softmax3(logits0, logits1, np.zeros(n))

    mean = expit(eta)
    alpha = np.clip(mean * kappa, 1.0, 100.0)
    beta = np.clip((1.0 - mean) * kappa, 1.0, 100.0)
    actual_mean = alpha / (alpha + beta)
    return probs, alpha, beta, actual_mean


def mixed_density_at(
    a: np.ndarray,
    pi0: np.ndarray,
    pi1: np.ndarray,
    pii: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    result = np.zeros(len(a), dtype=float)
    at0 = a == 0.0
    at1 = a == 1.0
    interior = (a > 0.0) & (a < 1.0)
    result[at0] = pi0[at0]
    result[at1] = pi1[at1]
    if interior.any():
        result[interior] = pii[interior] * np.exp(beta_log_pdf(a[interior], alpha[interior], beta[interior]))
    return result


def draw_mixed_treatment(
    rng: np.random.Generator,
    probs: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    return_latent: bool = False,
):
    # Draw every unit's latent interior value before category assignment. This
    # keeps factor comparisons paired even when endpoint probabilities change.
    interior_draw = rng.beta(alpha, beta)
    category_uniform = rng.random(len(alpha))
    cat = np.where(
        category_uniform < probs[:, 0],
        0,
        np.where(category_uniform < probs[:, 0] + probs[:, 1], 1, 2),
    )
    A = interior_draw.copy()
    A[cat == 0] = 0.0
    A[cat == 1] = 1.0
    if return_latent:
        return A, cat, interior_draw, category_uniform
    return A, cat


def outcome_baseline(
    case: CaseConfig,
    x1: np.ndarray,
    x2: np.ndarray,
    x3: np.ndarray,
    x4: np.ndarray,
    xcoord: np.ndarray,
    hidden_u: np.ndarray,
) -> np.ndarray:
    if case.outcome_form == "linear":
        base = 0.20 + 0.45 * x1 - 0.25 * x2 + 0.15 * x3
    else:
        coord_effect = 0.18 * np.sin(np.pi * xcoord) if case.covariate_law == "spatial_gmrf" else 0.0
        base = 0.25 + 0.60 * x1 - 0.38 * x2 + 0.22 * (x3 * x3 - 1.0) + coord_effect + 0.10 * x4
    if case.confounding == "hidden_C2_negative_control":
        base = base + 0.80 * hidden_u
    return base


def conditional_mean(case: CaseConfig, baseline: np.ndarray, x1: np.ndarray, a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    if case.outcome_form == "linear":
        return baseline + 0.50 * a
    return baseline + 0.55 * a - 0.35 * a * a + 0.25 * np.sin(2.0 * np.pi * a) + 0.20 * a * x1


def measurement_error(
    case: CaseConfig,
    a: np.ndarray,
    ue: np.ndarray,
    substrate: np.ndarray,
    z: np.ndarray,
) -> np.ndarray:
    """Generate the proposal-registered EO product-error components.

    One unit-level innovation is shared across its observed potential-outcome
    curve. Components are isolated in separate S9 fixtures and combined only in
    the explicitly named combined case.
    """
    a = np.asarray(a, dtype=float)
    if case.measurement_error == "none":
        return np.zeros_like(a, dtype=float)
    sigma = 0.035 + 0.006 * np.asarray(ue, dtype=float)
    random_term = sigma * np.asarray(z, dtype=float)
    treatment_term = case.measurement_lambda * (a - 0.5)
    substrate_term = case.measurement_substrate_lambda * np.asarray(substrate, dtype=float)
    if case.measurement_error == "random":
        return random_term
    if case.measurement_error == "treatment_correlated":
        return random_term + treatment_term
    if case.measurement_error == "substrate_bias":
        return random_term + substrate_term
    if case.measurement_error == "combined":
        return random_term + treatment_term + substrate_term
    raise Stage3BError(f"Unknown measurement-error mode: {case.measurement_error}")


def fitted_graph_edges(case: CaseConfig, basis: GraphBasis, seed: int) -> pd.DataFrame:
    if case.fitted_graph == "true_queen":
        return basis.edges_queen.copy()
    if case.fitted_graph == "rook":
        return basis.edges_rook.copy()
    if case.fitted_graph == "queen_omit50":
        rng = np.random.default_rng(derived_seed(seed, case.case_id + "_FIT_GRAPH"))
        keep = rng.random(len(basis.edges_queen)) < 0.50
        frame = basis.edges_queen.loc[keep].copy().reset_index(drop=True)
        require(len(frame) > 0, "Omitted-edge graph is empty")
        return frame
    raise Stage3BError(f"Unknown fitted graph: {case.fitted_graph}")


def source_seed_row(stage3a: Dict[str, Any], scenario_id: str, replication: int = 1) -> Dict[str, int]:
    seeds = stage3a["frames"]["seed_registry.csv"]
    source_scenario = scenario_id if scenario_id.startswith("S") and scenario_id[1:].isdigit() else "S4"
    require(1 <= int(replication) <= 20, "Stage 3B only permits the frozen pilot replications 1-20")
    rows = seeds[(seeds["phase"] == "pilot") & (seeds["scenario_id"] == source_scenario) & (seeds["replication"] == int(replication))]
    require(len(rows) == 1, f"Missing pilot seed row for {source_scenario}")
    row = rows.iloc[0]
    return {c: int(row[c]) for c in ["data_seed", "split_seed", "nuisance_seed", "orbit_seed", "measurement_seed", "target_draw_seed"]}


def q_at_observed(a: np.ndarray, target_dose: float, bandwidth: Optional[float], endpoint_audited: bool) -> np.ndarray:
    if target_dose == 0.0:
        return (a == 0.0).astype(float) if endpoint_audited else np.zeros(len(a))
    if target_dose == 1.0:
        return (a == 1.0).astype(float) if endpoint_audited else np.zeros(len(a))
    require(bandwidth is not None, "Interior target requires bandwidth")
    return truncated_normal_density(a.copy(), float(target_dose), float(bandwidth))


def summarize_weights(logw: np.ndarray) -> Dict[str, float]:
    finite = np.isfinite(logw)
    if not finite.any():
        return {"positive_weight_count": 0, "oracle_ess": 0.0, "max_normalized_weight": 1.0, "log_weight_range": float("inf")}
    values = logw[finite]
    m = float(np.max(values))
    weights = np.exp(values - m)
    total = float(np.sum(weights))
    norm = weights / total
    ess = float(1.0 / np.sum(norm * norm))
    return {
        "positive_weight_count": int(finite.sum()),
        "oracle_ess": ess,
        "max_normalized_weight": float(np.max(norm)),
        "log_weight_range": float(np.max(values) - np.min(values)),
    }


def mixed_law_normalization_error(
    probs: np.ndarray, alpha: np.ndarray, beta: np.ndarray, representative_units: int = 12
) -> float:
    indices = np.linspace(0, len(alpha) - 1, representative_units, dtype=int)
    x_raw, w_raw = np.polynomial.legendre.leggauss(1024)
    x = 0.5 * (x_raw + 1.0)
    w = 0.5 * w_raw
    errors: List[float] = []
    for idx in indices:
        density = np.exp(beta_log_pdf(x, np.full_like(x, alpha[idx]), np.full_like(x, beta[idx])))
        integral = float(np.sum(w * density))
        total = float(probs[idx, 0] + probs[idx, 1] + probs[idx, 2] * integral)
        errors.append(abs(total - 1.0))
    return float(max(errors))


def generate_case(
    case: CaseConfig,
    stage3a: Dict[str, Any],
    basis_cache: Dict[Tuple[Any, ...], GraphBasis],
    replication: int = VALIDATION_REPLICATION,
) -> Dict[str, Any]:
    key = (case.n_rows, case.n_cols, case.sample_design)
    if key not in basis_cache:
        basis_cache[key] = GraphBasis.create(case.n_rows, case.n_cols, case.sample_design)
    basis = basis_cache[key]
    source = source_seed_row(stage3a, case.scenario_id, replication=replication)
    data_seed = derived_seed(source["data_seed"], case.common_random_group)
    measurement_seed = derived_seed(source["measurement_seed"], case.common_random_group)
    target_seed = derived_seed(source["target_draw_seed"], case.common_random_group)
    orbit_seed = derived_seed(source["orbit_seed"], case.case_id)
    covariate_seed = derived_seed(data_seed, "COVARIATES")
    treatment_seed = derived_seed(data_seed, "TREATMENT")
    residual_seed = derived_seed(data_seed, "RESIDUAL")
    temporal_seed = derived_seed(data_seed, "TEMPORAL")
    rng_cov = np.random.default_rng(covariate_seed)
    rng_treatment = np.random.default_rng(treatment_seed)
    rng_residual = np.random.default_rng(residual_seed)
    rng_measure = np.random.default_rng(measurement_seed)
    rng_temporal = np.random.default_rng(temporal_seed)

    n = basis.n
    rows = np.repeat(np.arange(case.n_rows), case.n_cols)
    cols = np.tile(np.arange(case.n_cols), case.n_rows)
    xcoord = (cols - (case.n_cols - 1) / 2.0) / max((case.n_cols - 1) / 2.0, 1.0)
    ycoord = (rows - (case.n_rows - 1) / 2.0) / max((case.n_rows - 1) / 2.0, 1.0)
    roles = role_assignment(case.n_rows, case.n_cols, case.sample_design)
    information_component = np.where(
        roles == "nuisance_training",
        "independent_nuisance_training",
        np.where(roles == "support_audit", "independent_support_audit", "calibration_target_inference_graph"),
    )

    if case.covariate_law == "iid":
        x1 = rng_cov.normal(0.0, 0.80, size=n)
        x2 = rng_cov.normal(0.0, 0.80, size=n)
        hidden_u = rng_cov.normal(0.0, 0.70, size=n)
        eig_x1 = eig_x2 = 1.0 / (0.80 * 0.80)
        eig_u = 1.0 / (0.70 * 0.70)
    elif case.covariate_law == "spatial_gmrf":
        x1, eig_x1 = basis.gmrf_draw(0.45, 0.80, rng_cov)
        x2, eig_x2 = basis.gmrf_draw(0.25, 0.80, rng_cov)
        hidden_u, eig_u = basis.gmrf_draw(0.50, 0.70, rng_cov)
    else:
        raise Stage3BError(f"Unknown covariate law: {case.covariate_law}")
    x3_base = rng_cov.normal(size=n)
    x4 = rng_cov.uniform(-1.0, 1.0, size=n)
    substrate = rng_cov.normal(size=n)

    # The scalable N2/N3 target-design object is deliberately one-dimensional
    # and excludes raw coordinates. Observational-design slots are nuisance +
    # calibration; target-design slots are support-audit + final-test. A normal
    # location shift gives a known, everywhere-positive ratio
    # r(d)=exp(delta*d-delta^2/2). All ordinary cases use the identity ratio.
    target_design_mask = np.isin(roles, ["support_audit", "test_target"])
    x3 = x3_base.copy()
    if case.target_design_mode == "identity":
        oracle_design_ratio = np.ones(n, dtype=float)
    elif case.target_design_mode == "normal_location_shift":
        require(case.target_design_delta > 0.0, "Target-design shift requires positive delta")
        x3[target_design_mask] += case.target_design_delta
        oracle_design_ratio = np.exp(case.target_design_delta * x3 - 0.5 * case.target_design_delta ** 2)
    else:
        raise Stage3BError(f"Unknown target-design mode: {case.target_design_mode}")
    log_oracle_design_ratio = np.log(oracle_design_ratio)

    probs, alpha, beta, beta_mean = treatment_parameters(case, x1, x2, x3, x4, xcoord, ycoord, hidden_u)
    A, cat, latent_interior_draw, treatment_category_uniform = draw_mixed_treatment(
        rng_treatment, probs, alpha, beta, return_latent=True
    )
    gA = mixed_density_at(A, probs[:, 0], probs[:, 1], probs[:, 2], alpha, beta)
    require(np.isfinite(gA).all() and (gA > 0).all(), f"Non-positive oracle treatment likelihood in {case.case_id}")
    normalization_error = mixed_law_normalization_error(probs, alpha, beta)

    baseline = outcome_baseline(case, x1, x2, x3, x4, xcoord, hidden_u)
    # Common-random-number design: every factor variant in a registered group uses
    # the same standard-normal innovation. Changing rho or the residual transform
    # changes only the registered residual-law factor, not the underlying random draw.
    residual_base_innovation = rng_residual.standard_normal(n)
    if case.residual_law == "iid_continuous":
        latent_z = 0.35 * residual_base_innovation
        residual = latent_z.copy()
        eig_r = 1.0 / (0.35 * 0.35)
        transform_power = 1.0
    else:
        latent_z, eig_r = basis.gmrf_draw_from_innovation(case.spatial_rho, 0.35, residual_base_innovation)
        if case.residual_law == "transformed_gmrf_power_1_5":
            transform_power = 1.5
            residual = np.sign(latent_z) * np.abs(latent_z) ** transform_power
        else:
            transform_power = 1.0
            residual = latent_z.copy()

    mean_A = conditional_mean(case, baseline, x1, A)
    y_true = mean_A + residual
    ue = np.clip(10.0 + 4.0 * np.abs(x2) + 2.0 * (x4 > 0) + rng_measure.normal(0.0, 1.5, n), 0.0, 40.0)
    z_measure_unit = rng_measure.normal(size=n)
    meas_fact = measurement_error(case, A, ue, substrate, z_measure_unit)
    y_observed = y_true + meas_fact
    factual_error = y_observed - y_true

    fitted_edges = fitted_graph_edges(case, basis, orbit_seed)
    units = pd.DataFrame(
        {
            "case_id": case.case_id,
            "scenario_id": case.scenario_id,
            "replication": int(replication),
            "seed_phase": "pilot",
            "common_random_group": case.common_random_group,
            "unit_id": [f"{case.case_id}_REP{int(replication):03d}_R{r:02d}_C{c:02d}" for r, c in zip(rows, cols)],
            "node_index": np.arange(n, dtype=int),
            "grid_row": rows,
            "grid_col": cols,
            "role": roles,
            "information_component": information_component,
            "covariate_law": case.covariate_law,
            "x_coord": xcoord,
            "y_coord": ycoord,
            "X_spatial_1": x1,
            "X_spatial_2": x2,
            "X_nonlinear_1": x3,
            "X_nonlinear_1_base": x3_base,
            "X_nonlinear_2": x4,
            "target_design_class": np.where(target_design_mask, "target_design", "observational_design"),
            "target_design_mode": case.target_design_mode,
            "target_design_delta": case.target_design_delta,
            "oracle_target_design_ratio": oracle_design_ratio,
            "log_oracle_target_design_ratio": log_oracle_design_ratio,
            "hidden_spatial_confounder": hidden_u,
            "hidden_confounder_observed_by_method": False,
            "substrate_proxy_hidden": substrate,
            "treatment_category": np.where(cat == 0, "atom_0", np.where(cat == 1, "atom_1", "interior")),
            "treatment_latent_interior_draw": latent_interior_draw,
            "treatment_category_uniform": treatment_category_uniform,
            "A": A,
            "pi_atom_0": probs[:, 0],
            "pi_atom_1": probs[:, 1],
            "pi_interior": probs[:, 2],
            "beta_alpha": alpha,
            "beta_beta": beta,
            "beta_mean": beta_mean,
            "oracle_g_mixed_at_A": gA,
            "outcome_baseline": baseline,
            "conditional_mean_at_A": mean_A,
            "residual_base_innovation": residual_base_innovation,
            "residual_latent_gaussian": latent_z,
            "residual_transform_power": transform_power,
            "shared_spatial_residual": residual,
            "Y_true_at_A": y_true,
            "ue_proxy": ue,
            "measurement_base_innovation": z_measure_unit,
            "measurement_error_mode": case.measurement_error,
            "measurement_treatment_lambda": case.measurement_lambda,
            "measurement_substrate_lambda": case.measurement_substrate_lambda,
            "measurement_error_at_A": meas_fact,
            "Y_observed_at_A": y_observed,
            "graph_degree_true": basis.W_queen.sum(axis=1).astype(int),
            "support_scale": case.support_scale,
            "endpoint_audited": case.endpoint_structure != "interior_only",
            "covariate_seed": covariate_seed,
            "treatment_seed": treatment_seed,
            "residual_seed": residual_seed,
            "measurement_seed": measurement_seed,
        }
    )

    truth_parts: List[pd.DataFrame] = []
    transport_parts: List[pd.DataFrame] = []
    target_parts: List[pd.DataFrame] = []
    support_rows: List[Dict[str, Any]] = []
    test_mask = roles == "test_target"
    calibration_mask = roles == "calibration"
    test_indices = np.where(test_mask)[0]

    target_specs: List[Tuple[float, Optional[float], str]] = []
    for d in PRIMARY_DOSES:
        bw = None if d in (0.0, 1.0) else PRIMARY_BANDWIDTH
        if case.primary_target_mode == "localized" and case.primary_target_dose is not None and abs(float(d) - float(case.primary_target_dose)) < 1e-12:
            bw = case.primary_bandwidth if d not in (0.0, 1.0) else None
        target_specs.append((float(d), bw, "registered_grid"))
    if case.primary_target_mode == "localized" and case.primary_target_dose is not None and all(abs(float(case.primary_target_dose) - d) > 1e-12 for d in PRIMARY_DOSES):
        target_specs.append((float(case.primary_target_dose), case.primary_bandwidth, "scenario_primary_extra"))
    target_specs = sorted(target_specs, key=lambda x: x[0])
    for target_dose, bandwidth, target_scope in target_specs:
        endpoint = target_dose in (0.0, 1.0)
        endpoint_audited = case.endpoint_structure != "interior_only"
        a_vec = np.full(n, target_dose, dtype=float)
        mean_dose = conditional_mean(case, baseline, x1, a_vec)
        meas_dose = measurement_error(case, a_vec, ue, substrate, z_measure_unit)
        truth_parts.append(
            pd.DataFrame(
                {
                    "case_id": case.case_id,
                    "scenario_id": case.scenario_id,
                    "replication": int(replication),
                    "unit_id": units["unit_id"],
                    "target_dose": target_dose,
                    "target_scope": target_scope,
                    "bandwidth": np.nan if endpoint else float(bandwidth),
                    "conditional_mean": mean_dose,
                    "shared_spatial_residual": residual,
                    "Y_true": mean_dose + residual,
                    "measurement_error": meas_dose,
                    "Y_observed": mean_dose + residual + meas_dose,
                    "truth_role": "hard_dose_secondary_truth",
                }
            )
        )

        q_obs = q_at_observed(A.copy(), target_dose, bandwidth, endpoint_audited)
        log_ratio = np.full(n, -np.inf, dtype=float)
        positive = q_obs > 0
        log_ratio[positive] = np.log(q_obs[positive]) - np.log(gA[positive])
        transport_parts.append(
            pd.DataFrame(
                {
                    "case_id": case.case_id,
                    "scenario_id": case.scenario_id,
                    "replication": int(replication),
                    "unit_id": units["unit_id"],
                    "role": roles,
                    "target_dose": target_dose,
                    "target_scope": target_scope,
                    "bandwidth": bandwidth,
                    "q_at_observed_A": q_obs,
                    "g_at_observed_A": gA,
                    "log_oracle_treatment_ratio": log_ratio,
                    "positive_target_weight": positive,
                    "endpoint_audited": endpoint_audited,
                }
            )
        )
        cal_summary = summarize_weights(log_ratio[calibration_mask])
        support_rows.append(
            {
                "case_id": case.case_id,
                "scenario_id": case.scenario_id,
                "replication": int(replication),
                "target_dose": target_dose,
                "target_scope": target_scope,
                "bandwidth": bandwidth,
                "calibration_count": int(calibration_mask.sum()),
                **cal_summary,
                "registered_overlap": case.overlap,
                "mathematical_positivity_holds": not (case.endpoint_structure == "interior_only" and endpoint),
                "scenario_stress_label": (
                    "endpoint_not_audited"
                    if (case.endpoint_structure == "interior_only" and endpoint)
                    else "small_calibration_information"
                    if case.case_id == "S8_SMALL_CAL"
                    else "severe_weight_concentration_expected"
                    if (case.case_id == "S8_SEVERE_TAIL" and target_dose >= 0.75)
                    else "poor_overlap_weight_concentration_expected"
                    if (case.scenario_id == "S5" and target_dose >= 0.90)
                    else "ordinary_support_diagnostic"
                ),
                "operational_eligibility_status": "deferred_until_pilot_threshold_freeze",
                "threshold_status": "diagnostic_only_thresholds_freeze_after_pilot",
            }
        )

        if endpoint and not endpoint_audited:
            target_parts.append(
                pd.DataFrame(
                    {
                        "case_id": case.case_id,
                        "scenario_id": case.scenario_id,
                        "replication": int(replication),
                        "unit_id": units.loc[test_mask, "unit_id"].to_numpy(),
                        "target_dose": target_dose,
                        "target_scope": target_scope,
                        "bandwidth": np.nan,
                        "intervention_type": "unaudited_endpoint",
                        "draw_status": "R02_ENDPOINT_NOT_AUDITED",
                        "A_star": np.nan,
                        "q_at_A_star": np.nan,
                        "g_at_A_star": np.nan,
                        "log_oracle_ratio_at_A_star": np.nan,
                        "conditional_mean_at_A_star": np.nan,
                        "shared_spatial_residual": residual[test_mask],
                        "Y_true_at_A_star": np.nan,
                        "measurement_error_at_A_star": np.nan,
                        "Y_observed_at_A_star": np.nan,
                        "target_draw_seed": target_seed,
                        "target_draw_subseed": derived_seed(target_seed, f"TARGET_{target_dose:.12g}_{target_scope}"),
                        "measurement_seed": measurement_seed,
                        "target_supported_by_generator": False,
                    }
                )
            )
            continue

        target_subseed = derived_seed(target_seed, f"TARGET_{target_dose:.12g}_{target_scope}")
        rng_target_local = np.random.default_rng(target_subseed)
        if endpoint:
            astar = np.full(len(test_indices), target_dose, dtype=float)
            qstar = np.ones(len(test_indices), dtype=float)
            intervention_type = "audited_endpoint_atom"
        else:
            require(bandwidth is not None, "Interior target bandwidth missing")
            astar = draw_truncated_normal(rng_target_local, target_dose, float(bandwidth), len(test_indices))
            qstar = truncated_normal_density(astar.copy(), target_dose, float(bandwidth))
            intervention_type = "truncated_gaussian_kernel_on_open_unit_interval"
        gstar = mixed_density_at(
            astar,
            probs[test_mask, 0],
            probs[test_mask, 1],
            probs[test_mask, 2],
            alpha[test_mask],
            beta[test_mask],
        )
        logstar = np.log(qstar) - np.log(gstar)
        mean_star = conditional_mean(case, baseline[test_mask], x1[test_mask], astar)
        meas_star = measurement_error(case, astar, ue[test_mask], substrate[test_mask], z_measure_unit[test_mask])
        target_supported = np.isfinite(logstar) & (gstar > 0)
        target_parts.append(
            pd.DataFrame(
                {
                    "case_id": case.case_id,
                    "scenario_id": case.scenario_id,
                    "replication": int(replication),
                    "unit_id": units.loc[test_mask, "unit_id"].to_numpy(),
                    "target_dose": target_dose,
                    "target_scope": target_scope,
                    "bandwidth": np.nan if endpoint else float(bandwidth),
                    "intervention_type": intervention_type,
                    "draw_status": "drawn",
                    "A_star": astar,
                    "q_at_A_star": qstar,
                    "g_at_A_star": gstar,
                    "log_oracle_treatment_ratio_at_A_star": logstar,
                    "conditional_mean_at_A_star": mean_star,
                    "shared_spatial_residual": residual[test_mask],
                    "Y_true_at_A_star": mean_star + residual[test_mask],
                    "measurement_error_at_A_star": meas_star,
                    "Y_observed_at_A_star": mean_star + residual[test_mask] + meas_star,
                    "target_draw_seed": target_seed,
                    "target_draw_subseed": target_subseed,
                    "measurement_seed": measurement_seed,
                    "target_supported_by_generator": target_supported,
                }
            )
        )

    observational_targets = pd.DataFrame()
    if case.primary_target_mode == "observational":
        probs_t = probs[test_mask]
        alpha_t, beta_t = alpha[test_mask], beta[test_mask]
        observational_target_subseed = derived_seed(target_seed, "OBSERVATIONAL_TARGET")
        rng_target_observational = np.random.default_rng(observational_target_subseed)
        astar_obs, _ = draw_mixed_treatment(rng_target_observational, probs_t, alpha_t, beta_t)
        mean_obs = conditional_mean(case, baseline[test_mask], x1[test_mask], astar_obs)
        meas_obs = measurement_error(case, astar_obs, ue[test_mask], substrate[test_mask], z_measure_unit[test_mask])
        observational_targets = pd.DataFrame(
            {
                "case_id": case.case_id,
                "scenario_id": case.scenario_id,
                "replication": int(replication),
                "unit_id": units.loc[test_mask, "unit_id"].to_numpy(),
                "intervention_type": "observational_mixed_law",
                "A_star": astar_obs,
                "q_equals_g": True,
                "oracle_treatment_ratio": 1.0,
                "conditional_mean_at_A_star": mean_obs,
                "shared_spatial_residual": residual[test_mask],
                "Y_true_at_A_star": mean_obs + residual[test_mask],
                "measurement_error_at_A_star": meas_obs,
                "Y_observed_at_A_star": mean_obs + residual[test_mask] + meas_obs,
                "target_draw_seed": target_seed,
                "target_draw_subseed": observational_target_subseed,
            }
        )

    temporal = pd.DataFrame()
    if case.temporal_dependence:
        phi = 0.70
        temporal_parts: List[pd.DataFrame] = []
        state = residual.copy()
        for year_index, year in enumerate((2023, 2024, 2025)):
            if year_index > 0:
                innovation, _ = basis.gmrf_draw(case.spatial_rho, 0.35, rng_temporal)
                state = phi * state + math.sqrt(1.0 - phi * phi) * innovation
            year_mean = mean_A + 0.03 * year_index
            temporal_parts.append(
                pd.DataFrame(
                    {
                        "case_id": case.case_id,
                        "scenario_id": case.scenario_id,
                        "replication": int(replication),
                        "unit_id": units["unit_id"],
                        "year": year,
                        "role": roles,
                        "A": A,
                        "temporal_phi": phi,
                        "spatiotemporal_residual": state,
                        "Y_true": year_mean + state,
                        "row_split_forbidden": True,
                    }
                )
            )
        temporal = pd.concat(temporal_parts, ignore_index=True)

    target_design_truth = units[[
        "case_id", "scenario_id", "replication", "unit_id", "node_index", "role",
        "target_design_class", "target_design_mode", "target_design_delta",
        "X_nonlinear_1", "oracle_target_design_ratio", "log_oracle_target_design_ratio",
    ]].copy()
    target_design_truth["target_design_ratio_feature"] = "X_nonlinear_1"
    target_design_truth["ratio_training_role"] = np.where(
        target_design_truth.role == "nuisance_training", "observational_training",
        np.where(target_design_truth.role == "support_audit", "target_training", "not_ratio_training"),
    )
    target_design_truth["ratio_evaluation_role"] = np.where(
        target_design_truth.role == "calibration", "observational_evaluation",
        np.where(target_design_truth.role == "test_target", "target_evaluation", "not_ratio_evaluation"),
    )
    n_design_obs = int((roles == "nuisance_training").sum())
    n_design_target = int((roles == "support_audit").sum())
    require(n_design_obs > 0 and n_design_target > 0, "Target-design ratio training roles are empty")
    target_design_truth["ratio_training_n_observational"] = n_design_obs
    target_design_truth["ratio_training_n_target"] = n_design_target
    target_design_truth["class_prior_odds_n0_over_n1"] = n_design_obs / n_design_target
    target_design_truth["analytic_ratio_normalization"] = 1.0
    target_design_truth["ratio_training_dependence_regime"] = "iid_design_feature_X_nonlinear_1"
    target_design_truth["class_prior_odds_required"] = True

    diagnostics = {
        "case_id": case.case_id,
        "scenario_id": case.scenario_id,
        "common_random_group": case.common_random_group,
        "replication": int(replication),
        "seed_phase": "pilot",
        "unit_count": n,
        "nuisance_training_count": int((roles == "nuisance_training").sum()),
        "support_audit_count": int((roles == "support_audit").sum()),
        "calibration_count": int(calibration_mask.sum()),
        "test_target_count": int(test_mask.sum()),
        "true_queen_edge_count": int(len(basis.edges_queen)),
        "fitted_edge_count": int(len(fitted_edges)),
        "atom_0_count": int((A == 0.0).sum()),
        "atom_1_count": int((A == 1.0).sum()),
        "interior_count": int(((A > 0.0) & (A < 1.0)).sum()),
        "treatment_min": float(A.min()),
        "treatment_max": float(A.max()),
        "oracle_g_min": float(gA.min()),
        "oracle_g_max": float(gA.max()),
        "mixed_treatment_normalization_max_abs_error": float(normalization_error),
        "residual_min_precision_eigenvalue": float(eig_r),
        "covariate_min_precision_eigenvalue": float(min(eig_x1, eig_x2, eig_u)),
        "residual_law": case.residual_law,
        "residual_transform_power": transform_power,
        "measurement_error_sd": float(np.std(factual_error)),
        "measurement_error_mode": case.measurement_error,
        "measurement_treatment_lambda": case.measurement_lambda,
        "measurement_substrate_lambda": case.measurement_substrate_lambda,
        "measurement_error_potential_process": "shared_unit_innovation_with_explicit_random_treatment_and_substrate_components",
        "target_design_mode": case.target_design_mode,
        "target_design_delta": case.target_design_delta,
        "target_design_ratio_feature": "X_nonlinear_1",
        "target_design_ratio_analytic_normalization": 1.0,
        "measurement_error_A_correlation": float(np.corrcoef(A, factual_error)[0, 1]) if np.std(factual_error) > 0 else 0.0,
        "hidden_confounding_active": case.confounding == "hidden_C2_negative_control",
        "temporal_dependence_active": case.temporal_dependence,
        "support_scale": case.support_scale,
        "covariate_law": case.covariate_law,
        "information_graph_cut_after_cols": ";".join(str(x) for x in basis.cut_after_cols),
        "data_seed": data_seed,
        "covariate_seed": covariate_seed,
        "treatment_seed": treatment_seed,
        "residual_seed": residual_seed,
        "temporal_seed": temporal_seed,
        "measurement_seed": measurement_seed,
        "target_draw_seed": target_seed,
        "orbit_seed": orbit_seed,
        "dense_precision_inverse_formed": False,
    }

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        hard_truth_frame = pd.concat(truth_parts, ignore_index=True)
        transport_truth_frame = pd.concat(transport_parts, ignore_index=True)
        target_draw_frame = pd.concat(target_parts, ignore_index=True)
    return {
        "units": units,
        "hard_truth": hard_truth_frame,
        "treatment_transport_truth": transport_truth_frame,
        "target_design_truth": target_design_truth,
        "target_draws": target_draw_frame,
        "observational_target_draws": observational_targets,
        "support_diagnostics": pd.DataFrame(support_rows),
        "temporal": temporal,
        "fitted_edges": fitted_edges.assign(case_id=case.case_id, scenario_id=case.scenario_id, replication=int(replication)),
        "true_edges": basis.edges_queen,
        "diagnostics": diagnostics,
    }


def build_support_180m_map() -> pd.DataFrame:
    rows = np.repeat(np.arange(25), 25)
    cols = np.tile(np.arange(25), 25)
    agg_r = rows // 2
    agg_c = cols // 2
    frame = pd.DataFrame(
        {
            "unit_90m": [f"R{r:02d}_C{c:02d}" for r, c in zip(rows, cols)],
            "grid_row_90m": rows,
            "grid_col_90m": cols,
            "unit_180m": [f"AGG_R{r:02d}_C{c:02d}" for r, c in zip(agg_r, agg_c)],
            "grid_row_180m": agg_r,
            "grid_col_180m": agg_c,
            "anchored_rule": "floor(grid_index/2)",
        }
    )
    sizes = frame.groupby("unit_180m").size().rename("member_count")
    frame = frame.merge(sizes, on="unit_180m", how="left")
    frame["aggregation_weight"] = 1.0 / frame["member_count"]
    return frame


def build_paired_s10_aggregation(units: pd.DataFrame, support_map: pd.DataFrame) -> pd.DataFrame:
    source = units[units["case_id"] == "S10_90M"].copy()
    require(len(source) == 625, "Missing S10_90M source for paired aggregation")
    source["map_key"] = [f"R{int(r):02d}_C{int(c):02d}" for r, c in zip(source.grid_row, source.grid_col)]
    merged = source.merge(support_map, left_on="map_key", right_on="unit_90m", how="left", validate="one_to_one")
    require(merged.unit_180m.notna().all(), "Incomplete S10 aggregation map")
    role_nunique = merged.groupby("unit_180m")["role"].nunique()
    require((role_nunique == 1).all(), "S10 aggregation crosses frozen role boundaries")
    numeric_mean = [
        "X_spatial_1", "X_spatial_2", "X_nonlinear_1", "X_nonlinear_2",
        "A", "outcome_baseline", "conditional_mean_at_A", "shared_spatial_residual",
        "Y_true_at_A", "ue_proxy", "measurement_error_at_A", "Y_observed_at_A",
    ]
    grouped = merged.groupby(["unit_180m", "grid_row_180m", "grid_col_180m"], as_index=False)
    agg = grouped[numeric_mean].mean()
    meta = grouped.agg(
        member_count=("unit_90m", "size"),
        role=("role", "first"),
        source_case_id=("case_id", "first"),
        replication=("replication", "first"),
    )
    out = meta.merge(agg, on=["unit_180m", "grid_row_180m", "grid_col_180m"], validate="one_to_one")
    out["case_id"] = "S10_PAIRED_AGGREGATION"
    out["scenario_id"] = "S10"
    out["support_scale"] = "180m_anchored_paired_aggregation"
    out["treatment_likelihood_status"] = "not_closed_form_for_aggregated_mixed_treatment; theorem branch uses S10_180M support-level rerun"
    return out


def build_exact_orbit_fixtures(
    units: pd.DataFrame, target_draws: pd.DataFrame, true_edges: pd.DataFrame
) -> Dict[str, Any]:
    """Build theorem-ready fixed-slot/movable-payload fixtures.

    Calibration slots carry factual observed payloads. The unique target slot
    carries the realized localized-intervention treatment and its simulation
    truth response from Stage 3B, not the factual observational target payload.
    Stage 3D must replace the target response by each candidate y during G3
    inversion; the stored truth value is an oracle coverage-test reference.
    """
    source_case = "S4_RHO060"
    base = units[units["case_id"] == source_case].copy()
    require(len(base) == 625, "Missing S4_RHO060 fixture source")
    target_reference = target_draws[
        (target_draws.case_id == source_case)
        & (np.isclose(target_draws.target_dose.astype(float), 0.90))
        & (target_draws.draw_status == "drawn")
    ].copy()
    require(target_reference.unit_id.is_unique and len(target_reference) == int((base.role == "test_target").sum()), "Missing unique target-draw fixture references")
    target_lookup = target_reference.set_index("unit_id")

    coords8 = [(11, 18), (11, 19), (12, 18), (12, 19), (13, 18), (13, 19), (14, 19), (12, 20)]
    coords6 = [(11, 19), (12, 18), (12, 19), (13, 18), (13, 19), (12, 20)]

    def select_coords(coords: Sequence[Tuple[int, int]]) -> pd.DataFrame:
        order = {tuple(rc): i for i, rc in enumerate(coords)}
        frame = base[base.apply(lambda r: (int(r.grid_row), int(r.grid_col)) in order, axis=1)].copy()
        frame["_fixture_order"] = [order[(int(r), int(c))] for r, c in zip(frame.grid_row, frame.grid_col)]
        frame = frame.sort_values("_fixture_order").drop(columns="_fixture_order").reset_index(drop=True)
        require(len(frame) == len(coords), "Could not construct exact-orbit fixture coordinates")
        require((frame.role == "test_target").sum() == 1, "Exact-orbit fixture must contain exactly one target slot")
        require((frame.role == "calibration").sum() == len(frame) - 1, "All non-target exact-orbit slots must be calibration slots")
        return frame

    selected6 = select_coords(coords6)
    selected8 = select_coords(coords8)

    def make_fixture(frame: pd.DataFrame, duplicate_first_pair: bool = False) -> Dict[str, Any]:
        slots = frame[["unit_id", "node_index", "grid_row", "grid_col", "role"]].to_dict(orient="records")
        payloads: List[Dict[str, Any]] = []
        target_truth_reference: Dict[str, Any] | None = None
        for row in frame.itertuples(index=False):
            if row.role == "test_target":
                ref = target_lookup.loc[row.unit_id]
                payload = {
                    "A": float(ref.A_star),
                    "Y": float(ref.Y_observed_at_A_star),
                    "payload_origin": "realized_localized_target_truth_reference",
                }
                target_truth_reference = {
                    "unit_id": row.unit_id,
                    "target_dose": float(ref.target_dose),
                    "bandwidth": float(ref.bandwidth),
                    "A_star": float(ref.A_star),
                    "Y_true_at_A_star": float(ref.Y_true_at_A_star),
                    "Y_observed_at_A_star": float(ref.Y_observed_at_A_star),
                    "G3_rule": "replace target payload Y by candidate y for every inversion state; stored Y is oracle truth for simulation coverage testing",
                }
            else:
                payload = {
                    "A": float(row.A),
                    "Y": float(row.Y_observed_at_A),
                    "payload_origin": "factual_calibration_observation",
                }
            payloads.append(payload)
        require(target_truth_reference is not None, "Target truth reference missing from orbit fixture")
        if duplicate_first_pair:
            require(slots[0]["role"] == "calibration" and slots[1]["role"] == "calibration", "Duplicate fixture pair must be calibration slots")
            payloads[1] = dict(payloads[0])
        node_set = set(int(x) for x in frame.node_index)
        internal = true_edges[true_edges.source_node.isin(node_set) & true_edges.target_node.isin(node_set)][["source_node", "target_node"]]
        boundary = true_edges[true_edges.source_node.isin(node_set) ^ true_edges.target_node.isin(node_set)][["source_node", "target_node"]]
        keys = [(float(payload["A"]), float(payload["Y"])) for payload in payloads]
        multiplicities: Dict[str, int] = {}
        for key in keys:
            label = f"A={key[0]:.17g}|Y={key[1]:.17g}"
            multiplicities[label] = multiplicities.get(label, 0) + 1
        distinct = math.factorial(len(payloads))
        for count in multiplicities.values():
            distinct //= math.factorial(count)
        target_positions = [i for i, slot in enumerate(slots) if slot["role"] == "test_target"]
        require(len(target_positions) == 1, "Exact-orbit fixture target-slot count failed")
        adjacency: Dict[int, set[int]] = {int(x): set() for x in node_set}
        for edge in internal.itertuples(index=False):
            adjacency[int(edge.source_node)].add(int(edge.target_node))
            adjacency[int(edge.target_node)].add(int(edge.source_node))
        seen = {next(iter(node_set))}
        frontier = list(seen)
        while frontier:
            node = frontier.pop()
            for neighbor in adjacency[node] - seen:
                seen.add(neighbor)
                frontier.append(neighbor)
        require(seen == node_set, "Exact-orbit fixture block is not connected")
        return {
            "fixed_slots": slots,
            "movable_payloads": payloads,
            "payload_definition": ["A", "Y"],
            "target_slot_position": int(target_positions[0]),
            "target_slot_node_index": int(slots[target_positions[0]]["node_index"]),
            "target_truth_reference": target_truth_reference,
            "candidate_response_replacement_required": True,
            "calibration_slot_count": int(len(slots) - 1),
            "payload_multiplicities": multiplicities,
            "distinct_permutations": distinct,
            "internal_edges": internal.to_dict(orient="records"),
            "boundary_edges": boundary.to_dict(orient="records"),
            "block_connected": True,
        }

    fixture6 = make_fixture(selected6)
    fixture8 = make_fixture(selected8)
    fixture_dup = make_fixture(selected6, duplicate_first_pair=True)
    require(fixture6["distinct_permutations"] == math.factorial(6), "Unexpected duplicate in unique size-6 payload")
    require(fixture8["distinct_permutations"] == math.factorial(8), "Unexpected duplicate in unique size-8 payload")
    require(fixture_dup["distinct_permutations"] == math.factorial(6) // 2, "Duplicate quotient fixture failed")
    return {
        "fixture_version": "1.3",
        "source_case_id": source_case,
        "source_target_dose": 0.90,
        "slot_payload_separation": "unit/node/geometry/role are fixed slot data; only (A,Y) is movable",
        "target_inclusion_rule": "every exact-orbit fixture contains exactly one realized localized target payload and calibration slots only otherwise",
        "candidate_inversion_rule": "replace target response by candidate y before each G3 orbit evaluation",
        "size6_unique": fixture6,
        "size8_unique": fixture8,
        "size6_one_duplicate_pair": fixture_dup,
        "scientific_scope": "fixtures only for later G2/G3 orbit, boundary, target-inclusion, candidate-replacement, and duplicate-quotient tests; no conformal method is implemented in Stage 3B",
    }

def dataframe_hash(frame: pd.DataFrame) -> str:
    text = frame.to_csv(index=False, lineterminator="\n", float_format="%.17g")
    return sha256_bytes(text.encode("utf-8"))


def software_inventory() -> Dict[str, Any]:
    versions: Dict[str, str] = {}
    for module in ("numpy", "pandas"):
        obj = __import__(module)
        versions[module] = str(getattr(obj, "__version__", "unknown"))
    return {
        "python": platform.python_version(),
        "packages": versions,
        "script_version": SCRIPT_VERSION,
        "created_utc": utc_now(),
    }


def build_method_feature_contract() -> Dict[str, Any]:
    return {
        "allowed_observed_nuisance_features": [
            "X_spatial_1", "X_spatial_2", "X_nonlinear_1", "X_nonlinear_2", "x_coord", "y_coord"
        ],
        "allowed_static_design_fields": ["role", "graph_degree_true", "support_scale"],
        "target_design_ratio_features": ["X_nonlinear_1"],
        "target_design_ratio_excluded_fields": ["x_coord", "y_coord", "role", "graph_degree_true", "support_scale"],
        "target_design_ratio_training_roles": {"observational": "nuisance_training", "target": "support_audit"},
        "target_design_ratio_evaluation_roles": {"observational": "calibration", "target": "test_target"},
        "forbidden_method_predictors": [
            "hidden_spatial_confounder", "substrate_proxy_hidden", "X_nonlinear_1_base",
            "target_design_class", "target_design_mode", "target_design_delta",
            "shared_spatial_residual",
            "residual_base_innovation", "residual_latent_gaussian", "measurement_base_innovation",
            "treatment_latent_interior_draw", "treatment_category_uniform",
            "oracle_g_mixed_at_A", "oracle_target_design_ratio", "log_oracle_target_design_ratio",
            "ue_proxy", "measurement_error_mode", "measurement_treatment_lambda", "measurement_substrate_lambda",
            "outcome_baseline",
            "conditional_mean_at_A", "Y_true_at_A", "Y_observed_at_A",
            "measurement_error_at_A", "target_draw_seed", "measurement_seed"
        ],
        "ST3_rule": "hidden_spatial_confounder is truth-only and must be omitted from fitted adjustment",
        "S9_rule": "ue_proxy may be used only by the measurement-error/quality layer, not the treatment propensity",
        "target_design_rule": "the N2/N3 design ratio uses X_nonlinear_1 only; raw coordinates are excluded to preserve an explicit common-support density-ratio model",
    }


def build_generator_contract(case_count: int) -> Dict[str, Any]:
    return {
        "stage": "3B",
        "script_version": SCRIPT_VERSION,
        "scope": "controlled scenario generator and realized localized-target draw engine",
        "primary_target": "random localized stochastic potential outcome Y_i(A_star)",
        "registered_target_doses": list(PRIMARY_DOSES),
        "primary_bandwidth": PRIMARY_BANDWIDTH,
        "validation_case_count": case_count,
        "supported_seed_phase": "pilot",
        "supported_replications": list(range(1, 21)),
        "default_validation_replication": 1,
        "production_experiments_run": False,
        "methods_implemented": False,
        "coverage_evaluated": False,
        "hard_dose_truth_role": "secondary generator diagnostic and sensitivity truth",
        "causal_negative_control": "ST3 violates C2 by an omitted spatial field; conformal validity must not be interpreted as causal identification repair",
        "temporal_negative_control": "ST2 preserves repeated unit-year dependence and forbids row-wise random splitting",
        "endpoint_rule": "genuine endpoint atoms are exact; interior-only reference refuses endpoint intervention",
        "measurement_rule": "S9 isolates random, treatment-correlated, and substrate-specific error components, also supplies one combined case, and stores latent and observed targets separately using one shared unit-level measurement innovation across the observed potential-outcome curve",
        "treatment_transport_rule": "q_h(A|a0)/g(A|X) is treatment/intervention transport and is never mislabeled as the N2/N3 target-design ratio",
        "target_design_transport_rule": "ordinary cases use oracle ratio one; S4_RHO060_DESIGN_SHIFT uses IID design object D=X_nonlinear_1 with N(0,1) observational and N(delta,1) target laws, giving r(D)=exp(delta*D-delta^2/2); the exported n0/n1 class-prior factor must be used",
        "support_rule": "mathematical positivity is separated from practical weight concentration; operational ESS/weight thresholds freeze after the later pilot",
        "information_separation_rule": "nuisance-training and support-audit nodes are independent graph components; calibration and target remain in one spatial inference graph",
        "reduction_rule": "S1/S2/S3/S7 use IID pretreatment covariates; S1/S3/S7 use IID residuals so benign/treatment-only reductions are not contaminated by latent spatial fields",
        "target_substream_rule": "each target dose/scope uses an order-independent sub-seed derived from the dedicated target-draw seed",
        "S10_rule": "the controlled 180 m case is a complete support-level rerun on a 13x13 anchored lattice; a separate 90-to-180 mapping fixture is also supplied for later paired aggregation sensitivity",
        "dense_precision_inverse_formed": False,
        "next_stage": "Stage3C implements M1-M5 against these frozen generator outputs; Stage3D implements exact G2/G3",
        "exact_orbit_fixture_rule": "fixed slot metadata is separated from movable (A,Y) payloads; the target payload uses a realized A_star and oracle truth response, Stage3D must replace Y by candidate y, and duplicate quotient fixtures contain actual duplicate calibration payloads",
    }
