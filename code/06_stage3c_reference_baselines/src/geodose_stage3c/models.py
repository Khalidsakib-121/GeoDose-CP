from __future__ import annotations

import hashlib
import json
from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import betainc, betaln, expit, logit
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover
    XGBRegressor = None

from . import __version__

FEATURES = [
    "X_spatial_1",
    "X_spatial_2",
    "X_nonlinear_1",
    "X_nonlinear_2",
    "x_coord",
    "y_coord",
]


def _json_hash(payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _dataframe_content_hash(df: pd.DataFrame, columns: list[str], identity: str = "unit_id") -> str:
    missing = [c for c in [identity] + columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for fit hash: {missing}")
    if df[identity].duplicated().any():
        raise ValueError(f"Duplicate identity values in fit hash: {identity}")
    ordered = df[[identity] + columns].copy().sort_values(identity, kind="mergesort").reset_index(drop=True)
    raw = ordered.to_csv(index=False, lineterminator="\n", float_format="%.17g").encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class OutcomeRegressor:
    def __init__(self, kind: str, random_state: int):
        self.kind = kind
        self.random_state = int(random_state)
        if kind == "rf":
            self.hyperparameters = {
                "n_estimators": 240,
                "max_features": 0.85,
                "min_samples_leaf": 5,
                "bootstrap": True,
                "random_state": self.random_state,
                "n_jobs": 1,
            }
            self.model = RandomForestRegressor(**self.hyperparameters)
        elif kind == "xgb":
            if XGBRegressor is None:
                raise RuntimeError("xgboost is not installed")
            self.hyperparameters = {
                "n_estimators": 220,
                "max_depth": 3,
                "learning_rate": 0.04,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "reg_lambda": 1.0,
                "objective": "reg:squarederror",
                "random_state": self.random_state,
                "n_jobs": 1,
                "tree_method": "hist",
                "verbosity": 0,
            }
            self.model = XGBRegressor(**self.hyperparameters)
        else:
            raise ValueError(f"Unknown predictor kind {kind}")
        self.fit_id = ""
        self.training_data_hash = ""

    @staticmethod
    def design(df: pd.DataFrame, a: np.ndarray | None = None) -> np.ndarray:
        treatment = df["A"].to_numpy(float) if a is None else np.asarray(a, dtype=float)
        return np.column_stack([treatment, df[FEATURES].to_numpy(float)])

    def fit(self, df: pd.DataFrame, y_col: str = "Y_observed_at_A") -> "OutcomeRegressor":
        columns = ["A"] + FEATURES + [y_col]
        self.training_data_hash = _dataframe_content_hash(df, columns)
        self.model.fit(self.design(df), df[y_col].to_numpy(float))
        self.fit_id = _json_hash(
            {
                "package_version": __version__,
                "kind": self.kind,
                "features": ["A"] + FEATURES,
                "outcome": y_col,
                "hyperparameters": self.hyperparameters,
                "training_data_hash": self.training_data_hash,
            }
        )[:24]
        return self

    def predict_factual(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict(self.design(df))

    def predict_target(self, unit_df: pd.DataFrame, a_star: np.ndarray) -> np.ndarray:
        return self.model.predict(self.design(unit_df, a_star))


class CategoryModel:
    def __init__(self, mode: str, random_state: int):
        self.mode = mode
        self.random_state = int(random_state)
        self.scaler = StandardScaler()
        self.theta: np.ndarray | None = None
        self.constant: np.ndarray | None = None
        self.success = False
        self.hyperparameters = {
            "ridge_features": 0.10,
            "ridge_intercepts": 0.01,
            "optimizer": "L-BFGS-B",
            "maxiter": 1500,
            "ftol": 1e-11,
        }

    @staticmethod
    def _softmax(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        beta = theta.reshape(2, x.shape[1])
        logits = np.column_stack([x @ beta[0], x @ beta[1], np.zeros(len(x))])
        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        return exp_logits / exp_logits.sum(axis=1, keepdims=True)

    @classmethod
    def _nll(cls, theta: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
        probs = np.clip(cls._softmax(theta, x), 1e-15, 1.0)
        loss = -float(np.log(probs[np.arange(len(y)), y]).sum())
        beta = theta.reshape(2, x.shape[1])
        loss += 0.10 * float(np.sum(beta[:, 1:] ** 2)) + 0.01 * float(np.sum(beta[:, 0] ** 2))
        return loss

    def fit(self, df: pd.DataFrame) -> "CategoryModel":
        labels = df["treatment_category"].map({"atom_0": 0, "atom_1": 1, "interior": 2}).to_numpy(int)
        counts = np.bincount(labels, minlength=3).astype(float) + 0.5
        if self.mode == "misspecified":
            self.constant = counts / counts.sum()
            self.success = True
            return self
        z = self.scaler.fit_transform(df[FEATURES].to_numpy(float))
        x = np.column_stack([np.ones(len(z)), z])
        init = np.zeros((2, x.shape[1]), dtype=float)
        init[0, 0] = np.log(counts[0] / counts[2])
        init[1, 0] = np.log(counts[1] / counts[2])
        result = minimize(
            self._nll,
            init.ravel(),
            args=(x, labels),
            method="L-BFGS-B",
            bounds=[(-12.0, 12.0)] * init.size,
            options={"maxiter": 1500, "ftol": 1e-11},
        )
        self.theta = result.x
        self.success = bool(result.success and np.all(np.isfinite(result.x)))
        if not self.success:
            self.constant = counts / counts.sum()
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if self.constant is not None:
            return np.repeat(self.constant[None, :], len(df), axis=0)
        z = self.scaler.transform(df[FEATURES].to_numpy(float))
        x = np.column_stack([np.ones(len(z)), z])
        out = self._softmax(np.asarray(self.theta), x)
        out = np.clip(out, 1e-12, 1.0)
        return out / out.sum(axis=1, keepdims=True)


class BetaInteriorModel:
    def __init__(self, mode: str):
        self.mode = mode
        self.scaler = StandardScaler()
        self.theta: np.ndarray | None = None
        self.constant_mean: float | None = None
        self.constant_kappa: float | None = None
        self.success = False
        self.hyperparameters = {
            "optimizer": "L-BFGS-B",
            "maxiter": 1500,
            "ftol": 1e-10,
            "kappa_bounds": [2.0, 100.0],
        }

    @staticmethod
    def _nll(theta: np.ndarray, x: np.ndarray, a: np.ndarray) -> float:
        beta = theta[:-1]
        kappa = np.exp(theta[-1])
        mu = expit(x @ beta)
        aa = np.maximum(mu * kappa, 1e-4)
        bb = np.maximum((1.0 - mu) * kappa, 1e-4)
        loglik = (aa - 1.0) * np.log(a) + (bb - 1.0) * np.log1p(-a) - betaln(aa, bb)
        if not np.all(np.isfinite(loglik)):
            return 1e30
        return float(-np.sum(loglik) + 1e-5 * np.sum(beta[1:] ** 2))

    def fit(self, df: pd.DataFrame) -> "BetaInteriorModel":
        interior = df[(df["A"] > 0.0) & (df["A"] < 1.0)].copy()
        if len(interior) < 12:
            self.constant_mean = 0.5
            self.constant_kappa = 6.0
            self.success = False
            return self
        a = np.clip(interior["A"].to_numpy(float), 1e-8, 1.0 - 1e-8)
        if self.mode == "misspecified":
            mean = float(np.clip(a.mean(), 0.02, 0.98))
            variance = float(max(a.var(ddof=1), 1e-4))
            kappa = max(mean * (1.0 - mean) / variance - 1.0, 2.0)
            self.constant_mean = mean
            self.constant_kappa = min(kappa, 100.0)
            self.success = True
            return self
        z = self.scaler.fit_transform(interior[FEATURES].to_numpy(float))
        x = np.column_stack([np.ones(len(z)), z])
        init_beta, *_ = np.linalg.lstsq(x, logit(a), rcond=None)
        init = np.r_[init_beta, np.log(8.0)]
        result = minimize(
            self._nll,
            init,
            args=(x, a),
            method="L-BFGS-B",
            bounds=[(-12.0, 12.0)] * x.shape[1] + [(np.log(2.0), np.log(100.0))],
            options={"maxiter": 1500, "ftol": 1e-10},
        )
        self.theta = result.x
        self.success = bool(result.success and np.all(np.isfinite(result.x)))
        if not self.success:
            mean = float(np.clip(a.mean(), 0.02, 0.98))
            variance = float(max(a.var(ddof=1), 1e-4))
            self.constant_mean = mean
            self.constant_kappa = min(max(mean * (1.0 - mean) / variance - 1.0, 2.0), 100.0)
        return self

    def parameters(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        n = len(df)
        if self.constant_mean is not None:
            mu = np.full(n, self.constant_mean)
            kappa = np.full(n, float(self.constant_kappa))
            return mu * kappa, (1.0 - mu) * kappa
        z = self.scaler.transform(df[FEATURES].to_numpy(float))
        x = np.column_stack([np.ones(n), z])
        mu = expit(x @ np.asarray(self.theta)[:-1])
        kappa = np.exp(np.asarray(self.theta)[-1])
        return np.maximum(mu * kappa, 1e-4), np.maximum((1.0 - mu) * kappa, 1e-4)


class MixedPropensityModel:
    def __init__(self, mode: str, random_state: int):
        if mode not in {"estimated", "misspecified"}:
            raise ValueError(mode)
        self.mode = mode
        self.random_state = int(random_state)
        self.category = CategoryModel(mode, random_state)
        self.interior = BetaInteriorModel(mode)
        self.fit_id = ""
        self.training_data_hash = ""

    def fit(self, df: pd.DataFrame) -> "MixedPropensityModel":
        self.training_data_hash = _dataframe_content_hash(df, FEATURES + ["A", "treatment_category"])
        self.category.fit(df)
        self.interior.fit(df)
        self.fit_id = _json_hash(
            {
                "package_version": __version__,
                "kind": "mixed_propensity",
                "mode": self.mode,
                "random_state": self.random_state,
                "features": FEATURES,
                "category_hyperparameters": self.category.hyperparameters,
                "interior_hyperparameters": self.interior.hyperparameters,
                "training_data_hash": self.training_data_hash,
            }
        )[:24]
        return self

    def components(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        probs = self.category.predict_proba(df)
        alpha, beta = self.interior.parameters(df)
        return probs, alpha, beta

    def density(self, df: pd.DataFrame, a: np.ndarray | None = None) -> np.ndarray:
        aval = df["A"].to_numpy(float) if a is None else np.asarray(a, dtype=float)
        probs, alpha, beta = self.components(df)
        out = np.zeros(len(df), dtype=float)
        atom0 = aval == 0.0
        atom1 = aval == 1.0
        interior = (aval > 0.0) & (aval < 1.0)
        out[atom0] = probs[atom0, 0]
        out[atom1] = probs[atom1, 1]
        if interior.any():
            x = np.clip(aval[interior], 1e-12, 1.0 - 1e-12)
            aa = alpha[interior]
            bb = beta[interior]
            logpdf = (aa - 1.0) * np.log(x) + (bb - 1.0) * np.log1p(-x) - betaln(aa, bb)
            out[interior] = probs[interior, 2] * np.exp(logpdf)
        return np.maximum(out, 1e-300)

    def normalization_audit(self, df: pd.DataFrame) -> pd.DataFrame:
        probs, alpha, beta = self.components(df)
        return pd.DataFrame(
            {
                "unit_id": df["unit_id"].astype(str).to_numpy(),
                "pi_atom_0_hat": probs[:, 0],
                "pi_atom_1_hat": probs[:, 1],
                "pi_interior_hat": probs[:, 2],
                "beta_alpha_hat": alpha,
                "beta_beta_hat": beta,
                "interior_integral": betainc(alpha, beta, np.ones(len(df))) - betainc(alpha, beta, np.zeros(len(df))),
                "total_mixed_mass": probs[:, 0] + probs[:, 1] + probs[:, 2] * (betainc(alpha, beta, np.ones(len(df))) - betainc(alpha, beta, np.zeros(len(df)))),
            }
        )

    def diagnostics(self, audit_df: pd.DataFrame) -> Dict[str, float]:
        labels = audit_df["treatment_category"].map({"atom_0": 0, "atom_1": 1, "interior": 2}).to_numpy(int)
        probs = self.category.predict_proba(audit_df)
        category_loss = float(log_loss(labels, probs, labels=[0, 1, 2]))
        fitted = self.density(audit_df)
        oracle = audit_df["oracle_g_mixed_at_A"].to_numpy(float)
        return {
            "category_log_loss": category_loss,
            "mean_abs_log_density_error": float(np.mean(np.abs(np.log(fitted) - np.log(oracle)))),
            "rmse_log_density": float(np.sqrt(np.mean((np.log(fitted) - np.log(oracle)) ** 2))),
            "interior_fit_success": bool(self.interior.success),
            "category_fit_success": bool(self.category.success),
        }


def outcome_diagnostics(model: OutcomeRegressor, audit_df: pd.DataFrame) -> Dict[str, float]:
    pred = model.predict_factual(audit_df)
    observed = audit_df["Y_observed_at_A"].to_numpy(float)
    return {
        "rmse_observed_support_audit": float(mean_squared_error(observed, pred) ** 0.5),
        "mae_observed_support_audit": float(np.mean(np.abs(observed - pred))),
    }


class TargetDesignRatioModel:
    """Separate N2/N3 infrastructure; never inserted into Stage 3C baselines."""

    def __init__(self, random_state: int):
        self.random_state = int(random_state)
        self.hyperparameters = {"C": 1e4, "solver": "lbfgs", "max_iter": 1000, "random_state": self.random_state}
        self.model = LogisticRegression(**self.hyperparameters)
        self.fit_id = ""
        self.n0 = 0
        self.n1 = 0
        self.training_data_hash = ""

    def fit(self, observational: pd.DataFrame, target: pd.DataFrame) -> "TargetDesignRatioModel":
        obs = observational[["unit_id", "X_nonlinear_1"]].copy()
        obs["design_class"] = 0
        tar = target[["unit_id", "X_nonlinear_1"]].copy()
        tar["design_class"] = 1
        combined = pd.concat([obs, tar], ignore_index=True)
        self.training_data_hash = _dataframe_content_hash(combined, ["X_nonlinear_1", "design_class"])
        x = combined[["X_nonlinear_1"]].to_numpy(float)
        y = combined["design_class"].to_numpy(int)
        self.n0 = len(obs)
        self.n1 = len(tar)
        self.model.fit(x, y)
        self.fit_id = _json_hash(
            {
                "package_version": __version__,
                "kind": "target_design_ratio",
                "feature": "X_nonlinear_1",
                "hyperparameters": self.hyperparameters,
                "n0": self.n0,
                "n1": self.n1,
                "training_data_hash": self.training_data_hash,
            }
        )[:24]
        return self

    def ratio(self, df: pd.DataFrame) -> np.ndarray:
        p = np.clip(self.model.predict_proba(df[["X_nonlinear_1"]].to_numpy(float))[:, 1], 1e-12, 1.0 - 1e-12)
        return (p / (1.0 - p)) * (self.n0 / self.n1)
