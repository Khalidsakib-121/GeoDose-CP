from __future__ import annotations

import numpy as np
import pandas as pd

from geodose_stage3c.methods import build_interval

from .io import IntegrationError


def replay_calibration_audit(audit: pd.DataFrame, tolerance: float) -> tuple[pd.DataFrame, dict]:
    keys = ["query_id", "case_id", "scenario_id", "predictor", "nuisance_variant", "method"]
    rows = []
    max_q = 0.0
    status_mismatch = 0
    groups = 0
    for key, g in audit.groupby(keys, sort=True, dropna=False):
        g = g.sort_values("score", kind="mergesort")
        scores = g["score"].to_numpy(float)
        logw = g["log_weight"].to_numpy(float)
        target_logw = float(g["target_log_weight"].iloc[0])
        center = float(g["center"].iloc[0])
        alpha = float(g["alpha"].iloc[0])
        archived_q = float(g["archived_quantile"].iloc[0])
        archived_status = str(g["archived_status"].iloc[0])
        result = build_interval(center, scores, logw, target_logw, alpha)
        if np.isinf(archived_q) and np.isinf(result.quantile):
            qdiff = 0.0
        elif np.isnan(archived_q) and np.isnan(result.quantile):
            qdiff = 0.0
        else:
            qdiff = abs(float(result.quantile) - archived_q)
        max_q = max(max_q, qdiff)
        status_ok = result.interval_status == archived_status
        status_mismatch += int(not status_ok)
        rows.append(dict(zip(keys, key)) | {
            "calibration_n": len(g),
            "archived_quantile": archived_q,
            "replayed_quantile": result.quantile,
            "quantile_abs_error": qdiff,
            "archived_status": archived_status,
            "replayed_status": result.interval_status,
            "status_match": status_ok,
            "pass": bool(status_ok and qdiff <= tolerance),
        })
        groups += 1
    out = pd.DataFrame(rows)
    summary = {
        "groups_replayed": groups,
        "max_abs_quantile_error": max_q,
        "status_mismatch_count": status_mismatch,
        "all_groups_pass": bool(len(out) > 0 and out["pass"].all()),
        "principal_methods_replayed": sorted(out[out["method"].isin(["M1", "M2", "M5"])]["method"].unique().tolist()),
        "heuristics_replayed_but_not_promoted": sorted(out[out["method"].isin(["H1", "H4"])]["method"].unique().tolist()),
    }
    if not summary["all_groups_pass"]:
        raise IntegrationError("Frozen Stage3C baseline replay failed")
    return out, summary
