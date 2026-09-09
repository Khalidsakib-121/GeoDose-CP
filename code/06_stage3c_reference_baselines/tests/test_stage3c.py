from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from geodose_stage3c.graph import (
    build_graph_cache,
    deterministic_graph_safe_set,
    geographic_distance_logweights,
    verify_independent_set,
)
from geodose_stage3c.io import Stage3CError, safe_prepare_output_dir
from geodose_stage3c.models import _dataframe_content_hash
from geodose_stage3c.runner import derived_seed, source_scenario_id
from geodose_stage3c.weights import intervention_density, weighted_conformal_quantile


def run() -> None:
    treatment = np.array([0.0, 0.2, 1.0])
    assert np.array_equal(intervention_density(treatment, 0.0, None, True), np.array([1.0, 0.0, 0.0]))
    assert np.array_equal(intervention_density(treatment, 1.0, None, True), np.array([0.0, 0.0, 1.0]))
    assert intervention_density(treatment, 0.0, None, False).sum() == 0.0

    result = weighted_conformal_quantile(np.array([1.0, 2.0, 3.0]), np.zeros(3), 0.0, 0.20)
    assert np.isinf(result.q)
    result = weighted_conformal_quantile(np.array([1.0, 2.0, 3.0]), np.zeros(3), np.log(0.01), 0.25)
    assert result.q == 3.0
    q1 = weighted_conformal_quantile(np.array([1.0, 2.0, 3.0]), np.log([1.0, 2.0, 3.0]), np.log(2.0), 0.2)
    q2 = weighted_conformal_quantile(np.array([1.0, 2.0, 3.0]), np.log([1.0, 2.0, 3.0]) + 100.0, np.log(2.0) + 100.0, 0.2)
    assert q1.q == q2.q and q1.status == q2.status

    units = pd.DataFrame({"node_index": range(6)})
    edges = pd.DataFrame({"source_node": [0, 1, 2, 3, 4], "target_node": [1, 2, 3, 4, 5]})
    cache = build_graph_cache(units, edges, np.array([5]))
    selected = deterministic_graph_safe_set(cache, 5, np.array([0, 1, 2, 3, 4]))
    assert verify_independent_set(cache, 5, selected)
    assert np.array_equal(selected, deterministic_graph_safe_set(cache, 5, np.array([0, 1, 2, 3, 4])))
    h1 = geographic_distance_logweights(
        np.array([1.0, 0.0]),
        np.column_stack([np.linspace(0.0, 0.8, 5), np.zeros(5)]),
        0.25,
    )
    expected_h1 = -(np.linspace(0.0, 0.8, 5) - 1.0) ** 2 / (2.0 * 0.25 ** 2)
    assert np.allclose(h1, expected_h1, atol=1e-15, rtol=0)
    assert float(np.max(h1)) < 0.0  # target itself has the separate log-weight 0

    frame = pd.DataFrame(
        {
            "unit_id": ["u2", "u1"],
            "A": [0.2, 0.3],
            "X_spatial_1": [1.0, 2.0],
        }
    )
    h_a = _dataframe_content_hash(frame, ["A", "X_spatial_1"])
    frame2 = frame.copy()
    frame2.loc[0, "A"] = 0.25
    h_b = _dataframe_content_hash(frame2, ["A", "X_spatial_1"])
    assert h_a != h_b
    assert _dataframe_content_hash(frame.sample(frac=1.0, random_state=1), ["A", "X_spatial_1"]) == h_a
    try:
        _dataframe_content_hash(pd.concat([frame, frame.iloc[[0]]], ignore_index=True), ["A", "X_spatial_1"])
        raise AssertionError("duplicate fit identities were not rejected")
    except ValueError:
        pass

    assert source_scenario_id("S4") == "S4"
    assert source_scenario_id("ST2") == "S4"
    assert derived_seed(123, "OUTCOME_RF") == derived_seed(123, "OUTCOME_RF")
    assert derived_seed(123, "OUTCOME_RF") != derived_seed(123, "OUTCOME_XGB")

    with tempfile.TemporaryDirectory() as temp:
        package = Path(temp)
        out = package / "outputs_stage3c_reference"
        out.mkdir()
        (out / "stale.txt").write_text("stale", encoding="utf-8")
        safe_prepare_output_dir(out, package, overwrite=True)
        assert out.exists() and not any(out.iterdir())
        try:
            safe_prepare_output_dir(package, package, overwrite=True)
            raise AssertionError("unsafe package-root deletion was not blocked")
        except Stage3CError:
            pass

    print("STAGE3C UNIT TESTS PASSED")


if __name__ == "__main__":
    run()
