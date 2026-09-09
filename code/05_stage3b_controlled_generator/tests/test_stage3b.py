from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from geodose_stage3b.core import (  # noqa: E402
    GraphBasis,
    build_case_registry,
    build_exact_orbit_fixtures,
    draw_mixed_treatment,
    draw_truncated_normal,
    require,
    role_assignment,
    summarize_weights,
    truncated_normal_density,
)


def run_tests() -> None:
    cases = build_case_registry()
    require(len(cases) == 27, "Case registry count failed")
    require(set(f"S{i}" for i in range(1, 11)).issubset(set(cases.scenario_id)), "S1-S10 missing")
    require({"ST1", "ST2", "ST3"}.issubset(set(cases.scenario_id)), "Stress cases missing")
    require(cases.case_id.is_unique, "Case IDs not unique")
    require(set(cases.loc[cases.scenario_id.isin(["S1", "S2", "S3", "S7"]), "covariate_law"]) == {"iid"}, "Reduction covariate law failed")
    require(set(cases.loc[cases.scenario_id.isin(["S1", "S3", "S7"]), "residual_law"]) == {"iid_continuous"}, "Reduction residual law failed")
    require(set(cases.loc[cases.scenario_id == "ST1", "common_random_group"]) == {"ST1_COMMON"}, "ST1 common-random group failed")
    require(set(cases.loc[cases.scenario_id == "S9", "measurement_error"]) == {"random", "treatment_correlated", "substrate_bias", "combined"}, "S9 measurement component mapping failed")
    require(set(cases.loc[cases.target_design_mode != "identity", "case_id"]) == {"S4_RHO060_DESIGN_SHIFT"}, "Target-design validation case failed")
    require(float(cases.loc[cases.case_id == "S1_BASE", "primary_bandwidth"].iloc[0]) == 0.10, "S1 primary bandwidth drifted from Stage 3A")

    # Supported reference designs.
    basis25 = GraphBasis.create(25, 25, "primary")
    require(len(basis25.edges_queen) == 2206, "25x25 role-separated queen edge count failed")
    require(len(basis25.edges_rook) == 1150, "25x25 role-separated rook edge count failed")
    basis13 = GraphBasis.create(13, 13, "primary")
    require(len(basis13.edges_queen) == 526, "13x13 role-separated queen edge count failed")
    rng = np.random.default_rng(123)
    draw, eig = basis13.gmrf_draw(0.6, 0.35, rng)
    require(len(draw) == 169 and np.isfinite(draw).all(), "GMRF draw failed")
    require(eig > 0.0, "GMRF positive-definiteness failed")

    # No edge may cross the independent nuisance/support/final-inference components.
    roles = role_assignment(25, 25, "primary")
    components = np.where(roles == "nuisance_training", 0, np.where(roles == "support_audit", 1, 2))
    require((components[basis25.edges_queen.source_node.to_numpy()] == components[basis25.edges_queen.target_node.to_numpy()]).all(), "Information-component graph separation failed")
    cal_test = ((roles[basis25.edges_queen.source_node.to_numpy()] == "calibration") & (roles[basis25.edges_queen.target_node.to_numpy()] == "test_target")) | ((roles[basis25.edges_queen.source_node.to_numpy()] == "test_target") & (roles[basis25.edges_queen.target_node.to_numpy()] == "calibration"))
    require(cal_test.any(), "Calibration-target graph link missing")

    rng_a = np.random.default_rng(456)
    rng_b = np.random.default_rng(456)
    a = draw_truncated_normal(rng_a, 0.10, 0.10, 5000)
    b = draw_truncated_normal(rng_b, 0.10, 0.10, 5000)
    require(np.array_equal(a, b), "Target draw determinism failed")
    require(((a > 0.0) & (a < 1.0)).all(), "Target draw support failed")
    q = truncated_normal_density(a.copy(), 0.10, 0.10)
    require(np.isfinite(q).all() and (q > 0).all(), "Target density failed")

    # Latent interior draws are paired even when endpoint probabilities change.
    alpha = np.full(100, 2.5)
    beta = np.full(100, 3.5)
    mixed_probs = np.tile([0.2, 0.1, 0.7], (100, 1))
    interior_probs = np.tile([0.0, 0.0, 1.0], (100, 1))
    A_mixed, cat, latent_mixed, uniform_mixed = draw_mixed_treatment(
        np.random.default_rng(999), mixed_probs, alpha, beta, return_latent=True
    )
    A_interior, _, latent_interior, uniform_interior = draw_mixed_treatment(
        np.random.default_rng(999), interior_probs, alpha, beta, return_latent=True
    )
    require(np.array_equal(latent_mixed, latent_interior), "Paired latent interior treatment draw failed")
    require(np.array_equal(uniform_mixed, uniform_interior), "Paired treatment-category uniform failed")
    require(np.allclose(A_mixed[cat == 2], A_interior[cat == 2], atol=0, rtol=0), "Paired observed interior treatment failed")

    summary = summarize_weights(np.array([0.0, 0.0, 0.0]))
    require(abs(summary["oracle_ess"] - 3.0) < 1e-12, "ESS identity failed")
    require(abs(summary["max_normalized_weight"] - 1.0 / 3.0) < 1e-12, "Normalized weight failed")

    # Orbit fixture must separate fixed slots from movable payloads and really contain a duplicate pair.
    n = 625
    rows = np.repeat(np.arange(25), 25)
    cols = np.tile(np.arange(25), 25)
    units = pd.DataFrame({
        "case_id": "S4_RHO060",
        "grid_row": rows,
        "grid_col": cols,
        "node_index": np.arange(n),
        "unit_id": [f"U{i}" for i in range(n)],
        "A": np.linspace(0.001, 0.999, n),
        "Y_true_at_A": np.linspace(-1, 1, n),
        "Y_observed_at_A": np.linspace(-1, 1, n),
        "role": role_assignment(25, 25, "primary"),
    })
    target_units = units[units.role == "test_target"].copy()
    target_draws = pd.DataFrame({
        "case_id": "S4_RHO060",
        "unit_id": target_units.unit_id.to_numpy(),
        "target_dose": 0.90,
        "bandwidth": 0.10,
        "draw_status": "drawn",
        "A_star": np.linspace(0.80, 0.99, len(target_units)),
        "Y_true_at_A_star": np.linspace(1.0, 2.0, len(target_units)),
        "Y_observed_at_A_star": np.linspace(1.0, 2.0, len(target_units)),
    })
    fixtures = build_exact_orbit_fixtures(units, target_draws, basis25.edges_queen)
    require(fixtures["size6_unique"]["distinct_permutations"] == math.factorial(6), "Unique orbit fixture failed")
    require(fixtures["size6_one_duplicate_pair"]["distinct_permutations"] == math.factorial(6) // 2, "Duplicate orbit quotient failed")
    require(fixtures["size6_one_duplicate_pair"]["movable_payloads"][0] == fixtures["size6_one_duplicate_pair"]["movable_payloads"][1], "Duplicate movable payload missing")
    for fixture_name in ["size6_unique", "size8_unique", "size6_one_duplicate_pair"]:
        fixture = fixtures[fixture_name]
        roles_fixture = [slot["role"] for slot in fixture["fixed_slots"]]
        require(roles_fixture.count("test_target") == 1, f"Target-slot fixture failed: {fixture_name}")
        require(roles_fixture.count("calibration") == len(roles_fixture) - 1, f"Calibration fixture failed: {fixture_name}")
        require(fixture["block_connected"] is True, f"Connected fixture failed: {fixture_name}")
        require(fixture["candidate_response_replacement_required"] is True, f"Candidate replacement contract failed: {fixture_name}")
        target_pos = int(fixture["target_slot_position"])
        require(fixture["movable_payloads"][target_pos]["payload_origin"] == "realized_localized_target_truth_reference", f"Target payload origin failed: {fixture_name}")


if __name__ == "__main__":
    run_tests()
    print("STAGE3B GENERATOR UNIT TESTS PASSED")
