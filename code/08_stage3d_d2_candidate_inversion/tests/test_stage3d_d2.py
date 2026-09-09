from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from geodose_stage3d.candidate_inversion import (
    CandidateEvaluator,
    certified_score_partition,
    outer_components_from_cache,
    point_in_components,
    summarize_components,
)
from geodose_stage3d.d2_io import (
    build_d2_fixture,
    derive_seed,
    frozen_orbit_seed,
    registered_candidate_domain,
    stage3b_data,
    verify_d2_inputs,
)
from geodose_stage3d.exact_law import prepare_orbit, vectorized_orbit_evaluation
from geodose_stage3d.io import read_yaml
from geodose_stage3d.topology_validation import run_topology_validation


def evaluator_for(evaluation_id: str, tie_uniform: float = 0.37):
    data = stage3b_data(ROOT)
    fixture = build_d2_fixture(data, ROOT, evaluation_id)
    prepared = prepare_orbit(fixture, minimum_precision_eigenvalue_required=1e-10)
    evaluator = CandidateEvaluator(
        prepared,
        alpha=0.10,
        tie_uniform=tie_uniform,
        score_abs_tolerance=1e-12,
        score_rel_tolerance=1e-12,
        non_gaussian_min_abs_residual=1e-14,
    )
    return data, fixture, prepared, evaluator


def test_input_hashes_and_d1_inheritance() -> None:
    audit = verify_d2_inputs(ROOT)
    assert all(row["expected_sha256"] == row["observed_sha256"] for row in audit.values())


def test_frozen_tie_seed() -> None:
    base = frozen_orbit_seed(ROOT, "S4", 1)
    first = derive_seed(base, "D2_G3_TIE_UNIFORM", "fixture")
    second = derive_seed(base, "D2_G3_TIE_UNIFORM", "fixture")
    other = derive_seed(base, "D2_G3_TIE_UNIFORM", "other")
    assert first == second
    assert first != other


def test_domain_is_fixed_and_uses_no_outcomes() -> None:
    contract = read_yaml(ROOT / "configs" / "stage3d_d2_candidate_contract.yaml")
    domain = registered_candidate_domain(contract, "S4_RHO060")
    assert domain["domain_type"] == "fixed_registered_domain_truncated_reference"
    assert domain["lower"] == -8.0 and domain["upper"] == 8.0
    assert domain["target_truth_used"] is False
    assert domain["calibration_outcomes_used"] is False
    assert domain["support_audit_outcomes_used"] is False
    assert domain["nuisance_training_outcomes_used"] is False
    assert domain["unbounded_claim"] is False


def test_fast_probability_engine_matches_frozen_d1() -> None:
    for evaluation_id in ("G6_GAUSS_INTERIOR_TRUTH", "G6_NONGAUSSIAN_INTERIOR_TRUTH"):
        _, fixture, prepared, evaluator = evaluator_for(evaluation_id)
        for candidate in (fixture.target_payload_truth, fixture.target_payload_truth + 0.75):
            fast = evaluator.evaluate(candidate, keep_state=True)
            frozen = vectorized_orbit_evaluation(prepared, candidate, non_gaussian_min_abs_residual=1e-14)
            if fast.singular_conservative_inclusion:
                continue
            assert fast.state_probability is not None
            assert fast.permutations is not None
            assert np.array_equal(fast.permutations, frozen.permutations)
            assert np.max(np.abs(fast.state_probability - frozen.probability)) < 2e-12


def test_candidate_collision_uses_candidate_specific_quotient() -> None:
    _, _, prepared, evaluator = evaluator_for("G6_GAUSS_INTERIOR_TRUTH")
    target_a = float(prepared.A_payload[prepared.target_payload_index])
    collisions = [
        float(prepared.Y_payload_reference[index])
        for index, a_value in enumerate(prepared.A_payload.tolist())
        if index != prepared.target_payload_index and float(a_value) == target_a
    ]
    if collisions:
        result = evaluator.evaluate(collisions[0])
        assert result.quotient_changed_from_reference


def test_score_and_observed_assignment() -> None:
    _, fixture, _, evaluator = evaluator_for("G6_GAUSS_INTERIOR_TRUTH")
    result = evaluator.evaluate(fixture.target_payload_truth, keep_state=True)
    assert result.permutations is not None and result.state_score is not None
    assert np.array_equal(result.permutations[0], np.arange(len(fixture.slots)))
    assert abs(float(result.state_score[0]) - result.observed_score) < 1e-15


def test_certified_pvalue_mass_identity() -> None:
    _, fixture, _, evaluator = evaluator_for("G6_GAUSS_INTERIOR_TRUTH", tie_uniform=0.25)
    result = evaluator.evaluate(fixture.target_payload_truth)
    if not result.singular_conservative_inclusion:
        assert abs(
            result.conservative_p
            - (result.certified_strict_mass + result.numerically_ambiguous_mass + result.structural_tie_mass)
        ) < 2e-12
        assert abs(
            result.randomized_p
            - (
                result.certified_strict_mass
                + result.numerically_ambiguous_mass
                + 0.25 * result.structural_tie_mass
            )
        ) < 2e-12
        assert abs(
            result.certified_strict_mass
            + result.numerically_ambiguous_mass
            + result.structural_tie_mass
            + result.certified_less_mass
            - 1.0
        ) < 2e-12
    assert result.randomized_p <= result.conservative_p + 1e-14


def test_near_tie_is_not_randomized_downward() -> None:
    probability = np.asarray([0.4, 0.3, 0.3])
    scores = np.asarray([1.0, 1.0 + 5e-13, 0.5])
    structural = np.asarray([True, False, False])
    part = certified_score_partition(
        probability,
        scores,
        1.0,
        structural,
        absolute_tolerance=1e-12,
        relative_tolerance=0.0,
    )
    assert abs(float(part["tie_mass"]) - 0.4) < 1e-15
    assert abs(float(part["ambiguous_mass"]) - 0.3) < 1e-15
    u = 0.2
    safe_randomized = float(part["strict_mass"]) + float(part["ambiguous_mass"]) + u * float(part["tie_mass"])
    unsafe_old_rule = float(part["strict_mass"]) + u * (float(part["tie_mass"]) + float(part["ambiguous_mass"]))
    assert safe_randomized > unsafe_old_rule
    assert abs(safe_randomized - 0.38) < 1e-15


def test_one_uniform_held_across_candidates() -> None:
    _, fixture, _, evaluator = evaluator_for("G6_GAUSS_INTERIOR_TRUTH", tie_uniform=0.123456)
    values = [evaluator.evaluate(fixture.target_payload_truth + shift).tie_uniform for shift in (-1.0, 0.0, 1.0)]
    assert values == [0.123456, 0.123456, 0.123456]


def test_transformed_singularity_is_set_enlargement() -> None:
    _, _, prepared, evaluator = evaluator_for("G6_NONGAUSSIAN_INTERIOR_TRUTH")
    from geodose_stage3d.candidate_inversion import candidate_landmarks

    landmarks = candidate_landmarks(prepared, -8.0, 8.0)
    assert landmarks["transform_singularity"]
    result = evaluator.evaluate(landmarks["transform_singularity"][0])
    assert result.singular_conservative_inclusion
    assert result.randomized_p == 1.0 and result.conservative_p == 1.0
    assert result.state_probability is None


def test_outer_component_flags_are_honest() -> None:
    class R:
        def __init__(self, y: float, accept: bool):
            self.candidate_y = y
            self.randomized_accept = accept
            self.conservative_accept = accept

    class E:
        def __init__(self):
            self.rows = [R(-2.0, False), R(-1.0, True), R(1.0, True), R(2.0, False)]
        def cached_results(self):
            return self.rows

    components = outer_components_from_cache(E(), -2.0, 2.0, mode="conservative")
    assert len(components) == 1
    c = components[0]
    assert c["lower"] == -2.0 and c["left_closed"] is False
    assert c["upper"] == 2.0 and c["right_closed"] is False
    assert c["outer_numerical_set"] is True
    assert point_in_components(0.0, components)
    assert not point_in_components(-2.0, components)


def test_component_summary_has_no_unbounded_claim() -> None:
    components = [
        {
            "lower": -2.0,
            "upper": -1.0,
            "width": 1.0,
            "left_closed": False,
            "right_closed": False,
        },
        {
            "lower": 1.0,
            "upper": 3.0,
            "width": 2.0,
            "left_closed": False,
            "right_closed": True,
        },
    ]
    summary = summarize_components(components, -8.0, 8.0)
    assert summary["set_status"] == "disconnected_outer_numerical"
    assert summary["hull_inflation"] == 2.0
    assert summary["unbounded_claim"] is False
    assert summary["domain_truncated_set"] is True


def test_end_to_end_topology_validation() -> None:
    rows = run_topology_validation(
        absolute_tolerance=1e-6,
        relative_tolerance=1e-5,
        maximum_depth=40,
    )
    assert len(rows) == 5
    assert all(row["validation_pass"] for row in rows)
    assert {row["fixture_id"] for row in rows} == {
        "continuous_open_interval",
        "disconnected_two_interval",
        "empty_set",
        "certified_isolated_point",
        "closed_step_discontinuity",
    }


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
    print(f"STAGE3D D2 UNIT TESTS PASSED ({len(tests)} tests)")


if __name__ == "__main__":
    main()
