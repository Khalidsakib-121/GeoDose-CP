from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .candidate_inversion import (
    outer_components_from_cache,
    point_in_components,
    refine_boundaries,
)
from .io import require


@dataclass(frozen=True)
class AnalyticResult:
    candidate_y: float
    randomized_p: float
    conservative_p: float
    randomized_accept: bool
    conservative_accept: bool


class AnalyticEvaluator:
    def __init__(self, p_function: Callable[[float], float], alpha: float = 0.10):
        self.p_function = p_function
        self.alpha = float(alpha)
        self._cache: dict[str, AnalyticResult] = {}

    def evaluate(self, candidate_y: float) -> AnalyticResult:
        candidate_y = float(candidate_y)
        key = candidate_y.hex()
        if key not in self._cache:
            p_value = float(self.p_function(candidate_y))
            require(0.0 <= p_value <= 1.0, "D2_TOPOLOGY_VALIDATION_FAILED: invalid analytic p")
            accepted = bool(p_value > self.alpha)
            self._cache[key] = AnalyticResult(
                candidate_y=candidate_y,
                randomized_p=p_value,
                conservative_p=p_value,
                randomized_accept=accepted,
                conservative_accept=accepted,
            )
        return self._cache[key]

    def cached_results(self) -> list[AnalyticResult]:
        return sorted(self._cache.values(), key=lambda item: item.candidate_y)


def _component_signature(components: list[dict]) -> list[tuple[float, float, bool, bool]]:
    return [
        (
            float(item["lower"]),
            float(item["upper"]),
            bool(item["left_closed"]),
            bool(item["right_closed"]),
        )
        for item in components
    ]


def run_topology_validation(
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
    maximum_depth: int,
) -> list[dict]:
    lower, upper = -3.0, 3.0
    alpha = 0.10
    fixtures = [
        {
            "fixture_id": "continuous_open_interval",
            "p": lambda y: float(np.clip(alpha + 0.20 * (1.0 - y * y), 0.0, 1.0)),
            "landmarks": [-1.0, 1.0],
            "certified_boundaries": {-1.0: "rejected_transition", 1.0: "rejected_transition"},
            "certified_points": set(),
            "expected": [(-1.0, 1.0, False, False)],
            "truth_accept": lambda y: -1.0 < y < 1.0,
        },
        {
            "fixture_id": "disconnected_two_interval",
            "p": lambda y: 0.20 if (-2.0 < y < -1.0 or 1.0 < y < 2.0) else 0.05,
            "landmarks": [-2.0, -1.0, 1.0, 2.0],
            "certified_boundaries": {
                -2.0: "rejected_transition",
                -1.0: "rejected_transition",
                1.0: "rejected_transition",
                2.0: "rejected_transition",
            },
            "certified_points": set(),
            "expected": [(-2.0, -1.0, False, False), (1.0, 2.0, False, False)],
            "truth_accept": lambda y: (-2.0 < y < -1.0) or (1.0 < y < 2.0),
        },
        {
            "fixture_id": "empty_set",
            "p": lambda y: 0.05,
            "landmarks": [],
            "certified_boundaries": {},
            "certified_points": set(),
            "expected": [],
            "truth_accept": lambda y: False,
        },
        {
            "fixture_id": "certified_isolated_point",
            "p": lambda y: 0.20 if float(y).hex() == float(0.0).hex() else 0.05,
            "landmarks": [0.0],
            "certified_boundaries": {},
            "certified_points": {0.0},
            "expected": [(0.0, 0.0, True, True)],
            "truth_accept": lambda y: float(y).hex() == float(0.0).hex(),
        },
        {
            "fixture_id": "closed_step_discontinuity",
            "p": lambda y: 0.20 if y >= 0.0 else 0.05,
            "landmarks": [0.0],
            "certified_boundaries": {0.0: "accepted_right"},
            "certified_points": set(),
            "expected": [(0.0, 3.0, True, True)],
            "truth_accept": lambda y: y >= 0.0,
        },
    ]

    rows: list[dict] = []
    for fixture in fixtures:
        evaluator = AnalyticEvaluator(fixture["p"], alpha=alpha)
        base = np.linspace(lower, upper, 1025, dtype=float)
        points = np.asarray(sorted(set(base.tolist() + fixture["landmarks"] + [lower, upper])), dtype=float)
        for value in points:
            evaluator.evaluate(float(value))
        boundaries = refine_boundaries(
            evaluator,
            points,
            mode="conservative",
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
            maximum_depth=maximum_depth,
        )
        components = outer_components_from_cache(
            evaluator,
            lower,
            upper,
            mode="conservative",
            certified_exact_points=set(fixture["certified_points"]),
            certified_boundaries=dict(fixture["certified_boundaries"]),
        )
        observed = _component_signature(components)
        expected = fixture["expected"]
        exact_signature_match = observed == expected

        dense = np.linspace(lower, upper, 20001, dtype=float)
        false_negative = 0
        false_positive = 0
        for value in dense:
            direct = bool(fixture["truth_accept"](float(value)))
            represented = point_in_components(float(value), components)
            false_negative += int(direct and not represented)
            false_positive += int(represented and not direct)

        # For certified analytic fixtures the complete topology must match, not
        # merely provide an outer approximation.
        passed = bool(exact_signature_match and false_negative == 0 and false_positive == 0)
        require(passed, f"D2_TOPOLOGY_VALIDATION_FAILED: {fixture['fixture_id']}")
        rows.append(
            {
                "fixture_id": fixture["fixture_id"],
                "expected_component_count": len(expected),
                "observed_component_count": len(observed),
                "expected_signature": repr(expected),
                "observed_signature": repr(observed),
                "boundary_bracket_count": len(boundaries),
                "dense_reference_points": len(dense),
                "false_negative_count": false_negative,
                "false_positive_count": false_positive,
                "openness_closedness_match": exact_signature_match,
                "validation_pass": passed,
            }
        )
    return rows
