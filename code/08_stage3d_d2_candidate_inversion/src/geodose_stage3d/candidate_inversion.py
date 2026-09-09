from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.special import logsumexp

from .exact_law import candidate_permutations
from .graph_precision import gaussian_local_log_factor
from .io import Stage3DError, require
from .residual_transform import affine_residual_matrix, inverse_power_transform
from .types import PreparedOrbit


@dataclass(frozen=True)
class CandidateResult:
    candidate_y: float
    distinct_states: int
    observed_score: float
    certified_strict_mass: float
    structural_tie_mass: float
    numerically_ambiguous_mass: float
    certified_less_mass: float
    conservative_p: float
    randomized_p: float
    tie_uniform: float
    conservative_accept: bool
    randomized_accept: bool
    randomized_exact_tie_claim: bool
    singular_conservative_inclusion: bool
    singular_zero_state_count: int
    probability_sum: float | None
    positive_probability_states: int | None
    zero_probability_states: int | None
    quotient_changed_from_reference: bool
    elapsed_seconds: float
    state_probability: np.ndarray | None
    state_score: np.ndarray | None
    state_category: np.ndarray | None
    permutations: np.ndarray | None

    @property
    def strict_mass(self) -> float:
        """Backward-readable alias: certified strict-greater mass."""
        return self.certified_strict_mass

    @property
    def tie_mass(self) -> float:
        """Backward-readable alias: certified structural-tie mass only."""
        return self.structural_tie_mass


@dataclass(frozen=True)
class BoundaryBracket:
    mode: str
    left_y: float
    right_y: float
    left_accept: bool
    right_accept: bool
    accepted_y: float
    rejected_y: float
    outer_endpoint: float
    outer_endpoint_closed: bool
    midpoint: float
    width: float
    depth: int
    representation: str


def certified_score_partition(
    probability: np.ndarray,
    scores: np.ndarray,
    observed_score: float,
    structural_tie: np.ndarray,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, np.ndarray | float]:
    """Partition score mass without randomizing numerically ambiguous states."""
    probability = np.asarray(probability, dtype=float)
    scores = np.asarray(scores, dtype=float)
    structural_tie = np.asarray(structural_tie, dtype=bool)
    require(probability.shape == scores.shape == structural_tie.shape, "D2_PVALUE_INVALID: partition shape")
    tolerance = float(absolute_tolerance) + float(relative_tolerance) * np.maximum(
        np.abs(scores), abs(float(observed_score))
    )
    require(
        np.all(np.abs(scores[structural_tie] - float(observed_score)) <= tolerance[structural_tie]),
        "D2_PVALUE_INVALID: structural tie score mismatch",
    )
    certified_greater = scores > float(observed_score) + tolerance
    certified_less = scores < float(observed_score) - tolerance
    ambiguous = ~(certified_greater | certified_less | structural_tie)
    require(
        np.all(
            certified_greater.astype(int)
            + certified_less.astype(int)
            + structural_tie.astype(int)
            + ambiguous.astype(int)
            == 1
        ),
        "D2_PVALUE_INVALID: score categories do not partition states",
    )
    return {
        "certified_greater": certified_greater,
        "certified_less": certified_less,
        "structural_tie": structural_tie,
        "ambiguous": ambiguous,
        "strict_mass": float(np.sum(probability[certified_greater])),
        "tie_mass": float(np.sum(probability[structural_tie])),
        "ambiguous_mass": float(np.sum(probability[ambiguous])),
        "less_mass": float(np.sum(probability[certified_less])),
    }


class CandidateEvaluator:
    """Evaluate the exact candidate-specific G2 law and G3 upper-tail masses.

    Numerical score comparisons are partitioned into four disjoint categories:

    * certified strict greater;
    * certified structural tie (identical numerical movable payload at the
      fixed target slot);
    * numerically ambiguous near-tie;
    * certified strict less.

    Ambiguous mass is included in full in both p-values.  This prevents a
    tolerance-based near-tie from being randomized downward.  The randomized
    p-value is exact only when the ambiguous mass is zero.
    """

    def __init__(
        self,
        prepared: PreparedOrbit,
        *,
        alpha: float,
        tie_uniform: float,
        score_abs_tolerance: float,
        score_rel_tolerance: float,
        non_gaussian_min_abs_residual: float,
    ) -> None:
        require(0.0 < alpha < 1.0, "D2_CONTRACT_MISMATCH: alpha")
        require(0.0 <= tie_uniform <= 1.0, "D2_CONTRACT_MISMATCH: tie uniform")
        self.prepared = prepared
        self.alpha = float(alpha)
        self.tie_uniform = float(tie_uniform)
        self.score_abs_tolerance = float(score_abs_tolerance)
        self.score_rel_tolerance = float(score_rel_tolerance)
        self.non_gaussian_min_abs_residual = float(non_gaussian_min_abs_residual)
        self._cache: dict[str, CandidateResult] = {}
        self._collision_values = {
            float(prepared.Y_payload_reference[index])
            for index, a_value in enumerate(prepared.A_payload.tolist())
            if index != prepared.target_payload_index
            and float(a_value) == float(prepared.A_payload[prepared.target_payload_index])
        }
        reference_candidate = float(prepared.fixture.candidate_y)
        if reference_candidate in self._collision_values:
            reference_candidate = reference_candidate + 0.123456789
        self._default_permutations = candidate_permutations(prepared, reference_candidate)
        self.reference_state_count = int(len(self._default_permutations))

    @staticmethod
    def _key(candidate_y: float) -> str:
        value = float(candidate_y)
        require(np.isfinite(value), "D2_SCORE_INVALID: candidate nonfinite")
        return value.hex()

    def evaluate(self, candidate_y: float, *, keep_state: bool = False) -> CandidateResult:
        key = self._key(candidate_y)
        if key in self._cache and (not keep_state or self._cache[key].state_score is not None):
            return self._cache[key]

        started = time.perf_counter()
        candidate_y = float(candidate_y)
        permutations = (
            candidate_permutations(self.prepared, candidate_y)
            if candidate_y in self._collision_values
            else self._default_permutations
        )
        identity = np.arange(permutations.shape[1], dtype=permutations.dtype)
        require(np.array_equal(permutations[0], identity), "D2_OBSERVED_ASSIGNMENT_INVALID")

        fixture = self.prepared.fixture
        y_payload = self.prepared.Y_payload_reference.copy()
        y_payload[self.prepared.target_payload_index] = candidate_y
        residual, _ = affine_residual_matrix(
            permutations,
            y_payload,
            self.prepared.outcome_mean,
            self.prepared.outcome_scale,
        )
        target_scores = np.abs(residual[:, fixture.target_slot_position])
        require(np.all(np.isfinite(target_scores)), "D2_SCORE_INVALID")
        observed_score = float(target_scores[0])

        singular = False
        singular_count = 0
        if fixture.residual_law == "transformed_gmrf_power_1_5":
            zero_mask = np.abs(residual) < self.non_gaussian_min_abs_residual
            singular_count = int(np.count_nonzero(np.any(zero_mask, axis=1)))
            singular = singular_count > 0

        if singular:
            # A finite set of transformed-density singular landmarks is handled
            # by the preregistered set-enlarging rule.  No pointwise orbit-pmf
            # claim is made at these candidates.
            result = CandidateResult(
                candidate_y=candidate_y,
                distinct_states=int(len(permutations)),
                observed_score=observed_score,
                certified_strict_mass=math.nan,
                structural_tie_mass=math.nan,
                numerically_ambiguous_mass=math.nan,
                certified_less_mass=math.nan,
                conservative_p=1.0,
                randomized_p=1.0,
                tie_uniform=self.tie_uniform,
                conservative_accept=True,
                randomized_accept=True,
                randomized_exact_tie_claim=False,
                singular_conservative_inclusion=True,
                singular_zero_state_count=singular_count,
                probability_sum=None,
                positive_probability_states=None,
                zero_probability_states=None,
                quotient_changed_from_reference=int(len(permutations)) != self.reference_state_count,
                elapsed_seconds=time.perf_counter() - started,
                state_probability=None,
                state_score=target_scores if keep_state else None,
                state_category=None,
                permutations=permutations if keep_state else None,
            )
            self._cache[key] = result
            return result

        rows = np.arange(permutations.shape[1])[None, :]
        treatment_matrix = self.prepared.observational_g[rows, permutations].copy()
        treatment_matrix[:, fixture.target_slot_position] = self.prepared.intervention_q[
            permutations[:, fixture.target_slot_position]
        ]
        if np.any(treatment_matrix < 0.0) or np.any(~np.isfinite(treatment_matrix)):
            raise Stage3DError("D2_ORBIT_EVALUATION_FAILED: invalid treatment factor")
        log_treatment_matrix = np.full(treatment_matrix.shape, -np.inf, dtype=float)
        positive_treatment = treatment_matrix > 0.0
        log_treatment_matrix[positive_treatment] = np.log(treatment_matrix[positive_treatment])
        log_treatment = np.sum(log_treatment_matrix, axis=1)

        _, log_j_matrix = affine_residual_matrix(
            permutations,
            y_payload,
            self.prepared.outcome_mean,
            self.prepared.outcome_scale,
        )
        log_outcome_jacobian = np.sum(log_j_matrix, axis=1)

        if fixture.residual_law == "gaussian_gmrf":
            log_residual = gaussian_local_log_factor(
                residual,
                self.prepared.Q_BB,
                self.prepared.Q_BD,
                self.prepared.boundary_residual,
            )
        elif fixture.residual_law == "transformed_gmrf_power_1_5":
            latent, log_abs_dz_dr = inverse_power_transform(
                residual,
                fixture.transform_power,
                minimum_abs_residual=self.non_gaussian_min_abs_residual,
            )
            log_residual = gaussian_local_log_factor(
                latent,
                self.prepared.Q_BB,
                self.prepared.Q_BD,
                self.prepared.boundary_latent_z,
            ) + np.sum(log_abs_dz_dr, axis=1)
        else:
            raise Stage3DError(f"D2_ORBIT_EVALUATION_FAILED: {fixture.residual_law}")

        log_total = log_treatment + log_outcome_jacobian + log_residual
        finite = np.isfinite(log_total)
        require(np.any(finite), "D2_ORBIT_EVALUATION_FAILED: all weights zero")
        probability = np.zeros(len(log_total), dtype=float)
        probability[finite] = np.exp(log_total[finite] - float(logsumexp(log_total[finite])))
        require(abs(float(probability.sum()) - 1.0) <= 5e-12, "D2_PVALUE_INVALID: probability sum")

        scores = target_scores
        target_slot_payload_indices = permutations[:, fixture.target_slot_position]
        observed_payload_index = int(permutations[0, fixture.target_slot_position])
        observed_a = float(self.prepared.A_payload[observed_payload_index])
        observed_y = float(y_payload[observed_payload_index])
        assigned_a = self.prepared.A_payload[target_slot_payload_indices]
        assigned_y = y_payload[target_slot_payload_indices]
        structural_tie = (assigned_a == observed_a) & (assigned_y == observed_y)
        partition = certified_score_partition(
            probability,
            scores,
            observed_score,
            structural_tie,
            absolute_tolerance=self.score_abs_tolerance,
            relative_tolerance=self.score_rel_tolerance,
        )
        certified_greater = partition["certified_greater"]
        certified_less = partition["certified_less"]
        ambiguous = partition["ambiguous"]
        strict_mass = float(partition["strict_mass"])
        tie_mass = float(partition["tie_mass"])
        ambiguous_mass = float(partition["ambiguous_mass"])
        less_mass = float(partition["less_mass"])
        mass_total = strict_mass + tie_mass + ambiguous_mass + less_mass
        require(abs(mass_total - 1.0) <= 5e-12, "D2_PVALUE_INVALID: category mass sum")

        # The conservative p-value includes all non-certified-less states.
        conservative = strict_mass + ambiguous_mass + tie_mass
        # Only certified structural ties are randomized.  Numerically ambiguous
        # mass is included in full, making this safe against near-tie ambiguity.
        randomized = strict_mass + ambiguous_mass + self.tie_uniform * tie_mass
        require(randomized <= conservative + 1e-13, "D2_PVALUE_INVALID: randomized exceeds conservative")

        categories = np.full(len(scores), "ambiguous_included", dtype=object)
        categories[certified_greater] = "certified_strict_greater"
        categories[certified_less] = "certified_strict_less"
        categories[structural_tie] = "certified_structural_tie"

        result = CandidateResult(
            candidate_y=candidate_y,
            distinct_states=int(len(permutations)),
            observed_score=observed_score,
            certified_strict_mass=strict_mass,
            structural_tie_mass=tie_mass,
            numerically_ambiguous_mass=ambiguous_mass,
            certified_less_mass=less_mass,
            conservative_p=min(1.0, max(0.0, conservative)),
            randomized_p=min(1.0, max(0.0, randomized)),
            tie_uniform=self.tie_uniform,
            conservative_accept=bool(conservative > self.alpha),
            randomized_accept=bool(randomized > self.alpha),
            randomized_exact_tie_claim=bool(ambiguous_mass == 0.0),
            singular_conservative_inclusion=False,
            singular_zero_state_count=0,
            probability_sum=float(probability.sum()),
            positive_probability_states=int(np.count_nonzero(probability > 0.0)),
            zero_probability_states=int(np.count_nonzero(probability == 0.0)),
            quotient_changed_from_reference=int(len(permutations)) != self.reference_state_count,
            elapsed_seconds=time.perf_counter() - started,
            state_probability=probability if keep_state else None,
            state_score=scores if keep_state else None,
            state_category=categories if keep_state else None,
            permutations=permutations if keep_state else None,
        )
        self._cache[key] = result
        return result

    def cached_results(self) -> list[CandidateResult]:
        return sorted(self._cache.values(), key=lambda value: value.candidate_y)


def candidate_landmarks(prepared: PreparedOrbit, lower: float, upper: float) -> dict[str, list[float]]:
    fixture = prepared.fixture
    target_index = prepared.target_payload_index
    target_a = float(prepared.A_payload[target_index])
    target_slot = fixture.target_slot_position
    target_mean = float(prepared.outcome_mean[target_slot, target_index])
    target_scale = float(prepared.outcome_scale[target_slot, target_index])

    score_ties: list[float] = []
    for payload_index in range(len(prepared.A_payload)):
        if payload_index == target_index:
            continue
        fixed_score = abs(
            (
                float(prepared.Y_payload_reference[payload_index])
                - float(prepared.outcome_mean[target_slot, payload_index])
            )
            / float(prepared.outcome_scale[target_slot, payload_index])
        )
        score_ties.extend(
            [target_mean - target_scale * fixed_score, target_mean + target_scale * fixed_score]
        )

    collisions = [
        float(prepared.Y_payload_reference[index])
        for index, value in enumerate(prepared.A_payload.tolist())
        if index != target_index and float(value) == target_a
    ]

    singular: list[float] = []
    if fixture.residual_law == "transformed_gmrf_power_1_5":
        for slot in range(len(fixture.slots)):
            singular.append(float(prepared.outcome_mean[slot, target_index]))

    def unique_inside(values: Iterable[float]) -> list[float]:
        return sorted({float(v) for v in values if np.isfinite(v) and lower <= float(v) <= upper})

    return {
        "score_tie": unique_inside(score_ties),
        "payload_collision": unique_inside(collisions),
        "transform_singularity": unique_inside(singular),
    }


def initial_candidate_points(
    lower: float,
    upper: float,
    landmarks: dict[str, list[float]],
    *,
    coarse_points: int,
    staggered_points: int,
    certification_points: int,
    flank_relative: float,
    flank_minimum: float,
) -> tuple[np.ndarray, dict[str, set[float]]]:
    require(coarse_points >= 3, "D2_CONTRACT_MISMATCH: coarse grid")
    require(staggered_points == coarse_points - 1, "D2_CONTRACT_MISMATCH: staggered grid")
    require(certification_points >= coarse_points, "D2_CONTRACT_MISMATCH: certification grid")

    coarse = np.linspace(lower, upper, int(coarse_points), dtype=float)
    staggered = 0.5 * (coarse[:-1] + coarse[1:])
    certification = np.linspace(lower, upper, int(certification_points), dtype=float)
    source_map: dict[str, set[float]] = {
        "coarse": set(float(x) for x in coarse),
        "staggered": set(float(x) for x in staggered),
        "certification_grid": set(float(x) for x in certification),
        "score_tie_landmark": set(landmarks["score_tie"]),
        "payload_collision_landmark": set(landmarks["payload_collision"]),
        "transform_singularity_landmark": set(landmarks["transform_singularity"]),
        "landmark_flank": set(),
    }
    width = upper - lower
    delta = max(float(flank_minimum), float(flank_relative) * max(1.0, width))
    for category in ("score_tie", "payload_collision", "transform_singularity"):
        for value in landmarks[category]:
            if value - delta > lower:
                source_map["landmark_flank"].add(float(value - delta))
            if value + delta < upper:
                source_map["landmark_flank"].add(float(value + delta))
    points = sorted(set().union(*source_map.values(), {float(lower), float(upper)}))
    return np.asarray(points, dtype=float), source_map


def acceptance(result: CandidateResult, mode: str) -> bool:
    if mode == "randomized":
        return bool(result.randomized_accept)
    if mode == "conservative":
        return bool(result.conservative_accept)
    raise Stage3DError(f"Unknown tie mode: {mode}")


def refine_boundaries(
    evaluator,
    initial_points: np.ndarray,
    *,
    mode: str,
    absolute_tolerance: float,
    relative_tolerance: float,
    maximum_depth: int,
) -> list[BoundaryBracket]:
    points = np.asarray(sorted(set(float(v) for v in initial_points)), dtype=float)
    results = [evaluator.evaluate(float(v)) for v in points]
    brackets: list[BoundaryBracket] = []
    for left, right, left_result, right_result in zip(points[:-1], points[1:], results[:-1], results[1:]):
        left_accept = acceptance(left_result, mode)
        right_accept = acceptance(right_result, mode)
        if left_accept == right_accept:
            continue
        a, b = float(left), float(right)
        aa, bb = bool(left_accept), bool(right_accept)
        depth = 0
        while depth < maximum_depth:
            midpoint = 0.5 * (a + b)
            tolerance = max(float(absolute_tolerance), float(relative_tolerance) * max(1.0, abs(midpoint)))
            if b - a <= tolerance:
                break
            mid_accept = acceptance(evaluator.evaluate(midpoint), mode)
            if mid_accept == aa:
                a = midpoint
                aa = mid_accept
            else:
                b = midpoint
                bb = mid_accept
            depth += 1
        midpoint = 0.5 * (a + b)
        tolerance = max(float(absolute_tolerance), float(relative_tolerance) * max(1.0, abs(midpoint)))
        require(b - a <= tolerance, "D2_BOUNDARY_UNRESOLVED")
        accepted_y = a if aa else b
        rejected_y = b if aa else a
        brackets.append(
            BoundaryBracket(
                mode=mode,
                left_y=a,
                right_y=b,
                left_accept=aa,
                right_accept=bb,
                accepted_y=float(accepted_y),
                rejected_y=float(rejected_y),
                outer_endpoint=float(rejected_y),
                outer_endpoint_closed=False,
                midpoint=float(midpoint),
                width=float(b - a),
                depth=int(depth),
                representation="coverage_preserving_outer_boundary_bracket",
            )
        )
    return brackets


def outer_components_from_cache(
    evaluator,
    lower: float,
    upper: float,
    *,
    mode: str,
    certified_exact_points: set[float] | None = None,
    certified_boundaries: dict[float, str] | None = None,
) -> list[dict[str, float | int | bool | str]]:
    """Build a finite-domain, coverage-preserving outer numerical set.

    A numerical transition is bounded by one accepted and one rejected point.
    The returned endpoint uses the rejected-side point and excludes that point
    (open endpoint), thereby including the entire unresolved bracket without
    including a point already certified rejected.
    """
    certified_exact_points = certified_exact_points or set()
    certified_boundaries = certified_boundaries or {}
    values = [result for result in evaluator.cached_results() if lower <= result.candidate_y <= upper]
    require(
        values
        and float(values[0].candidate_y).hex() == float(lower).hex()
        and float(values[-1].candidate_y).hex() == float(upper).hex(),
        "D2_CONTRACT_MISMATCH: domain endpoints not evaluated",
    )
    flags = [acceptance(result, mode) for result in values]
    components: list[dict[str, float | int | bool | str]] = []
    index = 0
    component_id = 0
    while index < len(values):
        if not flags[index]:
            index += 1
            continue
        start = index
        while index + 1 < len(values) and flags[index + 1]:
            index += 1
        end = index
        component_id += 1

        accepted_start = float(values[start].candidate_y)
        accepted_end = float(values[end].candidate_y)
        exact_isolated = (
            start == end
            and accepted_start in certified_exact_points
            and (start == 0 or not flags[start - 1])
            and (end == len(values) - 1 or not flags[end + 1])
        )
        if exact_isolated:
            lower_y = upper_y = accepted_start
            left_closed = right_closed = True
            left_source = right_source = "certified_exact_included_landmark"
            left_direct_accept = right_direct_accept = True
            representation = "certified_exact_point_component"
        else:
            if start == 0:
                lower_y = float(lower)
                left_closed = True
                left_source = "accepted_registered_domain_boundary"
                left_direct_accept = True
            elif accepted_start in certified_boundaries and certified_boundaries[accepted_start] == "accepted_right":
                lower_y = accepted_start
                left_closed = True
                left_source = "certified_exact_closed_left_boundary"
                left_direct_accept = True
            elif float(values[start - 1].candidate_y) in certified_boundaries and certified_boundaries[float(values[start - 1].candidate_y)] == "rejected_transition":
                lower_y = float(values[start - 1].candidate_y)
                left_closed = False
                left_source = "certified_exact_open_left_boundary"
                left_direct_accept = False
            else:
                lower_y = float(values[start - 1].candidate_y)
                left_closed = False
                left_source = "rejected_side_outer_boundary"
                left_direct_accept = False
            if end == len(values) - 1:
                upper_y = float(upper)
                right_closed = True
                right_source = "accepted_registered_domain_boundary"
                right_direct_accept = True
            elif accepted_end in certified_boundaries and certified_boundaries[accepted_end] == "accepted_left":
                upper_y = accepted_end
                right_closed = True
                right_source = "certified_exact_closed_right_boundary"
                right_direct_accept = True
            elif float(values[end + 1].candidate_y) in certified_boundaries and certified_boundaries[float(values[end + 1].candidate_y)] == "rejected_transition":
                upper_y = float(values[end + 1].candidate_y)
                right_closed = False
                right_source = "certified_exact_open_right_boundary"
                right_direct_accept = False
            else:
                upper_y = float(values[end + 1].candidate_y)
                right_closed = False
                right_source = "rejected_side_outer_boundary"
                right_direct_accept = False
            representation = (
                "certified_exact_topology_component"
                if left_source.startswith("certified_exact") or right_source.startswith("certified_exact")
                else "coverage_preserving_outer_numerical_component"
            )

        components.append(
            {
                "tie_mode": mode,
                "component_id": component_id,
                "lower": lower_y,
                "upper": upper_y,
                "left_closed": left_closed,
                "right_closed": right_closed,
                "width": max(0.0, upper_y - lower_y),
                "isolated_point": bool(lower_y == upper_y and left_closed and right_closed),
                "representation": representation,
                "left_endpoint_source": left_source,
                "right_endpoint_source": right_source,
                "left_endpoint_direct_accept": left_direct_accept,
                "right_endpoint_direct_accept": right_direct_accept,
                "exact_g3_boundary_claim": bool(
                    exact_isolated
                    or left_source.startswith("certified_exact")
                    or right_source.startswith("certified_exact")
                ),
                "outer_numerical_set": bool(
                    not exact_isolated
                    and not (left_source.startswith("certified_exact") and right_source.startswith("certified_exact"))
                ),
            }
        )
        index += 1
    return components


def point_in_components(
    value: float,
    components: list[dict[str, float | int | bool | str]],
    tolerance: float = 0.0,
) -> bool:
    value = float(value)
    for component in components:
        lower = float(component["lower"])
        upper = float(component["upper"])
        left_closed = bool(component.get("left_closed", True))
        right_closed = bool(component.get("right_closed", True))
        left_ok = value > lower - tolerance if not left_closed else value >= lower - tolerance
        right_ok = value < upper + tolerance if not right_closed else value <= upper + tolerance
        if left_ok and right_ok:
            return True
    return False


def summarize_components(
    components: list[dict[str, float | int | bool | str]],
    lower: float,
    upper: float,
) -> dict[str, float | int | str | bool | None]:
    if not components:
        return {
            "set_status": "empty_within_registered_domain",
            "component_count": 0,
            "raw_total_width": 0.0,
            "hull_lower": None,
            "hull_upper": None,
            "hull_width": 0.0,
            "hull_inflation": 0.0,
            "full_registered_domain": False,
            "left_domain_truncated": False,
            "right_domain_truncated": False,
            "domain_truncated_set": True,
            "unbounded_claim": False,
        }
    widths = [float(component["width"]) for component in components]
    raw_width = float(sum(widths))
    hull_lower = float(min(float(component["lower"]) for component in components))
    hull_upper = float(max(float(component["upper"]) for component in components))
    hull_width = float(hull_upper - hull_lower)
    hull_inflation = max(0.0, hull_width - raw_width)
    left_truncated = abs(hull_lower - lower) <= 1e-12
    right_truncated = abs(hull_upper - upper) <= 1e-12
    full_domain = len(components) == 1 and left_truncated and right_truncated
    if full_domain:
        status = "full_registered_domain"
    elif len(components) == 1:
        status = "connected_domain_truncated" if (left_truncated or right_truncated) else "connected_outer_numerical"
    else:
        status = "disconnected_domain_truncated" if (left_truncated or right_truncated) else "disconnected_outer_numerical"
    return {
        "set_status": status,
        "component_count": int(len(components)),
        "raw_total_width": raw_width,
        "hull_lower": hull_lower,
        "hull_upper": hull_upper,
        "hull_width": hull_width,
        "hull_inflation": hull_inflation,
        "full_registered_domain": bool(full_domain),
        "left_domain_truncated": bool(left_truncated),
        "right_domain_truncated": bool(right_truncated),
        "domain_truncated_set": True,
        "unbounded_claim": False,
    }


def intersect_component_sets(
    left_components: list[dict[str, float | int | bool | str]],
    right_components: list[dict[str, float | int | bool | str]],
    *,
    tie_mode: str,
) -> list[dict[str, float | int | bool | str]]:
    """Intersect finite component unions while preserving endpoint topology."""
    intersections: list[dict[str, float | int | bool | str]] = []
    for left in left_components:
        for right in right_components:
            left_lower = float(left["lower"])
            right_lower = float(right["lower"])
            left_upper = float(left["upper"])
            right_upper = float(right["upper"])
            lower = max(left_lower, right_lower)
            upper = min(left_upper, right_upper)
            if lower > upper:
                continue
            if lower == left_lower and lower == right_lower:
                lower_closed = bool(left["left_closed"]) and bool(right["left_closed"])
            elif lower == left_lower:
                lower_closed = bool(left["left_closed"])
            else:
                lower_closed = bool(right["left_closed"])
            if upper == left_upper and upper == right_upper:
                upper_closed = bool(left["right_closed"]) and bool(right["right_closed"])
            elif upper == left_upper:
                upper_closed = bool(left["right_closed"])
            else:
                upper_closed = bool(right["right_closed"])
            if lower == upper and not (lower_closed and upper_closed):
                continue
            intersections.append(
                {
                    "tie_mode": tie_mode,
                    "component_id": 0,
                    "lower": lower,
                    "upper": upper,
                    "left_closed": lower_closed,
                    "right_closed": upper_closed,
                    "width": max(0.0, upper - lower),
                    "isolated_point": bool(lower == upper and lower_closed and upper_closed),
                    "representation": "randomized_outer_intersect_conservative_outer",
                    "left_endpoint_source": "intersection",
                    "right_endpoint_source": "intersection",
                    "left_endpoint_direct_accept": bool(left.get("left_endpoint_direct_accept", False)),
                    "right_endpoint_direct_accept": bool(left.get("right_endpoint_direct_accept", False)),
                    "exact_g3_boundary_claim": False,
                    "outer_numerical_set": True,
                }
            )
    intersections.sort(key=lambda item: (float(item["lower"]), float(item["upper"])))
    # Merge overlaps or touching intervals when the shared point is included.
    merged: list[dict[str, float | int | bool | str]] = []
    for component in intersections:
        if not merged:
            merged.append(component)
            continue
        previous = merged[-1]
        overlap = float(component["lower"]) < float(previous["upper"])
        touching_included = (
            float(component["lower"]) == float(previous["upper"])
            and (bool(component["left_closed"]) or bool(previous["right_closed"]))
        )
        if overlap or touching_included:
            if float(component["upper"]) > float(previous["upper"]):
                previous["upper"] = component["upper"]
                previous["right_closed"] = component["right_closed"]
                previous["right_endpoint_source"] = component["right_endpoint_source"]
            elif float(component["upper"]) == float(previous["upper"]):
                previous["right_closed"] = bool(previous["right_closed"]) or bool(component["right_closed"])
            previous["width"] = max(0.0, float(previous["upper"]) - float(previous["lower"]))
            previous["isolated_point"] = bool(
                float(previous["lower"]) == float(previous["upper"])
                and bool(previous["left_closed"])
                and bool(previous["right_closed"])
            )
        else:
            merged.append(component)
    for index, component in enumerate(merged, start=1):
        component["component_id"] = index
    return merged
