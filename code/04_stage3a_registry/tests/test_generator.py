from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from geodose_stage3a.core import generate_preview


def test_generator_preview_contract():
    preview, hard_truth, localized_truth, edges, diagnostics = generate_preview()
    assert len(preview) == 625
    assert len(hard_truth) == 4375
    assert len(localized_truth) == 4375
    assert len(edges) == 2352
    assert (preview.loc[preview.treatment_category == "atom_0", "A"] == 0).all()
    assert (preview.loc[preview.treatment_category == "atom_1", "A"] == 1).all()
    assert diagnostics["dense_inverse_formed"] is False
    assert diagnostics["realization_dependent_centering_or_scaling"] is False
    assert diagnostics["mixed_treatment_normalization_max_abs_error"] < 1e-6
    assert diagnostics["realized_target_draws_required_before_coverage"] is True
    assert min(diagnostics["precision_min_eigenvalues"].values()) > 0
    assert set(localized_truth.artifact_scope) == {"analytic_conditional_mean_functional_only"}
    for endpoint in [0.0, 1.0]:
        hard = hard_truth.loc[hard_truth.dose == endpoint].sort_values("unit_id")
        local = localized_truth.loc[localized_truth.target_dose == endpoint].sort_values("unit_id")
        assert np.allclose(hard.Y_true.to_numpy(), local.localized_Y_true_mean.to_numpy())


def test_generator_is_deterministic():
    p1, h1, l1, e1, d1 = generate_preview()
    p2, h2, l2, e2, d2 = generate_preview()
    assert p1.equals(p2)
    assert h1.equals(h2)
    assert l1.equals(l2)
    assert e1.equals(e2)
    assert d1 == d2


if __name__ == "__main__":
    test_generator_preview_contract()
    test_generator_is_deterministic()
    print("STAGE3A GENERATOR UNIT TESTS PASSED")
