import copy

import pytest

from validation.numerics import compare, indexed, jpt, softmax


def test_softmax_invariant_to_full_vocabulary_normalization():
    # A shared log-normalizer cancels even when checkpoint temperature is not one.
    assert softmax([2.0, -1.0, 0.5], 1.7) == pytest.approx(
        softmax([-98.0, -101.0, -99.5], 1.7), abs=1e-14
    )
    assert sum(softmax([1000.0, 999.0], 0.5)) == pytest.approx(1)


def test_tolerance_is_relative_to_reference_not_observed():
    tolerance = {"probability_atol": 0.001, "probability_rtol": 0.01}
    assert compare([0.101999], [0.1], tolerance)
    assert not compare([0.10201], [0.1], tolerance)
    with pytest.raises(ValueError):
        compare([0.5], [0.5, 0.5], tolerance)
    with pytest.raises(ValueError):
        indexed([{"id": "same"}, {"id": "same"}], "id")


def test_noul_uses_public_true_probability_and_candidate_order_is_checked():
    reference = {"cases": [{"case": 0, "answers": {
        "q": {"type": "noul", "noul": 0.95},
        "c": {"type": "choice", "choice": "a", "probabilities": {"a": 0.7, "b": 0.3}},
    }}]}
    actual = copy.deepcopy(reference)
    actual["cases"][0]["answers"]["q"]["noul"] = 0.955
    tolerance = {"probability_atol": 0.001, "probability_rtol": 0.01}
    assert jpt(reference, actual, tolerance)["failed_cases"] == 0
    actual["cases"][0]["answers"]["c"]["probabilities"] = {"b": 0.3, "a": 0.7}
    with pytest.raises(ValueError, match="order"):
        jpt(reference, actual, tolerance)
