import copy
import math

import pytest

from validation.acceptance import compare_answers

POLICY = dict(
    policy_id="bounded-drift-v1",
    probability_atol=0.03,
    probability_rtol=0,
    require_argmax_match=True,
)


def choice(a, b):
    return dict(type="choice", probabilities={"a": a, "b": b}, choice="a" if a >= b else "b")


def test_boundary_and_real_excess():
    expected = {"q": choice(0.7, 0.3)}
    assert compare_answers({"q": choice(0.73, 0.27)}, expected, POLICY)["passed"]
    assert not compare_answers({"q": choice(0.730000000002, 0.269999999998)}, expected, POLICY)[
        "passed"
    ]


def test_flip_under_tolerance_and_first_tie():
    result = compare_answers({"q": choice(0.499, 0.501)}, {"q": choice(0.501, 0.499)}, POLICY)
    assert not result["passed"] and result["argmax_flip_count"] == 1
    assert result["questions"][0]["choice_changed"]
    assert compare_answers({"q": choice(0.5, 0.5)}, {"q": choice(0.51, 0.49)}, POLICY)["passed"]
    assert compare_answers(
        {"q": choice(0.499, 0.501)},
        {"q": choice(0.501, 0.499)},
        POLICY | {"require_argmax_match": False},
    )["passed"]


@pytest.mark.parametrize(
    "change",
    [
        lambda x: x["q"]["probabilities"].update(a=math.nan),
        lambda x: x["q"]["probabilities"].pop("b"),
        lambda x: x["q"].update(probabilities={"b": 0.3, "a": 0.7}),
        lambda x: x["q"].update(choice="b"),
        lambda x: x["q"]["probabilities"].update(a=0.9),
        lambda x: x.update(other=x.pop("q")),
        lambda x: x["q"].update(type="unknown"),
    ],
)
def test_invalid_answers(change):
    expected = {"q": choice(0.7, 0.3)}
    actual = copy.deepcopy(expected)
    change(actual)
    with pytest.raises(ValueError):
        compare_answers(actual, expected, POLICY)


@pytest.mark.parametrize(
    "change",
    [
        dict(probability_atol=-1),
        dict(probability_rtol=math.inf),
        dict(probability_atol=math.nan),
        dict(require_argmax_match=1),
    ],
)
def test_invalid_policy(change):
    with pytest.raises(ValueError):
        compare_answers({"q": choice(0.7, 0.3)}, {"q": choice(0.7, 0.3)}, POLICY | change)


def test_score_noul_multiple_questions():
    expected = {
        "n": dict(type="noul", noul=0.5),
        "s": dict(
            type="score",
            probabilities={"0": 0.6, "1": 0.3, "2": 0.1},
            score=0.5,
            legend={"0": "low", "1": "mid", "2": "high"},
        ),
    }
    actual = copy.deepcopy(expected)
    actual["n"]["noul"] = 0.501
    actual["s"].update(probabilities={"0": 0.58, "1": 0.31, "2": 0.11}, score=0.53)
    result = compare_answers(actual, expected, POLICY)
    assert result["question_count"] == 2 and result["failed_questions"] == ["n"]
    assert result["questions"][0]["noul_threshold_flip"]
    assert result["questions"][1]["expected_score_error"] == pytest.approx(0.03)
    assert result["questions"][1]["normalized_expected_score_error"] == pytest.approx(0.015)
    actual["s"]["score"] = 0.8
    with pytest.raises(ValueError):
        compare_answers(actual, expected, POLICY)


def test_relative_tolerance_and_no_mutation():
    expected, actual = {"q": choice(0.7, 0.3)}, {"q": choice(0.72, 0.28)}
    saved = copy.deepcopy((actual, expected, POLICY))
    assert compare_answers(
        actual, expected, POLICY | {"probability_atol": 0, "probability_rtol": 0.1}
    )["passed"]
    assert (actual, expected, POLICY) == saved


def test_no_silent_normalization_and_probability_boolean_rejected():
    expected = {"q": choice(0.7, 0.3)}
    actual = {"q": choice(0.7, 0.3000005)}
    result = compare_answers(actual, expected, POLICY)
    assert result["max_probability_error"] == pytest.approx(0.0000005)
    assert result["questions"][0]["total_variation"] == pytest.approx(0.00000025)
    actual["q"]["probabilities"]["a"] = True
    with pytest.raises(ValueError):
        compare_answers(actual, expected, POLICY)


def test_score_argmax_flip_is_gate_even_when_expected_score_error_small():
    expected = {
        "s": dict(
            type="score",
            probabilities={"0": 0.501, "1": 0.499},
            score=0.499,
            legend={"0": "low", "1": "high"},
        )
    }
    actual = copy.deepcopy(expected)
    actual["s"].update(probabilities={"0": 0.499, "1": 0.501}, score=0.501)
    result = compare_answers(actual, expected, POLICY)
    assert not result["passed"] and result["argmax_flip_count"] == 1
    assert result["questions"][0]["expected_score_error"] == pytest.approx(0.002)


@pytest.mark.parametrize("kind", ["choice", "score", "noul"])
def test_author_reference_optional_confidence(kind):
    if kind == "choice":
        answer = choice(0.7, 0.3)
    elif kind == "score":
        answer = dict(
            type="score",
            probabilities={"0": 0.7, "1": 0.3},
            score=0.3,
            legend={"0": "low", "1": "high"},
        )
    else:
        answer = dict(type="noul", noul=0.7)
    reference = answer | {"confidence": 0.91}
    assert compare_answers({"q": answer}, {"q": reference}, POLICY)["passed"]
    for invalid in (math.nan, -0.1, 1.1, True):
        with pytest.raises(ValueError):
            compare_answers({"q": answer}, {"q": reference | {"confidence": invalid}}, POLICY)
    with pytest.raises(ValueError):
        compare_answers({"q": answer}, {"q": reference | {"unknown": 1}}, POLICY)


def test_neohorse_noul_distribution_validated():
    actual = {"q": dict(type="noul", noul=0.7)}
    expected = {"q": dict(type="noul", noul=0.7, probabilities={"false": 0.3, "true": 0.7})}
    assert compare_answers(actual, expected, POLICY)["passed"]
    for mapping in (
        {"true": 0.7, "false": 0.3},
        {"false": 0.4, "true": 0.6},
        {"false": math.nan, "true": 0.7},
    ):
        with pytest.raises(ValueError):
            compare_answers(actual, {"q": expected["q"] | {"probabilities": mapping}}, POLICY)


def test_explicit_reference_rounding_only_and_reported_score():
    probs = {str(i): 0.0039 for i in range(255)}
    reference = {
        "s": dict(type="score", probabilities=probs, score=126.95, legend={k: k for k in probs})
    }
    actual_probs = {k: 1 / 255 for k in probs}
    actual = {"s": reference["s"] | {"probabilities": actual_probs, "score": 127.0}}
    with pytest.raises(ValueError):
        compare_answers(actual, reference, POLICY)
    result = compare_answers(actual, reference, POLICY, reference_probability_decimals=4)
    row = result["questions"][0]
    assert result["passed"]
    assert row["reference_rounding"]["normalization_tolerance"] == pytest.approx(0.012751)
    assert row["reference_rounding"]["score_consistency_tolerance"] == pytest.approx(1.619301)
    assert row["reported_score_error"] == pytest.approx(0.05)
    assert row["expected_score_error"] != row["reported_score_error"]
    with pytest.raises(ValueError):
        compare_answers(reference, reference, POLICY, reference_probability_decimals=4)
    for digits in (True, -1, 9, 4.0):
        with pytest.raises(ValueError):
            compare_answers(actual, reference, POLICY, reference_probability_decimals=digits)


def test_reference_rounding_does_not_expand_probability_gate():
    expected = {"q": choice(0.7, 0.3)}
    actual = {"q": choice(0.731, 0.269)}
    assert not compare_answers(actual, expected, POLICY, reference_probability_decimals=4)["passed"]
    with pytest.raises(ValueError, match="decimal places"):
        compare_answers(
            {"q": choice(0.70001, 0.29999)},
            {"q": choice(0.70001, 0.29999)},
            POLICY,
            reference_probability_decimals=4,
        )
