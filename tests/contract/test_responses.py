import pytest
from pydantic import ValidationError

from jevserve.contracts.responses import ChoiceAnswer, DecisionResponse, NoulAnswer, ScoreAnswer


@pytest.mark.parametrize(
    "probabilities",
    [{"a": float("nan")}, {"a": float("inf")}, {"a": 1.1, "b": -0.1}, {"a": 0.2, "b": 0.2}, {}],
)
def test_invalid_distribution(probabilities):
    with pytest.raises(ValidationError):
        ChoiceAnswer(type="choice", choice="a", probabilities=probabilities)


def test_stable_tie_order():
    assert ChoiceAnswer(type="choice", choice="b", probabilities={"b": 0.5, "a": 0.5}).choice == "b"
    with pytest.raises(ValidationError):
        ChoiceAnswer(type="choice", choice="a", probabilities={"b": 0.5, "a": 0.5})


def test_score_is_ordinal_expectation():
    data = dict(
        type="score",
        score=0.75,
        probabilities={"0": 0.25, "1": 0.75},
        legend={"0": "low", "1": "high"},
    )
    assert ScoreAnswer(**data).score == 0.75
    for change in [{"score": 75.0}, {"legend": {"1": "high", "0": "low"}}, {"score": float("nan")}]:
        with pytest.raises(ValidationError):
            ScoreAnswer(**{**data, **change})


def test_noul_and_explicit_null_calibration():
    response = DecisionResponse(
        answers={"n": NoulAnswer(type="noul", noul=0.7)},
        request_id="r",
        model_revision="a" * 40,
        adapter_revision="dev",
        execution_profile="test-only",
        probability_kind="candidate_softmax",
    )
    assert response.model_dump()["calibration_id"] is None
