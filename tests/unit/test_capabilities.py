import pytest

from jevserve.contracts.capabilities import Capabilities
from jevserve.contracts.requests import DecisionRequest


def caps(**changes):
    return Capabilities(
        **dict(
            question_types=["choice", "noul", "score"],
            max_choice_candidates=3,
            max_score_levels=2,
            max_tokens_per_candidate=100,
            max_candidate_sequences=3,
            max_total_input_tokens=150,
        )
        | changes
    )


def test_work_budget_and_capability_are_separate():
    request = DecisionRequest.model_validate(
        {
            "state": "x",
            "questions": {
                "q": {"type": "choice", "instructions": "Pick", "criteria": {"a": None, "b": None}}
            },
        }
    )
    caps().validate_request(request)
    for limits in [
        caps(question_types=["noul"]),
        caps(max_choice_candidates=1),
        caps(max_candidate_sequences=1),
    ]:
        with pytest.raises(ValueError):
            limits.validate_request(request)


@pytest.mark.parametrize("lengths", [[], [101], [100, 51], [1, 1, 1, 1], [True], [0], [-1]])
def test_token_budgets(lengths):
    with pytest.raises(ValueError):
        caps().validate_token_lengths(lengths)


def test_exact_budget_boundary():
    caps().validate_token_lengths([100, 50])
