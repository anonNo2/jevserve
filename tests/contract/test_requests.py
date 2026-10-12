import subprocess
import sys

import pytest
from pydantic import ValidationError

from jevserve.contracts.requests import DecisionRequest, parse_request


def request():
    return {
        "state": "I was charged twice.",
        "questions": {
            "team": {
                "type": "choice",
                "instructions": "Choose a team.",
                "criteria": {"billing": "Payments", "technical": "Problems"},
            }
        },
    }


@pytest.mark.parametrize("count", [0, 17])
def test_question_count_rejected(count):
    data = request()
    data["questions"] = {str(i): data["questions"]["team"] for i in range(count)}
    with pytest.raises(ValidationError):
        DecisionRequest.model_validate(data)


def test_order_and_roundtrip():
    data = request()
    data["questions"]["a"] = {"type": "score", "instructions": "Rate.", "criteria": ["low", "high"]}
    data["questions"]["n"] = {"type": "noul", "instructions": "Is it paid?"}
    result = DecisionRequest.model_validate(data)
    assert list(result.questions) == ["team", "a", "n"]
    assert list(result.questions["team"].criteria) == ["billing", "technical"]
    assert parse_request(result.model_dump_json()) == result


@pytest.mark.parametrize(
    "state", ["", "  ", {}, None, [], 7, {"v": float("nan")}, {"v": [float("inf")]}, {1: "bad"}]
)
def test_invalid_state(state):
    data = request()
    data["state"] = state
    with pytest.raises(ValidationError):
        DecisionRequest.model_validate(data)


@pytest.mark.parametrize("instructions", ["", "  ", None, 123])
def test_invalid_instructions(instructions):
    data = request()
    data["questions"]["team"]["instructions"] = instructions
    with pytest.raises(ValidationError):
        DecisionRequest.model_validate(data)


@pytest.mark.parametrize(
    "question",
    [
        {"type": "choice", "instructions": "Pick.", "criteria": {}},
        {"type": "choice", "instructions": "Pick.", "criteria": {"": "bad"}},
        {"type": "choice", "instructions": "Pick.", "criteria": {"a": ""}},
        {"type": "score", "instructions": "Rate.", "criteria": ["one"]},
        {"type": "score", "instructions": "Rate.", "criteria": ["one", ""]},
        {"type": "noul", "instructions": "Yes?", "criteria": {"maybe": "Maybe"}},
        {"type": "unknown", "instructions": "What?"},
        {"type": "noul", "instructions": "Yes?", "extra": True},
    ],
)
def test_bad_question(question):
    with pytest.raises(ValidationError):
        DecisionRequest.model_validate({"state": "text", "questions": {"q": question}})


@pytest.mark.parametrize("deadline", [0, -1, 30001, True, "10", 1.5, float("inf")])
def test_deadline(deadline):
    with pytest.raises(ValidationError):
        DecisionRequest.model_validate({**request(), "deadline_ms": deadline})


@pytest.mark.parametrize(
    "raw",
    [
        '{"state":"a","state":"b","questions":{}}',
        '{"state":{"n":NaN},"questions":{}}',
        '{"state":{"n":Infinity},"questions":{}}',
        '{"state":"a","questions":{"q":{},"q":{}}}',
    ],
)
def test_strict_json(raw):
    with pytest.raises(ValueError):
        parse_request(raw)


def test_unknown_fields_and_question_id():
    for data in [
        {**request(), "typo": 1},
        {"state": "x", "questions": {" ": {"type": "noul", "instructions": "Y?"}}},
    ]:
        with pytest.raises(ValidationError):
            DecisionRequest.model_validate(data)


def test_protocol_limits_are_not_model_limits():
    data = request()
    data["questions"]["team"]["criteria"] = {str(i): None for i in range(256)}
    assert len(DecisionRequest.model_validate(data).questions["team"].criteria) == 256


def test_cpu_import_does_not_import_gpu_modules():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import jevserve.contracts.requests, jevserve.contracts.responses; "
            'import sys; assert not ({"torch", "vllm", "transformers"} & sys.modules.keys())',
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
