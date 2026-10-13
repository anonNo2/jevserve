"""Example contract and failure tests; no server, GPU, or network needed."""

import httpx
import pytest

from examples import business_examples as examples
from jevserve.client import JevServeError
from jevserve.contracts.responses import DecisionResponse


@pytest.mark.parametrize(
    "scenario,kind", list(zip(examples.SCENARIOS, ("choice", "noul", "score")))
)
def test_fixtures_are_bounded_native_requests(scenario, kind):
    request = examples.load_request(scenario)
    assert len(request.questions) == 1
    assert next(iter(request.questions.values())).type == kind
    assert request.deadline_ms == 30000


@pytest.mark.parametrize(
    "failure,status",
    [
        (JevServeError(429, "private-body", "private-id"), 1),
        (httpx.ReadTimeout("private-url"), 1),
        (httpx.ConnectError("private-url"), 1),
        (ValueError("private-body"), 2),
    ],
)
def test_failure_is_sanitized_without_retry(monkeypatch, capsys, failure, status):
    calls = []

    def fail(self, request):
        calls.append(request)
        raise failure

    monkeypatch.setattr(examples.Client, "decide", fail)
    assert examples.main(["ticket-routing"]) == status
    output = capsys.readouterr()
    assert len(calls) == 1
    assert output.out == ""
    assert "private" not in output.err


def test_url_credentials_rejected_before_client_creation(monkeypatch, capsys):
    def fail(*args, **kwargs):
        pytest.fail("client must not be created for a credential-bearing URL")

    monkeypatch.setattr(examples, "Client", fail)
    assert examples.main(["ticket-routing", "--url", "http://user:secret@localhost"]) == 2
    assert "secret" not in capsys.readouterr().err


def test_wrong_response_question_rejected():
    # Test-only synthetic response; the actual example always calls the SDK.
    response = DecisionResponse.model_validate(
        {
            "answers": {"unrelated": {"type": "noul", "noul": 0.5}},
            "request_id": "test",
            "model_revision": "0" * 40,
            "adapter_revision": "test",
            "execution_profile": "test",
            "probability_kind": "candidate_softmax",
            "calibration_id": None,
        }
    )
    with pytest.raises(ValueError, match="question mismatch"):
        examples.check_response(examples.load_request("content-moderation"), response)
