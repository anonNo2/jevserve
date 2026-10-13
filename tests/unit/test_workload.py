"""CPU-only workload evidence and failure boundaries."""

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

import validation.workload as workload
from validation.workload import measure, percentile, request_body, run


def response(answer):
    return {
        "answers": {"q": answer},
        "request_id": "test",
        "model_revision": "a" * 40,
        "adapter_revision": "test",
        "execution_profile": "test",
        "probability_kind": "candidate_softmax",
        "calibration_id": None,
    }


def options(tmp_path, **updates):
    args = dict(
        output=tmp_path / "report.json",
        concurrency=3,
        requests=3,
        duration=1,
        interval=0,
        kind="noul",
        candidates=2,
        api_key_file=None,
        url="http://test",
        observe_gpu=False,
        image_digest="sha256:" + "a" * 64,
    )
    return SimpleNamespace(**(args | updates))


def mock_service(monkeypatch, failure_path=None, status=503, payload=None):
    original = httpx.AsyncClient

    def handler(request):
        if request.url.path == failure_path:
            return httpx.Response(status, json=payload or {"private": "secret-body"})
        if request.url.path == "/v1/models":
            return httpx.Response(
                200, json={"models": [{"id": "test-model", "revision": "a" * 40}]}
            )
        if request.url.path == "/readyz":
            return httpx.Response(200, json={})
        return httpx.Response(200, json=response({"type": "noul", "noul": 0.5}))

    monkeypatch.setattr(
        workload.httpx,
        "AsyncClient",
        lambda **kw: original(transport=httpx.MockTransport(handler), **kw),
    )


def test_nearest_rank_percentiles_include_tail():
    values = list(range(1, 101))
    assert percentile(values, 0.5) == 50
    assert percentile(values, 0.95) == 95
    assert percentile(values, 0.99) == 99
    assert percentile([], 0.99) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,payload,error",
    [
        (200, response({"type": "noul", "noul": 0.25}), None),
        (200, response({"type": "noul", "noul": 1.25}), "invalid_response"),
        (200, {"answers": {}}, "invalid_response"),
        (503, {}, "http_503"),
    ],
)
async def test_measured_failures_are_not_successes(status, payload, error):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda req: httpx.Response(status, json=payload)),
        base_url="http://test",
    ) as client:
        elapsed, failure = await measure(client, request_body("noul", 2), "a" * 40)
    assert elapsed >= 0
    assert failure == error


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind,answer,error",
    [
        (
            "choice",
            {"type": "choice", "choice": "other", "probabilities": {"other": 1.0}},
            "invalid_candidates",
        ),
        (
            "choice",
            {"type": "choice", "choice": "missing", "probabilities": {"other": 1.0}},
            "invalid_response",
        ),
        (
            "score",
            {
                "type": "score",
                "score": 0.5,
                "probabilities": {"0": 0.5, "1": 0.5},
                "legend": {"0": "wrong", "1": "rubric"},
            },
            "invalid_score_rubric",
        ),
        (
            "score",
            {
                "type": "score",
                "score": 1.0,
                "probabilities": {"0": 0.5, "1": 0.5},
                "legend": {"0": "Urgency level 0", "1": "Urgency level 1"},
            },
            "invalid_response",
        ),
    ],
)
async def test_contract_and_request_semantics(kind, answer, error):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda req: httpx.Response(200, json=response(answer))),
        base_url="http://test",
    ) as client:
        _, failure = await measure(client, request_body(kind, 2), "a" * 40)
    assert failure == error


@pytest.mark.asyncio
@pytest.mark.parametrize("interval", [0, 1])
async def test_workload_request_limit_does_not_pass_duration_gate(tmp_path, monkeypatch, interval):
    mock_service(monkeypatch)
    args = options(tmp_path, interval=interval)
    assert await run(args) == 0
    report = json.loads(args.output.read_text())
    assert report["completed_requests"] == 3
    assert report["successful_requests"] == 3
    assert report["status"] == "passed"
    assert report["termination_reason"] == "request_limit"
    assert report["duration_gate_passed"] is False
    assert report["elapsed_seconds"] < args.duration


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path,stage",
    [
        ("/readyz", "readiness"),
        ("/v1/models", "metadata"),
        ("/v1/decisions", "warmup"),
    ],
)
async def test_preflight_failure_has_sanitized_report(tmp_path, monkeypatch, path, stage):
    mock_service(monkeypatch, failure_path=path)
    args = options(tmp_path)
    assert await run(args) == 1
    text = args.output.read_text()
    report = json.loads(text)
    assert report["status"] == "failed"
    assert report["stage"] == stage
    assert report["error_code"] == "http_503"
    assert report["duration_gate_passed"] is False
    assert report["completed_requests"] == 0
    assert report["elapsed_seconds"] == 0
    assert "secret-body" not in text


@pytest.mark.parametrize("url", ["http://user:SECRET@test", "http://test?key=SECRET"])
def test_cli_rejects_credential_url_without_echo(tmp_path, capsys, url):
    result = workload.main(
        [
            "--url",
            url,
            "--image-digest",
            "sha256:" + "a" * 64,
            "--output",
            str(tmp_path / "report.json"),
        ]
    )
    assert result == 2
    assert capsys.readouterr().err == "error_code=invalid_configuration\n"


def test_cli_http_failure_has_no_traceback_or_body(tmp_path, monkeypatch, capsys):
    mock_service(monkeypatch, failure_path="/readyz", status=401)
    result = workload.main(
        [
            "--url",
            "http://test",
            "--image-digest",
            "sha256:" + "a" * 64,
            "--output",
            str(tmp_path / "report.json"),
        ]
    )
    assert result == 1
    assert capsys.readouterr().err == "error_code=workload_failed\n"


@pytest.mark.asyncio
async def test_bad_key_is_reported_without_value(tmp_path, monkeypatch):
    mock_service(monkeypatch)
    key_file = tmp_path / "key"
    key_file.write_text("SECRET\nINTERNAL_NEWLINE")
    args = options(tmp_path, api_key_file=key_file)
    assert await run(args) == 1
    text = args.output.read_text()
    assert "SECRET" not in text
    assert json.loads(text)["stage"] == "authentication"


@pytest.mark.asyncio
async def test_interruption_records_actual_elapsed_without_duration_success(tmp_path, monkeypatch):
    mock_service(monkeypatch)

    async def interrupted_sleep(seconds):
        raise asyncio.CancelledError

    monkeypatch.setattr(workload.asyncio, "sleep", interrupted_sleep)
    args = options(tmp_path, requests=100, interval=1)
    with pytest.raises(asyncio.CancelledError):
        await run(args)
    report = json.loads(args.output.read_text())
    assert report["status"] == "interrupted"
    assert report["duration_gate_passed"] is False
    assert 0 < report["elapsed_seconds"] < args.duration
    assert report["completed_requests"] == 3


@pytest.mark.asyncio
async def test_duration_gate_uses_elapsed_after_final_interval(tmp_path, monkeypatch):
    mock_service(monkeypatch)
    args = options(tmp_path, requests=100, interval=1)
    assert await run(args) == 0
    report = json.loads(args.output.read_text())
    assert report["status"] == "passed"
    assert report["termination_reason"] == "duration"
    assert report["duration_gate_passed"] is True
    assert report["elapsed_seconds"] >= args.duration
    assert report["throughput_requests_per_second"] == (
        report["completed_requests"] / report["elapsed_seconds"]
    )


@pytest.mark.asyncio
async def test_failed_batch_stops_and_counts_failures(tmp_path, monkeypatch):
    mock_service(monkeypatch)
    calls = 0

    async def measure_then_fail(client, body, expected_revision):
        nonlocal calls
        calls += 1
        return 0.01, None if calls == 1 else "http_503"

    monkeypatch.setattr(workload, "measure", measure_then_fail)
    args = options(tmp_path, requests=100)
    assert await run(args) == 1
    report = json.loads(args.output.read_text())
    assert calls == 4  # One warmup, one concurrent batch; no retries.
    assert report["completed_requests"] == 3
    assert report["successful_requests"] == 0
    assert report["failures"] == {"http_503": 3}
    assert report["status"] == "failed"
    assert report["duration_gate_passed"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("changed_call", [1, 3])
async def test_revision_change_fails_warmup_or_later_batch(tmp_path, monkeypatch, changed_call):
    original = httpx.AsyncClient
    decision_calls = 0

    def handler(request):
        nonlocal decision_calls
        if request.url.path == "/readyz":
            return httpx.Response(200, json={})
        if request.url.path == "/v1/models":
            return httpx.Response(
                200, json={"models": [{"id": "test-model", "revision": "a" * 40}]}
            )
        decision_calls += 1
        payload = response({"type": "noul", "noul": 0.5})
        if decision_calls >= changed_call:
            payload["model_revision"] = "b" * 40
        return httpx.Response(200, json=payload)

    monkeypatch.setattr(
        workload.httpx,
        "AsyncClient",
        lambda **kw: original(transport=httpx.MockTransport(handler), **kw),
    )
    args = options(tmp_path, concurrency=1, requests=10)
    assert await run(args) == 1
    report = json.loads(args.output.read_text())
    assert report["status"] == "failed"
    assert report["model"]["revision"] == "a" * 40
    assert report["duration_gate_passed"] is False
    assert decision_calls == changed_call
    if changed_call == 1:
        assert report["stage"] == "warmup"
        assert report["error_code"] == "model_revision_mismatch"
        assert report["completed_requests"] == 0
    else:
        assert report["stage"] == "workload"
        assert report["completed_requests"] == 2
        assert report["successful_requests"] == 1
        assert report["failures"] == {"model_revision_mismatch": 1}


@pytest.mark.asyncio
@pytest.mark.parametrize("revision", [None, "main", "g" * 40, 123])
async def test_metadata_requires_pinned_revision_before_warmup(tmp_path, monkeypatch, revision):
    original = httpx.AsyncClient
    paths = []

    def handler(request):
        paths.append(request.url.path)
        if request.url.path == "/readyz":
            return httpx.Response(200, json={})
        model = {"id": "test-model"}
        if revision is not None:
            model["revision"] = revision
        return httpx.Response(200, json={"models": [model]})

    monkeypatch.setattr(
        workload.httpx,
        "AsyncClient",
        lambda **kw: original(transport=httpx.MockTransport(handler), **kw),
    )
    args = options(tmp_path)
    assert await run(args) == 1
    report = json.loads(args.output.read_text())
    assert report["stage"] == "metadata"
    assert report["status"] == "failed"
    assert report["duration_gate_passed"] is False
    assert report["completed_requests"] == 0
    assert paths == ["/readyz", "/v1/models"]
