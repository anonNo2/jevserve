import httpx
import pytest

from jevserve.cli import wait_for_ready


def clock(monkeypatch):
    now = [0.0]
    monkeypatch.setattr("jevserve.cli.time.monotonic", lambda: now[0])
    monkeypatch.setattr(
        "jevserve.cli.time.sleep", lambda seconds: now.__setitem__(0, now[0] + seconds)
    )


def test_startup_poll_waits_for_readiness_without_inference(monkeypatch):
    clock(monkeypatch)
    seen = []

    def handle(request):
        seen.append(request.url.path)
        return httpx.Response(503 if len(seen) < 3 else 200)

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        wait_for_ready(client, "http://test", 10)
    assert seen == ["/readyz"] * 3


def test_zero_wait_fails_immediately_and_wait_budget_is_bounded(monkeypatch):
    clock(monkeypatch)
    seen = []

    def handle(request):
        seen.append(request.url.path)
        return httpx.Response(503)

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(httpx.HTTPStatusError):
            wait_for_ready(client, "http://test", 0)
        assert len(seen) == 1
        with pytest.raises(TimeoutError):
            wait_for_ready(client, "http://test", 2)
        with pytest.raises(ValueError):
            wait_for_ready(client, "http://test", 1801)
    assert len(seen) == 4


def test_wrong_endpoint_is_not_retried(monkeypatch):
    clock(monkeypatch)
    with httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(404))) as client:
        with pytest.raises(httpx.HTTPStatusError):
            wait_for_ready(client, "http://test", 600)


def test_startup_can_wait_before_http_listener_exists(monkeypatch):
    clock(monkeypatch)
    calls = []

    def handle(request):
        calls.append(request.url.path)
        if len(calls) == 1:
            raise httpx.ConnectError("listener not started", request=request)
        return httpx.Response(200)

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        wait_for_ready(client, "http://test", 10)
    assert calls == ["/readyz", "/readyz"]
