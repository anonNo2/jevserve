import asyncio

import httpx
import pytest
from conftest import FakeRuntime, TestTokenizer

from jevserve.adapters.open_jev import OpenJevAdapter
from jevserve.config import ServerConfig
from jevserve.contracts.capabilities import Capabilities
from jevserve.server.app import create_app


def request(**extra):
    return {
        "state": "I was charged twice.",
        "questions": {
            "team": {
                "type": "choice",
                "instructions": "Choose a team.",
                "criteria": {"billing": "Payments", "technical": "Software"},
            }
        },
        **extra,
    }


def app_for(fake, *, prepared_model=None, **config):
    adapter = OpenJevAdapter(
        TestTokenizer(),
        temperature=1.5,
        capabilities=Capabilities(
            question_types=["choice", "noul", "score"],
            max_choice_candidates=255,
            max_score_levels=10,
            max_tokens_per_candidate=4096,
            max_candidate_sequences=256,
            max_total_input_tokens=65536,
        ),
    )
    return create_app(fake, adapter, ServerConfig(**config), prepared_model=prepared_model)


@pytest.mark.asyncio
async def test_success_health_and_stable_errors():
    fake = FakeRuntime()
    app = app_for(fake)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client,
    ):
        response = await client.post("/v1/decisions", json=request())
        assert response.status_code == 200
        body = response.json()
        assert body["answers"]["team"]["choice"] == "billing"
        assert body["calibration_id"] is None
        assert body["request_id"] == response.headers["x-request-id"]
        bad = await client.post(
            "/v1/decisions", content='{"state":"private","state":"private2","questions":{}}'
        )
        assert bad.status_code == 400 and "private" not in bad.text
        assert bad.json()["request_id"] == bad.headers["x-request-id"]
        assert (await client.get("/readyz")).status_code == 200
        fake.is_healthy = False
        assert (await client.get("/readyz")).status_code == 503
        assert (await client.get("/livez")).status_code == 200
    assert fake.closed


@pytest.mark.asyncio
async def test_timeout_releases_capacity():
    fake = FakeRuntime()
    app = app_for(fake)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client,
    ):
        fake.delay = 1
        result = await client.post("/v1/decisions", json=request(deadline_ms=10))
        assert result.status_code == 504
        assert app.state.admission.active == 0
        assert fake.aborted_ids


@pytest.mark.asyncio
async def test_auth_body_and_capability_limits(tmp_path):
    key = tmp_path / "key"
    key.write_text("secret-value")
    fake = FakeRuntime()
    app = app_for(fake, api_key_file=key, body_limit_bytes=500)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client,
    ):
        result = await client.post("/v1/decisions", json=request())
        assert result.status_code == 401 and result.json()["request_id"]
        client.headers["Authorization"] = "Bearer secret-value"
        assert (await client.post("/v1/decisions", content="x" * 501)).status_code == 413
        data = request()
        data["state"] = "a " * 3000
        assert (await client.post("/v1/decisions", json=data)).status_code == 413
        assert "secret-value" not in (await client.get("/metrics")).text


@pytest.mark.asyncio
async def test_overload_is_bounded():
    fake = FakeRuntime()
    app = app_for(fake, max_active=1, max_waiting=0)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client,
    ):
        fake.delay = 0.1
        first = asyncio.create_task(client.post("/v1/decisions", json=request()))
        while app.state.admission.active == 0:
            await asyncio.sleep(0.001)
        assert (await client.post("/v1/decisions", json=request())).status_code == 429
        assert (await first).status_code == 200


@pytest.mark.asyncio
@pytest.mark.parametrize("during_encode", [False, True])
async def test_disconnect_aborts_and_releases_admission(monkeypatch, during_encode):
    import json
    import threading

    entered = threading.Event()
    release = threading.Event()
    original = OpenJevAdapter.encode

    def slow_encode(self, parsed):
        entered.set()
        release.wait(1)
        return original(self, parsed)

    if during_encode:
        monkeypatch.setattr(OpenJevAdapter, "encode", slow_encode)

    fake = FakeRuntime()
    fake.delay = 10
    app = app_for(fake)
    incoming = asyncio.Queue()
    await incoming.put(
        {"type": "http.request", "body": json.dumps(request()).encode(), "more_body": False}
    )
    sent = []

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/decisions",
        "raw_path": b"/v1/decisions",
        "query_string": b"",
        "headers": [],
        "client": ("test", 1),
        "server": ("test", 80),
    }
    async with app.router.lifespan_context(app):
        running = asyncio.create_task(app(scope, incoming.get, send))
        try:
            async with asyncio.timeout(2):
                while app.state.admission.active == 0 or (during_encode and not entered.is_set()):
                    await asyncio.sleep(0.001)
                await incoming.put({"type": "http.disconnect"})
                await running
            assert app.state.admission.active == 0
            assert fake.aborted_ids
            if during_encode:
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app), base_url="http://test"
                ) as client:
                    assert (await client.get("/readyz")).status_code == 503
                    assert "jevserve_encoding_active 1.0" in (await client.get("/metrics")).text
        finally:
            release.set()


@pytest.mark.asyncio
async def test_queue_deadline_and_engine_failure_do_not_leak_input():
    fake = FakeRuntime()
    app = app_for(fake, max_active=1, max_waiting=1)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client,
    ):
        fake.delay = 0.1
        first = asyncio.create_task(client.post("/v1/decisions", json=request()))
        while app.state.admission.active == 0:
            await asyncio.sleep(0.001)
        queued = await client.post("/v1/decisions", json=request(deadline_ms=10))
        assert queued.status_code == 504
        assert app.state.admission.waiting == 0
        assert (await first).status_code == 200

        async def fail(*args):
            raise RuntimeError("private engine payload")

        fake.infer = fail
        failed = await client.post("/v1/decisions", json=request())
        assert failed.status_code == 503 and "private" not in failed.text
        assert app.state.admission.active == 0


@pytest.mark.asyncio
async def test_selected_model_metadata_and_systemone_routing(tmp_path):
    from jevserve.artifacts.catalog import source_manifest
    from jevserve.artifacts.manifest import PreparedModel

    manifest = source_manifest("kirp/jpt-9b")
    app = app_for(FakeRuntime(), prepared_model=PreparedModel(manifest=manifest, path=tmp_path))
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client,
    ):
        models = (await client.get("/v1/models")).json()
        assert models["models"][0]["id"] == manifest.model_id
        answer = await client.post("/v1/decisions", json=request())
        assert answer.json()["model_revision"] == manifest.revision
        accepted = await client.post("/v1/systemone", json=request(model=manifest.model_id))
        assert accepted.status_code == 200
        assert accepted.json()["model"] == manifest.model_id
        rejected = await client.post("/v1/systemone", json=request(model="ZefanCai/Open-Jev-2B"))
        assert rejected.status_code == 422


@pytest.mark.asyncio
async def test_slow_encoding_keeps_health_responsive_and_deadline_prompt(monkeypatch):
    import threading
    import time

    original = OpenJevAdapter.encode
    entered = threading.Event()
    release = threading.Event()

    def slow_encode(self, parsed):
        entered.set()
        release.wait(0.5)
        return original(self, parsed)

    monkeypatch.setattr(OpenJevAdapter, "encode", slow_encode)
    app = app_for(FakeRuntime(), drain_seconds=1)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client,
    ):
        started = time.monotonic()
        pending = asyncio.create_task(client.post("/v1/decisions", json=request(deadline_ms=40)))
        try:
            while not entered.is_set():
                await asyncio.sleep(0.001)
            assert (await client.get("/livez")).status_code == 200
            result = await pending
            assert result.status_code == 504
            assert time.monotonic() - started < 0.3
            assert (await client.get("/readyz")).status_code == 503
            assert (await client.post("/v1/decisions", json=request())).status_code == 503
            metrics = (await client.get("/metrics")).text
            assert "jevserve_encoding_active 1.0" in metrics
        finally:
            release.set()
            await pending
        async with asyncio.timeout(1):
            while (await client.get("/readyz")).status_code != 200:
                await asyncio.sleep(0.001)
        assert (await client.post("/v1/decisions", json=request())).status_code == 200


@pytest.mark.asyncio
async def test_cancelled_encoding_does_not_admit_replacement_work(monkeypatch):
    import threading

    original = OpenJevAdapter.encode
    entered = threading.Event()
    release = threading.Event()

    def slow_encode(self, parsed):
        entered.set()
        release.wait(0.5)
        return original(self, parsed)

    monkeypatch.setattr(OpenJevAdapter, "encode", slow_encode)
    app = app_for(FakeRuntime(), max_active=1, max_waiting=0, drain_seconds=1)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client,
    ):
        pending = asyncio.create_task(client.post("/v1/decisions", json=request()))
        try:
            while not entered.is_set():
                await asyncio.sleep(0.001)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
            assert (await client.get("/readyz")).status_code == 503
            for _ in range(8):
                assert (await client.post("/v1/decisions", json=request())).status_code == 503
            assert "jevserve_encoding_active 1.0" in (await client.get("/metrics")).text
        finally:
            release.set()


@pytest.mark.asyncio
async def test_internal_errors_have_sanitized_stage_diagnostics(caplog):
    import json
    import logging

    class BrokenRuntime(FakeRuntime):
        async def infer(self, encoded, request_id, deadline):
            raise RuntimeError("private-prompt secret-token /sensitive/path customer-id")

    app = app_for(BrokenRuntime())
    with caplog.at_level(logging.ERROR, logger="jevserve.server"):
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client,
        ):
            result = await client.post("/v1/decisions", json=request())
            assert result.status_code == 503
            records = [r for r in caplog.records if r.name == "jevserve.server"]
            assert records
            diagnostic = json.loads(records[0].getMessage())
            assert diagnostic["stage"] == "infer"
            assert diagnostic["code"] == "engine_failed"
            assert diagnostic["category"] == "runtime"
            assert records[0].exc_info is None
            metrics = (await client.get("/metrics")).text
            assert 'jevserve_errors_total{code="engine_failed",stage="infer"} 1.0' in metrics
            for sensitive in ("private-prompt", "secret-token", "/sensitive/path", "customer-id"):
                assert sensitive not in caplog.text + metrics + result.text
            assert result.json()["request_id"] not in caplog.text + metrics


@pytest.mark.asyncio
async def test_encoder_waiters_remain_bounded_and_expire_without_encoding(monkeypatch):
    import threading

    original = OpenJevAdapter.encode
    entered = threading.Event()
    release = threading.Event()
    encodes = []

    def slow_encode(self, parsed):
        encodes.append(parsed.state)
        entered.set()
        release.wait(1)
        return original(self, parsed)

    monkeypatch.setattr(OpenJevAdapter, "encode", slow_encode)
    app = app_for(FakeRuntime(), max_active=2, max_waiting=1)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client,
    ):
        first = asyncio.create_task(client.post("/v1/decisions", json=request()))
        second = third = None
        try:
            async with asyncio.timeout(1):
                while not entered.is_set():
                    await asyncio.sleep(0.001)
                second = asyncio.create_task(
                    client.post("/v1/decisions", json=request(state="second", deadline_ms=100))
                )
                while app.state.admission.active != 2:
                    await asyncio.sleep(0.001)
                third = asyncio.create_task(
                    client.post("/v1/decisions", json=request(state="third", deadline_ms=100))
                )
                while app.state.admission.waiting != 1:
                    await asyncio.sleep(0.001)
                assert (await client.post("/v1/decisions", json=request())).status_code == 429
                assert (await second).status_code == 504
                assert (await third).status_code == 504
                assert (await client.get("/readyz")).status_code == 200
                assert encodes == ["I was charged twice."]
        finally:
            release.set()
            await asyncio.gather(*(t for t in (first, second, third) if t is not None))
        assert first.result().status_code == 200


def test_hung_encoder_does_not_hold_process_shutdown():
    import subprocess
    import sys

    code = """
import asyncio
import threading
import time
from jevserve.server.encoding import Encoder

class HungAdapter:
    def encode(self, parsed):
        threading.Event().wait()

async def main():
    encoder = Encoder(HungAdapter())
    work = asyncio.create_task(encoder.encode(None))
    while not encoder.active:
        await asyncio.sleep(0.001)
    work.cancel()
    await asyncio.gather(work, return_exceptions=True)
    await encoder.close(time.monotonic() + 0.01)
    print("shutdown completed", flush=True)

asyncio.run(main())
"""
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=3, check=True
    )
    assert completed.stdout.strip() == "shutdown completed"


@pytest.mark.asyncio
async def test_shutdown_shares_one_drain_budget_with_running_encoder(monkeypatch):
    import threading
    import time

    entered = threading.Event()
    release = threading.Event()
    original = OpenJevAdapter.encode

    def slow_encode(self, parsed):
        entered.set()
        release.wait(3)
        return original(self, parsed)

    monkeypatch.setattr(OpenJevAdapter, "encode", slow_encode)
    app = app_for(FakeRuntime(), drain_seconds=1)
    pending = None
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as client:
            async with app.router.lifespan_context(app):
                pending = asyncio.create_task(client.post("/v1/decisions", json=request()))
                async with asyncio.timeout(1):
                    while not entered.is_set():
                        await asyncio.sleep(0.001)
                started = time.monotonic()
            assert time.monotonic() - started < 1.7
    finally:
        release.set()
        if pending is not None:
            await asyncio.gather(pending, return_exceptions=True)


@pytest.mark.asyncio
async def test_abort_failure_is_sanitized_and_disables_readiness(caplog):
    class BrokenAbort(FakeRuntime):
        async def infer(self, encoded, request_id, deadline):
            raise RuntimeError("secret-inference-text")

        async def abort(self, request_id):
            raise OSError("secret-abort-text /private/path")

    app = app_for(BrokenAbort())
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client,
    ):
        result = await client.post("/v1/decisions", json=request())
        assert result.status_code == 503
        assert (await client.get("/readyz")).status_code == 503
        metrics = (await client.get("/metrics")).text
        assert 'jevserve_errors_total{code="engine_failed",stage="abort"} 1.0' in metrics
        for private in ("secret-inference-text", "secret-abort-text", "/private/path"):
            assert private not in result.text + caplog.text + metrics
