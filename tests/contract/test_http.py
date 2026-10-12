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


def app_for(fake, **config):
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
    return create_app(fake, adapter, ServerConfig(**config))


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
async def test_disconnect_aborts_and_releases_admission():
    import json

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
        async with asyncio.timeout(2):
            while app.state.admission.active == 0:
                await asyncio.sleep(0.001)
            await incoming.put({"type": "http.disconnect"})
            await running
        assert app.state.admission.active == 0
        assert fake.aborted_ids


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
