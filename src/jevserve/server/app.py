"""Bounded native HTTP API with one runtime owner and sanitized errors."""

import asyncio
import hmac
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp, Receive, Scope, Send

from jevserve.adapters.base import ModelAdapter
from jevserve.adapters.open_jev import InvalidModelOutput
from jevserve.artifacts.catalog import DEFAULT_MODEL, baseline
from jevserve.artifacts.manifest import PreparedModel
from jevserve.config import ServerConfig
from jevserve.contracts.errors import HTTP_STATUS, ErrorCode
from jevserve.contracts.requests import DecisionRequest, parse_request
from jevserve.contracts.responses import DecisionResponse
from jevserve.contracts.systemone import SystemOneRequest, systemone_response
from jevserve.runtime.base import Runtime
from jevserve.server.admission import Admission, Draining, Overloaded


def error(code: ErrorCode, request_id: str) -> JSONResponse:
    return JSONResponse(
        {
            "error": {"code": code.value, "message": code.value.replace("_", " ")},
            "request_id": request_id,
        },
        status_code=HTTP_STATUS[code],
    )


class RequestEnvelope:
    def __init__(self, app: ASGIApp, responses: Counter, duration: Histogram):
        self.app = app
        self.responses = responses
        self.duration = duration

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        measured = scope["path"] in ("/v1/decisions", "/v1/systemone")
        request_id = uuid.uuid4().hex
        scope.setdefault("state", {}).update(request_id=request_id, started=time.monotonic())

        async def identified(message: Any) -> None:
            if message["type"] == "http.response.start":
                if measured:
                    self.responses.labels(str(message["status"])).inc()
                message.setdefault("headers", []).append((b"x-request-id", request_id.encode()))
            await send(message)

        try:
            await self.app(scope, receive, identified)
        finally:
            if measured:
                self.duration.observe(time.monotonic() - scope["state"]["started"])


def create_app(
    runtime: Runtime,
    adapter: ModelAdapter,
    config: ServerConfig,
    *,
    prepared_model: PreparedModel | None = None,
    on_fatal: Callable[[], None] | None = None,
) -> FastAPI:
    admission = Admission(config.max_active, config.max_waiting)
    body_admission = Admission(config.max_active + config.max_waiting, 0)
    registry = CollectorRegistry()
    responses = Counter(
        "jevserve_decisions_total", "Decision responses", ["status"], registry=registry
    )
    active: set[asyncio.Task[Any]] = set()
    ready = False
    key: str | None = None
    duration = Histogram(
        "jevserve_decision_seconds", "Time spent handling decisions", registry=registry
    )
    Gauge("jevserve_active_requests", "Admitted decisions", registry=registry).set_function(
        lambda: admission.active
    )
    Gauge("jevserve_waiting_requests", "Queued decisions", registry=registry).set_function(
        lambda: admission.waiting
    )
    Gauge("jevserve_ready", "Runtime ready", registry=registry).set_function(
        lambda: int(ready and runtime.healthy())
    )

    async def monitor() -> None:
        nonlocal ready
        while True:
            await asyncio.sleep(0.25)
            if not runtime.healthy():
                ready = False
                await admission.close()
                if on_fatal is not None:
                    on_fatal()
                return

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        nonlocal ready, key
        if config.api_key_file is not None:
            key = config.api_key_file.read_text().strip()
            if not key or "\n" in key:
                raise ValueError("API key file must contain one nonempty line")
        monitor_task = None
        try:
            if prepared_model is not None:
                await runtime.start(prepared_model)
                warm_request = parse_request(
                    b'{"state":"Warmup","questions":{"ready":{"type":"noul","instructions":"Is this a warmup?"}}}'
                )
                deadline = time.monotonic() + config.max_deadline_ms / 1000
                async with asyncio.timeout_at(deadline):
                    adapter.decode(
                        await runtime.infer(adapter.encode(warm_request), "warmup", deadline)
                    )
            ready = runtime.healthy()
            monitor_task = asyncio.create_task(monitor())
            yield
        finally:
            if monitor_task is not None:
                monitor_task.cancel()
                await asyncio.gather(monitor_task, return_exceptions=True)
            ready = False
            await body_admission.close()
            await admission.close()
            try:
                await admission.wait_empty(time.monotonic() + config.drain_seconds)
            except TimeoutError:
                for task in tuple(active):
                    task.cancel()
                await asyncio.gather(*tuple(active), return_exceptions=True)
            await runtime.close()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(RequestEnvelope, responses=responses, duration=duration)
    app.state.admission = admission

    def authorized(request: Request) -> bool:
        return key is None or hmac.compare_digest(
            request.headers.get("authorization", "").encode(), ("Bearer " + key).encode()
        )

    @app.get("/livez")
    async def live() -> dict[str, str]:
        return {"status": "live"}

    @app.get("/readyz")
    async def readiness() -> JSONResponse:
        healthy = ready and runtime.healthy()
        return JSONResponse(
            {"status": "ready" if healthy else "not_ready"}, status_code=200 if healthy else 503
        )

    @app.get("/metrics")
    async def metrics(request: Request) -> Response:
        if not authorized(request):
            return error(ErrorCode.UNAUTHORIZED, request.state.request_id)
        return Response(generate_latest(registry), headers={"Content-Type": CONTENT_TYPE_LATEST})

    @app.get("/v1/models")
    async def models(request: Request) -> Response:
        if not authorized(request):
            return error(ErrorCode.UNAUTHORIZED, request.state.request_id)
        lock = baseline()
        return JSONResponse(
            {
                "models": [
                    {
                        "id": lock["model_id"],
                        "revision": lock["revision"],
                        "capabilities": adapter.capabilities().model_dump(),
                    }
                ]
            }
        )

    @app.post("/v1/systemone")
    @app.post("/v1/decisions")
    async def decisions(request: Request) -> Response:
        request_id = request.state.request_id
        if not authorized(request):
            return error(ErrorCode.UNAUTHORIZED, request_id)
        if not ready or not runtime.healthy():
            return error(ErrorCode.NOT_READY, request_id)
        task = asyncio.current_task()
        assert task is not None
        active.add(task)
        try:
            result = await execute(request)
            return result
        except TimeoutError:
            await runtime.abort(request_id)
            return error(ErrorCode.DEADLINE_EXCEEDED, request_id)
        except (Overloaded, Draining) as exc:
            return error(
                ErrorCode.OVERLOADED if isinstance(exc, Overloaded) else ErrorCode.NOT_READY,
                request_id,
            )
        except asyncio.CancelledError:
            await runtime.abort(request_id)
            raise
        except Exception:
            await runtime.abort(request_id)
            return error(ErrorCode.ENGINE_FAILED, request_id)
        finally:
            active.discard(task)

    async def watch_disconnect(request: Request) -> None:
        while True:
            if (await request.receive())["type"] == "http.disconnect":
                return

    async def execute(request: Request) -> Response:
        request_id = request.state.request_id
        deadline = request.state.started + config.deadline_ms / 1000
        async with body_admission.lease(deadline), asyncio.timeout_at(deadline):
            body = bytearray()
            async for chunk in request.stream():
                if len(body) + len(chunk) > config.body_limit_bytes:
                    return error(ErrorCode.BODY_TOO_LARGE, request_id)
                body.extend(chunk)
            try:
                parsed = (
                    SystemOneRequest.from_json(bytes(body))
                    if request.url.path == "/v1/systemone"
                    else parse_request(bytes(body))
                )
            except (ValueError, RecursionError):
                return error(ErrorCode.INVALID_REQUEST, request_id)
        if isinstance(parsed, SystemOneRequest) and parsed.model != DEFAULT_MODEL:
            return error(ErrorCode.UNSUPPORTED_CAPABILITY, request_id)
        if parsed.deadline_ms is not None:
            if parsed.deadline_ms > config.max_deadline_ms:
                return error(ErrorCode.INVALID_REQUEST, request_id)
            deadline = min(deadline, request.state.started + parsed.deadline_ms / 1000)
        if len(parsed.questions) > config.max_questions:
            return error(ErrorCode.UNSUPPORTED_CAPABILITY, request_id)
        operation = asyncio.create_task(infer(parsed, request_id, deadline))
        disconnected = asyncio.create_task(watch_disconnect(request))
        try:
            done, _ = await asyncio.wait(
                (operation, disconnected), return_when=asyncio.FIRST_COMPLETED
            )
            if disconnected in done:
                operation.cancel()
                await runtime.abort(request_id)
                return Response(status_code=499)
            return await operation
        finally:
            operation.cancel()
            disconnected.cancel()
            await asyncio.gather(operation, disconnected, return_exceptions=True)

    async def infer(parsed: DecisionRequest, request_id: str, deadline: float) -> Response:
        async with admission.lease(deadline), asyncio.timeout_at(deadline):
            try:
                adapter.capabilities().validate_request(parsed)
            except ValueError:
                return error(ErrorCode.UNSUPPORTED_CAPABILITY, request_id)
            try:
                encoded = adapter.encode(parsed)
            except ValueError:
                return error(ErrorCode.CONTEXT_TOO_LONG, request_id)
            raw = await runtime.infer(encoded, request_id, deadline)
            try:
                answers = adapter.decode(raw)
            except InvalidModelOutput:
                return error(ErrorCode.INVALID_MODEL_OUTPUT, request_id)
            response = DecisionResponse(
                answers=answers,
                request_id=request_id,
                model_revision=baseline()["revision"],
                adapter_revision="open-jev-v1",
                execution_profile="qwen35-unmerged-v1",
                probability_kind="candidate_softmax",
            )
            if isinstance(parsed, SystemOneRequest):
                return JSONResponse(
                    systemone_response(answers, encoded.input_tokens),
                    headers={
                        "x-model-revision": response.model_revision,
                        "x-execution-profile": response.execution_profile,
                    },
                )
            return JSONResponse(response.model_dump())

    return app
