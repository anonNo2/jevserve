"""Export versioned OpenAPI contracts without importing the HTTP or GPU runtime.

Run from a source checkout: python -m validation.api_schema [--check].
Schemas derive from the actual Pydantic parsers; HTTP operations are explicit
because the live routes deliberately consume a bounded raw Request body.
"""

import argparse
import json
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from jevserve import __version__
from jevserve.contracts.capabilities import Capabilities
from jevserve.contracts.errors import HTTP_STATUS, ErrorResponse
from jevserve.contracts.requests import Contract, DecisionRequest, Text
from jevserve.contracts.responses import (
    ChoiceAnswer,
    DecisionResponse,
    NoulAnswer,
    Probability,
    ScoreAnswer,
)
from jevserve.contracts.systemone import SystemOneRequest


class SystemOneChoiceAnswer(ChoiceAnswer):
    confidence: Probability


class SystemOneScoreAnswer(ScoreAnswer):
    confidence: Probability


SystemOneAnswer = Annotated[
    SystemOneChoiceAnswer | NoulAnswer | SystemOneScoreAnswer, Field(discriminator="type")
]


class Usage(Contract):
    input_tokens: Annotated[int, Field(ge=0)]
    output_tokens: Annotated[int, Field(ge=0)]


class SystemOneResponse(Contract):
    model: Text
    answers: Annotated[dict[Text, SystemOneAnswer], Field(min_length=1, max_length=16)]
    usage: Usage


class ServedModel(Contract):
    id: Text
    revision: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    capabilities: Capabilities


class ModelsResponse(Contract):
    models: Annotated[list[ServedModel], Field(min_length=1, max_length=1)]


class LiveResponse(Contract):
    status: Literal["live"]


class ReadyResponse(Contract):
    status: Literal["ready"]


class NotReadyResponse(Contract):
    status: Literal["not_ready"]


def _reference(name: str) -> dict[str, str]:
    return {"$ref": f"#/components/schemas/{name}"}


def _response(name: str, description: str) -> dict[str, Any]:
    return {
        "description": description,
        "headers": {
            "x-request-id": {
                "description": "Server-generated correlation ID; also present in native/error bodies.",
                "schema": {"type": "string"},
            }
        },
        "content": {"application/json": {"schema": _reference(name)}},
    }


def _components() -> dict[str, Any]:
    schemas: dict[str, Any] = {}
    models: tuple[type[BaseModel], ...] = (
        DecisionRequest,
        SystemOneRequest,
        DecisionResponse,
        SystemOneResponse,
        ErrorResponse,
        ModelsResponse,
        LiveResponse,
        ReadyResponse,
        NotReadyResponse,
    )
    for model in models:
        schema = model.model_json_schema(ref_template="#/components/schemas/{model}")
        definitions = schema.pop("$defs", {})
        for name, definition in {**definitions, model.__name__: schema}.items():
            if name in schemas and schemas[name] != definition:
                raise ValueError(f"Conflicting schema component: {name}")
            schemas[name] = definition
    return {
        "schemas": schemas,
        "securitySchemes": {
            "BearerAuth": {
                "type": "http",
                "scheme": "bearer",
                "description": "Required on protected routes only when api_key_file is configured. "
                "Health routes stay public. Use TLS at a trusted reverse proxy outside localhost.",
            }
        },
    }


def _decision_operation(compatibility: bool) -> dict[str, Any]:
    responses: dict[str, Any] = {}
    codes: dict[int, list[str]] = {}
    for code, status in HTTP_STATUS.items():
        codes.setdefault(status, []).append(code.value)
    for status, values in codes.items():
        responses[str(status)] = _response("ErrorResponse", ", ".join(values))
        responses[str(status)]["x-error-codes"] = values
        responses[str(status)]["content"]["application/json"]["schema"] = {
            "allOf": [
                _reference("ErrorResponse"),
                {"properties": {"error": {"properties": {"code": {"enum": values}}}}},
            ]
        }
    responses["200"] = _response(
        "SystemOneResponse" if compatibility else "DecisionResponse",
        "Text Choice/Noul/Score answers. Probabilities/confidence are not calibrated accuracy.",
    )
    if compatibility:
        responses["200"]["headers"].update(
            {
                "x-model-revision": {"schema": {"type": "string", "pattern": "^[0-9a-f]{40}$"}},
                "x-execution-profile": {"schema": {"type": "string"}},
            }
        )
    responses["499"] = {
        "description": "Client disconnect observed after body parsing; engine work is aborted. "
        "Empty body; disconnected callers generally cannot receive this response.",
        "headers": {"x-request-id": {"schema": {"type": "string"}}},
    }
    return {
        "operationId": "systemone" if compatibility else "decide",
        "summary": "typesafe-sdk 0.7.4 text-question subset"
        if compatibility
        else "Native text decisions",
        "description": "One configured model; no routing or fallback. Strict JSON rejects duplicate "
        "keys and nonfinite values, and forbids undeclared request/question fields. "
        "Candidate and question insertion order is significant. Runtime model/token/work budgets "
        "are checked beyond this schema. deadline_ms only shortens the administrator deadline, "
        "measured from request entry, including body upload, queueing, encoding and inference. "
        "Requests above the administrator maximum are invalid. Disconnect/deadline aborts inference; "
        "Python encoding already running in a worker thread cannot be forcibly stopped."
        + (" model must exactly equal the single served /v1/models ID." if compatibility else ""),
        "security": [{"BearerAuth": []}, {}],
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {
                    "schema": _reference("SystemOneRequest" if compatibility else "DecisionRequest")
                }
            },
        },
        "responses": responses,
    }


def export_documents() -> dict[str, dict[str, Any]]:
    """Return stable documents; no FastAPI introspection or server construction."""
    protected = [{"BearerAuth": []}, {}]
    unauthorized = _response("ErrorResponse", "unauthorized")
    paths: dict[str, Any] = {
        "/livez": {
            "get": {
                "operationId": "liveness",
                "summary": "API process liveness",
                "security": [],
                "responses": {
                    "200": _response("LiveResponse", "Live; not an inference readiness check")
                },
            }
        },
        "/readyz": {
            "get": {
                "operationId": "readiness",
                "summary": "Runtime readiness",
                "security": [],
                "responses": {
                    "200": _response("ReadyResponse", "Ready"),
                    "503": _response(
                        "NotReadyResponse", "Not ready; health body, not ErrorResponse"
                    ),
                },
            }
        },
        "/metrics": {
            "get": {
                "operationId": "metrics",
                "summary": "Prometheus metrics",
                "security": protected,
                "responses": {
                    "200": {
                        "description": "Prometheus text exposition",
                        "headers": {"x-request-id": {"schema": {"type": "string"}}},
                        "content": {"text/plain": {"schema": {"type": "string"}}},
                    },
                    "401": unauthorized,
                },
            }
        },
        "/v1/models": {
            "get": {
                "operationId": "models",
                "summary": "Single configured model and request budgets",
                "description": "Metadata is not a readiness or numerical certification claim.",
                "security": protected,
                "responses": {
                    "200": _response("ModelsResponse", "Configured model"),
                    "401": unauthorized,
                },
            }
        },
        "/v1/decisions": {"post": _decision_operation(False)},
        "/v1/systemone": {"post": _decision_operation(True)},
    }
    common = {
        "openapi": "3.1.0",
        "jsonSchemaDialect": "https://json-schema.org/draft/2020-12/schema",
        "info": {
            "title": "JevServe HTTP API",
            "version": __version__,
            "description": "Development preview: native API and bounded SystemOne compatibility. "
            "Text only; no streaming, image inputs, batch jobs or general OpenAI compatibility. "
            "Static source contract; live /openapi.json, /docs and /redoc are disabled. "
            "JSON Schema cannot express duplicate-key rejection, finite recursive state values, "
            "candidate insertion order, probability normalization or ordinal score equations. "
            "See docs/api.md and runtime parsers for these additional invariants.",
        },
        "components": _components(),
    }
    return {
        "openapi-native.json": {**common, "paths": paths},
        "openapi-systemone.json": {
            **common,
            "info": {**common["info"], "title": "JevServe SystemOne compatibility subset"},
            "paths": {path: item for path, item in paths.items() if path != "/v1/decisions"},
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if committed exports drift")
    parser.add_argument("--output-dir", type=Path, default=Path("compat"))
    args = parser.parse_args()
    stale = []
    for name, document in export_documents().items():
        path = args.output_dir / name
        encoded = json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        if args.check:
            if not path.is_file() or path.read_text() != encoded:
                stale.append(str(path))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(encoded)
    if stale:
        parser.exit(1, "API exports differ: " + ", ".join(stale) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
