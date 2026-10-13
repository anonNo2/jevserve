"""Exported contracts stay tied to parsers and actual CPU HTTP responses."""

import json
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
from conftest import FakeRuntime
from jsonschema import Draft202012Validator
from test_http import app_for, request

from jevserve.contracts.requests import parse_request
from validation.api_schema import export_documents

ROOT = Path(__file__).resolve().parents[2]


def validator(document, schema):
    return Draft202012Validator({**schema, "components": document["components"]})


def response_schema(document, path, status=200):
    return document["paths"][path][
        "get" if path in ("/livez", "/readyz", "/v1/models") else "post"
    ]["responses"][str(status)]["content"]["application/json"]["schema"]


def test_committed_exports_match_and_all_references_resolve():
    for filename, document in export_documents().items():
        assert json.loads((ROOT / "compat" / filename).read_text()) == document
        for schema in document["components"]["schemas"].values():
            Draft202012Validator.check_schema(schema)

        def check(value):
            if isinstance(value, dict):
                if "$ref" in value:
                    target = document
                    for segment in value["$ref"].removeprefix("#/").split("/"):
                        target = target[segment]
                for child in value.values():
                    check(child)
            elif isinstance(value, list):
                for child in value:
                    check(child)

        check(document)


def test_schema_export_has_no_server_or_gpu_imports():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from validation.api_schema import export_documents; "
            "export_documents(); import sys; "
            "assert not {'fastapi', 'uvicorn', 'torch', 'vllm', 'transformers'} & sys.modules.keys()",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_document_covers_real_routes_and_optional_auth():
    document = export_documents()["openapi-native.json"]
    app = app_for(FakeRuntime())
    assert set(document["paths"]) == {route.path for route in app.routes}
    assert app.openapi_url is None and app.docs_url is None and app.redoc_url is None
    for path, item in document["paths"].items():
        operation = next(iter(item.values()))
        assert operation["security"] == (
            [] if path in ("/livez", "/readyz") else [{"BearerAuth": []}, {}]
        )
    assert "/v1/decisions" not in export_documents()["openapi-systemone.json"]["paths"]


@pytest.mark.parametrize(
    "body",
    [
        request(),
        {
            "state": {"nested": [None, 1, {"ok": True}]},
            "questions": {
                "ok": {"type": "noul", "instructions": "Is it okay?", "criteria": {"true": "Yes"}},
                "score": {"type": "score", "instructions": "Rate", "criteria": ["Low", "High"]},
            },
        },
    ],
)
def test_request_schema_accepts_real_parser_payloads(body):
    document = export_documents()["openapi-native.json"]
    parse_request(json.dumps(body))
    schema = document["paths"]["/v1/decisions"]["post"]["requestBody"]["content"][
        "application/json"
    ]["schema"]
    validator(document, schema).validate(body)


@pytest.mark.parametrize(
    "mutation",
    [
        {"unknown": True},
        {"deadline_ms": True},
        {"state": "   "},
        {"deadline_ms": 0},
        {"questions": {}},
    ],
)
def test_request_schema_rejects_invalid_shapes(mutation):
    document = export_documents()["openapi-native.json"]
    body = request(**mutation)
    with pytest.raises(ValueError):
        parse_request(json.dumps(body))
    schema = document["paths"]["/v1/decisions"]["post"]["requestBody"]["content"][
        "application/json"
    ]["schema"]
    assert not validator(document, schema).is_valid(body)


@pytest.mark.asyncio
async def test_actual_native_systemone_models_health_and_error_responses():
    document = export_documents()["openapi-native.json"]
    fake = FakeRuntime()
    app = app_for(fake)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client,
    ):
        model = (await client.get("/v1/models")).json()["models"][0]["id"]
        body = request()
        body["questions"].update(
            {
                "ok": {"type": "noul", "instructions": "Is it okay?"},
                "score": {"type": "score", "instructions": "Rate", "criteria": ["Low", "High"]},
            }
        )
        for path in ("/v1/decisions", "/v1/systemone"):
            payload = body if path.endswith("decisions") else {**body, "model": model}
            result = await client.post(path, json=payload)
            assert result.status_code == 200
            validator(document, response_schema(document, path)).validate(result.json())
            malformed = await client.post(path, json={})
            assert malformed.status_code == 400
            validator(document, response_schema(document, path, 400)).validate(malformed.json())
        for path in ("/livez", "/readyz", "/v1/models"):
            result = await client.get(path)
            validator(document, response_schema(document, path)).validate(result.json())
        fake.is_healthy = False
        result = await client.get("/readyz")
        assert result.status_code == 503
        validator(document, response_schema(document, "/readyz", 503)).validate(result.json())


def test_export_check_detects_missing_or_modified_files(tmp_path):
    command = [sys.executable, "-m", "validation.api_schema", "--output-dir", str(tmp_path)]
    missing = subprocess.run([*command, "--check"], capture_output=True, text=True)
    assert missing.returncode == 1 and "API exports differ" in missing.stderr
    subprocess.run(command, check=True, capture_output=True)
    subprocess.run([*command, "--check"], check=True, capture_output=True)
    (tmp_path / "openapi-native.json").write_text("{}\n")
    changed = subprocess.run([*command, "--check"], capture_output=True, text=True)
    assert changed.returncode == 1 and "openapi-native.json" in changed.stderr


def test_status_specific_error_codes_and_systemone_required_model():
    document = export_documents()["openapi-native.json"]
    schema = response_schema(document, "/v1/decisions", 400)
    payload = {
        "error": {"code": "invalid_request", "message": "invalid request"},
        "request_id": "example",
    }
    validator(document, schema).validate(payload)
    payload["error"]["code"] = "overloaded"
    assert not validator(document, schema).is_valid(payload)
    schema = document["paths"]["/v1/systemone"]["post"]["requestBody"]["content"][
        "application/json"
    ]["schema"]
    assert not validator(document, schema).is_valid(request())
    validator(document, schema).validate(request(model="configured-model-id"))


@pytest.mark.asyncio
async def test_auth_error_and_metrics_wire_format(tmp_path):
    key = tmp_path / "api-key"
    key.write_text("test-only-key")
    document = export_documents()["openapi-native.json"]
    app = app_for(FakeRuntime(), api_key_file=key)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client,
    ):
        for path in ("/metrics", "/v1/models", "/v1/decisions", "/v1/systemone"):
            method = "get" if path in ("/metrics", "/v1/models") else "post"
            result = await client.request(method, path)
            assert result.status_code == 401
            schema = document["paths"][path][method]["responses"]["401"]["content"][
                "application/json"
            ]["schema"]
            validator(document, schema).validate(result.json())
        assert (await client.get("/livez")).status_code == 200
        assert (await client.get("/readyz")).status_code == 200
        result = await client.get("/metrics", headers={"Authorization": "Bearer test-only-key"})
        assert result.status_code == 200
        assert result.headers["content-type"].startswith("text/plain")
        assert "jevserve_ready" in result.text
