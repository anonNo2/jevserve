"""Native synchronous SDK: explicit timeout, no retries, validated responses."""

from pathlib import Path
from typing import Any, Self

import httpx

from jevserve.contracts.requests import DecisionRequest
from jevserve.contracts.responses import DecisionResponse


class JevServeError(RuntimeError):
    def __init__(self, status: int, code: str, request_id: str | None):
        super().__init__(f"JevServe returned HTTP {status}: {code}")
        self.status = status
        self.code = code
        self.request_id = request_id


class Client:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        *,
        timeout: float = 35,
        api_key_file: Path | None = None,
    ):
        headers = {}
        if api_key_file is not None:
            headers["Authorization"] = "Bearer " + api_key_file.read_text().strip()
        self._http = httpx.Client(
            base_url=base_url,
            timeout=timeout,
            headers=headers,
            follow_redirects=False,
            transport=httpx.HTTPTransport(retries=0),
        )

    def decide(self, request: DecisionRequest | dict[str, Any]) -> DecisionResponse:
        parsed = (
            request
            if isinstance(request, DecisionRequest)
            else DecisionRequest.model_validate(request)
        )
        response = self._http.post("/v1/decisions", json=parsed.model_dump(exclude_none=True))
        if response.is_error:
            code = "http_error"
            try:
                code = str(response.json()["error"]["code"])
            except (ValueError, KeyError, TypeError):
                pass
            raise JevServeError(response.status_code, code, response.headers.get("x-request-id"))
        return DecisionResponse.model_validate(response.json())

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
