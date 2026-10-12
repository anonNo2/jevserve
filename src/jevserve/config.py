"""Validated administrator settings; no engine imports or ambient env parsing."""

from typing import Annotated, Self

from pydantic import Field, model_validator

from jevserve.contracts.requests import Contract, Text


class ServerConfig(Contract):
    host: Text = "127.0.0.1"
    port: Annotated[int, Field(ge=1, le=65535)] = 8000
    api_workers: Annotated[int, Field(ge=1, le=1)] = 1
    body_limit_bytes: Annotated[int, Field(ge=1, le=1048576)] = 1048576
    max_questions: Annotated[int, Field(ge=1, le=16)] = 16
    max_waiting: Annotated[int, Field(ge=0)] = 64
    max_active: Annotated[int, Field(ge=1)] = 16
    deadline_ms: Annotated[int, Field(ge=1, le=30000)] = 30000
    max_deadline_ms: Annotated[int, Field(ge=1, le=30000)] = 30000
    drain_seconds: Annotated[int, Field(ge=0, le=30)] = 30

    @model_validator(mode="after")
    def valid_deadline(self) -> Self:
        if self.deadline_ms > self.max_deadline_ms:
            raise ValueError("default deadline exceeds administrator maximum")
        return self
