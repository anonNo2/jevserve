"""Strict native protocol. Model-specific budgets are checked by capabilities."""

import json
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StringConstraints, field_validator

Text = Annotated[str, StringConstraints(strict=True, min_length=1, pattern=r"\S")]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class ChoiceQuestion(Contract):
    type: Literal["choice"]
    instructions: Text
    criteria: Annotated[dict[Text, Text | None], Field(min_length=1)]


class NoulQuestion(Contract):
    type: Literal["noul"]
    instructions: Text
    criteria: dict[Literal["true", "false"], Text] | None = None


class ScoreQuestion(Contract):
    type: Literal["score"]
    instructions: Text
    criteria: Annotated[list[Text], Field(min_length=2)]


Question = Annotated[ChoiceQuestion | NoulQuestion | ScoreQuestion, Field(discriminator="type")]


class DecisionRequest(Contract):
    state: Text | Annotated[dict[str, JsonValue], Field(min_length=1)]
    questions: Annotated[dict[Text, Question], Field(min_length=1, max_length=16)]
    deadline_ms: Annotated[int, Field(gt=0, le=30000)] | None = None

    @field_validator("state")
    @classmethod
    def finite_state(cls, value: str | dict[str, JsonValue]) -> str | dict[str, JsonValue]:
        # Recursively reject nonfinite numbers, including within nested arrays.
        json.dumps(value, allow_nan=False)
        return value

    @classmethod
    def from_json(cls, raw: str | bytes) -> Self:
        return cls.model_validate(
            json.loads(raw, object_pairs_hook=_unique, parse_constant=_constant)
        )


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def _constant(value: str) -> None:
    raise ValueError("nonfinite JSON number")


def parse_request(raw: str | bytes) -> DecisionRequest:
    """HTTP must enforce its byte limit before calling this strict JSON parser."""
    return DecisionRequest.from_json(raw)
