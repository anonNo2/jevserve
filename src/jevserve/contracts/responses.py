"""Typed normalized probabilities, never a calibrated accuracy claim."""

import math
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from jevserve.contracts.requests import Contract, Text

Probability = Annotated[float, Field(ge=0, le=1)]


class Distribution(Contract):
    probabilities: Annotated[dict[Text, Probability], Field(min_length=1)]

    @model_validator(mode="after")
    def normalized(self) -> Self:
        if not math.isclose(sum(self.probabilities.values()), 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("probabilities must sum to one")
        return self


class ChoiceAnswer(Distribution):
    type: Literal["choice"]
    choice: Text

    @model_validator(mode="after")
    def selected(self) -> Self:
        # Stable first candidate wins ties.
        if self.choice != max(self.probabilities, key=lambda key: self.probabilities[key]):
            raise ValueError("choice must be the first maximum-probability candidate")
        return self


class NoulAnswer(Contract):
    type: Literal["noul"]
    noul: Probability


class ScoreAnswer(Distribution):
    type: Literal["score"]
    score: Annotated[float, Field(ge=0)]
    legend: dict[Text, Text]

    @model_validator(mode="after")
    def ordinal(self) -> Self:
        expected_keys = [str(i) for i in range(len(self.probabilities))]
        if (
            len(expected_keys) < 2
            or list(self.probabilities) != expected_keys
            or list(self.legend) != expected_keys
        ):
            raise ValueError("Score requires ordered levels 0 through n-1 and matching legend")
        expected = sum(i * p for i, p in enumerate(self.probabilities.values()))
        if not math.isclose(self.score, expected, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("Score must equal ordinal expected value")
        return self


Answer = Annotated[ChoiceAnswer | NoulAnswer | ScoreAnswer, Field(discriminator="type")]


class DecisionResponse(Contract):
    answers: Annotated[dict[Text, Answer], Field(min_length=1, max_length=16)]
    request_id: Text
    model_revision: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    adapter_revision: Text
    execution_profile: Text
    probability_kind: Literal["candidate_softmax"]
    calibration_id: Text | None = None
