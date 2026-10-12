"""Immutable engine input and scalar readout contracts."""

from dataclasses import dataclass
from typing import Literal, Protocol

from jevserve.contracts.capabilities import Capabilities
from jevserve.contracts.requests import DecisionRequest
from jevserve.contracts.responses import ChoiceAnswer, NoulAnswer, ScoreAnswer

Answers = dict[str, ChoiceAnswer | NoulAnswer | ScoreAnswer]


@dataclass(frozen=True)
class ReadoutGroup:
    question_id: str
    kind: Literal["choice", "noul", "score"]
    keys: tuple[str, ...]
    legend: tuple[str, ...]
    sequence_count: int


@dataclass(frozen=True)
class EncodedDecision:
    question_ids: tuple[str, ...]
    engine_inputs: tuple[tuple[int, ...], ...]
    groups: tuple[ReadoutGroup, ...]
    input_tokens: int
    candidate_count: int
    readout_positions: tuple[int, ...]
    candidate_token_ids: tuple[tuple[int, ...], ...] = ()
    candidate_positions: tuple[tuple[int, ...], ...] = ()


@dataclass(frozen=True)
class RawDecision:
    encoded: EncodedDecision
    scores: tuple[float, ...]
    candidate_logits: tuple[tuple[float, ...], ...] = ()


class ModelAdapter(Protocol):
    def capabilities(self) -> Capabilities: ...
    def encode(self, request: DecisionRequest) -> EncodedDecision: ...
    def decode(self, raw: RawDecision) -> Answers: ...
