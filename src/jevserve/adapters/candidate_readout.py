"""Stable typed readout of ordered candidate logits or log probabilities."""

import math

from jevserve.adapters.base import Answers, RawDecision
from jevserve.adapters.open_jev import InvalidModelOutput
from jevserve.contracts.responses import ChoiceAnswer, NoulAnswer, ScoreAnswer


def decode_candidates(raw: RawDecision, temperature: float) -> Answers:
    if raw.scores or len(raw.candidate_logits) != len(raw.encoded.groups):
        raise InvalidModelOutput("invalid candidate logprob row count")
    answers: Answers = {}
    for group, row in zip(raw.encoded.groups, raw.candidate_logits, strict=True):
        if len(row) != len(group.keys) or not all(math.isfinite(v) for v in row):
            raise InvalidModelOutput("invalid candidate logprobs")
        peak = max(row)
        weights = [math.exp((v - peak) / temperature) for v in row]
        total = math.fsum(weights)
        probabilities = [v / total for v in weights]
        mapping = dict(zip(group.keys, probabilities, strict=True))
        if group.kind == "noul":
            answers[group.question_id] = NoulAnswer(type="noul", noul=mapping["true"])
        elif group.kind == "choice":
            answers[group.question_id] = ChoiceAnswer(
                type="choice",
                choice=max(mapping, key=mapping.__getitem__),
                probabilities=mapping,
            )
        else:
            answers[group.question_id] = ScoreAnswer(
                type="score",
                score=math.fsum(i * p for i, p in enumerate(probabilities)),
                probabilities=mapping,
                legend=dict(zip(group.keys, group.legend, strict=True)),
            )
    return answers
