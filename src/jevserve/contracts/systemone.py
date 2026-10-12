"""Text-question subset of typesafe-sdk 0.7.4, with explicit model selection.

Confidence formulas follow Open-Jev's MIT-licensed metrics (see NOTICE).
They describe distribution concentration, not calibrated correctness.
"""

from typing import Any

from jevserve.adapters.base import Answers
from jevserve.artifacts.catalog import DEFAULT_MODEL
from jevserve.contracts.requests import DecisionRequest, Text


class SystemOneRequest(DecisionRequest):
    model: Text


def systemone_response(answers: Answers, input_tokens: int) -> dict[str, Any]:
    result = {}
    for key, answer in answers.items():
        value = answer.model_dump()
        if answer.type == "choice":
            probabilities = list(answer.probabilities.values())
            n = len(probabilities)
            value["confidence"] = 1.0 if n == 1 else (max(probabilities) - 1 / n) / (1 - 1 / n)
        elif answer.type == "score":
            probabilities = list(answer.probabilities.values())
            n = len(probabilities)
            mode = max(range(n), key=probabilities.__getitem__)
            distance = sum(p * abs(i - mode) for i, p in enumerate(probabilities))
            center = (n - 1) / 2
            uniform = sum(abs(i - center) for i in range(n)) / n
            value["confidence"] = max(0.0, 1 - distance / uniform)
        result[key] = value
    return {
        "model": DEFAULT_MODEL,
        "answers": result,
        "usage": {"input_tokens": input_tokens, "output_tokens": 0},
    }
