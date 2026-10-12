"""Open-Jev independent-candidate scalar-head semantics.

Prompt/readout semantics follow Zefan-Cai/Open-Jev, pinned in the baseline lock.
See licenses/Open-Jev-MIT.txt and NOTICE. No engine or GPU imports.
"""

import json
import math
from typing import Protocol

from jevserve.adapters.base import Answers, EncodedDecision, RawDecision, ReadoutGroup
from jevserve.contracts.capabilities import Capabilities
from jevserve.contracts.requests import DecisionRequest
from jevserve.contracts.responses import ChoiceAnswer, NoulAnswer, ScoreAnswer


class Tokenizer(Protocol):
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str: ...
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...


class InvalidModelOutput(ValueError):
    pass


class OpenJevAdapter:
    def __init__(self, tokenizer: Tokenizer, *, temperature: float, capabilities: Capabilities):
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("invalid checkpoint temperature")
        if (
            capabilities.max_choice_candidates > 255
            or capabilities.max_score_levels > 10
            or capabilities.max_tokens_per_candidate > 4096
        ):
            raise ValueError("profile exceeds author model limits")
        self._tokenizer = tokenizer
        self._temperature = temperature
        self._capabilities = capabilities.model_copy(deep=True)

    def capabilities(self) -> Capabilities:
        return self._capabilities.model_copy(deep=True)

    def encode(self, request: DecisionRequest) -> EncodedDecision:
        self._capabilities.validate_request(request)
        state = (
            request.state
            if isinstance(request.state, str)
            else json.dumps(request.state, ensure_ascii=False, sort_keys=True, allow_nan=False)
        )
        groups, inputs = [], []
        for question_id, question in request.questions.items():
            instructions = question.instructions
            legend: tuple[str, ...] = ()
            keys: tuple[str, ...]
            if question.type == "noul":
                for key, label in [("true", "Yes"), ("false", "No")]:
                    if question.criteria and key in question.criteria:
                        instructions += f"\n{label} means: {question.criteria[key]}"
                suffixes = ["Is the answer to this question yes? Answer Yes or No."]
                keys = ("false", "true")
            else:
                if question.type == "choice":
                    keys = tuple(question.criteria)
                    candidates = [
                        key if description is None else f"{key}: {description}"
                        for key, description in question.criteria.items()
                    ]
                else:
                    candidates = question.criteria
                    keys = tuple(str(i) for i in range(len(candidates)))
                    legend = tuple(candidates)
                suffixes = [
                    f"Proposed answer: {candidate}\nIs this proposed answer correct? Answer Yes or No."
                    for candidate in candidates
                ]
            prefix = f"Context:\n{state}\n\nQuestion: {instructions}\n"
            for suffix in suffixes:
                rendered = self._tokenizer.apply_chat_template(
                    [{"role": "user", "content": prefix + suffix}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                tokens = tuple(self._tokenizer.encode(rendered, add_special_tokens=True))
                inputs.append(tokens)
                # Reject excess work before tokenizing further candidates.
                self._capabilities.validate_token_lengths([len(row) for row in inputs])
            groups.append(ReadoutGroup(question_id, question.type, keys, legend, len(suffixes)))
        lengths = [len(row) for row in inputs]
        self._capabilities.validate_token_lengths(lengths)
        return EncodedDecision(
            tuple(request.questions),
            tuple(inputs),
            tuple(groups),
            sum(lengths),
            len(inputs),
            tuple(n - 1 for n in lengths),
        )

    def decode(self, raw: RawDecision) -> Answers:
        if len(raw.scores) != len(raw.encoded.engine_inputs) or any(
            not math.isfinite(v) for v in raw.scores
        ):
            raise InvalidModelOutput("invalid number of scalar scores or nonfinite model output")
        answers: Answers = {}
        offset = 0
        for group in raw.encoded.groups:
            scores = raw.scores[offset : offset + group.sequence_count]
            offset += group.sequence_count
            logits = (0.0, scores[0]) if group.kind == "noul" else scores
            # Subtract the maximum before division to avoid positive overflow.
            maximum = max(logits)
            exponentials = [math.exp((value - maximum) / self._temperature) for value in logits]
            total = sum(exponentials)
            probabilities = [value / total for value in exponentials]
            mapping = dict(zip(group.keys, probabilities, strict=True))
            if group.kind == "choice":
                index = max(range(len(probabilities)), key=probabilities.__getitem__)
                answers[group.question_id] = ChoiceAnswer(
                    type="choice", choice=group.keys[index], probabilities=mapping
                )
            elif group.kind == "noul":
                answers[group.question_id] = NoulAnswer(type="noul", noul=probabilities[1])
            else:
                answers[group.question_id] = ScoreAnswer(
                    type="score",
                    probabilities=mapping,
                    score=sum(i * p for i, p in enumerate(probabilities)),
                    legend=dict(zip(group.keys, group.legend, strict=True)),
                )
        return answers
