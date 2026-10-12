"""NeoHorse pointer prompts, with independent causal question rows and no truncation.

Based on the pinned NeoHorse/kev Apache-2.0 implementation; see NOTICE.
"""

import re
from typing import Any

from jevserve.adapters.base import Answers, EncodedDecision, RawDecision, ReadoutGroup
from jevserve.adapters.candidate_readout import decode_candidates
from jevserve.contracts.capabilities import Capabilities
from jevserve.contracts.requests import DecisionRequest

SPECIAL = ("<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>")
CONTROL = re.compile(r"<\|([A-Za-z0-9_]+)\|>")


def render(value: Any, indent: int = 0) -> str:
    pad = "  " * indent
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return "\n".join(f"{pad}- {render(v, indent + 1).lstrip()}" for v in value)
    return "\n".join(
        f"{pad}{k}:\n{render(v, indent + 1)}"
        if isinstance(v, (dict, list))
        else f"{pad}{k}: {render(v)}"
        for k, v in value.items()
    )


class NeoHorseAdapter:
    def __init__(self, tokenizer: Any, capabilities: Capabilities, *, temperature: float = 1.0):
        self.temperature = temperature
        self.tokenizer = tokenizer
        self._capabilities = capabilities.model_copy(deep=True)
        self.special = tuple(tokenizer.convert_tokens_to_ids(t) for t in SPECIAL)
        if len(set(self.special)) != 5 or any(
            t is None or t == tokenizer.unk_token_id for t in self.special
        ):
            raise ValueError("checkpoint is missing decision delimiter tokens")

    def capabilities(self) -> Capabilities:
        return self._capabilities.model_copy(deep=True)

    def _tokens(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(CONTROL.sub(r"<¦\1¦>", text), add_special_tokens=False))

    def encode(self, request: DecisionRequest) -> EncodedDecision:
        self._capabilities.validate_request(request)
        state_id, question_id, option_id, end_id, decide_id = self.special
        state = [state_id] + self._tokens(render(request.state))
        if len(state) > 2048:
            raise ValueError("state exceeds 2048 tokens")
        groups, inputs, positions = [], [], []
        for qid, q in request.questions.items():
            legend: tuple[str, ...] = ()
            keys: tuple[str, ...]
            if q.type == "noul":
                criteria = q.criteria or {}
                keys = ("false", "true")
                options = [
                    "no" + (": " + criteria["false"] if "false" in criteria else ""),
                    "yes" + (": " + criteria["true"] if "true" in criteria else ""),
                ]
            elif q.type == "choice":
                keys = tuple(q.criteria)
                options = [k if v is None else f"{k}: {v}" for k, v in q.criteria.items()]
            else:
                keys = tuple(str(i) for i in range(len(q.criteria)))
                legend = tuple(q.criteria)
                options = list(q.criteria)
            tokens = state + [question_id] + self._tokens(q.instructions)
            ends = []
            for option in options:
                tokens.extend([option_id, *self._tokens(option), end_id])
                ends.append(len(tokens) - 1)
            tokens.append(decide_id)
            inputs.append(tuple(tokens))
            positions.append(tuple(ends))
            groups.append(ReadoutGroup(qid, q.type, keys, legend, 1))
            self._capabilities.validate_token_lengths([len(row) for row in inputs])
        lengths = [len(row) for row in inputs]
        return EncodedDecision(
            tuple(request.questions),
            tuple(inputs),
            tuple(groups),
            sum(lengths),
            sum(len(row) for row in positions),
            tuple(n - 1 for n in lengths),
            candidate_positions=tuple(positions),
        )

    def decode(self, raw: RawDecision) -> Answers:
        return decode_candidates(raw, self.temperature)
