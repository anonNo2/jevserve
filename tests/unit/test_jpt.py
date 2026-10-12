import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from jevserve.adapters.base import RawDecision
from jevserve.adapters.jpt import JptAdapter
from jevserve.adapters.open_jev import InvalidModelOutput
from jevserve.contracts.capabilities import Capabilities
from jevserve.contracts.requests import DecisionRequest

FIXTURE = json.loads(Path("validation/reports/jpt-encoding.json").read_text())


class RecordedTokenizer:
    """Only frozen author-tokenizer outputs, with no models, GPU imports or downloads."""

    def __init__(self):
        self.known = {FIXTURE["probe"]: FIXTURE["probe_tokens"]}
        for case in FIXTURE["cases"]:
            self.known.update(zip(case["prompts"], case["tokens"], strict=True))
        self.labels = dict(zip(FIXTURE["labels"], FIXTURE["label_ids"], strict=True))

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == dict(tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return (
            "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
            + "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )

    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        if text in self.known:
            return self.known[text]
        if text.startswith(FIXTURE["probe"] + " "):
            label = text[len(FIXTURE["probe"]) + 1 :]
            return FIXTURE["probe_tokens"] + (
                [self.labels[label]] if label in self.labels else [1, 2]
            )
        raise AssertionError("Prompt differs from the frozen upstream encoding")

    def decode(self, ids):
        return " " + FIXTURE["labels"][FIXTURE["label_ids"].index(ids[0])]


@pytest.fixture
def adapter():
    return JptAdapter(
        RecordedTokenizer(),
        temperature=1.14,
        capabilities=Capabilities(
            question_types=["choice", "noul", "score"],
            min_choice_candidates=2,
            max_choice_candidates=255,
            max_score_levels=255,
            max_tokens_per_candidate=4095,
            max_candidate_sequences=256,
            max_total_input_tokens=65536,
        ),
    )


@pytest.mark.parametrize("case", FIXTURE["cases"])
def test_real_author_encoding_snapshot(adapter, case):
    encoded = adapter.encode(DecisionRequest.model_validate(case["request"]))
    assert encoded.engine_inputs == tuple(tuple(row) for row in case["tokens"])
    assert [list(g.keys) for g in encoded.groups] == case["keys"]
    assert encoded.input_tokens == sum(map(len, case["tokens"]))
    assert encoded.candidate_token_ids == tuple(
        tuple(FIXTURE["label_ids"][: len(keys)]) for keys in case["keys"]
    )


def test_noul_uses_true_first_and_checkpoint_temperature_once(adapter):
    encoded = adapter.encode(DecisionRequest.model_validate(FIXTURE["cases"][1]["request"]))
    raw = RawDecision(encoded, (), ((-1.0, -3.0),))
    assert adapter.decode(raw)["urgent"].noul == pytest.approx(1 / (1 + math.exp(-2 / 1.14)))
    with pytest.raises(InvalidModelOutput):
        adapter.decode(replace(raw, candidate_logits=((-1.0,),)))
    with pytest.raises(InvalidModelOutput):
        adapter.decode(replace(raw, candidate_logits=((-1.0, math.nan),)))


def test_score_ties_and_extreme_logits(adapter):
    encoded = adapter.encode(DecisionRequest.model_validate(FIXTURE["cases"][3]["request"]))
    assert adapter.decode(RawDecision(encoded, (), ((0.0, 0.0, 0.0),)))["q"].score == pytest.approx(
        1
    )
    result = adapter.decode(RawDecision(encoded, (), ((-10000.0, 0.0, -10000.0),)))["q"]
    assert result.probabilities == {"0": 0.0, "1": 1.0, "2": 0.0}
    encoded = adapter.encode(DecisionRequest.model_validate(FIXTURE["cases"][0]["request"]))
    assert adapter.decode(RawDecision(encoded, (), ((0.0, 0.0),)))["team"].choice == "billing"


def test_reject_unsupported_choice_and_token_budget(adapter):
    request = DecisionRequest.model_validate(FIXTURE["cases"][0]["request"])
    request.questions["team"].criteria = {"only": None}
    with pytest.raises(ValueError, match="candidate limit"):
        adapter.encode(request)
    adapter._capabilities.max_tokens_per_candidate = 1
    with pytest.raises(ValueError, match="context length"):
        adapter.encode(DecisionRequest.model_validate(FIXTURE["cases"][1]["request"]))
