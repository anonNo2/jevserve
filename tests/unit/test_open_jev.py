import math

import pytest

from jevserve.adapters.base import RawDecision
from jevserve.adapters.open_jev import InvalidModelOutput, OpenJevAdapter
from jevserve.contracts.capabilities import Capabilities
from jevserve.contracts.requests import DecisionRequest


class Tokenizer:
    def apply_chat_template(self, messages, **options):
        assert options == dict(tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return messages[0]["content"]

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens
        return list(text.encode())


def adapter():
    return OpenJevAdapter(
        Tokenizer(),
        temperature=2.0,
        capabilities=Capabilities(
            question_types=["choice", "noul", "score"],
            max_choice_candidates=255,
            max_score_levels=10,
            max_tokens_per_candidate=4096,
            max_candidate_sequences=256,
            max_total_input_tokens=65536,
        ),
    )


def encoded():
    return adapter().encode(
        DecisionRequest.model_validate(
            {
                "state": {"z": 1, "a": "中"},
                "questions": {
                    "c": {
                        "type": "choice",
                        "instructions": "Choose.",
                        "criteria": {"b": None, "a": "A"},
                    },
                    "n": {
                        "type": "noul",
                        "instructions": "Yes?",
                        "criteria": {"false": "No", "true": "Yes"},
                    },
                    "s": {"type": "score", "instructions": "Rate.", "criteria": ["low", "high"]},
                },
            }
        )
    )


def test_candidate_order_and_readout_positions():
    value = encoded()
    assert value.question_ids == ("c", "n", "s")
    assert [group.keys for group in value.groups] == [("b", "a"), ("false", "true"), ("0", "1")]
    assert len(value.engine_inputs) == 5
    assert value.readout_positions == tuple(len(row) - 1 for row in value.engine_inputs)
    assert value.input_tokens == sum(map(len, value.engine_inputs))
    first = bytes(value.engine_inputs[0]).decode()
    assert "Proposed answer: b\n" in first
    assert '"a": "中", "z": 1' in first


def test_temperature_once_noul_mapping_and_score_expectation():
    result = adapter().decode(RawDecision(encoded(), (0.0, 2.0, 2.0, 0.0, 2.0)))
    p = 1 / (1 + math.exp(-1))
    assert result["c"].choice == "a"
    assert result["c"].probabilities["a"] == pytest.approx(p)
    assert result["n"].noul == pytest.approx(p)
    assert result["s"].score == pytest.approx(p)
    assert list(result["s"].legend) == ["0", "1"]


@pytest.mark.parametrize(
    "scores", [(0.0,), (0.0, float("nan"), 0.0, 0.0, 0.0), (0.0, float("inf"), 0.0, 0.0, 0.0)]
)
def test_bad_raw_fails(scores):
    with pytest.raises(InvalidModelOutput):
        adapter().decode(RawDecision(encoded(), scores))


def test_extreme_logits_are_stable():
    result = adapter().decode(RawDecision(encoded(), (-1e10, 1e10, -1e10, 1e10, -1e10)))
    assert result["c"].probabilities == {"b": 0.0, "a": 1.0}
    assert result["n"].noul == 0.0
    assert result["s"].score == 0.0
