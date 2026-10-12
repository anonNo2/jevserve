"""Replay real-tokenizer golden encodings without tokenizer/GPU dependencies in CI."""

import json
from pathlib import Path

from jevserve.adapters.open_jev import OpenJevAdapter
from jevserve.contracts.capabilities import Capabilities
from jevserve.contracts.requests import DecisionRequest

ROOT = Path(__file__).resolve().parents[2]


def test_prompt_and_readout_match_recorded_author_encoding():
    report = json.loads((ROOT / "validation/reports/open-jev-encoding.json").read_text())
    tokens = {}
    for case in report["cases"]:
        for prompt, row in zip(case["prompts"], case["engine_inputs"], strict=True):
            if prompt in tokens:
                assert tokens[prompt] == row
            tokens[prompt] = row

    class ReplayTokenizer:
        def apply_chat_template(self, messages, **options):
            assert options == dict(
                tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            return messages[0]["content"]

        def encode(self, text, *, add_special_tokens):
            assert add_special_tokens
            return tokens[text]

    adapter = OpenJevAdapter(
        ReplayTokenizer(),
        temperature=1.518796342858676,
        capabilities=Capabilities(
            question_types=["choice", "noul", "score"],
            max_choice_candidates=255,
            max_score_levels=10,
            max_tokens_per_candidate=4096,
            max_candidate_sequences=256,
            max_total_input_tokens=65536,
        ),
    )
    for case in report["cases"]:
        encoded = adapter.encode(DecisionRequest.model_validate(case["request"]))
        assert encoded.engine_inputs == tuple(tuple(row) for row in case["engine_inputs"])
        assert encoded.question_ids == tuple(case["question_ids"])
        assert encoded.readout_positions == tuple(case["readout_positions"])
        assert [list(group.keys) for group in encoded.groups] == case["answer_keys"]
