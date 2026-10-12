"""Decode actual author GPU logits; this is a CPU regression, not a new GPU run."""

import hashlib
import json
from pathlib import Path

import pytest

from jevserve.adapters.base import EncodedDecision, RawDecision, ReadoutGroup
from jevserve.adapters.open_jev import OpenJevAdapter
from jevserve.contracts.capabilities import Capabilities

ROOT = Path(__file__).resolve().parents[2]


def test_actual_gpu_reference_logits_decode_like_author():
    lock = json.loads((ROOT / "models/locks/open-jev-2b.json").read_text())
    path = ROOT / lock["reference_report"]["path"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == lock["reference_report"]["sha256"]
    report = json.loads(path.read_text())
    assert report["case_count"] == len(report["cases"]) == 113
    assert report["device_type"] == "cuda" and report["failures"] == 0

    class UnusedTokenizer:
        def apply_chat_template(self, *args, **kwargs):
            raise AssertionError("readout must not tokenize")

        def encode(self, *args, **kwargs):
            raise AssertionError("readout must not tokenize")

    adapter = OpenJevAdapter(
        UnusedTokenizer(),
        temperature=lock["temperature"],
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
        groups = tuple(
            ReadoutGroup(
                record["id"],
                record["kind"],
                tuple(record["answer_keys"]),
                tuple(record.get("legend", {}).values()),
                1 if record["kind"] == "noul" else len(record["options"]),
            )
            for record in case["records"]
        )
        rows = tuple(
            tuple(token for token, mask in zip(row, masks, strict=True) if mask)
            for row, masks in zip(
                case["encoding"]["input_ids"], case["encoding"]["attention_mask"], strict=True
            )
        )
        encoded = EncodedDecision(
            tuple(group.question_id for group in groups),
            rows,
            groups,
            sum(map(len, rows)),
            len(rows),
            tuple(len(row) - 1 for row in rows),
        )
        scores = tuple(
            value
            for record, row in zip(case["records"], case["logits"], strict=True)
            for value in (row[1:] if record["kind"] == "noul" else row)
        )
        actual = adapter.decode(RawDecision(encoded, scores))
        for question_id, answer in actual.items():
            expected = case["response"]["answers"][question_id]
            if answer.type == "noul":
                assert answer.noul == pytest.approx(expected["noul"], abs=1e-12)
            else:
                assert answer.probabilities == pytest.approx(expected["probabilities"], abs=1e-12)
                if answer.type == "choice":
                    assert answer.choice == expected["choice"]
                else:
                    assert answer.score == pytest.approx(expected["score"], abs=1e-12)
                    assert answer.legend == expected["legend"]
