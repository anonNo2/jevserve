"""CPU guards for diagnostic head experiments; no mock inference certification."""

import copy
import json
import subprocess
import sys

import pytest

from validation.jpt_diagnostic import summarize
from validation.jpt_head_probe import INPUT_KEYS, VARIANTS, aggregate, comparisons, verify_runtime


def reports():
    row = {
        "question_id": "q",
        "keys": ["a", "b"],
        "input_ids": [1, 2],
        "candidate_token_ids": [3, 4],
        **summarize([1, 0], 1),
    }
    reference = {
        "model_id": "m",
        "revision": "r",
        "source_identity": "s",
        "upstream_revision": "u",
        "temperature": 1,
        "cases": [{"case": 0, "rows": [row]}],
    }
    runtime = {
        **reference,
        "mode": "runtime_logits",
        "reference_sha256": "digest",
        "cases": [
            {
                "case": 0,
                "inputs": [{k: row[k] for k in INPUT_KEYS}],
                "runs": {"sequential": [summarize([1, 0], 1)]},
            }
        ],
    }
    return reference, runtime


def test_report_link_checks_exact_digest():
    reference, runtime = reports()
    verify_runtime(runtime, reference, "digest")
    with pytest.raises(ValueError, match="exact HF"):
        verify_runtime(runtime, reference, "other")
    with pytest.raises(ValueError, match="provenance"):
        verify_runtime({**runtime, "revision": "other"}, reference, "digest")


@pytest.mark.parametrize(
    "field, values", [("candidate_token_ids", [4, 3]), ("keys", ["b", "a"]), ("input_ids", [1, 3])]
)
def test_runtime_candidate_order_and_tokens_are_bound(field, values):
    reference, runtime = reports()
    runtime["cases"][0]["inputs"][0][field] = values
    with pytest.raises(ValueError, match="input IDs"):
        verify_runtime(runtime, reference, "digest")


def test_runtime_rejects_wrong_counts_and_stale_probabilities():
    reference, runtime = reports()
    for key, values in [
        ("probabilities", [0.5, 0.5]),
        ("raw_logits", [1]),
        ("centered_logits", [0, -2]),
    ]:
        altered = copy.deepcopy(runtime)
        altered["cases"][0]["runs"]["sequential"][0][key] = values
        with pytest.raises(ValueError):
            verify_runtime(altered, reference, "digest")
    runtime["cases"][0]["runs"]["sequential"] = []
    with pytest.raises(ValueError, match="question count"):
        verify_runtime(runtime, reference, "digest")


def test_comparisons_keep_hf_relative_tolerance_direction_and_summary():
    tolerance = {"probability_atol": 0.001, "probability_rtol": 0.01}
    a = summarize([0, 1], 1)
    b = summarize([10, 11], 1)
    variants = {name: a for name in VARIANTS} | {"B_last_full_bf16": b}
    runtime = summarize([1, 0], 1)
    result = comparisons(variants, a, {"batch": runtime}, tolerance)
    assert result["B_last_full_bf16_vs_A"]["max_raw_logits_error"] == 10
    assert result["B_last_full_bf16_vs_A"]["max_centered_logits_error"] == 0
    assert not result["vllm_batch_vs_A_original"]["probability_parity_passed"]
    summary = aggregate([{"rows": [{"comparisons": result}, {"comparisons": result}]}])
    assert summary["vllm_batch_vs_A_original"]["questions"] == 2
    assert summary["vllm_batch_vs_A_original"]["failed_questions"] == 2
    assert summary["vllm_batch_vs_A_original"]["argmax_changes"] == 2
    assert summary["B_last_full_bf16_vs_A"]["failed_questions"] == 0


def test_head_probe_import_and_help_are_gpu_free():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys,json; import validation.jpt_head_probe; "
            "print(json.dumps([m for m in ('torch','vllm','transformers') if m in sys.modules]))",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout) == []
    subprocess.run(
        [sys.executable, "-m", "validation.jpt_head_probe", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
