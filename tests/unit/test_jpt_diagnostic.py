"""CPU-only diagnosis math and candidate-order guard coverage."""

import json
import subprocess
import sys

import pytest

from jevserve.adapters.base import EncodedDecision, ReadoutGroup
from validation.jpt_diagnostic import (
    check_encoding,
    compare,
    summarize,
    verify_reference,
)

TOLERANCE = {"probability_atol": 0.001, "probability_rtol": 0.01}


def test_common_logit_shift_is_not_probability_error():
    expected = summarize([5, 3, -2], 1.14)
    actual = summarize([105, 103, 98], 1.14)
    result = compare(actual, expected, TOLERANCE)
    assert result["max_raw_logits_error"] == 100
    assert result["max_centered_logits_error"] == 0
    assert result["max_probabilities_error"] == 0
    assert result["probability_parity_passed"]


def test_temperature_once_and_candidate_order():
    result = summarize([0, 2], 2)
    assert result["probabilities"] == pytest.approx([0.26894142137, 0.73105857863])
    assert result["centered_logits"] == [-2, 0]
    reversed_result = summarize([2, 0], 2)
    difference = compare(reversed_result, result, TOLERANCE)
    assert difference["argmax_changed"]
    assert not difference["probability_parity_passed"]


@pytest.mark.parametrize(
    "values,temperature",
    [([], 1), ([float("nan")], 1), ([float("inf")], 1), ([1], 0), ([1], float("inf")), ([1], -1)],
)
def test_rejects_invalid_logits_and_temperature(values, temperature):
    with pytest.raises(ValueError):
        summarize(values, temperature)


def test_stable_extreme_logits_and_single_candidate():
    assert summarize([10000], 1)["probabilities"] == [1]
    assert summarize([-10000, 10000], 1)["probabilities"] == [0, 1]


def test_count_mismatch_is_not_silently_zipped():
    with pytest.raises(ValueError, match="candidate count"):
        compare(summarize([1, 2], 1), summarize([1], 1), TOLERANCE)


def test_probability_gate_is_directional_frozen_tolerance():
    expected = summarize([0, 0], 1)
    actual = summarize([0, 0.03], 1)
    assert not compare(actual, expected, TOLERANCE)["probability_parity_passed"]


def test_encoding_guard_detects_candidate_permutation_and_tokens():
    rows = [
        {
            "question_id": "q",
            "keys": ["a", "b"],
            "input_ids": [10, 20],
            "candidate_token_ids": [42, 31],
        }
    ]
    encoded = EncodedDecision(
        ("q",),
        ((10, 20),),
        (ReadoutGroup("q", "choice", ("a", "b"), (), 1),),
        2,
        2,
        (1,),
        ((42, 31),),
    )
    check_encoding(encoded, rows)
    for key, value in [
        ("candidate_token_ids", [31, 42]),
        ("input_ids", [10, 21]),
        ("keys", ["b", "a"]),
        ("question_id", "other"),
    ]:
        with pytest.raises(ValueError, match="order differ"):
            check_encoding(encoded, [{**rows[0], key: value}])


def test_rejects_stale_reference_model_or_cases(monkeypatch):
    monkeypatch.setattr("validation.jpt_diagnostic.requests", lambda: [{"state": "synthetic"}])
    selected = {
        "model_id": "m",
        "revision": "r",
        "temperature": 1,
        "upstream_code": {"revision": "u"},
    }
    report = {
        "model_id": "m",
        "revision": "r",
        "mode": "reference_logits",
        "source_identity": "s",
        "temperature": 1,
        "upstream_revision": "u",
        "cases": [{"case": 0, "request": {"state": "synthetic"}}],
    }
    verify_reference(report, selected, "s", "reference_logits")
    for key, value in [
        ("revision", "stale"),
        ("source_identity", "stale"),
        ("temperature", 2),
        ("upstream_revision", "stale"),
        ("cases", []),
    ]:
        with pytest.raises(ValueError):
            verify_reference({**report, key: value}, selected, "s", "reference_logits")


def test_module_import_and_help_do_not_import_gpu_packages():
    code = (
        "import sys, json; import validation.jpt_diagnostic; "
        "print(json.dumps([x for x in ('torch', 'vllm', 'transformers') if x in sys.modules]))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )
    assert json.loads(result.stdout) == []
    subprocess.run(
        [sys.executable, "-m", "validation.jpt_diagnostic", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    "module,mode,extra",
    [
        ("jpt_diagnostic", ["reference"], []),
        ("jpt_diagnostic", ["runtime"], []),
        ("jpt_head_probe", [], ["--runtime", "unused-runtime.json"]),
        ("jpt_hidden_probe", ["hf"], []),
        ("jpt_hidden_probe", ["runtime"], ["--hidden-reference", "unused-hidden.json"]),
    ],
)
@pytest.mark.parametrize("existing_kind", ["report", "dangling_symlink", "directory"])
def test_diagnostic_clis_refuse_existing_output_before_loading_models(
    tmp_path, module, mode, extra, existing_kind
):
    output = tmp_path / "report.json"
    if existing_kind == "report":
        output.write_text('{"previous_success": true}\n')
    elif existing_kind == "dangling_symlink":
        output.symlink_to("missing-target.json")
    else:
        output.mkdir()
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "validation." + module,
            *mode,
            "--model",
            "invalid/model",
            "--model-dir",
            "unused-model",
            "--upstream",
            "unused-upstream",
            "--reference",
            "unused-reference.json",
            "--output",
            str(output),
            *extra,
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "output already exists; choose a new report path" in result.stderr
    assert "Traceback" not in result.stderr
    if existing_kind == "report":
        assert output.read_text() == '{"previous_success": true}\n'
    elif existing_kind == "dangling_symlink":
        assert str(output.readlink()) == "missing-target.json"
        assert not (tmp_path / "missing-target.json").exists()
    else:
        assert output.is_dir()
        assert list(output.iterdir()) == []
