"""CPU contracts for bounded actual attention capture."""

import json
import subprocess
import sys

import pytest

from validation.jpt_attention_probe import finalize_arithmetic, select_attention_fixture


def test_only_original_first_fixture_is_selected():
    ref = {"cases": [{"case": 0, "rows": [{"input_ids": [1, 2]}]}, {"case": 1, "rows": []}]}
    case, row = select_attention_fixture(ref)
    assert case["case"] == 0 and row["input_ids"] == [1, 2]
    ref["cases"][0]["rows"][0]["input_ids"] = [1]
    with pytest.raises(ValueError):
        select_attention_fixture(ref)


def test_final_metadata_requires_all_eight_factors_and_restoration():
    from validation.jpt_arithmetic_probe import selected_factors

    factors = dict.fromkeys(selected_factors("hf_torch_decay_combined"), 1)
    metadata = {
        "removed": True,
        "forwards": 1,
        "calls": 8,
        "calls_per_forward": [8],
        "factor_calls_per_forward": [factors],
    }
    report = {}
    finalize_arithmetic(report, metadata)
    assert report["valid_trial"] and not report["certified"]
    with pytest.raises(ValueError):
        finalize_arithmetic({}, metadata | {"removed": False})


def test_help_and_import_are_gpu_free():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys,json;import validation.jpt_attention_probe;"
            'print(json.dumps([m for m in ("torch","vllm","transformers") if m in sys.modules]))',
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout) == []
    subprocess.run(
        [sys.executable, "-m", "validation.jpt_attention_probe", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
