"""CPU boundaries for one-fixture arithmetic layer diagnostics."""

import json
import subprocess
import sys

import pytest

from validation.jpt_arithmetic_layers import finalize_report


def baseline_metadata():
    return {"removed": True, "forwards": 1, "calls": 0, "calls_per_forward": [0]}


def test_report_retains_layer_provenance_and_requires_one_question_coverage():
    report = {
        "mode": "runtime_layers",
        "reference_sha256": "original",
        "layer_reference_sha256": "layers",
        "source_identity": "weights",
    }
    finalize_report(report, baseline_metadata(), "baseline")
    assert report["mode"] == "runtime_arithmetic_layers"
    assert report["reference_sha256"] == "original"
    assert report["layer_reference_sha256"] == "layers"
    assert report["source_identity"] == "weights"
    assert report["valid_trial"] is True
    with pytest.raises(ValueError):
        finalize_report(
            {"mode": "runtime_layers"}, baseline_metadata() | {"forwards": 2}, "baseline"
        )


def test_unrestored_or_wrong_report_rejected():
    with pytest.raises(ValueError):
        finalize_report(
            {"mode": "runtime_layers"}, baseline_metadata() | {"removed": False}, "baseline"
        )
    with pytest.raises(ValueError):
        finalize_report({"mode": "hf_layers"}, baseline_metadata(), "baseline")


def test_help_and_import_do_not_load_gpu_libraries():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json,sys;import validation.jpt_arithmetic_layers;"
            'print(json.dumps([x for x in ("torch","transformers","vllm") if x in sys.modules]))',
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout) == []
    subprocess.run(
        [sys.executable, "-m", "validation.jpt_arithmetic_layers", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )


def test_combined_layers_require_every_factor_on_the_selected_question():
    from validation.jpt_arithmetic_probe import selected_factors

    mode = "hf_torch_projection_combined"
    factors = dict.fromkeys(selected_factors(mode), 1)
    metadata = {
        "removed": True,
        "forwards": 1,
        "calls": len(factors),
        "calls_per_forward": [len(factors)],
        "factor_calls_per_forward": [factors],
    }
    report = {"mode": "runtime_layers"}
    finalize_report(report, metadata, mode)
    assert report["certified"] is False
    assert report["production_changed"] is False
    with pytest.raises(ValueError):
        finalize_report(
            {"mode": "runtime_layers"}, metadata | {"factor_calls_per_forward": [{}]}, mode
        )
