"""CPU checks for hidden-probe comparison and worker-control boundaries."""

import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from validation.jpt_diagnostic import summarize
from validation.jpt_hidden_probe import (
    HiddenProbeWorker,
    hidden_difference,
    rpc,
    validate_capture,
    validate_hidden_reference,
)


def test_hidden_comparison_preserves_exact_zero_and_norm():
    assert hidden_difference([1, 2], [1, 2])["unequal_elements"] == 0
    result = hidden_difference([4, 0], [1, 4])
    assert result["unequal_elements"] == 2
    assert result["max_abs_error"] == 4
    assert result["l2_error"] == 5
    assert result["relative_l2_error"] == pytest.approx(5 / 17**0.5)
    assert hidden_difference([1], [0])["relative_l2_error"] is None


@pytest.mark.parametrize(
    "left,right", [([], []), ([1], [1, 2]), ([float("nan")], [1]), ([1], [float("inf")])]
)
def test_hidden_comparison_rejects_invalid_vectors(left, right):
    with pytest.raises(ValueError):
        hidden_difference(left, right)


def test_capture_binds_weight_bytes_candidate_order_and_identity():
    expected = {
        "candidate_token_ids": [3, 4],
        "weight_sha256": "hash",
        "weight_shape": [2, 2],
        "weight_dtype": "torch.bfloat16",
        "hidden": [1, 2],
    }
    captured = {**expected, "tag": "question"}
    assert validate_capture(captured, expected, "question")["max_abs_error"] == 0
    for key, value in [
        ("tag", "other"),
        ("candidate_token_ids", [4, 3]),
        ("weight_sha256", "other"),
        ("weight_shape", [1, 2]),
        ("weight_dtype", "torch.float32"),
    ]:
        with pytest.raises(ValueError):
            validate_capture({**captured, key: value}, expected, "question")


def test_hidden_report_links_original_logit_report():
    source = {
        "question_id": "q",
        "keys": ["a", "b"],
        "input_ids": [1, 2],
        "candidate_token_ids": [3, 4],
    }
    row = {
        **source,
        "hidden": [1, 2],
        "hidden_dtype": "torch.bfloat16",
        "hidden_shape": [1, 2],
        "fp32": summarize([3, 4], 1),
    }
    reference = {
        "model_id": "m",
        "revision": "r",
        "source_identity": "s",
        "upstream_revision": "u",
        "temperature": 1,
        "cases": [{"case": 0, "request": {}, "rows": [source]}],
    }
    hidden = {
        **reference,
        "mode": "hf_hidden",
        "reference_sha256": "digest",
        "cases": [{"case": 0, "request": {}, "rows": [row]}],
    }
    validate_hidden_reference(hidden, reference, "digest")
    with pytest.raises(ValueError, match="exact HF"):
        validate_hidden_reference(hidden, reference, "other")
    row["candidate_token_ids"] = [4, 3]
    with pytest.raises(ValueError, match="inputs differ"):
        validate_hidden_reference(hidden, reference, "digest")


def test_worker_extension_can_restore_original_method_and_refuses_double_arm(monkeypatch):
    def original(x):
        return x

    model = SimpleNamespace(compute_logits=original, lm_head=SimpleNamespace(tp_size=1))

    class Worker(HiddenProbeWorker):
        def get_model(self):
            return model

    worker = Worker()
    worker.jpt_hidden_install()
    assert model.compute_logits is not original
    with pytest.raises(ValueError, match="already installed"):
        worker.jpt_hidden_install()
    monkeypatch.setattr(
        "validation.jpt_hidden_probe.candidate_weights",
        lambda head, ids: (None, {"weight_sha256": "hash"}),
    )
    with pytest.raises(ValueError, match="weight bytes"):
        worker.jpt_hidden_arm("q", [3, 4], "wrong")
    worker.jpt_hidden_arm("q", [3, 4], "hash")
    with pytest.raises(ValueError, match="not drained"):
        worker.jpt_hidden_arm("q2", [3, 4], "hash")
    with pytest.raises(ValueError, match="exactly one"):
        worker.jpt_hidden_take()
    worker.jpt_hidden_remove()
    assert model.compute_logits is original
    assert not hasattr(worker, "_jpt_hidden_state")


@pytest.mark.asyncio
async def test_rpc_uses_named_methods_and_refuses_multiple_workers():
    calls = []

    async def collective(method, timeout, args):
        calls.append((method, timeout, args))
        return [{"ok": True}]

    runtime = SimpleNamespace(_engine=SimpleNamespace(collective_rpc=collective))
    assert await rpc(runtime, "jpt_hidden_arm", "q", [3, 4], "hash") == {"ok": True}
    assert calls == [("jpt_hidden_arm", 30, ("q", [3, 4], "hash"))]

    async def multiple(*args, **kwargs):
        return [{}, {}]

    runtime._engine.collective_rpc = multiple
    with pytest.raises(ValueError, match="one diagnostic worker"):
        await rpc(runtime, "jpt_hidden_take")


def test_import_and_help_do_not_load_gpu_packages():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys,json; import validation.jpt_hidden_probe; "
            "print(json.dumps([m for m in ('torch','vllm','transformers') if m in sys.modules]))",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout) == []
    subprocess.run(
        [sys.executable, "-m", "validation.jpt_hidden_probe", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
