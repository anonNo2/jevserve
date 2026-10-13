"""CPU-only integrity and boundary comparison contracts."""

import base64
import hashlib
import json
import struct
import subprocess
import sys

import pytest

from validation.jpt_layer_probe import compare_boundaries, decode_matrix, select_fixture


def matrix(words, shape=(2, 2)):
    raw = struct.pack("<" + "H" * len(words), *words)
    return {
        "shape": list(shape),
        "dtype": "torch.bfloat16",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bf16_le_base64": base64.b64encode(raw).decode(),
    }


def test_full_token_comparison_finds_difference_before_last_row():
    expected = {"embedding": matrix([0, 0, 0, 0]), "final_norm": matrix([0, 0, 0, 0])}
    actual = {"embedding": matrix([0, 0, 0, 0]), "final_norm": matrix([0x3F80, 0, 0, 0])}
    result = compare_boundaries(actual, expected)
    assert result["first_unequal_boundary"] == "final_norm"
    assert result["boundaries"][1]["all_tokens"]["max_abs_error"] == 1
    assert result["boundaries"][1]["last_token"]["max_abs_error"] == 0


@pytest.mark.parametrize(
    "change",
    [
        {"sha256": "bad"},
        {"shape": [2, 3]},
        {"dtype": "torch.float32"},
        {"bf16_le_base64": "!!!"},
        {"shape": [True, 4]},
    ],
)
def test_corrupt_capture_rejected(change):
    with pytest.raises(ValueError):
        decode_matrix({**matrix([0, 0, 0, 0]), **change})


def test_nonfinite_and_boundary_mismatch_rejected():
    with pytest.raises(ValueError):
        decode_matrix(matrix([0x7F80, 0, 0, 0]))
    with pytest.raises(ValueError):
        compare_boundaries({"embedding": matrix([0] * 4)}, {"final_norm": matrix([0] * 4)})


def test_subset_is_explicit_and_bounded():
    reference = {"cases": [{"case": "c", "request": {}, "rows": [{"input_ids": [1, 2]}]}]}
    assert select_fixture(reference, 0, 0)[1]["input_ids"] == [1, 2]
    for case, row in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
        with pytest.raises(ValueError):
            select_fixture(reference, case, row)
    reference["cases"][0]["rows"][0]["input_ids"] = [1] * 1025
    with pytest.raises(ValueError):
        select_fixture(reference, 0, 0)


def test_import_has_no_gpu_dependencies():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys,json; import validation.jpt_layer_probe; "
            'print(json.dumps([m for m in ("torch","vllm","transformers") if m in sys.modules]))',
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout) == []


def test_norm_hooks_copy_normalized_tuple_member_without_mutating_output():
    from types import SimpleNamespace

    from validation.jpt_layer_probe import install_hooks

    class Module:
        def register_forward_hook(self, hook):
            self.hook = hook
            return SimpleNamespace(remove=lambda: None)

    class Tensor:
        def __init__(self, values):
            self.values = values

        def detach(self):
            return self

        def clone(self):
            return Tensor(self.values.copy())

    modules = [Module() for _ in range(4)]
    body = SimpleNamespace(
        embed_tokens=modules[0],
        norm=modules[3],
        layers=[SimpleNamespace(input_layernorm=modules[1], post_attention_layernorm=modules[2])],
    )
    state = {"armed": None, "captures": {}}
    handles = install_hooks(body, state)
    normalized, residual = Tensor([1, 2]), Tensor([9, 9])
    output = (normalized, residual)
    assert modules[1].hook(modules[1], (), output) is None
    assert state["captures"] == {}
    state["armed"] = 2
    assert modules[1].hook(modules[1], (), output) is None
    saved = state["captures"]["layer.0.input_norm"]
    normalized.values[0] = 3
    assert saved.values == [1, 2]
    assert residual.values == [9, 9]
    with pytest.raises(ValueError, match="multiple forwards"):
        modules[1].hook(modules[1], (), output)
    assert len(handles) == 4
