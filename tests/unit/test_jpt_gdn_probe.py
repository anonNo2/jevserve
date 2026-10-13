"""CPU contracts for typed internal GDN capture."""

import base64
import hashlib
import struct

import pytest

from validation.jpt_gdn_probe import compare_captures, decode_tensor


def record(values, dtype="torch.float32"):
    raw = struct.pack("<" + ("f" if dtype == "torch.float32" else "H") * len(values), *values)
    return {
        "shape": [1, len(values)],
        "dtype": dtype,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "data_base64": base64.b64encode(raw).decode(),
    }


def test_compare_preserves_typed_beta_difference():
    a = {"beta": record([0.5], "torch.float32")}
    e = {"beta": record([0x3F00], "torch.bfloat16")}
    result = compare_captures(a, e)
    assert result["first_unequal_boundary"] is None
    assert result["boundaries"][0]["actual_dtype"] == "torch.float32"
    assert result["boundaries"][0]["expected_dtype"] == "torch.bfloat16"


@pytest.mark.parametrize(
    "change", [{"sha256": "bad"}, {"shape": [2, 2]}, {"dtype": "torch.float64"}]
)
def test_typed_integrity_rejected(change):
    with pytest.raises(ValueError):
        decode_tensor(record([1.0]) | change)


def test_nonfinite_rejected():
    with pytest.raises(ValueError):
        decode_tensor(record([float("inf")]))


def test_missing_boundary_rejected():
    with pytest.raises(ValueError):
        compare_captures({"a": record([1.0])}, {"b": record([1.0])})


def test_patch_restores_class_lookup_and_incomplete_capture_rejected():
    from validation.jpt_gdn_probe import Capture

    class Owner:
        def method(self):
            return "original"

    owner = Owner()
    capture = Capture(2)
    capture.patch(owner, "method", lambda: "patched")
    assert owner.method() == "patched"
    with pytest.raises(ValueError, match="missing internal"):
        capture.finish()
    capture.close()
    assert owner.method() == "original"
    assert "method" not in vars(owner)


def test_capture_copies_before_inplace_reuse():
    from validation.jpt_gdn_probe import Capture

    class Tensor:
        def __init__(self, values):
            self.values = values

        def numel(self):
            return len(self.values)

        def detach(self):
            return self

        def clone(self):
            return Tensor(self.values.copy())

    capture = Capture(2)
    source = Tensor([1, 2])
    capture.add("recurrence", source)
    source.values[:] = [3, 4]
    assert capture.values["recurrence"].values == [1, 2]
    with pytest.raises(ValueError, match="duplicate"):
        capture.add("recurrence", source)


def test_import_and_help_do_not_load_gpu_libraries():
    import json
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys,json; import validation.jpt_gdn_probe; "
            'print(json.dumps([m for m in ("torch","vllm","transformers") if m in sys.modules]))',
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout) == []
    subprocess.run(
        [sys.executable, "-m", "validation.jpt_gdn_probe", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )


def test_passive_capture_requires_frozen_logits_unchanged():
    from validation.jpt_gdn_probe import frozen_logit_check

    result = frozen_logit_check([1.0, 2.0], [1.0, 2.0])
    assert result["exact_match"] is True
    assert result["difference"]["max_abs_error"] == 0
    with pytest.raises(ValueError, match="frozen HF logits"):
        frozen_logit_check([1.0, 2.25], [1.0, 2.0])
