import base64
import copy
import hashlib
import subprocess
import sys

import pytest

from validation.jpt_attention_replay import verify_pair


def pair():
    raw = b"\x80\x3f"
    tensor = dict(
        shape=[1, 1],
        dtype="torch.bfloat16",
        data_base64=base64.b64encode(raw).decode(),
        sha256=hashlib.sha256(raw).hexdigest(),
    )
    hf = dict(
        mode="hf_attention",
        selection=dict(case_index=0, row_index=0, case=0, layer_index=3),
        frozen_logit_comparison=dict(exact_match=True),
        capture=dict(
            boundaries={
                name: copy.deepcopy(tensor) for name in ("attention_input", "projected_q", "gate")
            }
        ),
    )
    rt = copy.deepcopy(hf)
    rt.update(mode="runtime_attention", valid_trial=True, attention_reference_sha256="abc")
    return hf, rt


def test_pair_requires_exact_inputs_and_link():
    hf, rt = pair()
    verify_pair(hf, rt, "abc")
    with pytest.raises(ValueError):
        verify_pair(hf, rt, "wrong")
    rt["capture"]["boundaries"]["gate"]["sha256"] = "bad"
    with pytest.raises(ValueError):
        verify_pair(hf, rt, "abc")


def test_pair_requires_matching_fixture():
    hf, rt = pair()
    rt["fixture"] = {"input_ids": [4]}
    with pytest.raises(ValueError):
        verify_pair(hf, rt, "abc")


def test_import_help_no_gpu_modules():
    subprocess.run(
        [
            sys.executable,
            "-c",
            'import sys; import validation.jpt_attention_replay; assert not any(m in sys.modules for m in ("torch","vllm","transformers"))',
        ],
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "validation.jpt_attention_replay", "--help"],
        check=True,
        capture_output=True,
    )
