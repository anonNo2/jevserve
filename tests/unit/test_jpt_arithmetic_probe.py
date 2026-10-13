"""Diagnostic guards remain GPU-import free."""

from types import SimpleNamespace

import pytest

from validation.jpt_arithmetic_probe import check_prefill_metadata, validate_mode


def test_unknown_or_combined_modes_are_rejected():
    for mode in ("", "all", "hf_beta,hf_qk_l2norm"):
        with pytest.raises(ValueError):
            validate_mode(mode)


def test_only_one_fresh_unpadded_prefill_is_supported():
    valid = dict(
        num_prefills=1,
        num_decodes=0,
        num_spec_decodes=0,
        num_prefill_tokens=7,
        spec_sequence_masks=None,
    )
    check_prefill_metadata(SimpleNamespace(**valid), 7, [False])
    for update in (
        {"num_prefills": 2},
        {"num_decodes": 1},
        {"num_spec_decodes": 1},
        {"num_prefill_tokens": 6},
        {"spec_sequence_masks": object()},
    ):
        with pytest.raises(ValueError):
            check_prefill_metadata(SimpleNamespace(**(valid | update)), 7, [False])
    with pytest.raises(ValueError):
        check_prefill_metadata(SimpleNamespace(**valid), 7, [True])
    with pytest.raises(ValueError):
        check_prefill_metadata(SimpleNamespace(**valid), 7, [])


def test_gated_norm_patch_covers_packed_prefill_bypass():
    from validation.jpt_arithmetic_probe import patch_gated_norm_paths

    class Norm:
        def forward(self, x, gate):
            raise AssertionError("original norm should be replaced")

    class GDN:
        def __init__(self):
            self.norm = Norm()

        def _rms_norm_gated_cuda(self, x, gate, out):
            raise AssertionError("packed prefill should be replaced")

    class Output:
        def copy_(self, value):
            self.value = value

    mixer = GDN()
    model = SimpleNamespace(modules=lambda: [mixer, mixer.norm])
    patched = []

    def patch(owner, name, value):
        patched.append((owner, name))
        setattr(owner, name, value)

    def replacement(norm, x, gate):
        assert norm is mixer.norm
        return (x, gate)

    assert patch_gated_norm_paths(model, Norm, GDN, patch, replacement) == 2
    assert mixer.norm.forward("x", "gate") == ("x", "gate")
    out = Output()
    assert mixer._rms_norm_gated_cuda("x", "gate", out) is None
    assert out.value == ("x", "gate")
    assert (mixer, "_rms_norm_gated_cuda") in patched


def test_trial_requires_each_question_to_execute_and_restore():
    from validation.jpt_arithmetic_probe import validate_trial

    metadata = dict(removed=True, forwards=2, calls=4, calls_per_forward=[2, 2])
    assert validate_trial(metadata, 2, "hf_beta")
    for update in (
        {"removed": False},
        {"forwards": 1},
        {"calls": 5},
        {"calls_per_forward": [0, 4]},
        {"calls_per_forward": [4]},
    ):
        assert not validate_trial(metadata | update, 2, "hf_beta")
    noop = dict(removed=True, forwards=2, calls=0, calls_per_forward=[0, 0])
    assert validate_trial(noop, 2, "baseline")
    assert not validate_trial(noop, 2, "hf_beta")


def test_chunk_guard_enforces_fresh_single_sequence_and_head_layout():
    from validation.jpt_arithmetic_probe import check_chunk_shapes

    valid = [
        (1, 7, 2, 128),
        (1, 7, 2, 128),
        (1, 7, 4, 128),
        (1, 7, 4),
        (1, 7, 4),
        (1, 4, 128, 128),
        [0, 7],
    ]
    assert check_chunk_shapes(*valid) == 2
    for index, replacement in (
        (0, (2, 7, 2, 128)),
        (5, (1, 2, 128, 128)),
        (6, [0, 3, 7]),
        (4, (1, 7, 2)),
    ):
        changed = list(valid)
        changed[index] = replacement
        with pytest.raises(ValueError):
            check_chunk_shapes(*changed)


def test_combined_trial_requires_every_factor_for_every_question():
    from validation.jpt_arithmetic_probe import selected_factors, validate_trial

    factors = set(selected_factors("hf_torch_combined"))
    assert len(factors) == 5
    valid = dict(
        removed=True,
        forwards=1,
        calls=5,
        calls_per_forward=[5],
        factor_calls_per_forward=[dict.fromkeys(factors, 1)],
    )
    assert validate_trial(valid, 1, "hf_torch_combined")
    for rows in ([], [{"hf_beta": 5}], [dict.fromkeys(factors, 0)]):
        assert not validate_trial(
            valid | {"factor_calls_per_forward": rows}, 1, "hf_torch_combined"
        )


def test_extended_combination_does_not_change_original_five_factors():
    from validation.jpt_arithmetic_probe import selected_factors

    five = selected_factors("hf_torch_combined")
    six = selected_factors("hf_torch_conv_combined")
    assert len(five) == 5
    assert six == (*five, "hf_conv_materialize")
    assert "hf_conv_core" not in five + six
    assert selected_factors("hf_conv_core") == ("hf_conv_core",)
    assert selected_factors("baseline") == ()


def test_convolution_guard_rejects_cached_and_packed_inputs():
    from validation.jpt_arithmetic_probe import check_conv_shapes

    check_conv_shapes((16, 7), (16, 4), (2, 16, 3), [False], [0, 7], [1])
    for states, lengths, slots in (
        ([True], [0, 7], [1]),
        ([False], [0, 3, 7], [1]),
        ([False], [0, 7], [-1]),
    ):
        with pytest.raises(ValueError):
            check_conv_shapes((16, 7), (16, 4), (2, 16, 3), states, lengths, slots)


def test_z_slice_requires_declared_noninterleaved_packed_shape():
    from validation.jpt_arithmetic_probe import z_weight_slice

    assert z_weight_slice(
        1024, 2048, 1024, (6144, 1024), [1024, 1024, 2048, 2048], 1, False, None
    ) == (4096, 6144)
    for sizes, tp, interleaved, bias in (
        ([6144], 1, False, None),
        ([1024, 1024, 2048, 2048], 2, False, None),
        ([1024, 1024, 2048, 2048], 1, True, None),
        ([1024, 1024, 2048, 2048], 1, False, object()),
    ):
        with pytest.raises(ValueError):
            z_weight_slice(1024, 2048, 1024, (6144, 1024), sizes, tp, interleaved, bias)


def test_projection_combination_uses_full_conv_and_preserves_older_modes():
    from validation.jpt_arithmetic_probe import selected_factors

    five = selected_factors("hf_torch_combined")
    seven = selected_factors("hf_torch_projection_combined")
    assert seven == (*five, "hf_conv_core", "hf_z_projection")
    assert len(seven) == 7
    assert "hf_conv_materialize" not in seven
    assert selected_factors("hf_torch_conv_combined") == (*five, "hf_conv_materialize")


def test_decay_combination_extends_exactly_seven_factor_trial():
    from validation.jpt_arithmetic_probe import selected_factors

    seven = selected_factors("hf_torch_projection_combined")
    eight = selected_factors("hf_torch_decay_combined")
    assert len(seven) == 7
    assert eight == (*seven, "hf_decay_g")
    assert selected_factors("hf_decay_g") == ("hf_decay_g",)


def test_full_convolution_preserves_downstream_triton_feature_stride():
    from validation.jpt_arithmetic_probe import preserve_conv_layout

    torch = pytest.importorskip("torch")
    tokens, channels, width = 7, 16, 4
    x = (
        torch.arange(tokens * channels, dtype=torch.float32, device="cpu")
        .reshape(tokens, channels)
        .to(torch.bfloat16)
        .transpose(0, 1)
    )
    native = torch.empty_like(x)
    weights = torch.ones((channels, 1, width), dtype=torch.bfloat16, device="cpu")
    raw = torch.nn.functional.conv1d(x.unsqueeze(0), weights, padding=width - 1, groups=channels)
    activated = torch.nn.functional.silu(raw[:, :, :tokens]).squeeze(0)
    bad = activated.transpose(0, 1)
    # The real Triton preparation kernel supplies token stride only and assumes
    # unit feature stride: reproduce its address calculation without a GPU.
    bad_kernel_view = bad.as_strided((tokens, channels), (bad.stride(0), 1))
    assert not torch.equal(bad_kernel_view, bad)
    original_stride = native.stride()
    fixed = preserve_conv_layout(native, activated)
    assert fixed is native
    assert fixed.stride() == original_stride
    assert torch.equal(fixed, activated)
    prepared = fixed.transpose(0, 1)
    assert prepared.stride(1) == 1
    kernel_view = prepared.as_strided((tokens, channels), (prepared.stride(0), 1))
    assert torch.equal(kernel_view, prepared)
    with pytest.raises(ValueError):
        preserve_conv_layout(activated, activated)
