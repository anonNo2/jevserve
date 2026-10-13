"""Single-factor, diagnostic-only Qwen3.5 arithmetic counterfactuals.

Run with the same runtime arguments as jpt_hidden_probe plus --arithmetic-mode.
Single modes change one operation; combined modes list their factors explicitly. Reports are not certification evidence.
"""

import hashlib
import importlib.metadata
import inspect
import json
import re
import sys
import tempfile
from pathlib import Path
from types import MethodType, SimpleNamespace

from validation.jpt_hidden_probe import HiddenProbeWorker, HiddenRuntime, rpc

MODES = (
    "baseline",
    "hf_gated_norm",
    "hf_residual_norm",
    "hf_qk_l2norm",
    "hf_beta",
    "hf_chunk_core",
    "hf_torch_combined",
    "hf_conv_materialize",
    "hf_conv_core",
    "hf_torch_conv_combined",
    "hf_z_projection",
    "hf_torch_projection_combined",
    "hf_decay_g",
    "hf_torch_decay_combined",
)


FIVE_FACTORS = ("hf_gated_norm", "hf_residual_norm", "hf_qk_l2norm", "hf_beta", "hf_chunk_core")
COMBINATIONS = {
    "hf_torch_combined": FIVE_FACTORS,
    "hf_torch_conv_combined": (*FIVE_FACTORS, "hf_conv_materialize"),
    "hf_torch_projection_combined": (*FIVE_FACTORS, "hf_conv_core", "hf_z_projection"),
    "hf_torch_decay_combined": (*FIVE_FACTORS, "hf_conv_core", "hf_z_projection", "hf_decay_g"),
}


def selected_factors(mode):
    validate_mode(mode)
    return () if mode == "baseline" else COMBINATIONS.get(mode, (mode,))


def trial_scope(mode):
    if mode == "baseline":
        return "matched unpatched control"
    if mode in COMBINATIONS:
        return f"combined {len(COMBINATIONS[mode])}-factor HF torch-path counterfactual; not certification"
    return "single-operator HF torch-path counterfactual; not certification"


def preserve_conv_layout(native, replacement):
    """Retain native [channels,tokens] layout required by downstream Triton."""
    if (
        native.ndim != 2
        or native.stride(0) != 1
        or native.shape != replacement.shape
        or native.dtype != replacement.dtype
    ):
        raise ValueError("convolution output must preserve native unit channel stride and dtype")
    native.copy_(replacement)
    if native.transpose(0, 1).stride(-1) != 1:
        raise ValueError("post-convolution preparation requires unit feature stride")
    return native


def check_conv_shapes(x, weight, state, initial, lengths, slots):
    if (
        len(x) != 2
        or len(weight) != 2
        or len(state) != 3
        or x[1] <= 1
        or weight[0] != x[0]
        or weight[1] not in (2, 3, 4)
        or state[1] != x[0]
        or state[2] < weight[1] - 1
        or initial != [False]
        or lengths != [0, x[1]]
        or len(slots) != 1
        or type(slots[0]) is not int
        or not 0 <= slots[0] < state[0]
    ):
        raise ValueError(
            "convolution trial requires one fresh unpadded sequence and valid cache slot"
        )


def z_weight_slice(key_dim, value_dim, hidden_size, shape, output_sizes, tp, interleaved, bias):
    sizes = [key_dim, key_dim, value_dim, value_dim]
    if (
        tp != 1
        or interleaved
        or bias is not None
        or list(output_sizes) != sizes
        or tuple(shape) != (sum(sizes), hidden_size)
    ):
        raise ValueError("unsupported packed QKVZ projection layout")
    return 2 * key_dim + value_dim, sum(sizes)


def tensor_fingerprint(tensor):
    import torch

    if tensor.dtype != torch.bfloat16:
        raise ValueError("checkpoint comparison requires exact BF16 tensors, no conversion")
    value = tensor.detach().contiguous().cpu()
    return {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "sha256": hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest(),
    }


def verify_gdn_weights(model, checkpoint, gdn_type):
    """Compare loaded Z and conv tensors against the already verified package."""
    from safetensors import safe_open

    checkpoint = Path(checkpoint)
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        raise ValueError("expected verified local checkpoint directory")
    wanted, proofs = {}, {}
    for name, module in model.named_modules():
        if not isinstance(module, gdn_type):
            continue
        match = re.search(r"(?:^|\.)layers\.(\d+)\.linear_attn$", name)
        if match is None:
            raise ValueError("cannot map GDN module to checkpoint layer")
        index = int(match.group(1))
        if index in proofs:
            raise ValueError("duplicate GDN layer")
        projection = module.in_proj_qkvz
        start, end = z_weight_slice(
            module.key_dim,
            module.value_dim,
            module.hidden_size,
            projection.weight.shape,
            projection.output_sizes,
            module.tp_size,
            module.gqa_interleaved_layout,
            projection.bias,
        )
        if type(projection.quant_method).__name__ != "UnquantizedLinearMethod":
            raise ValueError("Z projection trial requires native unquantized linear")
        prefix = f"model.language_model.layers.{index}.linear_attn"
        proof = {"layer_index": index, "z_slice": [start, end], "tensors": {}}
        proofs[index] = proof
        for suffix, tensor in (
            ("in_proj_z.weight", projection.weight[start:end]),
            ("conv1d.weight", module.conv1d.weight),
        ):
            key = prefix + "." + suffix
            wanted[key] = tensor_fingerprint(tensor)
            proof["tensors"][key] = wanted[key]
    if not proofs:
        raise ValueError("no GDN weights found")
    found = set()
    for file in sorted(checkpoint.glob("*.safetensors")):
        if file.is_symlink() or not file.is_file():
            raise ValueError("unsafe checkpoint tensor file")
        with safe_open(file, framework="pt", device="cpu") as source:
            for key in set(source.keys()) & set(wanted):
                if key in found:
                    raise ValueError("duplicate checkpoint tensor key")
                if tensor_fingerprint(source.get_tensor(key)) != wanted[key]:
                    raise ValueError(f"loaded GDN tensor differs from checkpoint: {key}")
                found.add(key)
    if found != set(wanted):
        raise ValueError("checkpoint missing required GDN Z/convolution weights")
    return [proofs[index] for index in sorted(proofs)]


def verify_decay_weights(model, checkpoint, gdn_type):
    """Require loaded decay parameters to equal HF's effective BF16 values."""
    import torch
    from safetensors import safe_open

    checkpoint = Path(checkpoint)
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        raise ValueError("expected verified local checkpoint directory")
    wanted, proofs = {}, []
    for name, module in model.named_modules():
        if not isinstance(module, gdn_type):
            continue
        match = re.search(r"(?:^|\.)layers\.(\d+)\.linear_attn$", name)
        if match is None:
            raise ValueError("cannot identify decay parameter layer")
        index = int(match.group(1))
        for leaf in ("A_log", "dt_bias"):
            value = getattr(module, leaf).detach().cpu()
            if value.dtype not in (torch.bfloat16, torch.float32):
                raise ValueError("unsupported decay parameter dtype")
            effective = value.to(torch.bfloat16)
            if not torch.equal(value.float(), effective.float()):
                raise ValueError("loaded decay parameter differs from effective HF BF16 values")
            key = f"model.language_model.layers.{index}.linear_attn.{leaf}"
            if key in wanted:
                raise ValueError("duplicate loaded decay parameter")
            wanted[key] = tensor_fingerprint(effective)
            proofs.append(
                {
                    "layer_index": index,
                    "checkpoint_key": key,
                    "loaded_dtype": str(value.dtype),
                    "effective_bf16": wanted[key],
                }
            )
    if not wanted:
        raise ValueError("no GDN decay parameters found")
    found = set()
    for file in sorted(checkpoint.glob("*.safetensors")):
        if file.is_symlink() or not file.is_file():
            raise ValueError("unsafe checkpoint tensor file")
        with safe_open(file, framework="pt", device="cpu") as source:
            for key in set(source.keys()) & set(wanted):
                if key in found:
                    raise ValueError("duplicate checkpoint decay parameter")
                if tensor_fingerprint(source.get_tensor(key).to(torch.bfloat16)) != wanted[key]:
                    raise ValueError(f"decay parameter differs from HF checkpoint cast: {key}")
                found.add(key)
    if found != set(wanted):
        raise ValueError("checkpoint missing decay parameters")
    return proofs


ENGINE_PROFILE = {
    "dtype": "bfloat16",
    "max_model_len": 4096,
    "mamba_ssm_cache_dtype": "float32",
    "enforce_eager": True,
    "max_num_seqs": 1,
    "max_num_batched_tokens": 4096,
    "enable_chunked_prefill": False,
    "enable_prefix_caching": False,
}


def implementation_hashes(hf, gdn, norm_class, gated_module):
    owners = {
        "probe": sys.modules[__name__],
        "hf_qwen3_5": hf,
        "vllm_gdn": gdn,
        "vllm_layernorm": norm_class,
        "vllm_prefill_prep": gdn.fused_post_conv_prep,
        "vllm_fla_chunk": gdn.fla_chunk_gated_delta_rule,
        "vllm_gated_norm": gated_module,
        "vllm_causal_conv": gdn.causal_conv1d_fn,
    }
    return {
        name: hashlib.sha256(Path(inspect.getfile(inspect.unwrap(owner))).read_bytes()).hexdigest()
        for name, owner in owners.items()
    }


def validate_mode(mode):
    if mode not in MODES:
        raise ValueError(f"expected one arithmetic mode from {MODES}")
    return mode


def check_prefill_metadata(metadata, tokens, initial_states):
    if (
        tokens <= 1
        or metadata.num_prefills != 1
        or metadata.num_decodes != 0
        or metadata.num_spec_decodes != 0
        or metadata.num_prefill_tokens != tokens
        or metadata.spec_sequence_masks is not None
        or initial_states != [False]
    ):
        raise ValueError("arithmetic probe requires one fresh unpadded prefill, no decode/cache")


def check_chunk_shapes(q, k, v, g, beta, state, cu_seqlens):
    if (
        len(q) != 4
        or len(v) != 4
        or q != k
        or q[0] != 1
        or q[1] <= 1
        or q[:2] != v[:2]
        or q[2] <= 0
        or v[2] % q[2]
        or g != v[:3]
        or beta != g
        or state != (1, v[2], v[3], q[3])
        or cu_seqlens != [0, q[1]]
    ):
        raise ValueError("unsupported HF chunk counterfactual shapes or packed sequence")
    return v[2] // q[2]


def patch_gated_norm_paths(model, norm_type, gdn_type, patch, replacement):
    """Cover both ordinary Module.forward and packed-prefill in-place norm."""
    patched = 0
    for module in model.modules():
        if isinstance(module, norm_type):
            patch(module, "forward", MethodType(replacement, module))
            patched += 1
        if isinstance(module, gdn_type):

            def packed(this, x, output_gate, out):
                out.copy_(replacement(this.norm, x, output_gate))

            patch(module, "_rms_norm_gated_cuda", MethodType(packed, module))
            patched += 1
    return patched


def combined_coverage(metadata, questions, mode):
    expected = set(selected_factors(mode))
    factors = metadata.get("factor_calls_per_forward", [])
    return len(factors) == questions and all(
        set(row) == expected and all(count > 0 for count in row.values()) for row in factors
    )


def validate_trial(metadata, questions, mode):
    counts = metadata.get("calls_per_forward", []) if metadata else []
    return bool(
        metadata
        and metadata.get("removed")
        and metadata.get("forwards") == questions
        and len(counts) == questions
        and all(c == 0 if mode == "baseline" else c > 0 for c in counts)
        and sum(counts) == metadata.get("calls")
        and (mode not in COMBINATIONS or combined_coverage(metadata, questions, mode))
    )


class ArithmeticProbeWorker(HiddenProbeWorker):
    def jpt_arithmetic_install(self, mode):
        validate_mode(mode)
        if hasattr(self, "_jpt_arithmetic"):
            raise ValueError("remove previous arithmetic trial before installing another")
        for name, version in (("vllm", "0.29.0"), ("transformers", "5.10.4"), ("torch", "2.13.0")):
            if importlib.metadata.version(name).split("+")[0] != version:
                raise ValueError(f"arithmetic probe requires pinned {name} {version}")
        import torch
        from transformers.models.qwen3_5 import modeling_qwen3_5 as hf
        from vllm.forward_context import get_forward_context
        from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNormGated
        from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn
        from vllm.third_party.flash_linear_attention.ops import layernorm_guard
        from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

        source_hashes = implementation_hashes(hf, gdn, GemmaRMSNorm, layernorm_guard)

        model = self.get_model()
        active = set(selected_factors(mode))
        weight_proofs = (
            verify_gdn_weights(model, self.model_config.model, gdn.QwenGatedDeltaNetAttention)
            if active & {"hf_z_projection", "hf_conv_materialize", "hf_conv_core"}
            else []
        )
        decay_proofs = (
            verify_decay_weights(model, self.model_config.model, gdn.QwenGatedDeltaNetAttention)
            if "hf_decay_g" in active
            else []
        )
        state = {
            "mode": mode,
            "undo": [],
            "calls": 0,
            "forwards": 0,
            "validated": False,
            "calls_per_forward": [],
            "factor_calls_per_forward": [],
        }
        self._jpt_arithmetic = state

        def guard(_module, args, kwargs):
            state["validated"] = False
            tokens = kwargs.get("input_ids", args[0] if args else None)
            if tokens is None or tokens.ndim != 1:
                raise ValueError("expected flattened unpadded token input")
            metadata = get_forward_context().attn_metadata
            if not isinstance(metadata, dict):
                raise ValueError("expected per-layer attention metadata")
            found = 0
            for value in metadata.values():
                if isinstance(value, GDNAttentionMetadata):
                    initial = value.has_initial_state
                    states = [] if initial is None else initial.detach().cpu().tolist()
                    check_prefill_metadata(value, tokens.numel(), states)
                    found += 1
            if not found:
                raise ValueError("missing GDN prefill metadata")
            state["validated"] = True
            state["forwards"] += 1
            state["calls_per_forward"].append(0)
            state["factor_calls_per_forward"].append({})

        def count(factor):
            if not state["validated"]:
                raise ValueError("arithmetic patch called outside guarded prefill")
            state["calls"] += 1
            state["calls_per_forward"][-1] += 1
            factors = state["factor_calls_per_forward"][-1]
            factors[factor] = factors.get(factor, 0) + 1

        def patch(owner, name, replacement):
            # Preserve instance-vs-class attribute lookup when restoring methods.
            had_own = name in vars(owner)
            old = getattr(owner, name)
            state["undo"].append((owner, name, old, had_own))
            setattr(owner, name, replacement)

        try:
            state["guard"] = model.register_forward_pre_hook(guard, with_kwargs=True)
            patched = 0
            if "hf_gated_norm" in active:
                for module in model.modules():
                    if isinstance(module, RMSNormGated):
                        if (
                            not module.norm_before_gate
                            or module.activation not in ("silu", "swish")
                            or module.group_size is not None
                        ):
                            raise ValueError("unsupported gated norm configuration")

                def gated(this, x, z=None):
                    count("hf_gated_norm")
                    if x.dtype != torch.bfloat16 or z is None or this.weight.dtype != x.dtype:
                        raise ValueError("expected BF16 gated RMSNorm operands")
                    proxy = SimpleNamespace(weight=this.weight, variance_epsilon=this.eps)
                    return hf.Qwen3_5RMSNormGated.forward(proxy, x, z)

                patched = patch_gated_norm_paths(
                    model, RMSNormGated, gdn.QwenGatedDeltaNetAttention, patch, gated
                )
            if "hf_chunk_core" in active:

                def chunk(
                    this,
                    q,
                    k,
                    v,
                    g,
                    beta,
                    initial_state,
                    output_final_state,
                    cu_seqlens=None,
                    chunk_indices=None,
                    chunk_offsets=None,
                    use_qk_l2norm_in_kernel=True,
                    core_attn_out=None,
                ):
                    count("hf_chunk_core")
                    if (
                        use_qk_l2norm_in_kernel
                        or not output_final_state
                        or initial_state is None
                        or cu_seqlens is None
                        or q.dtype != torch.bfloat16
                        or k.dtype != q.dtype
                        or v.dtype != q.dtype
                        or initial_state.dtype != torch.float32
                        or g.dtype != torch.float32
                        or beta.dtype != torch.float32
                    ):
                        raise ValueError("unsupported HF chunk counterfactual operands")
                    repeats = check_chunk_shapes(
                        tuple(q.shape),
                        tuple(k.shape),
                        tuple(v.shape),
                        tuple(g.shape),
                        tuple(beta.shape),
                        tuple(initial_state.shape),
                        cu_seqlens.cpu().tolist(),
                    )
                    if torch.count_nonzero(initial_state).item() != 0:
                        raise ValueError("HF chunk counterfactual requires fresh zero state")
                    if repeats != 1:
                        q = q.repeat_interleave(repeats, dim=2)
                        k = k.repeat_interleave(repeats, dim=2)
                    output, final = hf.torch_chunk_gated_delta_rule(
                        q,
                        k,
                        v,
                        g,
                        beta,
                        chunk_size=64,
                        initial_state=initial_state.transpose(-1, -2).contiguous(),
                        output_final_state=True,
                        use_qk_l2norm_in_kernel=False,
                    )
                    final = final.transpose(-1, -2).contiguous()
                    if core_attn_out is not None:
                        core_attn_out.reshape(-1)[: output.numel()].copy_(output.reshape(-1))
                    return output, final

                for module in model.modules():
                    if isinstance(module, gdn.ChunkGatedDeltaRule):
                        patch(module, "forward", MethodType(chunk, module))
                        patched += 1
            if "hf_residual_norm" in active:
                for module in model.modules():
                    if isinstance(module, GemmaRMSNorm):
                        original = module.forward

                        def residual(this, x, residual=None, _original=original):
                            if residual is None:
                                return _original(x)
                            count("hf_residual_norm")
                            if x.dtype != torch.bfloat16 or residual.dtype != x.dtype:
                                raise ValueError("expected BF16 residual operands")
                            summed = x + residual
                            return _original(summed), summed

                        patch(module, "forward", MethodType(residual, module))
                        patched += 1
            if active & {"hf_qk_l2norm", "hf_beta", "hf_decay_g"}:
                original = gdn.fused_post_conv_prep
                signature = inspect.signature(original)

                def preparation(*args, **kwargs):
                    bound = signature.bind(*args, **kwargs)
                    bound.apply_defaults()
                    values = bound.arguments
                    if values["conv_output"].stride(-1) != 1:
                        raise ValueError("GDN preparation requires unit feature stride")
                    if not values["apply_l2norm"] or values["output_g_exp"]:
                        raise ValueError("unsupported prefill preparation variant")
                    if (
                        values["conv_output"].dtype != torch.bfloat16
                        or values["b"].dtype != torch.bfloat16
                    ):
                        raise ValueError("expected BF16 prefill operands")
                    if "hf_qk_l2norm" in active:
                        values["apply_l2norm"] = False
                    q, k, v, g, beta = original(**values)
                    if "hf_qk_l2norm" in active:
                        count("hf_qk_l2norm")
                        q = hf.l2norm(q, dim=-1, eps=1e-6).contiguous()
                        k = hf.l2norm(k, dim=-1, eps=1e-6).contiguous()
                    if "hf_beta" in active:
                        count("hf_beta")
                        # HF GDN forward uses b.sigmoid() before its FP32 recurrence.
                        beta = values["b"].sigmoid().float().contiguous()
                    if "hf_decay_g" in active:
                        count("hf_decay_g")
                        if values["a"].dtype != torch.bfloat16:
                            raise ValueError("HF decay trial requires BF16 projection output")
                        g = (
                            -values["A_log"].float().exp()
                            * torch.nn.functional.softplus(
                                values["a"].float() + values["dt_bias"].to(torch.bfloat16)
                            )
                        ).contiguous()
                    return q, k, v, g, beta

                patch(gdn, "fused_post_conv_prep", preparation)
                patched += 1
            if "hf_z_projection" in active:
                for module in model.modules():
                    if not isinstance(module, gdn.QwenGatedDeltaNetAttention):
                        continue
                    projection = module.in_proj_qkvz
                    start, end = z_weight_slice(
                        module.key_dim,
                        module.value_dim,
                        module.hidden_size,
                        projection.weight.shape,
                        projection.output_sizes,
                        module.tp_size,
                        module.gqa_interleaved_layout,
                        projection.bias,
                    )
                    native_projection = projection.forward
                    # These weights were byte-checked against the pinned HF key above.
                    z_weights = projection.weight[start:end].detach().contiguous()

                    def z_project(
                        this,
                        x,
                        _native=native_projection,
                        _start=start,
                        _end=end,
                        _weights=z_weights,
                    ):
                        count("hf_z_projection")
                        if x.ndim != 2 or x.dtype != torch.bfloat16:
                            raise ValueError("expected unpadded BF16 projection input")
                        output, bias = _native(x)
                        if (
                            bias is not None
                            or output.dtype != x.dtype
                            or tuple(output.shape) != (x.shape[0], _end)
                        ):
                            raise ValueError("unexpected native packed projection output")
                        z = torch.nn.functional.linear(x.unsqueeze(0), _weights).squeeze(0)
                        replaced = output.clone()
                        replaced[:, _start:_end].copy_(z)
                        return replaced, bias

                    patch(projection, "forward", MethodType(z_project, projection))
                    patched += 1
            if active & {"hf_conv_materialize", "hf_conv_core"}:
                conv_original = gdn.causal_conv1d_fn
                conv_signature = inspect.signature(conv_original)
                conv_mode = (
                    "hf_conv_materialize" if "hf_conv_materialize" in active else "hf_conv_core"
                )

                def convolution(*args, **kwargs):
                    count(conv_mode)
                    bound = conv_signature.bind(*args, **kwargs)
                    bound.apply_defaults()
                    values = bound.arguments
                    x, weight, cache = values["x"], values["weight"], values["conv_states"]
                    if (
                        values["activation"] not in ("silu", "swish")
                        or x.dtype != torch.bfloat16
                        or weight.dtype != x.dtype
                        or cache is None
                        or cache.dtype != x.dtype
                        or values["has_initial_state"] is None
                        or values["query_start_loc"] is None
                        or values["cache_indices"] is None
                    ):
                        raise ValueError("expected BF16 fresh-prefill convolution operands")
                    slots = values["cache_indices"].detach().cpu().tolist()
                    check_conv_shapes(
                        tuple(x.shape),
                        tuple(weight.shape),
                        tuple(cache.shape),
                        values["has_initial_state"].detach().cpu().tolist(),
                        values["query_start_loc"].detach().cpu().tolist(),
                        slots,
                    )
                    if slots[0] in (values["pad_slot_id"], values["null_block_id"]):
                        raise ValueError("convolution trial cannot use padded/null cache slot")
                    if values["bias"] is not None and values["bias"].dtype != x.dtype:
                        raise ValueError("convolution bias must match BF16 operands")
                    if conv_mode == "hf_conv_materialize":
                        values["activation"] = None
                    # Always execute native state updates exactly once. Only returned
                    # activations change; no cached continuation is accepted by guards.
                    native = conv_original(**values)
                    if native.dtype != x.dtype or native.shape != x.shape:
                        raise ValueError("native convolution output shape/dtype differs")
                    if conv_mode == "hf_conv_materialize":
                        return preserve_conv_layout(native, torch.nn.functional.silu(native))
                    convolved = torch.nn.functional.conv1d(
                        x.unsqueeze(0),
                        weight.unsqueeze(1),
                        values["bias"],
                        padding=weight.shape[1] - 1,
                        groups=x.shape[0],
                    )
                    activated = torch.nn.functional.silu(convolved[:, :, : x.shape[1]]).squeeze(0)
                    return preserve_conv_layout(native, activated)

                patch(gdn, "causal_conv1d_fn", convolution)
                patched += 1
            if not patched and mode != "baseline":
                raise ValueError("no matching arithmetic targets found")
            source = Path(inspect.getfile(hf))
            state["metadata"] = {
                "mode": mode,
                "engine_profile": dict(ENGINE_PROFILE),
                "implementation_sha256": source_hashes,
                "factors": sorted(active - {"baseline"}),
                "checkpoint_weight_verification": weight_proofs,
                "decay_weight_verification": decay_proofs,
                "patched_targets": patched,
                "hf_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "scope": trial_scope(mode),
            }
            return state["metadata"]
        except BaseException:
            self.jpt_arithmetic_remove()
            raise

    def jpt_arithmetic_remove(self):
        state = self._jpt_arithmetic
        for owner, name, original, had_own in reversed(state["undo"]):
            if had_own:
                setattr(owner, name, original)
            else:
                delattr(owner, name)
        if "guard" in state:
            state["guard"].remove()
        result = {
            "removed": True,
            "calls": state["calls"],
            "forwards": state["forwards"],
            "calls_per_forward": state["calls_per_forward"],
            "factor_calls_per_forward": state["factor_calls_per_forward"],
        }
        del self._jpt_arithmetic
        return result


class ArithmeticRuntime(HiddenRuntime):
    def __init__(self, *args, arithmetic_mode, **kwargs):
        super().__init__(*args, **kwargs)
        self.arithmetic_mode = validate_mode(arithmetic_mode)
        self.arithmetic_metadata = None
        self.arithmetic_restored = None

    def _create_engine(self, path):
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM

        return AsyncLLM.from_engine_args(
            AsyncEngineArgs(
                model=str(path),
                **ENGINE_PROFILE,
                runner="generate",
                trust_remote_code=False,
                gpu_memory_utilization=self.gpu_memory_utilization,
                cpu_offload_gb=0,
                max_logprobs=255,
                logprobs_mode="processed_logits",
                generation_config="vllm",
                limit_mm_per_prompt={"image": 0, "video": 0},
                seed=0,
                enable_log_requests=False,
                disable_log_stats=True,
                worker_extension_cls="validation.jpt_arithmetic_probe.ArithmeticProbeWorker",
            )
        )

    async def start(self, model):
        await super().start(model)
        self.arithmetic_metadata = await rpc(self, "jpt_arithmetic_install", self.arithmetic_mode)

    async def close(self):
        try:
            if self.arithmetic_metadata is not None and self.healthy():
                self.arithmetic_restored = await rpc(self, "jpt_arithmetic_remove")
                self.arithmetic_metadata = dict(
                    self.arithmetic_metadata, **self.arithmetic_restored
                )
        finally:
            await super().close()


def main():
    # Reuse the established provenance/capture/weight checks and CLI unchanged.
    # Factory injection is process-local, restored even on failure, never production state.
    import argparse

    from validation import jpt_hidden_probe as hidden

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--arithmetic-mode", choices=MODES, required=True)
    options, remaining = parser.parse_known_args()
    if not remaining or remaining[0] != "runtime":
        parser.error("arithmetic probe supports runtime mode only")
    if "--help" in remaining or "-h" in remaining:
        previous_argv = sys.argv
        try:
            sys.argv = [previous_argv[0], *remaining]
            hidden.main()
        finally:
            sys.argv = previous_argv
        return
    instances = []

    def factory(*args, **kwargs):
        instance = ArithmeticRuntime(*args, arithmetic_mode=options.arithmetic_mode, **kwargs)
        instances.append(instance)
        return instance

    output_parser = argparse.ArgumentParser(add_help=False)
    output_parser.add_argument("--output", type=Path, required=True)
    output, without_output = output_parser.parse_known_args(remaining)
    if output.output.exists() or output.output.is_symlink():
        parser.error("output already exists; choose a new report path")
    output.output.parent.mkdir(parents=True, exist_ok=True)
    old_runtime, old_argv = hidden.HiddenRuntime, sys.argv
    with tempfile.TemporaryDirectory(prefix=".arithmetic-", dir=output.output.parent) as staging:
        intermediate = Path(staging) / "hidden.json"
        try:
            hidden.HiddenRuntime = factory
            sys.argv = [old_argv[0], *without_output, "--output", str(intermediate)]
            hidden.main()
        finally:
            hidden.HiddenRuntime, sys.argv = old_runtime, old_argv
        report = json.loads(intermediate.read_text())
    metadata = instances[0].arithmetic_metadata
    valid = validate_trial(metadata, report["summary"]["questions"], options.arithmetic_mode)
    report.update(
        mode="runtime_arithmetic",
        arithmetic_mode=options.arithmetic_mode,
        arithmetic=metadata,
        valid_trial=valid,
        scope=trial_scope(options.arithmetic_mode),
    )
    with output.output.open("x") as destination:
        destination.write(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    if not valid:
        raise RuntimeError(
            "arithmetic trial coverage/restoration failed; report marked valid_trial=false"
        )
    print("Saved arithmetic diagnostic", output.output)


if __name__ == "__main__":
    main()
