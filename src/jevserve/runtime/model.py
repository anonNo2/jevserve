"""Text-only scalar-head model; derived from mode-io/vllm-jev's Apache-2.0 adapter.

Only vLLM workers import this module. The actual scheduler/backbone/pooler are vLLM.
"""

import torch
from vllm.model_executor.models.adapters import as_seq_cls_model
from vllm.model_executor.models.interfaces import IsHybrid, SupportsMRoPE
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration


class JevQwen35CausalLM(Qwen3_5ForCausalLM, IsHybrid, SupportsMRoPE):
    is_hybrid = True
    supports_mrope = True

    def get_mrope_input_positions(self, input_tokens, mm_features):
        if mm_features:
            raise ValueError("JevServe supports text only")
        positions = torch.arange(len(input_tokens), dtype=torch.long)
        return positions.unsqueeze(0).expand(3, -1).contiguous(), 0

    @classmethod
    def get_mamba_state_dtype_from_config(cls, config):
        return Qwen3_5ForConditionalGeneration.get_mamba_state_dtype_from_config(config)

    @classmethod
    def get_mamba_state_shape_from_config(cls, config):
        return Qwen3_5ForConditionalGeneration.get_mamba_state_shape_from_config(config)

    @classmethod
    def get_mamba_state_copy_func(cls):
        return Qwen3_5ForConditionalGeneration.get_mamba_state_copy_func()


class _NativeClassifier(as_seq_cls_model(JevQwen35CausalLM)):
    """Final hidden state and FP32 scalar head; activations applied in the adapter."""


class FixedFP32LoRA(torch.nn.Module):
    """One fixed adapter; PEFT-equivalent FP32 residuals, with no merged weights."""

    def __init__(self, base, parts, scale):
        super().__init__()
        self.base = base
        self.scale = scale
        self.slices = []
        for i, (start, a, b) in enumerate(parts):
            self.register_buffer(f"a{i}", a.to(device=base.weight.device, dtype=torch.float32))
            self.register_buffer(f"b{i}", b.to(device=base.weight.device, dtype=torch.float32))
            self.slices.append((start, start + b.shape[0]))

    def forward(self, x):
        output, bias = self.base(x)
        work = x.float()
        for i, (start, end) in enumerate(self.slices):
            delta = (
                torch.nn.functional.linear(
                    torch.nn.functional.linear(work, getattr(self, f"a{i}")), getattr(self, f"b{i}")
                )
                * self.scale
            )
            output[..., start:end] = (output[..., start:end].float() + delta).to(output.dtype)
        return output, bias


# The base class remains native vLLM; this adds only the fixed LoRA residual path.


class JevServeQwen35ForSequenceClassification(_NativeClassifier):
    def __init__(self, *, vllm_config, prefix=""):
        if vllm_config.parallel_config.tensor_parallel_size != 1:
            raise ValueError("JevServe alpha requires one GPU")
        if vllm_config.lora_config is not None or vllm_config.quant_config is not None:
            raise ValueError("fixed FP32 adapter does not support dynamic LoRA or quantization")
        self._checkpoint_path = vllm_config.model_config.model
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    def load_weights(self, weights):
        import json
        from pathlib import Path

        from safetensors.torch import load_file

        loaded = super().load_weights(weights)
        path = Path(self._checkpoint_path) / "lora"
        config = json.loads((path / "adapter_config.json").read_text())
        scale = config["lora_alpha"] / config["r"]
        if config["r"] != 8 or scale != 2 or config["lora_dropout"] != 0:
            raise ValueError("unexpected Open-Jev LoRA configuration")
        values = load_file(path / "adapter_model.safetensors", device="cpu")
        pairs = {}
        for name, value in values.items():
            if value.dtype != torch.float32:
                raise ValueError("LoRA weights must remain FP32")
            name = name.removeprefix("base_model.model.")
            module, side = name.rsplit(".lora_", 1)
            pairs.setdefault(module, {})[side] = value
        groups = {}
        for name, pair in pairs.items():
            parent, leaf = name.rsplit(".", 1)
            target = (
                "qkv_proj"
                if leaf in ("q_proj", "k_proj", "v_proj")
                else "in_proj_qkvz"
                if leaf == "in_proj_qkv"
                else leaf
            )
            groups.setdefault(parent + "." + target, {})[leaf] = pair
        for name, parts in groups.items():
            module = self.get_submodule(name)
            ordered = []
            if name.endswith(".qkv_proj"):
                if set(parts) != {"q_proj", "k_proj", "v_proj"}:
                    raise ValueError("incomplete QKV LoRA")
                ordered = [parts[key] for key in ("q_proj", "k_proj", "v_proj")]
            else:
                if len(parts) != 1:
                    raise ValueError("ambiguous LoRA mapping")
                ordered = list(parts.values())
            offset = 0
            mapped = []
            for pair in ordered:
                a, b = pair["A.weight"], pair["B.weight"]
                if b.shape[1] != a.shape[0] or a.shape[1] != module.weight.shape[1]:
                    raise ValueError("LoRA dimensions mismatch")
                mapped.append((offset, a, b))
                offset += b.shape[0]
            if offset > module.weight.shape[0]:
                raise ValueError("LoRA output slice exceeds projection")
            parent, leaf = name.rsplit(".", 1)
            setattr(self.get_submodule(parent), leaf, FixedFP32LoRA(module, mapped, scale))
            if loaded is not None:
                loaded = {
                    name + ".base." + key[len(name) + 1 :] if key.startswith(name + ".") else key
                    for key in loaded
                }
        return loaded
