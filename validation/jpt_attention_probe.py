"""Actual layer-3 attention boundaries: original HF versus eight-factor runtime.

Fixed original case 0, row 0; no attention arithmetic replacement or certification.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import sys
from pathlib import Path

from validation import jpt_gdn_probe as gdn_probe
from validation import jpt_layer_probe as layers
from validation.jpt_arithmetic_probe import (
    ENGINE_PROFILE,
    ArithmeticProbeWorker,
    ArithmeticRuntime,
    validate_trial,
)
from validation.jpt_hidden_probe import candidate_weights, rpc

ARITHMETIC_MODE = "hf_torch_decay_combined"
NAMES = (
    "attention_input",
    "projected_q",
    "projected_k",
    "projected_v",
    "gate",
    "postrope_q",
    "postrope_k",
    "postrope_v",
    "attention_output",
    "gated",
    "out_projection",
)


def select_attention_fixture(reference):
    case, row = layers.select_fixture(reference, 0, 0)
    if len(row["input_ids"]) <= 1:
        raise ValueError("attention probe requires multi-token prefill")
    return case, row


def finalize_arithmetic(report, metadata):
    if not validate_trial(metadata, 1, ARITHMETIC_MODE):
        raise ValueError("eight-factor coverage/restoration failed")
    report.update(
        arithmetic=metadata,
        arithmetic_mode=ARITHMETIC_MODE,
        valid_trial=True,
        certified=False,
        production_changed=False,
    )


class AttentionCapture(gdn_probe.Capture):
    def __init__(self, tokens):
        super().__init__(tokens)
        self.metadata = {}
        self.auxiliary = {}

    def finish(self):
        if set(self.values) != set(NAMES):
            raise ValueError("missing actual attention boundaries")
        return {name: gdn_probe.encode_tensor(self.values[name], self.tokens) for name in NAMES}

    def pre_hook(self, module, name):
        def save(_module, inputs):
            self.add(name, inputs[0])

        self.handles.append(module.register_forward_pre_hook(save))

    def auxiliary_report(self):
        return {
            name: gdn_probe.encode_tensor(value, self.tokens)
            for name, value in self.auxiliary.items()
        }


def hf_install(body, capture):
    import torch
    from transformers.models.qwen3_5 import modeling_qwen3_5 as hf

    module = body.layers[3].self_attn
    if body.layers[3].layer_type != "full_attention":
        raise ValueError("expected first full-attention layer at index 3")
    original_interface = hf.ALL_ATTENTION_FUNCTIONS.get_interface
    capture.metadata = {
        "layer_index": 3,
        "hf_attention_implementation": module.config._attn_implementation,
        "source_sha256": gdn_probe.source_hash(hf),
    }

    def q_projection(_module, _inputs, output):
        q, gate = output.reshape(capture.tokens, -1, module.head_dim * 2).chunk(2, dim=-1)
        capture.add("projected_q", q)
        capture.add("gate", gate)

    capture.handles.append(module.q_proj.register_forward_hook(q_projection))
    capture.pre_hook(module.q_proj, "attention_input")
    capture.hook(module.k_proj, "projected_k")
    capture.hook(module.v_proj, "projected_v")
    capture.pre_hook(module.o_proj, "gated")
    capture.hook(module.o_proj, "out_projection")

    def rotary(_module, inputs, output):
        if "rope_cos" in capture.auxiliary:
            raise ValueError("multiple rotary embedding calls")
        capture.metadata["position_ids"] = inputs[1].detach().cpu().tolist()
        capture.auxiliary["rope_cos"] = output[0].detach().clone()
        capture.auxiliary["rope_sin"] = output[1].detach().clone()

    capture.handles.append(body.rotary_emb.register_forward_hook(rotary))

    def get_interface(*args, **kwargs):
        original = original_interface(*args, **kwargs)

        def attention(current, query, key, value, mask, *a, **kw):
            if current is not module:
                return original(current, query, key, value, mask, *a, **kw)
            if any(
                x.ndim != 4 or x.shape[0] != 1 or x.shape[2] != capture.tokens
                for x in (query, key, value)
            ):
                raise ValueError("expected actual HF [1,heads,tokens,dim] attention inputs")
            capture.metadata.update(
                interface_module=original.__module__,
                interface_name=original.__qualname__,
                interface_source_sha256=gdn_probe.source_hash(original),
                attention_mask=None
                if mask is None
                else {
                    "shape": list(mask.shape),
                    "dtype": str(mask.dtype),
                    "sha256": hashlib.sha256(
                        mask.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
                    ).hexdigest(),
                },
                scaling=kw.get("scaling"),
                is_causal=current.is_causal,
                backend_scope="selected HF interface; CUDA kernel dispatch not profiled",
            )
            for name, tensor in [("postrope_q", query), ("postrope_k", key), ("postrope_v", value)]:
                capture.add(name, tensor.transpose(1, 2))
            result = original(current, query, key, value, mask, *a, **kw)
            if result[0].ndim != 4 or list(result[0].shape[:2]) != [1, capture.tokens]:
                raise ValueError("expected HF attention output [1,tokens,heads,dim]")
            capture.add("attention_output", result[0])
            return result

        return attention

    capture.patch(hf.ALL_ATTENTION_FUNCTIONS, "get_interface", get_interface)


def runtime_install(body, capture):
    from vllm.model_executor.models import qwen3_next

    module = body.layers[3].self_attn
    if (
        body.layers[3].layer_type != "full_attention"
        or not module.attn_output_gate
        or module.qkv_proj.tp_size != 1
    ):
        raise ValueError("requires single-worker gated full attention at layer 3")
    capture.metadata = {
        "layer_index": 3,
        "source_sha256": gdn_probe.source_hash(qwen3_next),
        "fused_qk_norm_rope_gate": module.use_fused_qk_norm_rope_gate,
        "attention_impl_module": type(module.attn.impl).__module__,
        "attention_impl_class": type(module.attn.impl).__qualname__,
        "attention_backend": module.attn.backend.name,
        "scaling": module.scaling,
    }
    original_project = module._project_qkv_gate

    def project(qkv, positions):
        if qkv.shape[0] != capture.tokens:
            raise ValueError("unpadded full-sequence attention required")
        q_gate, k, v = qkv.split([2 * module.q_size, module.kv_size, module.kv_size], dim=-1)
        q, gate = q_gate.reshape(capture.tokens, module.num_heads, 2 * module.head_dim).chunk(
            2, dim=-1
        )
        for name, tensor in [
            ("projected_q", q),
            ("projected_k", k),
            ("projected_v", v),
            ("gate", gate),
        ]:
            capture.add(name, tensor)
        capture.metadata["position_ids"] = positions.detach().cpu().tolist()
        flat_positions = positions if positions.ndim == 1 else positions[0]
        if positions.ndim == 2:
            import torch

            if not torch.equal(positions, flat_positions.unsqueeze(0).expand_as(positions)):
                raise ValueError("only equal text position streams supported")
        capture.metadata["rope_cache_layout"] = (
            "selected [cos_half,sin_half] rows before fused/eager application"
        )
        capture.auxiliary["rope_cache_rows"] = (
            module.rotary_emb.cos_sin_cache[flat_positions].detach().clone()
        )
        result = original_project(qkv, positions)
        for name, tensor in zip(
            ("postrope_q", "postrope_k", "postrope_v"), result[:3], strict=True
        ):
            capture.add(name, tensor)
        return result

    capture.patch(module, "_project_qkv_gate", project)
    capture.pre_hook(module.qkv_proj, "attention_input")
    capture.hook(module.attn, "attention_output")
    capture.pre_hook(module.o_proj, "gated")
    capture.hook(module.o_proj, "out_projection")


class AttentionProbeWorker(ArithmeticProbeWorker):
    def jpt_attention_install(self, tokens):
        if hasattr(self, "_jpt_attention"):
            raise ValueError("attention probe already installed")
        capture = AttentionCapture(tokens)
        self._jpt_attention = capture
        try:
            self.jpt_hidden_install()
            runtime_install(self.get_model().language_model.model, capture)
            return {"installed": True}
        except BaseException:
            capture.close()
            if hasattr(self, "_jpt_hidden_state"):
                self.jpt_hidden_remove()
            del self._jpt_attention
            raise

    def jpt_attention_take(self):
        capture = self._jpt_attention
        return {
            **self.jpt_hidden_take(),
            "boundaries": capture.finish(),
            "operation_metadata": capture.metadata,
            "rotary_operands": capture.auxiliary_report(),
        }

    def jpt_attention_remove(self):
        self._jpt_attention.close()
        del self._jpt_attention
        return self.jpt_hidden_remove()


class AttentionRuntime(ArithmeticRuntime):
    def _create_engine(self, path):
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM

        return AsyncLLM.from_engine_args(
            AsyncEngineArgs(
                model=str(path),
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
                worker_extension_cls="validation.jpt_attention_probe.AttentionProbeWorker",
                **ENGINE_PROFILE,
            )
        )


def hf_stage(model, case, row, report):
    import torch
    from llm2jev.backends import HF

    from jevserve.contracts.requests import DecisionRequest

    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("real CUDA required")
    backend = HF(str(model.path), device="cuda", dtype="bfloat16")
    layers.check_encoding(
        layers.adapter(backend.tok, report["temperature"]).encode(
            DecisionRequest.model_validate(case["request"])
        ),
        case["rows"],
    )
    capture = AttentionCapture(len(row["input_ids"]))
    try:
        hf_install(backend.model.model.language_model, capture)
        with torch.no_grad(), backend.lock:
            result = backend.model(
                input_ids=torch.tensor([row["input_ids"]], device=backend.model.device)
            )
            _, metadata = candidate_weights(
                backend.model.get_output_embeddings(), row["candidate_token_ids"]
            )
            report["capture"] = {
                **metadata,
                "boundaries": capture.finish(),
                "original_logits": result.logits[0, -1, row["candidate_token_ids"]]
                .float()
                .cpu()
                .tolist(),
                "operation_metadata": capture.metadata,
                "rotary_operands": capture.auxiliary_report(),
            }
            report["frozen_logit_comparison"] = gdn_probe.frozen_logit_check(
                report["capture"]["original_logits"], row["raw_logits"]
            )
    finally:
        capture.close()


async def runtime_stage(args, model, case, row, expected, report):
    from dataclasses import replace

    from transformers import AutoTokenizer

    from jevserve.contracts.requests import DecisionRequest

    native = layers.adapter(
        AutoTokenizer.from_pretrained(model.path, local_files_only=True, trust_remote_code=False),
        report["temperature"],
    )
    encoded = native.encode(DecisionRequest.model_validate(case["request"]))
    layers.check_encoding(encoded, case["rows"])
    single = replace(
        encoded,
        question_ids=encoded.question_ids[:1],
        engine_inputs=encoded.engine_inputs[:1],
        groups=encoded.groups[:1],
        candidate_token_ids=encoded.candidate_token_ids[:1],
        readout_positions=encoded.readout_positions[:1],
        input_tokens=len(row["input_ids"]),
        candidate_count=len(row["candidate_token_ids"]),
    )
    runtime = AttentionRuntime(
        args.cache_dir,
        gpu_memory_utilization=args.gpu_memory_utilization,
        arithmetic_mode=ARITHMETIC_MODE,
    )
    installed = False
    try:
        await runtime.start(model)
        await rpc(runtime, "jpt_attention_install", len(row["input_ids"]))
        installed = True
        await rpc(
            runtime,
            "jpt_hidden_arm",
            "attention-fixture",
            row["candidate_token_ids"],
            expected["weight_sha256"],
        )
        result = await runtime.infer(
            single, "attention-fixture", layers.asyncio.get_running_loop().time() + args.timeout
        )
        capture = await rpc(runtime, "jpt_attention_take")
        for key in ("weight_sha256", "weight_shape", "weight_dtype"):
            if capture[key] != expected[key]:
                raise ValueError("candidate weight identity differs")
        if list(result.candidate_logits[0]) != capture["original_logits"]:
            raise ValueError("runtime and captured head logits differ")
        report["capture"] = capture
        report["comparison"] = gdn_probe.compare_captures(
            capture["boundaries"], expected["boundaries"]
        )
    finally:
        try:
            if installed and runtime.healthy():
                await rpc(runtime, "jpt_attention_remove")
        finally:
            await runtime.close()
    finalize_arithmetic(report, runtime.arithmetic_metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["hf", "runtime"])
    parser.add_argument("--model", required=True)
    for name in ("model-dir", "upstream", "reference", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--attention-reference", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=Path(".local/jpt-attention-cache"))
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error("output exists")
    if args.mode == "runtime" and args.attention_reference is None:
        parser.error("runtime requires --attention-reference")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout must be finite and positive")
    if not math.isfinite(args.gpu_memory_utilization) or not 0 < args.gpu_memory_utilization < 1:
        parser.error("GPU memory utilization must lie between zero and one")
    if os.environ.get("VLLM_ALLOW_INSECURE_SERIALIZATION", "0") != "0":
        parser.error("insecure serialization forbidden")
    selected = layers.profile(args.model)
    if selected["adapter"] != "jpt":
        parser.error("only pinned JPT profiles supported")
    layers.verify_upstream(args.upstream, selected["upstream_code"]["revision"])
    model = layers.load_model_directory(args.model_dir, layers.source_manifest(args.model))
    raw = args.reference.read_bytes()
    reference = json.loads(raw)
    layers.verify_reference(reference, selected, model.manifest.identity, "reference_logits")
    case, row = select_attention_fixture(reference)
    pins = {"torch": "2.13.0", "transformers": "5.10.4"}
    if args.mode == "runtime":
        pins["vllm"] = "0.29.0"
    packages = {name: importlib.metadata.version(name) for name in pins}
    if any(packages[name].split("+")[0] != version for name, version in pins.items()):
        raise ValueError("requires pinned runtime packages")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    sys.path.insert(0, str(args.upstream.resolve()))
    report = {
        "schema_version": 1,
        "mode": args.mode + "_attention",
        "model_id": args.model,
        "revision": selected["revision"],
        "source_identity": model.manifest.identity,
        "upstream_revision": selected["upstream_code"]["revision"],
        "temperature": selected["temperature"],
        "reference_sha256": hashlib.sha256(raw).hexdigest(),
        "packages": packages,
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "selection": {"case_index": 0, "row_index": 0, "case": case["case"], "layer_index": 3},
        "fixture": {key: row[key] for key in layers.INPUT_KEYS},
        "scope": "one synthetic fixture actual full-attention boundaries; runtime uses eight-factor counterfactual; no certification",
        "insecure_serialization": False,
        "certified": False,
        "production_changed": False,
        "schedule": "sequential single fresh unpadded question",
        "runtime_profile": dict(ENGINE_PROFILE),
    }
    if args.mode == "hf":
        hf_stage(model, case, row, report)
    else:
        attention_bytes = args.attention_reference.read_bytes()
        expected = json.loads(attention_bytes)
        for key in (
            "model_id",
            "revision",
            "source_identity",
            "upstream_revision",
            "temperature",
            "reference_sha256",
            "selection",
            "fixture",
        ):
            if expected.get(key) != report[key]:
                raise ValueError("attention reference provenance/fixture differs")
        if (
            expected.get("mode") != "hf_attention"
            or list(expected["capture"]["boundaries"]) != list(NAMES)
            or not expected.get("frozen_logit_comparison", {}).get("exact_match")
        ):
            raise ValueError("expected complete frozen-logit-matching HF attention report")
        gdn_probe.compare_captures(
            expected["capture"]["boundaries"], expected["capture"]["boundaries"]
        )
        report["attention_reference_sha256"] = hashlib.sha256(attention_bytes).hexdigest()
        layers.asyncio.run(runtime_stage(args, model, case, row, expected["capture"], report))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(report, output, ensure_ascii=False, allow_nan=False)
        output.write("\n")
    print("Saved attention diagnostic", args.output)


if __name__ == "__main__":
    main()
