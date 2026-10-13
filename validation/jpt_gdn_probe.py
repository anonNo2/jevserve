"""First-layer actual GDN boundaries on one synthetic fixture; no arithmetic changes."""

import base64
import hashlib
import inspect
import json
import math
import struct
import sys
from pathlib import Path

from validation import jpt_layer_probe as layers
from validation.jpt_hidden_probe import HiddenProbeWorker, candidate_weights, hidden_difference, rpc

NAMES = (
    "projected_qkv",
    "projected_z",
    "projected_b",
    "projected_a",
    "postconv_q",
    "postconv_k",
    "postconv_v",
    "normalized_q",
    "normalized_k",
    "g",
    "beta",
    "recurrence",
    "gated",
    "out_projection",
)


def decode_tensor(record):
    shape, dtype = record["shape"], record["dtype"]
    if (
        dtype not in ("torch.bfloat16", "torch.float32")
        or len(shape) != 2
        or any(type(v) is not int or v <= 0 for v in shape)
        or shape[0] > layers.MAX_TOKENS
        or math.prod(shape) > layers.MAX_ELEMENTS
    ):
        raise ValueError("invalid typed capture shape/dtype")
    itemsize = 2 if dtype == "torch.bfloat16" else 4
    if len(record["data_base64"]) > 4 * ((math.prod(shape) * itemsize + 2) // 3):
        raise ValueError("capture exceeds shape bound")
    try:
        raw = base64.b64decode(record["data_base64"], validate=True)
    except Exception as exc:
        raise ValueError("invalid capture encoding") from exc
    if (
        len(raw) != math.prod(shape) * itemsize
        or hashlib.sha256(raw).hexdigest() != record["sha256"]
    ):
        raise ValueError("capture shape/hash differs")
    if itemsize == 2:
        values = [
            struct.unpack("<f", struct.pack("<I", w[0] << 16))[0]
            for w in struct.iter_unpack("<H", raw)
        ]
    else:
        values = [v[0] for v in struct.iter_unpack("<f", raw)]
    if not all(math.isfinite(v) for v in values):
        raise ValueError("nonfinite capture")
    return values


def encode_tensor(tensor, tokens):
    import torch

    if (
        tensor.dtype not in (torch.bfloat16, torch.float32)
        or sys.byteorder != "little"
        or tensor.numel() > layers.MAX_ELEMENTS
        or tensor.numel() % tokens
    ):
        raise ValueError("invalid bounded activation")
    matrix = tensor.detach().reshape(tokens, -1).contiguous().cpu()
    if not bool(torch.isfinite(matrix).all()):
        raise ValueError("nonfinite activation")
    raw = matrix.view(torch.uint8).numpy().tobytes()
    return {
        "shape": list(matrix.shape),
        "dtype": str(matrix.dtype),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "data_base64": base64.b64encode(raw).decode("ascii"),
    }


def compare_captures(actual, expected):
    if list(actual) != list(expected) or not 1 <= len(actual) <= len(NAMES):
        raise ValueError("capture names/order differ")
    results = []
    for name in expected:
        a, e = actual[name], expected[name]
        if a["shape"] != e["shape"]:
            raise ValueError("capture shapes differ")
        av, ev = decode_tensor(a), decode_tensor(e)
        width = a["shape"][1]
        results.append(
            {
                "boundary": name,
                "actual_dtype": a["dtype"],
                "expected_dtype": e["dtype"],
                "all_tokens": hidden_difference(av, ev),
                "last_token": hidden_difference(av[-width:], ev[-width:]),
            }
        )
    return {
        "first_unequal_boundary": next(
            (r["boundary"] for r in results if r["all_tokens"]["unequal_elements"]), None
        ),
        "boundaries": results,
    }


class Capture:
    def __init__(self, tokens):
        if not 1 < tokens <= layers.MAX_TOKENS:
            raise ValueError("requires bounded multi-token fixture")
        self.tokens = tokens
        self.values = {}
        self.handles = []
        self.undo = []
        self.active = False

    def add(self, name, tensor):
        if name in self.values:
            raise ValueError("duplicate GDN boundary")
        if tensor.numel() > layers.MAX_ELEMENTS:
            raise ValueError("capture exceeds element limit")
        self.values[name] = tensor.detach().clone()

    def patch(self, owner, name, fn):
        self.undo.append((owner, name, getattr(owner, name), name in vars(owner)))
        setattr(owner, name, fn)

    def hook(self, module, name):
        def save(_module, _inputs, output):
            self.add(name, output[0] if isinstance(output, tuple) else output)

        self.handles.append(module.register_forward_hook(save))

    def finish(self):
        if set(self.values) != set(NAMES):
            raise ValueError(
                "missing internal GDN captures: " + ",".join(set(NAMES) - set(self.values))
            )
        return {name: encode_tensor(self.values[name], self.tokens) for name in NAMES}

    def close(self):
        for handle in self.handles:
            handle.remove()
        for owner, name, original, own in reversed(self.undo):
            if own:
                setattr(owner, name, original)
            else:
                delattr(owner, name)
        self.values.clear()


def source_hash(owner):
    return hashlib.sha256(Path(inspect.getfile(owner)).read_bytes()).hexdigest()


def frozen_logit_check(actual, expected):
    difference = hidden_difference(actual, expected)
    if difference["unequal_elements"]:
        raise ValueError("passive capture differs from frozen HF logits")
    return {"exact_match": True, "difference": difference}


def hf_install(module, capture):
    from transformers.models.qwen3_5 import modeling_qwen3_5 as hf

    if (
        module.chunk_gated_delta_rule is not hf.torch_chunk_gated_delta_rule
        or type(module.norm) is not hf.Qwen3_5RMSNormGated
    ):
        raise ValueError("GDN capture requires observed HF torch fallback")
    for attr, name in [
        ("in_proj_qkv", "projected_qkv"),
        ("in_proj_z", "projected_z"),
        ("in_proj_b", "projected_b"),
        ("in_proj_a", "projected_a"),
        ("norm", "gated"),
        ("out_proj", "out_projection"),
    ]:
        capture.hook(getattr(module, attr), name)
    original_chunk, original_norm = module.chunk_gated_delta_rule, hf.l2norm
    norm_count = 0

    def normalized(x, *args, **kwargs):
        nonlocal norm_count
        result = original_norm(x, *args, **kwargs)
        if capture.active:
            if norm_count >= 2:
                raise ValueError("unexpected HF normalization count")
            capture.add(("normalized_q", "normalized_k")[norm_count], result)
            norm_count += 1
        return result

    def chunk(query, key, value, **kwargs):
        if kwargs.get("initial_state") is not None or not kwargs.get("use_qk_l2norm_in_kernel"):
            raise ValueError("expected fresh HF prefill with actual in-function normalization")
        for name, value_ in [
            ("postconv_q", query),
            ("postconv_k", key),
            ("postconv_v", value),
            ("g", kwargs["g"]),
            ("beta", kwargs["beta"]),
        ]:
            capture.add(name, value_)
        capture.active = True
        try:
            result = original_chunk(query, key, value, **kwargs)
        finally:
            capture.active = False
        capture.add("recurrence", result[0])
        return result

    capture.patch(hf, "l2norm", normalized)
    capture.patch(module, "chunk_gated_delta_rule", chunk)
    return {
        "backend": "HF torch_chunk_gated_delta_rule",
        "source_sha256": source_hash(hf),
        "qk_layout": "actual HF repeated value-head order",
        "normalized_qk": "actual operations",
    }


def runtime_install(module, capture):
    from vllm.forward_context import get_forward_context
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

    if module.tp_size != 1 or module.gqa_interleaved_layout or not module.enable_fused_gdn_decode:
        raise ValueError("requires pinned single-worker packed Qwen3.5 path")
    ratio = module.num_v_heads // module.num_k_heads
    if ratio < 1 or module.num_v_heads % module.num_k_heads:
        raise ValueError("unsupported head ratio")
    original_core = module._forward_core_fused_norm
    original_prep = gdn.fused_post_conv_prep
    original_gate = module._rms_norm_gated_cuda

    def core(*args, **kwargs):
        bound = inspect.signature(original_core).bind(*args, **kwargs)
        v = bound.arguments
        metadata = get_forward_context().attn_metadata[module.prefix]
        if (
            not isinstance(metadata, GDNAttentionMetadata)
            or metadata.num_prefills != 1
            or metadata.num_decodes
            or metadata.num_spec_decodes
            or metadata.num_prefill_tokens != capture.tokens
            or metadata.num_actual_tokens != capture.tokens
            or metadata.has_initial_state is None
            or metadata.has_initial_state.detach().cpu().tolist() != [False]
        ):
            raise ValueError("requires one fresh unpadded GDN prefill")
        for name, value in [
            ("projected_qkv", v["mixed_qkv"]),
            ("projected_z", v["output_gate"]),
            ("projected_b", v["b"]),
            ("projected_a", v["a"]),
        ]:
            capture.add(name, value)
        capture.active = True
        try:
            return original_core(*args, **kwargs)
        finally:
            capture.active = False

    def prep(*args, **kwargs):
        bound = inspect.signature(original_prep).bind(*args, **kwargs)
        bound.apply_defaults()
        v = bound.arguments
        if capture.active:
            if not v["apply_l2norm"] or v["output_g_exp"]:
                raise ValueError("unsupported GDN preparation")
            qdim, vdim = module.key_dim, module.value_dim
            q, k, value = v["conv_output"].split([qdim, qdim, vdim], dim=-1)
            for name, tensor in [("postconv_q", q), ("postconv_k", k)]:
                capture.add(
                    name,
                    tensor.reshape(
                        capture.tokens, module.num_k_heads, module.head_k_dim
                    ).repeat_interleave(ratio, dim=1),
                )
            capture.add("postconv_v", value)
        result = original_prep(*args, **kwargs)
        if capture.active:
            q, k, value, g, beta = result
            capture.add("normalized_q", q.repeat_interleave(ratio, dim=1))
            capture.add("normalized_k", k.repeat_interleave(ratio, dim=1))
            capture.add("g", g)
            capture.add("beta", beta)
        return result

    def gated(x, output_gate, out):
        # x and out alias on packed prefill: save recurrence BEFORE original.
        capture.add("recurrence", x)
        result = original_gate(x, output_gate, out)
        capture.add("gated", out)
        return result

    capture.patch(module, "_forward_core_fused_norm", core)
    capture.patch(gdn, "fused_post_conv_prep", prep)
    capture.patch(module, "_rms_norm_gated_cuda", gated)
    capture.hook(module.out_proj, "out_projection")
    return {
        "backend": type(module.chunk_gated_delta_rule).__name__,
        "gdn_prefill_backend": module.chunk_gated_delta_rule.gdn_prefill_backend,
        "chunk_dispatch": module.chunk_gated_delta_rule._forward_method.__name__,
        "source_sha256": source_hash(gdn),
        "qk_layout": "vLLM key heads repeated to HF value-head order for comparison only",
        "head_repeat": ratio,
        "normalized_qk": "actual operations",
    }


class GDNProbeWorker(HiddenProbeWorker):
    def jpt_gdn_install(self, tokens):
        if hasattr(self, "_jpt_gdn"):
            raise ValueError("GDN probe already installed")
        model = self.get_model()
        body = model.language_model.model
        capture = Capture(tokens)
        self._jpt_gdn = capture
        try:
            self.jpt_hidden_install()
            return runtime_install(body.layers[0].linear_attn, capture)
        except BaseException:
            capture.close()
            if hasattr(self, "_jpt_hidden_state"):
                self.jpt_hidden_remove()
            del self._jpt_gdn
            raise

    def jpt_gdn_take(self):
        return {**self.jpt_hidden_take(), "boundaries": self._jpt_gdn.finish()}

    def jpt_gdn_remove(self):
        self._jpt_gdn.close()
        del self._jpt_gdn
        return self.jpt_hidden_remove()


class GDNRuntime(layers.LayerRuntime):
    def _create_engine(self, path):
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM

        return AsyncLLM.from_engine_args(
            AsyncEngineArgs(
                model=str(path),
                runner="generate",
                dtype="bfloat16",
                max_model_len=4096,
                trust_remote_code=False,
                gpu_memory_utilization=self.gpu_memory_utilization,
                cpu_offload_gb=0,
                enable_prefix_caching=False,
                mamba_ssm_cache_dtype="float32",
                enforce_eager=True,
                max_num_seqs=8,
                max_num_batched_tokens=4096,
                max_logprobs=255,
                logprobs_mode="processed_logits",
                generation_config="vllm",
                limit_mm_per_prompt={"image": 0, "video": 0},
                seed=0,
                enable_log_requests=False,
                disable_log_stats=True,
                worker_extension_cls="validation.jpt_gdn_probe.GDNProbeWorker",
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
    capture = Capture(len(row["input_ids"]))
    try:
        report["operation_metadata"] = hf_install(
            backend.model.model.language_model.layers[0].linear_attn, capture
        )
        with torch.no_grad(), backend.lock:
            tokens = torch.tensor([row["input_ids"]], device=backend.model.device)
            result = backend.model(input_ids=tokens)
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
            }
            report["frozen_logit_comparison"] = frozen_logit_check(
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
    i = args.row_index
    single = replace(
        encoded,
        question_ids=encoded.question_ids[i : i + 1],
        engine_inputs=encoded.engine_inputs[i : i + 1],
        groups=encoded.groups[i : i + 1],
        candidate_token_ids=encoded.candidate_token_ids[i : i + 1],
        readout_positions=encoded.readout_positions[i : i + 1],
        input_tokens=len(row["input_ids"]),
        candidate_count=len(row["candidate_token_ids"]),
    )
    runtime = GDNRuntime(args.cache_dir, gpu_memory_utilization=args.gpu_memory_utilization)
    installed = False
    try:
        await runtime.start(model)
        report["operation_metadata"] = await rpc(runtime, "jpt_gdn_install", len(row["input_ids"]))
        installed = True
        await rpc(
            runtime,
            "jpt_hidden_arm",
            "gdn-fixture",
            row["candidate_token_ids"],
            expected["weight_sha256"],
        )
        result = await runtime.infer(
            single, "gdn-fixture", layers.asyncio.get_running_loop().time() + args.timeout
        )
        capture = await rpc(runtime, "jpt_gdn_take")
        for key in ("weight_sha256", "weight_shape", "weight_dtype"):
            if capture[key] != expected[key]:
                raise ValueError("candidate weight identity differs")
        if list(result.candidate_logits[0]) != capture["original_logits"]:
            raise ValueError("runtime and head logits differ")
        report["capture"] = capture
        report["comparison"] = compare_captures(capture["boundaries"], expected["boundaries"])
    finally:
        try:
            if installed and runtime.healthy():
                await rpc(runtime, "jpt_gdn_remove")
        finally:
            await runtime.close()


def main():
    import argparse
    import importlib.metadata
    import os

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["hf", "runtime"])
    parser.add_argument("--model", required=True)
    for name in ("model-dir", "upstream", "reference", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--gdn-reference", type=Path)
    parser.add_argument("--case-index", type=int, default=0)
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--cache-dir", type=Path, default=Path(".local/jpt-gdn-cache"))
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error("output exists")
    if args.mode == "runtime" and args.gdn_reference is None:
        parser.error("runtime requires --gdn-reference")
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
    case, row = layers.select_fixture(reference, args.case_index, args.row_index)
    pins = {"torch": "2.13.0", "transformers": "5.10.4"}
    if args.mode == "runtime":
        pins["vllm"] = "0.29.0"
    packages = {name: importlib.metadata.version(name) for name in pins}
    if any(packages[name].split("+")[0] != pin for name, pin in pins.items()):
        raise ValueError("requires pinned runtime packages")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    sys.path.insert(0, str(args.upstream.resolve()))
    report = {
        "schema_version": 1,
        "mode": args.mode + "_gdn",
        "model_id": args.model,
        "revision": selected["revision"],
        "source_identity": model.manifest.identity,
        "upstream_revision": selected["upstream_code"]["revision"],
        "temperature": selected["temperature"],
        "reference_sha256": hashlib.sha256(raw).hexdigest(),
        "packages": packages,
        "selection": {
            "case_index": args.case_index,
            "row_index": args.row_index,
            "case": case["case"],
        },
        "fixture": {key: row[key] for key in layers.INPUT_KEYS},
        "scope": "one synthetic fixture, first linear attention layer actual boundaries; no certification",
        "insecure_serialization": False,
        "arithmetic_changed": False,
        "schedule": "sequential single unpadded question",
        "runtime_profile": "same engine arguments as original hidden/layer probes",
    }
    if args.mode == "hf":
        hf_stage(model, case, row, report)
    else:
        gdn_bytes = args.gdn_reference.read_bytes()
        expected = json.loads(gdn_bytes)
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
                raise ValueError("GDN reference provenance/fixture differs")
        if expected.get("mode") != "hf_gdn" or list(expected["capture"]["boundaries"]) != list(
            NAMES
        ):
            raise ValueError("expected complete HF GDN capture")
        compare_captures(expected["capture"]["boundaries"], expected["capture"]["boundaries"])
        report["gdn_reference_sha256"] = hashlib.sha256(gdn_bytes).hexdigest()
        layers.asyncio.run(runtime_stage(args, model, case, row, expected["capture"], report))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(report, output, ensure_ascii=False, allow_nan=False)
        output.write("\n")
    print("Saved first-layer GDN diagnostic", args.output)


if __name__ == "__main__":
    main()
