"""Bounded single-question JPT layer diagnosis; named RPCs, never certification."""

import argparse
import asyncio
import base64
import hashlib
import importlib.metadata
import json
import math
import os
import struct
import sys
from dataclasses import replace
from pathlib import Path

from jevserve.artifacts.catalog import profile, source_manifest
from jevserve.artifacts.verify import load_model_directory
from validation.jpt_diagnostic import adapter, check_encoding, verify_reference
from validation.jpt_hidden_probe import (
    INPUT_KEYS,
    HiddenProbeWorker,
    HiddenRuntime,
    candidate_weights,
    hidden_difference,
    rpc,
)
from validation.model_matrix import verify_upstream

MAX_TOKENS = 1024
MAX_ELEMENTS = 16_777_216
MAX_BOUNDARIES = 130


def select_fixture(reference, case_index, row_index):
    if not 0 <= case_index < len(reference["cases"]):
        raise ValueError("invalid case index")
    case = reference["cases"][case_index]
    if not 0 <= row_index < len(case["rows"]):
        raise ValueError("invalid row index")
    row = case["rows"][row_index]
    if not 1 <= len(row["input_ids"]) <= MAX_TOKENS:
        raise ValueError("selected fixture exceeds token bound")
    return case, row


def decode_matrix(record):
    shape = record["shape"]
    if (
        record["dtype"] != "torch.bfloat16"
        or len(shape) != 2
        or any(type(v) is not int or v <= 0 for v in shape)
        or shape[0] > MAX_TOKENS
        or math.prod(shape) > MAX_ELEMENTS
    ):
        raise ValueError("invalid bounded BF16 matrix")
    encoded = record["bf16_le_base64"]
    if len(encoded) > 4 * ((math.prod(shape) * 2 + 2) // 3):
        raise ValueError("matrix payload exceeds shape bound")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise ValueError("invalid matrix encoding") from exc
    if len(raw) != math.prod(shape) * 2 or hashlib.sha256(raw).hexdigest() != record["sha256"]:
        raise ValueError("matrix bytes/shape/hash differ")
    values = [
        struct.unpack("<f", struct.pack("<I", word[0] << 16))[0]
        for word in struct.iter_unpack("<H", raw)
    ]
    if not all(math.isfinite(v) for v in values):
        raise ValueError("nonfinite activation")
    return values


def encode_matrix(tensor, tokens):
    import torch

    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if (
        tensor.ndim != 2
        or tensor.shape[0] != tokens
        or tensor.dtype != torch.bfloat16
        or tensor.numel() > MAX_ELEMENTS
        or sys.byteorder != "little"
    ):
        raise ValueError("expected full unpadded bounded BF16 sequence")
    raw = tensor.detach().contiguous().cpu().view(torch.uint16).numpy().tobytes()
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("nonfinite activation")
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bf16_le_base64": base64.b64encode(raw).decode("ascii"),
    }


def compare_boundaries(actual, expected):
    if list(actual) != list(expected) or not 1 <= len(actual) <= MAX_BOUNDARIES:
        raise ValueError("boundary names/order differ")
    rows = []
    for name in expected:
        a, e = actual[name], expected[name]
        if a["shape"] != e["shape"]:
            raise ValueError("boundary shapes differ")
        av, ev = decode_matrix(a), decode_matrix(e)
        width = e["shape"][1]
        rows.append(
            {
                "boundary": name,
                "all_tokens": hidden_difference(av, ev),
                "last_token": hidden_difference(av[-width:], ev[-width:]),
            }
        )
    return {
        "first_unequal_boundary": next(
            (r["boundary"] for r in rows if r["all_tokens"]["unequal_elements"]), None
        ),
        "boundaries": rows,
    }


def boundary_modules(body):
    if not 1 <= len(body.layers) <= (MAX_BOUNDARIES - 2) // 2:
        raise ValueError("unsupported number of layers")
    result = [("embedding", body.embed_tokens)]
    for index, layer in enumerate(body.layers):
        result.extend(
            [
                (f"layer.{index}.input_norm", layer.input_layernorm),
                (f"layer.{index}.post_attention_norm", layer.post_attention_layernorm),
            ]
        )
    result.append(("final_norm", body.norm))
    return result


def install_hooks(body, state):
    handles = []
    try:
        for name, module in boundary_modules(body):

            def capture(_module, _inputs, output, name=name):
                if state["armed"] is None:
                    return
                if name in state["captures"]:
                    raise ValueError("multiple forwards/chunked prefill are unsupported")
                # vLLM fused RMSNorm returns (normalized, residual). Only the
                # normalized tensor matches HF; decoder outputs are not aligned.
                normalized = output[0] if isinstance(output, tuple) else output
                state["captures"][name] = normalized.detach().clone()

            handles.append(module.register_forward_hook(capture))
    except BaseException:
        for handle in handles:
            handle.remove()
        raise
    return handles


def drain_boundaries(state, body, tokens):
    expected = [name for name, _ in boundary_modules(body)]
    if list(state["captures"]) != expected:
        raise ValueError("missing or reordered layer captures")
    return {name: encode_matrix(value, tokens) for name, value in state["captures"].items()}


class LayerProbeWorker(HiddenProbeWorker):
    def jpt_layer_install(self):
        if hasattr(self, "_jpt_layer_state"):
            raise ValueError("layer probe already installed")
        result = self.jpt_hidden_install()
        language = getattr(self.get_model(), "language_model", self.get_model())
        body = language.model
        state = {"armed": None, "captures": {}, "body": body}
        try:
            state["handles"] = install_hooks(body, state)
        except BaseException:
            self.jpt_hidden_remove()
            raise
        self._jpt_layer_state = state
        return result

    def jpt_layer_arm(self, tag, ids, weight_sha256, tokens):
        state = self._jpt_layer_state
        if state["armed"] is not None or state["captures"]:
            raise ValueError("previous layer capture not drained")
        if type(tokens) is not int or not 1 <= tokens <= MAX_TOKENS:
            raise ValueError("invalid token count")
        self.jpt_hidden_arm(tag, ids, weight_sha256)
        state["armed"] = tokens
        return {"armed": tag}

    def jpt_layer_take(self):
        state = self._jpt_layer_state
        try:
            boundaries = drain_boundaries(state, state["body"], state["armed"])
            return {**self.jpt_hidden_take(), "boundaries": boundaries}
        finally:
            state["armed"] = None
            state["captures"].clear()

    def jpt_layer_remove(self):
        for handle in self._jpt_layer_state["handles"]:
            handle.remove()
        del self._jpt_layer_state
        return self.jpt_hidden_remove()


class LayerRuntime(HiddenRuntime):
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
                worker_extension_cls="validation.jpt_layer_probe.LayerProbeWorker",
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
    check_encoding(
        adapter(backend.tok, report["temperature"]).encode(
            DecisionRequest.model_validate(case["request"])
        ),
        case["rows"],
    )
    body = backend.model.model.language_model
    state = {"armed": len(row["input_ids"]), "captures": {}}
    handles = install_hooks(body, state)
    head = backend.model.get_output_embeddings()
    try:
        with torch.no_grad(), backend.lock:
            tokens = torch.tensor([row["input_ids"]], device=backend.model.device)
            result = backend.model(input_ids=tokens)
            _, metadata = candidate_weights(head, row["candidate_token_ids"])
            report["capture"] = {
                **metadata,
                "boundaries": drain_boundaries(state, body, len(row["input_ids"])),
                "original_logits": result.logits[0, -1, row["candidate_token_ids"]]
                .float()
                .cpu()
                .tolist(),
            }
    finally:
        for handle in handles:
            handle.remove()


async def runtime_stage(args, model, case, row, expected, report):
    from transformers import AutoTokenizer

    from jevserve.contracts.requests import DecisionRequest

    native = adapter(
        AutoTokenizer.from_pretrained(model.path, local_files_only=True, trust_remote_code=False),
        report["temperature"],
    )
    encoded = native.encode(DecisionRequest.model_validate(case["request"]))
    check_encoding(encoded, case["rows"])
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
    runtime = LayerRuntime(args.cache_dir, gpu_memory_utilization=args.gpu_memory_utilization)
    installed = False
    try:
        await runtime.start(model)
        report["worker"] = await rpc(runtime, "jpt_layer_install")
        installed = True
        await rpc(
            runtime,
            "jpt_layer_arm",
            "layer-fixture",
            row["candidate_token_ids"],
            expected["weight_sha256"],
            len(row["input_ids"]),
        )
        result = await runtime.infer(
            single, "layer-fixture", asyncio.get_running_loop().time() + args.timeout
        )
        capture = await rpc(runtime, "jpt_layer_take")
        if capture["original_logits"] != list(result.candidate_logits[0]):
            raise ValueError("captured head differs from runtime logits")
        for key in ("weight_sha256", "weight_shape", "weight_dtype"):
            if capture[key] != expected[key]:
                raise ValueError("candidate weight identity differs")
        report["capture"] = capture
        report["comparison"] = compare_boundaries(capture["boundaries"], expected["boundaries"])
    finally:
        try:
            if installed and runtime.healthy():
                await rpc(runtime, "jpt_layer_remove")
        finally:
            await runtime.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["hf", "runtime"])
    parser.add_argument("--model", required=True)
    for name in ("model-dir", "upstream", "reference", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--layer-reference", type=Path)
    parser.add_argument("--case-index", type=int, default=0)
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--cache-dir", type=Path, default=Path(".local/jpt-layer-cache"))
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error("output exists; choose a new report path")
    if args.mode == "runtime" and args.layer_reference is None:
        parser.error("runtime requires --layer-reference")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout must be finite and positive")
    if not math.isfinite(args.gpu_memory_utilization) or not 0 < args.gpu_memory_utilization < 1:
        parser.error("GPU memory utilization must lie between zero and one")
    if os.environ.get("VLLM_ALLOW_INSECURE_SERIALIZATION", "0") != "0":
        parser.error("insecure serialization must be disabled")
    selected = profile(args.model)
    if selected["adapter"] != "jpt":
        parser.error("only pinned JPT profiles are supported")
    verify_upstream(args.upstream, selected["upstream_code"]["revision"])
    model = load_model_directory(args.model_dir, source_manifest(args.model))
    reference_bytes = args.reference.read_bytes()
    reference = json.loads(reference_bytes)
    verify_reference(reference, selected, model.manifest.identity, "reference_logits")
    case, row = select_fixture(reference, args.case_index, args.row_index)
    pins = {"torch": "2.13.0", "transformers": "5.10.4"}
    if args.mode == "runtime":
        pins["vllm"] = "0.29.0"
    packages = {name: importlib.metadata.version(name) for name in pins}
    if any(packages[name].split("+")[0] != pin for name, pin in pins.items()):
        raise ValueError("layer probe requires pinned runtime packages")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    sys.path.insert(0, str(args.upstream.resolve()))
    report = {
        "schema_version": 1,
        "mode": args.mode + "_layers",
        "model_id": args.model,
        "revision": selected["revision"],
        "source_identity": model.manifest.identity,
        "upstream_revision": selected["upstream_code"]["revision"],
        "temperature": selected["temperature"],
        "packages": packages,
        "reference_sha256": hashlib.sha256(reference_bytes).hexdigest(),
        "selection": {
            "case_index": args.case_index,
            "row_index": args.row_index,
            "case": case["case"],
        },
        "fixture": {key: row[key] for key in INPUT_KEYS},
        "scope": "one synthetic unpadded question; full token boundaries; no certification or causal attribution",
        "schedule": "sequential single question",
        "insecure_serialization": False,
        "boundaries": "embedding; normalized input to token mixer; normalized input to MLP; final norm",
        "runtime_profile": "BF16, eager, FP32 Mamba cache, no prefix cache; unchanged hidden probe profile",
    }
    if args.mode == "hf":
        hf_stage(model, case, row, report)
    else:
        raw = args.layer_reference.read_bytes()
        expected = json.loads(raw)
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
                raise ValueError("layer reference provenance or fixture differs")
        if expected.get("mode") != "hf_layers":
            raise ValueError("expected HF layer reference")
        compare_boundaries(expected["capture"]["boundaries"], expected["capture"]["boundaries"])
        report["layer_reference_sha256"] = hashlib.sha256(raw).hexdigest()
        asyncio.run(runtime_stage(args, model, case, row, expected["capture"], report))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(report, output, ensure_ascii=False, allow_nan=False)
        output.write("\n")
    print("Saved layer diagnostic", args.output)


if __name__ == "__main__":
    main()
