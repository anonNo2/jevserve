"""Two-process JPT pre-head diagnosis, using named worker RPCs only."""

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import math
import os
import sys
from pathlib import Path

from jevserve.artifacts.catalog import profile, source_manifest
from jevserve.artifacts.verify import load_model_directory
from validation.jpt_diagnostic import (
    ROOT,
    DiagnosticRuntime,
    adapter,
    check_encoding,
    compare,
    summarize,
    verify_reference,
)
from validation.model_matrix import verify_upstream

INPUT_KEYS = ("question_id", "keys", "input_ids", "candidate_token_ids")


def hidden_difference(actual, expected):
    if (
        not actual
        or len(actual) != len(expected)
        or not all(math.isfinite(v) for v in [*actual, *expected])
    ):
        raise ValueError("hidden vectors must have equal nonzero length and finite values")
    delta = [a - b for a, b in zip(actual, expected, strict=True)]
    l2 = math.sqrt(math.fsum(v * v for v in delta))
    norm = math.sqrt(math.fsum(v * v for v in expected))
    return {
        "elements": len(delta),
        "unequal_elements": sum(v != 0 for v in delta),
        "max_abs_error": max(abs(v) for v in delta),
        "l2_error": l2,
        "relative_l2_error": l2 / norm if norm else None,
    }


def validate_capture(capture, expected, tag):
    if capture["tag"] != tag or capture["candidate_token_ids"] != expected["candidate_token_ids"]:
        raise ValueError("capture identity or candidate order differs")
    if capture["weight_sha256"] != expected["weight_sha256"]:
        raise ValueError("candidate weights differ between HF and vLLM")
    if (
        capture["weight_shape"] != expected["weight_shape"]
        or capture["weight_dtype"] != expected["weight_dtype"]
    ):
        raise ValueError("candidate weight shape/dtype differs")
    return hidden_difference(capture["hidden"], expected["hidden"])


def candidate_weights(head, ids):
    import torch

    if head.weight.dtype != torch.bfloat16 or getattr(head, "bias", None) is not None:
        raise ValueError("expected bias-free BF16 head")
    weights = head.weight[ids].detach().contiguous()
    # Hash exact BF16 bytes, not decimal text. Shape/dtype/order are checked separately.
    digest = hashlib.sha256(weights.cpu().view(torch.uint16).numpy().tobytes()).hexdigest()
    return weights, {
        "weight_sha256": digest,
        "weight_shape": list(weights.shape),
        "weight_dtype": str(weights.dtype),
    }


def project(hidden, weights):
    import torch
    import torch.nn.functional as functional

    if hidden.dtype != torch.bfloat16 or list(hidden.shape) != [1, weights.shape[1]]:
        raise ValueError("expected one BF16 hidden row")
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
            result = functional.linear(hidden.float(), weights.float())
        return result[0].cpu().tolist()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


class HiddenProbeWorker:
    """Diagnostic extension imported by name; RPC payloads are JSON-compatible."""

    def jpt_hidden_install(self):
        model = self.get_model()
        if hasattr(self, "_jpt_hidden_state"):
            raise ValueError("hidden probe already installed")
        language = getattr(model, "language_model", model)
        head = language.lm_head
        if getattr(head, "tp_size", 1) != 1:
            raise ValueError("hidden probe requires one tensor-parallel worker")
        original = model.compute_logits
        state = {"armed": None, "captures": [], "original": original, "head": head}
        self._jpt_hidden_state = state

        def capture(hidden_states, *args, **kwargs):
            armed = state["armed"]
            hidden = hidden_states.detach().clone() if armed else None
            result = original(hidden_states, *args, **kwargs)
            if armed:
                if result is None or list(hidden.shape[:1]) != [1] or hidden.ndim != 2:
                    raise ValueError("capture requires one sampled hidden row")
                if state["captures"]:
                    raise ValueError("more than one compute_logits call for armed question")
                state["captures"].append((hidden, result[0, armed["ids"]].detach().clone()))
            return result

        model.compute_logits = capture
        return {"installed": True, "model_class": type(model).__name__}

    def jpt_hidden_arm(self, tag, ids, weight_sha256):
        state = self._jpt_hidden_state
        if state["armed"] is not None or state["captures"]:
            raise ValueError("previous capture not drained")
        if not 1 <= len(ids) <= 255 or len(set(ids)) != len(ids):
            raise ValueError("invalid candidate IDs")
        weights, metadata = candidate_weights(state["head"], ids)
        if metadata["weight_sha256"] != weight_sha256:
            raise ValueError("candidate weight bytes differ from HF")
        state["armed"] = {"tag": tag, "ids": ids, "weights": weights, "metadata": metadata}
        return {"armed": tag}

    def jpt_hidden_take(self):
        state = self._jpt_hidden_state
        if state["armed"] is None or len(state["captures"]) != 1:
            raise ValueError("expected exactly one completed capture")
        armed = state["armed"]
        hidden, raw = state["captures"][0]
        try:
            return {
                "tag": armed["tag"],
                "candidate_token_ids": armed["ids"],
                **armed["metadata"],
                "hidden": hidden[0].float().cpu().tolist(),
                "hidden_dtype": str(hidden.dtype),
                "hidden_shape": list(hidden.shape),
                "original_logits": raw.float().cpu().tolist(),
                "original_logits_dtype": str(raw.dtype),
                "fp32_logits": project(hidden, armed["weights"]),
            }
        finally:
            state["armed"] = None
            state["captures"].clear()

    def jpt_hidden_remove(self):
        state = self._jpt_hidden_state
        self.get_model().compute_logits = state["original"]
        del self._jpt_hidden_state
        return {"removed": True}


class HiddenRuntime(DiagnosticRuntime):
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
                worker_extension_cls="validation.jpt_hidden_probe.HiddenProbeWorker",
            )
        )


async def rpc(runtime, method, *args):
    rows = await runtime._engine.collective_rpc(method, timeout=30, args=args)
    if len(rows) != 1:
        raise ValueError("expected exactly one diagnostic worker")
    return rows[0]


def hf_stage(model, reference, report):
    import torch
    from llm2jev.backends import HF

    from jevserve.contracts.requests import DecisionRequest

    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("real CUDA required")
    backend = HF(str(model.path), device="cuda", dtype="bfloat16")
    native = adapter(backend.tok, report["temperature"])
    head = backend.model.get_output_embeddings()
    captured = {}

    def pre_hook(_module, inputs):
        hidden = inputs[0]
        if list(hidden.shape[:2]) != [1, len(current["input_ids"])]:
            raise ValueError("expected original full-sequence head input")
        captured["hidden"] = hidden[:, -1, :].detach().clone()

    def post_hook(_module, _inputs, output):
        if "raw" in captured:
            raise ValueError("expected one head forward")
        captured["raw"] = output[0, -1, current["candidate_token_ids"]].detach().clone()

    pre, post = head.register_forward_pre_hook(pre_hook), head.register_forward_hook(post_hook)
    try:
        with torch.no_grad():
            for case in reference["cases"]:
                check_encoding(
                    native.encode(DecisionRequest.model_validate(case["request"])), case["rows"]
                )
                rows = []
                for current in case["rows"]:
                    captured.clear()
                    tokens = torch.tensor([current["input_ids"]], device=backend.model.device)
                    with backend.lock:
                        backend.model(input_ids=tokens)
                    hidden, raw = captured["hidden"], captured["raw"]
                    weights, metadata = candidate_weights(head, current["candidate_token_ids"])
                    original = summarize(raw.float().cpu().tolist(), report["temperature"])
                    rows.append(
                        {
                            **{k: current[k] for k in INPUT_KEYS},
                            **metadata,
                            "hidden": hidden[0].float().cpu().tolist(),
                            "hidden_dtype": str(hidden.dtype),
                            "hidden_shape": list(hidden.shape),
                            "original": original,
                            "fp32": summarize(project(hidden, weights), report["temperature"]),
                            "original_vs_reference": compare(
                                original, current, report["tolerance"]
                            ),
                        }
                    )
                    del hidden, raw, weights, tokens
                report["cases"].append(
                    {"case": case["case"], "request": case["request"], "rows": rows}
                )
                print("HF hidden case", case["case"], "executed", flush=True)
    finally:
        pre.remove()
        post.remove()


async def runtime_stage(args, model, reference, hidden_reference, report):
    from dataclasses import replace

    from transformers import AutoTokenizer

    from jevserve.contracts.requests import DecisionRequest

    native = adapter(
        AutoTokenizer.from_pretrained(model.path, local_files_only=True, trust_remote_code=False),
        report["temperature"],
    )
    runtime = HiddenRuntime(args.cache_dir, gpu_memory_utilization=args.gpu_memory_utilization)
    installed = False
    try:
        await runtime.start(model)
        report["worker"] = await rpc(runtime, "jpt_hidden_install")
        installed = True
        for case, hidden_case in zip(reference["cases"], hidden_reference["cases"], strict=True):
            encoded = native.encode(DecisionRequest.model_validate(case["request"]))
            check_encoding(encoded, hidden_case["rows"])
            rows = []
            for index, expected in enumerate(hidden_case["rows"]):
                tag = f"hidden-{case['case']}-{index}"
                await rpc(
                    runtime,
                    "jpt_hidden_arm",
                    tag,
                    expected["candidate_token_ids"],
                    expected["weight_sha256"],
                )
                single = replace(
                    encoded,
                    question_ids=encoded.question_ids[index : index + 1],
                    engine_inputs=encoded.engine_inputs[index : index + 1],
                    groups=encoded.groups[index : index + 1],
                    candidate_token_ids=encoded.candidate_token_ids[index : index + 1],
                    readout_positions=encoded.readout_positions[index : index + 1],
                    input_tokens=len(encoded.engine_inputs[index]),
                    candidate_count=len(expected["candidate_token_ids"]),
                )
                result = await runtime.infer(
                    single, tag, asyncio.get_running_loop().time() + args.timeout
                )
                captured = await rpc(runtime, "jpt_hidden_take")
                delta = validate_capture(captured, expected, tag)
                if list(result.candidate_logits[0]) != captured["original_logits"]:
                    raise ValueError("captured head logits differ from processed_logits result")
                projected = summarize(captured["fp32_logits"], report["temperature"])
                original = summarize(captured["original_logits"], report["temperature"])
                rows.append(
                    {
                        **{k: expected[k] for k in INPUT_KEYS},
                        **captured,
                        "hidden_difference": delta,
                        "fp32": projected,
                        "original": original,
                        "fp32_vs_hf_fp32": compare(
                            projected, expected["fp32"], report["tolerance"]
                        ),
                        "original_vs_hf_original": compare(
                            original, expected["original"], report["tolerance"]
                        ),
                    }
                )
            report["cases"].append({"case": case["case"], "rows": rows})
            print("Runtime hidden case", case["case"], "executed", flush=True)
    finally:
        try:
            if installed and runtime.healthy():
                await rpc(runtime, "jpt_hidden_remove")
        finally:
            await runtime.close()


def validate_hidden_reference(hidden, reference, digest):
    for key in ("model_id", "revision", "source_identity", "upstream_revision", "temperature"):
        if hidden.get(key) != reference.get(key):
            raise ValueError("hidden reference provenance differs")
    if hidden.get("mode") != "hf_hidden" or hidden.get("reference_sha256") != digest:
        raise ValueError("hidden reference must link exact HF logit report")
    if len(hidden.get("cases", [])) != len(reference["cases"]):
        raise ValueError("hidden reference case count differs")
    for a, b in zip(hidden["cases"], reference["cases"], strict=True):
        if (
            a["case"] != b["case"]
            or a["request"] != b["request"]
            or len(a["rows"]) != len(b["rows"])
        ):
            raise ValueError("hidden reference cases differ")
        for row, expected in zip(a["rows"], b["rows"], strict=True):
            if any(row[k] != expected[k] for k in INPUT_KEYS):
                raise ValueError("hidden reference inputs differ")
            hidden_difference(row["hidden"], row["hidden"])
            if row["hidden_dtype"] != "torch.bfloat16" or row["hidden_shape"] != [
                1,
                len(row["hidden"]),
            ]:
                raise ValueError("hidden shape/dtype differs")
            if row["fp32"] != summarize(row["fp32"]["raw_logits"], hidden["temperature"]):
                raise ValueError("hidden projection derived values differ")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["hf", "runtime"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument(
        "--reference", type=Path, required=True, help="original HF logit diagnostic"
    )
    parser.add_argument("--hidden-reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path(".local/jpt-hidden-cache"))
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error("output already exists; choose a new report path")
    if args.mode == "runtime" and args.hidden_reference is None:
        parser.error("runtime requires --hidden-reference")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout must be finite and positive")
    if args.output.resolve() in {p.resolve() for p in (args.reference, args.hidden_reference) if p}:
        parser.error("output must not overwrite inputs")
    if os.environ.get("VLLM_ALLOW_INSECURE_SERIALIZATION", "0") != "0":
        parser.error("hidden probe requires insecure serialization disabled")
    selected = profile(args.model)
    if selected["adapter"] != "jpt":
        parser.error("only pinned JPT profiles are supported")
    verify_upstream(args.upstream, selected["upstream_code"]["revision"])
    model = load_model_directory(args.model_dir, source_manifest(args.model))
    data = args.reference.read_bytes()
    reference, digest = json.loads(data), hashlib.sha256(data).hexdigest()
    verify_reference(reference, selected, model.manifest.identity, "reference_logits")
    packages = {n: importlib.metadata.version(n) for n in ("torch", "transformers")}
    if any(
        packages[n].split("+")[0] != v
        for n, v in {"torch": "2.13.0", "transformers": "5.10.4"}.items()
    ):
        raise ValueError("hidden probe requires pinned runtime packages")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    sys.path.insert(0, str(args.upstream.resolve()))
    report = {
        "schema_version": 1,
        "mode": args.mode + "_hidden",
        "model_id": args.model,
        "revision": selected["revision"],
        "source_identity": model.manifest.identity,
        "upstream_revision": selected["upstream_code"]["revision"],
        "temperature": selected["temperature"],
        "reference_sha256": digest,
        "packages": packages,
        "cases": [],
        "tolerance": json.loads((ROOT / "compat/parity-profile.json").read_text()),
        "scope": "pre-head difference diagnosis; not attribution to a specific layer or certification",
        "projection": "candidate-only FP32 from original BF16 operands; TF32/autocast disabled",
        "schedule": "sequential single question",
        "insecure_serialization": False,
    }
    if args.mode == "hf":
        hf_stage(model, reference, report)
    else:
        hidden_bytes = args.hidden_reference.read_bytes()
        hidden_reference = json.loads(hidden_bytes)
        validate_hidden_reference(hidden_reference, reference, digest)
        report["hidden_reference_sha256"] = hashlib.sha256(hidden_bytes).hexdigest()
        report["packages"]["vllm"] = importlib.metadata.version("vllm")
        report["gpu_memory_utilization"] = args.gpu_memory_utilization
        asyncio.run(runtime_stage(args, model, reference, hidden_reference, report))
        rows = [row for case in report["cases"] for row in case["rows"]]
        report["summary"] = {
            "questions": len(rows),
            "different_hidden_questions": sum(
                row["hidden_difference"]["unequal_elements"] > 0 for row in rows
            ),
            "max_hidden_abs_error": max(row["hidden_difference"]["max_abs_error"] for row in rows),
            "fp32_failed_questions": sum(
                not row["fp32_vs_hf_fp32"]["probability_parity_passed"] for row in rows
            ),
            "max_fp32_probability_error": max(
                row["fp32_vs_hf_fp32"]["max_probabilities_error"] for row in rows
            ),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print("Saved hidden diagnostic", args.output)


if __name__ == "__main__":
    main()
