"""Offline JPT candidate-logit diagnosis; GPU imports are confined to execution."""

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import math
import os
import sys
from dataclasses import replace
from pathlib import Path

from jevserve.artifacts.catalog import profile, source_manifest
from jevserve.artifacts.verify import load_model_directory
from jevserve.runtime.token_backend import VllmTokenRuntime
from validation.model_matrix import requests, verify_upstream

ROOT = Path(__file__).resolve().parents[1]


def summarize(values, temperature):
    if not values or not all(math.isfinite(v) for v in values):
        raise ValueError("candidate logits must be nonempty and finite")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    centered = [v - max(values) for v in values]
    weights = [math.exp(v / temperature) for v in centered]
    return {
        "raw_logits": list(values),
        "centered_logits": centered,
        "temperature": temperature,
        "probabilities": [v / math.fsum(weights) for v in weights],
    }


def compare(actual, expected, tolerance):
    if len(actual["raw_logits"]) != len(expected["raw_logits"]):
        raise ValueError("candidate count differs")
    result = {}
    for key in ("raw_logits", "centered_logits", "probabilities"):
        result["max_" + key + "_error"] = max(
            abs(a - b) for a, b in zip(actual[key], expected[key], strict=True)
        )
    result["probability_parity_passed"] = all(
        abs(a - b) <= tolerance["probability_atol"] + tolerance["probability_rtol"] * abs(b)
        for a, b in zip(actual["probabilities"], expected["probabilities"], strict=True)
    )
    result["argmax_changed"] = max(
        range(len(actual["probabilities"])), key=actual["probabilities"].__getitem__
    ) != max(range(len(expected["probabilities"])), key=expected["probabilities"].__getitem__)
    return result


def verify_reference(report, selected, manifest_identity, mode):
    if (
        report.get("model_id") != selected["model_id"]
        or report.get("revision") != selected["revision"]
        or report.get("mode") != mode
    ):
        raise ValueError("reference model/revision/mode mismatch")
    if mode == "reference_logits" and (
        report.get("source_identity") != manifest_identity
        or report.get("temperature") != selected["temperature"]
        or report.get("upstream_revision") != selected["upstream_code"]["revision"]
    ):
        raise ValueError("diagnostic reference provenance mismatch")
    cases = report.get("cases", [])
    fixture = requests()
    if len(cases) != len(fixture) or any(
        row.get("case") != i or row.get("request") != data
        for i, (row, data) in enumerate(zip(cases, fixture, strict=True))
    ):
        raise ValueError("reference cases differ from synthetic matrix")


def adapter(tokenizer, temperature):
    from jevserve.adapters.jpt import JptAdapter
    from jevserve.contracts.capabilities import Capabilities

    return JptAdapter(
        tokenizer,
        temperature=temperature,
        capabilities=Capabilities(
            question_types=["choice", "noul", "score"],
            min_choice_candidates=2,
            noul_candidate_cost=2,
            max_choice_candidates=255,
            max_score_levels=255,
            max_tokens_per_candidate=4095,
            max_candidate_sequences=256,
            max_total_input_tokens=65536,
        ),
    )


def check_encoding(encoded, rows):
    if len(encoded.groups) != len(rows):
        raise ValueError("question count differs")
    for i, row in enumerate(rows):
        if (
            row["question_id"] != encoded.question_ids[i]
            or row["keys"] != list(encoded.groups[i].keys)
            or row["input_ids"] != list(encoded.engine_inputs[i])
            or row["candidate_token_ids"] != list(encoded.candidate_token_ids[i])
        ):
            raise ValueError("author/runtime input IDs or candidate order differ")


def reference_stage(args, selected, model, report, baseline, tolerance):
    import torch
    from llm2jev.backends import HF
    from llm2jev.engine import LLM2Jev
    from llm2jev.prompt import render
    from llm2jev.scoring import answer, softmax

    from jevserve.contracts.requests import DecisionRequest

    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("real CUDA required")
    backend = HF(str(model.path), device="cuda", dtype="bfloat16")
    author = LLM2Jev(backend.processor or backend.tok, backend, temperature=selected["temperature"])
    native = adapter(backend.tok, selected["temperature"])
    report["parameter_dtypes"] = sorted({str(p.dtype) for p in backend.model.parameters()})
    report["head_parameter_dtypes"] = sorted(
        {str(p.dtype) for p in backend.model.get_output_embeddings().parameters()}
    )
    report["schedule"] = "one unpadded prompt at a time, author HF.score"
    captured = []

    def capture(_module, _inputs, output):
        # Copy only selected logits; never retain the full vocabulary or GPU tensor.
        logits = output.logits[0, -1]
        captured.append((str(logits.dtype), logits[current_ids].float().cpu().tolist()))

    hook = backend.model.register_forward_hook(capture)
    try:
        for case in baseline["cases"]:
            data = case["request"]
            _, prompts, media = render(
                author.processor, data["state"], data["questions"], author.labels
            )
            if media:
                raise ValueError("text-only diagnostic")
            rows, answers = [], {}
            for qid, (text, keys) in prompts.items():
                current_ids = author.ids[: len(keys)]
                captured.clear()
                logprobs, _ = backend.score(text, [], current_ids)
                if len(captured) != 1:
                    raise ValueError("expected one author forward")
                dtype, logits = captured[0]
                values = summarize(logits, selected["temperature"])
                author_probs = softmax(logprobs, selected["temperature"])
                if any(
                    abs(a - b) > 1e-6
                    for a, b in zip(values["probabilities"], author_probs, strict=True)
                ):
                    raise ValueError("captured logits do not reconstruct author probabilities")
                answers[qid] = answer(data["questions"][qid], keys, author_probs)
                expected_answer = case["answers"][qid]
                expected = (
                    [expected_answer["noul"], 1 - expected_answer["noul"]]
                    if expected_answer["type"] == "noul"
                    else list(expected_answer["probabilities"].values())
                )
                if expected_answer["type"] != "noul" and list(
                    expected_answer["probabilities"]
                ) != list(keys):
                    raise ValueError("existing reference candidate order differs")
                if not all(
                    abs(a - b)
                    <= tolerance["probability_atol"] + tolerance["probability_rtol"] * abs(b)
                    for a, b in zip(author_probs, expected, strict=True)
                ):
                    raise ValueError("author rerun differs from existing HF reference")
                rows.append(
                    {
                        "question_id": qid,
                        "keys": list(keys),
                        "input_ids": backend.tok.encode(text, add_special_tokens=False),
                        "candidate_token_ids": list(current_ids),
                        "logits_dtype": dtype,
                        "author_logprobs": logprobs,
                        "existing_reference_max_probability_error": max(
                            abs(a - b) for a, b in zip(author_probs, expected, strict=True)
                        ),
                        **values,
                    }
                )
            check_encoding(native.encode(DecisionRequest.model_validate(data)), rows)
            report["cases"].append({**case, "answers": answers, "rows": rows})
    finally:
        hook.remove()
        author.pool.shutdown(wait=True)


class DiagnosticRuntime(VllmTokenRuntime):
    """Production inference/abort path with only the output representation changed."""

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
                cpu_offload_gb=self.cpu_offload_gb,
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
            )
        )


async def runtime_stage(args, selected, model, report, baseline, tolerance):
    from transformers import AutoTokenizer

    from jevserve.contracts.requests import DecisionRequest

    tokenizer = AutoTokenizer.from_pretrained(
        model.path, local_files_only=True, trust_remote_code=False
    )
    native = adapter(tokenizer, selected["temperature"])
    for case in baseline["cases"]:
        check_encoding(native.encode(DecisionRequest.model_validate(case["request"])), case["rows"])
    runtime = DiagnosticRuntime(
        args.cache_dir,
        gpu_memory_utilization=args.gpu_memory_utilization,
        cpu_offload_gb=args.cpu_offload_gb,
    )
    report["schedule"] = args.schedule
    report["logprobs_mode"] = "processed_logits"
    report["sampling_temperature"] = 1.0
    report["parameter_dtype_config"] = "bfloat16"
    report["logits_dtype"] = "float32 (pinned sampler conversion; pre-sampler dtype unobserved)"
    report["cpu_offload_gb"] = args.cpu_offload_gb
    report["gpu_memory_utilization"] = args.gpu_memory_utilization
    try:
        await runtime.start(model)
        for case in baseline["cases"]:
            encoded = native.encode(DecisionRequest.model_validate(case["request"]))
            modes = ["sequential", "batch"] if args.schedule == "both" else [args.schedule]
            runs = {}
            for mode in modes:
                if mode == "batch":
                    result = await runtime.infer(
                        encoded,
                        f"diag-{case['case']}-batch",
                        asyncio.get_running_loop().time() + args.timeout,
                    )
                    logits = result.candidate_logits
                else:
                    logits = []
                    for i in range(len(encoded.engine_inputs)):
                        single = replace(
                            encoded,
                            question_ids=encoded.question_ids[i : i + 1],
                            engine_inputs=encoded.engine_inputs[i : i + 1],
                            groups=encoded.groups[i : i + 1],
                            candidate_token_ids=encoded.candidate_token_ids[i : i + 1],
                            readout_positions=encoded.readout_positions[i : i + 1],
                            input_tokens=len(encoded.engine_inputs[i]),
                            candidate_count=len(encoded.candidate_token_ids[i]),
                        )
                        result = await runtime.infer(
                            single,
                            f"diag-{case['case']}-{i}",
                            asyncio.get_running_loop().time() + args.timeout,
                        )
                        logits.append(result.candidate_logits[0])
                runs[mode] = [summarize(row, selected["temperature"]) for row in logits]
            row = {
                "case": case["case"],
                "runs": runs,
                "comparisons": {},
                "inputs": [
                    {
                        k: reference_row[k]
                        for k in ("question_id", "keys", "input_ids", "candidate_token_ids")
                    }
                    for reference_row in case["rows"]
                ],
            }
            for mode, values in runs.items():
                row["comparisons"][mode + "_vs_hf"] = [
                    compare(a, b, tolerance) for a, b in zip(values, case["rows"], strict=True)
                ]
            if len(runs) == 2:
                row["comparisons"]["batch_vs_sequential"] = [
                    compare(a, b, tolerance)
                    for a, b in zip(runs["batch"], runs["sequential"], strict=True)
                ]
            report["cases"].append(row)
    finally:
        await runtime.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["reference", "runtime"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="existing matrix HF report for reference; diagnostic HF report for runtime",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--schedule", choices=["sequential", "batch", "both"], default="both")
    parser.add_argument("--cache-dir", type=Path, default=Path(".local/jpt-diagnostic-cache"))
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--cpu-offload-gb", type=float, default=0)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error("output already exists; choose a new report path")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout must be finite and positive")
    if args.output.resolve() == args.reference.resolve():
        parser.error("output must not overwrite the input reference")
    selected = profile(args.model)
    if selected["adapter"] != "jpt":
        parser.error("only pinned JPT profiles are supported")
    verify_upstream(args.upstream, selected["upstream_code"]["revision"])
    model = load_model_directory(args.model_dir, source_manifest(args.model))
    baseline_bytes = args.reference.read_bytes()
    baseline = json.loads(baseline_bytes)
    verify_reference(
        baseline,
        selected,
        model.manifest.identity,
        "reference" if args.mode == "reference" else "reference_logits",
    )
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    sys.path.insert(0, str(args.upstream.resolve()))
    tolerance = json.loads((ROOT / "compat/parity-profile.json").read_text())
    report = {
        "schema_version": 1,
        "model_id": args.model,
        "revision": selected["revision"],
        "source_identity": model.manifest.identity,
        "upstream_revision": selected["upstream_code"]["revision"],
        "mode": args.mode + "_logits",
        "temperature": selected["temperature"],
        "reference_sha256": hashlib.sha256(baseline_bytes).hexdigest(),
        "tolerance": tolerance,
        "cases": [],
        "scope": "synthetic candidate-logit diagnosis, not certification or accuracy",
        "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers")},
    }
    for name, expected in {"torch": "2.13.0", "transformers": "5.10.4"}.items():
        if report["packages"][name].split("+")[0] != expected:
            raise ValueError("diagnostic requires pinned runtime packages")
    if args.mode == "reference":
        reference_stage(args, selected, model, report, baseline, tolerance)
    else:
        report["packages"]["vllm"] = importlib.metadata.version("vllm")
        asyncio.run(runtime_stage(args, selected, model, report, baseline, tolerance))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print("Saved diagnostic report", args.output)


if __name__ == "__main__":
    main()
