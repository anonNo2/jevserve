"""Diagnostic-only JPT head shape/precision counterfactuals on identical HF hidden states."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path

from jevserve.artifacts.catalog import profile, source_manifest
from jevserve.artifacts.verify import load_model_directory
from validation.jpt_diagnostic import (
    ROOT,
    adapter,
    check_encoding,
    compare,
    summarize,
    verify_reference,
)
from validation.model_matrix import verify_upstream

INPUT_KEYS = ("question_id", "keys", "input_ids", "candidate_token_ids")
VARIANTS = (
    "A_original",
    "B_last_full_bf16",
    "C_last_candidates_bf16",
    "D_last_candidates_fp32",
    "D_round_bf16",
)


def verify_runtime(runtime, reference, reference_digest):
    for key in ("model_id", "revision", "source_identity", "upstream_revision", "temperature"):
        if runtime.get(key) != reference.get(key):
            raise ValueError("runtime/reference provenance mismatch")
    if (
        runtime.get("mode") != "runtime_logits"
        or runtime.get("reference_sha256") != reference_digest
    ):
        raise ValueError("runtime must link the exact HF diagnostic report")
    if len(runtime.get("cases", [])) != len(reference["cases"]):
        raise ValueError("runtime case count differs")
    for actual, expected in zip(runtime["cases"], reference["cases"], strict=True):
        if actual["case"] != expected["case"] or actual.get("inputs") != [
            {key: row[key] for key in INPUT_KEYS} for row in expected["rows"]
        ]:
            raise ValueError("runtime input IDs or candidate order differ")
        if not actual.get("runs") or set(actual["runs"]) - {"sequential", "batch"}:
            raise ValueError("invalid runtime schedule")
        for rows in actual["runs"].values():
            if len(rows) != len(expected["rows"]):
                raise ValueError("runtime question count differs")
            for row, ref_row in zip(rows, expected["rows"], strict=True):
                if len(row["raw_logits"]) != len(ref_row["candidate_token_ids"]):
                    raise ValueError("runtime candidate count differs")
                if row != summarize(row["raw_logits"], reference["temperature"]):
                    raise ValueError("runtime derived values differ from raw logits")


def comparisons(variants, reference, runtime_rows, tolerance):
    result = {
        f"{name}_vs_A": compare(values, variants["A_original"], tolerance)
        for name, values in variants.items()
        if name != "A_original"
    }
    result["A_vs_hf_reference"] = compare(variants["A_original"], reference, tolerance)
    result["C_vs_B"] = compare(
        variants["C_last_candidates_bf16"], variants["B_last_full_bf16"], tolerance
    )
    result["D_round_vs_D"] = compare(
        variants["D_round_bf16"], variants["D_last_candidates_fp32"], tolerance
    )
    for mode, runtime_row in runtime_rows.items():
        for name, values in variants.items():
            # Keep HF-side probabilities as the denominator for frozen relative tolerance.
            result[f"vllm_{mode}_vs_{name}"] = compare(runtime_row, values, tolerance)
    return result


def aggregate(cases):
    result = {}
    for case in cases:
        for row in case["rows"]:
            for name, values in row["comparisons"].items():
                item = result.setdefault(
                    name,
                    {
                        "questions": 0,
                        "failed_questions": 0,
                        "argmax_changes": 0,
                        "max_raw_logits_error": 0,
                        "max_centered_logits_error": 0,
                        "max_probabilities_error": 0,
                    },
                )
                item["questions"] += 1
                item["failed_questions"] += not values["probability_parity_passed"]
                item["argmax_changes"] += values["argmax_changed"]
                for key in (
                    "max_raw_logits_error",
                    "max_centered_logits_error",
                    "max_probabilities_error",
                ):
                    item[key] = max(item[key], values[key])
    return result


def probe(args, selected, model, reference, runtime, report):
    import torch
    import torch.nn.functional as functional
    from llm2jev.backends import HF

    from jevserve.contracts.requests import DecisionRequest

    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("real CUDA required")
    backend = HF(str(model.path), device="cuda", dtype="bfloat16")
    native = adapter(backend.tok, selected["temperature"])
    for case in reference["cases"]:
        check_encoding(native.encode(DecisionRequest.model_validate(case["request"])), case["rows"])
    head = backend.model.get_output_embeddings()
    if head.weight.dtype != torch.bfloat16 or head.bias is not None:
        raise ValueError("expected pinned bias-free BF16 head")
    report["head_weight_shape"] = list(head.weight.shape)
    report["head_weight_dtype"] = str(head.weight.dtype)
    report["parameter_dtypes"] = sorted({str(p.dtype) for p in backend.model.parameters()})
    report["original_matmul_settings"] = {
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
    }
    captured = {}

    def pre_hook(_module, inputs):
        hidden = inputs[0]
        if list(hidden.shape[:2]) != [1, len(current["input_ids"])]:
            raise ValueError("expected original full-sequence head input")
        if hidden.dtype != torch.bfloat16:
            raise ValueError("expected BF16 head input")
        captured["hidden"] = hidden[:, -1, :].detach().clone()
        captured["input_shape"] = list(hidden.shape)
        captured["input_stride"] = list(hidden.stride())
        captured["hidden_dtype"] = str(hidden.dtype)

    def post_hook(_module, _inputs, output):
        if "A" in captured or output.dtype != torch.bfloat16:
            raise ValueError("expected one BF16 head forward")
        captured["A"] = output[0, -1, current["candidate_token_ids"]].float().cpu().tolist()
        captured["output_dtype"] = str(output.dtype)

    pre = head.register_forward_pre_hook(pre_hook)
    post = head.register_forward_hook(post_hook)
    try:
        with torch.no_grad():
            for case, runtime_case in zip(reference["cases"], runtime["cases"], strict=True):
                rows = []
                for index, current in enumerate(case["rows"]):
                    captured.clear()
                    tokens = torch.tensor([current["input_ids"]], device=backend.model.device)
                    # Same author HF.score forward call; supplied IDs were verified above.
                    # Discard full output immediately. Hooks retain only final hidden/candidates.
                    with backend.lock:
                        backend.model(input_ids=tokens)
                    if "A" not in captured or "hidden" not in captured:
                        raise ValueError("head hooks did not capture one original forward")
                    hidden = captured.pop("hidden")
                    ids = current["candidate_token_ids"]
                    weights = head.weight[ids]
                    a = captured.pop("A")
                    b_tensor = functional.linear(hidden, head.weight)[0, ids]
                    c_tensor = functional.linear(hidden, weights)[0]
                    b = b_tensor.float().cpu().tolist()
                    c = c_tensor.float().cpu().tolist()
                    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
                    try:
                        torch.backends.cuda.matmul.allow_tf32 = False
                        with torch.autocast(device_type="cuda", enabled=False):
                            d = functional.linear(hidden.float(), weights.float())[0]
                        d_values = d.cpu().tolist()
                        rounded_tensor = d.to(torch.bfloat16)
                        rounded = rounded_tensor.float().cpu().tolist()
                    finally:
                        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
                    variants = {
                        name: summarize(values, selected["temperature"])
                        for name, values in zip(VARIANTS, (a, b, c, d_values, rounded), strict=True)
                    }
                    rows.append(
                        {
                            **{key: current[key] for key in INPUT_KEYS},
                            **captured,
                            "last_hidden_shape": list(hidden.shape),
                            "last_hidden_stride": list(hidden.stride()),
                            "candidate_weight_shape": list(weights.shape),
                            "variants": variants,
                            "variant_output_dtypes": dict(
                                zip(
                                    VARIANTS,
                                    (
                                        captured["output_dtype"],
                                        str(b_tensor.dtype),
                                        str(c_tensor.dtype),
                                        str(d.dtype),
                                        str(rounded_tensor.dtype),
                                    ),
                                    strict=True,
                                )
                            ),
                            "comparisons": comparisons(
                                variants,
                                current,
                                {
                                    mode: values[index]
                                    for mode, values in runtime_case["runs"].items()
                                },
                                report["tolerance"],
                            ),
                        }
                    )
                    del hidden, weights, d, tokens, b_tensor, c_tensor, rounded_tensor
                report["cases"].append({"case": case["case"], "rows": rows})
                print("Head probe case", case["case"], "executed", flush=True)
    finally:
        pre.remove()
        post.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True, help="HF diagnostic report")
    parser.add_argument("--runtime", type=Path, required=True, help="vLLM diagnostic report")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error("output already exists; choose a new report path")
    if args.output.resolve() in {args.reference.resolve(), args.runtime.resolve()}:
        parser.error("output must not overwrite input reports")
    selected = profile(args.model)
    if selected["adapter"] != "jpt":
        parser.error("only pinned JPT profiles are supported")
    verify_upstream(args.upstream, selected["upstream_code"]["revision"])
    model = load_model_directory(args.model_dir, source_manifest(args.model))
    reference_bytes, runtime_bytes = args.reference.read_bytes(), args.runtime.read_bytes()
    reference, runtime = json.loads(reference_bytes), json.loads(runtime_bytes)
    reference_digest = hashlib.sha256(reference_bytes).hexdigest()
    verify_reference(reference, selected, model.manifest.identity, "reference_logits")
    verify_runtime(runtime, reference, reference_digest)
    packages = {name: importlib.metadata.version(name) for name in ("torch", "transformers")}
    for name, expected in {"torch": "2.13.0", "transformers": "5.10.4"}.items():
        if packages[name].split("+")[0] != expected:
            raise ValueError("head probe requires pinned runtime packages")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    sys.path.insert(0, str(args.upstream.resolve()))
    report = {
        "schema_version": 1,
        "mode": "head_counterfactual",
        "model_id": args.model,
        "revision": selected["revision"],
        "source_identity": model.manifest.identity,
        "upstream_revision": selected["upstream_code"]["revision"],
        "temperature": selected["temperature"],
        "reference_sha256": reference_digest,
        "runtime_sha256": hashlib.sha256(runtime_bytes).hexdigest(),
        "packages": packages,
        "tolerance": json.loads((ROOT / "compat/parity-profile.json").read_text()),
        "scope": "same-HF-hidden diagnostic counterfactual; no production precision change",
        "schedule": "one original unpadded HF forward per question, then head-only projections",
        "fp32_counterfactual": {
            "allow_tf32": False,
            "autocast": False,
            "operands": "original BF16 hidden/weights cast to FP32",
        },
        "cases": [],
    }
    probe(args, selected, model, reference, runtime, report)
    report["summary"] = aggregate(report["cases"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print("Saved head counterfactual report", args.output)


if __name__ == "__main__":
    main()
