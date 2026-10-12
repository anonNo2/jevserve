"""Real pretokenized vLLM parity for the pinned compiled Open-Jev profile."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path

from jevserve.artifacts.catalog import source_manifest
from jevserve.artifacts.compile import verify_compiled


def main() -> int:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify_compiled(args.model_path, source_manifest().identity)
    root = args.cache_dir.resolve()
    os.environ.update(
        HF_HUB_OFFLINE="1",
        HF_HOME=str(root / "hf"),
        VLLM_CACHE_ROOT=str(root / "vllm"),
        TRITON_CACHE_DIR=str(root / "triton"),
        VLLM_PLUGINS="jevserve",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
        VLLM_NO_USAGE_STATS="1",
    )
    from vllm import LLM, PoolingParams

    reference_path = Path("validation/reports/open-jev-reference.json")
    reference = json.loads(reference_path.read_text())
    tolerance = json.loads(Path("compat/parity-profile.json").read_text())
    engine = LLM(
        model=str(args.model_path),
        runner="pooling",
        convert="none",
        dtype="bfloat16",
        max_model_len=4096,
        trust_remote_code=False,
        gpu_memory_utilization=0.5,
        enforce_eager=True,
        enable_prefix_caching=False,
        mamba_ssm_cache_dtype="float32",
        max_num_seqs=16,
        max_num_batched_tokens=4096,
        seed=0,
        disable_log_stats=True,
        pooler_config={"task": "classify", "seq_pooling_type": "LAST", "use_activation": False},
    )
    cases = []
    try:
        for case in reference["cases"]:
            prompts = [
                {"prompt_token_ids": [t for t, m in zip(row, mask, strict=True) if m]}
                for row, mask in zip(
                    case["encoding"]["input_ids"], case["encoding"]["attention_mask"], strict=True
                )
            ]
            output = engine.encode(
                prompts,
                pooling_task="classify",
                use_tqdm=False,
                pooling_params=PoolingParams(task="classify", use_activation=False),
            )
            scores = [float(value.outputs.data[0]) for value in output]
            assert all(math.isfinite(value) for value in scores)
            offset = 0
            probabilities = []
            maximum = 0.0
            passed = True
            flips = []
            for record, expected in zip(case["records"], case["probabilities"], strict=True):
                count = 1 if record["kind"] == "noul" else len(expected)
                raw = scores[offset : offset + count]
                offset += count
                if record["kind"] == "noul":
                    raw = [0.0, raw[0]]
                exps = [math.exp((v - max(raw)) / reference["temperature"]) for v in raw]
                actual = [v / sum(exps) for v in exps]
                probabilities.append(actual)
                for x, y in zip(actual, expected, strict=True):
                    maximum = max(maximum, abs(x - y))
                    passed &= abs(x - y) <= tolerance["probability_atol"] + tolerance[
                        "probability_rtol"
                    ] * abs(y)
                if max(range(len(actual)), key=actual.__getitem__) != max(
                    range(len(expected)), key=expected.__getitem__
                ):
                    flips.append(record["id"])
            cases.append(
                {
                    "id": case["id"],
                    "within_tolerance": passed,
                    "max_probability_error": maximum,
                    "scores": scores,
                    "probabilities": probabilities,
                    "label_flips": flips,
                }
            )
    finally:
        engine.llm_engine.engine_core.shutdown()
    report = {
        "case_count": len(cases),
        "failures": sum(not c["within_tolerance"] for c in cases),
        "max_probability_error": max(c["max_probability_error"] for c in cases),
        "label_flip_count": sum(len(c["label_flips"]) for c in cases),
        "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        "source_identity": source_manifest().identity,
        "packages": {
            n: importlib.metadata.version(n) for n in ("vllm", "torch", "transformers", "peft")
        },
        "execution_profile": "qwen35-unmerged-v1",
        "tolerance": tolerance,
        "cases": cases,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("cases", "tolerance")}))
    return 0 if report["failures"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
