"""Small reproducible model smoke/parity matrix. Reference and HTTP runs are separate GPU gates."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

from jevserve.artifacts.catalog import profile, source_manifest
from jevserve.artifacts.verify import load_model_directory
from jevserve.cli import sample_request


def requests():
    fixture = json.loads(Path("validation/reports/jpt-encoding.json").read_text())
    return [sample_request()] + [c["request"] for c in fixture["cases"]]


def verify_upstream(directory, revision):
    actual = subprocess.check_output(
        ["git", "-C", str(directory), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(directory), "status", "--porcelain", "--untracked-files=no"], text=True
    ).strip()
    if actual != revision or dirty:
        raise ValueError("upstream reference source must be clean and pinned")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["reference", "http"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--reference-wheel", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selected = profile(args.model)
    report = {
        "model_id": args.model,
        "revision": selected["revision"],
        "mode": args.mode,
        "precision": selected.get("dtype", "bfloat16"),
        "scope": "synthetic text smoke/parity, not model accuracy or certification",
        "cases": [],
    }
    if args.mode == "reference":
        if args.model_dir is None or args.upstream is None:
            parser.error("reference requires --model-dir and --upstream")
        model = load_model_directory(args.model_dir, source_manifest(args.model))
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        if selected["adapter"] == "jpt":
            verify_upstream(args.upstream, selected["upstream_code"]["revision"])
            sys.path.insert(0, str(args.upstream.resolve()))
            import torch
            from llm2jev.backends import HF
            from llm2jev.engine import LLM2Jev

            torch.set_num_threads(4)
            backend = HF(str(model.path), device="cuda", dtype="bfloat16")
            reference = LLM2Jev(
                backend.processor or backend.tok, backend, temperature=selected["temperature"]
            )

            def call(data):
                return reference(data["state"], data["questions"])
        elif selected["adapter"] == "neohorse":
            verify_upstream(args.upstream, selected["upstream_code"]["revision"])
            sys.path.insert(0, str(args.upstream.resolve() / "jev/package/src"))
            if args.reference_wheel:
                sys.path.append(str(args.reference_wheel.resolve()))
            from neohorse_decision._inference import load_bundle, predict

            tok, backbone = load_bundle(str(model.path))

            def call(data):
                return predict(tok, backbone, data, max_state=2048, max_branch=4096)["answers"]
        elif selected["adapter"] == "kev":
            provenance = json.loads((model.path / "checkpoint/provenance.json").read_text())
            # Check the vendored reference sources against the checkpoint's provenance.
            if provenance["git_commit"] != selected["upstream_code"]["revision"]:
                raise ValueError("Kev reference source revision mismatch")
            source_hashes = provenance["source_hashes"]
            required_sources = {"kev/model.py", "kev/api.py", "kev/checkpoint.py"}
            if not required_sources.issubset(source_hashes):
                raise ValueError("incomplete Kev reference source hashes")
            for name, expected in source_hashes.items():
                if name in ("kev/model.py", "kev/api.py", "kev/checkpoint.py"):
                    if hashlib.sha256((args.upstream / name).read_bytes()).hexdigest() != expected:
                        raise ValueError("Kev reference source hash mismatch")
            sys.path.insert(0, str(args.upstream.resolve()))
            import torch
            from kev.api import SystemOneRequest, to_answers, to_record
            from kev.checkpoint import Checkpoint, LoadOptions

            torch.set_num_threads(4)
            torch.backends.cuda.matmul.allow_tf32 = False
            checkpoint = Checkpoint(str(model.path / "checkpoint"))
            checkpoint.meta.base = str(model.path / "base")
            checkpoint.meta.base_revision = None
            tok, backbone = checkpoint.load("cuda", LoadOptions(dtype=torch.float32, attn="sdpa"))

            def call(data):
                record, meta = to_record(SystemOneRequest(**data))
                encoded = backbone.encode(tok, record, strict=True, max_state=2048, max_branch=4096)
                with torch.inference_mode():
                    probs = [p.tolist() for p in backbone.probs(encoded)]
                return to_answers(probs, meta)
        else:
            parser.error("use validation.reference for scalar Open-Jev models")
        report["upstream_revision"] = selected["upstream_code"]["revision"]
        report["packages"] = {
            name: importlib.metadata.version(name) for name in ["torch", "transformers"]
        }
        for index, data in enumerate(requests()):
            start = time.monotonic()
            result = call(data)
            report["cases"].append(
                {
                    "case": index,
                    "request": data,
                    "answers": result,
                    "elapsed_seconds": round(time.monotonic() - start, 4),
                }
            )
            print("Reference case", index, "passed", flush=True)
    else:
        reference = json.loads(args.reference.read_text()) if args.reference else None
        tolerance = json.loads(Path("compat/parity-profile.json").read_text())
        report["tolerance_rule"] = "abs(actual-reference) <= atol + rtol*abs(reference)"
        report["probability_atol"] = tolerance["probability_atol"]
        report["probability_rtol"] = tolerance["probability_rtol"]
        failures, flips = 0, 0
        with httpx.Client(timeout=35) as client:
            client.get(args.url + "/readyz").raise_for_status()
            served = client.get(args.url + "/v1/models")
            served.raise_for_status()
            assert served.json()["models"][0]["id"] == args.model
            for index, data in enumerate(requests()):
                start = time.monotonic()
                response = client.post(args.url + "/v1/decisions", json=data)
                row = {
                    "case": index,
                    "status": response.status_code,
                    "elapsed_seconds": round(time.monotonic() - start, 4),
                }
                if response.is_error:
                    row["error"] = response.json()
                    report["cases"].append(row)
                    args.output.parent.mkdir(exist_ok=True, parents=True)
                    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
                response.raise_for_status()
                result = response.json()
                assert result["model_revision"] == selected["revision"]
                row["answers"] = result["answers"]
                row["execution_profile"] = result["execution_profile"]
                row["max_probability_error"] = 0.0
                row["parity_passed"] = None
                if reference:
                    assert (
                        reference["model_id"] == args.model
                        and reference["revision"] == selected["revision"]
                    )
                    expected = reference["cases"][index]["answers"]
                    passed = True
                    for qid, answer in result["answers"].items():
                        other = expected[qid]
                        left = (
                            list(answer["probabilities"].values())
                            if "probabilities" in answer
                            else [answer["noul"]]
                        )
                        right = (
                            list(other["probabilities"].values())
                            if answer["type"] != "noul"
                            else [other["noul"]]
                        )
                        if answer["type"] != "noul":
                            assert list(answer["probabilities"]) == list(other["probabilities"])
                        if len(left) != len(right):
                            raise ValueError("candidate count differs")
                        differences = [abs(x - y) for x, y in zip(left, right, strict=True)]
                        row["max_probability_error"] = max(
                            row["max_probability_error"], max(differences)
                        )
                        passed &= all(
                            abs(x - y)
                            <= tolerance["probability_atol"]
                            + tolerance["probability_rtol"] * abs(y)
                            for x, y in zip(left, right, strict=True)
                        )
                        if answer["type"] == "choice":
                            flips += answer["choice"] != other["choice"]
                    row["parity_passed"] = passed
                    failures += not passed
                report["cases"].append(row)
                print("HTTP case", index, "executed; parity:", row["parity_passed"], flush=True)
            sample = sample_request() | {"model": args.model}
            response = client.post(args.url + "/v1/systemone", json=sample)
            response.raise_for_status()
            assert response.json()["model"] == args.model
            report["systemone_smoke"] = "passed"
            wrong = client.post(
                args.url + "/v1/systemone", json=sample | {"model": "unconfigured/model"}
            )
            assert wrong.status_code == 422
            report["unknown_model_rejected"] = True
        from typesafe_sdk import Choice, Noul, RetryPolicy, Score, TypeSafeClient

        from jevserve.client import Client

        with Client(args.url) as native:
            result = native.decide(sample_request())
            assert result.model_revision == selected["revision"]
        report["native_client_smoke"] = "passed"
        with TypeSafeClient(
            api_key="local-validation",
            model=args.model,
            base_url=args.url,
            retry=RetryPolicy(max_retries=0),
        ) as sdk:
            result = sdk.system_one(
                state="I was charged twice and want a refund.",
                questions={
                    "team": Choice(
                        instructions="Choose a team",
                        criteria={"billing": "Payments", "technical": "Software"},
                    ),
                    "urgent": Noul(instructions="Is this urgent?"),
                    "priority": Score(instructions="Rate urgency", criteria=["Low", "High"]),
                },
            )
            assert result.model == args.model
            assert result.choices["team"].choice in ("billing", "technical")
            assert 0 <= result.nouls["urgent"].noul <= 1
            assert 0 <= result.scores["priority"].score <= 1
        report["official_sdk_smoke"] = "typesafe-sdk==" + importlib.metadata.version("typesafe-sdk")
        report["parity_failed_cases"] = failures if reference else None
        report["choice_label_flips"] = flips if reference else None
    args.output.parent.mkdir(exist_ok=True, parents=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("Saved", args.output, flush=True)


if __name__ == "__main__":
    main()
