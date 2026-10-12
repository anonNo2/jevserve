"""Small CLI; GPU dependencies are imported only for serving."""

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx

from jevserve.artifacts.catalog import DEFAULT_MODEL, model_ids, profile, source_manifest
from jevserve.artifacts.prepare import prepare_model
from jevserve.artifacts.verify import load_model_directory
from jevserve.client import Client


def sample_request() -> dict[str, Any]:
    return {
        "state": "I was charged twice and want a refund. No urgency.",
        "questions": {
            "team": {
                "type": "choice",
                "instructions": "Which team should handle this?",
                "criteria": {"technical": "Software and setup", "billing": "Payments and refunds"},
            },
            "urgent": {"type": "noul", "instructions": "Does this need an urgent response?"},
            "priority": {
                "type": "score",
                "instructions": "How urgent is this request?",
                "criteria": ["Routine", "Soon", "Immediate"],
            },
        },
    }


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="jevserve")
    sub = root.add_subparsers(dest="command", required=True)
    for name in ("serve", "prepare"):
        command = sub.add_parser(name)
        command.add_argument("--model", default=os.environ.get("JEVSERVE_MODEL", DEFAULT_MODEL))
        command.add_argument(
            "--model-dir",
            type=Path,
            help="Read-only pinned weight directory; verify locally without downloads",
        )
        command.add_argument(
            "--cache-dir",
            type=Path,
            default=Path(os.environ.get("JEVSERVE_CACHE_DIR", ".cache/jevserve")),
        )
        command.add_argument("--offline", action="store_true")
        command.add_argument("--token-file", type=Path)
        if name == "serve":
            command.add_argument(
                "--allow-uncertified",
                action="store_true",
                help="Explicitly run the development model profile; parity and container certification are incomplete",
            )
            command.add_argument("--host", default=os.environ.get("JEVSERVE_HOST", "127.0.0.1"))
            command.add_argument("--port", type=int, default=8000)
            command.add_argument("--api-key-file", type=Path)
            command.add_argument("--gpu-memory-utilization", type=float, default=0.5)
            command.add_argument(
                "--cpu-offload-gb",
                type=float,
                default=0,
                help="Explicit CPU weight offload budget; never enabled automatically",
            )
            command.add_argument("--max-active", type=int, default=16)
            command.add_argument("--max-waiting", type=int, default=64)
    models = sub.add_parser("models")
    models.add_argument("action", choices=["list"])
    sub.add_parser("doctor")
    for name in ("check", "benchmark"):
        command = sub.add_parser(name)
        command.add_argument("--url", default="http://127.0.0.1:8000")
        command.add_argument("--api-key-file", type=Path)
        if name == "check":
            command.add_argument(
                "--wait-seconds",
                type=int,
                default=0,
                help="Wait up to 1800 seconds for readiness before one inference check",
            )
        if name == "benchmark":
            command.add_argument("--requests", type=int, default=20)
            command.add_argument("--output", type=Path, required=True)
    return root


def wait_for_ready(http: httpx.Client, url: str, wait_seconds: int) -> None:
    """Poll readiness only; inference is still attempted exactly once."""
    if not 0 <= wait_seconds <= 1800:
        raise ValueError("readiness wait must be 0..1800 seconds")
    deadline = time.monotonic() + wait_seconds
    while True:
        try:
            remaining = deadline - time.monotonic()
            timeout = max(0.001, min(5.0, remaining)) if wait_seconds else 5.0
            response = http.get(url.rstrip("/") + "/readyz", timeout=timeout)
            if response.status_code != 503 or not wait_seconds:
                response.raise_for_status()
                return
        except httpx.TransportError:
            if not wait_seconds:
                raise
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("model did not become ready within the requested budget")
        time.sleep(min(1.0, remaining))


def run(args: argparse.Namespace) -> None:
    if args.command == "models":
        print(
            json.dumps(
                {
                    "models": [
                        {
                            "id": model_id,
                            "revision": profile(model_id)["revision"],
                            "adapter": profile(model_id)["adapter"],
                            "license": profile(model_id)["license"],
                            "status": "uncertified_development",
                        }
                        for model_id in model_ids()
                    ]
                },
                indent=2,
            )
        )
    elif args.command == "doctor":
        packages = {}
        for name in ("jevserve", "vllm", "torch", "transformers", "peft"):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = "not installed"
        try:
            gpu = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,memory.used,driver_version",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            observation = gpu.stdout.strip() if gpu.returncode == 0 else "unavailable"
        except (OSError, subprocess.TimeoutExpired):
            observation = "unavailable"
        print(
            json.dumps(
                {
                    "packages": packages,
                    "gpu_observation": observation,
                    "certification": "incomplete",
                },
                indent=2,
            )
        )
    elif args.command in ("prepare", "serve"):
        if args.command == "serve" and not args.allow_uncertified:
            raise ValueError(
                "No certified profile yet; development serving requires --allow-uncertified"
            )
        selected = profile(args.model)
        manifest = source_manifest(args.model)
        model = (
            load_model_directory(args.model_dir, manifest)
            if args.model_dir
            else prepare_model(
                manifest,
                args.cache_dir / "sources",
                offline=args.offline,
                token_file=args.token_file,
            )
        )
        if args.command == "prepare":
            print(
                json.dumps(
                    {
                        "model": model.manifest.model_id,
                        "identity": model.manifest.identity,
                        "path": str(model.path),
                    }
                )
            )
            return
        import uvicorn
        from transformers import AutoTokenizer

        from jevserve.adapters.open_jev import OpenJevAdapter
        from jevserve.config import ServerConfig
        from jevserve.contracts.capabilities import Capabilities
        from jevserve.runtime.base import Runtime
        from jevserve.runtime.vllm_backend import VllmRuntime
        from jevserve.server.app import create_app

        config = ServerConfig(
            host=args.host,
            port=args.port,
            api_key_file=args.api_key_file,
            max_active=args.max_active,
            max_waiting=args.max_waiting,
        )
        adapter: Any
        runtime: Runtime
        if selected["adapter"] == "open_jev":
            if args.cpu_offload_gb:
                raise ValueError("CPU offload is not supported by the Open-Jev pooling profile")
            tokenizer = AutoTokenizer.from_pretrained(
                model.path / "base", local_files_only=True, trust_remote_code=False
            )
            adapter = OpenJevAdapter(
                tokenizer,
                temperature=selected["temperature"],
                capabilities=Capabilities(
                    question_types=["choice", "noul", "score"],
                    max_choice_candidates=255,
                    max_score_levels=10,
                    max_tokens_per_candidate=4096,
                    max_candidate_sequences=256,
                    max_total_input_tokens=65536,
                ),
            )
            runtime = VllmRuntime(
                args.cache_dir, gpu_memory_utilization=args.gpu_memory_utilization
            )
            adapter_revision, execution_profile = "open-jev-v1", "qwen35-unmerged-v1"
        elif selected["adapter"] == "jpt":
            from jevserve.adapters.jpt import JptAdapter
            from jevserve.runtime.token_backend import VllmTokenRuntime

            tokenizer = AutoTokenizer.from_pretrained(
                model.path, local_files_only=True, trust_remote_code=False
            )
            adapter = JptAdapter(
                tokenizer,
                temperature=selected["temperature"],
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
            runtime = VllmTokenRuntime(
                args.cache_dir,
                gpu_memory_utilization=args.gpu_memory_utilization,
                cpu_offload_gb=args.cpu_offload_gb,
            )
            adapter_revision, execution_profile = "jpt-llm2jev-2b252d5", "qwen35-token-bf16-v1"
            if args.cpu_offload_gb:
                execution_profile += f"-offload-{args.cpu_offload_gb:g}g"
        elif selected["adapter"] in ("neohorse", "kev"):
            from jevserve.adapters.neohorse import NeoHorseAdapter
            from jevserve.runtime.torch_pointer import TorchPointerRuntime

            if args.cpu_offload_gb:
                raise ValueError("CPU offload is not supported by the pointer reference profile")
            tokenizer = AutoTokenizer.from_pretrained(
                model.path / ("tokenizer" if selected["adapter"] == "neohorse" else "base"),
                local_files_only=True,
                trust_remote_code=False,
            )
            adapter = NeoHorseAdapter(
                tokenizer,
                Capabilities(
                    question_types=["choice", "noul", "score"],
                    max_choice_candidates=255,
                    max_score_levels=255,
                    max_tokens_per_candidate=4096,
                    max_candidate_sequences=256,
                    max_total_input_tokens=8192,
                    max_padded_input_tokens=8192,
                    noul_candidate_cost=2,
                ),
                temperature=selected["temperature"],
            )
            runtime = TorchPointerRuntime(gpu_memory_utilization=args.gpu_memory_utilization)
            adapter_revision = selected["adapter"] + "-pointer-v1"
            execution_profile = "torch-pointer-" + (
                "fp32-v1" if selected["adapter"] == "kev" else "bf16-v1"
            )
        else:
            raise ValueError("model adapter is not implemented")
        engine_failed = False

        def fatal() -> None:
            nonlocal engine_failed
            engine_failed = True
            server.should_exit = True

        app = create_app(
            runtime,
            adapter,
            config,
            prepared_model=model,
            on_fatal=fatal,
            adapter_revision=adapter_revision,
            execution_profile=execution_profile,
            token_readout=selected["adapter"] == "jpt",
            pointer_readout=selected["adapter"] in ("neohorse", "kev"),
        )
        server = uvicorn.Server(
            uvicorn.Config(
                app,
                host=config.host,
                port=config.port,
                workers=1,
                access_log=False,
                timeout_graceful_shutdown=config.drain_seconds,
            )
        )
        server.run()
        if engine_failed or not server.started:
            raise RuntimeError("runtime startup or engine failed")
    elif args.command == "check":
        with httpx.Client(timeout=10) as http:
            wait_for_ready(http, args.url, args.wait_seconds)
        with Client(args.url, api_key_file=args.api_key_file) as client:
            print(client.decide(sample_request()).model_dump_json(indent=2))
    elif args.command == "benchmark":
        if not 1 <= args.requests <= 1000:
            raise ValueError("benchmark requests must be 1..1000")
        elapsed = []
        with Client(args.url, api_key_file=args.api_key_file) as client:
            for _ in range(args.requests):
                start = time.monotonic()
                response = client.decide(sample_request())
                elapsed.append(time.monotonic() - start)
        ordered = sorted(elapsed)
        report = {
            "requests": len(elapsed),
            "concurrency": 1,
            "total_seconds": sum(elapsed),
            "p50_seconds": ordered[(len(ordered) - 1) // 2],
            "p95_seconds": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
            "model_revision": response.model_revision,
            "execution_profile": response.execution_profile,
            "latency_seconds": elapsed,
        }
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in report.items() if k != "latency_seconds"}))


def main(argv: list[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if not values or (values[0].startswith("-") and values[0] not in ("-h", "--help")):
        values.insert(0, "serve")
    args = parser().parse_args(values)
    if args.command == "serve" and not args.allow_uncertified:
        print(
            "No certified model profile is available. For development, pass --allow-uncertified; see docs/development/STATUS.md for failed and pending gates.",
            file=sys.stderr,
        )
        return 2
    try:
        run(args)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        # Never echo network exceptions, tokens, requests or private filesystem paths.
        print(
            f"JevServe {args.command} failed ({type(exc).__name__}). Check configuration, artifact integrity and runtime diagnostics.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
