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

from jevserve.artifacts.catalog import DEFAULT_MODEL, baseline, source_manifest
from jevserve.artifacts.prepare import prepare_model
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
        command.add_argument("--model", default=DEFAULT_MODEL)
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
            command.add_argument("--max-active", type=int, default=16)
            command.add_argument("--max-waiting", type=int, default=64)
    models = sub.add_parser("models")
    models.add_argument("action", choices=["list"])
    sub.add_parser("doctor")
    for name in ("check", "benchmark"):
        command = sub.add_parser(name)
        command.add_argument("--url", default="http://127.0.0.1:8000")
        command.add_argument("--api-key-file", type=Path)
        if name == "benchmark":
            command.add_argument("--requests", type=int, default=20)
            command.add_argument("--output", type=Path, required=True)
    return root


def run(args: argparse.Namespace) -> None:
    if args.command == "models":
        lock = baseline()
        print(
            json.dumps(
                {
                    "models": [
                        {
                            "id": lock["model_id"],
                            "revision": lock["revision"],
                            "status": "uncertified_development",
                        }
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
        model = prepare_model(
            source_manifest(args.model),
            args.cache_dir / "sources",
            offline=args.offline,
            token_file=args.token_file,
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
        from jevserve.runtime.vllm_backend import VllmRuntime
        from jevserve.server.app import create_app

        config = ServerConfig(
            host=args.host,
            port=args.port,
            api_key_file=args.api_key_file,
            max_active=args.max_active,
            max_waiting=args.max_waiting,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model.path / "base", local_files_only=True, trust_remote_code=False
        )
        adapter = OpenJevAdapter(
            tokenizer,
            temperature=baseline()["temperature"],
            capabilities=Capabilities(
                question_types=["choice", "noul", "score"],
                max_choice_candidates=255,
                max_score_levels=10,
                max_tokens_per_candidate=4096,
                max_candidate_sequences=256,
                max_total_input_tokens=65536,
            ),
        )
        runtime = VllmRuntime(args.cache_dir, gpu_memory_utilization=args.gpu_memory_utilization)
        engine_failed = False

        def fatal() -> None:
            nonlocal engine_failed
            engine_failed = True
            server.should_exit = True

        app = create_app(runtime, adapter, config, prepared_model=model, on_fatal=fatal)
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
            http.get(args.url.rstrip("/") + "/readyz").raise_for_status()
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
