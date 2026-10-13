"""Bounded HTTP workload/soak runner; measures service execution, not model accuracy."""

import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from jevserve.contracts.responses import DecisionResponse


def percentile(values, fraction):
    if not values:
        return None
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)]


def request_body(kind, candidates):
    question = {"type": kind, "instructions": "Classify the urgency of this support request."}
    if kind == "choice":
        question["criteria"] = {f"level_{i}": f"Urgency level {i}" for i in range(candidates)}
    elif kind == "score":
        question["criteria"] = [f"Urgency level {i}" for i in range(candidates)]
    return {
        "state": "The payment failed. Please help before tomorrow.",
        "questions": {"q": question},
    }


def snapshot(path, report):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, path)


def gpu_memory():
    try:
        value = subprocess.check_output(
            ["nvidia-smi", "--id=0", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            timeout=5,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return int(value)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


async def measure(client, body, expected_revision):
    start = time.monotonic()
    try:
        response = await client.post("/v1/decisions", json=body)
        elapsed = time.monotonic() - start
        if response.status_code != 200:
            return elapsed, f"http_{response.status_code}"
        data = DecisionResponse.model_validate(response.json())
        if data.model_revision != expected_revision:
            return elapsed, "model_revision_mismatch"
        if list(data.answers) != list(body["questions"]):
            return elapsed, "invalid_answer"
        for key, question in body["questions"].items():
            answer = data.answers[key]
            if answer.type != question["type"]:
                return elapsed, "invalid_answer"
            if answer.type == "choice" and list(answer.probabilities) != list(question["criteria"]):
                return elapsed, "invalid_candidates"
            if answer.type == "score" and list(answer.legend.values()) != question["criteria"]:
                return elapsed, "invalid_score_rubric"
        return elapsed, None
    except httpx.HTTPError:
        return time.monotonic() - start, "transport_error"
    except (ValueError, TypeError, AttributeError):
        return time.monotonic() - start, "invalid_response"


async def run(args):
    if not re.fullmatch(r"sha256:[a-f0-9]{64}", args.image_digest):
        raise ValueError("registry digest required")
    if not math.isfinite(args.duration) or not math.isfinite(args.interval):
        raise ValueError("finite duration and interval required")
    if args.output.exists():
        raise ValueError("output already exists; use a fresh output path")
    if not 1 <= args.concurrency <= 16 or not 1 <= args.requests <= 1000000:
        raise ValueError("concurrency or request limit out of range")
    if not 1 <= args.duration <= 259200 or not 0 <= args.interval <= 3600:
        raise ValueError("duration/interval out of range")
    if not 2 <= args.candidates <= 255:
        raise ValueError("candidate count out of range")
    url = urlsplit(args.url)
    if (
        url.scheme not in ("http", "https")
        or not url.hostname
        or url.username is not None
        or url.password is not None
        or url.query
        or url.fragment
        or url.path not in ("", "/")
    ):
        raise ValueError("invalid origin URL")
    # Accessing port rejects malformed port strings before any network request.
    _ = url.port
    body = request_body(args.kind, args.candidates)
    report = {
        "schema_version": 1,
        "scope": "bounded synthetic HTTP execution; no accuracy/parity certification",
        "model": None,
        "image_digest": args.image_digest,
        "request_sha256": hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest(),
        "question_type": args.kind,
        "candidates": args.candidates if args.kind != "noul" else 2,
        "concurrency": args.concurrency,
        "interval_seconds": args.interval,
        "requested_duration_seconds": args.duration,
        "request_limit": args.requests,
        "warmup_seconds": None,
        "status": "running",
        "termination_reason": None,
        "stage": "authentication",
        "error_code": None,
        "duration_gate_passed": False,
        "elapsed_seconds": 0,
        "completed_requests": 0,
        "successful_requests": 0,
        "failures": {},
        "throughput_requests_per_second": None,
        "gpu_memory_note": "sampled GPU 0 device total incl runtime reservation; not minimum model VRAM",
    }
    snapshot(args.output, report)
    start = None
    latencies = []
    failures = {}
    try:
        headers = {}
        if args.api_key_file:
            key = args.api_key_file.read_text().strip()
            if not key or any(ord(char) < 33 or ord(char) > 126 for char in key):
                raise ValueError("invalid key file")
            headers["Authorization"] = "Bearer " + key
        async with httpx.AsyncClient(base_url=args.url, headers=headers, timeout=60) as client:
            report["stage"] = "readiness"
            health = await client.get("/readyz")
            health.raise_for_status()
            report["stage"] = "metadata"
            metadata = await client.get("/v1/models")
            metadata.raise_for_status()
            model = metadata.json()["models"][0]
            if (
                not isinstance(model, dict)
                or not isinstance(model.get("id"), str)
                or not isinstance(model.get("revision"), str)
                or not re.fullmatch(r"[a-f0-9]{40}", model["revision"])
            ):
                raise ValueError("invalid model metadata")
            report["model"] = model
            expected_revision = model["revision"]
            report["stage"] = "warmup"
            warmup, warmup_error = await measure(client, body, expected_revision)
            report["warmup_seconds"] = warmup
            if warmup_error:
                report.update(
                    status="failed", termination_reason="warmup_failed", error_code=warmup_error
                )
                return 1
            baseline = gpu_memory() if args.observe_gpu else None
            peak = baseline
            report["stage"] = "workload"
            start = time.monotonic()
            while len(latencies) < args.requests and time.monotonic() - start < args.duration:
                count = min(args.concurrency, args.requests - len(latencies))
                results = await asyncio.gather(
                    *(measure(client, body, expected_revision) for _ in range(count))
                )
                for latency, failure in results:
                    latencies.append(latency)
                    if failure:
                        failures[failure] = failures.get(failure, 0) + 1
                elapsed = time.monotonic() - start
                observed = gpu_memory() if args.observe_gpu else None
                if observed is not None:
                    peak = max(peak or 0, observed)
                report.update(
                    elapsed_seconds=elapsed,
                    completed_requests=len(latencies),
                    failures=failures,
                    successful_requests=len(latencies) - sum(failures.values()),
                    throughput_requests_per_second=len(latencies) / elapsed,
                    latency_seconds={
                        "p50": percentile(latencies, 0.5),
                        "p95": percentile(latencies, 0.95),
                        "p99": percentile(latencies, 0.99),
                        "max": max(latencies),
                    },
                    sampled_gpu_baseline_mib=baseline,
                    sampled_gpu_peak_mib=peak,
                )
                snapshot(args.output, report)
                if failures:
                    report["termination_reason"] = "failed_batch"
                    report["error_code"] = "failed_batch"
                    break
                if len(latencies) >= args.requests and elapsed < args.duration:
                    report["termination_reason"] = "request_limit"
                    break
                await asyncio.sleep(min(args.interval, max(0, args.duration - elapsed)))
            elapsed = time.monotonic() - start
            report["status"] = "failed" if failures else "passed"
            report["duration_gate_passed"] = (
                not failures
                and report["termination_reason"] != "request_limit"
                and elapsed >= args.duration
            )
            report["termination_reason"] = report["termination_reason"] or "duration"
    except asyncio.CancelledError:
        report.update(status="interrupted", termination_reason="signal", error_code="interrupted")
        raise
    except httpx.HTTPStatusError as exc:
        report.update(
            status="failed",
            termination_reason="preflight_failed",
            error_code=f"http_{exc.response.status_code}",
        )
    except httpx.HTTPError:
        report.update(
            status="failed", termination_reason="transport_failed", error_code="transport_error"
        )
    except OSError:
        report.update(status="failed", termination_reason="io_failed", error_code="io_error")
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, httpx.InvalidURL):
        report.update(
            status="failed",
            termination_reason="invalid_input_or_response",
            error_code="invalid_input_or_response",
        )
    finally:
        if start is not None:
            elapsed = time.monotonic() - start
            report["elapsed_seconds"] = elapsed
            report["throughput_requests_per_second"] = len(latencies) / elapsed if elapsed else None
        if report["status"] == "running":
            report.update(status="interrupted", termination_reason="interrupted")
        if report["status"] != "passed":
            report["duration_gate_passed"] = False
        snapshot(args.output, report)
    return int(report["status"] != "passed")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--kind", choices=("choice", "noul", "score"), default="choice")
    parser.add_argument("--candidates", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--duration", type=float, default=300)
    parser.add_argument("--interval", type=float, default=0)
    parser.add_argument("--observe-gpu", action="store_true")
    args = parser.parse_args(argv)

    async def controlled():
        task = asyncio.current_task()
        asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, task.cancel)
        try:
            return await run(args)
        finally:
            asyncio.get_running_loop().remove_signal_handler(signal.SIGTERM)

    try:
        status = asyncio.run(controlled())
    except (asyncio.CancelledError, KeyboardInterrupt):
        print("error_code=interrupted", file=sys.stderr)
        return 130
    except (ValueError, TypeError, httpx.InvalidURL):
        print("error_code=invalid_configuration", file=sys.stderr)
        return 2
    except OSError:
        print("error_code=io_error", file=sys.stderr)
        return 2
    if status:
        print("error_code=workload_failed", file=sys.stderr)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
