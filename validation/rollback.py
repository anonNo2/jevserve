"""Serial old/new/old Open-Jev-2B container execution drill using an existing cache."""

import argparse
import fcntl
import hashlib
import json
import re
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

import httpx

from jevserve.artifacts.catalog import profile
from jevserve.contracts.requests import DecisionRequest
from jevserve.contracts.responses import DecisionResponse
from validation.run import ROOT, ValidationFailure, assert_idle, call, checked, interrupted

MODEL = "ZefanCai/Open-Jev-2B"
LABEL = "org.jevserve.rollback"


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--docker-host", required=True)
    result.add_argument(
        "--from-image", required=True, help="Locally available repository@sha256:digest"
    )
    result.add_argument(
        "--to-image", required=True, help="Locally available repository@sha256:digest"
    )
    result.add_argument(
        "--cache-volume", required=True, help="Existing prepared Open-Jev-2B volume"
    )
    result.add_argument("--output-dir", required=True, type=Path)
    result.add_argument("--port", required=True, type=int)
    result.add_argument(
        "--timeout", type=int, default=1800, help="Per-stage readiness timeout seconds"
    )
    return result


def validate(args):
    for image in (args.from_image, args.to_image):
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._/:-]*@sha256:[a-f0-9]{64}", image):
            raise ValueError("pinned repository digest required")
    if args.from_image.rsplit("@", 1)[1] == args.to_image.rsplit("@", 1)[1]:
        raise ValueError("two distinct image digests required")
    if not args.docker_host.startswith("unix:///") or any(
        char in args.docker_host for char in "\r\n\0"
    ):
        raise ValueError("explicit local Docker Unix socket required")
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}", args.cache_volume):
        raise ValueError("existing named cache volume required")
    if not 1024 <= args.port <= 65535 or not 1 <= args.timeout <= 1800:
        raise ValueError("port or timeout out of range")
    args.output_dir = args.output_dir.absolute()


def write_summary(path, summary):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n")
    temporary.replace(path)


def container_command(args, name, image):
    return [
        "docker",
        "--host",
        args.docker_host,
        "run",
        "--detach",
        "--pull=never",
        "--name",
        name,
        "--label",
        f"{LABEL}={name}",
        "--gpus",
        "device=0",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,size=2g",
        "--shm-size",
        "2g",
        "--publish",
        f"127.0.0.1:{args.port}:8000",
        "--mount",
        f"type=volume,src={args.cache_volume},dst=/cache",
        image,
        "serve",
        "--allow-uncertified",
        "--offline",
        "--host",
        "0.0.0.0",
        "--cache-dir",
        "/cache",
    ]


def cleanup(docker, name):
    """Remove only an exact container whose unique ownership label matches."""
    result = {"status": "failed", "container_exit_code": None, "stop_returncode": None}
    try:
        raw = call(docker + ["inspect", "--format", "{{json .Config.Labels}}", name])
        if raw.returncode:
            # Distinguish an absent container from daemon/inspect failure.
            ids = checked(docker + ["ps", "--all", "--quiet", "--filter", f"name=^/{name}$"])
            if not ids:
                result["status"] = "not_created"
            return result
        if json.loads(raw.stdout).get(LABEL) != name:
            result["error_code"] = "ownership_mismatch"
            return result
        try:
            stopped = call(docker + ["stop", "--time", "65", name], timeout=90)
            result["stop_returncode"] = stopped.returncode
            state = json.loads(checked(docker + ["inspect", "--format", "{{json .State}}", name]))
            result["container_exit_code"] = state.get("ExitCode")
            if (
                stopped.returncode == 0
                and state.get("Running") is False
                and state.get("ExitCode") == 0
            ):
                result["status"] = "passed"
        finally:
            removed = call(docker + ["rm", "--force", name], timeout=90)
            result["remove_returncode"] = removed.returncode
            if removed.returncode:
                result["status"] = "failed"
    except Exception:
        result["status"] = "failed"
        result["error_code"] = "cleanup_failed"
    return result


def sample(port, request):
    with httpx.Client(
        base_url=f"http://127.0.0.1:{port}", timeout=35, trust_env=False, follow_redirects=False
    ) as client:
        metadata = client.get("/v1/models")
        metadata.raise_for_status()
        models = metadata.json()["models"]
        if (
            len(models) != 1
            or models[0]["id"] != MODEL
            or models[0]["revision"] != profile(MODEL)["revision"]
        ):
            raise ValidationFailure("model_identity_mismatch")
        response = client.post("/v1/decisions", json=request.model_dump(exclude_none=True))
        response.raise_for_status()
        answer = DecisionResponse.model_validate(response.json())
    if answer.model_revision != models[0]["revision"] or list(answer.answers) != list(
        request.questions
    ):
        raise ValidationFailure("response_identity_mismatch")
    for key, question in request.questions.items():
        value = answer.answers[key]
        if value.type != question.type:
            raise ValidationFailure("response_type_mismatch")
        if value.type == "choice" and list(value.probabilities) != list(question.criteria):
            raise ValidationFailure("response_candidates_mismatch")
        if value.type == "score" and list(value.legend.values()) != question.criteria:
            raise ValidationFailure("response_rubric_mismatch")
    return models[0], answer.model_dump()


def compare(first, second):
    """Descriptive cross-run differences; deliberately no author parity threshold."""
    maximum = 0.0
    choice_changes = []
    score_deltas = {}
    for key, before in first["answers"].items():
        after = second["answers"][key]
        if before["type"] == "noul":
            maximum = max(maximum, abs(before["noul"] - after["noul"]))
        else:
            for label, probability in before["probabilities"].items():
                maximum = max(maximum, abs(probability - after["probabilities"][label]))
            if before["type"] == "choice" and before["choice"] != after["choice"]:
                choice_changes.append(key)
            if before["type"] == "score":
                score_deltas[key] = after["score"] - before["score"]
    return {
        "max_absolute_probability_difference": maximum,
        "choice_changes": choice_changes,
        "score_deltas": score_deltas,
        "author_parity": "not_measured",
    }


def error_code(error):
    if isinstance(error, ValidationFailure):
        return error.code
    if isinstance(error, KeyboardInterrupt):
        return "interrupted"
    if isinstance(error, subprocess.TimeoutExpired):
        return "timeout"
    if isinstance(error, BlockingIOError):
        return "validation_locked"
    if isinstance(error, httpx.HTTPStatusError):
        return f"http_{error.response.status_code}"
    return "execution_error"


def execute(args):
    validate(args)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summary_path = args.output_dir / "summary.json"
    summary = {
        "schema_version": 1,
        "report_type": "jevserve-rollback",
        "status": "running",
        "model_id": MODEL,
        "model_revision": profile(MODEL)["revision"],
        "scope": "serial cached old/new/old execution; not accuracy, author parity, or certification",
        "real_gpu": False,
        "stage": "preflight",
        "stages": [],
        "comparisons": {},
        "input_cache_preserved": True,
    }
    write_summary(summary_path, summary)
    docker = ["docker", "--host", args.docker_host]
    lock = None
    try:
        (ROOT / ".local").mkdir(exist_ok=True)
        lock = (ROOT / ".local/validation-gpu.lock").open("a")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        checked(docker + ["volume", "inspect", args.cache_volume])
        if checked(docker + ["ps", "--quiet", "--filter", f"volume={args.cache_volume}"]):
            raise ValidationFailure("cache_volume_in_use")
        images = {}
        for image in (args.from_image, args.to_image):
            image_id = checked(docker + ["image", "inspect", "--format", "{{.Id}}", image])
            if not re.fullmatch(r"sha256:[a-f0-9]{64}", image_id):
                raise ValidationFailure("invalid_image_identity")
            images[image] = image_id
        if images[args.from_image] == images[args.to_image]:
            raise ValidationFailure("identical_image_ids")
        raw = (ROOT / "examples/decision.json").read_bytes()
        request = DecisionRequest.from_json(raw)
        summary["request_sha256"] = hashlib.sha256(raw).hexdigest()
        summary["source_commit"] = checked(["git", "rev-parse", "HEAD"])
        summary["source_dirty"] = bool(checked(["git", "status", "--porcelain"]))
        run_id = uuid.uuid4().hex
        responses = []
        for index, (label, image) in enumerate(
            (
                ("old_before", args.from_image),
                ("new", args.to_image),
                ("old_after", args.from_image),
            )
        ):
            summary["stage"] = label
            stage = {
                "name": label,
                "status": "running",
                "image": image,
                "image_digest": image.rsplit("@", 1)[1],
                "image_id": images[image],
                "startup_to_readiness_seconds": None,
                "readiness_returncode": None,
                "cleanup": {"status": "not_needed"},
            }
            summary["stages"].append(stage)
            name = f"jevserve-rollback-{run_id}-{index}"
            stage["container_name"] = name
            attempted = False
            started = None
            try:
                assert_idle()
                stage["gpu_idle_preflight"] = "passed"
                write_summary(summary_path, summary)
                attempted = True
                started = time.monotonic()
                checked(container_command(args, name, image), timeout=120)
                actual = checked(docker + ["inspect", "--format", "{{.Image}}", name])
                stage["container_image_id"] = actual
                if actual != images[image]:
                    raise ValidationFailure("container_image_mismatch")
                ready = call(
                    docker
                    + ["exec", name, "jevserve", "check", "--wait-seconds", str(args.timeout)],
                    timeout=args.timeout + 30,
                )
                stage["readiness_returncode"] = ready.returncode
                stage["startup_elapsed_seconds"] = time.monotonic() - started
                if ready.returncode:
                    raise ValidationFailure("readiness_failed")
                stage["startup_to_readiness_seconds"] = stage["startup_elapsed_seconds"]
                metadata, answer = sample(args.port, request)
                stage["model"] = metadata
                artifact = args.output_dir / f"{label}-sample.json"
                artifact.write_text(json.dumps(answer, indent=2) + "\n")
                stage["sample"] = {
                    "file": artifact.name,
                    "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                }
                responses.append(answer)
                stage["status"] = "passed"
                summary["real_gpu"] = True
            except (Exception, KeyboardInterrupt) as error:
                stage["status"] = "failed"
                stage["error_code"] = error_code(error)
                if started is not None:
                    stage.setdefault("startup_elapsed_seconds", time.monotonic() - started)
                raise
            finally:
                handlers = {
                    sig: signal.signal(sig, signal.SIG_IGN)
                    for sig in (signal.SIGINT, signal.SIGTERM)
                }
                try:
                    if attempted:
                        stage["cleanup"] = cleanup(docker, name)
                        if stage["cleanup"]["status"] != "passed":
                            stage["status"] = "failed"
                    write_summary(summary_path, summary)
                finally:
                    for sig, handler in handlers.items():
                        signal.signal(sig, handler)
            if stage["status"] != "passed":
                raise ValidationFailure("cleanup_failed")
        summary["comparisons"] = {
            "old_to_new": compare(responses[0], responses[1]),
            "old_before_to_old_after": compare(responses[0], responses[2]),
        }
        summary.update(status="passed", stage="complete")
    except (Exception, KeyboardInterrupt) as error:
        summary.update(status="failed", error_code=error_code(error))
    finally:
        if lock:
            lock.close()
        write_summary(summary_path, summary)
    print(json.dumps({"status": summary["status"], "error_code": summary.get("error_code")}))
    return 0 if summary["status"] == "passed" else 1


def main(argv=None):
    args = parser().parse_args(argv)
    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        return execute(args)
    except (ValueError, OSError):
        print("error_code=invalid_configuration_or_output", file=sys.stderr)
        return 2
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    raise SystemExit(main())
