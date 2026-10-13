"""Serial, offline author-reference and container validation (Linux, source checkout)."""

import argparse
import fcntl
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import uuid
from pathlib import Path

from jevserve.artifacts.catalog import model_ids, profile, source_manifest
from jevserve.artifacts.verify import load_model_directory

ROOT = Path(__file__).resolve().parents[1]


class ValidationFailure(RuntimeError):
    """Stable public failure code; detailed subprocess output remains in private logs."""

    def __init__(self, code):
        super().__init__(code)
        self.code = code


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=model_ids(), required=True)
    for key in ("model-dir", "upstream", "reference-python", "output-dir"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--image", required=True, help="Locally available repository@sha256:digest")
    p.add_argument("--docker-host", required=True, help="Explicit local unix:// Docker socket")
    p.add_argument("--reference-wheel", type=Path)
    p.add_argument("--reference-wheel-sha256")
    p.add_argument("--port", type=int, default=18792)
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument(
        "--plan", action="store_true", help="Print local commands; do not execute or verify bytes"
    )
    return p


def validate(args):
    if not re.fullmatch(r"[^\s]+@sha256:[a-f0-9]{64}", args.image):
        raise ValueError("image must be pinned by repository digest")
    if not args.docker_host.startswith("unix:///"):
        raise ValueError("only a local unix:/// Docker host is supported")
    if not 1024 <= args.port <= 65535 or not 1 <= args.timeout <= 1800:
        raise ValueError("port or timeout out of range")
    for key in ("model_dir", "upstream", "reference_python", "output_dir"):
        value = getattr(args, key).absolute()
        if any(char in str(value) for char in ",\n\r"):
            raise ValueError("paths cannot contain commas or newlines")
        setattr(args, key, value)
    if profile(args.model)["adapter"] == "neohorse":
        if not args.reference_wheel or not re.fullmatch(
            "[a-f0-9]{64}", args.reference_wheel_sha256 or ""
        ):
            raise ValueError("NeoHorse requires a reference wheel and its expected SHA-256")
        args.reference_wheel = args.reference_wheel.absolute()


def commands(args, name):
    out = args.output_dir
    scalar = profile(args.model)["adapter"] == "open_jev"
    reference = [str(args.reference_python), "-m"]
    if scalar:
        reference += [
            "validation.reference",
            "--model",
            args.model,
            "--upstream",
            str(args.upstream),
            "--checkpoint",
            str(args.model_dir / "checkpoint"),
            "--base",
            str(args.model_dir / "base"),
            "--cache",
            str(out / "reference-cache"),
        ]
    else:
        reference += [
            "validation.model_matrix",
            "reference",
            "--model",
            args.model,
            "--upstream",
            str(args.upstream),
            "--model-dir",
            str(args.model_dir),
            "--reference-cache",
            str(out / "reference-cache"),
        ]
        if args.reference_wheel:
            reference += ["--reference-wheel", str(args.reference_wheel)]
    reference += ["--output", str(out / "reference.json")]
    docker = ["docker", "--host", args.docker_host]
    container = docker + [
        "run",
        "--detach",
        "--pull=never",
        "--name",
        name,
        "--label",
        "org.jevserve.validation=" + name,
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
        f"type=bind,src={args.model_dir},dst=/model,readonly",
        "--mount",
        f"type=volume,src={name},dst=/cache",
        args.image,
        "serve",
        "--allow-uncertified",
        "--model",
        args.model,
        "--model-dir",
        "/model",
        "--offline",
        "--host",
        "0.0.0.0",
    ]
    http = [sys.executable, "-m"]
    http += (
        ["validation.http"]
        if scalar
        else ["validation.model_matrix", "http", "--model", args.model]
    )
    http += [
        "--url",
        f"http://127.0.0.1:{args.port}",
        "--reference",
        str(out / "reference.json"),
        "--output",
        str(out / "http.json"),
    ]
    return {
        "reference": reference,
        "container": container,
        "readiness": docker
        + ["exec", name, "jevserve", "check", "--wait-seconds", str(args.timeout)],
        "http": http,
    }


def call(command, *, timeout=60, env=None, log=None):
    stream = log.open("w") if log else None
    try:
        with subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            text=True,
            start_new_session=True,
            stdout=stream or subprocess.PIPE,
            stderr=subprocess.STDOUT if stream else subprocess.PIPE,
        ) as process:
            try:
                stdout, stderr = process.communicate(timeout=timeout)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                os.killpg(process.pid, signal.SIGKILL)
                process.communicate()
                raise
            return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    finally:
        if stream:
            stream.close()


def checked(command, **kwargs):
    result = call(command, **kwargs)
    if result.returncode:
        raise ValidationFailure("subprocess_failed")
    return result.stdout.strip() if result.stdout else ""


def assert_idle():
    active = checked(["nvidia-smi", "--id=0", "--query-compute-apps=pid", "--format=csv,noheader"])
    if active:
        raise ValidationFailure("gpu_busy")


def execute(args):
    """Write summary even on failure; return 0 success, 1 parity failure, 2 execution failure."""
    validate(args)
    name = "jevserve-validation-" + uuid.uuid4().hex
    plan = commands(args, name)
    if args.plan:
        print(json.dumps(plan, indent=2))
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=False)  # Never reuse stale reports.
    (args.output_dir / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    summary = {
        "schema_version": 1,
        "report_type": "jevserve-validation",
        "real_gpu": False,
        "model_id": args.model,
        "model_revision": profile(args.model)["revision"],
        "image": args.image,
        "image_digest": args.image.rsplit("@", 1)[1],
        "models": [{"model_id": args.model, "revision": profile(args.model)["revision"]}],
        "source_manifest_identity": source_manifest(args.model).identity,
        "scope": "synthetic text execution/parity only; no certification or vision gate",
        "execution": "failed",
        "parity": "not_run",
        "cleanup": "not_needed",
        "stage": "preflight",
    }
    docker = ["docker", "--host", args.docker_host]
    env = os.environ | {
        "PYTHONPATH": str(ROOT / "src") + os.pathsep + str(ROOT),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "OMP_NUM_THREADS": "4",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    attempted = False
    lock = None
    try:
        if any(key.startswith("JEV_") for key in os.environ):
            raise ValueError("unset JEV_* overrides before validation")
        (ROOT / ".local").mkdir(exist_ok=True)
        lock = (ROOT / ".local/validation-gpu.lock").open("w")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        summary["source_commit"] = checked(["git", "rev-parse", "HEAD"])
        summary["source_dirty"] = bool(checked(["git", "status", "--porcelain"]))
        summary["source_hashes"] = {
            str(p.relative_to(ROOT)): sha256(p)
            for p in sorted(
                [
                    *(ROOT / "src").rglob("*.py"),
                    *(ROOT / "validation").glob("*.py"),
                    ROOT / "compat/parity-profile.json",
                    ROOT / "compat/reference-requirements.txt",
                    ROOT / "compat/runtime-requirements.txt",
                    ROOT / "validation/reports/jpt-encoding.json",
                    *(ROOT / "src/jevserve/data").glob("*.json"),
                ]
            )
        }
        load_model_directory(args.model_dir, source_manifest(args.model))
        selected = profile(args.model)
        upstream_revision = selected.get("upstream_code", {}).get("revision")
        if not upstream_revision:
            upstream_revision = json.loads((ROOT / "models/locks/open-jev-2b.json").read_text())[
                "reference_revision"
            ]
        summary["upstream_revision"] = upstream_revision
        # Kev uses checkpoint-provenance hashes of vendored sources in model_matrix.
        if selected["adapter"] != "kev":
            actual = checked(["git", "-C", str(args.upstream), "rev-parse", "HEAD"])
            dirty = checked(
                ["git", "-C", str(args.upstream), "status", "--porcelain", "--untracked-files=all"]
            )
            if actual != upstream_revision or dirty:
                raise ValidationFailure("upstream_unpinned_or_dirty")
        if args.reference_wheel:
            digest = sha256(args.reference_wheel)
            if digest != args.reference_wheel_sha256:
                raise ValueError("reference wheel SHA-256 mismatch")
            summary["reference_wheel"] = {"filename": args.reference_wheel.name, "sha256": digest}
        summary["image_id"] = checked(
            docker + ["image", "inspect", "--format", "{{.Id}}", args.image]
        )
        assert_idle()
        gpu_uuid = checked(["nvidia-smi", "--id=0", "--query-gpu=uuid", "--format=csv,noheader"])
        if not re.fullmatch(r"GPU-[a-fA-F0-9-]+", gpu_uuid):
            raise ValidationFailure("gpu_identity_unavailable")
        env["CUDA_VISIBLE_DEVICES"] = gpu_uuid
        summary["stage"] = "reference"
        checked(
            plan["reference"], timeout=args.timeout, env=env, log=args.output_dir / "reference.log"
        )
        reference = json.loads((args.output_dir / "reference.json").read_text())
        if reference.get("model_revision", reference.get("revision")) != summary[
            "model_revision"
        ] or not reference.get("cases"):
            raise ValueError("reference report missing cases or revision mismatch")
        assert_idle()
        summary["stage"] = "container"
        attempted = True
        checked(plan["container"], timeout=120, env=env, log=args.output_dir / "start.log")
        summary["container_image_id"] = checked(
            docker + ["inspect", "--format", "{{.Image}}", name]
        )
        if summary["container_image_id"] != summary["image_id"]:
            raise ValueError("container image identity mismatch")
        summary["stage"] = "readiness"
        checked(
            plan["readiness"],
            timeout=args.timeout + 30,
            env=env,
            log=args.output_dir / "readiness.log",
        )
        summary["stage"] = "http_sdk"
        result = call(plan["http"], timeout=args.timeout, env=env, log=args.output_dir / "http.log")
        report = json.loads((args.output_dir / "http.json").read_text())
        failures = report.get("failures", report.get("parity_failed_cases"))
        if (
            type(failures) is not int
            or failures < 0
            or not report.get("cases")
            or not report.get("official_sdk_smoke")
            or result.returncode not in (0, 1)
            or (result.returncode == 1 and not (selected["adapter"] == "open_jev" and failures > 0))
        ):
            raise ValueError("incomplete HTTP/SDK report")
        if report.get("model_revision", report.get("revision")) != summary["model_revision"]:
            raise ValueError("HTTP model revision mismatch")
        summary.update(
            real_gpu=True,
            execution="passed",
            parity="failed" if failures else "passed",
            parity_failed_cases=failures,
            stage="complete",
        )
    except (Exception, KeyboardInterrupt) as error:
        summary["error_type"] = type(error).__name__  # Messages may contain private paths.
        summary["error_code"] = (
            error.code
            if isinstance(error, ValidationFailure)
            else "timeout"
            if isinstance(error, subprocess.TimeoutExpired)
            else "interrupted"
            if isinstance(error, KeyboardInterrupt)
            else "validation_locked"
            if isinstance(error, BlockingIOError)
            else "invalid_artifact_or_report"
            if isinstance(error, ValueError)
            else "execution_error"
        )
    finally:
        # A second Ctrl-C/SIGTERM must not interrupt resource cleanup or summary writing.
        previous_handlers = {
            sig: signal.signal(sig, signal.SIG_IGN) for sig in (signal.SIGINT, signal.SIGTERM)
        }
        try:
            if attempted:
                summary["cleanup"] = "passed"
                for command, log in [
                    (docker + ["logs", name], "container.log"),
                    (docker + ["stop", "--time", "65", name], "stop.log"),
                    (docker + ["rm", "--force", name], "cleanup-container.log"),
                    (docker + ["volume", "rm", name], "cleanup-volume.log"),
                ]:
                    try:
                        checked(command, timeout=90, log=args.output_dir / log)
                    except Exception:
                        summary["cleanup"] = "failed"
                if summary["cleanup"] == "failed":
                    summary["execution"] = "failed"
            if lock:
                lock.close()
            summary["reports"] = {
                name: sha256(args.output_dir / name)
                for name in ("reference.json", "http.json")
                if (args.output_dir / name).is_file()
            }
            (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        finally:
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)
    print(json.dumps(summary, indent=2))
    return 2 if summary["execution"] != "passed" else int(summary["parity"] != "passed")


def interrupted(signum, frame):
    raise KeyboardInterrupt


def main():
    signal.signal(signal.SIGTERM, interrupted)
    args = parser().parse_args()
    try:
        return execute(args)
    except (ValueError, FileExistsError) as error:
        parser().error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
