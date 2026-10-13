"""Rollback orchestration tests use no Docker daemon, GPU or network."""

import json
import subprocess
from types import SimpleNamespace

import pytest

import validation.rollback as rollback
from jevserve.contracts.responses import DecisionResponse
from validation.run import ValidationFailure


@pytest.fixture
def args(tmp_path):
    return SimpleNamespace(
        docker_host="unix:///test/docker.sock",
        from_image="example/jevserve@sha256:" + "a" * 64,
        to_image="example/jevserve@sha256:" + "b" * 64,
        cache_volume="jevserve-existing",
        output_dir=tmp_path / "report",
        port=18000,
        timeout=60,
    )


@pytest.fixture
def fake_runtime(tmp_path, monkeypatch):
    (tmp_path / "examples").mkdir()
    (tmp_path / "examples/decision.json").write_bytes(
        (rollback.ROOT / "examples/decision.json").read_bytes()
    )
    monkeypatch.setattr(rollback, "ROOT", tmp_path)
    events = []
    active = {}

    def checked(command, **kwargs):
        if command[0] == "git":
            return "a" * 40 if command[1] == "rev-parse" else ""
        operation = command[3:]
        events.append(operation)
        if operation[:2] == ["volume", "inspect"]:
            return "{}"
        if operation[:2] == ["ps", "--quiet"]:
            return ""
        if operation[:2] == ["image", "inspect"]:
            return "sha256:" + ("1" if operation[-1].endswith("a" * 64) else "2") * 64
        if operation[0] == "run":
            assert not active, "containers must be serial"
            name = operation[operation.index("--name") + 1]
            image = next(value for value in operation if "@sha256:" in value)
            active[name] = "sha256:" + ("1" if image.endswith("a" * 64) else "2") * 64
            return "container-id"
        if operation[0] == "inspect":
            return active[operation[-1]]
        raise AssertionError(operation)

    def call(command, **kwargs):
        assert command[3] == "exec"
        events.append(command[3:])
        return subprocess.CompletedProcess(command, 0, "", "")

    def idle():
        events.append(["gpu_idle"])
        assert not active

    def cleanup(docker, name):
        events.append(["cleanup", name])
        del active[name]
        return {"status": "passed", "container_exit_code": 0, "stop_returncode": 0}

    answer = {
        "answers": {
            "team": {
                "type": "choice",
                "choice": "billing",
                "probabilities": {"technical": 0.25, "billing": 0.75},
            },
            "urgent": {"type": "noul", "noul": 0.2},
            "priority": {
                "type": "score",
                "score": 0.5,
                "probabilities": {"0": 0.6, "1": 0.3, "2": 0.1},
                "legend": {"0": "Routine", "1": "Soon", "2": "Immediate"},
            },
        },
        "request_id": "test",
        "model_revision": rollback.profile(rollback.MODEL)["revision"],
        "adapter_revision": "test",
        "execution_profile": "test",
        "probability_kind": "candidate_softmax",
        "calibration_id": None,
    }
    DecisionResponse.model_validate(answer)

    def sample(port, request):
        events.append(["sample"])
        return {"id": rollback.MODEL, "revision": answer["model_revision"]}, answer

    monkeypatch.setattr(rollback, "checked", checked)
    monkeypatch.setattr(rollback, "call", call)
    monkeypatch.setattr(rollback, "assert_idle", idle)
    monkeypatch.setattr(rollback, "cleanup", cleanup)
    monkeypatch.setattr(rollback, "sample", sample)
    return events, active


def test_old_new_old_is_serial_and_preserves_cache(args, fake_runtime):
    events, active = fake_runtime
    assert rollback.execute(args) == 0
    report = json.loads((args.output_dir / "summary.json").read_text())
    assert report["status"] == "passed"
    assert [stage["image"] for stage in report["stages"]] == [
        args.from_image,
        args.to_image,
        args.from_image,
    ]
    assert all(stage["readiness_returncode"] == 0 for stage in report["stages"])
    assert all(stage["startup_to_readiness_seconds"] >= 0 for stage in report["stages"])
    assert len([event for event in events if event == ["gpu_idle"]]) == 3
    runs = [event for event in events if event[0] == "run"]
    assert len(runs) == 3
    assert all("--pull=never" in event and "--model-dir" not in event for event in runs)
    assert all(f"type=volume,src={args.cache_volume},dst=/cache" in event for event in runs)
    assert len({event[event.index("--name") + 1] for event in runs}) == 3
    assert report["comparisons"]["old_before_to_old_after"]["author_parity"] == "not_measured"
    assert not active


@pytest.mark.parametrize("failure", [ValidationFailure("http_failed"), KeyboardInterrupt()])
def test_failure_or_interrupt_cleans_current_container_and_stops(
    args, fake_runtime, monkeypatch, failure
):
    events, active = fake_runtime

    def fail(*arguments):
        raise failure

    monkeypatch.setattr(rollback, "sample", fail)
    assert rollback.execute(args) == 1
    report = json.loads((args.output_dir / "summary.json").read_text())
    assert report["status"] == "failed"
    assert report["error_code"] in ("http_failed", "interrupted")
    assert len(report["stages"]) == 1
    assert report["stages"][0]["cleanup"]["status"] == "passed"
    assert not active
    assert len([event for event in events if event[0] == "run"]) == 1


def test_busy_gpu_never_starts_container(args, fake_runtime, monkeypatch):
    events, _ = fake_runtime

    def busy():
        raise ValidationFailure("gpu_busy")

    monkeypatch.setattr(rollback, "assert_idle", busy)
    assert rollback.execute(args) == 1
    report = json.loads((args.output_dir / "summary.json").read_text())
    assert report["error_code"] == "gpu_busy"
    assert not any(event[0] == "run" for event in events)


def test_failed_cleanup_blocks_next_stage(args, fake_runtime, monkeypatch):
    events, _ = fake_runtime
    monkeypatch.setattr(rollback, "cleanup", lambda docker, name: {"status": "failed"})
    assert rollback.execute(args) == 1
    assert len([event for event in events if event[0] == "run"]) == 1


def test_cleanup_never_removes_foreign_container_or_any_volume(monkeypatch):
    commands = []

    def call(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(
            command, 0, json.dumps({rollback.LABEL: "someone-else"}), ""
        )

    monkeypatch.setattr(rollback, "call", call)
    result = rollback.cleanup(["docker", "--host", "unix:///test"], "jevserve-rollback-owned")
    assert result["status"] == "failed"
    assert result["error_code"] == "ownership_mismatch"
    assert len(commands) == 1


def test_cleanup_records_exit_and_removes_exact_container_only(monkeypatch):
    commands = []
    name = "jevserve-rollback-owned"

    def call(command, **kwargs):
        commands.append(command)
        output = json.dumps({rollback.LABEL: name}) if command[3] == "inspect" else ""
        return subprocess.CompletedProcess(command, 0, output, "")

    monkeypatch.setattr(rollback, "call", call)
    monkeypatch.setattr(
        rollback, "checked", lambda command: json.dumps({"Running": False, "ExitCode": 0})
    )
    result = rollback.cleanup(["docker", "--host", "unix:///test"], name)
    assert result["status"] == "passed"
    assert result["container_exit_code"] == 0
    assert [command[3] for command in commands] == ["inspect", "stop", "rm"]
    assert all(command[-1] == name for command in commands)
    assert not any("volume" in command for command in commands)


@pytest.mark.parametrize(
    "field,value",
    [
        ("from_image", "example/jevserve:latest"),
        ("cache_volume", "/host/path"),
        ("docker_host", "tcp://remote:2375"),
        ("port", 80),
    ],
)
def test_invalid_configuration_is_rejected(args, field, value):
    setattr(args, field, value)
    with pytest.raises(ValueError):
        rollback.validate(args)


def test_readiness_failure_records_exit_and_cleans(args, fake_runtime, monkeypatch):
    _, active = fake_runtime
    monkeypatch.setattr(
        rollback,
        "call",
        lambda command, **kw: subprocess.CompletedProcess(command, 3, "", "PRIVATE"),
    )
    assert rollback.execute(args) == 1
    text = (args.output_dir / "summary.json").read_text()
    stage = json.loads(text)["stages"][0]
    assert stage["readiness_returncode"] == 3
    assert stage["startup_to_readiness_seconds"] is None
    assert stage["cleanup"]["status"] == "passed"
    assert "PRIVATE" not in text
    assert not active


def test_nonzero_container_exit_fails_cleanup_but_still_removes(monkeypatch):
    commands = []
    name = "jevserve-rollback-owned"

    def call(command, **kwargs):
        commands.append(command)
        output = json.dumps({rollback.LABEL: name}) if command[3] == "inspect" else ""
        return subprocess.CompletedProcess(command, 0, output, "")

    monkeypatch.setattr(rollback, "call", call)
    monkeypatch.setattr(
        rollback, "checked", lambda command: json.dumps({"Running": False, "ExitCode": 137})
    )
    result = rollback.cleanup(["docker", "--host", "unix:///test"], name)
    assert result["status"] == "failed"
    assert result["container_exit_code"] == 137
    assert commands[-1][3:] == ["rm", "--force", name]


def test_cleanup_attempts_removal_even_if_stop_times_out(monkeypatch):
    commands = []
    name = "jevserve-rollback-owned"

    def call(command, **kwargs):
        commands.append(command)
        if command[3] == "stop":
            raise subprocess.TimeoutExpired(command, 90)
        output = json.dumps({rollback.LABEL: name}) if command[3] == "inspect" else ""
        return subprocess.CompletedProcess(command, 0, output, "")

    monkeypatch.setattr(rollback, "call", call)
    result = rollback.cleanup(["docker", "--host", "unix:///test"], name)
    assert result["status"] == "failed"
    assert commands[-1][3:] == ["rm", "--force", name]


def test_in_use_cache_refused_before_container_launch(args, fake_runtime, monkeypatch):
    events, _ = fake_runtime
    original = rollback.checked

    def checked(command, **kwargs):
        if command[3:5] == ["ps", "--quiet"]:
            return "existing-container"
        return original(command, **kwargs)

    monkeypatch.setattr(rollback, "checked", checked)
    assert rollback.execute(args) == 1
    report = json.loads((args.output_dir / "summary.json").read_text())
    assert report["error_code"] == "cache_volume_in_use"
    assert not any(event[0] == "run" for event in events)


def test_sample_rejects_wrong_model_before_inference(monkeypatch):
    import httpx

    from jevserve.contracts.requests import DecisionRequest

    original = httpx.Client
    requests = []

    def handler(request):
        requests.append(request.url.path)
        return httpx.Response(200, json={"models": [{"id": "wrong-model", "revision": "a" * 40}]})

    monkeypatch.setattr(
        rollback.httpx,
        "Client",
        lambda **kw: original(transport=httpx.MockTransport(handler), **kw),
    )
    request = DecisionRequest.from_json((rollback.ROOT / "examples/decision.json").read_bytes())
    with pytest.raises(ValidationFailure, match="model_identity_mismatch"):
        rollback.sample(18000, request)
    assert requests == ["/v1/models"]


def test_distinct_digests_resolving_to_same_image_are_rejected(args, fake_runtime, monkeypatch):
    events, active = fake_runtime
    original = rollback.checked

    def checked(command, **kwargs):
        if command[3:5] == ["image", "inspect"]:
            return "sha256:" + "1" * 64
        return original(command, **kwargs)

    monkeypatch.setattr(rollback, "checked", checked)
    assert rollback.execute(args) == 1
    report = json.loads((args.output_dir / "summary.json").read_text())
    assert report["status"] == "failed"
    assert report["stage"] == "preflight"
    assert report["error_code"] == "identical_image_ids"
    assert report["stages"] == []
    assert not any(event[0] == "run" for event in events)
    assert not active
