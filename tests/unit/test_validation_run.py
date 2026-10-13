import json
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

from validation import run


@pytest.fixture
def args(tmp_path):
    return run.parser().parse_args(
        [
            "--model",
            "kirp/jpt-0.8b",
            "--model-dir",
            str(tmp_path / "weights"),
            "--upstream",
            str(tmp_path / "author"),
            "--reference-python",
            "/usr/bin/python3",
            "--output-dir",
            str(tmp_path / "output"),
            "--docker-host",
            "unix:///tmp/test.sock",
            "--image",
            "example/jevserve@sha256:" + "a" * 64,
        ]
    )


def test_plan_no_execution(args, monkeypatch, capsys):
    args.plan = True
    monkeypatch.setattr(run, "call", Mock(side_effect=AssertionError("must not execute")))
    assert run.execute(args) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["reference"][2:4] == ["validation.model_matrix", "reference"]
    assert "--pull=never" in plan["container"]
    assert not args.output_dir.exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("image", "example:latest"),
        ("docker_host", "tcp://host:2375"),
        ("port", 0),
        ("timeout", 0),
        ("timeout", 1801),
        ("model_dir", Path("/tmp/comma,path")),
    ],
)
def test_bad_inputs(args, field, value):
    setattr(args, field, value)
    with pytest.raises(ValueError):
        run.validate(args)


def test_scalar_and_neohorse_commands(args):
    args.model = "ZefanCai/Open-Jev-9B"
    plan = run.commands(args, "jevserve-validation-test")
    assert "validation.reference" in plan["reference"]
    assert "--base" in plan["reference"]
    assert "validation.http" in plan["http"]
    args.model = "TokenRhythm/NeoHorse-Jev-4B"
    with pytest.raises(ValueError, match="wheel"):
        run.validate(args)


@pytest.fixture
def orchestration(args, monkeypatch):
    # Production uses verified real artifacts; the double exists only in this CPU test.
    monkeypatch.setattr(run, "load_model_directory", lambda *a: None)
    calls = []
    selected = run.profile(args.model)

    def fake(command, **kwargs):
        calls.append(command)
        stdout = ""
        if command[:3] == ["git", "rev-parse", "HEAD"]:
            stdout = "b" * 40
        elif "rev-parse" in command:
            stdout = selected["upstream_code"]["revision"]
        elif "--query-gpu=uuid" in command:
            stdout = "GPU-12345678-abcd"
        elif "inspect" in command:
            stdout = "sha256:" + "c" * 64
        elif "validation.model_matrix" in command:
            report = {"revision": selected["revision"], "cases": [{"case": 0}]}
            if "http" in command:
                report.update(parity_failed_cases=2, official_sdk_smoke="typesafe-sdk==0.7.4")
            Path(command[command.index("--output") + 1]).write_text(json.dumps(report))
        return subprocess.CompletedProcess(command, 0, stdout, "")

    monkeypatch.setattr(run, "call", fake)
    return calls, fake


def test_parity_failure_is_execution_success(args, orchestration):
    calls, _ = orchestration
    assert run.execute(args) == 1
    summary = json.loads((args.output_dir / "summary.json").read_text())
    assert summary["execution"] == "passed"
    assert summary["parity"] == "failed"
    assert summary["cleanup"] == "passed"
    assert str(args.output_dir) not in json.dumps(summary)
    reference_index = next(i for i, c in enumerate(calls) if "reference" in c)
    start_index = next(i for i, c in enumerate(calls) if "run" in c)
    assert reference_index < start_index
    assert calls[-1][-3:-1] == ["volume", "rm"]
    assert calls[-1][-1].startswith("jevserve-validation-")


@pytest.mark.parametrize("failure", ["reference", "run", "exec", "http", "stop"])
def test_failures_report_and_cleanup(args, orchestration, monkeypatch, failure):
    calls, fake = orchestration

    def broken(command, **kwargs):
        if failure in command:
            calls.append(command)
            raise subprocess.TimeoutExpired(command, 1)
        return fake(command, **kwargs)

    monkeypatch.setattr(run, "call", broken)
    assert run.execute(args) == 2
    summary = json.loads((args.output_dir / "summary.json").read_text())
    assert summary["execution"] == "failed"
    cleanup = [c for c in calls if "rm" in c]
    assert len(cleanup) == (0 if failure == "reference" else 2)


def test_occupied_gpu_never_starts_reference(args, orchestration, monkeypatch):
    calls, fake = orchestration

    def busy(command, **kwargs):
        if command[0] == "nvidia-smi":
            return subprocess.CompletedProcess(command, 0, "1234", "")
        return fake(command, **kwargs)

    monkeypatch.setattr(run, "call", busy)
    assert run.execute(args) == 2
    assert not any("reference" in c for c in calls)


def test_existing_output_refused(args):
    args.output_dir.mkdir()
    with pytest.raises(FileExistsError):
        run.execute(args)


def test_timeout_kills_only_own_process_group(monkeypatch):
    process = Mock(pid=12345)
    process.communicate.side_effect = [subprocess.TimeoutExpired(["own-command"], 1), ("", "")]
    context = Mock()
    context.__enter__ = Mock(return_value=process)
    context.__exit__ = Mock(return_value=False)
    popen = Mock(return_value=context)
    kill = Mock()
    monkeypatch.setattr(run.subprocess, "Popen", popen)
    monkeypatch.setattr(run.os, "killpg", kill)
    with pytest.raises(subprocess.TimeoutExpired):
        run.call(["own-command"], timeout=1)
    assert popen.call_args.kwargs["start_new_session"] is True
    kill.assert_called_once_with(12345, run.signal.SIGKILL)


def test_failed_sdk_report_cannot_pass(args, orchestration, monkeypatch):
    _, fake = orchestration

    def incomplete(command, **kwargs):
        result = fake(command, **kwargs)
        if "http" in command:
            path = args.output_dir / "http.json"
            data = json.loads(path.read_text())
            del data["official_sdk_smoke"]
            path.write_text(json.dumps(data))
        return result

    monkeypatch.setattr(run, "call", incomplete)
    assert run.execute(args) == 2
    data = json.loads((args.output_dir / "summary.json").read_text())
    assert data["real_gpu"] is False
    assert data["parity"] == "not_run"


def test_reference_uses_checked_physical_gpu_uuid(args, orchestration, monkeypatch):
    _, fake = orchestration
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5")
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "FASTEST_FIRST")

    def inspect(command, **kwargs):
        if "reference" in command:
            assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "GPU-12345678-abcd"
        return fake(command, **kwargs)

    monkeypatch.setattr(run, "call", inspect)
    assert run.execute(args) == 1


def test_interrupt_cleanup_ignores_second_signals(args, orchestration, monkeypatch):
    _, fake = orchestration

    def interrupted(command, **kwargs):
        if "exec" in command:
            raise KeyboardInterrupt
        if "stop" in command or "rm" in command:
            assert run.signal.getsignal(run.signal.SIGTERM) == run.signal.SIG_IGN
            assert run.signal.getsignal(run.signal.SIGINT) == run.signal.SIG_IGN
        return fake(command, **kwargs)

    monkeypatch.setattr(run, "call", interrupted)
    previous = run.signal.getsignal(run.signal.SIGTERM)
    assert run.execute(args) == 2
    assert run.signal.getsignal(run.signal.SIGTERM) == previous
    summary = json.loads((args.output_dir / "summary.json").read_text())
    assert summary["error_code"] == "interrupted"
    assert summary["cleanup"] == "passed"


def test_sigterm_becomes_interrupt():
    with pytest.raises(KeyboardInterrupt):
        run.interrupted(run.signal.SIGTERM, None)
