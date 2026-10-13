"""Run with isolated Python (-I) outside the checkout after installing its wheel."""

import argparse
import contextlib
import importlib.metadata
import importlib.util
import io
import json
import socket
import sys
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", action="store_true")
    parser.add_argument("--runtime", action="store_true")
    args = parser.parse_args()
    assert sys.flags.isolated, "Run Python with -I to exclude source and user site packages"

    def no_network(*_args, **_kwargs):
        raise AssertionError("installed package smoke attempted network access")

    socket.create_connection = no_network
    socket.socket.connect = no_network
    socket.socket.connect_ex = no_network

    import jevserve
    from jevserve.cli import main as cli
    from jevserve.cli import sample_request
    from jevserve.client import Client
    from jevserve.contracts.requests import DecisionRequest

    package_path = Path(jevserve.__file__).resolve()
    assert {"site-packages", "dist-packages"} & set(package_path.parts), package_path
    assert not Path.cwd().joinpath("src/jevserve").exists()
    DecisionRequest.model_validate(sample_request())
    with Client("http://127.0.0.1:1"):
        pass
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        try:
            cli(["--help"])
        except SystemExit as exc:
            assert exc.code == 0
        assert cli(["models", "list"]) == 0
    assert "ZefanCai/Open-Jev-2B" in stdout.getvalue()
    with contextlib.redirect_stderr(io.StringIO()):
        assert cli([]) == 2, "serving must require explicit development opt-in"
    with tempfile.TemporaryDirectory() as folder:
        cache = Path(folder) / "missing"
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            assert cli(["prepare", "--offline", "--cache-dir", str(cache)]) == 1
        assert "ArtifactMissing" in stderr.getvalue(), stderr.getvalue()
        assert not cache.exists(), "offline preparation must not create a cache"
    gpu = {"torch", "vllm", "transformers", "peft", "safetensors"}
    server = {"fastapi", "uvicorn", "prometheus_client"}
    assert not (server & sys.modules.keys()), "client/CLI imported server dependencies"
    if args.server or args.runtime:
        from jevserve.server.app import create_app

        assert callable(create_app)
        for module in server:
            assert importlib.util.find_spec(module) is not None, module
    else:
        for module in server:
            assert importlib.util.find_spec(module) is None, module
    assert not (gpu & sys.modules.keys()), "CPU operation imported GPU dependencies"
    if not args.runtime:
        for module in gpu:
            assert importlib.util.find_spec(module) is None, module
    print(json.dumps({"mode": "runtime" if args.runtime else "server" if args.server else "client",
                      "version": importlib.metadata.version("jevserve"), "status": "passed"}))


if __name__ == "__main__":
    main()
