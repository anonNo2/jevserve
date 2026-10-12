"""Reject incomplete evidence at release time; discovery alone is not certification."""

import hashlib
import json
import re
from pathlib import Path, PurePosixPath


def _match(value, pattern, label):
    if not isinstance(value, str) or re.fullmatch(pattern, value) is None:
        raise ValueError(f"invalid immutable {label}")


def _files(files):
    if not isinstance(files, list) or not files:
        raise ValueError("missing file inventory")
    seen = set()
    for item in files:
        name = item["path"]
        path = PurePosixPath(name)
        if (
            not name
            or path.is_absolute()
            or ".." in path.parts
            or "\\" in name
            or name != str(path)
            or name in seen
        ):
            raise ValueError("invalid or duplicate artifact path")
        seen.add(name)
        _match(item["sha256"], r"[0-9a-f]{64}", "file digest")
        if type(item["size"]) is not int or item["size"] < 0:
            raise ValueError("invalid file size")


def validate_baseline(engine, model):
    for value in (
        engine["source_revision"],
        engine["plugin_revision"],
        model["revision"],
        model["reference_revision"],
        model["base"]["revision"],
    ):
        _match(value, r"[0-9a-f]{40}", "revision")
    _match(engine["image_digest"], r"sha256:[0-9a-f]{64}", "image digest")
    _files(model["files"])
    _files(model["base"]["files"])
    for field in ("tokenizer_digest", "template_digest"):
        _match(model[field], r"[0-9a-f]{64}", field)
    if model["readout"] != "scalar_head" or model["dtype"] != "bfloat16":
        raise ValueError("unexpected baseline execution semantics")


def validate_release(engine, model, root: Path):
    validate_baseline(engine, model)
    if engine["status"] != "certified" or model["status"] != "certified":
        raise ValueError("baseline is not certified")
    for gate in ("gpu_reference", "gpu_parity", "docker_run", "compose", "offline"):
        if model.get("gates", {}).get(gate) != "passed":
            raise ValueError(f"missing certification gate: {gate}")
    report = model["reference_report"]
    _match(report["sha256"], r"[0-9a-f]{64}", "reference report digest")
    path = (root / report["path"]).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("reference report outside repository")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != report["sha256"]:
        raise ValueError("reference report digest mismatch")
    evidence = json.loads(data)
    if evidence["model_revision"] != model["revision"] or evidence["case_count"] < 100:
        raise ValueError("reference report does not certify this model")
    if evidence["failures"] != 0 or evidence["device_type"] != "cuda":
        raise ValueError("reference report lacks successful GPU execution")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    validate_release(
        json.loads((root / "compat/engine-lock.json").read_text()),
        json.loads((root / "models/locks/open-jev-2b.json").read_text()),
        root,
    )
