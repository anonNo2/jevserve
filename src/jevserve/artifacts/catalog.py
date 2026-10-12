"""Pinned model inventories, independent of the runtime image and weight storage."""

import json
from importlib.resources import files
from typing import Any, cast

from jevserve.artifacts.manifest import ArtifactFile, ModelManifest

DEFAULT_MODEL = "ZefanCai/Open-Jev-2B"
PROFILES = {
    "jaredpalmer/kev-4b": "kev-4b.json",
    "TokenRhythm/NeoHorse-Jev-4B": "neohorse-jev-4b.json",
    "ZefanCai/Open-Jev-9B": "open-jev-9b.json",
    "kirp/jpt-0.8b": "jpt-0.8b.json",
    "kirp/jpt-4b": "jpt-4b.json",
    "kirp/jpt-9b": "jpt-9b.json",
}


def baseline() -> dict[str, Any]:
    return cast(
        dict[str, Any], json.loads(files("jevserve").joinpath("data/open-jev-2b.json").read_text())
    )


def source_manifest(model_id: str = DEFAULT_MODEL) -> ModelManifest:
    lock = profile(model_id)
    if "base" not in lock:
        return ModelManifest(
            model_id=model_id,
            revision=lock["revision"],
            files=tuple(
                ArtifactFile(
                    **item,
                    source_repo=model_id,
                    source_revision=lock["revision"],
                    source_path=item["path"],
                )
                for item in lock["files"]
            ),
        )
    artifacts = []
    for prefix, source in [("base", lock["base"]), ("checkpoint", lock)]:
        for item in source["files"]:
            artifacts.append(
                ArtifactFile(
                    path=prefix + "/" + item["path"],
                    sha256=item["sha256"],
                    size=item["size"],
                    source_repo=source["model_id"],
                    source_revision=source["revision"],
                    source_path=item["path"],
                )
            )
    return ModelManifest(model_id=model_id, revision=lock["revision"], files=tuple(artifacts))


def profile(model_id: str = DEFAULT_MODEL) -> dict[str, Any]:
    if model_id == DEFAULT_MODEL:
        return baseline() | {"adapter": "open_jev", "license": "apache-2.0"}
    if model_id not in PROFILES:
        raise ValueError("model is not in the configured registry")
    return cast(
        dict[str, Any],
        json.loads(files("jevserve").joinpath("data/" + PROFILES[model_id]).read_text()),
    )


def model_ids() -> tuple[str, ...]:
    return (DEFAULT_MODEL, *PROFILES)
