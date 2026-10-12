"""Packaged source inventory for the one explicitly supported implementation target."""

import json
from importlib.resources import files
from typing import Any, cast

from jevserve.artifacts.manifest import ArtifactFile, ModelManifest

DEFAULT_MODEL = "ZefanCai/Open-Jev-2B"


def baseline() -> dict[str, Any]:
    return cast(
        dict[str, Any], json.loads(files("jevserve").joinpath("data/open-jev-2b.json").read_text())
    )


def source_manifest(model_id: str = DEFAULT_MODEL) -> ModelManifest:
    if model_id != DEFAULT_MODEL:
        raise ValueError("model is not in the configured registry")
    lock = baseline()
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
