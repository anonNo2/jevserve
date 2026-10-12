"""Explicit registry; baseline discovery never implicitly enables a serving model."""

from collections.abc import Mapping
from pathlib import Path

from jevserve.artifacts.manifest import ModelManifest


class ModelRegistry:
    def __init__(self, entries: Mapping[str, ModelManifest]) -> None:
        self._entries = dict(entries)
        if any(key != value.model_id for key, value in self._entries.items()):
            raise ValueError("registry model identity mismatch")

    def resolve(self, model_id: str) -> ModelManifest:
        try:
            return self._entries[model_id]
        except KeyError:
            raise ValueError("model is not in the configured registry") from None

    @staticmethod
    def local_manifest(path: Path) -> ModelManifest:
        return ModelManifest.model_validate_json(path.read_bytes())
