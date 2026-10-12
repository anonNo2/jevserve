"""Preparation manifest, separate from baseline discovery and certification reports."""

import hashlib
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Revision = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Repo = Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+$")]


def relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or value == "."
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in value
        or str(path) != value
        or any(ord(c) < 32 for c in value)
    ):
        raise ValueError("artifact paths must be normalized relative paths")
    return value


class Immutable(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class ArtifactFile(Immutable):
    path: str
    source_repo: Repo
    source_revision: Revision
    source_path: str
    sha256: Digest
    size: Annotated[int, Field(ge=0)]

    @field_validator("path", "source_path")
    @classmethod
    def safe_path(cls, value: str) -> str:
        return relative_path(value)

    @field_validator("path")
    @classmethod
    def reserved(cls, value: str) -> str:
        if PurePosixPath(value).parts[0] in ("READY", "manifest.json"):
            raise ValueError("reserved package path")
        return value


class ModelManifest(Immutable):
    schema_version: Literal[1] = 1
    model_id: Repo
    revision: Revision
    files: Annotated[tuple[ArtifactFile, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def no_collisions(self) -> Self:
        paths = [item.path for item in self.files]
        if len(set(paths)) != len(paths):
            raise ValueError("duplicate artifact path")
        for name in paths:
            if any(str(parent) in paths for parent in PurePosixPath(name).parents):
                raise ValueError("artifact directory collides with file")
        return self

    @property
    def identity(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()


class PreparedModel(Immutable):
    manifest: ModelManifest
    path: Path
