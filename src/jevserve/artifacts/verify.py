"""Every cache hit verifies bytes and rejects symlinks, including directories."""

import hashlib
from pathlib import Path

from jevserve.artifacts.manifest import ModelManifest, PreparedModel


class ArtifactCorrupt(ValueError):
    pass


def verify_files(path: Path, manifest: ModelManifest) -> None:
    if path.is_symlink() or not path.is_dir():
        raise ArtifactCorrupt("invalid package directory")
    allowed = {item.path for item in manifest.files} | {"manifest.json", "READY"}
    for entry in path.rglob("*"):
        if entry.is_symlink() or (
            not entry.is_dir() and str(entry.relative_to(path)) not in allowed
        ):
            raise ArtifactCorrupt("unexpected file or symlink in model package")
    for item in manifest.files:
        file = path / item.path
        for component in (file, *file.parents):
            if component == path:
                break
            if component.is_symlink():
                raise ArtifactCorrupt("symlinks are not permitted in model packages")
        if not file.is_file() or file.stat().st_size != item.size:
            raise ArtifactCorrupt("artifact missing or size mismatch")
        with file.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != item.sha256:
            raise ArtifactCorrupt("artifact digest mismatch")


def verify_package(path: Path, manifest: ModelManifest) -> PreparedModel:
    verify_files(path, manifest)
    for name in ("manifest.json", "READY"):
        if (path / name).is_symlink() or not (path / name).is_file():
            raise ArtifactCorrupt("package metadata missing or unsafe")
    expected = manifest.model_dump_json().encode()
    if (path / "manifest.json").stat().st_size != len(expected):
        raise ArtifactCorrupt("package manifest size mismatch")
    if (path / "manifest.json").read_bytes() != expected:
        raise ArtifactCorrupt("package manifest mismatch")
    if (path / "READY").stat().st_size != 64 or (path / "READY").read_text() != manifest.identity:
        raise ArtifactCorrupt("package readiness identity mismatch")
    return PreparedModel(manifest=manifest, path=path)
