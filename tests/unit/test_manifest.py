import hashlib

import pytest
from pydantic import ValidationError

from jevserve.artifacts.manifest import ArtifactFile, ModelManifest


def item(**changes):
    return ArtifactFile(
        **dict(
            path="weights.bin",
            source_repo="test/model",
            source_revision="a" * 40,
            source_path="weights.bin",
            sha256=hashlib.sha256(b"weights").hexdigest(),
            size=7,
        )
        | changes
    )


@pytest.mark.parametrize(
    "path", ["/a", "../a", "a/../b", "a\\b", ".", "a//b", "a/./b", "READY", "manifest.json"]
)
def test_invalid_path(path):
    with pytest.raises(ValidationError):
        item(path=path)


@pytest.mark.parametrize(
    "changes",
    [
        {"source_revision": "main"},
        {"sha256": "bad"},
        {"size": -1},
        {"source_repo": "https://user:secret@example.com"},
        {"source_path": "../escape"},
    ],
)
def test_bad_source(changes):
    with pytest.raises(ValidationError):
        item(**changes)


def test_immutable_identity_and_duplicates():
    a = ModelManifest(model_id="test/model", revision="a" * 40, files=(item(),))
    b = ModelManifest(model_id="test/model", revision="b" * 40, files=(item(),))
    assert a.identity != b.identity
    with pytest.raises(ValidationError):
        a.revision = "b" * 40
    with pytest.raises(ValidationError):
        ModelManifest(model_id="test/model", revision="a" * 40, files=(item(), item()))
