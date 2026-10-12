import json
from pathlib import Path

import pytest

from jevserve.artifacts.compile import RECIPE, digest, verify_compiled
from jevserve.artifacts.verify import ArtifactCorrupt


def compiled(tmp_path):
    (tmp_path / "weights").write_bytes(b"original")
    (tmp_path / "compiled.json").write_text(
        json.dumps(
            {
                "recipe": RECIPE,
                "source_identity": "identity",
                "files": {"weights": digest(tmp_path / "weights")},
            }
        )
    )
    return tmp_path


def test_compiled_cache_rejects_corruption_extra_files_and_identity(tmp_path):
    path = compiled(tmp_path)
    verify_compiled(path, "identity")
    with pytest.raises(ArtifactCorrupt):
        verify_compiled(path, "different-source")
    (path / "weights").write_bytes(b"changed")
    with pytest.raises(ArtifactCorrupt):
        verify_compiled(path, "identity")
    compiled(path)
    (path / "surprise").write_text("untracked")
    with pytest.raises(ArtifactCorrupt):
        verify_compiled(path, "identity")


def test_compiled_cache_rejects_symlinks(tmp_path):
    path = compiled(tmp_path)
    (path / "link").symlink_to(Path("weights"))
    with pytest.raises(ArtifactCorrupt):
        verify_compiled(path, "identity")
