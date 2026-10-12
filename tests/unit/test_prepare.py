import hashlib
from concurrent.futures import ThreadPoolExecutor

import pytest

from jevserve.artifacts.manifest import ArtifactFile, ModelManifest
from jevserve.artifacts.prepare import ArtifactMissing, DownloadError, prepare_model
from jevserve.artifacts.verify import ArtifactCorrupt


def manifest():
    return ModelManifest(
        model_id="test/model",
        revision="a" * 40,
        files=(
            ArtifactFile(
                path="nested/weights.bin",
                source_repo="test/model",
                source_revision="a" * 40,
                source_path="weights.bin",
                sha256=hashlib.sha256(b"weights").hexdigest(),
                size=7,
            ),
        ),
    )


class Downloader:
    def __init__(self):
        self.calls = 0

    def __call__(self, artifact, target, token):
        self.calls += 1
        target.write_bytes(b"weights")


def test_prepare_cache_hit_and_offline(tmp_path):
    download = Downloader()
    prepared = prepare_model(manifest(), tmp_path, downloader=download)
    assert prepared.path.joinpath("nested/weights.bin").read_bytes() == b"weights"
    assert prepare_model(manifest(), tmp_path, offline=True, downloader=download) == prepared
    assert prepare_model(manifest(), tmp_path, downloader=download) == prepared
    assert download.calls == 1


def test_offline_missing_never_downloads(tmp_path):
    download = Downloader()
    with pytest.raises(ArtifactMissing):
        prepare_model(manifest(), tmp_path, offline=True, downloader=download)
    assert download.calls == 0
    assert not list(tmp_path.iterdir())


def test_interrupted_download_is_not_ready(tmp_path):
    def broken(item, target, token):
        target.write_bytes(b"half")
        raise OSError("connection lost")

    with pytest.raises(DownloadError):
        prepare_model(manifest(), tmp_path, downloader=broken)
    assert not (tmp_path / manifest().identity).exists()
    assert not list(tmp_path.glob("*.staging-*"))


def test_bad_download_digest(tmp_path):
    def wrong(item, target, token):
        target.write_bytes(b"WRONG!!")

    with pytest.raises(ArtifactCorrupt):
        prepare_model(manifest(), tmp_path, downloader=wrong)
    assert not (tmp_path / manifest().identity).exists()


def test_cache_corruption_fails_without_redownload(tmp_path):
    download = Downloader()
    prepared = prepare_model(manifest(), tmp_path, downloader=download)
    prepared.path.joinpath("nested/weights.bin").write_bytes(b"WRONG!!")
    with pytest.raises(ArtifactCorrupt):
        prepare_model(manifest(), tmp_path, offline=True, downloader=download)
    assert download.calls == 1


def test_concurrent_prepare_downloads_once(tmp_path):
    download = Downloader()
    with ThreadPoolExecutor(max_workers=4) as pool:
        outputs = list(
            pool.map(lambda _: prepare_model(manifest(), tmp_path, downloader=download), range(4))
        )
    assert all(result == outputs[0] for result in outputs)
    assert download.calls == 1


def test_symlink_file_rejected(tmp_path):
    prepared = prepare_model(manifest(), tmp_path, downloader=Downloader())
    file = prepared.path / "nested/weights.bin"
    file.unlink()
    outside = tmp_path / "outside"
    outside.write_bytes(b"weights")
    file.symlink_to(outside)
    with pytest.raises(ArtifactCorrupt):
        prepare_model(manifest(), tmp_path, offline=True)


def test_missing_ready_and_manifest_tampering(tmp_path):
    prepared = prepare_model(manifest(), tmp_path, downloader=Downloader())
    prepared.path.joinpath("manifest.json").write_text("{}")
    with pytest.raises(ArtifactCorrupt):
        prepare_model(manifest(), tmp_path, offline=True)


def test_no_cleanup_of_unowned_staging(tmp_path):
    foreign = tmp_path / "someone.staging-other"
    foreign.mkdir()
    prepare_model(manifest(), tmp_path, downloader=Downloader())
    assert foreign.is_dir()


def test_token_is_read_from_file_not_logged(tmp_path):
    token_file = tmp_path / "secret"
    token_file.write_text("private-token\n")

    def download(item, target, token):
        assert token == "private-token"
        target.write_bytes(b"weights")

    result = prepare_model(
        manifest(), tmp_path / "cache", downloader=download, token_file=token_file
    )
    assert "private-token" not in repr(result)


def test_symlink_directory_and_final_rejected(tmp_path):
    prepared = prepare_model(manifest(), tmp_path, downloader=Downloader())
    nested = prepared.path / "nested"
    nested.rename(prepared.path / "moved")
    nested.symlink_to(prepared.path / "moved", target_is_directory=True)
    with pytest.raises(ArtifactCorrupt):
        prepare_model(manifest(), tmp_path, offline=True)


def test_lock_deadline(tmp_path):
    import fcntl

    with (tmp_path / (manifest().identity + ".lock")).open("w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        with pytest.raises(DownloadError, match="deadline"):
            prepare_model(manifest(), tmp_path, lock_timeout=0)


def test_disk_full_sanitized_and_cleanup(tmp_path):
    import errno

    def full(item, target, token):
        raise OSError(errno.ENOSPC, "private path/token")

    with pytest.raises(DownloadError, match="insufficient disk space") as error:
        prepare_model(manifest(), tmp_path, downloader=full)
    assert "private" not in str(error.value)
    assert not list(tmp_path.glob("*.staging-*"))


def test_downloader_does_not_forward_auth_on_redirect():
    import urllib.request

    from jevserve.artifacts.prepare import SafeRedirect

    request = urllib.request.Request(
        "https://huggingface.co/a", headers={"Authorization": "Bearer secret"}
    )
    handler = SafeRedirect()
    redirected = handler.redirect_request(request, None, 302, "", None, "https://cdn.example/a")
    assert redirected.get_header("Authorization") is None
    with pytest.raises(DownloadError):
        handler.redirect_request(request, None, 302, "", None, "http://cdn.example/a")


def test_unlisted_files_are_not_silently_trusted(tmp_path):
    prepared = prepare_model(manifest(), tmp_path, downloader=Downloader())
    (prepared.path / "config.json").write_text("{}")
    with pytest.raises(ArtifactCorrupt, match="unexpected"):
        prepare_model(manifest(), tmp_path, offline=True)


def test_offline_does_not_read_token_file(tmp_path):
    with pytest.raises(ArtifactMissing):
        prepare_model(manifest(), tmp_path, offline=True, token_file=tmp_path / "missing-secret")


def test_missing_ready_marker_is_not_a_cache_hit(tmp_path):
    prepared = prepare_model(manifest(), tmp_path, downloader=Downloader())
    (prepared.path / "READY").unlink()
    with pytest.raises(ArtifactCorrupt):
        prepare_model(manifest(), tmp_path, offline=True)


def test_keyboard_interrupt_cleans_owned_stage(tmp_path):
    def interrupted(item, target, token):
        target.write_bytes(b"half")
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        prepare_model(manifest(), tmp_path, downloader=interrupted)
    assert not list(tmp_path.glob("*.staging-*"))
    assert not (tmp_path / manifest().identity).exists()
