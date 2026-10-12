"""Linux file lock → staging → verification → atomic publish; offline never downloads."""

import errno
import fcntl
import os
import shutil
import tempfile
import time
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol
from urllib.parse import quote

from jevserve.artifacts.manifest import ArtifactFile, ModelManifest, PreparedModel
from jevserve.artifacts.verify import verify_files, verify_package


class ArtifactMissing(FileNotFoundError):
    pass


class DownloadError(RuntimeError):
    pass


class Downloader(Protocol):
    def __call__(self, artifact: ArtifactFile, target: Path, token: str | None) -> None: ...


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: object,
        code: int,
        msg: str,
        headers: object,
        newurl: str,
    ) -> urllib.request.Request | None:
        if not newurl.startswith("https://"):
            raise DownloadError("non-HTTPS artifact redirect rejected")
        # Hub may redirect to a signed CDN URL. Never forward bearer credentials.
        return urllib.request.Request(newurl, headers={"User-Agent": "jevserve/0.1"})


def download_file(artifact: ArtifactFile, target: Path, token: str | None) -> None:
    url = (
        f"https://huggingface.co/{artifact.source_repo}/resolve/"
        f"{artifact.source_revision}/{quote(artifact.source_path, safe='/')}"
    )
    headers = {"User-Agent": "jevserve/0.1"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    opener = urllib.request.build_opener(SafeRedirect())
    request = urllib.request.Request(url, headers=headers)
    # One attempt, bounded socket timeout; no automatic retry or model substitution.
    deadline = time.monotonic() + 1800
    with opener.open(request, timeout=30) as response, target.open("xb") as stream:
        size = 0
        while block := response.read(1024 * 1024):
            size += len(block)
            if size > artifact.size or time.monotonic() > deadline:
                raise DownloadError("artifact size or download time budget exceeded")
            stream.write(block)
        stream.flush()
        os.fsync(stream.fileno())


@contextmanager
def _lock(path: Path, timeout: float) -> Iterator[None]:
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise DownloadError("model preparation lock deadline exceeded") from None
                time.sleep(min(0.05, max(0, deadline - time.monotonic())))
        yield
    finally:
        os.close(fd)


def prepare_model(
    manifest: ModelManifest,
    cache_dir: Path,
    offline: bool = False,
    *,
    downloader: Downloader = download_file,
    token_file: Path | None = None,
    lock_timeout: float = 30,
) -> PreparedModel:
    if not 0 <= lock_timeout <= 1800:
        raise ValueError("lock_timeout must be between 0 and 1800 seconds")
    if cache_dir.is_symlink():
        raise DownloadError("cache root must not be a symlink")
    final = cache_dir / manifest.identity
    if final.exists() or final.is_symlink():
        return verify_package(final, manifest)
    if offline:
        raise ArtifactMissing("verified model is absent in offline mode")
    cache_dir.mkdir(parents=True, exist_ok=True)
    with _lock(cache_dir / (manifest.identity + ".lock"), lock_timeout):
        if final.exists() or final.is_symlink():
            return verify_package(final, manifest)
        if shutil.disk_usage(cache_dir).free < sum(item.size for item in manifest.files):
            raise DownloadError("insufficient disk space for model preparation")
        token = token_file.read_text().strip() if token_file else None
        if token is not None and (not token or "\n" in token or "\r" in token):
            raise DownloadError("invalid token file")
        stage = Path(tempfile.mkdtemp(prefix=manifest.identity + ".staging-", dir=cache_dir))
        try:
            for item in manifest.files:
                target = stage / item.path
                target.parent.mkdir(parents=True, exist_ok=True)
                downloader(item, target, token)
            verify_files(stage, manifest)
            (stage / "manifest.json").write_text(manifest.model_dump_json())
            (stage / "READY").write_text(manifest.identity)
            # Both directories are on the cache filesystem, under the same lock.
            os.rename(stage, final)
            return verify_package(final, manifest)
        except OSError as error:
            message = (
                "insufficient disk space"
                if error.errno == errno.ENOSPC
                else "artifact download or publication failed"
            )
            raise DownloadError(message) from None
        finally:
            if stage.exists():
                shutil.rmtree(stage)
