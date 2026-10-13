"""Synthetic release fixtures test policy, never constitute GPU evidence."""

import hashlib
import json
import subprocess
from urllib.error import HTTPError

import pytest

from validation import release


def digest(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def candidate(tmp_path):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(tmp_path), *args]).decode().strip()
    git("init", "-q")
    paths = sorted(release.DEPENDENCIES | {"src/jevserve/data/test.json"})
    sources = []
    for name in paths:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps({"model_id": "test/model", "revision": "a" * 40}).encode()
        path.write_bytes(data)
        sources.append({"path": name, "sha256": digest(data)})
    git("add", ".")
    git("-c", "user.name=Test Fixture", "-c", "user.email=fixture@example.invalid",
        "commit", "-qm", "Synthetic test data")
    commit = git("rev-parse", "HEAD")
    raw = tmp_path / "raw.json"
    raw.write_text('{"fixture": true}')
    reference = {"model_id": "test/model", "revision": "a" * 40, "mode": "reference",
                 "cases": [{"case": 0, "answers": {"test": {"noul": 0.5}}}]}
    http = {"model_id": "test/model", "revision": "a" * 40, "mode": "http",
            "cases": [{"case": 0, "status": 200, "parity_passed": False,
                       "answers": {"test": {"noul": 0.6}}}],
            "parity_failed_cases": 1, "official_sdk_smoke": "typesafe-sdk==0.7.4",
            "systemone_smoke": "passed", "native_client_smoke": "passed",
            "unknown_model_rejected": True}
    (tmp_path / "reference.json").write_text(json.dumps(reference))
    (tmp_path / "http.json").write_text(json.dumps(http))
    summary = {"report_type": "jevserve-validation", "schema_version": 1,
               "source_commit": commit, "source_dirty": False,
               "image_digest": "sha256:" + "b" * 64,
               "model_id": "test/model", "model_revision": "a" * 40,
               "real_gpu": True, "execution": "passed", "cleanup": "passed",
               "parity": "failed", "parity_failed_cases": 1, "reports": {
                   "reference.json": digest((tmp_path / "reference.json").read_bytes()),
                   "http.json": digest((tmp_path / "http.json").read_bytes())}}
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    evidence = {
        "schema_version": 1, "source_commit": commit, "image_digest": "sha256:" + "b" * 64,
        "real_gpu": True,
        "gates": dict.fromkeys(("cpu", "docker_run", "compose", "offline"), "passed"),
        "models": [{"model_id": "test/model", "revision": "a" * 40,
                    "execution": "passed", "parity": "failed", "parity_failed_cases": 1,
                    "reports": [{"path": "raw.json", "sha256": digest(raw.read_bytes())},
                                {"path": "summary.json", "sha256": digest(
                                    (tmp_path / "summary.json").read_bytes())}]}],
    }
    evidence["gate_reports"] = {}
    for gate in evidence["gates"]:
        document = {"report_type": "jevserve-release-gate", "schema_version": 1,
                    "source_commit": commit, "image_digest": evidence["image_digest"],
                    "gate": gate, "status": "passed", "commands": ["synthetic fixture"],
                    "reports": [{"path": "raw.json", "sha256": digest(raw.read_bytes())}]}
        data = json.dumps(document).encode()
        (tmp_path / f"{gate}.json").write_bytes(data)
        evidence["gate_reports"][gate] = {"path": f"{gate}.json", "sha256": digest(data)}
    manifest = {
        "schema_version": 1, "channel": "development-preview", "version": "0.1.0-dev2",
        "source_commit": commit, "image_repository": "anonwx/jevserve",
        "image_digest": evidence["image_digest"], "limitations": ["Synthetic fixture"],
        "source_files": sources, "failed_parity_models": ["test/model"],
    }
    def save():
        data = json.dumps(evidence).encode()
        (tmp_path / "evidence.json").write_bytes(data)
        manifest["evidence"] = {"path": "evidence.json", "sha256": digest(data)}
        return manifest, tmp_path
    save()
    return manifest, evidence, tmp_path, save


def test_preview_allows_declared_failed_parity(candidate):
    manifest, _, root, _ = candidate
    assert release.validate(manifest, root)["version"] == "0.1.0-dev2"


@pytest.mark.parametrize("channel", ["alpha", "rc", "stable"])
def test_certification_fails_closed(candidate, channel):
    manifest, _, root, _ = candidate
    manifest["channel"] = channel
    with pytest.raises(ValueError, match="certification"):
        release.validate(manifest, root)


@pytest.mark.parametrize("field,value", [
    ("source_commit", "c" * 40), ("image_digest", "sha256:" + "c" * 64),
    ("real_gpu", False), ("gates", {"cpu": "passed"}),
    ("models", []),
])
def test_mismatched_or_incomplete_evidence(candidate, field, value):
    _, evidence, _, save = candidate
    evidence[field] = value
    with pytest.raises(ValueError):
        release.validate(*save())


def test_revision_and_parity_are_not_optional(candidate):
    manifest, evidence, root, save = candidate
    evidence["models"][0]["revision"] = "c" * 40
    with pytest.raises(ValueError, match="binding"):
        release.validate(*save())
    evidence["models"][0]["revision"] = "a" * 40
    save()
    manifest["failed_parity_models"] = []
    with pytest.raises(ValueError, match="parity"):
        release.validate(manifest, root)


def test_tamper_and_source_locks(candidate):
    manifest, _, root, _ = candidate
    (root / "raw.json").write_text("tampered")
    with pytest.raises(ValueError, match="digest mismatch"):
        release.validate(manifest, root)
    manifest["source_files"].pop()
    with pytest.raises(ValueError, match="dependency"):
        release.validate(manifest, root)


@pytest.mark.parametrize("path", ["../outside", "/etc/passwd", "a/../b", "a\\b", "./x"])
def test_unsafe_paths(path):
    with pytest.raises(ValueError):
        release.relative(path)


def test_symlink_and_duplicate_json(candidate):
    manifest, _, root, _ = candidate
    (root / "link.json").symlink_to(root / "evidence.json")
    manifest["evidence"]["path"] = "link.json"
    with pytest.raises(ValueError, match="symlink"):
        release.validate(manifest, root)
    with pytest.raises(ValueError, match="duplicate"):
        release.read_json('{"status":"failed","status":"passed"}')


def test_requested_commit_must_match(candidate):
    manifest, _, root, _ = candidate
    with pytest.raises(ValueError, match="requested commit"):
        release.validate(manifest, root, expected_commit="c" * 40)


@pytest.mark.parametrize("existing,source_changed", [(False, False), (True, False), (False, True)])
def test_promotion_copies_exact_bytes_or_refuses(monkeypatch, existing, source_changed):
    monkeypatch.setenv("DOCKERHUB_USERNAME", "fixture")
    monkeypatch.setenv("DOCKERHUB_TOKEN", "fixture")
    data = b'{ "schemaVersion": 2 }\n'
    result = {"image_digest": "sha256:" + digest(data), "version": "0.1.0-dev2"}
    writes = []
    def request(url, *, headers=None, data=None, method="GET"):
        if "auth.docker.io" in url:
            return b'{"token":"fixture"}', {}
        if method == "PUT":
            writes.append(data)
            return b"", {}
        if url.endswith(result["version"]) and not existing and not writes:
            raise HTTPError(url, 404, "missing", {}, None)
        body = b"different" if source_changed else b'{ "schemaVersion": 2 }\n'
        return body, {"Content-Type": "application/vnd.oci.image.manifest.v1+json"}
    monkeypatch.setattr(release, "registry_request", request)
    if existing or source_changed:
        with pytest.raises(ValueError):
            release.promote(result)
        assert writes == []
    else:
        release.promote(result)
        assert writes == [data]


@pytest.mark.parametrize("field,value", [
    ("source_dirty", True), ("source_commit", "c" * 40),
    ("image_digest", "sha256:" + "c" * 64), ("real_gpu", False),
    ("cleanup", "failed"), ("parity", "passed"), ("report_type", "legacy"),
])
def test_runner_provenance_cannot_be_backfilled(candidate, field, value):
    _, evidence, root, save = candidate
    path = root / "summary.json"
    document = json.loads(path.read_bytes())
    document[field] = value
    path.write_text(json.dumps(document))
    evidence["models"][0]["reports"][1]["sha256"] = digest(path.read_bytes())
    with pytest.raises(ValueError):
        release.validate(*save())


def test_raw_runner_report_tamper(candidate):
    manifest, _, root, _ = candidate
    (root / "http.json").write_text("tampered")
    with pytest.raises(ValueError, match="digest mismatch"):
        release.validate(manifest, root)


def test_missing_gate_attestation(candidate):
    _, evidence, _, save = candidate
    del evidence["gate_reports"]["compose"]
    with pytest.raises(ValueError, match="gate reports"):
        release.validate(*save())


def test_registry_errors_do_not_allow_overwrite(monkeypatch):
    monkeypatch.setenv("DOCKERHUB_USERNAME", "fixture")
    monkeypatch.setenv("DOCKERHUB_TOKEN", "fixture")
    data = b'{}'
    result = {"image_digest": "sha256:" + digest(data), "version": "0.1.0-dev2"}
    def request(url, **kwargs):
        assert kwargs.get("method") != "PUT"
        if "auth.docker.io" in url:
            return b'{"token":"fixture"}', {}
        if url.endswith(result["version"]):
            raise HTTPError(url, 503, "unavailable", {}, None)
        return data, {"Content-Type": "application/vnd.oci.image.manifest.v1+json"}
    monkeypatch.setattr(release, "registry_request", request)
    with pytest.raises(ValueError, match="absence"):
        release.promote(result)


def raw_fixture(candidate, flavor):
    _, evidence, root, _ = candidate
    reference = json.loads((root / "reference.json").read_bytes())
    http = json.loads((root / "http.json").read_bytes())
    summary = json.loads((root / "summary.json").read_bytes())
    model = evidence["models"][0]
    reference_sha = "a" * 64
    if flavor == "open_jev":
        reference = {"model_revision": model["revision"], "device_type": "cuda",
                     "failures": 0, "case_count": 1,
                     "cases": [{"id": "test", "response": {"answers": {"x": {"noul": 0.5}}}}]}
        http = {"model_revision": model["revision"], "failures": 1, "case_count": 1,
                "reference_sha256": reference_sha, "native_rejected_case_ids": [],
                "official_sdk_smoke": "typesafe-sdk==0.7.4",
                "cases": [{"id": "test", "within_tolerance": False,
                           "answers": {"x": {"noul": 0.6}}}]}
    return reference, http, summary, model, reference_sha


@pytest.mark.parametrize("flavor", ["matrix", "open_jev"])
def test_known_raw_formats_are_checked(candidate, flavor):
    release.validate_raw_reports(*raw_fixture(candidate, flavor))


@pytest.mark.parametrize("flavor", ["matrix", "open_jev"])
@pytest.mark.parametrize("conflict", ["empty", "revision", "sdk", "http_status", "failure_count",
                                      "summary_count", "aggregate_count", "parity", "coverage",
                                      "placeholder", "reference_revision"])
def test_raw_conflicts_cannot_hide_behind_passed_summary(candidate, flavor, conflict):
    reference, http, summary, model, sha = raw_fixture(candidate, flavor)
    revision_key = "revision" if flavor == "matrix" else "model_revision"
    if conflict == "empty":
        http["cases"] = []
    elif conflict == "revision":
        http[revision_key] = "c" * 40
    elif conflict == "reference_revision":
        reference[revision_key] = "c" * 40
    elif conflict == "sdk":
        http["official_sdk_smoke"] = "failed"
    elif conflict == "http_status":
        http["cases"][0]["status"] = 500
    elif conflict == "failure_count":
        http["parity_failed_cases" if flavor == "matrix" else "failures"] = 0
    elif conflict == "summary_count":
        summary["parity_failed_cases"] = 0
    elif conflict == "aggregate_count":
        model["parity_failed_cases"] = 0
    elif conflict == "parity":
        model["parity"] = summary["parity"] = "passed"
    elif conflict == "coverage":
        http["cases"][0]["case" if flavor == "matrix" else "id"] = "different"
    else:
        reference = http = {"fixture": True}
    with pytest.raises(ValueError):
        release.validate_raw_reports(reference, http, summary, model, sha)


def test_rehashed_raw_contradiction_is_rejected(candidate):
    _, evidence, root, save = candidate
    path = root / "http.json"
    http = json.loads(path.read_bytes())
    http["cases"][0]["status"] = 500
    path.write_text(json.dumps(http))
    summary_path = root / "summary.json"
    summary = json.loads(summary_path.read_bytes())
    summary["reports"]["http.json"] = digest(path.read_bytes())
    summary_path.write_text(json.dumps(summary))
    evidence["models"][0]["reports"][1]["sha256"] = digest(summary_path.read_bytes())
    with pytest.raises(ValueError, match="HTTP execution failure"):
        release.validate(*save())


@pytest.mark.parametrize("name,model_id,failed", [
    ("jpt-0.8b", "kirp/jpt-0.8b", 6),
    ("jpt-4b", "kirp/jpt-4b", 6),
    ("jpt-9b", "kirp/jpt-9b", 5),
    ("open-jev-9b", "ZefanCai/Open-Jev-9B", 27),
])
def test_historical_report_format_only_not_release_eligibility(name, model_id, failed):
    from pathlib import Path

    reports = Path(__file__).resolve().parents[2] / "validation/reports"
    reference_bytes = (reports / f"{name}-reference.json").read_bytes()
    reference = json.loads(reference_bytes)
    http = json.loads((reports / f"{name}-container.json").read_bytes())
    model = {"model_id": model_id,
             "revision": reference.get("model_revision", reference.get("revision")),
             "parity": "failed", "parity_failed_cases": failed}
    release.validate_raw_reports(reference, http, {"parity_failed_cases": failed},
                                 model, digest(reference_bytes))


def test_historical_http_without_sdk_evidence_is_rejected():
    from pathlib import Path

    reports = Path(__file__).resolve().parents[2] / "validation/reports"
    reference_bytes = (reports / "open-jev-reference.json").read_bytes()
    reference = json.loads(reference_bytes)
    http = json.loads((reports / "native-http-parity.json").read_bytes())
    failures = http["failures"]
    model = {"model_id": "ZefanCai/Open-Jev-2B", "revision": reference["model_revision"],
             "parity": "failed", "parity_failed_cases": failures}
    with pytest.raises(ValueError, match="SDK"):
        release.validate_raw_reports(reference, http, {"parity_failed_cases": failures},
                                     model, digest(reference_bytes))
