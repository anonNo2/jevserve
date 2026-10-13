import hashlib
import json

import pytest

from jevserve.artifacts.catalog import profile, source_manifest
from validation.reassess import summarize_matrix


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(data) + "\n").encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def evidence(tmp_path):
    model = "kirp/jpt-0.8b"
    revision = profile(model)["revision"]
    image = "sha256:" + "a" * 64
    case = {
        "case": 0,
        "answers": {
            "q": {
                "type": "choice",
                "choice": "a",
                "probabilities": {"a": 0.7, "b": 0.3},
            }
        },
    }
    reference = {"model_id": model, "revision": revision, "mode": "reference", "cases": [case]}
    actual = {
        "case": 0,
        "status": 200,
        "parity_passed": False,
        "answers": {
            "q": {
                "type": "choice",
                "choice": "a",
                "probabilities": {"a": 0.72, "b": 0.28},
            }
        },
    }
    http = {
        "model_id": model,
        "revision": revision,
        "mode": "http",
        "cases": [actual],
        "systemone_smoke": "passed",
        "native_client_smoke": "passed",
        "unknown_model_rejected": True,
        "official_sdk_smoke": "typesafe-sdk==0.7.4",
        "parity_failed_cases": 1,
    }
    folder = tmp_path / "raw" / "jpt-0.8b"
    hashes = {
        "reference.json": write(folder / "reference.json", reference),
        "http.json": write(folder / "http.json", http),
    }
    summary = {
        "report_type": "jevserve-validation",
        "real_gpu": True,
        "model_id": model,
        "model_revision": revision,
        "source_manifest_identity": source_manifest(model).identity,
        "execution": "passed",
        "cleanup": "passed",
        "parity": "failed",
        "image_id": image,
        "container_image_id": image,
        "source_commit": "b" * 40,
        "parity_failed_cases": 1,
        "reports": hashes,
    }
    summary_hash = write(folder / "summary.json", summary)
    item = {
        "model_id": model,
        "model_revision": revision,
        "execution": "passed",
        "cleanup": "passed",
        "source_image_binding": "passed",
        "real_gpu": True,
        "parity": "failed",
        "parity_failed_cases": 1,
        "summary_sha256": summary_hash,
        "raw_reports": {name: {"sha256": digest} for name, digest in hashes.items()},
    }
    aggregate = {
        "report_type": "jevserve-local-source-image-acceptance",
        "image_identity_kind": "docker-local-id",
        "image_id": image,
        "source_commit": "b" * 40,
        "matrix": [item],
    }
    write(tmp_path / "aggregate.json", aggregate)
    write(
        tmp_path / "policy.json",
        {
            "policy_id": "bounded-drift-v1",
            "probability_atol": 0.03,
            "probability_rtol": 0.0,
            "require_argmax_match": True,
        },
    )
    return tmp_path


def run(root):
    return summarize_matrix(root / "aggregate.json", root / "raw", root / "policy.json")


def test_reassessment_preserves_strict_failure_and_is_not_new_gpu_certification(evidence):
    result = run(evidence)
    assert result["all_models_within_policy"]
    assert result["matrix"][0]["historical_strict_failed_cases"] == 1
    assert result["matrix"][0]["failed_cases"] == 0
    assert not result["new_gpu_run"] and not result["certified"]
    assert "image_digest" not in result
    assert "probabilities" not in json.dumps(result)


def test_reassessment_rejects_changed_raw_evidence(evidence):
    path = evidence / "raw/jpt-0.8b/http.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="hash"):
        run(evidence)


def test_reassessment_rejects_duplicate_models(evidence):
    path = evidence / "aggregate.json"
    data = json.loads(path.read_text())
    data["matrix"].append(data["matrix"][0])
    write(path, data)
    with pytest.raises(ValueError, match="duplicate"):
        run(evidence)


def test_reassessment_rejects_wrong_image_identity(evidence):
    path = evidence / "aggregate.json"
    data = json.loads(path.read_text())
    data["image_id"] = "sha256:" + "c" * 64
    write(path, data)
    with pytest.raises(ValueError, match="image"):
        run(evidence)


def test_reassessment_rejects_wrong_source_commit(evidence):
    path = evidence / "aggregate.json"
    data = json.loads(path.read_text())
    data["source_commit"] = "d" * 40
    write(path, data)
    with pytest.raises(ValueError, match="source commit"):
        run(evidence)


def test_reassessment_reports_failure_under_smaller_policy(evidence):
    path = evidence / "policy.json"
    data = json.loads(path.read_text())
    data["probability_atol"] = 0.01
    write(path, data)
    report = run(evidence)
    assert not report["all_models_within_policy"]
    assert report["matrix"][0]["failed_cases"] == 1
