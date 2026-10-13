"""CPU-only numerical policy reassessment of hash-bound source-image evidence."""

import argparse
import hashlib
import json
from pathlib import Path

from jevserve.artifacts.catalog import DEFAULT_MODEL, PROFILES, profile, source_manifest
from jevserve.contracts.requests import DecisionRequest
from validation.acceptance import compare_answers
from validation.release import match, read_json, require, validate_raw_reports

ROOT = Path(__file__).resolve().parents[1]
POLICY_FIELDS = ("policy_id", "probability_atol", "probability_rtol", "require_argmax_match")
KEV_ROUNDED_REFERENCE = "6d02f5d066cd34958dfd15ffa5d2f6f0f4c21a63"


def read(path, expected_hash=None):
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_hash is not None:
        require(digest == expected_hash, "evidence hash mismatch")
    return read_json(raw), digest


def accepted_cases(reference, http):
    matrix = reference.get("mode") == "reference"
    key = "case" if matrix else "id"
    expected = {row[key]: row for row in reference["cases"]}
    if not matrix:
        rejected = set(http.get("native_rejected_case_ids", []))
        for case_id, row in expected.items():
            try:
                DecisionRequest.model_validate(row["request"])
                valid = True
            except ValueError:
                valid = False
            require(valid == (case_id not in rejected), "incorrect native case exclusion")
    for actual in http["cases"]:
        case_id = actual[key]
        original = expected[case_id]
        answers = original["answers"] if matrix else original["response"]["answers"]
        yield case_id, actual["answers"], answers


def summarize_matrix(aggregate_path, evidence_dir, policy_path):
    aggregate, aggregate_sha = read(aggregate_path)
    policy, policy_sha = read(policy_path)
    require(all(key in policy for key in POLICY_FIELDS), "incomplete numerical policy")
    numerical_policy = {key: policy[key] for key in POLICY_FIELDS}
    require(
        aggregate.get("report_type") == "jevserve-local-source-image-acceptance",
        "expected source-image acceptance evidence",
    )
    require(
        aggregate.get("image_identity_kind") == "docker-local-id", "expected local image identity"
    )
    match(aggregate.get("image_id"), r"sha256:[a-f0-9]{64}", "image ID")
    match(aggregate.get("source_commit"), r"[a-f0-9]{40}", "source commit")
    require(
        isinstance(aggregate.get("matrix"), list) and aggregate["matrix"], "empty acceptance matrix"
    )
    models = [item["model_id"] for item in aggregate["matrix"]]
    require(len(models) == len(set(models)), "duplicate model in matrix")
    output = []
    for item in aggregate["matrix"]:
        model_id = item["model_id"]
        selected = profile(model_id)
        require(item["model_revision"] == selected["revision"], "model revision mismatch")
        require(
            item.get("execution") == "passed"
            and item.get("cleanup") == "passed"
            and item.get("real_gpu") is True
            and item.get("source_image_binding") == "passed",
            "incomplete source-image execution evidence",
        )
        slug = "open-jev-2b" if model_id == DEFAULT_MODEL else Path(PROFILES[model_id]).stem
        folder = evidence_dir / slug
        summary, summary_sha = read(folder / "summary.json", item["summary_sha256"])
        require(
            summary.get("report_type") == "jevserve-validation"
            and summary.get("real_gpu") is True
            and summary.get("model_id") == model_id
            and summary.get("model_revision") == selected["revision"]
            and summary.get("source_manifest_identity") == source_manifest(model_id).identity,
            "summary model/source identity mismatch",
        )
        require(
            summary.get("image_id") == summary.get("container_image_id") == aggregate["image_id"],
            "summary/container image identity mismatch",
        )
        require(
            summary.get("source_commit") == aggregate["source_commit"],
            "summary source commit mismatch",
        )
        require(
            summary.get("execution") == "passed" and summary.get("cleanup") == "passed",
            "summary execution not passed",
        )
        docs, hashes = {}, {}
        for filename in ("reference.json", "http.json"):
            expected_sha = item["raw_reports"][filename]["sha256"]
            require(summary["reports"][filename] == expected_sha, "summary report hash differs")
            docs[filename], hashes[filename] = read(folder / filename, expected_sha)
        reference, http = docs["reference.json"], docs["http.json"]
        validate_raw_reports(
            reference,
            http,
            summary,
            item | {"revision": selected["revision"]},
            hashes["reference.json"],
        )
        decimals = None
        if model_id == "jaredpalmer/kev-4b":
            require(
                reference.get("upstream_revision")
                == KEV_ROUNDED_REFERENCE
                == selected["upstream_code"]["revision"],
                "unknown Kev reference serialization",
            )
            decimals = 4
        cases = [
            {
                "case": case_id,
                **compare_answers(
                    actual, expected, numerical_policy, reference_probability_decimals=decimals
                ),
            }
            for case_id, actual, expected in accepted_cases(reference, http)
        ]
        require(cases, "no accepted HTTP cases")
        questions = [q for case in cases for q in case["questions"]]
        output.append(
            {
                "model_id": model_id,
                "model_revision": selected["revision"],
                "source_manifest_identity": summary["source_manifest_identity"],
                "reference_probability_decimals": decimals,
                "summary_sha256": summary_sha,
                "raw_report_sha256": hashes,
                "historical_strict_failed_cases": item["parity_failed_cases"],
                "passed": all(case["passed"] for case in cases),
                "case_count": len(cases),
                "question_count": sum(case["question_count"] for case in cases),
                "native_rejected_case_ids": http.get("native_rejected_case_ids", []),
                "failed_cases": sum(not case["passed"] for case in cases),
                "argmax_flip_count": sum(case["argmax_flip_count"] for case in cases),
                "max_probability_error": max(case["max_probability_error"] for case in cases),
                "max_total_variation": max(q["total_variation"] for q in questions),
                "max_expected_score_error": max(
                    (q["expected_score_error"] for q in questions if q["kind"] == "score"),
                    default=0,
                ),
                "max_normalized_expected_score_error": max(
                    (
                        q["normalized_expected_score_error"]
                        for q in questions
                        if q["kind"] == "score"
                    ),
                    default=0,
                ),
                "max_reported_score_error": max(
                    (q["reported_score_error"] for q in questions if q["kind"] == "score"),
                    default=0,
                ),
                "max_normalized_reported_score_error": max(
                    (
                        q["normalized_reported_score_error"]
                        for q in questions
                        if q["kind"] == "score"
                    ),
                    default=0,
                ),
                "cases": cases,
            }
        )
    return {
        "schema_version": 1,
        "report_type": "jevserve-numerical-policy-reassessment",
        "scope": "New policy applied to existing real GPU/HTTP evidence; synthetic text suite only.",
        "new_gpu_run": False,
        "certified": False,
        "release_eligible": False,
        "policy": policy,
        "policy_sha256": policy_sha,
        "input_aggregate_sha256": aggregate_sha,
        "source_commit": aggregate["source_commit"],
        "image_identity_kind": aggregate["image_identity_kind"],
        "image_id": aggregate["image_id"],
        "evaluator_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (
                "validation/acceptance.py",
                "validation/reassess.py",
                "validation/release.py",
            )
        },
        "all_models_within_policy": all(item["passed"] for item in output),
        "matrix": output,
        "limitations": [
            "Historical strict failures remain unchanged; this is a policy revision, not a numerical fix.",
            "No new workloads, batch/cache invariance, accuracy, calibration or vision certification.",
            "Total variation and Score expectation errors are reported but not additional gates.",
            "Pinned Kev reference serializes to four decimals; reference consistency checks account for rounding without renormalizing or changing probability thresholds.",
            "Model registry and release gates remain unchanged; local identity is not a registry digest.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=ROOT / "compat/numerical-acceptance-v1.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error("output already exists; choose a fresh path")
    report = summarize_matrix(args.aggregate, args.evidence_dir, args.policy)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "all_models_within_policy": report["all_models_within_policy"],
                "models": len(report["matrix"]),
                "new_gpu_run": False,
            }
        )
    )
    return 0 if report["all_models_within_policy"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
