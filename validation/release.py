"""Offline, fail-closed release review. No GPU imports or network access."""

import argparse
import base64
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path, PurePosixPath
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, Request, build_opener

DEPENDENCIES = {
    "compat/engine-lock.json", "compat/runtime-requirements.txt",
    "compat/reference-requirements.txt", "uv.lock", "pyproject.toml",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def match(value, pattern, label):
    require(isinstance(value, str) and re.fullmatch(pattern, value), f"invalid {label}")


def relative(value):
    require(isinstance(value, str), "path must be text")
    path = PurePosixPath(value)
    require(value and str(path) == value and not path.is_absolute()
            and ".." not in path.parts and "\\" not in value, "unsafe evidence path")
    return value


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.PIPE)


def read_json(data):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    return json.loads(data, object_pairs_hook=unique)


def checked_file(root, entry):
    name = relative(entry["path"])
    path = root / name
    require(not any(p.is_symlink() for p in [path, *path.parents]), "symlink evidence")
    require(path.resolve().is_relative_to(root.resolve()), "evidence outside repository")
    data = path.read_bytes()
    match(entry["sha256"], r"[0-9a-f]{64}", "evidence SHA-256")
    require(hashlib.sha256(data).hexdigest() == entry["sha256"], "evidence digest mismatch")
    return data


def validate_raw_reports(reference, http, summary, model, reference_sha):
    """Check the two report formats emitted by reference/http and model_matrix."""
    revision = model["revision"]
    for document in (reference, http):
        require(isinstance(document, dict), "raw report must be an object")
        require(document.get("model_revision", document.get("revision")) == revision,
                "raw report model revision mismatch")
        if "model_id" in document:
            require(document["model_id"] == model["model_id"], "raw report model ID mismatch")
        require(isinstance(document.get("cases"), list) and document["cases"],
                "raw report requires nonempty cases")
    matrix = reference.get("mode") == "reference"
    if matrix:
        require(http.get("mode") == "http" and reference.get("model_id") == model["model_id"]
                and http.get("model_id") == model["model_id"], "mixed raw report formats")
        case_key, parity_key, failures_key = "case", "parity_passed", "parity_failed_cases"
        require(http.get("systemone_smoke") == "passed"
                and http.get("native_client_smoke") == "passed"
                and http.get("unknown_model_rejected") is True, "incomplete HTTP smoke evidence")
    else:
        require("mode" not in reference and "mode" not in http, "unknown raw report format")
        require(reference.get("device_type") == "cuda"
                and type(reference.get("failures")) is int and reference["failures"] == 0,
                "reference GPU execution failed")
        for document in (reference, http):
            require(type(document.get("case_count")) is int
                    and document["case_count"] == len(document["cases"]),
                    "raw case count mismatch")
        require(http.get("reference_sha256") == reference_sha, "HTTP reference digest mismatch")
        case_key, parity_key, failures_key = "id", "within_tolerance", "failures"
    ids = []
    for case in reference["cases"]:
        require(isinstance(case, dict) and case_key in case and "error" not in case,
                "incomplete reference case")
        answers = case.get("answers") if matrix else case.get("response", {}).get("answers")
        require(isinstance(answers, dict) and answers, "reference case missing answers")
        ids.append(case[case_key])
    observed = []
    failures = 0
    for case in http["cases"]:
        require(isinstance(case, dict) and case_key in case and "error" not in case,
                "incomplete HTTP case")
        require((not matrix and "status" not in case) or case.get("status") == 200,
                "HTTP execution failure")
        require(isinstance(case.get("answers"), dict) and case["answers"],
                "HTTP case missing answers")
        require(type(case.get(parity_key)) is bool, "HTTP parity not measured")
        failures += not case[parity_key]
        observed.append(case[case_key])
    require(len(set(ids)) == len(ids) and len(set(observed)) == len(observed),
            "duplicate raw case IDs")
    rejected = [] if matrix else http.get("native_rejected_case_ids", [])
    require(isinstance(rejected, list) and len(set(rejected)) == len(rejected)
            and not set(observed) & set(rejected) and set(ids) == set(observed) | set(rejected),
            "reference/HTTP case coverage mismatch")
    match(http.get("official_sdk_smoke"), r"typesafe-sdk==[0-9]+\.[0-9]+\.[0-9]+",
          "official SDK execution evidence")
    for document, key in ((http, failures_key), (summary, "parity_failed_cases"),
                          (model, "parity_failed_cases")):
        require(type(document.get(key)) is int and document[key] == failures,
                "raw/summary/aggregate parity failure count mismatch")
    require(model["parity"] == ("failed" if failures else "passed"),
            "raw parity contradicts aggregate status")


def validate(manifest, root, *, expected_commit=None):
    require(manifest["schema_version"] == 1, "unknown release schema")
    # A small synthetic parity suite is insufficient to implement certification.
    require(manifest["channel"] == "development-preview",
            "alpha/rc/stable disabled: certification evidence policy not implemented")
    match(manifest["version"], r"[0-9]+\.[0-9]+\.[0-9]+-dev[0-9]+", "preview version")
    commit = manifest["source_commit"]
    match(commit, r"[0-9a-f]{40}", "source commit")
    require(expected_commit is None or commit == expected_commit, "requested commit mismatch")
    require(git(root, "rev-parse", f"{commit}^{{commit}}").decode().strip() == commit,
            "source commit does not exist")
    subprocess.run(["git", "-C", str(root), "merge-base", "--is-ancestor", commit, "HEAD"],
                   check=True, capture_output=True)
    require(manifest["image_repository"] == "anonwx/jevserve", "unexpected image repository")
    match(manifest["image_digest"], r"sha256:[0-9a-f]{64}", "registry manifest digest")
    require(isinstance(manifest["limitations"], list) and manifest["limitations"]
            and all(isinstance(x, str) and x.strip()
            for x in manifest["limitations"]), "preview limitations required")
    sources = manifest["source_files"]
    paths = [relative(item["path"]) for item in sources]
    require(len(paths) == len(set(paths)) and DEPENDENCIES <= set(paths),
            "missing or duplicate dependency locks")
    catalog = set(git(root, "ls-tree", "-r", "--name-only", commit,
                      "src/jevserve/data").decode().splitlines())
    catalog = {name for name in catalog if name.endswith(".json")}
    require(catalog and catalog <= set(paths), "all shipped model manifests required")
    models = {}
    for entry in sources:
        name = entry["path"]
        match(entry["sha256"], r"[0-9a-f]{64}", "source file digest")
        data = git(root, "show", f"{commit}:{name}")
        require(hashlib.sha256(data).hexdigest() == entry["sha256"], "source file digest mismatch")
        if name in catalog:
            model = read_json(data)
            match(model["revision"], r"[0-9a-f]{40}", "model revision")
            require(model["model_id"] not in models, "duplicate model")
            models[model["model_id"]] = model["revision"]
    evidence = read_json(checked_file(root, manifest["evidence"]))
    require(evidence["schema_version"] == 1, "unknown evidence schema")
    require(evidence["source_commit"] == commit, "evidence commit mismatch")
    require(evidence["image_digest"] == manifest["image_digest"], "evidence image mismatch")
    require(evidence["real_gpu"] is True, "real GPU evidence required")
    require(evidence["gates"] == dict.fromkeys(("cpu", "docker_run", "compose", "offline"),
                                               "passed"), "missing execution gate")
    require(set(evidence["gate_reports"]) == set(evidence["gates"]),
            "execution gate reports required")
    for gate, entry in evidence["gate_reports"].items():
        gate_report = read_json(checked_file(root, entry))
        require(gate_report.get("report_type") == "jevserve-release-gate"
                and gate_report.get("schema_version") == 1
                and gate_report.get("source_commit") == commit
                and gate_report.get("image_digest") == manifest["image_digest"]
                and gate_report.get("gate") == gate and gate_report.get("status") == "passed",
                "execution gate report binding mismatch")
        require(gate_report.get("commands") and gate_report.get("reports"),
                "execution gate lacks commands/raw evidence")
        for report in gate_report["reports"]:
            checked_file(root, report)
    observed = {}
    failed = []
    for item in evidence["models"]:
        key = item["model_id"]
        require(key not in observed, "duplicate model evidence")
        observed[key] = item["revision"]
        require(item["execution"] == "passed", "model execution failed")
        require(item["parity"] in ("passed", "failed"), "parity must be measured")
        if item["parity"] == "failed":
            failed.append(key)
        require(item["reports"], "missing raw model reports")
        summaries = []
        for report in item["reports"]:
            document = read_json(checked_file(root, report))
            if document.get("report_type") == "jevserve-validation":
                summaries.append((report, document))
        require(len(summaries) == 1, "one bound validation runner summary required per model")
        report, summary = summaries[0]
        require(summary.get("schema_version") == 1 and summary.get("source_commit") == commit
                and summary.get("source_dirty") is False
                and summary.get("image_digest") == manifest["image_digest"]
                and summary.get("model_id") == key
                and summary.get("model_revision") == item["revision"],
                "runner summary source/image/model binding mismatch")
        require(summary.get("real_gpu") is True and summary.get("execution") == "passed"
                and summary.get("cleanup") == "passed" and summary.get("parity") == item["parity"],
                "runner summary lacks completed real GPU evidence")
        raw_reports = summary.get("reports", {})
        require({"reference.json", "http.json"} <= set(raw_reports), "missing raw runner reports")
        documents = {}
        for name, sha in raw_reports.items():
            relative(name)
            data = checked_file(root, {"path": str(PurePosixPath(report["path"]).parent / name),
                                       "sha256": sha})
            if name in ("reference.json", "http.json"):
                documents[name] = read_json(data)
        validate_raw_reports(documents["reference.json"], documents["http.json"], summary,
                             item, raw_reports["reference.json"])
    require(observed == models, "evidence must cover exactly the shipped model revisions")
    require(sorted(manifest["failed_parity_models"]) == sorted(failed),
            "failed parity must be declared explicitly")
    return {"version": manifest["version"], "source_commit": commit,
            "image_repository": manifest["image_repository"],
            "image_digest": manifest["image_digest"], "channel": manifest["channel"]}


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("registry redirect refused")


def registry_request(url, *, headers=None, data=None, method="GET"):
    # Never follow a redirect with registry credentials; all hosts are fixed below.
    request = Request(url, headers=headers or {}, data=data, method=method)
    with build_opener(NoRedirect).open(request, timeout=60) as response:
        body = response.read(16 * 1024 * 1024 + 1)
        require(len(body) <= 16 * 1024 * 1024, "oversized registry response")
        return body, response.headers


def promote(result):
    """Retag exact existing manifest bytes in the same Docker Hub repository."""
    username = os.environ["DOCKERHUB_USERNAME"]
    password = os.environ["DOCKERHUB_TOKEN"]
    require(username and password, "Docker Hub credentials required")
    basic = base64.b64encode(f"{username}:{password}".encode()).decode()
    token_body, _ = registry_request(
        "https://auth.docker.io/token?service=registry.docker.io"
        "&scope=repository:anonwx/jevserve:pull,push",
        headers={"Authorization": f"Basic {basic}"})
    token = read_json(token_body)["token"]
    headers = {"Authorization": f"Bearer {token}", "Accept": ", ".join((
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.docker.distribution.manifest.v2+json"))}
    base = "https://registry-1.docker.io/v2/anonwx/jevserve/manifests/"
    digest = result["image_digest"]
    data, response_headers = registry_request(base + digest, headers=headers)
    require("sha256:" + hashlib.sha256(data).hexdigest() == digest,
            "registry source bytes differ from tested digest")
    target = base + result["version"]
    try:
        registry_request(target, headers=headers)
    except HTTPError as exc:
        require(exc.code == 404, "cannot establish target tag absence")
    else:
        raise ValueError("target tag already exists; refusing overwrite")
    media_type = response_headers["Content-Type"].split(";", 1)[0]
    require(media_type in headers["Accept"].split(", "), "unsupported manifest media type")
    # No build, layer upload, reserialization, annotations or platform selection.
    registry_request(target, headers={**headers, "Content-Type": media_type},
                     data=data, method="PUT")
    published, _ = registry_request(target, headers=headers)
    require("sha256:" + hashlib.sha256(published).hexdigest() == digest,
            "published tag digest mismatch; inspect registry before further action")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--github-output", type=Path)
    parser.add_argument("--promote", action="store_true", help="write reviewed registry tag")
    args = parser.parse_args()
    try:
        result = validate(read_json(args.manifest.read_bytes()), args.root.resolve(),
                          expected_commit=args.expected_commit)
        if args.promote:
            promote(result)
    except (ValueError, KeyError, TypeError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"release rejected: {type(exc).__name__}: {exc}\n")
    if args.github_output:
        with args.github_output.open("a") as stream:
            for key, value in result.items():
                stream.write(f"{key}={value}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
