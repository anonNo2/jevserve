"""Reanalyse committed numerical evidence without GPU dependencies or changed tolerances."""

import argparse
import hashlib
import json
import math
from pathlib import Path


def softmax(values, temperature):
    if not values or temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("finite positive temperature and nonempty logits required")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("nonfinite logits")
    exps = [math.exp((value - max(values)) / temperature) for value in values]
    return [value / sum(exps) for value in exps]


def error(actual, expected):
    return max(abs(a - b) for a, b in zip(actual, expected, strict=True))


def compare(actual, expected, tolerance):
    return all(
        abs(a - b) <= tolerance["probability_atol"] + tolerance["probability_rtol"] * abs(b)
        for a, b in zip(actual, expected, strict=True)
    )


def indexed(cases, key):
    result = {case[key]: case for case in cases}
    if len(result) != len(cases):
        raise ValueError("duplicate case IDs")
    return result


def pooling(reference, runtime, tolerance):
    actual_cases = indexed(runtime["cases"], "id")
    expected_cases = indexed(reference["cases"], "id")
    if actual_cases.keys() != expected_cases.keys():
        raise ValueError("case IDs differ")
    rows = []
    for case_id, expected in expected_cases.items():
        actual = actual_cases[case_id]
        offset = 0
        for record, logits, probabilities, observed in zip(
            expected["records"], expected["logits"], expected["probabilities"],
            actual["probabilities"], strict=True,
        ):
            count = 1 if record["kind"] == "noul" else len(logits)
            scores = actual["scores"][offset : offset + count]
            offset += count
            if record["kind"] == "noul":
                scores = [0.0, scores[0]]
            # Subtracting a shared constant isolates differences that softmax can observe.
            centered_expected = [v - logits[0] for v in logits]
            centered_actual = [v - scores[0] for v in scores]
            rows.append({
                "case": case_id, "question": record["id"], "kind": record["kind"],
                "candidate_count": len(logits),
                "centered_logit_max_error": error(centered_actual, centered_expected),
                "reference_reconstruction_error": error(
                    softmax(logits, reference["temperature"]), probabilities
                ),
                "runtime_reconstruction_error": error(
                    softmax(scores, reference["temperature"]), observed
                ),
                "max_probability_error": error(observed, probabilities),
                "within_tolerance": compare(observed, probabilities, tolerance),
            })
        if offset != len(actual["scores"]):
            raise ValueError("unconsumed runtime scores")
    return {
        "case_count": len(expected_cases), "question_count": len(rows),
        "failed_cases": len({r["case"] for r in rows if not r["within_tolerance"]}),
        "max_reference_reconstruction_error": max(r["reference_reconstruction_error"] for r in rows),
        "max_runtime_reconstruction_error": max(r["runtime_reconstruction_error"] for r in rows),
        "max_centered_logit_error": max(r["centered_logit_max_error"] for r in rows),
        "questions": rows,
    }


def answer_values(answer):
    if answer["type"] == "noul":
        return {"true": answer["noul"]}
    return answer["probabilities"]


def jpt(reference, runtime, tolerance):
    expected_cases = indexed(reference["cases"], "case")
    actual_cases = indexed(runtime["cases"], "case")
    if expected_cases.keys() != actual_cases.keys():
        raise ValueError("case IDs differ")
    groups = {}
    failed = set()
    flips = 0
    for case_id, expected in expected_cases.items():
        actual = actual_cases[case_id]
        if actual["answers"].keys() != expected["answers"].keys():
            raise ValueError("question IDs differ")
        for key, answer in expected["answers"].items():
            observed = actual["answers"][key]
            if answer["type"] != observed["type"]:
                raise ValueError("question types differ")
            a, b = answer_values(observed), answer_values(answer)
            if list(a) != list(b):
                raise ValueError("candidate order differs")
            kind = answer["type"]
            group = groups.setdefault(kind, {"questions": 0, "failed_questions": 0,
                                             "max_probability_error": 0.0})
            group["questions"] += 1
            passed = compare(list(a.values()), list(b.values()), tolerance)
            group["failed_questions"] += int(not passed)
            group["max_probability_error"] = max(
                group["max_probability_error"], error(list(a.values()), list(b.values()))
            )
            if not passed:
                failed.add(case_id)
            if kind == "choice":
                flips += observed["choice"] != answer["choice"]
    return {"case_count": len(expected_cases), "failed_cases": len(failed),
            "choice_label_flips": flips, "by_question_type": groups,
            "raw_logits_available": False}


def analyse(reports, tolerance_path):
    hashes = {}

    def read(path):
        raw = path.read_bytes()
        hashes[path.name] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    tolerance = read(tolerance_path)
    reference = read(reports / "open-jev-reference.json")
    runtime = read(reports / "vllm-pooling-parity.json")
    if runtime["reference_sha256"] != hashes["open-jev-reference.json"]:
        raise ValueError("pooling evidence refers to another reference")
    if runtime["tolerance"] != tolerance:
        raise ValueError("recorded tolerance differs from frozen profile")
    result = {"scope": "CPU reanalysis of historical real-GPU evidence; no new inference",
              "tolerance": tolerance,
              "open_jev_2b": pooling(reference, runtime, tolerance), "jpt": {}}
    for slug in ("jpt-0.8b", "jpt-4b", "jpt-9b"):
        ref, run = read(reports / f"{slug}-reference.json"), read(reports / f"{slug}-container.json")
        if (ref["model_id"], ref["revision"]) != (run["model_id"], run["revision"]):
            raise ValueError("model identity mismatch")
        result["jpt"][slug] = jpt(ref, run, tolerance)
    result["input_sha256"] = hashes
    result["root_cause_resolved"] = False
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, default=Path("validation/reports"))
    parser.add_argument("--tolerance", type=Path, default=Path("compat/parity-profile.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(analyse(args.reports, args.tolerance), indent=2) + "\n")
    return 0  # Successful analysis does not certify the analysed runtime.


if __name__ == "__main__":
    raise SystemExit(main())
