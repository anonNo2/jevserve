"""Actual HTTP parity against the frozen author GPU reference; no fake fallback."""

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path

import httpx

from jevserve.client import Client
from jevserve.contracts.requests import DecisionRequest


def main() -> int:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument(
        "--reference", type=Path, default=Path("validation/reports/open-jev-reference.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text())
    tolerance = json.loads(Path("compat/parity-profile.json").read_text())
    cases = []
    rejected = []
    with Client(args.url) as client:
        for case in reference["cases"]:
            try:
                request = DecisionRequest.model_validate(case["request"])
            except ValueError:
                rejected.append(case["id"])
                continue
            response = client.decide(request)
            maximum = 0.0
            passed = True
            flips = []
            for key, answer in response.answers.items():
                expected = case["response"]["answers"][key]
                if answer.type == "noul":
                    actual = [1 - answer.noul, answer.noul]
                    original = [1 - expected["noul"], expected["noul"]]
                else:
                    actual = list(answer.probabilities.values())
                    original = list(expected["probabilities"].values())
                    assert list(answer.probabilities) == list(expected["probabilities"])
                for x, y in zip(actual, original, strict=True):
                    maximum = max(maximum, abs(x - y))
                    passed &= abs(x - y) <= tolerance["probability_atol"] + tolerance[
                        "probability_rtol"
                    ] * abs(y)
                    assert math.isfinite(x)
                if max(range(len(actual)), key=actual.__getitem__) != max(
                    range(len(original)), key=original.__getitem__
                ):
                    flips.append(key)
            cases.append(
                {
                    "id": case["id"],
                    "within_tolerance": passed,
                    "max_probability_error": maximum,
                    "label_flips": flips,
                    "answers": {
                        key: answer.model_dump() for key, answer in response.answers.items()
                    },
                }
            )
    report = {
        "case_count": len(cases),
        "native_rejected_case_ids": rejected,
        "failures": sum(not case["within_tolerance"] for case in cases),
        "max_probability_error": max(case["max_probability_error"] for case in cases),
        "label_flip_count": sum(len(case["label_flips"]) for case in cases),
        "model_revision": response.model_revision,
        "execution_profile": response.execution_profile,
        "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        "tolerance": tolerance,
        "transport": "real HTTP, sequential requests",
        "cases": cases,
    }
    # Exercise the public SDK against the selected model after native parity.
    from typesafe_sdk import Choice, Noul, RetryPolicy, Score, TypeSafeClient

    metadata = httpx.get(args.url + "/v1/models", timeout=10)
    metadata.raise_for_status()
    model_id = metadata.json()["models"][0]["id"]
    with TypeSafeClient(
        api_key="local-validation", model=model_id, base_url=args.url,
        retry=RetryPolicy(max_retries=0),
    ) as sdk:
        result = sdk.system_one(
            state="I was charged twice and want a refund.",
            questions={
                "team": Choice(instructions="Choose a team", criteria={
                    "billing": "Payments", "technical": "Software",
                }),
                "urgent": Noul(instructions="Is this urgent?"),
                "priority": Score(instructions="Rate urgency", criteria=["Low", "High"]),
            },
        )
        assert result.model == model_id
        assert result.choices["team"].choice in ("billing", "technical")
        assert 0 <= result.nouls["urgent"].noul <= 1
        assert 0 <= result.scores["priority"].score <= 1
    report["official_sdk_smoke"] = "typesafe-sdk==" + importlib.metadata.version("typesafe-sdk")
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key not in ("cases", "tolerance")}
        )
    )
    return 0 if report["failures"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
