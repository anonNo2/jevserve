"""Run one real text decision; no retries, mock outputs or external business writes."""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from jevserve.client import Client, JevServeError
from jevserve.contracts.requests import DecisionRequest
from jevserve.contracts.responses import DecisionResponse

SCENARIOS = ("ticket-routing", "content-moderation", "priority-scoring")


def load_request(scenario: str) -> DecisionRequest:
    if scenario not in SCENARIOS:
        raise ValueError("unknown scenario")
    path = Path(__file__).parent / "business" / f"{scenario}.json"
    return DecisionRequest.from_json(path.read_bytes())


def check_response(request: DecisionRequest, response: DecisionResponse) -> None:
    """Reject valid-shaped answers that belong to a different question or rubric."""
    if list(response.answers) != list(request.questions):
        raise ValueError("response question mismatch")
    for key, question in request.questions.items():
        answer = response.answers[key]
        if answer.type != question.type:
            raise ValueError("response answer type mismatch")
        if question.type == "choice":
            if list(answer.probabilities) != list(question.criteria):
                raise ValueError("response candidate mismatch")
        if question.type == "score":
            if list(answer.legend.values()) != question.criteria:
                raise ValueError("response score rubric mismatch")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", choices=SCENARIOS)
    parser.add_argument("--url", default=os.getenv("JEVSERVE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--timeout", type=float, default=35, help="HTTP timeout in seconds")
    args = parser.parse_args(argv)
    try:
        url = urlsplit(args.url)
        if (
            url.scheme not in ("http", "https")
            or not url.hostname
            or url.username is not None
            or url.password is not None
            or url.query
            or url.fragment
            or url.path not in ("", "/")
            or not math.isfinite(args.timeout)
            or args.timeout <= 0
        ):
            raise ValueError("invalid connection settings")
        request = load_request(args.scenario)
        with Client(args.url, timeout=args.timeout, api_key_file=args.api_key_file) as client:
            response = client.decide(request)
        check_response(request, response)
    except JevServeError as exc:
        # Do not echo a remote error body/code, request content, URL or credential.
        print(
            f"Service rejected the request (HTTP {exc.status}); see docs/examples.md.",
            file=sys.stderr,
        )
        return 1
    except httpx.TimeoutException:
        print("Request timed out; no retry was attempted.", file=sys.stderr)
        return 1
    except httpx.HTTPError:
        print("HTTP connection/protocol failure; no retry was attempted.", file=sys.stderr)
        return 1
    except (OSError, ValueError, httpx.InvalidURL):
        print(
            "Invalid configuration, unreadable input/key file, or invalid response.",
            file=sys.stderr,
        )
        return 2
    print(json.dumps({"scenario": args.scenario, "response": response.model_dump()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
