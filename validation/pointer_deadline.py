"""Exercise an active pointer deadline against an explicitly disposable service.

This intentionally terminates the selected pointer worker; restart is required.
Run only against a dedicated validation container after its normal smoke suite.
"""

import argparse
import json
import time
from pathlib import Path

import httpx


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with httpx.Client(base_url=args.url, timeout=15) as client:
        client.get("/readyz").raise_for_status()
        metadata = client.get("/v1/models")
        metadata.raise_for_status()
        model = metadata.json()["models"][0]
        if model["id"] not in ("jaredpalmer/kev-4b", "TokenRhythm/NeoHorse-Jev-4B"):
            raise ValueError("this gate requires a dedicated pointer-model service")
        data = {
            "state": "Choose the most appropriate label for a billing request.",
            "questions": {
                "wide": {
                    "type": "choice",
                    "instructions": "Select the matching label.",
                    "criteria": {f"option_{i}": None for i in range(255)},
                }
            },
            "deadline_ms": 100,
        }
        start = time.monotonic()
        result = client.post("/v1/decisions", json=data)
        elapsed = time.monotonic() - start
        assert result.status_code == 504, result.status_code
        unavailable = False
        for _ in range(40):
            try:
                response = client.get("/readyz")
                if response.status_code == 503:
                    unavailable = True
                    break
            except httpx.TransportError:
                unavailable = True
                break
            time.sleep(0.25)
        assert unavailable, "worker should fail closed after active cancellation"
    report = {
        "model_id": model["id"],
        "scope": "real GPU pointer active deadline; dedicated disposable container",
        "deadline_ms": 100,
        "http_status": result.status_code,
        "elapsed_seconds": round(elapsed, 4),
        "service_unavailable_after_deadline": unavailable,
        "automatic_restart": False,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("Active deadline and fail-closed readiness passed")


if __name__ == "__main__":
    main()
