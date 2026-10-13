<p align="center">
  <img src="docs/assets/brand/jevserve-logo.png" alt="JevServe" width="480">
</p>

<p align="center"><strong>Turn text into decisions.</strong></p>
<p align="center">Self-hosted decision models. One GPU. One HTTP API.</p>
<p align="center"><a href="README.zh-CN.md">简体中文</a> · <a href="#quick-start">Quick Start</a> · <a href="docs/quickstart.md">Deployment guide</a> · <a href="docs/models.md">Supported models & tests</a> · <a href="compat/README.md">API compatibility</a></p>

JevServe runs open-weight Jev-style decision models on your NVIDIA GPU. Send text and a question; get a typed answer with probabilities. Use it to route support tickets, flag urgent messages, or score content against a rubric.

| Ask the model to… | Question type | Answer |
|---|---|---|
| Choose a support team | **Choice** | A label and probabilities for each option |
| Decide whether a message is urgent | **Noul** | A probability between 0 and 1 |
| Rate priority from low to high | **Score** | An expected level and its distribution |

**One reusable image, separately mounted weights.** Adapters cover Open-Jev, JPT, NeoHorse and Kev, using vLLM or a named Torch backend. Includes a Python client, bounded request queues, cancellation, health checks and offline mode. See [model-specific test status and licenses](docs/models.md).

> **Developer preview.** Reassessment of existing GPU evidence passes the initial `bounded-drift-v1` policy on the tested corpus; historical strict parity still fails for Open-Jev/JPT. `--allow-uncertified` remains required. This is not full model certification or measured accuracy. See [numerical scope and limitations](#status-and-limitations).

## Quick Start

You need **Linux x86_64, an NVIDIA GPU, Docker, and NVIDIA Container Toolkit**. The host does not need Python or CUDA Toolkit. Tested host: RTX A6000. See [deployment requirements](docs/quickstart.md#requirements).

Run the [Docker Hub image](https://hub.docker.com/r/anonwx/jevserve) without a local build:

```bash
docker run -d --name jevserve --gpus device=0 --shm-size=1g \
  -p 127.0.0.1:8000:8000 \
  --mount type=volume,source=jevserve-cache,target=/cache \
  anonwx/jevserve@sha256:4b758ad1000f4d38501fd83f0a9c2b32f0cb91968d427b170a2c2b0d9a613aff \
  serve --allow-uncertified
docker exec jevserve jevserve check --wait-seconds 1800
```

First startup downloads and verifies approximately 4.6 GB of pinned model files, then compiles the runtime cache; allow several minutes. Later starts reuse the `jevserve-cache` volume. Once ready, the API listens at `127.0.0.1:8000`.

Or use the bundled startup helper:

```bash
git clone https://github.com/anonNo2/jevserve.git
cd jevserve
./scripts/quickstart.sh --image anonwx/jevserve:0.1.0-dev1
```

The helper starts the service, waits for readiness, and sends a real request covering all three question types. Developers can use `--build` to build from source. [Supported models, test environment and evidence →](docs/models.md)

### Mount a model, reuse the image

With an already prepared, pinned weight directory:

```bash
./scripts/quickstart.sh --image anonwx/jevserve:0.1.0-dev1 \
  --model kirp/jpt-0.8b --model-dir /absolute/path/to/verified-model
```

Weights are mounted read-only; runtime caches use a separate volume. Change `--model` and `--model-dir` to select another model without rebuilding the image. Each container loads one model. [Prepare weights, directory layout and mounted Compose](docs/quickstart.md#mounted-models). JPT weights use CC-BY-NC-4.0; check the license before use.

### Make a decision

```bash
curl -fsS http://127.0.0.1:8000/v1/decisions \
  -H 'Content-Type: application/json' \
  -d '{
    "state": "I was charged twice and want a refund.",
    "questions": {
      "team": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {
          "billing": "Payments and refunds",
          "technical": "Software and setup"
        }
      }
    }
  }'
```

Read `answers.team.choice` for the selected team and `answers.team.probabilities` for the candidate distribution. Responses also identify the model revision and execution profile. [Try all three types](examples/decision.json).

### Docker Compose

From the repository checkout:

```bash
JEVSERVE_IMAGE=anonwx/jevserve:0.1.0-dev1 docker compose up --pull always --no-build -d
docker compose exec jevserve jevserve check --wait-seconds 1800
```

Use either Quick Start or Compose on port 8000. [Custom ports, GPU selection, authentication, offline startup, and cleanup →](docs/quickstart.md)

## Use Python

The native client accepts the same request dictionary as the HTTP API:

```python
from jevserve.client import Client

with Client("http://127.0.0.1:8000") as client:
    result = client.decide(
        {
            "state": "Production is down for all users.",
            "questions": {
                "urgent": {
                    "type": "noul",
                    "instructions": "Does this need an urgent response?",
                }
            },
        }
    )
    print(result.answers["urgent"].noul)
```

Install the client from this checkout with `pip install .` in a Python 3.12 virtual environment. The source client installs Pydantic and HTTPX without server or GPU dependencies, and does not retry requests. Source server development uses `pip install ".[server]"`; GPU execution still requires the separate pinned runtime. These source improvements are not included in the published dev1 image. A tested text-question subset of `typesafe-sdk==0.7.4` is available at `/v1/systemone`; see [compatibility](compat/README.md).

## Status and limitations

- **Models:** 7 pinned profiles tested with real GPU inference and containers on RTX A6000; text only, one model and one GPU per container. [Individual test results](docs/models.md).
- **Numerics:** all 7 profiles pass the initial `bounded-drift-v1` policy on 270 recorded requests / 285 questions: absolute probability error ≤0.03 (3 percentage points) and unchanged argmax for every distribution. This reanalyses existing GPU results; it adds no GPU run or arithmetic fix. Historical strict failures remain unchanged, and passing this finite corpus does not certify unseen inputs. [Reassessment](validation/reports/source-13eed9b-bounded-drift.json) · [Per-model scope](docs/models.md).
- **Source image:** the unpublished `13eed9b` candidate passed real inference/SDK checks for all 7 profiles and offline, read-only Compose checks for Open-Jev-2B and NeoHorse. Its immutable local Docker ID is not a published registry digest; published dev1 is unchanged. [CPU/source binding](validation/reports/source-13eed9b-image-cpu.json) · [GPU/Compose evidence](validation/reports/source-13eed9b-image-gpu.json).
- **Lifecycle and soak:** the new image passed concurrent requests, health, deadline and recovery checks. Client cancellation was exercised; server disconnect propagation was not independently verified. The old dev1 24-hour soak failed after 16,271.64 seconds; 1,589 of 1,590 requests succeeded. No completed 24/72-hour gate or clean-host acceptance. [Soak failure](validation/reports/soak-24h-failed.json).
- **Inputs:** up to 16 questions and 255 Choice candidates. Score limits, minimum candidates and token/work budgets vary by model; inspect `/v1/models`. No silent truncation, precision change, model fallback or retry.
- **Release:** development preview, not an alpha-certified or production release. [Current status](docs/current-status.md) · [Detailed evidence](docs/development/STATUS.md).

## Documentation and contributing

[API reference](docs/api.md) · [Deployment](docs/quickstart.md) · [Development setup](docs/development/IMPLEMENTATION.md) · [Architecture](docs/plans/JevServe-design.md) · [Model details](docs/models.md) · [Contributing](CONTRIBUTING.md)

```bash
uv sync --locked --extra server
uv run --extra server pytest -q
uv run --extra server ruff check .
uv run --extra server mypy
```

CPU tests need no GPU or model downloads. Bug reports and contributions are welcome; please include your runtime version and a minimal reproducible example.

[Reproduce validation](docs/validation.md) · [Numerical diagnosis](docs/numerical-diagnosis.md) · [Release process](docs/releasing.md) · [Open-source hardening plan](docs/plans/open-source-hardening.md)

[Business examples](docs/examples.md) · [Model contribution guide](docs/contributing-models.md) · [Performance and soak](docs/performance-and-soak.md) · [Security reporting](SECURITY.md) · [Image SBOM](docs/sbom.md)

## License

Apache-2.0. Upstream code and model licenses are documented in [NOTICE](NOTICE) and [model details](docs/models.md).

JevServe is an independent project, unaffiliated with TypeSafe AI or JevServe-Bench. It does not include TypeSafe's official Jev weights.
