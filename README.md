<p align="center">
  <img src="docs/assets/brand/jevserve-logo.png" alt="JevServe" width="480">
</p>

<p align="center"><strong>Turn text into decisions.</strong></p>
<p align="center">Self-hosted decision models. One GPU. One HTTP API.</p>
<p align="center"><a href="README.zh-CN.md">简体中文</a> · <a href="#quick-start">Quick Start</a> · <a href="docs/quickstart.md">Deployment guide</a> · <a href="docs/models.md">Supported models & tests</a> · <a href="compat/README.md">API compatibility</a></p>

JevServe runs open Jev-style decision models on your NVIDIA GPU. Send text and a question; get a typed answer with probabilities. Use it to route support tickets, flag urgent messages, or score content against a rubric.

| Ask the model to… | Question type | Answer |
|---|---|---|
| Choose a support team | **Choice** | A label and probabilities for each option |
| Decide whether a message is urgent | **Noul** | A probability between 0 and 1 |
| Rate priority from low to high | **Score** | An expected level and its distribution |

Built on **vLLM**, with a Python client, bounded request queues, cancellation, health checks, and an offline mode. Currently serves **[Open-Jev-2B](https://huggingface.co/ZefanCai/Open-Jev-2B)**.

> **Developer preview.** Real GPU inference works; numerical alignment with the author's implementation is not yet certified. See [known limitations](#status-and-limitations) before relying on the results.

## Quick Start

You need **Linux x86_64, an NVIDIA GPU, Docker, and NVIDIA Container Toolkit**. The host does not need Python or CUDA Toolkit. Tested host: RTX A6000. See [deployment requirements](docs/quickstart.md#requirements).

Run the [Docker Hub image](https://hub.docker.com/r/anonwx/jevserve) without a local build:

```bash
docker run -d --name jevserve --gpus device=0 --shm-size=1g \
  -p 127.0.0.1:8000:8000 \
  --mount type=volume,source=jevserve-cache,target=/cache \
  anonwx/jevserve:0.1.0-dev0
docker exec jevserve jevserve check --wait-seconds 1800
```

First startup downloads and verifies approximately 4.6 GB of pinned model files, then compiles the runtime cache; allow several minutes. Later starts reuse the `jevserve-cache` volume. Once ready, the API listens at `127.0.0.1:8000`.

Or use the bundled startup helper:

```bash
git clone https://github.com/anonNo2/jevserve.git
cd jevserve
./scripts/quickstart.sh --image anonwx/jevserve:0.1.0-dev0
```

The helper starts the service, waits for readiness, and sends a real request covering all three question types. Developers can use `--build` to build from source. [Supported models, test environment and evidence →](docs/models.md)

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
JEVSERVE_IMAGE=anonwx/jevserve:0.1.0-dev0 docker compose up --pull always --no-build -d
docker compose exec jevserve jevserve check --wait-seconds 1800
```

Use either Quick Start or Compose on port 8000. [Custom ports, GPU selection, authentication, offline startup, and cleanup →](docs/quickstart.md)

## Use Python

The native client accepts the same request dictionary as the HTTP API:

```python
from jevserve.client import Client

with Client("http://127.0.0.1:8000") as client:
    result = client.decide({
        "state": "Production is down for all users.",
        "questions": {
            "urgent": {
                "type": "noul",
                "instructions": "Does this need an urgent response?",
            }
        },
    })
    print(result.answers["urgent"].noul)
```

Install the client from this checkout with `pip install .` in a Python 3.12 virtual environment. The client does not install GPU dependencies or retry requests. A tested text-question subset of `typesafe-sdk==0.7.4` is available at `/v1/systemone`; see [compatibility](compat/README.md).

## Status and limitations

- **Model:** Open-Jev-2B; text only, one model and one GPU per container.
- **Numerics:** the latest pooling comparison failed 38/113 cases at the frozen probability tolerance; no label flips were observed. These probabilities are not calibrated accuracy. [Reports](docs/development/STATUS.md).
- **Inputs:** up to 16 questions; Choice up to 255 candidates; Score 2–10 text levels. Additional token/work limits apply. No silent truncation, precision change, model fallback, or retry.
- **Release:** development preview, not an alpha-certified or production release. [Current container and release evidence](docs/development/STATUS.md).

## Documentation and contributing

[Deployment](docs/quickstart.md) · [Development setup](docs/development/IMPLEMENTATION.md) · [Architecture](docs/plans/JevServe-design.md) · [Model details](docs/models.md) · [Contributing](CONTRIBUTING.md)

```bash
uv sync --locked
uv run pytest -q
uv run ruff check .
uv run mypy
```

CPU tests need no GPU or model downloads. Bug reports and contributions are welcome; please include your runtime version and a minimal reproducible example.

## License

Apache-2.0. Upstream code and model licenses are documented in [NOTICE](NOTICE) and [model details](docs/models.md).

JevServe is an independent project, unaffiliated with TypeSafe AI or JevServe-Bench. It does not include TypeSafe's official Jev weights.
