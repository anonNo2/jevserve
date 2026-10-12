# JevServe

**Serve Jev-style models. One command.**

[简体中文](README.zh-CN.md) · [Design](docs/plans/JevServe-design.md) · [Evidence and next steps](docs/development/STATUS.md)

JevServe serves text Choice, Noul and Score decisions on one Linux NVIDIA GPU. One API process owns one vLLM engine. It is independent of **TypeSafe AI** and **JevServe-Bench**; it does not distribute TypeSafe's official Jev weights or imply their endorsement.

**Development milestone:** the CLI → HTTP → vLLM → real `ZefanCai/Open-Jev-2B` path runs on an RTX A6000. Native SDK, bounded admission, deadlines, cancellation, health checks, metrics and verified offline artifact loading are implemented. **No model is certified:** the current engine failed 38/113 pretokenized reference comparisons at the frozen tolerance (maximum probability error 0.0218; no label flips). Docker creation on the development host fails before the entrypoint; image build and both container inference paths remain unverified. No image or package has been published.

## Run from source

Use a project-local Python 3.12 environment and a compatible NVIDIA host. The runtime lock is large and includes CUDA dependencies. Do not replace system Python or CUDA.

```bash
uv venv --python 3.12 .venv-runtime
uv pip sync --python .venv-runtime/bin/python --require-hashes compat/runtime-requirements.txt
uv pip install --python .venv-runtime/bin/python --no-deps .
.venv-runtime/bin/jevserve serve --allow-uncertified --cache-dir .cache/jevserve
```

The development flag acknowledges incomplete certification; it does not relax parity thresholds. First start downloads exact source revisions, verifies hashes, derives the text checkpoint without merging FP32 LoRA weights, loads the GPU engine and warms it up. Readiness becomes available after startup. Subsequent starts reverify the cache; add `--offline` to prohibit model downloads. [Detailed setup and operation](docs/development/IMPLEMENTATION.md).

```bash
.venv-runtime/bin/jevserve check
curl -fsS http://127.0.0.1:8000/v1/decisions \
  -H 'Content-Type: application/json' \
  -d '{"state":"I was charged twice.","questions":{"team":{"type":"choice","instructions":"Which team should handle this?","criteria":{"billing":"Payments and refunds","technical":"Technical problems"}}}}'
```

The native Python client is `jevserve.client.Client`; `decide()` accepts a `DecisionRequest` or dictionary and returns a validated response. It performs no retries.

## Container paths — implemented, not yet verified

The Dockerfile pins the upstream vLLM image by digest. The Compose configuration passes structural validation. The following still require successful build and real GPU verification on a working Docker host:

```bash
docker build -t jevserve:dev .
docker run --rm --name jevserve-dev --gpus device=0 --shm-size=1g \
  -p 127.0.0.1:8000:8000 -v jevserve-cache:/cache \
  jevserve:dev serve --allow-uncertified
# Alternative:
docker compose up --build -d
```

## API and limits

| Endpoint | Behavior |
|---|---|
| `POST /v1/decisions` | Native answers and execution metadata |
| `POST /v1/systemone` | Tested text-question subset of `typesafe-sdk==0.7.4`; exact model ID required |
| `GET /v1/models` | Model revision and capability limits |
| `GET /livez`, `/readyz` | API liveness and engine readiness |
| `GET /metrics` | Prometheus counters, latency and admission gauges |

Native state is nonempty text or a JSON object. Instructions and descriptions are text; Choice permits null descriptions. Score accepts 2–10 ordered levels. The compatibility endpoint does not promise the hosted API's full object/array question schema or model aliases. [Compatibility details](compat/README.md).

Requests are bounded by 1 MiB, 16 questions, 256 candidate sequences, 4096 tokens per candidate and 65536 total input tokens. Defaults: 16 active requests, 64 waiting, 30-second deadline. Inputs are never silently truncated. Bearer authentication uses `--api-key-file`; health endpoints remain public. Bind to loopback by default.

Probabilities describe the supplied candidate set, **not calibrated accuracy**. `calibration_id` is null; Score is an ordinal expectation. There is no fake production engine, automatic retry, model fallback, quantization or precision fallback.

## Development and release gates

```bash
uv sync --locked
uv run pytest -q
uv run ruff check .
uv run mypy
uv run python -m build --no-isolation
```

CPU tests require no GPU imports or model downloads. Author GPU reference, engine parity, container build, Docker/Compose inference and long-duration soak are separate gates. See [STATUS](docs/development/STATUS.md), [decisions](docs/development/DECISIONS.md) and [contribution guide](CONTRIBUTING.md).

Alpha targets one image, one model and one GPU. Kubernetes, training, a web UI, multimodal, multi-node and a Triton implementation are out of scope. A second model is optional. Project code is Apache-2.0; upstream code notices and model licenses are preserved separately in [NOTICE](NOTICE) and `licenses/`.
