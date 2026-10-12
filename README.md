# JevServe

**Serve Jev-style models. One command.**

[简体中文](README.zh-CN.md) · [Design](docs/plans/JevServe-design.md) · [Implementation plan](docs/plans/JevServe-task-plan.md) · [Development handoff](docs/development/HANDOFF.md)

JevServe is a planned open-source inference server for Jev-style / System One decision models. Its goal is to make typed decisions as easy to serve as an LLM: one container image, one command, a documented API, and reproducible model validation.

> **Status: implementation started; CPU contracts, verified artifact preparation and an Open-Jev adapter are implemented. Its encoding matches 110 author/real-tokenizer cases; 113 author GPU reference cases passed. JevServe/vLLM inference and container certification remain pending.** The Python development package is installable; inference, CLI and containers are not implemented. No model is certified or image published. The quickstart commands below remain targets. See [development setup](docs/development/IMPLEMENTATION.md) and [current evidence](docs/development/STATUS.md).

## What we are building

- **Simple deployment:** Docker or Docker Compose on a single Linux NVIDIA GPU host.
- **Typed decisions:** Choice, Noul (yes/no), and Score with explicit probability semantics.
- **A proven execution engine:** vLLM scheduling and batching with bounded model adapters.
- **Reproducible artifacts:** pinned weights, tokenizers, templates, runtime versions, and image digests.
- **Operational essentials:** readiness, bounded queues, deadlines, cancellation, metrics, offline packages, and rollback.
- **Evidence-based support:** real GPU tests and reference comparisons for every certified configuration.

The first milestone is a working alpha repository, not feature parity or maturity parity with vLLM. Triton, Kubernetes, training, a web UI, multimodal input, and automatic model switching are outside the first release.

## Target quickstart — not available yet

Prerequisites for the future runtime: Linux x86_64, a compatible NVIDIA GPU and driver, Docker Engine, and NVIDIA Container Toolkit. Host Python and CUDA Toolkit should not be required.

After the Dockerfile and runtime have been implemented, a local development build should run with:

```bash
docker build -t jevserve:dev .
docker run --rm --gpus 'device=0' --shm-size=2g \
  -p 127.0.0.1:8000:8000 \
  -v jevserve-cache:/var/lib/jevserve \
  jevserve:dev serve --model ZefanCai/Open-Jev-2B
```

The release quickstart will use an actual published image instead of requiring a build. The alternative target is:

```bash
docker compose up -d
```

On the first online start, the container will prepare and verify the pinned model; later starts will reuse the cache. Offline operation will require a prepared model package. Wait for readiness before inference:

```bash
curl -fsS http://127.0.0.1:8000/readyz
curl -fsS http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"state":"I was charged twice.","questions":{"team":{"type":"choice","instructions":"Which team should handle this?","criteria":{"billing":"Payments and refunds","technical":"Technical problems"}}}}'
```

No example response is presented as a measured model result. Authentication and deployment settings will be documented alongside the implemented service.

## Planned model support

| Model | Role | Current status |
|---|---|---|
| `ZefanCai/Open-Jev-2B` | First adapter target | Pinned author GPU reference: 113 cases passed; JevServe/vLLM and containers unverified |
| `convaiinnovations/laya` | Optional second adapter | Planned; not required for alpha |
| TypeSafe's official Jev | Not distributed by this project | No official weights included |

A model name in a plan does not mean it is supported. The release matrix will bind each certification to a checkpoint revision, GPU, precision, runtime, and test report. Historical upstream claims and versions must be rechecked before implementation.

## Planned API

| Endpoint | Purpose |
|---|---|
| `POST /v1/decisions` | Native typed decisions and execution metadata |
| `POST /v1/systemone` | Version-tested compatibility interface |
| `GET /v1/models` | Deployed model and capabilities |
| `GET /livez`, `GET /readyz` | Liveness and model readiness |
| `GET /metrics` | Prometheus metrics |

A normalized probability is not automatically calibrated accuracy. The initial service will report `calibration_id: null` unless a validated calibration artifact is explicitly used.

## Start development here

1. Read [AGENTS.md](AGENTS.md), the [handoff](docs/development/HANDOFF.md), the [design](docs/plans/JevServe-design.md), and the [task plan](docs/plans/JevServe-task-plan.md).
2. Use [STATUS.md](docs/development/STATUS.md) to locate the next task: **T00 — complete the real reference and container baseline**, alongside the CPU work recorded there.
3. Keep CPU tests and GPU certification separate. A mock, successful image build, or upstream README is not GPU validation.
4. Record commands, versions, and evidence before marking a task complete.

Implementation phases: baseline and contracts → model package and adapter → real vLLM inference → API/CLI → Docker/Compose → reliability and validation → release documentation.

See [CONTRIBUTING.md](CONTRIBUTING.md) for working conventions. The design and task plan currently use Chinese; the English README is the public entry point.

## Independence and license

JevServe is an independent project, not affiliated with or endorsed by TypeSafe AI or JevServe-Bench. “Jev-style” describes the class of decision models; this repository does not provide TypeSafe's official Jev weights.

Repository materials are licensed under [Apache-2.0](LICENSE). Model weights and third-party components retain their own licenses. See [NOTICE](NOTICE). Repository, package, and image publication namespaces have not been reserved or published.
