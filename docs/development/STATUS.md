# Development status

## Current milestone

A usable **development prototype** now loads pinned Jev-style models and
serves Choice/Noul/Score over HTTP. CLI, native SDK, bounded lifecycle, artifact
preparation, Dockerfile and Compose framework are implemented. This is **not an
alpha release or certified model**: frozen numerical parity still fails. A real
development image now builds and serves all three question types with no network
and a read-only root filesystem. Cold startup and Compose real inference also
pass. The shared daemon is unchanged. Source is now public on
[GitHub](https://github.com/anonNo2/jevserve), and its first remote CPU CI run passed.
The tested development image is public as `anonwx/jevserve:0.1.0-dev0`; anonymous
manifest access, digest pull and offline CLI checks passed. No Python package is published.

## Task tracking

| Task | Status | Remaining evidence |
|---|---|---|
| T00 Baseline | Author reference complete; engine measured | GPU parity failed; isolated Docker CPU/GPU baseline passed |
| T01 Contracts | CPU complete | 168 CPU tests across current implementation |
| T02 Artifacts | Cold online preparation and real container offline inference passed | Broader release/fault matrix remains |
| T03 Adapter | Encoding/readout and fixed FP32 LoRA implemented | Numerical alignment still fails; not certified |
| T04 Runtime | Real GPU/HTTP inference works | Resolve 38/113 latest pretokenized and 36/110 HTTP parity failures |
| T05 API/lifecycle | Native and SDK subset implemented | Broader release/fault/soak coverage remains |
| T06 CLI/SDK | CLI/SDK and one-command cold container quickstart passed | Separate clean-host installation not run |
| T07 Docker | Image build, cold online and offline GPU inference passed | Docker Hub development preview published and verified; numerical certification failed |
| T08 Compose/offline/rollback | Compose real GPU and container offline inference passed | Image-version rollback drill not run |
| T09 Multi-model extension | Seven pinned adapters/profiles integrated; JPT-0.8B GPU/container paths work | Larger model checks in progress; JPT probability parity fails |
| T10 Observability/benchmark | Metrics and short sequential benchmark measured | Capacity study and 24/72-hour soak not run |
| T11 CI/release | CPU workflow written; local checks pass | Remote CPU CI passed; certified image promotion/release absent |
| T12 Open-source acceptance | Incomplete | Parity, rollback, clean-host and soak acceptance |

## Session evidence — 2026-10-12

### T00: actual upstream discovery, not certification

Read AGENTS, HANDOFF, STATUS, design, full task plan and local server notes.
Starting Git worktree was clean. Host GPU observation: RTX A6000, 49140 MiB,
11 MiB used, 0% utilization, driver 580.95.05. These are observations, not a
reservation or supported profile. Public sanitized report:
`validation/reports/t00-environment.json`; private context: `.local/SERVER.md`.

Exact probes:

```bash
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu,driver_version --format=csv
docker info --format '{{range $k,$v := .Runtimes}}{{$k}} {{end}}'
docker run --rm --pull=never --name jevserve-t00-gpu-probe \
  --gpus device=0 --network none --read-only --entrypoint nvidia-smi \
  python:3.10-slim --query-gpu=name,memory.total,driver_version --format=csv
docker run --rm --pull=never --name jevserve-t00-cpu-probe \
  --network none --read-only --entrypoint python python:3.10-slim --version
docker manifest inspect vllm/vllm-openai:v0.29.0
```

Both container probes failed before entrypoint execution with Docker TTRPC
`unsupported protocol`. The CPU probe shows this is not solely an NVIDIA
passthrough problem. Existing shared services, daemon, drivers and runtime
settings were not changed. The cached Python 3.10 image was only a daemon probe,
not a development or inference environment. Registry manifest inspection passed. Later,
`uv pip compile --python-version 3.12 .local/upstream/vllm-jev/pyproject.toml`
resolved 201 candidate engine dependencies successfully; this is not an engine
installation or inference result.

Verified immutable sources and inspected licenses/model/tokenizer/readout code:

- Open-Jev-2B: `0c7aa498b1627be8da4acf34c863ff0ee0a92785`.
- Qwen3.5-2B base/tokenizer: `15852e8c16360a2fea060d615a32b45270f8a8fc`.
- Author source: `bd4118882f733574a3250a4b65fe4d884130c08b`.
- vllm-jev: `eae6b2d6dd06fb014118c99f660261cf4ecf04f9`, version 0.3.6.
- vLLM: `98dff2a81d747d1dba01a47f939f48c3526d4206`, version 0.29.0.
- Official linux/amd64 image manifest:
  `sha256:082ca6f035279109041ffd3fe0695cb568b29bc580b35c4f297a66a08b216c1b`.

Sources, byte inventories and semantics are recorded in `docs/models.md`,
`models/locks/open-jev-2b.json` and `compat/engine-lock.json`. Checkpoint files
were downloaded to staging, verified against pinned Hub Git/LFS digests and
atomically published locally. Base and dependency acquisition outcomes are
recorded below as they finish. No weights are tracked. The candidate engine's
Python/Torch/CUDA values are declarations, not observations from a running image.

Local uv 0.13.0 installed Python 3.12.15 under `.local/python`; `.venv` is the
CPU environment. `.local/reference-venv` is isolated because author and engine
Transformers/PEFT requirements differ. Hash-pinned reference requirements live
in `compat/reference-requirements.txt`. No global Python/CUDA package changes.

`tests/contract/test_release_lock.py` first failed collection because the checker
was absent, then passed 6 tests after implementation. Release validation remains
intentionally failing for pending certification. The reference runner now defines
113 unique synthetic requests covering three types, Chinese, structured state,
null descriptions, empty-state author behavior, one candidate and an exact tie.
It requires real CUDA, records tokens/masks/logits/probabilities and refuses
modified source or artifact bytes. The later real GPU run is recorded below.
Parity tolerances were frozen before measurement in `compat/parity-profile.json`.

### T01: installable CPU contracts

Added pinned `pyproject.toml` / `uv.lock`, the `jevserve` package, strict request,
response/error/capability models and validated administrator settings. Tests
cover question count, stable order/ties, empty/nonfinite input, duplicate JSON,
invalid fields, ordinal expectations, model work budgets and CPU-only imports.
Request/config tests first failed collection, then passed after implementation.

```bash
UV_CACHE_DIR="$PWD/.local/uv-cache" UV_PYTHON_INSTALL_DIR="$PWD/.local/python" .local/bin/uv sync --locked
.venv/bin/pytest tests/contract/test_requests.py tests/unit/test_config.py -q
.venv/bin/ruff check .
.venv/bin/mypy
UV_CACHE_DIR="$PWD/.local/uv-cache" .local/bin/uv build
```

Wheel and sdist built. A fresh `.local/wheel-test` Python 3.12 environment installed
the wheel; `python -I` imported contracts/artifact preparation and verified that
Torch, Transformers and vLLM were absent from loaded modules. Archive inspection
found no `.local`, secrets or model weights. No CLI is claimed.

### T02: safe preparation library

Added immutable source/file manifests, registry lookup, full cache verification,
Linux advisory locking, staging and atomic publication. Offline mode never
creates a cache or reads a token when missing. Online download is HTTPS only,
bounded, and strips Authorization on redirects. Unknown models fail closed.
The registry is empty until a model is certified. Missing/corrupt/unlisted files,
symlinks and path traversal fail; corrupt final packages are not auto-replaced.

Preparation/manifest tests first failed collection, then passed after the real
implementation. Meaningful failure tests include concurrent preparation, lock
expiry, interrupted download, disk-full sanitization, corrupted byte hashes,
missing readiness metadata, unowned staging preservation and token handling.

```bash
.venv/bin/pytest tests/unit/test_prepare.py tests/unit/test_manifest.py -q
.venv/bin/pytest -q
.venv/bin/ruff check .
.venv/bin/mypy
```

A real HTTPS preparation of all 19 checkpoint-repository files also passed,
followed by offline cache re-verification of those real bytes. Report:
`validation/reports/open-jev-prepare.json`. This is artifact verification, not
model inference.

Most recent suite: **116 passed**. Ruff clean; mypy checks all 15 package source
files without issues. Self-review found and fixed acceptance of undeclared cache
files and added metadata size checks. Independent agent review was not used
because active session instructions disallow unrequested delegation.

### T03: CPU adapter and real-tokenizer encoding parity

Implemented immutable encoded inputs/readout groups and scalar-output decoding.
Preserves independent candidates, question/label order, zero-based Score levels,
false/true Noul mapping, checkpoint temperature and no truncation. Reused prompt
semantics carry the author MIT notice in `licenses/Open-Jev-MIT.txt` and NOTICE.
No vLLM model registration or engine bridge is implemented yet.

The adapter tests first failed collection, then passed. An isolated tokenizer
CPU environment used Transformers 5.10.2, tokenizers 0.22.2 and Jinja2 3.1.6.
The first encoding run diagnosed missing Jinja2; after installing that exact
version, **110 real-tokenizer cases matched the author implementation**:

```bash
PYTHONPATH=src HF_HUB_OFFLINE=1 .local/tokenizer-venv/bin/python -m validation.encoding \
  --upstream .local/upstream/Open-Jev --tokenizer .local/models/base.staging \
  --output validation/reports/open-jev-encoding.json
```

After base publication, use `.local/models/base` for `--tokenizer`. The report
records complete golden inputs/readout positions and is hashed in the model
lock. Unit tests replay those recorded token rows and verify prompts/metadata;
that replay is explicitly a test double, not a production tokenizer or inference
fallback. Author GPU inference has now passed; replaying all 113 recorded GPU logit rows
through the adapter also matches author probabilities, choices and ordinal
scores (CPU regression tolerance 1e-12). This is not a vLLM engine comparison.

## Unrun gates

vLLM parity, container GPU passthrough, Docker build,
docker-run/Compose inference, service offline inference, cancellation under GPU
load, official SDK compatibility, performance/soak and clean-host installation
are not passed. No fake runtime or substitute model was introduced.

## Real GPU reference completion — 2026-10-12

All base files completed SHA-256 verification and were atomically published to
`.local/models/base`. The 56-package reference environment installed from
hash-verified wheels; `uv pip check --python .local/reference-venv/bin/python`
passed. Large PyPI transfers were slow; identical locked wheels were retrieved
through a public mirror and verified against the original PyPI/requirements
hashes. No dependency version or model precision was changed to accelerate this.

The exact reference command below completed successfully, offline, on RTX A6000:

```bash
.local/reference-venv/bin/python -m validation.reference \
  --upstream .local/upstream/Open-Jev \
  --checkpoint .local/models/checkpoint --base .local/models/base \
  --cache .local/reference-cache \
  --output validation/reports/open-jev-reference.json
```

Result: **113 real GPU cases, zero failures**. Report includes input token IDs,
attention masks, raw logits, probabilities and typed author responses. Exact-tie
case observed `[0.5, 0.5]`. Runtime observed: Python 3.12.15, Torch 2.12.0+cu130,
CUDA 13.0, Transformers 5.10.2, PEFT 0.19.1, Accelerate 1.13.0. Backbone parameters
include BF16 and FP32 (LoRA); head is FP32. The author's ordinary Torch path was
used; optional FLA/causal-conv1d fast kernels were not installed. Prefix cache was
off, and saved temperature was retained. This is a reference baseline, not a
performance or calibrated-accuracy claim.

The report and reference-requirements SHA-256 are bound in the inventory/report.
Only `gpu_reference` changes to passed; engine, model certification and other
gates remain pending. GPU returned to 11 MiB used after process exit. The whole
CPU suite now passes **116 tests**, including replay of these actual GPU logits.
`python -m validation.locks` exits nonzero with `baseline is not certified`, as
required. First implementation commit: `4908d0d`; reference evidence is committed
separately in the following commit (see `git log --oneline`).

## Development runtime milestone — 2026-10-12

The engine environment installed the exact `compat/runtime-requirements.txt`
(200 packages, wheel hashes). Observed: Python 3.12.15, vLLM 0.29.0, Torch 2.13.0,
CUDA 13.0, Transformers 5.10.4 and PEFT 0.20.0. Source/base hashes remain unchanged.
A safe combined source package is compiled atomically into a text-only checkpoint:
BF16 base tensors are unchanged; the original FP32 LoRA and head are retained.

Numerical experiments (fixed atol=0.001, rtol=0.01; never relaxed):

- BF16 merged LoRA: 35/113 failures, maximum probability error 0.0160318306.
  Rejected; report `validation/reports/bf16-merge-rejected.json`.
- Native vLLM dynamic FP32 LoRA: fails its FP16/BF16-only expansion-kernel assertion.
  No precision fallback. A bounded fixed FP32 LoRA module supplies the residuals.
- Fixed FP32 LoRA, native kernels: 33/113 pretokenized failures, maximum error
  0.0217803367, no label flips. `vllm-unmerged-initial.json`.
- Reference-style residual normalization experiment: 35/113 failures, maximum
  error 0.0195144937, no flips. `vllm-unmerged-norm.json`; experiment reverted.
- Final standalone reproduction with `validation.pooling`: **38/113 failures**,
  maximum error 0.0217803367, no flips. `vllm-pooling-parity.json`. Differences
  from the initial 33/113 run are retained as evidence of numerical variability;
  identical results across batch shapes/runs are not claimed.
- Actual HTTP: all 110 native requests completed, **36/110 fail parity**, same
  maximum error, no flips. Three empty-state reference cases are intentionally
  rejected by the native contract. `native-http-parity.json`.
- Serving-environment tokenizer: **110/110** author encoding comparisons pass.
  `engine-tokenizer-parity.json`. The tokenizer warning is not automatically
  “fixed,” because that could change IDs; actual IDs were compared instead.

The code has no fake production runtime or merged/low-precision fallback.
`--allow-uncertified` is required to serve this development profile. Its purpose
is to expose the incomplete certification, never to weaken the release gate.
The certification registry remains empty; `python -m validation.locks` still
fails with `baseline is not certified`.

Actual service observations:

- Started offline from verified cache; engine warmed before readiness. Initial
  kernel compilation took about 94 seconds on this host; startup is not an SLA.
- `jevserve check` returned real Choice/Noul/Score. Its refund sample selected
  billing with probability 0.9488854504; this is a model output, not an accuracy claim.
- Official `typesafe-sdk==0.7.4` made a real network call for all three types.
  Only the text-question subset is promised; see `compat/README.md`.
- Sixteen concurrent requests: 16 HTTP 200 responses. A 1 ms deadline returned
  504. Client cancellation returned active/waiting gauges to zero, followed by
  successful inference and readiness. `http-resilience.json`.
- Twenty sequential six-candidate/three-question requests: p50 0.141754 s,
  p95 0.170816 s, total 2.799689 s. `native-http-benchmark.json`. This short,
  synthetic, warmed workload is not a capacity/accuracy/soak claim.
- SIGTERM on our API process closed its engine and released GPU memory. SIGTERM
  on **only our worker** caused API nonzero exit (1); GPU returned to idle.
  `runtime-environment.json`. No other process or shared service was changed.
- `docker compose config --quiet` passed. Dockerfile build, `docker run` inference,
  Compose inference and container network isolation are **unrun**, not passed.
  The shared daemon's earlier CPU/GPU TTRPC failure remains a prerequisite blocker.

Verification: 130 CPU tests passed, Ruff and strict Mypy (28 CPU-visible source
modules) passed, wheel/sdist build passed. The GPU model module uses a dynamic
vLLM base class and is excluded from Mypy; real loading and inference validate
that boundary. Manual review covered cache integrity, cancellation, secret/body
logging, checkpoint dtypes and release claims. Independent agent review was not
available under the session's no-delegation instruction. CI workflow is written
but has not run remotely.

## Reproduction and exact next action

CPU:

```bash
uv sync --locked
uv run pytest -q
uv run ruff check .
uv run mypy
uv run python -m build --no-isolation
```

Real serving on this development host (check GPU occupancy first):

```bash
.local/engine-venv/bin/jevserve serve --allow-uncertified --offline \
  --cache-dir .local/runtime-cache --port 18791
.venv/bin/jevserve check --url http://127.0.0.1:18791
.venv/bin/python -m validation.http --url http://127.0.0.1:18791 \
  --output .local/http-parity.json
```

The HTTP parity command exits **1** for the measured tolerance failures. To
reproduce pretokenized pooling, run `python -m validation.pooling --help` in the
engine environment and supply the verified compiled path and cache directory.
Author reference reproduction remains the command in the preceding section.
For another host, use the README's pinned Python 3.12 installation and prepare
commands; the local cache paths are not committed artifacts.

**Next engineering action:** isolate numerical drift at per-layer/sequence-batch
boundaries between the author's Torch path and vLLM kernels while preserving
checkpoint precision and the existing tolerance. Do not certify on label agreement
alone. Independently, the host administrator must restore Docker container creation;
then run the existing CPU/GPU probes, build the pinned Dockerfile, and verify both
container paths, offline startup and rollback. No daemon restart is authorized
or performed here. Alpha/RC/stable gates remain open.

No push or publication was performed. Weights, caches, credentials and private
host notes remain ignored. Commits use the user's existing Git identity.

## README, identity and image preparation — 2026-10-12

Replaced both README entry points with a shared logo, plain description, three
concrete decision examples, one-command startup, HTTP/Python examples and links
to detailed evidence. Logo source/prompt: `docs/assets/brand/README.md`; selected
asset: `docs/assets/brand/jevserve-logo.png`. Deployment guide:
`docs/quickstart.md`. The image is explicitly a development preview; numerical
certification and stable/alpha promotion remain closed.

`scripts/quickstart.sh` builds or selects an image, preserves existing containers,
uses a project cache volume, polls readiness and performs one real three-type
decision request. `check --wait-seconds` bounds polling to 1800 seconds; decision
requests are not retried. Compose accepts `JEVSERVE_IMAGE`.

Observed checks: `pytest -q` **134 passed**, `ruff check .` passed, strict Mypy
passed for 28 modules, wheel/sdist build passed; `bash -n` and Compose config
validation passed. Local README/deployment links resolve.

A separate rootless Docker daemon with project-local state successfully ran
Python 3.12.15, native NVIDIA CDI and legacy `--gpus device=0` probes on the
RTX A6000. The shared daemon's running binary/on-disk shim mismatch was observed
read-only; it was not repaired or restarted. This supersedes the prerequisite
blocker above for this development session. Private commands/logs and state:
`.local/SERVER.md`, `.local/evidence/`, `.local/docker/`.

The fixed image builds and passes `pip check`, non-root CLI import, real
Choice/Noul/Score inference with `--network none --read-only`, and graceful
SIGTERM (exit 0, GPU returned to idle). Missing-cache offline preparation fails
with ArtifactMissing (exit 1), as required. Cold online startup from a completely
empty named volume passed all three decision types in 703.8 seconds, including
downloads and first warmup (one observation, not a benchmark/SLA). Both README
curl/Python examples passed. Compose reused this cache through a volume-only
override, passed all three types and readiness, then stopped with exit 0 and
released GPU memory. Its read-only root and security settings were preserved. Docker Hub publication was requested by the user, but
there is no configured registry login or confirmed destination namespace. The
namespace was requested; no credentials are written to tracked files.

The first actual build exposed two base-image extras incompatible with the
frozen lock (LMCache and the OpenTelemetry Prometheus exporter). They are now
removed; `pip check` is a required build step. The first GPU container exposed
FlashInfer trying to create `/.cache`. A UID 10001 service account and explicit
writable cache paths fix this while retaining a read-only root filesystem.
See D022/D023. These were real failures, not skipped checks.


### Final artifact and reproduction

Local tags `jevserve:dev` and `jevserve:0.1.0-dev0` identify the tested image:
`sha256:f27c15c1da40d0c02139672a426d41d5370e6ea061d209b1d3fc5aef53b73396`.
This is a local image identifier, **not a published Docker Hub pull reference**.
Image Python: 3.12.3; non-root UID/GID: 10001. Sanitized evidence:
[`validation/reports/container-smoke.json`](../../validation/reports/container-smoke.json).
The image uses the frozen runtime lock and pinned upstream image digest.

On a configured Docker host, reproduce with:

```bash
docker build -t jevserve:dev .
./scripts/quickstart.sh
curl -fsS http://127.0.0.1:8000/v1/decisions \
  -H 'Content-Type: application/json' --data-binary @examples/decision.json
docker stop --timeout 65 jevserve
docker compose up --no-build -d
docker compose exec jevserve jevserve check --wait-seconds 1800
docker compose down
```

This host uses an isolated daemon, so prefix Docker/helper commands with
`DOCKER_HOST="unix://$PWD/.local/docker/run/docker.sock"`. Actual smoke names
were `jevserve-offline-check`, `jevserve-quickstart-check`, and Compose project
`jevserve-compose-check`. The Compose override in `.local/compose-validation.yaml`
only maps its cache to the named volume `jevserve-quickstart-check-cache`.
The offline test used `--network none --read-only --tmpfs /tmp:size=2g,mode=1777`,
a verified source cache, and `serve --allow-uncertified --offline`; inference was
checked through `docker exec`. No host model path was used by the cold startup.

All validation services are stopped and the GPU returned to 11 MiB idle usage.
The shared daemon still has its original 40 running containers. Verified caches
and the project-local daemon remain available; no shared daemon restart, global
prune, driver change or unrelated-container modification occurred. Failed test
logs remain private; the failed probe container and intermediate candidate tag
were removed by exact project names. Review was local/manual; independent agent
review was not allowed by the active session instructions.

**Publication handoff at the end of that session (superseded below):** obtain the user's Docker Hub namespace and
local registry login, tag this already-tested image as that repository's
`0.1.0-dev0`, push, verify its remote digest, then update both README files with
the real pull command. Publication is already authorized by the user's request;
missing destination/authentication is the remaining publication blocker. No
credentials are requested in chat or committed. There is still no Git remote,
Git push, PyPI publication, alpha certification, rollback drill or soak claim.
Numerical drift investigation remains the next model-engineering task; existing
failed tolerance reports are unchanged.

Container outcomes were synchronized into engine/model metadata, with the
packaged model lock kept identical to its repository copy. The source-manifest
identity and all model/runtime bytes remain unchanged. The cold-start report
records the preceding runtime-identical image ID; the final metadata-refreshed
image was rechecked with the downloaded cache. Release validation still exits 1
with `baseline is not certified`, as required by failed numerical parity.


## Public development preview — 2026-10-12

The user authorized publication to their existing GitHub repository and logged
in to Docker Hub. The verified Docker identity is `anonwx`; GitHub SSH identity
is `anonNo2`. No credentials are in Git or command arguments.

- Source: https://github.com/anonNo2/jevserve, branch `main`. The initial push of
  `837204a686e74b88c37e2e98598679a2d965b957` was verified through the public GitHub
  API. Its [remote CPU CI run](https://github.com/anonNo2/jevserve/actions/runs/38246635531)
  completed successfully (environment installation, CPU checks and packaging).
- Docker Hub: https://hub.docker.com/r/anonwx/jevserve, tag `0.1.0-dev0`, Linux amd64.
  Registry index digest: `sha256:f27c15c1da40d0c02139672a426d41d5370e6ea061d209b1d3fc5aef53b73396`.
  This is the already-tested image from source commit `837204a`; publishing only
  added a registry tag. Runtime/model bytes were not rebuilt or changed.
- `docker push` exited 0. Docker Hub's anonymous tag API reported `active` and the
  same digest. An empty project-local Docker client configuration successfully
  inspected the manifest and pulled by digest. Existing local layers were reused;
  this does not constitute clean-host installation. A network-disabled, read-only
  container using the published digest listed the pinned Open-Jev-2B model and
  `uncertified_development` status, exit 0. The additional `unknown/unknown` index
  entry is build attestation, not a second supported CPU architecture.
- Both README files now link [supported models and tests](../models.md), with
  real Docker Hub, helper and Compose startup commands. The model matrix records
  only Open-Jev-2B as implemented/tested and links successful and failed evidence.
- Local review: `git diff --check`, documentation local-link checks, helper
  `bash -n`, published-image `docker compose config`, 134 CPU tests, Ruff, mypy
  (28 source files), and wheel/sdist build all passed. Existing Git history was
  checked for tracked credentials, weights and private host paths before push.
  Independent agent review was not permitted in this session.

Publication/reproduction commands (this host uses the project-local daemon
prefix documented above; an ordinary configured Docker host needs no prefix):

```bash
docker tag jevserve:0.1.0-dev0 anonwx/jevserve:0.1.0-dev0
docker push anonwx/jevserve:0.1.0-dev0
docker manifest inspect anonwx/jevserve:0.1.0-dev0
docker pull anonwx/jevserve@sha256:f27c15c1da40d0c02139672a426d41d5370e6ea061d209b1d3fc5aef53b73396
docker run --rm --network none --read-only \
  anonwx/jevserve@sha256:f27c15c1da40d0c02139672a426d41d5370e6ea061d209b1d3fc5aef53b73396 models list
git push -u origin main
```

The initial source push and remote CI are complete; these documentation/report
updates are the publication follow-up commit. No `latest`/stable tag or alpha
certification is claimed. Numerical parity, clean-host installation, image-version
rollback and 24/72-hour soak gates remain incomplete. GPU tests were not rerun
for this documentation-only publication: the registry digest exactly matches
the previously tested image in `validation/reports/container-smoke.json`.

**Exact next engineering action:** reproduce and investigate the frozen
Open-Jev-2B probability failures using the reference, pooling and HTTP reports;
keep the existing tolerances and pinned weights. Do not promote this image to
alpha/certified status until those failures and remaining release gates are resolved.


## Multi-model integration in progress — 2026-10-12

The user requested one reusable image with mounted weights and as many deployable
Jev checkpoints as possible on the current 48 GiB GPU, specifically mentioning
JEV-27B-VL, Xor 1.2 and JPT-9B. D024/D025 record this scope extension.

- Implemented, pending GPU validation: model catalog selection, read-only
  `--model-dir` verification, per-model HTTP metadata, model-aware helper and
  Compose mount override; JPT token readout, Open-Jev-9B inventory and NeoHorse
  explicit Torch pointer profile. No new model has passed a GPU gate yet.
- Pinned JPT-0.8B/4B/9B, Open-Jev-9B plus its base, and NeoHorse-Jev-4B inventories.
  JPT weights are CC BY-NC 4.0; they are not bundled into the Apache-2.0 image.
- JPT author/real-tokenizer comparison: 9 synthetic requests, 10 questions, all
  255 label IDs matched (`validation/reports/jpt-encoding.json`). 150 CPU tests
  passed before adding the pointer worker. Current image candidate builds as
  `jevserve:multimodel-dev`; it has not been published or certified.
- Sequential Hub transfers encountered interrupted connections. Validation-only
  acquisition now uses explicitly requested bounded range retries, validates
  the entire pinned SHA-256, and atomically publishes through the same preparer.
  Partial bytes remain staging/private; serving never retries inference.
- JEV-27B-VL's original weights occupy approximately 56.05 GB; Xor 1.2 approximately
  70.24 GB. Neither fits entirely in this GPU at the released BF16 precision.
  No quantized substitute or success is claimed. Their native protocols are being
  inspected; availability is not equivalent to integration.

Immediate next action: finish hash verification, run each model's real reference
then the common container's HTTP/native SDK/Compose/offline checks sequentially
on the GPU, record parity separately, and publish a new development image only
after its actual test matrix is complete.

### Multi-model verification checkpoint (in progress)

- CPU: `pytest -q`: **167 passed**; `ruff check .` and `mypy` passed.
  Protocol tests still import no GPU packages. Added selected-model routing,
  pointer cancellation and candidate-mask boundary regression coverage.
- Built the common `0.1.0.dev1` candidate image without model weights. New
  adapters/inventories: Open-Jev-9B, JPT-0.8B/4B/9B, NeoHorse-Jev-4B and Kev-4B.
  No new image has been published at this checkpoint.
- JPT-0.8B: pinned author implementation ran all 10 synthetic reference requests.
  Common-image HTTP ran all 10, including 255-option Choice/Score, and rejected
  an unconfigured SystemOne model. **5/10 requests failed** the frozen probability
  tolerance (`atol + rtol * abs(reference)`); max error 0.0273194739, no Choice
  label flips. This is not numerical certification.
- An initial targeted-token vLLM run failed at 255 options: pinned vLLM permits
  at most 128 `logprob_token_ids`. Replaced this with an explicit allowed-candidate
  mask and N processed logprobs, retaining one forward pass and applying the
  checkpoint temperature exactly once (D026). The corrected 255-option request
  executed successfully; the probability mismatch above remains recorded.
- Mounted JPT-0.8B Compose: actual inference passed with `network_mode: none`,
  a read-only weight mount/root filesystem, writable runtime cache and non-root
  service. Graceful stop returned GPU use to idle. Open-Jev-2B also passed a
  common-image offline regression earlier in this session.
- Larger inventories are downloading to ignored project-local staging. HTTP/2
  resets interrupted initial transfers; explicit resumable validation transfers
  retain chunks, validate byte ranges, then verify the complete pinned SHA-256
  before atomic publication. Download completion is not a GPU gate.
- Exact next action: finish verified downloads, run each author's GPU reference
  and then the common image's HTTP/native-client/official-SDK checks sequentially.
  Update the model matrix with observed outcomes, then publish the common dev1
  image and source to the already authorized Docker Hub/GitHub destinations.
  JEV-27B-VL/Xor 1.2 remain investigated candidates, not supported or tested
  profiles; their raw BF16 weights exceed this GPU's memory.

- Subsequent common-image JPT-0.8B rerun: all 10 HTTP cases, native client and
  `typesafe-sdk==0.7.4` passed execution. The latest reference/container pairing
  failed **6/10** probability comparisons with no Choice label flips; the report
  records the latest run, while the earlier 5/10 observation above remains
  historical. GPU reservation after this run was 24,688 MiB at the default 0.5
  budget, not a minimum memory requirement. Latest CPU suite: **168 passed**.
- Source-license audit corrected NeoHorse/Kev attribution to the actual pinned
  Apache-2.0 LICENSE and retained the vendored NOTICE. JPT protocol code remains
  MIT; model weight licenses are separate.

### Additional GPU evidence and precision audit (in progress)

- NeoHorse-Jev-4B: author GPU reference and common-container HTTP executed all
  10 synthetic cases, including 255-option Choice/Score. All probability checks
  passed; maximum absolute error 6.326030721481857e-8. Native client and official
  `typesafe-sdk==0.7.4` passed. Observed allocation after tests: 10,050 MiB.
- The same candidate image ran both JPT-0.8B and NeoHorse via Compose with network
  disabled, read-only mounted weights/root, UID 10001, and real Choice/Noul/Score.
  See `validation/reports/multimodel-compose.json` for exact image identity.
- Kev's first reference/container comparison passed, but a memory audit exposed
  an ignored FP32 loader argument: all 4,205,751,296 backbone parameters were
  BF16. Initial reports were retained as `kev-4b-initial-bf16-*` and explicitly
  invalidated for the declared precision profile. D028 records the correction.
  A CPU load with the explicit text config found all backbone parameters FP32,
  with no missing or mismatched checkpoint keys. Corrected GPU validation is
  underway; no FP32 success is claimed until it completes.
- Open-Jev-9B has now completed full pinned SHA-256 verification; GPU checks are
  queued behind Kev. JPT-4B/9B still download in resumable, verified staging.
- Reproduction: `python -m validation.model_matrix reference --model MODEL_ID
  --model-dir VERIFIED_DIRECTORY --upstream PINNED_SOURCE --output REPORT`, then
  `python -m validation.model_matrix http --model MODEL_ID --reference REPORT
  --url http://127.0.0.1:8000 --output HTTP_REPORT` against the selected container.

- Corrected Kev FP32 GPU reference and container rerun completed: **10/10**
  probability comparisons passed, no Choice flips; native client and official
  SDK passed. All reference backbone/head parameters were observed FP32; the
  serving worker also asserts backbone precision. Observed GPU allocation:
  **18,790 MiB**. Image identity and exact results are in `kev-4b-runtime.json`
  and `kev-4b-container.json`. This supersedes the invalid initial BF16 profile.
  CPU 168 tests, Ruff and mypy remain green. Next GPU task: Open-Jev-9B.

- Open-Jev-9B reference: **113 real GPU cases executed, zero runtime failures**
  using the original locked reference environment. The harness now reads its
  pinned inventory directly, keeping production Pydantic dependencies out of
  that independent environment. Container parity is still running.
- JPT-4B source download and complete hash verification passed. JPT-9B has
  verified its first two weight shards; remaining shards are still staging.
- GitHub main now includes `01ee8c5` (Kev precision correction and reports).
  Dev1 Docker publication remains pending final common-image regression.

- Open-Jev-9B common-container gate completed: **110 native HTTP cases** and
  official SDK executed successfully. Frozen probability tolerance failed
  **27/110**, maximum absolute error **0.019531584454720696**, zero label flips.
  Observed allocation after tests: **23,914 MiB** at the explicit default 0.5
  GPU budget. The reference had 113 cases; three empty-state cases are rejected
  by the native contract. Reports: `open-jev-9b-{reference,container,runtime}.json`.
  Numerical certification remains failed, separately from successful execution.

- Final candidate image `sha256:4b758ad1000f4d38501fd83f0a9c2b32f0cb91968d427b170a2c2b0d9a613aff`
  passed sequential mounted offline Compose inference for both JPT-0.8B and
  NeoHorse-Jev-4B. Network disabled, UID 10001, read-only root/weight mount and
  all three question types verified; `multimodel-compose.json` now records this
  final artifact. Open-Jev-2B regression and real pointer deadline gate are next.

- Final Open-Jev-2B common-image regression passed real Choice/Noul/Score with
  network disabled and read-only root, reusing the previously verified cache.
  Evidence: `validation/reports/open-jev-2b-common-image.json`.
- Real Kev FP32 active deadline test passed: a 255-option request with 100 ms
  deadline returned **504 in 0.3274 s** including cancellation cleanup; the
  container exited and GPU allocation returned to **11 MiB**, with restart
  policy `no`. Evidence: `validation/reports/pointer-deadline.json`; reproduce
  with `python -m validation.pointer_deadline --url DEDICATED_SERVICE
  --output REPORT` only against a disposable pointer-model validation instance.
  CPU cancellation tests and real GPU lifecycle evidence are separate.
- Exact next action: JPT-4B author/container matrix is running; finish JPT-9B
  verified download and its matrix, then publish the already verified common
  artifact and synchronize bilingual README, model results and registry digest.
