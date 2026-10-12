# Development status

## Current milestone

Implementation started on 2026-10-12. CPU contracts, artifact preparation and the Open-Jev adapter are
implemented; real-tokenizer encoding parity is verified. No inference server, CLI, Dockerfile or Compose service
exists yet. No model is certified and no release has been published.

## Task tracking

| Task | Status | Remaining evidence |
|---|---|---|
| Bootstrap | Complete | Historical documentation baseline |
| T00 Baseline | Reference complete; engine/container baseline incomplete | Author GPU 113/113 passed; candidate engine runtime and container startup still unverified |
| T01 Contracts | CPU complete | 116 total current CPU tests include T00/T02/T03; wheel build/import verified |
| T02 Artifacts | CPU implementation complete | Shared preparer exists; actual serve startup integration belongs to T06 |
| T03 Adapter | CPU encoding/readout implemented | Author GPU-logit decoding matches; vLLM registration/runtime parity remains unverified |
| T04 Runtime | Not started | Real vLLM GPU inference |
| T05 API/lifecycle | Not started | Protocol, overload, cancellation and failure tests |
| T06 CLI/SDK | Not started | Real end-to-end check and prepare/serve integration |
| T07 Docker | Blocked prerequisite; not started | Container daemon can create project containers, then real inference |
| T08 Compose/offline/rollback | Not started | Real deployment and recovery exercises |
| T09 Second model | Optional, not started | Independent reference/GPU certification |
| T10 Observability/benchmark | Not started | Measured reports and milestone-specific soak |
| T11 CI/release | Not started | Exact tested image promotion |
| T12 Open-source acceptance | Not started | Clean-host quickstart and complete alpha gates |

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

## Next action and reproduction

1. Resolve the remaining T00 engine/container baseline. Host administrator must
   diagnose Docker's TTRPC startup failure; after recovery rerun the two probes
   above, then inspect/start the pinned official engine image. No daemon restart
   or driver/runtime change was performed in this session.
2. Continue T03's bounded vLLM model registration and T04 runtime bridge against
   the committed author token/logit fixtures. Preserve LoRA, scalar head and
   temperature semantics. Do not mark `gpu_parity` passed on reference replay.
3. CPU reproduction: `uv sync --locked && uv run pytest -q`; then
   `uv run ruff check .`, `uv run mypy`, `uv build`.
4. GPU reference reproduction: install `compat/reference-requirements.txt` into
   the separate Python 3.12 reference environment with `--require-hashes`,
   recheck GPU occupancy, and execute the command above. Sources can be prepared
   with `uv run python -m validation.acquire --cache .local/baseline`; use the
   printed checkpoint/base paths. `--offline` verifies existing safe packages.

No remote push or publication was performed. No model/weight/cache/private host
file is committed. Commits use the user's existing Git identity.
