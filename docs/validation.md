# Reproduce model validation

`python -m validation.run` coordinates one pinned model at a time: verify local
weights, run the author's GPU reference, release that process, start one immutable
container, wait for readiness, run native HTTP and TypeSafe SDK checks, and remove
only the uniquely named container and cache volume it created. Run it from a
source checkout with the project's Python 3.12 development environment installed
(`uv sync --locked`). No GPU packages are imported by the coordinator or CPU tests.

This is a **text-only synthetic execution and probability comparison**, not an
accuracy, vision, performance, Compose, or release certification gate. Current
Open-Jev/JPT reports fail probability parity. NeoHorse/Kev small-suite passes do
not establish general model accuracy. Frozen tolerances in
`compat/parity-profile.json` are used without adjustment.

## Inputs and preparation

Use a Linux NVIDIA host, Docker with NVIDIA support, and GPU 0 free of compute
processes. The reference selects the physical GPU 0 UUID reported by NVIDIA,
overriding inherited CUDA device visibility so both stages use the same GPU.
The runner checks GPU occupancy before both GPU stages and takes a
checkout-local advisory lock. Other applications do not honor that lock; reserve
the GPU operationally. It never stops other users' processes. The Docker socket
must be an explicit **local** `unix:///...` socket; remote daemons cannot use these
local model mounts. Ensure image UID 10001 can read the model directory and its
parents; the runner does not change model permissions.

Prepare weights with `jevserve prepare --model MODEL_ID --cache-dir /path/to/cache
` (see `jevserve prepare --help` and [models](models.md)). Supply
the resulting verified source directory as `--model-dir`, not its parent cache.
Open-Jev/Kev source directories contain `base/` and `checkpoint/`; other models
have flat inventories. Full sizes/hashes and revisions are verified again offline.
The input must contain exactly the catalog's files (plus optional preparation
metadata), not arbitrary files from a Hub snapshot.

Pull the exact image in advance using the same Docker host:

```bash
export VALIDATION_DOCKER_HOST=unix:///var/run/docker.sock
export VALIDATION_IMAGE=anonwx/jevserve:0.1.0-dev1@sha256:4b758ad1000f4d38501fd83f0a9c2b32f0cb91968d427b170a2c2b0d9a613aff
docker --host "$VALIDATION_DOCKER_HOST" pull "$VALIDATION_IMAGE"
```

The runner uses `--pull=never` and offline model loading. HTTP uses a loopback port;
the container's network namespace is not disabled. This runner does not test
network isolation. Each run gets a fresh cache volume, so startup includes model
compilation. No model download or inference retry occurs inside the runner.
Readiness polls health, then `jevserve check` sends one inference request.
Unset all `JEV_*` environment overrides first.

## Author reference environments

Use a separate Python 3.12 virtual environment for the reference. Open-Jev uses
`compat/reference-requirements.txt` (Torch 2.12.0, Transformers 5.10.2, PEFT 0.19.1).
The reference verifies these versions. Other families use
`compat/runtime-requirements.txt` (Torch 2.13.0, Transformers 5.10.4, PEFT 0.20.0);
their reports record observed versions. They execute author source under JevServe
runtime packages, **not the author's original benchmark environment**. Install
with hash checking, for example:

```bash
uv venv --python 3.12 /path/to/reference-env
uv pip install --python /path/to/reference-env/bin/python --require-hashes \
  -r compat/reference-requirements.txt
```

Use runtime requirements instead for JPT/NeoHorse/Kev. Run the coordinator itself
with `.venv/bin/python`, which includes the native client and `typesafe-sdk==0.7.4`.
Do not use system Python or replace global CUDA/PyTorch.

Pin author source to these commits in a clean checkout (untracked files are also
rejected for Git-based references). Reference subprocesses set
`PYTHONDONTWRITEBYTECODE=1` so these runs do not introduce `__pycache__` files:


| Family | Repository | Commit |
|---|---|---|
| Open-Jev | `Zefan-Cai/Open-Jev` | `bd4118882f733574a3250a4b65fe4d884130c08b` |
| JPT | `tic-top/llm2jev` | `2b252d504972764211ef172c1155ac0fedc9c3de` |
| NeoHorse | `TokenRhythm/NeoHorse` | `c494069accfc2d08839675961c1fdfcc54ba0b92` |
| Kev | `jaredpalmer/kev` | `6d02f5d066cd34958dfd15ffa5d2f6f0f4c21a63` |

Kev may use the checkpoint's vendored reference sources: its runner checks
`kev/model.py`, `kev/api.py`, and `kev/checkpoint.py` against pinned checkpoint
provenance. Kev's upstream nested config ignores its outer FP32 setting; the
reference creates a separate config view setting both `dtype` and
`text_config.dtype` to float32, preserving original weights. Its report explicitly
records this override and observed parameter dtypes. This is not an unmodified
author benchmark.

NeoHorse additionally needs `neohorse_decision-1.0.0-py3-none-any.whl`, SHA-256
`9b653e635b2adc9861064005924dfdd884b91648b34b293f4de9ff5ada4b51c1`.
Download it separately from the package index (`uv pip download` is not required;
`python -m pip download --no-deps neohorse-decision==1.0.0` in a suitable environment
is sufficient), then pass `--reference-wheel /path/to/file.whl` and
`--reference-wheel-sha256` with that hash. The wheel is added to the reference
import path; the summary records its filename/hash. No wheel is fetched silently.

## Run and inspect

Replace the paths below with your prepared artifacts and checked-out author code.
`--output-dir` must not already exist; this prevents stale results from appearing
successful. `--plan` prints exact argument arrays without touching Docker, weights,
or output files. This plan includes your local paths; do not publish it unredacted.

```bash
.venv/bin/python -m validation.run \
  --model kirp/jpt-0.8b \
  --model-dir /path/to/verified-jpt-source \
  --upstream /path/to/llm2jev \
  --reference-python /path/to/runtime-reference-env/bin/python \
  --image "$VALIDATION_IMAGE" \
  --docker-host "$VALIDATION_DOCKER_HOST" \
  --output-dir /path/to/new-validation-output \
  --port 18792 --timeout 1800 --plan
```

Remove `--plan` to execute. The timeout applies to each inference/readiness stage,
not the entire run; accepted values are 1–1800 seconds (the readiness CLI limit).
SIGTERM becomes a controlled interruption; subsequent SIGTERM/SIGINT signals are
ignored during cleanup and summary writing. Reference timeout/interruption kills its own process group;
cleanup attempts a graceful container stop, then removal. The private `plan.json`
records command arguments and the exact resource name for recovery. Inference is never
retried. A failed cleanup is an execution failure. Inspect private cleanup logs
and remove only that run's exact resource names if Docker was unavailable.

`summary.json` separates `execution` (`passed`/`failed`), `parity`
(`passed`/`failed`/`not_run`) and `cleanup`. Exit codes are **0** for execution and
parity success, **1** for completed execution with parity failure, and **2** for
execution/input errors. Every started run writes a summary, including failures. The public `error_code` and `stage` identify failures without
exposing paths: for example `gpu_busy`, `upstream_unpinned_or_dirty`,
`subprocess_failed`, `timeout`, or `validation_locked`.
Invalid arguments or an existing output directory fail before creating a run.
The summary binds model revision, verified source-manifest identity, upstream
revision, image repository digest and local image ID, source Git commit/dirty flag,
source-file hashes, wheel metadata when applicable, and raw JSON report hashes.
`real_gpu=true` is set only after reference and container HTTP/SDK completion.

Publish the reviewed `summary.json`, `reference.json`, and `http.json` together.
The summary omits local paths, Docker socket and exception messages. `plan.json` and logs may
contain private paths/environment details and must be reviewed/redacted before
sharing. A source dirty flag and content hashes expose local changes; matching a
Git commit alone does not establish that the image contains those local changes.
A validation summary is not a release attestation.

For all seven profiles, run separate invocations sequentially with each family's
matching reference environment and a fresh output directory. Do not parallelize
GPU work. Open-Jev covers 113 reference cases and 110 HTTP-valid cases; the other
families use the smaller matrix from `validation/reports/jpt-encoding.json`.

CPU orchestration checks:

```bash
.venv/bin/pytest tests/unit/test_validation_run.py -q
```

## Observed end-to-end runs

The public coordinator completed on an idle development RTX A6000 on 2026-10-13:

| Profile | Reference/container/SDK/cleanup | Strict HTTP parity |
|---|---|---|
| JPT-0.8B | Passed | Failed, 6/10 cases |
| NeoHorse-Jev-4B | Passed | Passed, 10/10 synthetic cases |

Reports: [JPT](../validation/reports/hardening-jpt-0.8b) and
[NeoHorse](../validation/reports/hardening-neohorse-4b). The JPT run began at clean
source commit `fea51b2`; the NeoHorse run records a dirty tree while independent
documentation/diagnostic tools were being added. Neither retroactively certifies
the old image build/source binding. The original busy-GPU refusal is retained
separately; it is not replaced with a success claim. Other profiles were tested
historically, but not rerun through this new coordinator in this session.
