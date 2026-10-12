# Local implementation workflow

Use Python 3.12 and a project-local environment. The protocol and artifact code
has no Torch, Transformers or vLLM dependency:

```bash
uv sync --locked
uv run pytest -q
uv run ruff check .
uv run mypy
uv run python -m build
```

## Real development service

The CLI, HTTP service, native SDK and pinned vLLM bridge are implemented. The
runtime is explicitly **uncertified**: see STATUS for failed numerical parity
and blocked container gates. Use the source setup in the bilingual README.
The development host uses `.local/engine-venv`, not the CPU `.venv`.

```bash
.local/engine-venv/bin/jevserve serve --allow-uncertified --offline \
  --cache-dir .local/runtime-cache --port 18791
.venv/bin/jevserve check --url http://127.0.0.1:18791
.venv/bin/jevserve benchmark --url http://127.0.0.1:18791 \
  --requests 20 --output .local/benchmark.json
```

For another host, prepare with `jevserve prepare --cache-dir CACHE` before
using `--offline`. Set `--token-file` for a Hub credential file; never place its
contents in command arguments. `--api-key-file` enables constant-time Bearer
authentication on decisions/models/metrics; health endpoints stay unauthenticated.
The local default bind is 127.0.0.1. Container defaults use 0.0.0.0 internally
and the documented port publication binds the host to loopback.

Source and derived checkpoints are separate immutable cache directories.
Preparation verifies source hashes; compilation strips vision-only tensors,
renames text tensors and preserves base/head/LoRA values. The FP32 LoRA residual
path is fixed to this checkpoint and never switches to BF16 dynamic LoRA.
Startup verifies versions and warms the engine before readiness. Engine death
closes admission and makes the CLI exit nonzero. SIGTERM drains pending requests
within the configured budget and closes the engine. No access/body logging or
vLLM scheduler-input failure dumps are enabled.

`jevserve models list` reports development status. `doctor` reports installed
versions and a read-only GPU observation; it neither reserves the GPU nor
certifies Docker. `check` performs actual HTTP inference. `benchmark` is a
bounded sequential workload (up to 1000 requests), not a capacity or soak test.
The native SDK never retries. Official SDK compatibility is documented in
[compat/README.md](../../compat/README.md).

For Compose preparation/offline use, after a verified image build:

```bash
docker compose run --rm --no-deps jevserve prepare
docker compose run --rm --service-ports jevserve serve --allow-uncertified --offline
```

These container commands remain **unexecuted** while Docker creation is blocked.
Keep prepared caches when changing images; pin an actual built image ID/digest
for rollback. There is no certified release image or completed rollback drill.

## Artifact preparation

`prepare_model(manifest, cache_dir, offline=False)` uses an explicit immutable
`ModelManifest`. Only the pinned Open-Jev-2B development target is accepted; unknown IDs fail. Each
file binds its local path, immutable source revision, byte length and SHA-256.
The preparation identity hashes the full manifest, including base-model sources.

Online preparation locks the identity, checks free space, downloads into a
private staging directory, verifies all files, writes manifest and readiness
metadata, then atomically renames on the same filesystem. Failures clean only
the staging directory created by that invocation. A corrupt final package fails
closed; automatic repair or destructive cache replacement is not implemented.
The default downloader has one attempt, a 30-second socket timeout and an
1800-second per-file time budget. There is no retry on inference or automatic
model substitution. Tokens are read from files and removed on CDN redirects.

Offline missing packages fail before directory creation, token access or any
network call. Cache hits rehash all declared files, check metadata identity and
reject symlinks and extra files that could alter model loading. Cache directories
must be owned by a trusted administrator: this is not isolation against another
process with the same filesystem write privileges. Atomic publication prevents
readers seeing partially prepared packages; it does not promise power-loss
survival on every filesystem. Read-only verified caches work offline.

The preparation library verifies byte integrity and paths, not model-specific
head/tokenizer structure; the bounded converter/runtime additionally validate those before engine
startup. `validation.acquire` exercises this same preparer for baseline sources.

## Gates

CPU checks, package build, Docker execution and GPU model certification are
separate. The initial lock checker rejects the current candidate baseline for
release. It is a fail-closed baseline guard, not the complete T11 image-promotion
pipeline. Use STATUS for actual evidence and unrun checks.
