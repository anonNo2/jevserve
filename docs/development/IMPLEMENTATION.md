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

The package currently exposes Python contracts, artifact preparation and an
Open-Jev encoding/readout adapter. The adapter requires an explicitly supplied
tokenizer and verified model capabilities; it never constructs a fake engine. There is
no `jevserve` CLI or HTTP server yet. `DecisionRequest.from_json` / `parse_request`
reject duplicate JSON fields recursively; a future HTTP handler must call that
parser after enforcing the body limit. Calling Pydantic's generic JSON method
directly does not provide duplicate-field rejection. Native instructions and
candidate descriptions are text (Choice permits null descriptions); broader
upstream object/array description compatibility is deferred to T05.

`Capabilities` enforces model candidate and token work budgets separately from
request structure. The adapter must validate token lengths after encoding and
before admission to GPU work. A model's limits are not inferred from available
GPU memory. Responses reject invalid distributions, mismatched ordinal scores,
and unstable tie selection. These types are not evidence of a running model.

## Artifact preparation

`prepare_model(manifest, cache_dir, offline=False)` uses an explicit immutable
`ModelManifest`. Unknown registry IDs fail; there is no default model yet. Each
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
head/tokenizer structure; T03/T04 must additionally validate those before engine
startup. `validation.acquire` exercises this same preparer for baseline sources.

## Gates

CPU checks, package build, Docker execution and GPU model certification are
separate. The initial lock checker rejects the current candidate baseline for
release. It is a fail-closed baseline guard, not the complete T11 image-promotion
pipeline. Use STATUS for actual evidence and unrun checks.
