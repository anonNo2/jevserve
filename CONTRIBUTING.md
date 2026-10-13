# Contributing to JevServe

Start with a small, reproducible change. JevServe is a development preview for
self-hosted text Choice/Noul/Score inference; there is no certified model or
alpha/stable release. See [current status](docs/current-status.md) and the
[model matrix](docs/models.md) for the evidence and limitations.

## Set up and check a source checkout

Use project-local Python 3.12 and `uv`. No GPU, model download, Docker daemon or
system CUDA changes are needed for these checks:

```bash
uv sync --locked --extra server
.venv/bin/jevserve --help
.venv/bin/pytest -q
.venv/bin/ruff check .
.venv/bin/mypy
.venv/bin/python -m validation.api_schema --check
.venv/bin/python -m build --no-isolation
```

The `server` extra installs the CPU HTTP stack; it does not install a GPU runtime.
The base package contains the client and contracts. Use `uv run --extra server`
if you prefer `uv run` to direct `.venv/bin` commands, so dependency synchronization
retains the server extra. For real deployment use [Quick Start](docs/quickstart.md).
For runtime development see [IMPLEMENTATION.md](docs/development/IMPLEMENTATION.md).

## Pick work and navigate

The [contributor guide](docs/contributing.md) maps directories to tasks and checks.
Documentation/examples, focused CPU failure-path tests, API contract improvements,
and reproductions of an existing bug are useful starting points. Read
[API reference](docs/api.md) for protocol changes, or the
[model integration checklist](docs/contributing-models.md) before proposing a checkpoint.
The [hardening plan](docs/plans/open-source-hardening.md) tracks broader work; ask in
an issue before starting a large architecture change.

Use the [bug report](https://github.com/anonNo2/jevserve/issues/new?template=bug_report.md)
or [model request](https://github.com/anonNo2/jevserve/issues/new?template=model_request.md)
template. For security concerns, follow [SECURITY.md](SECURITY.md).

## Submit a pull request

Explain the problem, resulting behavior, and checks you actually ran. Include
failures and unrun checks. Keep the change focused, preserve unrelated edits, and
add meaningful behavior tests for changed contracts or failure paths. Keep English
and Chinese README capability statements aligned. A small contribution does not
require reading the historical plans or updating the maintainer's session log.

Preserve tokenizer/template semantics, candidate order, probability meaning and
precision. CPU tests, Docker builds, actual inference and numerical parity are
separate checks; label agreement is not probability parity. Never use test doubles
as production fallback. Inspect shared resources before GPU/Docker work and clean
up only resources belonging to your task.

Do not commit weights, caches, credentials, private data or host details. Keep
reused-code attribution/licenses; model licenses are separate from this project's
license. Publishing needs an authorized destination and reproducible release
evidence. JevServe is independent of TypeSafe AI and JevServe-Bench.

Maintainers and coding agents resuming an implementation session should additionally
follow [AGENTS.md](AGENTS.md), [HANDOFF.md](docs/development/HANDOFF.md), and
[STATUS.md](docs/development/STATUS.md). Session evidence and next actions belong in
STATUS; material architecture decisions belong in DECISIONS. These are continuation
records, not prerequisites for a typo fix or a focused first contribution.
