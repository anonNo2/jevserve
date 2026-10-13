# Contributing to JevServe

JevServe is a working development preview with a public [source repository](https://github.com/anonNo2/jevserve) and generic image `anonwx/jevserve:0.1.0-dev1`. Seven pinned model profiles have real GPU/container execution evidence. Open-Jev/JPT probability comparisons still fail; NeoHorse/Kev pass only a small synthetic suite. No model is certified and there is no alpha or stable release. See the [model matrix](docs/models.md) for text support, licenses and limitations; vision input is not implemented.

Read [AGENTS.md](AGENTS.md), the [handoff](docs/development/HANDOFF.md), and [current status](docs/development/STATUS.md). Choose work from the [open-source hardening plan](docs/plans/open-source-hardening.md). The original [design](docs/plans/JevServe-design.md) and [T00–T12 plan](docs/plans/JevServe-task-plan.md) preserve historical intent; their checkboxes and example commands are not current status or deployment instructions.

For a new checkpoint, follow the [model integration guide and acceptance checklist](docs/contributing-models.md). Use the [bug report](https://github.com/anonNo2/jevserve/issues/new?template=bug_report.md) or [model request](https://github.com/anonNo2/jevserve/issues/new?template=model_request.md) template for public issues. For a possible vulnerability, read [SECURITY.md](SECURITY.md) before posting any details.

Use Python 3.12 in a project-local environment. From the repository root, with `uv` installed:

```bash
uv sync --locked
uv run jevserve --help
uv run jevserve models list
uv run pytest -q
uv run ruff check .
uv run mypy
uv run python -m build --no-isolation
```

CPU checks need neither GPU packages nor model downloads. Use the [deployment guide](docs/quickstart.md) for the tested image digest, Docker and Compose commands; a source GPU runtime requires the separate pinned dependencies described in the [implementation workflow](docs/development/IMPLEMENTATION.md).

Preserve existing edits and keep changes focused. Add behavior tests for changed contracts or failure paths. In a pull request, explain the problem, resulting behavior and exact checks performed, including failures and unrun gates. Record task evidence and the next action in [STATUS.md](docs/development/STATUS.md); record material design changes in [DECISIONS.md](docs/development/DECISIONS.md).

Numerical validation must retain the frozen tolerances, tokenizer/template semantics, candidate order and precision. Real GPU results, CPU tests and container execution are separate gates; label agreement alone does not establish probability parity. Inspect shared GPU/Docker use before heavy work and clean up only resources created for your task.

Do not commit weights, private data, credentials, caches or local server details. Keep English and Chinese README capability/status statements aligned. Retain attribution and licenses for reused code; model licenses are separate from the repository license.

A release must identify the exact tested source revision, image digest and model revisions. Preserve reproducible evidence for performance claims. No Python package has been published. Pushing or publishing requires authorization and a real destination.
