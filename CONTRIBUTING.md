# Contributing to JevServe

The repository is currently a documentation bootstrap. Read [AGENTS.md](AGENTS.md), the [handoff](docs/development/HANDOFF.md), and the [implementation plan](docs/plans/JevServe-task-plan.md) before starting work.

Use a project-local environment or container. Preserve existing edits, keep changes task-sized, and include behavior-focused tests. CPU checks must not require model downloads; GPU tests must run real models. Record commands, outcomes and unrun checks in [STATUS.md](docs/development/STATUS.md).

Do not commit weights, private data, secrets, caches or local server details. Keep English and Chinese README capability/status statements aligned. Cite and retain licenses for reused code; model licenses are separate from the repository license.

A future release must identify the exact image digest and model revision that passed validation. Do not add performance claims without reproducible measurements. No package, image or public Git remote has been published during bootstrap.
