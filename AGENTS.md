# JevServe — instructions for coding agents

## Read first

Read `docs/development/HANDOFF.md`, `docs/development/STATUS.md`, `docs/plans/JevServe-design.md`, and `docs/plans/JevServe-task-plan.md` before changing implementation. If `.local/SERVER.md` exists, read it for this host's observed environment; it is intentionally not tracked. User instructions take precedence over this file and historical plans.

## Scope and decisions already made

- Brand: JevServe. Repo/package/CLI/image base name: `jevserve`. Naming discussion is complete; the user accepted proximity to JevServe-Bench. State independence from that project and TypeSafe AI.
- Target: single Linux NVIDIA host; one reusable runtime image for all supported models, externally mounted weights, one model per container, one GPU; `docker run` and Docker Compose are both first-class paths.
- vLLM first, bounded adapters. No Kubernetes, training platform, web UI, multi-node, automatic fallback, or empty Triton implementation in scope. The user expanded model integration to current Jev checkpoints including VL families; report text and vision support separately.
- At least one real Jev-style checkpoint must pass GPU and container checks. Default candidate: `ZefanCai/Open-Jev-2B`; availability, license and revision must be verified. Do not invent a substitute model or silently substitute ordinary JSON generation.
- CPU contracts, artifact preparation, real GPU inference, CLI and container definitions are implemented; consult STATUS for current verification and release blockers. Do not claim commands work until implemented and verified.

## Execution

When the user asks to start implementation, begin T00 and proceed in dependency order. Do not reopen settled architecture or ask for confirmation at every reversible step. The plan is the agreed direction, but evidence may require adjusting a dependency or interface; record concrete changes in `docs/development/DECISIONS.md`.

Historical skills named in the plan may not be installed on this host. If unavailable, follow the concrete tasks and tests directly; skill absence alone is not a blocker. Default to sequential local work. Delegate only when requested or allowed by the active session's instructions. Review changes before completion; obtain independent review when the active tooling and instructions allow it.

Plan test snippets illustrate contracts and are not complete executable fixtures. Implement their fixtures/imports/helpers and test the actual behavior. Fix plan inconsistencies when discovered; never create dummy assertions merely to match a snippet.

## Engineering rules

- Use a project-local Python 3.12 environment or the pinned container; do not replace system Python or globally upgrade CUDA/PyTorch.
- Keep protocol tests usable without GPU imports or downloads. Never allow test doubles as production fallback.
- Preserve tokenizer/template/readout semantics, candidate order, masks, and probability meaning. Softmax is not calibrated accuracy.
- Pin real upstream commits, weights and image digests. Historical version claims are candidates, not verified facts. Record inaccessible/nonexistent upstream resources as blockers, not successes.
- Bound body size, queue length, work cost and deadlines. Propagate cancellation; no implicit truncation, model substitution, precision change or retry.
- One API process owns one engine; do not multiply model copies using web worker count.
- Keep logs and metrics free of tokens, request bodies and high-cardinality identifiers. No secrets in source or command lines.
- Download to staging, verify hashes, and atomically publish. Offline mode must not attempt network access.
- Keep docs bilingual at the README level; sync capability/status changes across both README files.

## Shared-server hygiene

Inspect GPU use and Docker resources before heavy work. Do not kill others' GPU processes, restart the Docker daemon, change driver/runtime settings, run global prune, or delete unrelated volumes. Use project-scoped container/volume names. Do not modify files outside the project without a concrete need and appropriate authorization.

Do not commit weights, model caches, credentials, raw private data, or `.local/`. Published reports must redact host/user/network details unless intentionally public. Never push or publish without user authorization and a real destination.

## Verification and handoff

For each task, record commands, observed outcomes, artifacts, unrun checks and blockers in `docs/development/STATUS.md`. Commit coherent completed changes using the user's existing Git identity. Do not fabricate authorship or overwrite user edits.

CPU green, Docker build green, and real GPU certification are separate gates. Alpha needs real inference and both container paths. RC adds 24-hour soak; stable adds 72-hour soak and clean-host installation. Do not mark unexecuted gates passed.

Before ending an implementation session, update STATUS with completed tasks, exact next action and reproduction commands. Update DECISIONS for material deviations. Keep local machine findings in `.local/SERVER.md`.
