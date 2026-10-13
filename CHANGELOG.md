# Changelog

Versions below are development previews. JevServe is independent of
JevServe-Bench and TypeSafe AI. No alpha, RC or stable certification is claimed.

## Unreleased

- Bounded off-loop request encoding with cancellation accounting, shared shutdown
  deadline, sanitized error diagnostics and fail-closed abort cleanup.
- Lightweight client installation with optional server dependencies; isolated wheel,
  Compose and static API contract checks in CPU CI. Separate manual runtime-image job.
- Versioned OpenAPI exports and API reference; simplified contributor entry and
  current-status page. New source image requires explicit uncertified opt-in.
- Digest-bound dev1 advisory scan with initial triage and unresolved security actions.
  These source changes are not included in the published dev1 image.

- Real public-runner coverage for JPT and NeoHorse, JPT raw/head/hidden numerical
  diagnostics, bounded performance/soak and image rollback tools.
- Three executed business examples, model-contribution and issue/PR templates,
  security reporting policy, and a digest-bound SPDX inventory of dev1.
- Synthetic four-cell performance matrix and isolated corruption/offline/crash
  recovery evidence; probability certification remains incomplete.

- Public validation orchestration and evidence-bound release review tooling.
- Manually initiated, protected promotion of the already-tested registry digest.
  New policy has local CPU tests; remote promotion remains unexecuted.
- Release gates reject missing source/image/model bindings and preserve measured
  parity failures. Historical releases are not retrospectively certified.

## 0.1.0-dev1 — 2026-10-13

- Generic runtime with seven pinned profiles: Open-Jev-2B/9B,
  JPT-0.8B/4B/9B, NeoHorse-Jev-4B and Kev-4B, with externally mounted weights.
- All seven profiles executed real GPU/container inference. NeoHorse and Kev FP32
  passed small synthetic parity suites; Open-Jev and JPT probability parity still
  failed. This is not evidence of general model accuracy or full vision support.
- Published `anonwx/jevserve:0.1.0-dev1`, registry digest
  `sha256:4b758ad1000f4d38501fd83f0a9c2b32f0cb91968d427b170a2c2b0d9a613aff`.
  [Publication evidence](validation/reports/dev1-publication.json) records anonymous
  manifest access and digest pull. Historical evidence does not satisfy the new
  release source-binding schema; no source SHA is inferred here.

## 0.1.0-dev0 — 2026-10-12

- Initial Open-Jev-2B development runtime: Choice/Noul/Score HTTP API, CLI/SDK,
  artifact preparation, Docker and Compose paths, offline container inference.
- Published `anonwx/jevserve:0.1.0-dev0`, registry digest
  `sha256:f27c15c1da40d0c02139672a426d41d5370e6ea061d209b1d3fc5aef53b73396`.
- Numerical parity remained failed. See the historical publication section of
  [STATUS](docs/development/STATUS.md) for observed commands and limitations.
