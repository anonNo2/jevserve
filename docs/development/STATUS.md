# Development status

## Current milestone

Documentation/bootstrap only. No implementation or GPU certification yet.

## Task tracking

| Task | Status | Next evidence required |
|---|---|---|
| Bootstrap | Complete | Git baseline includes docs; private host notes remain local |
| T00 Baseline | Not started | Real upstream revisions, licenses, GPU/container profile and reference results |
| T01 Contracts | Not started | CPU tests and installable package |
| T02 Artifacts | Not started | Atomic preparation and offline tests |
| T03 Adapter | Not started | Reference encoding/readout parity |
| T04 Runtime | Not started | Real GPU inference |
| T05 API/lifecycle | Not started | Protocol, overload, cancellation and failure tests |
| T06 CLI/SDK | Not started | Real end-to-end check |
| T07 Docker | Not started | Fresh-cache single-container inference |
| T08 Compose/offline/rollback | Not started | Real deployment and recovery exercises |
| T09 Second model | Optional, not started | Independent reference/GPU certification |
| T10 Observability/benchmark | Not started | Measured reports and milestone-specific soak |
| T11 CI/release | Not started | Exact tested image promotion |
| T12 Open-source acceptance | Not started | Clean-host quickstart and complete alpha gates |

## Next action

On user instruction to implement, begin T00. Read `.local/SERVER.md`, recheck current GPU occupancy and verify container GPU passthrough. Host GPU visibility alone is insufficient.

## Bootstrap checks

- Remote destination was empty before initialization.
- Linux NVIDIA hardware, Docker/Compose versions and daemon access were inspected without changing system configuration.
- Docs contain explicit not-implemented status; no product code was added.
- Private host information is ignored via `.local/`.
- Markdown links, stale-name absence, document transfer checksums and clean Git status are checked during bootstrap delivery.

## Update format for future work

For each session add: task ID, files changed, exact commands, passed/failed/unrun checks, artifact paths, blockers, commit and next command. A blocked GPU gate must stay uncompleted even if CPU work can continue.
