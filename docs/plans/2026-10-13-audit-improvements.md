# Open-source audit improvements

The user approved implementation of the repository audit and explicitly requested
subagents on 2026-10-13. This extends the existing hardening plan; it does not
reopen model selection, relax numerical tolerances or authorize publication.

## Design and acceptance

- Keep the single API/engine owner. Move request encoding off the event loop with
  a single bounded worker and real completion accounting. Deadline/disconnect must
  not create replacement work while cancelled synchronous work remains alive.
  Test responsive health checks, deadline response, bounded overload and recovery.
- Emit allowlisted diagnostic events and finite error metrics. Never interpolate
  exception messages, request bodies, credentials or private paths.
- Keep native client dependencies to Pydantic and HTTPX; put HTTP serving packages
  in a server extra. Retain the tested Python 3.12 range. Verify installed wheels
  outside the checkout, without GPU packages; preserve pinned image dependencies.
- Extend CI with artifact installation and Compose validation. Full runtime image
  builds need a suitable hosted runner and must report unavailable capacity as
  unrun, never a successful container gate. Do not expose shared GPU runners to PRs.
- Export static versioned API schemas from contract models and explicit operation
  semantics, with drift tests. Keep live API docs disabled. Document all endpoints,
  errors, auth, timeout behavior and model-dependent limits.
- Simplify the contributor entry; keep historical evidence intact and maintain a
  concise user status. Synchronize English/Chinese README installation/status.
- Recheck security reporting and use the existing immutable SBOM for a real
  vulnerability assessment if the scanner/database are available. Report findings
  and triage limits; never treat package inventory as a clean vulnerability scan.
- Reuse numerical evidence to specify the next isolated experiment. The active
  24-hour NeoHorse soak owns the GPU; do not interrupt it, change precision,
  invent passing certification, or treat cached installation as a clean host.

## Ownership and verification

- Runtime subagent: server encoding, diagnostic events and behavior tests.
- Packaging subagent: dependency split, lock, Dockerfile, installation checks, CI.
- API/documentation subagent: static contracts, schema tests, contributor/API docs.
- Coordinator: security/evidence work, bilingual README, decisions/status, review,
  integration tests and coherent local commits using existing identity.

Run focused tests after each behavior change. After independent review, run the
whole CPU suite, Ruff, mypy, package build, fresh wheel installations, schema drift,
Compose validation and documentation-link checks. GPU/container execution,
clean-host installation and remote release promotion remain separate gates.

## Execution ledger

- Initial tree clean; CPU baseline 352 tests, Ruff and mypy passed in audit.
- Soak inspected: running, 1,434 successful requests, 14,674 elapsed seconds,
  no failures at snapshot. Project container healthy; GPU 9,216 MiB in use.
- Decision: keep Python 3.12 requirement until additional versions are tested;
  dependency separation alone does not prove a wider supported Python range.
- Decision: finish authorized local changes without re-asking approval of the
  preceding audit. Remote publication still requires a concrete reviewed result.

- Runtime, packaging/CI and static API/contributor tasks implemented; cross-review
  and independent final review complete. Shutdown deadline/abort safety and one
  skipped-gate status message were fixed and re-reviewed.
- Final whole CPU suite 374 passed, Ruff/mypy/schema drift passed. Wheel/sdist,
  fresh isolated client/server installs, dependency checks and Compose config
  checks passed. 220 Markdown file links resolved. No production image/GPU gate run.
- Grype 0.120.1 scan completed: 2,123 package/advisory matches retained in sanitized
  inventory, with scanner/database provenance and explicit incomplete triage.
- Remaining external gates: active soak completion; new-source container/GPU
  regression; numerical certification; clean-host/capacity/72-hour tests; real
  release rehearsal and administrator-enabled private vulnerability reporting.
- Decisions and exact next actions are retained in DECISIONS D033–D036 and the
  latest STATUS session. No remote push or publication performed.
