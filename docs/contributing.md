# Contributor guide

Start with [CONTRIBUTING.md](../CONTRIBUTING.md) for the Python 3.12 environment and
checks. You can work on documentation, clients and CPU contracts without owning an
NVIDIA GPU or downloading model weights.

## Where to work

| Directory | Responsibility | Useful checks |
|---|---|---|
| `src/jevserve/contracts/` | Strict requests, typed answers, errors and model budgets | `tests/contract/` |
| `src/jevserve/client.py` | Native synchronous/asynchronous HTTP client | Client contract tests |
| `src/jevserve/server/` | Authentication, bounds, admission and lifecycle | HTTP/admission tests |
| `src/jevserve/artifacts/` | Pinned model manifests, preparation, hash verification | Artifact/prepare tests |
| `src/jevserve/adapters/` | Model-specific encoding and answer readout | Adapter tests; later real parity |
| `src/jevserve/runtime/` | Engine ownership, abort and GPU backends | Runtime unit tests; separate GPU evidence |
| `validation/` | Reproducible validation/reporting tools and static API exporter | Relevant unit/contract tests |
| `compat/` | Engine locks, pinned requirements and exported OpenAPI documents | Lock tests and schema drift check |
| `examples/` | Small synthetic business examples | `tests/unit/test_examples.py` |
| `docs/` | Current usage and contributor guides | Review commands/links and bilingual README claims |
| `docs/development/`, `docs/plans/` | Maintainer handoff/evidence and historical design | Read when relevant to your change |

Run a focused file while iterating, for example:

```bash
.venv/bin/pytest -q tests/contract/test_requests.py
.venv/bin/pytest -q tests/contract/test_http.py
```

Run the complete CPU checks from CONTRIBUTING before submitting a behavior change.
Protocol changes should regenerate both [OpenAPI exports](api.md#updating-the-contract).
Examples and tests must use synthetic content and should not require network
access. Do not introduce GPU imports into the client or protocol contracts.

## Useful first contributions

- Reproduce a parser/client/lifecycle bug with a focused CPU test and minimal input.
- Improve an example or an error explanation while preserving the actual contract.
- Correct a broken documentation command/link and describe how you checked it.
- Improve a validation report's clarity without changing frozen numerical tolerances.

Model integration requires the [adapter checklist](contributing-models.md), pinned
upstream revisions/licenses and real inference evidence. Numerical work starts at
[numerical diagnosis](numerical-diagnosis.md); a passing label comparison does not
resolve probability drift. Scope substantial changes in an issue first.

## Pull request checklist

- State the concrete problem and before/after behavior.
- List executed tests, observed results, and any unrun GPU/container gates.
- Update affected user documentation and generated contracts.
- Keep secrets, caches, weights, private logs and `.local/` out of the diff.
- Preserve attribution and the distinction between implementation and certification.

Maintainer continuation records are in [HANDOFF](development/HANDOFF.md) and
[STATUS](development/STATUS.md). Ordinary small contributions need not rewrite these
records or read every historical session. Agents still follow the repository's
[AGENTS.md](../AGENTS.md) instructions.
