# Decision log

| ID | Decision | Reason |
|---|---|---|
| D001 | Brand JevServe; technical names jevserve | User selected the name and accepted similarity to JevServe-Bench |
| D002 | Linux NVIDIA, single host, Docker/Compose | Explicit user deployment constraint |
| D003 | Single image/container by default | Meet the vLLM-like minimal startup experience |
| D004 | vLLM first; bounded adapters | Reuse the inference engine and focus on correct model integration |
| D005 | No Triton implementation or Kubernetes in alpha | Avoid expanding scope before one real model works |
| D006 | Real GPU evidence required | Bootstrap or CPU-only success does not satisfy the product goal |
| D007 | Private host notes in ignored .local/ | Preserve development context without committing private infrastructure details |
| D008 | Planning docs are historical inputs to T00 | Upstream versions, models and docs need direct verification before use |

Add material changes with evidence, compatibility consequences and affected tasks. Do not treat this log as permission to change the user's agreed scope silently.

## Implementation decisions — 2026-10-12

- **D009 — Continue independent CPU work while T00 container execution is blocked.**
  CPU and GPU probes both fail during Docker shim creation with TTRPC unsupported
  protocol. No daemon/runtime/driver changes are authorized on this shared host.
  T01/T02 proceed; T00, runtime and container certification stay incomplete.
- **D010 — Isolate author reference and engine dependency environments.** Author
  source pins Transformers 5.10.2 / PEFT 0.19.1; candidate vLLM 0.29.0 requires
  Transformers >=5.10.4 and Torch 2.13.0. Their dependency declarations cannot
  share one frozen environment without changing the reference. Reference uses
  its own hash-pinned requirements; engine values remain declared, not observed.
- **D011 — Baseline inventory is distinct from a serving manifest.** Open-Jev-2B
  is an adapter/head package and requires the separately pinned Qwen base.
  Discovery locks record both, with explicit pending gates. The serving registry
  stays empty. T02 preparation manifests bind per-file source repositories and
  revisions and never imply certification.
- **D012 — Native structural validation is narrower than upstream compatibility.**
  Native state is nonempty text/object, descriptions/instructions are text,
  Choice null descriptions are allowed, candidate order is preserved. Model
  candidate/context limits live in Capabilities. T05 must separately freeze and
  test official SDK compatibility; T01 is not that compatibility claim.
- **D013 — Preserve upstream saved temperature without claiming calibration.**
  The checkpoint temperature is part of readout semantics. Apply it once;
  `calibration_id=null` remains the native default. Parity thresholds are fixed
  in `compat/parity-profile.json` before first measurement.
- **D014 — Stage T02 acceptance across its dependencies.** The safe preparer and
  offline/failure tests are completed without waiting for an HTTP service. T06
  must call this same function for first-start preparation and verify actual
  serve integration. T02 CPU success does not claim GPU/offline inference.
