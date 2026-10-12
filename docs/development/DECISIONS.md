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
- **D015 — Reject BF16 merged-LoRA export for the initial profile.** The isolated
  HF merge experiment changed probabilities by up to 0.0160318306 and failed the
  predeclared tolerance on 35/113 cases (no label flips). Evidence is in
  `validation/reports/bf16-merge-rejected.json`. Use original base tensor bytes
  plus independent FP32 LoRA weights, subject to real vLLM parity. Do not loosen
  the original tolerance or call the merged checkpoint certified.
- **D016 — Bound upstream reuse to the text Qwen3.5 pooling model.** Vendor only
  the small model registration bridge with Apache attribution. The service owns
  its admission/API/CLI and does not install unrelated Laya/multimodal adapters.
- **D017 — Fixed FP32 LoRA is a development profile, not a parity exemption.**
  Native vLLM dynamic LoRA rejects FP32 in its expansion kernel. The adapter
  preserves FP32 residuals around native linear layers with TP=1. Real pooling
  and HTTP execute, but frozen probability tolerances fail. Require the explicit
  `--allow-uncertified` flag, retain all failed reports and keep the release
  registry/gates closed. The reference-style normalization experiment was
  reverted because it did not resolve the discrepancy. No precision change.
- **D018 — Freeze a narrow official SDK compatibility claim.** Test the actual
  `typesafe-sdk==0.7.4` against ASGI and GPU HTTP. Support native text-question
  semantics plus an exact model ID; do not claim hosted account APIs, all JSON
  question shapes or aliases. Return actual candidate input-token counts and
  zero generated tokens; preserve the author's concentration formulas.
- **D019 — Continue framework work without declaring container success.**
  T05/T06, metrics and CPU CI can be implemented and exercised against host vLLM
  despite T00's Docker blocker. Dockerfile and Compose are reviewable artifacts;
  only Compose structural validation has passed. Container/alpha/soak gates
  remain mandatory and unpassed. No shared daemon/runtime changes were made.
