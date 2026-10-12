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
- **D020 — Ship a clearly labelled development image with a usable default.**
  The preview image's default command explicitly includes `--allow-uncertified`;
  source CLI serving still requires that flag. README and startup helper disclose
  numerical certification status. This is not a certified model-registry entry
  and must not be published as a stable/alpha-certified release. The helper polls
  readiness before one real three-question check; it never retries inference.
- **D021 — Isolate container validation from the shared daemon.** A project-local
  rootless Docker daemon and generated NVIDIA CDI specification allow CPU/GPU
  container validation without restarting or changing the shared Docker service.
  Its socket, state and host observations remain under ignored `.local/`.
  Both legacy `--gpus device=0` and explicit native CDI probes ran successfully.
- **D022 — Remove conflicting optional upstream image extras.** The first real
  image build found base-image LMCache and OpenTelemetry Prometheus exporter
  constraints conflicting with the frozen runtime lock. Neither is part of this
  engine profile (prefix caching is disabled; JevServe owns its metrics endpoint).
  Remove these two extras and require `pip check` during build; preserve the
  existing pinned runtime versions and failed numerical parity evidence.
- **D023 — Give the container service account a writable cache home.** Real
  non-root startup reached model inspection, then FlashInfer failed trying to
  create `/.cache`. Add the UID 10001 account with `/cache` as its home and route
  FlashInfer/XDG/CUDA caches explicitly into that volume. Retain non-root
  execution and Compose's read-only root filesystem; do not broaden permissions.

- **D024 — Expand to one model-independent runtime image with mounted weights.**
  User explicitly requested more Jev checkpoints (including JEV-27B-VL, Xor 1.2
  and JPT-9B) tested on the current 48 GiB host. The same image must select a
  pinned adapter/model at startup and accept an external read-only weight mount.
  One model/engine per container remains; model-specific images are not the design.
  Existing automatic preparation remains optional. Each model needs its own
  prompt/readout/calibration semantics, license and evidence. Larger BF16 models
  need an explicitly configured and separately tested offload/quantized profile;
  no automatic precision reduction or fallback. Text and vision evidence must
  be reported separately; the historical single-model alpha scope is superseded.

- **D025 — Use a named reference backend for NeoHorse's pointer head.** Its
  published merged backbone and pointer readout are not ordinary causal-LM
  generation. Integrate the native BF16-backbone/FP32-head Torch path as an
  explicit model profile in the same image, not as fallback from a failed vLLM
  call. One spawned GPU worker serializes inference; cancellation of active
  Torch work terminates that worker and makes the service unhealthy, requiring
  explicit restart. It never leaves an abandoned GPU thread or retries a request.
  vLLM remains the engine for Open-Jev and JPT. Benchmark/certification claims
  must identify the selected backend and the native-reference evidence.

- **D026 — Read all JPT candidates through an explicit vocabulary mask.** The
  first real 255-option request exposed vLLM 0.29's hard 128-entry
  `logprob_token_ids` limit, despite the model/tokenizer supporting 255 labels.
  Mask only non-candidate tokens, set `logprobs=N`, `top_k=-1`, `top_p=1` and
  sampling temperature 1, and read all N finite processed logprobs. Conditioning
  subtracts a shared log-normalizer, so the adapter's checkpoint-temperature
  softmax preserves the candidate distribution. Missing candidates fail closed.
  This is one forward pass, with no truncation, extra inference pass or retry.
  The initial 255-option failure and subsequent parity check are separate gates.

- **D027 — Keep Kev's evaluated FP32 profile explicit.** The pinned `head.pt`
  records FP32 training weights, full LoRA placement, no option isolation and no
  trained special-token embeddings. Load the pinned base in FP32, merge the
  adapter in FP32 and use its exact stored temperature
  `2.406050072164233`. Do not replace it with the rounded model-card value or
  silently select BF16 to save VRAM. The shared image pins the same Torch,
  Transformers, PEFT and safetensors versions for the pointer worker. NeoHorse
  separately retains its author's merged BF16-backbone/FP32-head loading path.

- **D028 — Verify the loaded precision, not only the loader argument.** A GPU
  memory audit found Transformers 5.10.4's multimodal-to-text config extraction
  discarded Kev's outer FP32 argument: the initial backbone was actually BF16.
  Those initial reports are retained with `precision_profile_valid: false`.
  Production now passes the extracted text config with explicit FP32 and checks
  every merged backbone parameter. Reference author code remains unchanged, but
  runs against a documented local configuration view setting both outer and
  nested dtype to FP32; weight bytes remain the pinned originals. This is a
  corrected loader profile, not an unmodified author benchmark environment.
