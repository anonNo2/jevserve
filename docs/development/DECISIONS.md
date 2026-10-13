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

- **D029 — Separate reviewed preview promotion from certification.** Release
  automation accepts only development-preview manifests binding an exact source
  commit, immutable registry digest, all shipped model revisions and hashed raw
  evidence. It promotes the same manifest bytes without a rebuild. Alpha/RC/stable
  remain disabled until their evidence policies exist. Historical dev0/dev1 reports
  are not retroactively assigned guessed commit bindings. New gate wrappers are
  reviewed attestations, not proof that hardware execution occurred.

- **D030 — Diagnose scores without redefining parity.** The CPU numerical report
  reanalyses existing real-GPU records with frozen tolerances and checkpoint
  temperature. Open-Jev probabilities reconstruct exactly from recorded scores;
  score differences remain unexplained. JPT reports lack raw logits. No causal
  attribution to BF16, precision change or tolerance relaxation follows from this
  analysis. Public validation defaults to an idle physical GPU 0, uses its UUID
  for the author reference, and refuses occupied hardware rather than evicting
  another workload.

- **D031 — Preserve production arithmetic while locating pre-head differences.**
  JPT-0.8B GPU probes verified identical input/candidate IDs and candidate BF16
  weight bytes. HF full-sequence/last-token/candidate-only BF16 head outputs were
  identical on the 13-question suite; substituting FP32 outputs changed semantics.
  Final hidden states differ between engines, and a common FP32 projection still
  exceeds frozen probability tolerance on 7/13 questions. Record a pre-head
  contribution, without guessing a particular kernel or changing production dtype.
  Diagnostic worker methods use named string RPCs, never insecure function
  serialization, and preserve the original inference return values.

- **D032 — Duration and request-count checks are different gates.** A workload
  completing its request cap early cannot pass a 24/72-hour gate, including when
  an interval would otherwise pad elapsed time. Reports distinguish completed
  requests, errors, actual duration and termination reason. Short synthetic
  performance, cached crash recovery, image rollback, clean-host install and
  sustained stability remain separate evidence. Long runs must have a fixed
  duration and explicit owner/cleanup; no automatic retry or restart hides failures.


- **D033 — Bound real encoding work beyond the caller lifetime.** One daemon-owned
  encoder serializes synchronous tokenizer calls outside the HTTP event loop.
  Cancellation/deadline does not release the real worker slot; readiness is false
  while orphaned encoding is still running and recovers after completion. No
  replacement worker or unbounded executor queue is created. Python cannot
  preempt native tokenizer code; permanently stuck code requires process exit.
  Request draining and encoder draining share one absolute shutdown deadline.
  Unexpected abort cleanup errors fail readiness closed rather than leaking raw
  framework tracebacks. Internal diagnostics use fixed stages/codes and an
  allowlisted exception category, without request IDs, private data or messages.

- **D034 — Separate client dependencies without inventing compatibility.** Core
  installation requires Pydantic and HTTPX; the `server` extra retains the pinned
  HTTP/metrics packages. Keep Python >=3.12,<3.13 until another interpreter is
  actually tested. The production runtime requirements/base image are unchanged.
  New source image defaults no longer silently add `--allow-uncertified`; the
  explicit development helper/Compose paths still supply it. Published dev1
  remains immutable and does not contain these changes.

- **D035 — Export contracts without weakening bounded raw-body handling.** Static
  OpenAPI documents combine the real Pydantic contracts with explicit endpoint
  definitions. Keep live Swagger/OpenAPI disabled; schema drift and actual HTTP
  response tests validate the exported contract. Cross-field ordering, probability
  normalization and model work limits remain runtime semantics, not promises that
  generic JSON Schema validation proves a request executable.

- **D036 — Keep container capacity and advisory evidence honest.** Every CPU CI
  run installs the built wheel outside the checkout and validates Compose/schema.
  Full vLLM builds use a separate trusted manual default-branch job with a dedicated
  large-disk CPU runner, never an untrusted PR on a shared GPU host. Its absence is
  unrun, not passed. Grype advisory matches against dev1's SBOM are published with
  scanner/database identity and no suppression; package matches do not establish
  exploitability. Runtime upgrades/removals require a newly validated candidate.

- **D037 — Validate unpublished candidates by local identity without release claims.**
  The offline validation runner accepts a full immutable Docker local image ID
  as an alternative to a registry reference. It checks the inspected and running
  image IDs, records `image_identity_kind=docker-local-id`, and omits
  `image_digest`; such reports cannot satisfy registry release gates. Docker 29
  on the observed project daemon returns an OCI index as its local ID, so it must
  not be described as the configuration digest. Candidate provenance separately
  compares installed package and embedded-wheel bytes against the source commit.
  No image publication is needed to perform local GPU/container acceptance.

- **D038 — Arithmetic counterfactuals require matched controls and actual coverage.**
  Qwen3.5 diagnostic substitutions remain outside production. Capture normalized
  HF/vLLM boundaries, account for split residuals and packed paths that bypass
  ordinary module forwards, and preserve intermediate dtypes in evidence.
  Single-factor experiments use a matched unpatched engine profile. A combined
  experiment tests only the explicitly named joint hypothesis. Record execution
  counts, restoration, source hashes and unchanged tolerances; reject missing
  coverage. Improved or unchanged probabilities do not certify a checkpoint or
  authorize a production precision change or HF fallback.
  Coverage alone does not prove correctness: a later-discovered diagnostic
  layout defect invalidates affected reports even when `valid_trial` was true.
  Preserve native convolution strides as well as shapes/dtypes when a downstream
  kernel assumes unit feature stride; retain the invalid run separately and
  bind corrected reruns to their implementation hashes.

- **D039 — Accept explicitly bounded drift with a separate versioned policy.**
  The user permits numerical drift. Adopt bounded-drift-v1 as the initial project
  threshold: per-candidate absolute probability error ≤ 0.03, zero relative
  tolerance, and unchanged first-in-order argmax for Choice/Noul/Score. The exact
  0.03 value is a project default, not a verbatim user requirement. Preserve old
  strict reports and thresholds; re-evaluate hash-bound saved GPU evidence with
  a CPU-only tool and record policy/evaluator hashes. This supersedes strict
  parity as the acceptance blocker for the current finite corpus, not historical
  observations. Do not change weights, temperature, production precision or
  reference values. Pinned Kev four-decimal serialization requires a reference-only
  internal-consistency budget, not wider acceptance tolerances or normalization.
  Report total variation and distribution-derived/returned Score errors separately.
  All seven current models pass this policy; that does not establish accuracy,
  calibration, vision support, new GPU execution or full release certification.
  Strict kernel alignment becomes optional research. Broader workload, lifecycle,
  soak and clean-host evidence remain separate operational/release work.
