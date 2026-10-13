# JPT candidate-logit diagnosis

This offline tool investigates the frozen probability-parity failures for the
pinned `kirp/jpt-0.8b`, `kirp/jpt-4b`, and `kirp/jpt-9b` text profiles. It does
not certify a model or change production precision, temperature, or tolerance.
Run from the repository root in the hash-pinned runtime Python environment.
The two stages must run serially on an available GPU; check shared-server GPU
and container use first. The tool itself does not reserve the GPU.

All three diagnostic CLIs require a new `--output` path. An existing file,
directory, or symlink is rejected before model loading, preserving prior
evidence. For a rerun, use a fresh name such as `jpt-0.8b-hf-run02.json` and
update downstream `--reference` / `--hidden-reference` arguments to that exact
new report. A failed run does not write a completed report; do not substitute
an older report for a missing result.

The reference stage verifies all model bytes and the clean, pinned author Git
revision, then uses the author's `HF.score`. A forward hook copies only the
requested candidate logits from that same forward pass, before the author's
FP32 full-vocabulary log-softmax. It checks that these logits reconstruct the
author probabilities within `1e-6` (a capture-consistency check, not a new
parity tolerance), and that the rerun matches the existing author reference
report under the frozen `compat/parity-profile.json` probability tolerances.
Author prompt IDs, candidate IDs and candidate order must also match the
production JPT adapter exactly. All synthetic matrix cases are required.

The runtime stage verifies the model again and checks the reference identity,
revision, temperature, fixture requests, and production encoding. It uses the
production `VllmTokenRuntime.infer` sampling and cancellation path. Its diagnostic
engine changes `logprobs_mode` to `processed_logits`; candidate masking,
sampling temperature 1, no top-k/top-p filtering, prefix caching off, BF16
model dtype, FP32 Mamba state cache, and eager execution remain as in production.
At these sampling settings, allowed candidate values retain their original
logits; only non-candidates are masked. The pinned vLLM sampler converts these
values to FP32. This conversion is recorded separately from model dtype and
does not establish the dtype of the preceding head calculation.

## Reproduce

Use a verified model package directory from `jevserve prepare` or the public
validation runner, and a checkout of `tic-top/llm2jev` at
`2b252d504972764211ef172c1155ac0fedc9c3de`. Replace the model directory below with
your actual verified package path. The supplied reference must be a
`validation.model_matrix reference` report for the same checkpoint; an HTTP
report is not accepted. No network download or dtype fallback is performed.

```bash
PYTHONPATH=src .local/engine-venv/bin/python -m validation.jpt_diagnostic reference \
  --model kirp/jpt-0.8b \
  --model-dir /path/to/verified/jpt-0.8b \
  --upstream .local/upstream/llm2jev \
  --reference validation/reports/jpt-0.8b-reference.json \
  --output .local/jpt-diagnostic/jpt-0.8b-hf.json

PYTHONPATH=src .local/engine-venv/bin/python -m validation.jpt_diagnostic runtime \
  --model kirp/jpt-0.8b \
  --model-dir /path/to/verified/jpt-0.8b \
  --upstream .local/upstream/llm2jev \
  --reference .local/jpt-diagnostic/jpt-0.8b-hf.json \
  --schedule both \
  --output .local/jpt-diagnostic/jpt-0.8b-runtime.json
```

Repeat with `kirp/jpt-4b` or `kirp/jpt-9b`, the matching package and existing
reference report. Both stages require Torch `2.13.0` and Transformers `5.10.4`;
the runtime also enforces vLLM `0.29.0` through the production startup guard.
The runtime defaults to GPU memory utilization `0.5`, no CPU offload, and a
120-second inference deadline per request or sequential question. Explicit
`--gpu-memory-utilization`, `--cpu-offload-gb`, and `--timeout` options are
available; offload and memory budget are recorded. Startup and HF forwards do
not have an in-process timeout; an external coordinator should bound whole
process execution. Runtime timeout propagates cancellation and engine cleanup.

`--schedule sequential` awaits each question before submitting the next.
`--schedule batch` submits the questions of each original request concurrently;
the vLLM scheduler decides actual microbatch membership. `--schedule both`
runs sequential then concurrent submission for each case on the same engine,
and records both against HF plus batch against sequential. This is a
submission-pattern comparison, not proof that every question shared one kernel
batch. HF uses one unpadded prompt at a time in all cases.

Reports retain raw and max-centered candidate logits, ordered probabilities,
checkpoint temperature, package versions, model/source identity, input IDs,
candidate IDs and the SHA-256 of the input reference report. They record maximum
raw-logit, centered-logit and probability differences, probability parity, and
argmax changes per question. A common logit offset changes raw differences but
not centered differences or conditional probabilities. A failure establishes a
numerical difference; it does not by itself attribute that difference to BF16,
head arithmetic, attention kernels, or scheduling. The two-stage tool does not perform FP32-head counterfactuals; the separate
head probe below provides that experiment.

A completed diagnostic command exits zero even when probability comparisons
fail: inspect `cases[].comparisons.*[].probability_parity_passed`. Source,
encoding, reference-reproduction or execution errors fail the command and do
not produce a completed report. Runtime `inputs` preserve the exact question
and candidate ordering; reference raw outputs are linked by report hash.
Outputs contain only the repository's synthetic matrix, and omit local model
paths and host identities. Review reports before publishing.

## CPU verification

```bash
.venv/bin/pytest tests/unit/test_jpt_diagnostic.py -q
.venv/bin/ruff check validation/jpt_diagnostic.py tests/unit/test_jpt_diagnostic.py
```

The CPU tests cover stable softmax, temperature, shift invariance, frozen
probability thresholds, candidate-order and provenance guards, invalid numeric
values, and GPU-free imports/help. They do not substitute for either real GPU
stage; actual execution evidence belongs in development STATUS and the report
artifacts.

## Same-hidden head counterfactual

`validation.jpt_head_probe` is a separate GPU process. Run it after both stages
above have exited. It verifies model bytes, author source, the HF report, and the
vLLM report's exact HF-report hash plus token/candidate order. It replays all
synthetic questions using the saved token IDs through the author-loaded model,
with the same unpadded forward signature as `HF.score`. Pre/post hooks on the
output head capture the last hidden vector and original candidate logits from
the same full-sequence forward. It then runs only these head projections:

| Name | Calculation |
|---|---|
| `A_original` | Original full-sequence, full-vocabulary BF16 head; select final candidates |
| `B_last_full_bf16` | Same final hidden, full-vocabulary BF16 head |
| `C_last_candidates_bf16` | Same final hidden, candidate-only BF16 head |
| `D_last_candidates_fp32` | Same BF16 hidden and candidate weights cast to FP32; TF32/autocast disabled |
| `D_round_bf16` | D rounded to BF16, then converted to FP32 for reporting |

```bash
PYTHONPATH=src .local/engine-venv/bin/python -m validation.jpt_head_probe \
  --model kirp/jpt-0.8b \
  --model-dir /path/to/verified/jpt-0.8b \
  --upstream .local/upstream/llm2jev \
  --reference .local/jpt-diagnostic/jpt-0.8b-hf.json \
  --runtime .local/jpt-diagnostic/jpt-0.8b-runtime.json \
  --output .local/jpt-diagnostic/jpt-0.8b-head.json

.venv/bin/pytest tests/unit/test_jpt_head_probe.py -q
```

This does not modify model parameters or production execution. TF32 is disabled
only for D and the prior setting is restored before the next original forward.
FP32 here means arithmetic on the existing BF16 values, not recovered FP32
checkpoint weights. No full-vocabulary FP32 weight copy is allocated. Reports
include original head shape/stride, captured hidden dtype, candidate projection
shape, original matmul settings, report hashes, per-question comparisons, and
aggregate question failure/error counts. The existing frozen tolerances are
used without changes. Comparisons against vLLM use each HF-side variant as the
reference denominator for relative tolerance.

A-versus-B isolates head input-shape sensitivity at fixed hidden. B-versus-C
checks whether candidate-only projection introduces its own shape sensitivity.
D-versus-rounded-D measures output rounding of that FP32 projection. Differences
between A/B/C and rounded-D also include arithmetic/kernel effects. A variant
being closer to vLLM is evidence of a possible contribution, not proof that the
HF/vLLM backbone hidden states agree; errors can cancel. A repeated original
forward differing from the saved HF reference is recorded under
`A_vs_hf_reference`, not silently discarded or called a passed reproduction.
The tool still finishes the counterfactual comparisons and exits zero on
numerical differences; source/input/execution failures exit nonzero. Check
`summary` before interpreting the result. As with the HF reference stage, use
an external coordinator to enforce a whole-process timeout.

## HF/vLLM hidden comparison

`validation.jpt_hidden_probe` runs HF and vLLM in separate processes, using the
same 10 synthetic requests and all their questions, submitted one at a time.
The HF stage records the final BF16 head input, exact ordered candidate weight
bytes hash, original logits, and a candidate-only FP32 projection. The runtime
stage requires that HF report and the original HF logit diagnostic report.
Run the HF command to completion before starting the runtime command:

```bash
PYTHONPATH=src .local/engine-venv/bin/python -m validation.jpt_hidden_probe hf \
  --model kirp/jpt-0.8b \
  --model-dir /path/to/verified/jpt-0.8b \
  --upstream .local/upstream/llm2jev \
  --reference .local/jpt-diagnostic/jpt-0.8b-hf.json \
  --output .local/jpt-diagnostic/jpt-0.8b-hidden-hf.json

PYTHONPATH=src .local/engine-venv/bin/python -m validation.jpt_hidden_probe runtime \
  --model kirp/jpt-0.8b \
  --model-dir /path/to/verified/jpt-0.8b \
  --upstream .local/upstream/llm2jev \
  --reference .local/jpt-diagnostic/jpt-0.8b-hf.json \
  --hidden-reference .local/jpt-diagnostic/jpt-0.8b-hidden-hf.json \
  --output .local/jpt-diagnostic/jpt-0.8b-hidden-runtime.json

.venv/bin/pytest tests/unit/test_jpt_hidden_probe.py -q
```

The dedicated engine loads `HiddenProbeWorker` via `worker_extension_cls`. Named
RPC methods install, arm, drain, and remove a wrapper around `compute_logits`;
all RPC arguments/results are plain strings, lists, numbers, and dictionaries.
The tool refuses an enabled `VLLM_ALLOW_INSECURE_SERIALIZATION`. It does not
expose a network service or change production code. The original head method
runs normally and its return value is preserved. Capturing is armed only after
startup, for exactly one question, and requires exactly one worker and one
sampled hidden row. Captured original candidate logits must equal the returned
`processed_logits` values exactly. Unexpected extra head calls fail the run
rather than silently associating the wrong hidden vector with a question.

Before each request, candidate weights must match the HF report's SHA-256 over
contiguous BF16 bytes in candidate order. Shape and dtype are checked separately.
Both projections use `[1,H]` hidden and `[K,H]` candidate weights converted from
BF16 to FP32, with TF32/autocast disabled and the prior TF32 setting restored.
The runtime projection occurs only after the original request completes.
Reports include hidden vectors, exact inequality counts, maximum absolute and
L2 differences, common-projection probability comparisons, and report hashes.
The same full-vector parity rule applies, including both Noul candidates;
these counts are not interchangeable with HTTP Noul-true-only checks.

A hidden-vector difference proves the engine inputs to the head differ. With
identical candidate weights and projection settings, differing FP32 projections
show how this difference affects the candidate readout. This localizes a
contribution to before the head, but cannot identify a particular attention,
recurrent, normalization, or other layer. It does not exclude an additional
head contribution. Projections can agree despite hidden differences in
unmeasured directions. Check HF `original_vs_reference` reproduction results
before interpreting the comparison. Numerical differences produce a completed
report, not a certification pass; execution/provenance/capture errors fail.
No hidden vectors or weight tensors from private user requests are accepted:
inputs are bound to the existing repository synthetic fixture.

Runtime defaults retain the production BF16 configuration, no CPU offload,
`0.5` GPU memory budget, and 120-second per-question inference deadline. The
memory budget and inference deadline can be set explicitly; each RPC has a
30-second timeout. Whole-process/startup/HF deadlines remain the responsibility
of the external coordinator. Hooks/wrappers are removed on normal or handled
failure paths, and the engine is closed. This probe measures numerical behavior;
its extra copies and RPC calls are unsuitable for latency benchmarking.
