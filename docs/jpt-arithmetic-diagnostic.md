# JPT arithmetic counterfactuals

`validation.jpt_arithmetic_probe` tests specific numerical differences between
pinned HF and vLLM Qwen3.5 implementations. It changes arithmetic only inside a
diagnostic worker, restores the patched methods before engine shutdown, and
leaves production code, checkpoint weights and probability tolerances unchanged.
A successful run is **not model certification**. These trials investigate the
existing [numerical discrepancy](numerical-diagnosis.md).

## Inputs and execution

Use the hash-pinned runtime environment: Torch 2.13.0, Transformers 5.10.4 and
vLLM 0.29.0. The tool verifies the prepared model package and clean pinned author
checkout. It requires two existing reports from the [JPT diagnostic workflow](jpt-diagnostic.md):

- `--reference`: the original HF candidate-logit report, mode `reference_logits`.
- `--hidden-reference`: the corresponding HF hidden report, mode `hf_hidden`,
  bound to the exact candidate-logit report by SHA-256.

Model identity, revision, temperature, source identity, input token IDs,
candidate order and ordered candidate-weight bytes must agree. These inputs
are synthetic matrix fixtures; do not replace them with private request logs.
No network download or production fallback is performed.

Inspect shared-server GPU and container use first. Run one process at a time
on an available GPU; the tool does not reserve the GPU. Each invocation loads
one engine and evaluates all reference questions sequentially. Prefix caching
and chunked prefill are disabled, `max_num_seqs=1`, and eager execution is used.
The worker checks that every forward is one fresh, unpadded prefill, with no
cached initial state or decode tokens. These engine arguments differ from the
older hidden probe, so run the **matched `baseline` mode** before interpreting
an arithmetic trial.

```bash
PYTHONPATH=src .local/engine-venv/bin/python -m validation.jpt_arithmetic_probe runtime \
  --arithmetic-mode baseline \
  --model kirp/jpt-0.8b \
  --model-dir /path/to/verified/jpt-0.8b \
  --upstream .local/upstream/llm2jev \
  --reference validation/reports/jpt-diagnostic/jpt-0.8b-hf.json \
  --hidden-reference validation/reports/jpt-diagnostic/jpt-0.8b-hidden-hf.json \
  --cache-dir .local/jpt-arithmetic-cache \
  --gpu-memory-utilization 0.5 \
  --timeout 120 \
  --output .local/jpt-arithmetic/baseline.json
```

Replace the model directory with your verified package path. The author
checkout must match the selected profile's pinned revision. Reuse the same
cache and engine arguments for every mode. Run the command again with one
mode from the table and a new output path. Existing output files or symlinks
are rejected. `--timeout` bounds each inference request; use an external
coordinator if a whole-process timeout is required.

## Modes

| Mode | Diagnostic change |
|---|---|
| `baseline` | No arithmetic replacement; matched engine arguments and capture guards |
| `hf_gated_norm` | Apply HF torch gated RMSNorm, including intermediate BF16 rounding, to vLLM recurrence output |
| `hf_residual_norm` | Materialize BF16 residual addition before the existing normalization |
| `hf_qk_l2norm` | Use HF torch BF16 q/k L2 normalization on vLLM convolution outputs |
| `hf_beta` | Evaluate beta sigmoid in BF16, then promote its result to FP32 for recurrence |
| `hf_chunk_core` | Use HF torch chunk recurrence with vLLM-prepared q/k, g and beta |
| `hf_torch_combined` | Combine the five changes above; this is a combined hypothesis, not an isolated-factor trial |
| `hf_conv_materialize` | Keep native convolution, materialize its BF16 output, then apply separate SiLU |
| `hf_conv_core` | Replace convolution output with HF BF16 grouped Conv1d, causal slicing and SiLU |
| `hf_torch_conv_combined` | Combine the original five changes with `hf_conv_materialize`; preserve the original five-factor mode unchanged |
| `hf_z_projection` | Recompute only packed Z output with a standalone BF16 linear projection and verified Z weights |
| `hf_torch_projection_combined` | Combine the original five changes with `hf_conv_core` and `hf_z_projection`; preserve the five- and six-factor modes unchanged |
| `hf_decay_g` | Recompute only GDN log-decay `g` with HF Torch exp and softplus on verified effective operands |
| `hf_torch_decay_combined` | Extend the seven-factor projection combination with `hf_decay_g`; preserve all earlier combinations unchanged |

The gated-norm trial covers both ordinary norm calls and the packed GDN path
that bypasses `norm.forward` and writes gated output in place. The chunk trial
repeats q/k heads into value-head order and transposes recurrent state between
vLLM `[N,H,V,K]` and HF `[N,H,K,V]` layouts. It requires fresh zero FP32 state and
one sequence; no decode, multi-sequence or cached-continuation support is implied.
Chunk q/k normalization remains disabled because preparation already performed
that operation. Optional vLLM output buffers retain their expected behavior.

Both convolution modes execute the native convolution exactly once to preserve
its cache updates. The materialization mode disables only its fused activation,
then applies SiLU to the returned BF16 tensor. The full convolution mode retains
native cache updates and replaces the returned activation with HF-style grouped
Conv1d and SiLU. That second mode tests the whole convolution operator, including
its reduction arithmetic and intermediate rounding; it does not isolate only
the SiLU boundary. Both require BF16 operands/cache and a valid fresh single
sequence, with no cached continuation or padded/null cache slot.

Replacement activations are copied into the native convolution output buffer.
Its unit channel stride is required by the downstream Triton preparation
kernel after transposition. Shape and dtype alone do not establish this
contract. An earlier eight-factor trial returned a contiguous `[C,T]` tensor,
violated that contract and is excluded from numerical evidence. A real CPU
Torch regression reproduces the wrong addressing and checks the corrected
layout; the corrected GPU trial uses a separate report.

Before a convolution or Z trial, the worker compares every loaded GDN Z weight
slice and convolution weight with the exact BF16 tensor in the verified
safetensors checkpoint. It checks shape, dtype and SHA-256 without converting
precision, and rejects missing, duplicate or mismatched tensors. Per-layer
fingerprints are retained in the report. This verifies loaded values against
checkpoint bytes; it is not a capture of a separate running HF model.

The Z trial requires unquantized, bias-free, noninterleaved QKVZ packing and
one tensor-parallel worker. It runs the native packed projection, preserves
Q/K/V, and replaces only Z using a standalone linear projection with HF-style
`[1,T,D]` input shape and the verified Z slice. It tests projection shape/kernel
arithmetic; it does not change weights or replace the entire packed projection.

The decay trial verifies every loaded GDN `A_log` and `dt_bias` value against
the checkpoint cast used by HF BF16 loading. It rejects loaded values that
would change when represented as BF16, including more precise FP32 values.
Only `g = -exp(A_log.float()) * softplus(a.float() + dt_bias)` is replaced; the
isolated mode leaves q/k/v and beta unchanged. Parameter fingerprints and
original loaded dtypes are recorded separately from effective BF16 values.

These substitutions specifically reproduce operations from the pinned **HF
torch path**. They do not establish that every HF installation uses that path,
or reproduce all HF operations. Other projections, full attention and engine operations remain possible
contributors; convolution is unchanged except in the explicitly named
convolution modes. The combined mode does
not convert the model to FP32 or constitute a compatible production correction.

## Interpreting reports

Final reports use `mode="runtime_arithmetic"`, identify `arithmetic_mode`, and
include `valid_trial`. A valid trial requires successful restoration and the
expected number of guarded forwards. Each non-baseline question must execute
a replacement; baseline requires zero replacement calls. Each combined mode also
requires every declared factor to execute for every question (five, six, seven or eight). Reports retain the
probe, HF and relevant vLLM implementation source hashes, package versions,
input-report hashes, explicit engine profile, patch counts and per-forward
execution counts. Earlier reports may require the separately recorded probe
source hash; do not infer missing provenance fields.

`valid_trial` records automated coverage checks, not proof that the diagnostic
implementation is correct. A subsequently discovered implementation defect
invalidates the affected trial even if those checks passed.

The original head output must equal the candidate logits returned through the
runtime inference path. Reports compare both those original logits and a
common candidate-only FP32 projection of the captured BF16 hidden vectors with
the HF hidden reference. The common projection disables TF32 and autocast only
for that projection. Original backbone and head computations retain their
configured arithmetic.

Exit zero means the diagnostic completed with valid coverage and restoration;
it does **not** mean probability parity passed. Inspect:

- `summary.max_hidden_abs_error` and `summary.fp32_failed_questions`.
- Each row's `original_vs_hf_original.probability_parity_passed` and
  `fp32_vs_hf_fp32.probability_parity_passed`.
- `arithmetic.calls_per_forward`, `arithmetic.factor_calls_per_forward` and
  `arithmetic.removed`.

The frozen comparison remains `atol=0.001`, `rtol=0.01` relative to HF.
Compare every trial with the matched baseline and retain failed cases.
Improvement can include cancellation between errors; deterioration does not
show an operation was equivalent. Labels agreeing or fewer failures does not
pass the original gate.

Intermediate hidden reports are staged privately. If inference completes but
coverage/restoration validation fails, the final report is explicitly marked
`valid_trial=false` and the process exits nonzero. Earlier execution failures
may leave no final report. Never use an invalid trial as arithmetic evidence.

CPU guard and capture-contract checks:

```bash
.venv/bin/python -m pytest -q \
  tests/unit/test_jpt_arithmetic_probe.py \
  tests/unit/test_jpt_hidden_probe.py
```

GPU results and release gates are recorded separately in
[STATUS](development/STATUS.md). This document describes the diagnostic
interface; it does not declare numerical parity, container certification or a
new image passed.
