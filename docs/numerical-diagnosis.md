# Numerical parity investigation

Status: **unresolved; no tolerance, production precision or backend changed**.
This analysis reuses historical real-GPU evidence. It is not a new GPU run or
certification. Inputs are SHA-256 bound in
[`numerical-diagnosis.json`](../validation/reports/numerical-diagnosis.json).

## Reproduce without a GPU

From the source checkout and CPU environment:

```bash
uv run python -m validation.numerics --output numerical-diagnosis.json
```

Exit zero means the analysis ran, not that parity passed. The tool checks report
identity, case IDs, candidate order and the frozen tolerance profile. For JPT,
Noul compares the public `noul` probability, preserving its original validator.
Open-Jev pooling retains its original two-element probability comparison; the
complementary-probability tolerance alone accounts for three of its failed cases
(`00-noul`, `13-noul`, `18-noul`). Neither historical comparison rule is changed.

## What the evidence establishes

For Open-Jev-2B, all 113 recorded reference and runtime probability vectors are
reconstructed exactly from their respective recorded logits and the checkpoint
temperature using the same stable softmax. The existing 38 failed cases remain.
The maximum difference between centered candidate logits is 0.1360604166984558.
Thus the recorded discrepancy already exists in the model/readout scores;
postprocessing these recorded scores does not explain it. This does **not** locate
the cause within the backbone, LoRA, head, kernels or batching.

The pooling runner feeds token IDs from the author reference directly, removing
independent tokenization from that particular comparison. This observation does
not certify every HTTP template or every model. See the separate encoding reports.

| JPT profile | Failed cases / 10 | Choice / Noul / Score failed questions | Choice label flips |
|---|---:|---|---:|
| 0.8B | 6 | 2 / 1 / 3 | 0 |
| 4B | 6 | 3 / 0 / 3 | 0 |
| 9B | 5 | 2 / 0 / 3 | 0 |

A case can contain multiple questions; these are question counts, not independent
case counts. The historical JPT reports contain answers but not raw logits or
intermediate hidden states. They establish a discrepancy but cannot establish a
BF16, masking or kernel root cause. Label agreement is not probability parity.

Mathematically, subtracting the same log-normalizer from every candidate cancels
in candidate softmax, including with a non-unit checkpoint temperature. Therefore
full-vocabulary versus candidate-only normalization **alone in exact arithmetic**
does not explain a discrepancy. Different rounding and preceding engine execution
still need measurement; this identity does not prove implementation equivalence.

## Next experiment and acceptance

On an available GPU, start with pinned JPT-0.8B and identical synthetic token IDs.
Capture author and runtime candidate IDs, raw logits, centered logits and final
probabilities. Compare single-question and batched execution under the same dtype,
then separately measure a diagnostic FP32 readout while preserving the production
profile. Record actual parameter/output dtypes, package versions, checkpoint
revision, device use and the immutable image. Free the reference engine before
starting the runtime. Do not change several factors in one experiment.

Retain `atol=0.001`, `rtol=0.01` relative to the reference. Any proposed fix needs
all affected cases plus real container/SDK regression and a newly tested image.
An isolated counterfactual, synthetic label agreement or a relaxed threshold does
not pass that gate. This experiment and the full numerical root cause remain open.

## 2026-10-13: real JPT-0.8B logits and fixed-hidden experiments

The [public diagnostic tools](jpt-diagnostic.md) ran against the pinned local
checkpoint and author code. Reports are under
[`validation/reports/jpt-diagnostic`](../validation/reports/jpt-diagnostic).
All 10 requests / 13 questions matched the author input token IDs and candidate
order. Captured HF logits reconstructed the author answers; the author rerun
matched the prior reference.

- Both sequential and within-request batched vLLM runs differ from HF; maximum
  probability error is 0.027319495161551854 and centered-logit error is 0.28125.
- Batched versus sequential execution itself changes one request beyond tolerance
  (maximum probability difference 0.0065495271359604645). Scheduling contributes
  to observed variation but cannot explain all HF/vLLM differences.
- For the same HF hidden state, original full-sequence BF16 projection,
  last-token/full-vocabulary BF16 and last-token/candidate-only BF16 are identical
  for all 13 questions. Candidate-only FP32 rounded back to BF16 is also identical.
  The tested HF head shapes therefore do not account for the discrepancy.
- Keeping FP32 projection outputs changes six questions beyond tolerance, with
  maximum probability change 0.009566225941695261 versus the original BF16 output.
  This is a diagnostic counterfactual, not a compatible production fix.

Diagnostic comparisons include both Noul outcomes and find seven failed request
cases versus HF; the public HTTP validator retains its original true-probability
rule and reports six. The thresholds and the public validator were not changed.
All recorded argmax comparisons in these probes remain unchanged.

All diagnostic requests completed and the process exited zero with GPU memory
released. vLLM logged an `EngineDeadError` during explicit shutdown; it was not a
failed request, but shutdown logging/behavior still needs separate investigation.
The HF fast-path-unavailable message identifies different implementations, not a
proven numerical cause. The next isolating experiment is to capture vLLM's final
hidden states and project both engines' hidden states with the same candidate
weights and arithmetic. The backbone/kernel contribution remains unresolved.

### Unified hidden-state projection result

The follow-up named-worker-RPC probe completed all 13 questions, without enabling
insecure function serialization or changing the original vLLM logits return value.
Ordered candidate BF16 weights matched byte-for-byte between engines (SHA-256,
shape and dtype checked per question). The captured vLLM head outputs also matched
the values actually returned by its processed-logit inference path.

All 13 final hidden vectors differ between HF and vLLM; maximum elementwise
absolute difference is **0.330078125**. Projecting both engines' hidden vectors
using matching candidate weights in FP32 with TF32/autocast disabled still fails
probability tolerance for **7/13 questions**, with maximum probability difference
**0.012281190571394651**.

This establishes a measurable **pre-head contribution** to the discrepancy in
this model/environment/sample set. It does not establish which backbone layer,
normalization, attention/recurrent kernel or accumulation path caused it, and it
does not rule out additional head arithmetic effects. Production precision,
backend and tolerances remain unchanged. Next work should compare corresponding
layer boundaries with the same token inputs and verified weights, rather than
replace BF16 outputs or relax the public parity gate.
