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
