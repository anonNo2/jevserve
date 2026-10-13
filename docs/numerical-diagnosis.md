# Numerical parity investigation

Status: **unresolved; no tolerance, production precision or backend changed**.
The initial analysis below reuses historical real-GPU evidence; subsequent
sections record new diagnostic GPU runs, none of which certify a model.
Historical inputs are SHA-256 bound in
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

Raw-logit, common-hidden-projection, normalized-layer, first-GDN and actual
full-attention captures below are complete. The next experiment replays
projection and rotary operations with fixed observed inputs. The previous 24-hour NeoHorse
soak failed after losing its project Docker daemon; its duration gate did not
pass. New source-image GPU checks completed separately. No production numerical
correction is claimed.

CPU source inspection of the pinned Transformers 5.10.4 / vLLM 0.29.0 code found
an important capture constraint: HF's `Qwen3_5DecoderLayer.forward` returns the
post-residual hidden tensor, while vLLM's inherited `Qwen3NextDecoderLayer.forward`
retains separate hidden and residual tensors and fuses addition into later norms.
Comparing those raw return objects as if they represented the same activation
would produce a misleading diagnosis. This structural difference is **not proof
of the numerical root cause**.

Capture matching normalized inputs to the token mixer and MLP first, plus
embedding and final-norm outputs. Record exact layer index/type, tensor shape,
dtype, token position and original residual components; never mutate production
activations to make the captures match. Start with one unpadded question at a time,
release HF before starting vLLM, and preserve the existing original-logit capture
consistency checks. Locate the first divergent matching boundary before testing
one kernel/arithmetic hypothesis at a time. A final-token-only comparison cannot
rule out earlier-token differences in causal attention/recurrent state.

Retain `atol=0.001`, `rtol=0.01` relative to the reference. Any proposed fix needs
all affected cases plus real container/SDK regression and a newly tested image.
An isolated counterfactual, synthetic label agreement or a relaxed threshold does
not pass that gate. The full numerical root cause remains open.

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
proven numerical cause. The subsequent experiment captured vLLM's final hidden states and projected
both engines' hidden states with the same candidate weights and arithmetic; its
result is below. The exact backbone/kernel cause remains unresolved.

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

### Layer and arithmetic evidence from the source-image acceptance session

The [layer probe](jpt-layer-diagnostic.md) captured all token positions for one
unpadded JPT-0.8B fixture. Embedding and layer-0 input normalization are exactly
equal; the first differing normalized boundary is layer-0 post-attention norm.
The [internal GDN probe](jpt-gdn-diagnostic.md) then found exact projected QKV,
A and B, but 95 unequal projected-Z elements. Post-convolution Q/K/V also differ.
The observed runtime dispatch is the Triton prefill backend with
`ChunkGatedDeltaRule.forward_native`. HF capture reproduced the frozen original
candidate logits exactly. Full activation dumps stay local; the reviewed
[aggregate evidence](../validation/reports/numerical-arithmetic-diagnosis.json)
contains comparisons and raw-report hashes.

[Controlled arithmetic trials](jpt-arithmetic-diagnostic.md) use a matched
unpatched baseline and verify execution/restoration. On the 13 questions, the
baseline has seven original-probability failures. Isolated residual rounding,
gated normalization, Q/K normalization, beta sigmoid and chunk-recurrence trials
have respectively four, seven, ten, seven and seven failures. Combining those
five factors still has nine failures. These counts compare all candidate
probabilities, including both Noul outcomes; they are not the public HTTP
request-level counts. They do not establish monotonic contributions: rounding
errors can reinforce or cancel one another.

The earlier residual-only report retained aggregate coverage but lacked
per-forward records. A fresh run under the current coverage contract confirmed
the same 4/13 original and 6/13 common-FP32-projection failures, with 48 residual
replacements per question and successful restoration. The legacy report remains
explicitly marked as having weaker coverage evidence.

The initial gated-normalization attempt was invalid because a packed path
bypassed the replacement. Its captures exactly matched the baseline; it is
excluded from valid arithmetic evidence. The repaired trial covers the packed
path and verifies calls for every question. The combined trial emitted an
`EngineDeadError` after explicit engine shutdown, after inference and restoration
had completed; the process exited zero and GPU memory was released. This remains
a separate teardown observation, not a passed production lifecycle assertion.

An eight-factor trial adds HF-style full convolution, standalone Z projection
and log-decay arithmetic. The first run was invalid: its diagnostic convolution
returned the wrong memory layout for the downstream Triton kernel. A real CPU
Torch regression reproduced the addressing error. The corrected implementation
preserves native output storage and rejects an incompatible feature stride;
independent review confirmed that contract. The invalid run is excluded.

The corrected eight-factor run has **5/13 original-probability failures** and
**2/13 common-FP32-projection failures** (maximum probability difference
0.008104384865228242). Maximum final-hidden error is 0.375. Every factor executed
for each question and was restored. This remains a failed numerical trial.

For the single layer fixture, the corrected combination makes every captured
normalized boundary from embedding through layer-3 input norm exactly equal
across all 97 tokens. The first remaining observed difference is layer-3
post-attention norm (42,968 unequal elements, maximum absolute error 0.0625).
That narrows the next investigation to the first full-attention block and its
residual/normalization interval. Equality of normalized boundaries does not
prove equality of every intermediate GDN or MLP tensor.

The subsequent [actual attention capture](jpt-attention-diagnostic.md) matched
HF's frozen logits exactly and retained full eight-factor runtime coverage.
Attention input, projected Q and gate are equal; projected K and V differ
(19,115 and 18,835 elements respectively). Post-RoPE Q also differs despite equal
projected Q, so K/V projection differences alone cannot explain all observed
boundaries. HF selected the SDPA interface, while vLLM used FlashAttention with
fused Q/K normalization and RoPE. This records dispatch, not a controlled
comparison of attention kernels with equal operands. See the
[redacted attention evidence](../validation/reports/jpt-attention-diagnosis.json).

The [fixed-input projection replay](jpt-attention-replay.md) used verified
checkpoint weights and the exactly shared observed input. Separate projections
reproduced HF's Q/K/V/gate byte-for-byte; a packed projection reproduced the
runtime's outputs byte-for-byte. This supports projection arithmetic as a source
of this fixture's K/V differences, while actual runtime weight storage equality
has not been captured. The observed cosine/sine operands also match after layout
alignment, leaving Q normalization/application arithmetic to isolate.

No production arithmetic or threshold was changed. The eight-factor combination
is insufficient for this suite. Loaded-weight verification, Q norm/RoPE isolation,
remaining attention differences and the unchanged default Open-Jev probability
gate are still required before a production correction can be certified.
