# First-layer JPT GDN diagnostic

`validation.jpt_gdn_probe` records actual internal operations in the first
linear-attention layer for one selected synthetic question. It helps locate a
numerical difference already observed between the layer's input and
post-attention normalization. It does not certify a model or change production
arithmetic. Both GPU capture stages have now executed on one 97-token
JPT-0.8B fixture (case 0, row 0). HF captured logits exactly matched the frozen
reference. The observed vLLM prefill backend was Triton (`forward_native`).

Projected QKV, B, and A matched exactly across every token. Projected Z differed
in 95 of 198,656 elements, with maximum absolute difference 0.03125. The
post-convolution Q, K, and V outputs also differed; post-convolution Q's maximum
absolute difference was 0.0625. These are separate observed differences, not
proof that one operation explains the final probability mismatch. See the
[measured diagnostic report](../validation/reports/numerical-arithmetic-diagnosis.json)
for all boundaries. This one-fixture result is not certification; arithmetic
counterfactual investigation remains separate.

Use the pinned diagnostic environment with Torch 2.13.0, Transformers 5.10.4,
and vLLM 0.29.0. Check GPU availability and run the stages sequentially.
The original HF logit report must match the verified model and upstream source.
Every output path must be new.

```bash
python -m validation.jpt_gdn_probe hf \
  --model kirp/jpt-0.8b --model-dir /weights/jpt \
  --upstream /sources/llm2jev --reference hf-logits.json \
  --case-index 0 --row-index 0 --output hf-gdn.json
python -m validation.jpt_gdn_probe runtime \
  --model kirp/jpt-0.8b --model-dir /weights/jpt \
  --upstream /sources/llm2jev --reference hf-logits.json \
  --case-index 0 --row-index 0 --gdn-reference hf-gdn.json \
  --output runtime-gdn.json
```

The selection must match across stages. The fixture must contain between 2 and
1,024 tokens. The runtime requires one fresh unpadded prefill, one tensor-parallel
worker, and the pinned Qwen3.5 packed execution path. HF must actually select the
Torch chunk implementation and Torch gated RMSNorm; an installed alternative
implementation is rejected rather than silently replaced.

Captured boundaries are:

| Boundary | Matching convention |
| --- | --- |
| Projected QKV, Z, B, A | HF separate projections versus unpacked vLLM fused projections |
| Post-convolution Q, K, V | Includes the convolution activation; Q/K aligned to value-head order |
| Normalized Q, K | Actual HF normalization calls and actual vLLM preparation outputs |
| G and beta | Original operand dtype preserved, including BF16 versus FP32 beta |
| Recurrence output | Before gated RMSNorm |
| Gated output | After gated RMSNorm, before output projection |
| Output projection | Token mixer result before residual addition |

HF already repeats Q/K to value-head order. The diagnostic repeats the captured
vLLM key heads solely for comparison. The runtime's packed path bypasses the
ordinary norm module, so capture wraps `_rms_norm_gated_cuda` directly. Its input
and output can alias: the recurrence is copied before the original operation
runs. Outputs and cache contents are otherwise unchanged.

Reports store all token rows as lossless little-endian BF16 or FP32 bytes,
base64 encoded with SHA256, dtype, and shape. Comparisons include every element
and the last token separately. They preserve differing operand dtypes instead of
requiring beta's storage dtype to match. Missing or repeated boundaries, changed
candidate weights, mismatched fixture identities, and nonfinite values fail the
run. RPC methods are named and insecure serialization is forbidden.

The first unequal boundary identifies the first observed difference in the
listed sequence; independent projections may diverge separately. It is not a
causal attribution. In particular, matching Q/K normalization and beta alone
does not align the recurrence: HF's Torch chunk implementation and vLLM's chunk
kernels use different intermediate precision and reduction operations. A
separate controlled arithmetic trial is needed to test a proposed cause.

These artifacts contain synthetic token IDs and full activations. Keep raw
reports local and publish only reviewed summaries. Capture overhead makes the
runs unsuitable for performance measurement. See also the
[layer-boundary diagnostic](jpt-layer-diagnostic.md).
