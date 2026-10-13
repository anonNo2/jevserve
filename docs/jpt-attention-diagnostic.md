# First full-attention layer diagnostic

`validation.jpt_attention_probe` compares actual operations at decoder layer 3
for original synthetic case 0, row 0. HF uses the original unmodified model.
The runtime uses the explicit eight-factor diagnostic arithmetic combination
`hf_torch_decay_combined`, whose preceding normalized layer boundaries matched
HF on this fixture. This does not establish equality of every unobserved
intermediate operation or certify a model. No attention arithmetic is replaced.

Run the stages sequentially after checking GPU availability:

```bash
python -m validation.jpt_attention_probe hf \
  --model kirp/jpt-0.8b --model-dir /weights/jpt \
  --upstream /sources/llm2jev --reference hf-logits.json \
  --output hf-attention.json
python -m validation.jpt_attention_probe runtime \
  --model kirp/jpt-0.8b --model-dir /weights/jpt \
  --upstream /sources/llm2jev --reference hf-logits.json \
  --attention-reference hf-attention.json --output runtime-attention.json
```

The reference must be the original frozen HF logit diagnostic. Model manifests,
upstream commit, package pins, fixture identity, candidate weight bytes and
reference hashes are checked. HF captured logits must exactly match the frozen
reference. The runtime uses the arithmetic diagnostic engine profile and requires
all eight factors to execute on its one question and be restored before a report
is published. Output paths must be new. CPU contracts are tested; GPU execution
has now completed for both stages on the fixed 97-token fixture.

The [measured attention report](../validation/reports/jpt-attention-diagnosis.json)
binds both raw reports and source hashes without publishing token IDs, positions,
or activation tensors. All 11 boundary comparisons were independently recomputed
from the hash-verified BF16 captures. HF logits matched the original frozen
reference exactly; each stage's logits and attention input also matched its
corresponding prior layer capture. Runtime coverage recorded all eight diagnostic
factors and successful restoration.

Attention input, projected Q, and gate matched exactly across all tokens.
The first unequal listed boundary was projected K: 19,115 of 49,664 elements
differed, with maximum absolute difference 0.03125. Projected V independently
differed in 18,835 elements (maximum 0.015625). Post-RoPE Q differed in 6,982 of
198,656 elements (maximum 0.03125), despite exact projected Q. These observations
identify separate intervals for investigation, not a single established cause.

HF selected the `sdpa_attention_forward` interface with no explicit mask and
causal attention enabled; its CUDA kernel was not profiled. vLLM used
`FLASH_ATTN` / `FlashAttentionImpl` with fused Q/K normalization and RoPE enabled.
These results apply to the eight-factor diagnostic runtime, not production
arithmetic, and do not establish probability parity or certification.

The probe copies the attention input, projected Q/K/V and gate, actual post-RoPE
Q/K/V, attention output before gating, gated output before output projection,
and the output projection result. Query/gate packing is split within each head.
HF attention-interface inputs are transposed from `[batch, heads, tokens, dim]`
into token-major order for comparison; KV heads retain their native count.

vLLM may fuse Q/K normalization and RoPE. The probe captures the fused operation's
actual outputs; it does not replay a normalization and label it as an observed
intermediate. Actual position IDs and rotary operands are recorded separately:
HF full cosine/sine arrays and vLLM selected cache rows `[cos_half, sin_half]`
retain their original representation and dtype. These auxiliary representations
are not automatically compared as identical layouts.

Metadata records HF's selected attention interface, source hash, causal flag,
mask shape/dtype/hash, and scaling; it does not claim which CUDA SDPA kernel was
selected. vLLM metadata records its backend, implementation class, and fused
normalization/RoPE flag. Causal mask infinities are hashed as bytes rather than
passed through finite-activation validation.

All compared activations use lossless BF16/FP32 bytes with shape, dtype and SHA256.
The report compares all tokens and the last token separately. The first unequal
listed boundary narrows the investigation; parallel projections and auxiliary
rotary operands require separate interpretation. Keep raw reports local because
they contain synthetic token IDs and full activations. No performance or
certification claim follows from this diagnostic.
