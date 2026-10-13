# JPT layer-boundary diagnostic

`validation.jpt_layer_probe` compares one explicitly selected synthetic question
from an existing verified HF logit report. It is a diagnostic subset, not model
certification. Run HF and runtime stages sequentially on an available GPU after
checking shared-host resources. Both stages verify the model manifest, upstream
commit, original report identity and pinned packages (Torch 2.13.0, Transformers
5.10.4; runtime additionally requires vLLM 0.29.0).

Both GPU stages have executed on one 97-token JPT-0.8B fixture (case 0, row 0).
Embedding and the first layer's input normalization matched across all 99,328
elements. The first observed unequal boundary was
`layer.0.post_attention_norm`: 72,999 elements differed, with maximum absolute
difference 0.125. This places the first observed discrepancy within the first
token-mixer/residual/normalization span; it does not identify a single causal
kernel or certify the model. The
[measured diagnostic report](../validation/reports/numerical-arithmetic-diagnosis.json)
contains the full boundary comparisons. A subsequent
[first-layer GDN capture](jpt-gdn-diagnostic.md) observes internal operations
within that span.

```bash
python -m validation.jpt_layer_probe hf \
  --model kirp/jpt-0.8b --model-dir /weights/jpt \
  --upstream /sources/llm2jev --reference hf-logits.json \
  --case-index 0 --row-index 0 --output hf-layers.json
python -m validation.jpt_layer_probe runtime \
  --model kirp/jpt-0.8b --model-dir /weights/jpt \
  --upstream /sources/llm2jev --reference hf-logits.json \
  --case-index 0 --row-index 0 --layer-reference hf-layers.json \
  --output runtime-layers.json
```

Use the exact profile identifier from the original report. Output paths must be
new. A fixture may contain at most 1,024 tokens. The probe captures embedding,
each decoder's input RMSNorm and post-attention RMSNorm outputs, and final
RMSNorm. It preserves all token rows as lossless little-endian BF16 bytes,
base64 encoded with SHA256, shape and dtype. It rejects missing, repeated,
reordered, nonfinite or differently shaped captures. JSON RPC uses named methods;
insecure serialization is forbidden. It copies activations without replacing
model outputs or changing production precision or tolerances.

vLLM fuses residual addition into RMSNorm and returns `(normalized, residual)`.
Only the normalized member matches the HF norm output. Raw decoder outputs are
not equivalent boundaries. The report compares every element and separately the
last token, identifying the first observed unequal boundary. This localizes the
first observed difference but does not prove which kernel caused it. Differences
within unobserved operations between boundaries remain possible.

The runtime keeps the existing hidden-probe profile: BF16, eager execution,
FP32 Mamba cache, disabled prefix caching, sequential single-question inference.
Captured candidate logits are checked against the runtime result; candidate head
weights are checked byte-for-byte against the HF capture. Hook overhead makes
these runs unsuitable for performance measurement. The fixture and activations
are diagnostic artifacts and should remain local; publish only reviewed summaries.
