# JPT fixed-input attention replay

`validation.jpt_attention_replay` replays layer 3 projection arithmetic using the
exact input captured by `jpt_attention_probe`. It does not start an engine or
modify production. The existing actual HF and eight-factor runtime reports must
link by SHA256, select the same original fixture, and have identical observed
attention input, projected Q and gate. Every encoded boundary is hash checked.

```sh
.local/engine-venv/bin/python -m validation.jpt_attention_replay \
  --model kirp/jpt-0.8b --model-dir /weights/jpt \
  --hf .local/numerical-acceptance/attention-hf.json \
  --runtime .local/numerical-acceptance/attention-runtime.json \
  --device cuda --output .local/numerical-acceptance/attention-replay.json
```

The caller must reserve the GPU first. `--device cpu` is available for a CPU
arithmetic comparison, but cannot establish that CUDA GEMMs reproduce the actual
GPU capture. Imports, `--help`, and default contract tests do not import GPU
libraries. Output must be a fresh path.

The prepared model package is fully verified against the pinned manifest before
loading the three original layer-3 projection tensors. Effective BF16 tensor
hashes and shapes are recorded. A separate `[1,tokens,hidden]` `F.linear` for Q/gate,
K and V is compared with one packed `[tokens,hidden]` `F.linear`. Both reconstructed
paths are compared against both observed captures. The report distinguishes
replayed tensors from observations and explicitly states that actual runtime
weights were not captured or proven equal. Exact reproduction of both paths
supports a projection arithmetic explanation for this fixture; any mismatch
leaves attribution open. This is not certification and changes no tolerance.

The report also compares actual HF cosine/sine operands with the duplicated
halves of actual runtime `[cos_half,sin_half]` cache rows. This operand comparison
does not replay or attribute Q/K normalization, RoPE application, or attention
backend arithmetic. Those require subsequent bounded experiments.

The measured CUDA replay on the original case 0, row 0 reproduced both observed
projection paths exactly: separate projections matched every captured HF Q,
gate, K and V element; the packed projection matched every captured runtime
element. Cross comparisons first differed at K. Together with the identical
captured projection input, this supports packed versus separate BF16 projection
arithmetic as the explanation for this fixture's observed K/V discrepancy.
Actual runtime weight storage was not independently captured; the replay uses
verified original checkpoint weights. The aligned actual cosine and sine values
also matched exactly, so unequal rotary operand values do not explain the Q
post-RoPE discrepancy in this fixture. Normalization and RoPE application
arithmetic remain unresolved, as do downstream attention effects. Strict
end-to-end numerical acceptance remains failed.

Public evidence: [fixed-input GPU replay](../validation/reports/jpt-attention-replay.json).
