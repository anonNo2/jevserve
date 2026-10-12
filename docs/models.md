# Model baseline and certification

No serving model is certified yet. `models/registry.yaml` is intentionally empty.
The discovery inventory in `models/locks/open-jev-2b.json` is not a production
serving manifest or proof of inference. Its status remains `pending`.

## Verified sources

- [Open-Jev-2B](https://huggingface.co/ZefanCai/Open-Jev-2B/tree/0c7aa498b1627be8da4acf34c863ff0ee0a92785): revision `0c7aa498b1627be8da4acf34c863ff0ee0a92785`.
- [Base Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B/tree/15852e8c16360a2fea060d615a32b45270f8a8fc): revision `15852e8c16360a2fea060d615a32b45270f8a8fc`.
- [Author implementation](https://github.com/Zefan-Cai/Open-Jev/tree/bd4118882f733574a3250a4b65fe4d884130c08b): MIT source, separate from the weights.
- [vllm-jev](https://github.com/mode-io/vllm-jev/tree/eae6b2d6dd06fb014118c99f660261cf4ecf04f9): Apache-2.0, declares vLLM 0.29.0.

The checkpoint and base licenses were read at the pinned revisions: Apache-2.0.
Keep their notices when redistributing. Source code is MIT. No model weights are
committed or publicly redistributed by this repository. The published training
mixture has redistribution caveats; reference inputs here are synthetic and do
not copy that dataset.

## Semantics that must be preserved

This checkpoint is LoRA plus a scalar head, requiring the separately pinned base
model. It is not a standalone generative checkpoint. The author compiles each
candidate independently, applies the tokenizer chat template with thinking off,
and reads the last non-padding hidden state through the FP32 scalar head.
The backbone defaults to BF16. Noul uses logits `[0, score]` in false/true order.
Choice and Score retain caller candidate order. Score is a zero-based ordinal
expectation. The saved temperature `1.518796342858676` is applied once before
softmax. Retaining that temperature does not establish calibration on user data;
JevServe's `calibration_id` remains null.

Author bounds are Choice 1–255, Score 2–10, 4096 tokens per candidate. Native
contracts deliberately separate structural validation from these model limits.
The native protocol restricts instructions/descriptions to text, rejects empty
state and unknown fields, and accepts at most 16 questions. Compatibility with an
official SDK remains untested; native schemas do not claim full System One parity.

## Candidate engine, not a tested runtime

`compat/engine-lock.json` records the actual registry linux/amd64 manifest digest,
upstream commit and declared versions. The container has not been started, so
observed runtime versions and a certified minimum driver remain unknown.
CUDA's [minor compatibility documentation](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
lists driver branch 580 for CUDA 13.x, with feature/PTX limitations. That table
alone does not certify this engine on any driver.

The author's Transformers 5.10.2/PEFT 0.19.1 environment is isolated from the
candidate vLLM environment (vLLM declares Transformers >=5.10.4 and Torch 2.13.0).
Do not upgrade the author environment to resolve engine dependencies. The
hash-pinned reference requirements are in `compat/reference-requirements.txt`.
See STATUS for whether installation and execution have actually passed.

## Reproduce the baseline

Use a project-local Python 3.12 environment. After `uv sync --locked`:

```bash
python -m validation.acquire --cache .local/baseline
# Prints checkpoint and base directories; pass those to the reference command.
python -m validation.acquire --cache .local/baseline --offline
```

Prepare a separate reference environment, clone the author repository, and
checkout the exact revision above. Install its inference requirements without
installing training/dataset extras:

```bash
uv venv --python 3.12 .local/reference-venv
uv pip sync --python .local/reference-venv/bin/python --require-hashes compat/reference-requirements.txt
.local/reference-venv/bin/python -m validation.reference \
  --upstream .local/upstream/Open-Jev \
  --checkpoint CHECKPOINT_DIRECTORY --base BASE_DIRECTORY \
  --cache .local/reference-cache \
  --output validation/reports/open-jev-reference.json
```

The runner verifies all inventory hashes and author commit before loading, uses
an isolated offline Hub snapshot, requires CUDA, and records tokens, masks,
logits and probabilities. No reference output is fabricated when dependencies or
weights are unavailable. `compat/parity-profile.json` freezes tolerances before
measurement. Empty-state author behavior is recorded separately from the stricter
native request contract. `python -m validation.locks` must fail until certification
and its report are complete.

## Encoding evidence

`validation/reports/open-jev-encoding.json` records 110 CPU cases using the pinned
real tokenizer and clean author source. The adapter matched every token sequence
and answer-key order. This does not execute the scalar head or certify inference.
The baseline lock binds the report digest. `validation.encoding` reproduces the
comparison in a tokenizer environment; ordinary CPU tests replay its token rows
without downloading models or importing a GPU library.

## Author GPU reference evidence

`validation/reports/open-jev-reference.json` contains 113 actual author GPU
executions on RTX A6000 with no failures. All source/weight hashes were checked
before offline loading. Observed Torch is 2.12.0+cu130; the ordinary author Torch
path ran without optional FLA/causal-conv1d kernels or prefix cache. Backbone/LoRA
parameters include BF16/FP32 and the scalar head is FP32. The saved temperature
was preserved. Replaying these logits through the native adapter matches the
author's probabilities and answer mapping. This does not certify vLLM or either
container deployment path; those gates remain pending.
