# Supported models and test results

**Currently supported: Open-Jev-2B only, as a development preview.** It has
passed real GPU serving, Docker, Compose, cold-start download and offline
execution checks. It has **not** passed numerical certification against the
author's reference implementation. “Tested” does not mean production-certified.

**目前仅支持 Open-Jev-2B（开发预览版）**：已通过真实 GPU 推理、Docker、Compose、
首次下载及离线运行测试；与作者实现的数值一致性尚未通过认证。

## Support matrix

| Model | JevServe support | Tested behavior | Certification |
|---|---|---|---|
| [`ZefanCai/Open-Jev-2B`](https://huggingface.co/ZefanCai/Open-Jev-2B) | Default model; text Choice, Noul and Score | Real NVIDIA GPU, native HTTP/client, official SDK text subset, Docker, Compose, cold download, offline, graceful stop | **Uncertified**: probability parity fails |
| `Qwen/Qwen3.5-2B` | Pinned backbone dependency of Open-Jev-2B | Loaded as part of the decision model above | Not a separately supported Jev endpoint |
| `convaiinnovations/laya` | Not implemented | Not tested by JevServe | Not supported |
| TypeSafe's official Jev weights / other checkpoints | Not included or accepted by the current model catalog | Not tested by JevServe | Not supported |

The model selector accepts only `ZefanCai/Open-Jev-2B`; there is no automatic
substitution or generic “any Hugging Face model” support. The certified registry
in `models/registry.yaml` remains empty. The locked development model remains
`pending` certification, even though container smoke gates have passed.

## What was actually tested

Observed on **Linux x86_64, RTX A6000 (48 GiB), NVIDIA driver 580.95.05**.
The serving image uses Python 3.12.3, vLLM 0.29.0, Torch 2.13.0+cu130,
Transformers 5.10.4 and PEFT 0.20.0. These observations do not establish a minimum
VRAM/driver requirement or support for every NVIDIA GPU.

| Check | Observed result | Evidence |
|---|---|---|
| Author's real GPU reference | 113 cases executed, 0 failures | [Reference report](../validation/reports/open-jev-reference.json) |
| Token encoding and answer order | 110 cases matched | [Encoding report](../validation/reports/open-jev-encoding.json) |
| vLLM probability parity | **38/113 failed** the frozen tolerance; no label flips observed | [Pooling report](../validation/reports/vllm-pooling-parity.json) |
| Native HTTP probability parity | **36/110 failed** the frozen tolerance; no label flips observed | [HTTP report](../validation/reports/native-http-parity.json) |
| Docker and Compose | Real Choice/Noul/Score responses; non-root service and graceful stop | [Container report](../validation/reports/container-smoke.json) |
| First startup and offline reuse | Empty-volume download/verification succeeded; real inference with network disabled and read-only root succeeded | [Container report](../validation/reports/container-smoke.json) |
| SDK compatibility | Native client and `typesafe-sdk==0.7.4` text-question subset exercised | [Supported API subset](../compat/README.md) |

Probabilities are **not calibrated accuracy**. Label agreement alone is not
numerical certification. A separate clean-host installation, image rollback
exercise and 24/72-hour soak are not complete. See [development status](development/STATUS.md)
for commands, scope, failures and remaining release gates.

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
state and unknown fields, and accepts at most 16 questions. The tested official SDK subset is documented in [API compatibility](../compat/README.md);
native schemas do not claim full System One parity.

## Pinned runtime and reference environment

`compat/engine-lock.json` records the pinned upstream linux/amd64 image digest,
source commit and observed serving-package versions. The tested JevServe image
adds the hash-pinned runtime dependencies and a non-root service account. Image
and container evidence live in [the container report](../validation/reports/container-smoke.json).
A certified minimum driver has not been established.

The author's Transformers 5.10.2/PEFT 0.19.1 environment remains isolated from
the vLLM serving environment. Do not upgrade the author environment to resolve
engine dependencies. Hash-pinned reference requirements are in
`compat/reference-requirements.txt`; serving requirements are in
`compat/runtime-requirements.txt`.

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
author's probabilities and answer mapping. The separately recorded Docker/Compose smoke checks now pass, but the failed
vLLM probability comparison still blocks numerical certification.
