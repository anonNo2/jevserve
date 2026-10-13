---
name: Model integration request / 模型接入请求
about: Propose a pinned Jev-style checkpoint with reference semantics and license evidence.
title: ""
labels: ""
assignees: ""
---

<!-- Read docs/contributing-models.md and docs/models.md first. 可用中文填写。
Do not attach weights, credentials or private datasets. Unknown facts may be
marked unknown; a request does not imply support or certification. -->

### Checkpoint and provenance / 模型来源

- Public checkpoint URL and full immutable revision, if available:
- Base-model/tokenizer dependencies and revisions:
- Author implementation URL and revision:
- Checkpoint, base-model and code licenses; restrictions/evidence links:
- Access or download restrictions (without credentials):

### Decision semantics / 决策语义

- Intended use and Choice/Noul/Score types needed:
- Decision head/readout, template, masks and checkpoint temperature:
- Candidate/context limits and supported languages, with sources:
- Text support requested:
- Vision support requested (separate preprocessing/reference evidence required):

### Reference and resource evidence / 参考与资源

- Runnable author reference and known dependency/precision requirements:
- Available GPU/VRAM for validation; measured needs vs estimates:
- Existing integration work or reproducible synthetic fixtures:
- What remains unknown or untested:

### Contribution scope / 贡献范围

Describe what you can contribute: provenance, adapter, CPU tests, real GPU
reference, Docker/Compose checks or documentation. Follow the acceptance checklist
in docs/contributing-models.md. One generic image uses external weights; there is
no automatic fallback, silent quantization or certification from label agreement.
