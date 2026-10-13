---
name: Bug report / 问题报告
about: Report a reproducible runtime, API, preparation or documentation problem.
title: ""
labels: ""
assignees: ""
---

<!-- Security concerns: read SECURITY.md before posting. This issue is public.
Remove tokens, private prompts, host/user names, addresses and local absolute paths.
可用中文填写。请删除凭据、私有输入和主机信息。 -->

### What happened / 实际与预期行为

Describe expected and observed behavior. For numerical differences, distinguish
probability mismatch, label change and execution failure. All current models are
development profiles; see docs/models.md for known parity failures.

### Reproduction / 复现

Provide minimal commands and a synthetic request, without credentials or private
data. Include CLI flags and relevant sanitized configuration. Do not attach full
environment dumps, model weights or unreviewed logs.

### Version and environment / 版本环境

- JevServe Git revision and whether locally modified:
- Image repository digest (a tag alone can move):
- Model ID, checkpoint revision and adapter/profile:
- Backend and actual precision, if known:
- Linux architecture, GPU model/VRAM, driver:
- Docker and Compose versions; `docker run`, Compose or source launch:
- Mounted/prepared weights, cold/cached startup and online/offline mode:
- Relevant sanitized `JEV_*` settings and `--allow-uncertified` usage:
- Other GPU workloads present (no process/user identifiers):

### Evidence / 证据

Include sanitized error code/status, smallest relevant log excerpt, exact check
commands and their exit codes. If reporting parity, include reference provenance,
frozen tolerance, probability error/label flips and reviewed report links. State
what was not run. A successful build or matching label does not prove parity.
