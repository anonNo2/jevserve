# Cached image upgrade and rollback drill

`validation.rollback` checks **old → new → old** container execution for the
default `ZefanCai/Open-Jev-2B` model. It runs one container at a time on GPU 0,
uses an existing named model/cache volume, and stops/removes each test container
before the next stage. It does not upgrade a production service or leave a
replacement service running after the drill.

这是固定镜像间的缓存升级与回滚演练。每阶段均等待真实服务就绪并发送
Choice/Noul/Score 请求；跨版本概率差仅用于观察，不代表作者实现数值对照通过，
也不构成模型准确率、长稳或干净主机认证。

## Preconditions

Use a repository checkout and its Python 3.12 CPU environment (`uv sync --locked`).
The host needs Docker, NVIDIA Container Toolkit and an available GPU 0. Both
pinned images must already exist on the explicitly selected **local Unix Docker
socket**. The tool never pulls images, downloads weights or switches Docker
contexts. The existing cache volume must contain a complete verified default
Open-Jev-2B cache prepared by an earlier successful startup.

Inspect GPU occupancy and the selected Docker daemon before starting. Stop your
own service using this GPU/cache for a maintenance window; do not stop unrelated
workloads. The tool checks for compute processes before each stage and refuses
an already-running container that uses the selected volume. It also holds the
project's validation GPU lock. These checks coordinate this project's validators;
they cannot reserve the device against another user's later launch.

The input cache is writable and is reused through all three versions. Compilation
caches may change; the tool does **not** snapshot or restore cache bytes. Use a
separately prepared validation volume if you need to preserve a production cache
exactly. No volume is created or deleted by the tool, and a missing volume fails
preflight instead of causing an implicit empty-cache download.

## Run the published development images

Replace the Docker socket and existing volume name with the ones you have
inspected. Choose an unused loopback port and a new output directory:

```bash
.venv/bin/python -m validation.rollback \
  --docker-host unix:///var/run/docker.sock \
  --from-image anonwx/jevserve@sha256:f27c15c1da40d0c02139672a426d41d5370e6ea061d209b1d3fc5aef53b73396 \
  --to-image anonwx/jevserve@sha256:4b758ad1000f4d38501fd83f0a9c2b32f0cb91968d427b170a2c2b0d9a613aff \
  --cache-volume jevserve-cache \
  --port 18794 \
  --output-dir .local/rollback-dev0-dev1
```

These are the published dev0 and dev1 digests, respectively; no mutable tag is
accepted. Different manifest digests must also resolve to different local image
IDs; otherwise preflight fails with `identical_image_ids`. Do not substitute a
local image ID for a registry digest. For a rootless
or project-local daemon, pass its actual Unix socket explicitly. The tool uses
`--pull=never`, read-only container roots, writable temporary storage and loopback
port publishing. Serving arguments shared by both versions are:

```text
serve --allow-uncertified --offline --host 0.0.0.0 --cache-dir /cache
```

There is no `--model-dir` option in this drill because dev0 does not support it.
There is also no arbitrary model selector: every stage must report the pinned
Open-Jev-2B model identity. Offline here means model acquisition is disabled;
the container still has networking for the local HTTP probe. The tool does not
configure authentication, expose a public port, or accept user payloads.

`--timeout` bounds each readiness check (default/max 1800 seconds); inference
requests use a 35-second client timeout. Decision requests are not retried.
If a prerequisite or stage fails, the remaining stages are not started.

## Evidence and failure handling

The new output directory contains:

- `summary.json`: source revision/dirty status, request hash, selected model,
  per-stage image digest and observed image ID, exact project container name,
  GPU idle preflight, startup-to-readiness elapsed time, readiness return code,
  cleanup return codes and final container exit code.
- `old_before-sample.json`, `new-sample.json`, `old_after-sample.json`: complete
  actual native API responses from the fixed public synthetic
  [`examples/decision.json`](../examples/decision.json), each hash-bound by the
  summary. Failed stages may have no sample.
- Descriptive old-to-new and old-before-to-old-after comparisons: maximum
  absolute probability difference, changed Choice selections and Score deltas.
  These have no pass/fail tolerance and explicitly mark author parity unmeasured.

A passed report means all three stages returned valid matching protocol responses,
ran the requested images and shut down with container exit code 0. It does not
assert that probabilities were identical or correct. Startup timing includes
Docker launch through readiness; this is a warm-cache observation, not cold
installation performance. Shared cache bytes are preserved as a volume, not
rolled back transactionally. Model license and numerical limitations remain in
[the model matrix](models.md).

The command exits **0** only if all three stages and their cleanup passed,
**1** on execution failure/interruption, and **2** on invalid configuration or
output setup failure. Reports are updated after each stage and on failure.
Errors use stable codes rather than subprocess output, endpoints or host paths.
The raw responses contain only the fixed synthetic example's labels and model
metadata; still review artifacts before publishing them.

SIGTERM/Ctrl-C interrupts the current operation, then cleanup checks that the
exact unique container's ownership label matches before stopping/removing it.
Repeated signals are ignored while that cleanup runs. Cleanup never removes
input volumes, restarts Docker, changes drivers, or stops unrelated containers.
If cleanup fails, the report fails too; inspect the recorded exact container name
on the selected daemon before manually completing cleanup. A failed upgrade does
not automatically restart a separate production instance.

CPU checks (no Docker, network or GPU):

```bash
.venv/bin/pytest tests/unit/test_rollback.py -q
```

A script and green CPU tests alone are not execution evidence. Record the actual
real-GPU drill in [STATUS](development/STATUS.md) before marking rollback tested.

## Observed dev0 → dev1 → dev0 drill

On 2026-10-13, the public tool completed all three default Open-Jev-2B stages
against the immutable published digests, using the same existing project cache.
Start through readiness/check took **130.699 s**, **108.293 s**, and **98.579 s**.
Every stage completed the real sample and stopped with container exit code **0**;
the input cache volume remained present. The sample's probabilities and choices
were identical in the old/new/old comparisons. This one sample does not establish
full cross-version numerical equivalence or author parity.

[Summary and original samples](../validation/reports/rollback-dev0-dev1).
This is a same-host cached image-version rollback, not a clean-host install,
a rollback of every model profile, or a cache-format migration guarantee.
