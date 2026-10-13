# Models and verification

JevServe uses **one generic runtime image with separately mounted weights**.
The catalog below selects a pinned adapter, checkpoint and execution profile;
it does not accept arbitrary chat models. Each container owns one model and one GPU.
Current API support is **text Choice, Noul and Score only**. No vision capability
is implied by a checkpoint's architecture or name.

JevServe 使用一个通用镜像，模型权重独立挂载。每个容器运行一个模型、使用一张 GPU。
下表区分“已接入”“真实推理测试”和“数值对照”，当前 API 仅支持文本，未实现图片输入。
所有模型均为开发预览；合成用例通过不代表准确率评测或生产认证。

## Current bounded-drift assessment

Following the decision to allow limited numerical drift, the initial project
policy `bounded-drift-v1` requires **absolute probability error ≤0.03** (3 percentage
points, not relative error) and **unchanged argmax for every distribution**.
Reassessment of the existing source-`13eed9b` GPU evidence passes both checks for
all seven profiles: **270 requests, 285 questions, zero argmax changes**. This is
CPU analysis of saved GPU results, not new GPU execution, an arithmetic fix or
certification for untested inputs. See the separately versioned
[bounded-drift report](../validation/reports/source-13eed9b-bounded-drift.json).

| Model | Recorded requests / questions | Maximum absolute probability error | Bounded policy |
|---|---:|---:|---|
| Open-Jev-2B | 110 / 110 | 0.02178034 | Passed |
| Open-Jev-9B | 110 / 110 | 0.01953158 | Passed |
| JPT-0.8B | 10 / 13 | 0.02731947 | Passed |
| JPT-4B | 10 / 13 | 0.02947989 | Passed |
| JPT-9B | 10 / 13 | 0.02507534 | Passed |
| NeoHorse-Jev-4B | 10 / 13 | 0.00000006326 | Passed |
| Kev-4B, configured FP32 | 10 / 13 | 0.00004988098 | Passed |

Historical strict tolerances and their failed results below are preserved.
Operator-level strict-parity research remains available, but it is no longer a
blocker for this corpus under the bounded policy. No serving arithmetic, model
precision, weights, candidate order or temperature changed. The certified registry
remains empty; `--allow-uncertified`, model licenses and all operational/release
gates still apply. The local image remains unpublished.

当前初始策略允许每项概率最多 0.03 的绝对误差，并要求所有分布的 argmax 不变；
7 个配置的既有 270 次请求、285 个问题均通过。这是对已保存 GPU 证据的重新评估，
不是新增 GPU 推理、算术修复或完整模型认证。历史严格失败保留；逐算子严格根因研究
转为可选，不再阻塞这批用例的新数值门禁。长稳、干净主机和发布门禁仍独立存在。

## Historical integration and strict comparison matrix

The table retains the original integration results. A new unpublished source
image (`13eed9b`) subsequently passed real inference/HTTP/SDK execution for all
seven profiles and offline, read-only Compose for Open-Jev-2B and NeoHorse.
Its frozen HTTP probability comparisons still failed: Open-Jev-2B **31/110**,
Open-Jev-9B **29/110**, and each JPT profile **5/10**; NeoHorse and Kev each passed
their 10-case synthetic suite. These are separately bound runs, not evidence that
changed failure counts establish a numerical improvement. See the
[source-image report](../validation/reports/source-13eed9b-image-gpu.json).

| Model | Backend / precision | Real GPU / container test | Historical strict comparison | Weight license |
|---|---|---|---|---|
| [Open-Jev-2B](https://huggingface.co/ZefanCai/Open-Jev-2B) | vLLM pooling, BF16 backbone + FP32 scalar head | Passed Docker, Compose, cold download and offline; common-image regression passed | **Failed:** 36/110 HTTP cases in the original suite | Apache-2.0 |
| [JPT-0.8B](https://huggingface.co/kirp/jpt-0.8b) | vLLM token readout, BF16 | 10 HTTP cases including 255 options; mounted offline Compose passed | **Failed:** 6/10; see [container report](../validation/reports/jpt-0.8b-container.json); no Choice label flips | CC-BY-NC-4.0 |
| [JPT-4B](https://huggingface.co/kirp/jpt-4b) | vLLM token readout, BF16 | 10 HTTP cases including 255 options and native/official SDK passed execution | **Failed:** 6/10 comparisons; no Choice label flips | CC-BY-NC-4.0 |
| [JPT-9B](https://huggingface.co/kirp/jpt-9b) | vLLM token readout, BF16 | 10 HTTP cases including 255 options and native/official SDK passed execution | **Failed:** 5/10 comparisons; no Choice label flips | CC-BY-NC-4.0 |
| [Open-Jev-9B](https://huggingface.co/ZefanCai/Open-Jev-9B) | vLLM pooling, BF16 backbone + FP32 scalar head | 110 native HTTP cases and official SDK passed execution | **Failed:** 27/110 comparisons; no label flips | Apache-2.0 |
| [NeoHorse-Jev-4B](https://huggingface.co/TokenRhythm/NeoHorse-Jev-4B) | Explicit Torch pointer backend, BF16 backbone + FP32 head | 10 HTTP cases, native/official SDK and mounted offline Compose passed | **Passed:** 10/10 synthetic comparisons; max error 6.33e-8 | Apache-2.0 |
| [Kev-4B](https://huggingface.co/jaredpalmer/kev-4b) | Explicit Torch pointer backend, FP32 | 10 HTTP cases and native/official SDK passed | **Passed:** 10/10 with documented FP32 loader correction | Apache-2.0 |

The certified registry remains empty. Models require `--allow-uncertified`.
Torch pointer execution is a selected model profile, never a fallback after a
vLLM failure. Active Torch cancellation terminates the GPU worker and requires
an explicit service restart; queued cancellation does not kill another request.

## Larger candidates on the current GPU

| Candidate | Verified observation | JevServe status |
|---|---|---|
| [autotrust/JEV-27B-VL](https://modelscope.cn/models/autotrust/JEV-27B-VL) | Published inventory ~56.05 GB; author recommends an 80 GB GPU. BF16 weights alone exceed this 48 GiB GPU. Custom decision bias/temperature and VL processing need their own adapter. | Not integrated or GPU-tested. No quantized/offloaded/vision profile is claimed. |
| [Xor 1.2, 35B-A3B](https://huggingface.co/juspay/xor/tree/release%2Fxor-1.2) | Published inventory ~70.24 GB; the 3B active-expert count does not make all BF16 weights fit in 48 GiB. Author serving uses a patched SGLang path and forward/reverse-option averaging. | Not integrated or GPU-tested. A smaller model is not silently substituted. |
| [Laya](https://huggingface.co/convaiinnovations/laya) | ModernBERT plus custom decision heads, with multiple variants | Investigated; no adapter or GPU result yet. |

An explicitly engineered CPU-offload or quantized profile might change the
memory requirement, but it needs separate protocol and probability tests. These
limitations describe the current implementation, not a claim that the hardware
can never run those models by other methods.

## Model-specific bounds

| Family | Choice candidates | Score levels | Context / work bounds |
|---|---|---|---|
| Open-Jev | 1–255 | 2–10 | 4,096 tokens per candidate; 65,536 summed input tokens |
| JPT | 2–255 | 2–255 | 4,095 input tokens per question plus one readout token; 65,536 summed tokens |
| NeoHorse / Kev | 1–255 | 2–255 | 2,048 state tokens; 4,096 per question row; 8,192 summed and padded tokens |

All profiles additionally bound questions, candidate work, queue length and
deadlines. See `/v1/models` for the served profile. Token templates, candidate
order and checkpoint temperature are preserved. JPT uses one masked candidate
readout per question; its checkpoint temperature is applied once. Softmax and
`confidence` are not calibrated accuracy. Score confidence follows each adapter
family's published definition; API usage counters describe this service's work,
not provider billing.

## Observed GPU allocation

Measured after the synthetic container tests on the 48 GiB RTX A6000:

| Profile | GPU memory used (MiB) |
|---|---:|
| JPT-0.8B | 24,688 |
| JPT-4B | 24,776 |
| JPT-9B | 25,333 (shared GPU) |
| Open-Jev-9B | 23,914 |
| NeoHorse-Jev-4B | 10,050 |
| Kev-4B FP32 | 18,790 |

These are observations, **not minimum VRAM**. vLLM reserves a cache pool under
its 0.5 memory budget, so the smaller JPT model does not imply a smaller observed
reservation. Torch profiles use an allocator cap instead. Peak usage depends on
request shape; these are not concurrent-load or capacity measurements. JPT-9B
ran with another process present (640 MiB total device usage before this test);
its row is total device usage, not isolated per-process usage. The reference and
container each used an explicit 0.5 GPU-memory budget.

## Pinned inventories

`prepare --model MODEL_ID` downloads to staging, checks every file's size and
SHA-256, then publishes atomically. `serve --model MODEL_ID --model-dir PATH
--offline` verifies mounted files without downloading. [Commands and directory
layout](quickstart.md#mounted-models).

| Model ID | Checkpoint revision | Temperature | Source download (decimal GB) |
|---|---|---|---|
| `ZefanCai/Open-Jev-2B` | `0c7aa498b1627be8da4acf34c863ff0ee0a92785` | 1.518796342858676 | 4.58 |
| `jaredpalmer/kev-4b` | `6cfce5c2fa4b4bd64026336ab649c5ca78857d52` | 2.406050072164233 | 9.50 |
| `TokenRhythm/NeoHorse-Jev-4B` | `434cb21d3a994a953d3ae5788405fcb2c4970554` | 1.0 | 9.15 |
| `ZefanCai/Open-Jev-9B` | `47e966881e489511c0c7f5633a9e1960a676a551` | 1.8969118766347646 | 19.36 |
| `kirp/jpt-0.8b` | `1431c0509bbc10772cd59964cc7af5835c8720d6` | 1.14 | 1.74 |
| `kirp/jpt-4b` | `78312f855b8bebf83ae7e9e8f05b5a0f73f519a9` | 1.036 | 9.11 |
| `kirp/jpt-9b` | `9fda2c7f123d36c4789c46c93737e80fc3a6c9fe` | 1.087 | 18.85 |

Full file hashes and pinned base-model dependencies are shipped in
[`src/jevserve/data`](../src/jevserve/data). Source download size is not minimum
VRAM. JPT weights are noncommercial under CC-BY-NC-4.0; the runtime's Apache-2.0
license does not replace any model license. Weights are downloaded separately
and are not redistributed in the Docker image.

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


## Reproduce the additional comparison suite

The [matrix harness](../validation/model_matrix.py) runs author reference and
container HTTP phases separately so they do not occupy the GPU together.
Use the pinned upstream source from the corresponding inventory, a verified
model directory and the same frozen tolerance in `compat/parity-profile.json`.
The suite includes Chinese text, structured state, candidate order, boundary
markers, multiple questions and 255-option Choice/Score requests. It is a small
synthetic compatibility suite, not a representative quality benchmark.

- [JPT author's token encoding replay](../validation/reports/jpt-encoding.json)
- [NeoHorse author's encoding and readout positions](../validation/reports/neohorse-encoding.json)
- [JPT-0.8B GPU reference](../validation/reports/jpt-0.8b-reference.json)
- [JPT-0.8B container probabilities](../validation/reports/jpt-0.8b-container.json)
- [JPT-4B container probabilities](../validation/reports/jpt-4b-container.json)
- [JPT-9B container probabilities](../validation/reports/jpt-9b-container.json)
- [NeoHorse-Jev-4B container probabilities](../validation/reports/neohorse-jev-4b-container.json)
- [Open-Jev-9B container comparison](../validation/reports/open-jev-9b-container.json)
- [Kev-4B FP32 reference and dtype audit](../validation/reports/kev-4b-reference.json)
- [Kev-4B FP32 container probabilities](../validation/reports/kev-4b-container.json)
- [Final-image Open-Jev-2B offline regression](../validation/reports/open-jev-2b-common-image.json)
- [Real GPU active timeout and memory release](../validation/reports/pointer-deadline.json)
- [Same-image mounted offline Compose](../validation/reports/multimodel-compose.json)

Record runtime, precision and revision with every result. A passing CPU suite,
a successful Docker build, a working GPU endpoint and probability parity are
separate gates. Full certification, clean-host installation and soak tests
remain outstanding. [Development status](development/STATUS.md).

Kev's FP32 reference uses the pinned author code with a local configuration view
setting both outer and text-config dtype to FP32. Transformers 5.10.4 otherwise
ignored the outer argument and loaded BF16. All loaded backbone parameter dtypes
are checked. Initial BF16 reports are retained but invalidated for this FP32
profile; see [D028](development/DECISIONS.md).

Further JPT-0.8B raw-logit/head/hidden diagnosis and public-runner JPT/NeoHorse
reproduction are documented in [numerical diagnosis](numerical-diagnosis.md) and
[validation](validation.md). These new execution reports do not change certification
status, precision, licenses or the supported text-only scope.
