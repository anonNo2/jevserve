# Codex 开发交接

更新：2026-10-13。JevServe 已实现 CLI、HTTP/SDK、模型准备、真实推理和 Docker/Compose 部署，当前交付为开发预览。Open-Jev-2B/9B、JPT-0.8B/4B/9B、NeoHorse-Jev-4B、Kev-4B 共 7 个固定模型配置均有真实 GPU/容器执行证据。对已有 GPU 证据的重新评估显示，7 个配置的 270 次请求、285 个问题均通过初始 `bounded-drift-v1`；Open-Jev/JPT 的历史严格概率对照仍失败且报告保留。尚无 alpha、稳定版或完整模型认证。

源码已公开于 [GitHub](https://github.com/anonNo2/jevserve)。通用镜像 `anonwx/jevserve:0.1.0-dev1` 已发布，其固定 digest 为 `sha256:4b758ad1000f4d38501fd83f0a9c2b32f0cb91968d427b170a2c2b0d9a613aff`；匿名拉取已通过，但复用了缓存层，不能视为干净主机安装。部署步骤见 [Quick Start](../quickstart.md)，模型许可证、能力与逐项证据见 [models.md](../models.md)。尚未发布 Python 包。

## 本轮审查改进（2026-10-13）

源码已加入有界异步编码、脱敏诊断、客户端/server 依赖拆分、静态 API 契约及安装/Compose CI。
已发布 dev1 镜像不包含这些改动；新的源码构建默认需要显式 `serve --allow-uncertified`。
新的本地 `13eed9b` 镜像已通过 7 个模型的真实作者对照执行、Docker HTTP/SDK 与清理，以及 Open-Jev-2B、NeoHorse 的断网只读 Compose 检查。安装后的 41 个源码/模型清单文件与该 commit 和镜像内 wheel 一致。本地 Docker ID 为 `sha256:369538a76f4dd2fb7b0e00bd02b580196ceba3d9659ed0dadaeee6e948ae0193`，不是已发布 registry digest；dev1 未更新。见 [CPU/源码绑定](../../validation/reports/source-13eed9b-image-cpu.json) 和 [GPU/Compose 汇总](../../validation/reports/source-13eed9b-image-gpu.json)。

用户最新允许一定数值漂移，本轮选用初始 `bounded-drift-v1`：每项概率绝对误差 ≤0.03（3 个百分点）且所有分布的 argmax 不变。具体 0.03 是本轮初始策略，不应描述为用户逐字指定的阈值。270 次已有 GPU 请求、285 个问题全部满足，最大概率误差约 0.02947989、argmax 变化为 0。见[独立重新评估报告](../../validation/reports/source-13eed9b-bounded-drift.json)。这是 CPU 重评已有 GPU 证据，不是新 GPU 运行或算术修复；旧严格报告不改写，有限用例通过不扩大为未测输入、准确率或完整认证结论。

严格 HTTP 概率对照仍失败：Open-Jev-2B 为 31/110，Open-Jev-9B 为 29/110，三个 JPT 配置各为 5/10；NeoHorse/Kev 仅在各 10 个合成用例通过。本轮层级、GDN、算术、注意力捕获及固定输入重放已完成；投影形状差异已被重放复现，但融合 Q 归一化/RoPE 等剩余差异尚未闭合，没有生产算术修正。最新完整 CPU 检查为 453 passed, 1 skipped，Ruff/mypy 通过。真实容器还通过并发、健康、deadline 和恢复检查；客户端取消已执行，但服务端断连传播与编码阶段取消未被该 smoke 独立验证。

Grype 已扫描 dev1 SPDX，包与公告匹配尚未完成分诊，见 [安全维护](../security-maintenance.md)。
私密漏洞报告仍关闭，当前会话没有 GitHub API 管理凭据。当前简版状态见
[Current status](../current-status.md)，完整审查与未完成项见 STATUS 最新会话。

## 接续工作

1. 阅读根目录 [AGENTS.md](../../AGENTS.md)、本文件和 [STATUS.md](STATUS.md)。STATUS 顶部汇总及最新会话记录反映当前进展；较早会话保留当时的失败和未执行状态。
2. 如存在，读取 `.local/SERVER.md`，并在重工作前重新检查资源占用。该文件仅用于本机观察，不提交到 Git。
3. 阅读历史 [设计](../plans/JevServe-design.md)、[T00–T12 计划](../plans/JevServe-task-plan.md) 和 [DECISIONS.md](DECISIONS.md) 理解约束与变更；不要重新从初始化或 T00 开始。
4. 按 [开源完善计划 O01–O08](../plans/open-source-hardening.md) 和 STATUS 最新未完成项接续。公开编排和三层 JPT 诊断已实测；24 小时 NeoHorse 长稳已提前失败，新源码镜像执行验收已完成，这批用例已通过新的有界漂移数值策略，下一步推进独立的运维/发布遗留项（重评已通过独立审查）。Open-Jev 的 base/FP32-LoRA 与 JPT 融合 Q 归一化/RoPE 严格根因研究转为可选，不再阻塞这批用例的数值接受；不要把有限用例通过写成完整模型认证。

开始时运行 `git status --short --branch`，保留已有工作。CPU 安装、CLI 和检查命令见 [CONTRIBUTING.md](../../CONTRIBUTING.md)；源码运行时开发见 [IMPLEMENTATION.md](IMPLEMENTATION.md)。不要把已有缓存或某次宿主观察当成新环境验证。部署命令以 Quick Start 为准。

## 已确定的边界

- 一个通用运行时镜像、外部挂载权重，一个容器拥有一个模型/引擎，单 Linux NVIDIA 主机、单 GPU；Docker 与 Compose 都是正式部署路径。
- vLLM 为主；NeoHorse/Kev 使用明确选择的 Torch pointer profile，不是在 vLLM 失败后自动回退。
- 当前 API 仅支持文本 Choice/Noul/Score。用户已扩展模型调查范围到 VL 家族，但图片输入未实现；JEV-27B-VL、Xor 1.2 和 Laya 尚未接入或完成 GPU 验证。
- JevServe 独立于 TypeSafe AI 和 JevServe-Bench。命名已确定；不重新讨论，也不引入 Kubernetes、训练平台、Web UI、多机或空壳 Triton 后端。
- 开发模型仍须显式使用 `--allow-uncertified`。保留历史严格容差和失败证据；新增有版本的有界漂移策略单独重评，不更改模型语义、精度或权重。新策略同时约束每项概率误差和全部分布的 argmax，不能仅凭标签一致、mock 推理或构建成功声称通过。

## 验证与交付

CPU 通过、容器执行通过和真实 GPU 数值认证分别记录。dev0 → dev1 → dev0 回滚实测已通过；24 小时 NeoHorse 长稳因项目 Docker 实例失联而提前失败，1590 次请求中 1589 次成功，持续 16271.64 秒，未通过时长门槛；项目实例已恢复，旧长稳容器已清理。独立干净主机安装、完成的 24/72 小时长稳仍待验收；现有小型对照不构成完整发布验收。新的有界策略只接受已评估用例的数值漂移，不构成算术等价证明；历史严格容差不变。若继续可选根因研究，仍须依据编码、读出、温度、精度和批处理证据，不无依据归因于 BF16。

精确复现命令、原始证据位置和下一步见 [STATUS 最新源码镜像会话](STATUS.md)。本机矩阵证据在 `.local/source-13eed9b-matrix-acceptance`，Compose 在 `.local/source-13eed9b-compose-retry`；这些目录已存在，重复运行必须选新目录。保存新重评策略与已有 GPU 原始证据的绑定，按已审查的重评报告推进剩余运维门禁；严格根因研究可另行安排。任何后续 GPU 工作先检查占用；若修改生产代码，必须重建镜像并重做受影响的源码绑定/真实对照。当前本地候选未发布，报告不含 `image_digest`，不能用于绕过发布门禁。

按当前会话授权执行；默认顺序工作，只有明确允许时委派，GPU 重工作保持串行。不得重启共享 Docker daemon、改驱动或清理他人资源。离线模式不访问网络，下载到 staging、校验后原子发布；不得提交权重、凭据或 `.local/`。

结束前更新 STATUS：实际命令、结果、证据路径、未运行检查、阻塞和精确下一步。重大设计变化记录到 DECISIONS，能力或状态变化同步中英文 README。按会话安排审查并提交完整变更；沿用会话中已有的 push/发布授权及目的地；未授权的新目的地或发布范围需要另行明确。用户可使用 [START-CODEX.md](START-CODEX.md) 中的提示接续。

最新上游对照：vllm-jev 0.3.7 的 no_splitk 思路已做本机固定投影实验，显著减少 packed/separate 差异，但改变的是 separate 输出，未改善 runtime 对冻结 HF 基准的匹配。不得将其直接设为生产修复；详见 STATUS 最新研究会话及 [实验报告](../../validation/reports/upstream-no-splitk-projection.json)。
