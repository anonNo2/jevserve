# Codex 开发交接

更新：2026-10-13。JevServe 已实现 CLI、HTTP/SDK、模型准备、真实推理和 Docker/Compose 部署，当前交付为开发预览。Open-Jev-2B/9B、JPT-0.8B/4B/9B、NeoHorse-Jev-4B、Kev-4B 共 7 个固定模型配置均有真实 GPU/容器执行证据。Open-Jev/JPT 概率对照仍失败；NeoHorse 与修正加载配置后的 Kev FP32 仅通过小型合成对照。尚无 alpha、稳定版或模型认证。

源码已公开于 [GitHub](https://github.com/anonNo2/jevserve)。通用镜像 `anonwx/jevserve:0.1.0-dev1` 已发布，其固定 digest 为 `sha256:4b758ad1000f4d38501fd83f0a9c2b32f0cb91968d427b170a2c2b0d9a613aff`；匿名拉取已通过，但复用了缓存层，不能视为干净主机安装。部署步骤见 [Quick Start](../quickstart.md)，模型许可证、能力与逐项证据见 [models.md](../models.md)。尚未发布 Python 包。

## 接续工作

1. 阅读根目录 [AGENTS.md](../../AGENTS.md)、本文件和 [STATUS.md](STATUS.md)。STATUS 顶部汇总及最新会话记录反映当前进展；较早会话保留当时的失败和未执行状态。
2. 如存在，读取 `.local/SERVER.md`，并在重工作前重新检查资源占用。该文件仅用于本机观察，不提交到 Git。
3. 阅读历史 [设计](../plans/JevServe-design.md)、[T00–T12 计划](../plans/JevServe-task-plan.md) 和 [DECISIONS.md](DECISIONS.md) 理解约束与变更；不要重新从初始化或 T00 开始。
4. 按 [开源完善计划 O01–O08](../plans/open-source-hardening.md) 和 STATUS 最新未完成项接续。当前优先项是公开验证编排、Open-Jev/JPT 数值偏差诊断及可追溯发布流程；以实际完成证据更新任务状态。

开始时运行 `git status --short --branch`，保留已有工作。CPU 安装、CLI 和检查命令见 [CONTRIBUTING.md](../../CONTRIBUTING.md)；源码运行时开发见 [IMPLEMENTATION.md](IMPLEMENTATION.md)。不要把已有缓存或某次宿主观察当成新环境验证。部署命令以 Quick Start 为准。

## 已确定的边界

- 一个通用运行时镜像、外部挂载权重，一个容器拥有一个模型/引擎，单 Linux NVIDIA 主机、单 GPU；Docker 与 Compose 都是正式部署路径。
- vLLM 为主；NeoHorse/Kev 使用明确选择的 Torch pointer profile，不是在 vLLM 失败后自动回退。
- 当前 API 仅支持文本 Choice/Noul/Score。用户已扩展模型调查范围到 VL 家族，但图片输入未实现；JEV-27B-VL、Xor 1.2 和 Laya 尚未接入或完成 GPU 验证。
- JevServe 独立于 TypeSafe AI 和 JevServe-Bench。命名已确定；不重新讨论，也不引入 Kubernetes、训练平台、Web UI、多机或空壳 Triton 后端。
- 开发模型必须显式使用 `--allow-uncertified`。保持冻结容差及原始语义，不用标签一致、mock 推理或构建成功替代数值认证。

## 验证与交付

CPU 通过、容器执行通过和真实 GPU 数值认证分别记录。回滚演练、独立干净主机安装、24/72 小时长稳仍需实际完成；现有小型对照不构成完整发布验收。数值偏差应按编码、读出、温度、精度和批处理逐层调查，没有证据不归因于 BF16，也不放宽容差。

按当前会话授权执行；默认顺序工作，只有明确允许时委派，GPU 重工作保持串行。不得重启共享 Docker daemon、改驱动或清理他人资源。离线模式不访问网络，下载到 staging、校验后原子发布；不得提交权重、凭据或 `.local/`。

结束前更新 STATUS：实际命令、结果、证据路径、未运行检查、阻塞和精确下一步。重大设计变化记录到 DECISIONS，能力或状态变化同步中英文 README。按会话安排审查并提交完整变更；沿用会话中已有的 push/发布授权及目的地；未授权的新目的地或发布范围需要另行明确。用户可使用 [START-CODEX.md](START-CODEX.md) 中的提示接续。
