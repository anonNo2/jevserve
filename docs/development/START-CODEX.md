# Start the next Codex session

在仓库根目录启动 Codex 后，可发送以下提示继续当前工作。实际进度以 [STATUS.md](STATUS.md) 为准；本提示不授权发布或重跑全部 GPU 验证。

```text
请继续完善 JevServe。先阅读 AGENTS.md、docs/development/HANDOFF.md、docs/development/STATUS.md、.local/SERVER.md（若存在），以及 docs/plans/JevServe-design.md、docs/plans/JevServe-task-plan.md、docs/plans/open-source-hardening.md 和 docs/development/DECISIONS.md。

已有公开源码和通用 dev1 镜像，7 个固定模型配置有真实 GPU/容器执行证据；当前仍是未认证开发预览。Open-Jev/JPT 数值对照失败，NeoHorse/Kev 仅通过小型合成对照。当前 API 只支持文本，不能宣称已支持图片或所有 VL 模型。

先检查 git status，保留现有变更，按 STATUS 最新记录和开源完善计划选择未完成工作。不要从初始化/T00 重做，不重复询问已确定的命名和单机单 GPU 架构。默认顺序执行；只有本会话明确授权时再委派。GPU/容器重工作前检查共享资源，并串行调度。

公开可复现的验证、数值偏差诊断和可追溯发布流程是近期重点。保持模型精度和冻结容差，不能用标签一致、CPU 测试或 Docker build 成功代替 GPU 认证。回滚、干净主机安装和 24/72 小时长稳需要各自实际证据，未执行就如实标记。

使用 Python 3.12 项目环境和固定依赖。每项工作完成后记录命令、结果、证据和下一步，更新 STATUS，必要时更新 DECISIONS，并同步中英文 README。审查变更后按当前授权提交，不改共享 Docker/驱动或他人资源，不做未授权的 push 或发布。遇到真实阻塞时记录证据并继续独立工作。
```
