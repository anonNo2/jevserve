# Start the next Codex session

在仓库根目录启动 Codex 后，将下面的提示发送给它。此提示供用户手动发送，不代表初始化阶段已开始实现。

```text
请开始实现 JevServe。先阅读 AGENTS.md、docs/development/HANDOFF.md、docs/development/STATUS.md、.local/SERVER.md（若存在）以及 docs/plans/ 中的设计和任务规划。

目标是完成可公开发布的开源 repo 雏形，至少一个真实 Jev 类模型能够用单条 docker run 或 docker compose up -d 拉起。部署范围是单机 Linux + NVIDIA。首版用 vLLM，不做 Kubernetes 或空壳 Triton 后端。

先核查现有 Git 变更和机器状态，从 T00 的真实上游/许可证/版本/GPU 容器验证开始，然后按依赖顺序实施。顺序执行即可，不重复征询已确定的命名和架构；根据证据修正规划中的不一致并记录理由。不要因为本机没有计划中提到的技能而停止。

不要用 mock 推理、仅构建成功或上游宣称代替真实 GPU 验收。README 要如实反映已实现能力。每完成任务更新 STATUS、保存验证证据并提交；不改动共享服务器的全局驱动、Docker 配置或他人资源，不做未授权的公开发布。遇到真实阻塞时说明证据并继续可独立完成的工作。
```
