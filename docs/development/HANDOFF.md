# Codex 开发交接

实施已于 2026-10-12 开始。下方初始化记录属于历史基线；当前完成项、阻塞项和下一步以 [STATUS.md](STATUS.md) 为准。CPU 开发命令见 [IMPLEMENTATION.md](IMPLEMENTATION.md)。

## 用户目标

交付一个完善的开源推理 repo 雏形，能像 vLLM 一样以一个容器极简拉起 Jev 类模型。主打真实可运行、模型结果可验证、离线/运维完整，而不是只有目录和 API 外壳。

用户已选择 Linux + NVIDIA、单机 Docker Compose，认可名称 `jevserve` 并接受与 JevServe-Bench 近名。默认是单容器，而非强制 gateway + worker 两套部署。架构和详细计划见 `docs/plans/`；它们来自先前需求讨论，不需要重新询问产品定位。

## 此次初始化完成了什么

- 上传架构设计与 13 项任务规划（T00–T12）。
- 编写中英文 README、AGENTS.md、交接/状态/决策文档、贡献指南和 Apache-2.0 许可证。
- 初始化 Git 主分支并提交文档基线；不配置虚构远端，不推送。
- 将服务器环境快照写入 `.local/SERVER.md`，避免把私人服务器地址提交到公开仓库。

没有实现推理服务、Python 包、Dockerfile、Compose、SDK 或 CI。没有下载权重、安装依赖、启动模型或完成 GPU 认证。README 命令属于目标体验，不能报告为现在可用。

## 阅读顺序

1. 根目录 `AGENTS.md`。
2. `docs/development/STATUS.md` 和本文件。
3. 本机 `.local/SERVER.md`（如果存在）。
4. `docs/plans/JevServe-design.md`。
5. `docs/plans/JevServe-task-plan.md`，从 T00 开始。

## 第一段实际工作

用户发出“开始开发”后，先执行 T00 的只读环境和上游核验：

1. `git status --short --branch`；检查已有变更，保留用户工作。
2. 检查 GPU 空闲情况、Docker daemon 权限、NVIDIA runtime 和 GPU 容器透传。不能根据宿主 nvidia-smi 成功就宣布容器 GPU 可用。
3. 核验目标权重和 vllm-jev 上游是否真实可访问，读取许可证和模型实现，固定 commit。历史版本号及搜索所得模型名称必须重新核实。
4. 选择实际存在且能在本机运行的 vLLM/CUDA/PyTorch 组合，建立本地 Python 3.12 环境或容器；不修改系统 Python。
5. 运行作者参考实现并保存对照 fixture、模型清单及证据，随后进入 T01。

上游资源不存在、网络失败或模型许可证不允许时，如实记录，完成其余独立工作；需要改变目标模型时向用户说明证据和替代选择，不能编造 lock 文件或 benchmark。

## 历史计划需要注意的地方

- “vLLM latest 文档有某接口”不能替代固定发行版验证。
- 计划中的测试片段需要补齐 fixture/import/helper；它们是行为约定，并非已运行代码。
- `FakeRuntime` 只能用于 CPU 测试，不能用于演示上线。
- `calibration_id=null` 是首版正常状态，不开展未经要求的校准平台建设。
- T09 第二模型可选；Triton 不属于本轮交付。
- 计划中的工程日是估算，不能据此跳过验收。
- 若本机没有 superpowers 等技能，直接按任务和验收工作，不为安装技能中断开发。

## 验收与后续会话

每完成一个任务更新 STATUS，记录真实命令和结果，并以本机既有 Git 身份提交。标明未执行的 GPU/长稳检查。不要生成虚构徽章、镜像 registry、PyPI 链接、星标数量或性能数字。

用户本次只要求初始化与交接，并未要求本会话启动 T00–T12。后续用户要求开发时再执行；不需要重新做命名/架构讨论。开源远端创建、push、镜像/软件包公开发布需用户给出目的地及授权。

供用户在服务器 Codex 会话中使用的启动提示见 [START-CODEX.md](START-CODEX.md)。
