# JevServe

**一条命令，部署决策模型。**

[English](README.md) · [架构设计](docs/plans/JevServe-design.md) · [任务规划](docs/plans/JevServe-task-plan.md) · [开发交接](docs/development/HANDOFF.md)

JevServe 计划成为面向 Jev 类 / System One 决策模型的开源推理服务框架。目标是让决策模型像 LLM 一样容易部署：一个容器镜像、一条启动命令、明确的 API，以及可复现的模型验证。

> **当前状态：已开始实施，CPU 协议、经校验的模型准备流程和 Open-Jev 适配器已实现。110 条真实 tokenizer 编码对照通过，GPU 推理仍未验证。** Python 开发包可安装；推理、CLI 和容器尚未实现，没有认证模型或已发布镜像。下方启动命令仍为目标接口。参见[开发安装](docs/development/IMPLEMENTATION.md)及[当前验证记录](docs/development/STATUS.md)。

## 项目目标

- **极简部署：** 单机 Linux + NVIDIA GPU，支持 Docker 与 Docker Compose。
- **类型化决策：** Choice、Noul（是/否）、Score，明确说明概率的真实含义。
- **复用成熟引擎：** 使用 vLLM 的调度和批处理，维护范围受控的模型适配器。
- **可复现产物：** 固定权重、tokenizer、模板、运行时版本和镜像 digest。
- **运行保障：** 就绪检查、有界队列、超时、取消、指标、离线包和回滚。
- **以证据认证：** 每个认证配置均有真实 GPU 测试和作者参考实现对照。

首个里程碑是可运行的 alpha 仓库，并不承诺达到 vLLM 的功能范围或成熟度。首版不包含 Triton、Kubernetes、训练、Web UI、多模态或自动切换模型。

## 目标启动体验——尚未实现

未来运行环境需具备 Linux x86_64、兼容的 NVIDIA GPU/驱动、Docker Engine 和 NVIDIA Container Toolkit。宿主机不应需要安装 Python 或 CUDA Toolkit。

Dockerfile 和推理功能实现后，本地开发镜像应可这样启动：

```bash
docker build -t jevserve:dev .
docker run --rm --gpus 'device=0' --shm-size=2g \
  -p 127.0.0.1:8000:8000 \
  -v jevserve-cache:/var/lib/jevserve \
  jevserve:dev serve --model ZefanCai/Open-Jev-2B
```

正式发布时提供真实可拉取的镜像，用户不必自行构建。另一条目标启动路径是：

```bash
docker compose up -d
```

首次在线启动由容器准备并校验已锁定模型，后续复用缓存；离线运行使用提前准备的模型包。等待服务就绪后调用：

```bash
curl -fsS http://127.0.0.1:8000/readyz
curl -fsS http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"state":"I was charged twice.","questions":{"team":{"type":"choice","instructions":"Which team should handle this?","criteria":{"billing":"Payments and refunds","technical":"Technical problems"}}}}'
```

这里不提供冒充实测结果的示例响应。鉴权和部署设置将随服务实现一并记录。

## 模型规划

| 模型 | 定位 | 当前状态 |
|---|---|---|
| `ZefanCai/Open-Jev-2B` | 首个适配目标 | 已核验来源、许可证和 revision；参考执行、GPU 和容器认证待完成 |
| `convaiinnovations/laya` | 可选第二适配器 | 规划中，不阻塞 alpha |
| TypeSafe 官方 Jev | 本项目不分发 | 不包含官方权重 |

写入计划不等于已经支持。发布矩阵将认证绑定到 checkpoint revision、GPU、精度、运行时和测试报告。历史检索所得上游版本和能力必须在实施前重新核验。

## API 规划

| 接口 | 用途 |
|---|---|
| `POST /v1/decisions` | 原生类型化决策与执行元数据 |
| `POST /v1/systemone` | 按版本测试的兼容接口 |
| `GET /v1/models` | 已部署模型与能力 |
| `GET /livez`、`GET /readyz` | 存活与模型就绪检查 |
| `GET /metrics` | Prometheus 指标 |

归一化概率不等于经过校准的正确率。初版默认返回 `calibration_id: null`，除非显式使用经过验证的校准产物。

## 从这里开始开发

1. 阅读 [AGENTS.md](AGENTS.md)、[开发交接](docs/development/HANDOFF.md)、[设计](docs/plans/JevServe-design.md)和[任务规划](docs/plans/JevServe-task-plan.md)。
2. 在 [STATUS.md](docs/development/STATUS.md) 查找下一任务：**T00——完成真实参考执行和容器基线**，并查看其中记录的 CPU 工作。
3. CPU 测试与 GPU 认证分开；mock、镜像构建成功或上游 README 均不能替代真实验证。
4. 保存命令、版本和证据后，才能标记任务完成。

执行顺序：基线与协议 → 模型包与适配器 → vLLM 真实推理 → API/CLI → Docker/Compose → 可靠性与验证 → 发布文档。

协作约定见 [CONTRIBUTING.md](CONTRIBUTING.md)。设计和任务规划当前以中文维护，英文 README 作为对外入口。

## 独立身份与许可证

JevServe 是独立项目，不隶属于 TypeSafe AI 或 JevServe-Bench，也不暗示获得其背书。“Jev 类”描述决策模型类别，本仓库不提供 TypeSafe 官方 Jev 权重。

仓库内容采用 [Apache-2.0](LICENSE)。模型权重和第三方组件遵循各自许可证，详见 [NOTICE](NOTICE)。仓库、软件包和镜像发布 namespace 尚未注册或发布。
