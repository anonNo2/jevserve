# JevServe

**一条命令，部署决策模型。**

[English](README.md) · [架构设计](docs/plans/JevServe-design.md) · [实测证据与下一步](docs/development/STATUS.md)

JevServe 在单机 Linux NVIDIA GPU 上提供文本 Choice、Noul 和 Score 决策服务。一个 API 进程拥有一个 vLLM 引擎。本项目独立于 **TypeSafe AI** 和 **JevServe-Bench**，不分发 TypeSafe 官方 Jev 权重，也不暗示获得其背书。

**开发阶段进展：** CLI → HTTP → vLLM → 真实 `ZefanCai/Open-Jev-2B` 已在 RTX A6000 上跑通。原生 SDK、有界准入、超时、取消、健康检查、指标及经校验的离线模型加载已实现。**尚无认证模型：** 当前引擎在固定容差下有 38/113 条预编码参考对照失败（最大概率偏差 0.0218，无标签翻转）。本机 Docker 在 entrypoint 执行前创建容器失败，镜像构建及两条容器推理路径仍未验证。没有发布镜像或软件包。

## 从源码运行

使用项目内 Python 3.12 环境和兼容的 NVIDIA 主机。运行依赖锁包含 CUDA，体积较大；不要替换系统 Python 或 CUDA。

```bash
uv venv --python 3.12 .venv-runtime
uv pip sync --python .venv-runtime/bin/python --require-hashes compat/runtime-requirements.txt
uv pip install --python .venv-runtime/bin/python --no-deps .
.venv-runtime/bin/jevserve serve --allow-uncertified --cache-dir .cache/jevserve
```

开发标志明确承认认证未完成，不会放宽对照容差。首次启动下载固定 revision、校验哈希，保留独立 FP32 LoRA 生成文本 checkpoint，然后加载 GPU 引擎并预热。启动完成后才提供 readiness。后续启动重新校验缓存；加 `--offline` 禁止模型下载。详见[开发与运维说明](docs/development/IMPLEMENTATION.md)。

```bash
.venv-runtime/bin/jevserve check
curl -fsS http://127.0.0.1:8000/v1/decisions \
  -H 'Content-Type: application/json' \
  -d '{"state":"I was charged twice.","questions":{"team":{"type":"choice","instructions":"Which team should handle this?","criteria":{"billing":"Payments and refunds","technical":"Technical problems"}}}}'
```

原生 Python 客户端为 `jevserve.client.Client`；`decide()` 接收 `DecisionRequest` 或字典，返回经校验的响应，不自动重试。

## 容器路径——已编写，尚未验证

Dockerfile 以 digest 固定上游 vLLM 镜像，Compose 配置已通过结构校验。以下命令仍需在 Docker 正常的主机上完成构建和真实 GPU 验证：

```bash
docker build -t jevserve:dev .
docker run --rm --name jevserve-dev --gpus device=0 --shm-size=1g \
  -p 127.0.0.1:8000:8000 -v jevserve-cache:/cache \
  jevserve:dev serve --allow-uncertified
# 或者：
docker compose up --build -d
```

## API 与限制

| 接口 | 行为 |
|---|---|
| `POST /v1/decisions` | 原生回答与执行元数据 |
| `POST /v1/systemone` | 经 `typesafe-sdk==0.7.4` 测试的文本问题子集；必须指定完整模型 ID |
| `GET /v1/models` | 模型 revision 与能力限制 |
| `GET /livez`、`/readyz` | API 存活与引擎就绪 |
| `GET /metrics` | Prometheus 计数、延迟与准入指标 |

原生 state 为非空文本或 JSON 对象，指令与描述为文本；Choice 描述可为 null。Score 接受 2–10 个有序等级。兼容端点不承诺托管 API 的全部对象/数组问题结构或模型别名。详见[兼容范围](compat/README.md)。

单请求上限为 1 MiB、16 个问题、256 条候选序列、每候选 4096 tokens、总计 65536 输入 tokens。默认最多 16 个执行请求、64 个等待请求、30 秒期限，不静默截断输入。`--api-key-file` 启用 Bearer 鉴权，健康端点保持公开；默认仅监听 loopback。

概率表示给定候选集上的分布，**不代表经过校准的正确率**。`calibration_id` 为 null，Score 是有序等级期望。没有生产假引擎、自动重试、模型回退、量化或精度回退。

## 开发与发布门禁

```bash
uv sync --locked
uv run pytest -q
uv run ruff check .
uv run mypy
uv run python -m build --no-isolation
```

CPU 测试无需 GPU 导入或模型下载。作者 GPU 参考、引擎对齐、容器构建、Docker/Compose 推理和长稳测试是独立门禁。详见 [STATUS](docs/development/STATUS.md)、[决策记录](docs/development/DECISIONS.md)和[贡献指南](CONTRIBUTING.md)。

Alpha 面向一个镜像、一个模型、一个 GPU；不包含 Kubernetes、训练、Web UI、多模态、多机或 Triton 实现。第二模型可选。项目代码采用 Apache-2.0；上游代码声明及模型许可证分别保留于 [NOTICE](NOTICE) 和 `licenses/`。
