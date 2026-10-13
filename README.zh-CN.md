<p align="center">
  <img src="docs/assets/brand/jevserve-logo.png" alt="JevServe" width="480">
</p>

<p align="center"><strong>文本进来，决策出来。</strong></p>
<p align="center">在自己的 GPU 上运行决策模型，通过一个 HTTP API 调用。</p>
<p align="center"><a href="README.md">English</a> · <a href="#快速开始">快速开始</a> · <a href="docs/quickstart.md">部署指南</a> · <a href="docs/models.md">支持模型与测试结果</a> · <a href="compat/README.md">API 兼容范围</a></p>

JevServe 将开放权重的 Jev 类决策模型封装成 HTTP 服务。发送文本和问题，获得带概率的类型化答案。你可以用它分派客服工单、识别紧急消息，或按规则给内容评等级。

| 想做什么 | 问题类型 | 返回什么 |
|---|---|---|
| 选择处理工单的团队 | **Choice** | 一个标签，以及各候选项的概率 |
| 判断消息是否紧急 | **Noul** | 0–1 之间的概率 |
| 按低、中、高评定优先级 | **Score** | 等级期望值，以及各等级的概率 |

**一个通用镜像，模型权重独立挂载。** 已接入 Open-Jev、JPT、NeoHorse、Kev 模型家族，使用 vLLM 或指定的 Torch 后端。提供 Python 客户端、有界请求队列、取消、健康检查和离线模式。各模型的实测状态和许可见[支持模型与测试结果](docs/models.md)。

> **开发预览版。** 对已有 GPU 证据的重新评估表明，实测用例通过初始 `bounded-drift-v1` 策略；Open-Jev/JPT 的历史严格对照仍失败。仍需 `--allow-uncertified`，这不是完整模型认证或准确率评测。详见[数值范围与限制](#当前状态与限制)。

## 快速开始

需要 **Linux x86_64、NVIDIA GPU、Docker 和 NVIDIA Container Toolkit**。宿主机不需要安装 Python 或 CUDA Toolkit。已测试的主机使用 RTX A6000，详见[运行要求](docs/quickstart.md#requirements)。

直接运行 [Docker Hub 镜像](https://hub.docker.com/r/anonwx/jevserve)，无需本地构建：

```bash
docker run -d --name jevserve --gpus device=0 --shm-size=1g \
  -p 127.0.0.1:8000:8000 \
  --mount type=volume,source=jevserve-cache,target=/cache \
  anonwx/jevserve@sha256:4b758ad1000f4d38501fd83f0a9c2b32f0cb91968d427b170a2c2b0d9a613aff \
  serve --allow-uncertified
docker exec jevserve jevserve check --wait-seconds 1800
```

首次启动会下载并校验约 4.6 GB 的固定版本模型文件，再编译运行缓存，可能需要数分钟。后续启动复用 `jevserve-cache` 卷。服务就绪后，在 `127.0.0.1:8000` 接收请求。

也可以使用封装好的启动脚本：

```bash
git clone https://github.com/anonNo2/jevserve.git
cd jevserve
./scripts/quickstart.sh --image anonwx/jevserve:0.1.0-dev1
```

脚本自动启动服务、等待就绪，并发送三种问题类型的真实请求。开发者可用 `--build` 从源码构建。[支持模型、测试环境与验证报告 →](docs/models.md)

### 挂载模型，复用同一个镜像

已经准备好固定版本权重时：

```bash
./scripts/quickstart.sh --image anonwx/jevserve:0.1.0-dev1 \
  --model kirp/jpt-0.8b --model-dir /absolute/path/to/verified-model
```

模型目录以只读方式挂载，运行缓存单独放在卷中。更换 `--model` 和 `--model-dir` 即可更换模型，无需重新构建镜像；每个容器只加载一个模型。[下载与校验权重、挂载目录格式、Compose 用法](docs/quickstart.md#mounted-models)。JPT 权重采用 CC-BY-NC-4.0，使用前请确认许可。

### 发起一个决策请求

```bash
curl -fsS http://127.0.0.1:8000/v1/decisions \
  -H 'Content-Type: application/json' \
  -d '{
    "state": "I was charged twice and want a refund.",
    "questions": {
      "team": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {
          "billing": "Payments and refunds",
          "technical": "Software and setup"
        }
      }
    }
  }'
```

从 `answers.team.choice` 读取选中的团队，从 `answers.team.probabilities` 读取候选分布。响应同时提供模型 revision 与执行配置标识。[三种问题类型的完整示例](examples/decision.json)。

### 使用 Docker Compose

在仓库目录中执行：

```bash
JEVSERVE_IMAGE=anonwx/jevserve:0.1.0-dev1 docker compose up --pull always --no-build -d
docker compose exec jevserve jevserve check --wait-seconds 1800
```

Quick Start 和 Compose 二选一使用端口 8000。[自定义端口、GPU、鉴权、离线启动与清理 →](docs/quickstart.md)

## 在 Python 中使用

原生客户端接受与 HTTP API 相同的请求字典：

```python
from jevserve.client import Client

with Client("http://127.0.0.1:8000") as client:
    result = client.decide(
        {
            "state": "Production is down for all users.",
            "questions": {
                "urgent": {
                    "type": "noul",
                    "instructions": "Does this need an urgent response?",
                }
            },
        }
    )
    print(result.answers["urgent"].noul)
```

在 Python 3.12 虚拟环境中，从本仓库运行 `pip install .` 安装客户端。源码客户端只安装 Pydantic 和 HTTPX，不安装服务端或 GPU 依赖，也不自动重试。源码服务端开发使用 `pip install ".[server]"`；GPU 执行仍需单独锁定的运行环境。这些源码改进尚未包含在已发布的 dev1 镜像中。`/v1/systemone` 支持经过测试的 `typesafe-sdk==0.7.4` 文本问题子集，详见[兼容范围](compat/README.md)。

## 当前状态与限制

- **模型：** 7 个固定版本模型已在 RTX A6000 完成真实 GPU 与容器测试；当前 API 仅支持文本，一个容器对应一个模型和一张 GPU。[逐模型测试结果](docs/models.md)。
- **数值：** 7 个配置在已记录的 270 次请求、285 个问题上通过初始 `bounded-drift-v1`：每项概率绝对误差 ≤0.03（3 个百分点），所有分布的 argmax 不变。这是重评已有 GPU 结果，没有新增 GPU 运行或算术修复；历史严格失败记录保留，有限用例通过不认证未测输入。[重新评估报告](validation/reports/source-13eed9b-bounded-drift.json) · [逐模型范围](docs/models.md)。
- **源码镜像：** 未发布的 `13eed9b` 候选镜像已通过全部 7 个配置的真实推理/SDK 检查，以及 Open-Jev-2B、NeoHorse 的离线、只读 Compose 检查。其不可变 Docker 本地 ID 不是已发布的 registry digest；已发布 dev1 不变。[CPU/源码绑定](validation/reports/source-13eed9b-image-cpu.json) · [GPU/Compose 证据](validation/reports/source-13eed9b-image-gpu.json)。
- **生命周期与长稳：** 新镜像通过并发请求、健康、截止时间与恢复检查；已执行客户端取消，但未独立验证服务端断连传播。旧 dev1 的 24 小时长稳在 16,271.64 秒后失败，1,590 次请求中 1,589 次成功；尚无完成的 24/72 小时门禁或干净主机验收。[长稳失败记录](validation/reports/soak-24h-failed.json)。
- **输入：** 最多 16 个问题；Choice 上限 255。Score 等级数、最少候选数和 token/计算量限制因模型而异，见 `/v1/models`。不静默截断、改变精度、切换模型或重试。
- **发布：** 当前是开发预览版，尚非通过 alpha 认证的版本或生产版。[当前状态](docs/current-status.md) · [详细验证记录](docs/development/STATUS.md)。

## 文档与贡献

[API 参考](docs/api.md) · [部署指南](docs/quickstart.md) · [开发环境](docs/development/IMPLEMENTATION.md) · [架构](docs/plans/JevServe-design.md) · [模型详情](docs/models.md) · [贡献指南](CONTRIBUTING.md)

```bash
uv sync --locked --extra server
uv run --extra server pytest -q
uv run --extra server ruff check .
uv run --extra server mypy
```

CPU 测试无需 GPU 或模型下载。欢迎提交问题和改进；报告问题时请附上运行版本和最小复现示例。

[复现验证](docs/validation.md) · [数值偏差诊断](docs/numerical-diagnosis.md) · [发布流程](docs/releasing.md) · [开源完善计划](docs/plans/open-source-hardening.md)

[业务示例](docs/examples.md) · [模型适配指南](docs/contributing-models.md) · [性能与长稳验证](docs/performance-and-soak.md) · [安全报告](SECURITY.md) · [镜像 SBOM](docs/sbom.md)

## 许可证

Apache-2.0。上游代码与模型的许可证见 [NOTICE](NOTICE) 和[模型详情](docs/models.md)。

JevServe 独立于 TypeSafe AI 与 JevServe-Bench，不包含 TypeSafe 官方 Jev 权重。
