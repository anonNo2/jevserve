# JevServe：单机 NVIDIA 决策模型服务设计

日期：2026-10-12。状态：供实施使用的设计基线，尚未实现或进行 GPU 验证。项目品牌名为 JevServe；仓库、Python 包、CLI 与镜像基础名称统一为 jevserve。

## 项目命名与独立身份

用户已接受与现有 JevServe-Bench 的近名情况，确定使用 JevServe / jevserve。JevServe 是独立开源项目，不隶属于 TypeSafe AI 或 JevServe-Bench，也不暗示获得其背书。README 应显式说明这一身份。

英文标语：Serve Jev-style models. One command.
中文定位：一条命令，部署决策模型。

公开名称近似项目参考：https://supercomputing-system-ai-lab.github.io/blogs/rethinking-llm-serving-with-jev/
PyPI 包名可注册性、GitHub 发布 namespace 和镜像 registry namespace 尚未确认；命名决定不代表已注册或占有这些名称。公开发布时核实实际 namespace，不改变既定 CLI 名称。

## 目标与边界

交付一个可公开发布、可继续扩展的开源仓库雏形。用户只需 Linux x86_64、支持的 NVIDIA GPU/驱动、Docker Engine 与 NVIDIA Container Toolkit；宿主机不安装 Python、CUDA Toolkit 或项目依赖。拉取镜像后，以一条 docker run 命令启动支持的开源 Jev 类模型，也可 docker compose up -d。首次启动可下载已锁定的模型；后续复用缓存。离线模式仅使用预先准备的模型包。

这里的 Jev 指 Jev-style / System One 开源决策模型，不代表提供 TypeSafe 官方 Jev 权重。发布文案必须明确这一点。

首版限定单机、单容器一个模型、单 GPU、文本 Choice/Noul/Score。默认支持目标为 ZefanCai/Open-Jev-2B；第二适配目标为 convaiinnovations/laya。两者能否进入认证清单由参考结果和 GPU 测试决定，不能只依据上游支持列表。首版 release 至少有一个真实 checkpoint 通过全部容器与 GPU 门禁。第二模型未通过时标记 experimental 或移出发布镜像，不能假装支持。

不包含 Kubernetes、集群控制面、训练平台、Web 管理界面、自动换模型、量化、热加载 LoRA、多模态和多机并行。Triton 保留设计扩展点，不交付空壳后端、不阻塞首版。

## 架构

默认一个镜像、一条启动命令、一个对外端口。API 与模型执行是逻辑分层，不要求独立服务；底层引擎可以创建自身 worker 进程。一个 API 进程拥有一个 engine 实例，禁止通过多个 Uvicorn worker 隐式复制模型。

请求流程：鉴权与大小限制 → 类型校验 → 模型能力校验 → 有界准入 → 模型输入编码 → vLLM 调度/批处理 → 专用 readout → 概率检查和可选弃权策略 → 结果与执行版本。

复用 vLLM 上游模型、pooling 或插件接口；对必要的模型专用 head 建立最小适配，不自研 GPU scheduler。原生标签 token 读取与专用决策 head 是不同策略，manifest 必须声明，不能静默相互替代。上游 vllm-jev 可作为参考或合法复用来源，但最终用户无须另行安装它；引入源码时保留许可证和 NOTICE。

模块边界：contracts 定义公共语义；adapters 定义编码、能力和 readout；runtime 封装 vLLM 版本差异；artifacts 管理不可变模型包；server 管理请求生命周期；cli 管理启动、检查和准备；validation/benchmarks 提供独立认证。

## 公共行为

- CLI：serve、models list、doctor、prepare、check、benchmark。serve 无参数使用随镜像发布的默认模型清单。
- API：POST /v1/decisions；POST /v1/systemone；GET /v1/models；GET /livez；GET /readyz；GET /metrics。
- /v1/systemone 兼容范围按冻结的协议及官方 SDK 版本测试，不承诺所有网关协议完全兼容。兼容响应不随意添加字段；版本信息可用响应头传递。
- /v1/decisions 返回 answers、model_revision、adapter_revision、execution_profile、probability_kind、calibration_id 和 request_id。calibration_id 默认 null，不把 softmax 称为校准概率。
- Choice 分布限定当前候选集合；Noul 的正负标签映射由适配器固定；Score 返回 0 起始有序等级期望及 legend，不当作业务金额等连续预测。
- 多问题请求首版整体成功或失败。单请求最多 16 问题；HTTP body 最大 1 MiB；候选数/上下文长度按模型 manifest 强校验；不自动截断。
- 时间预算从请求进入服务开始计算。默认 deadline 30 秒，客户端可缩短但不能超过管理员上限。队列默认最多 64 个等待请求、最多 16 个已准入请求；这些是可调初值，不是性能认证结论。token/候选预算还需受对应模型 profile 限制。
- 客户端断开、超时或任务取消时回收配额并调用 engine abort；不能承诺立即中断 GPU kernel。默认不重试，不换模型，不静默换精度。
- 错误语义：400 结构/选项错误；413 body 过大；422 模型不支持的能力/上下文；429 准入超限；503 未就绪/执行引擎失败；504 deadline；401 鉴权失败。响应含稳定 error.code 与 request_id，不暴露 traceback 或输入正文。

## 模型包与发布锁定

发布 artifact manifest 固定权重 commit、文件 digest、tokenizer/template digest、adapter revision、engine image digest、精度和支持的 GPU profile。模型 ID 可作为用户输入，但只解析到镜像内认证 registry；任意本地模型需提供经校验 manifest，不默认执行远程代码。

在线 prepare 与 serve 的首次下载调用同一实现：文件锁 → staging 目录 → 下载 → digest 校验 → 参考结构检查 → 原子 rename → ready 标记。仅启动阶段有网络下载，不在请求路径下载。缓存命中必须验证 manifest，不因目录存在而跳过校验。

离线模式缺文件立即失败，不访问网络。鉴权令牌通过 secret 文件读取，不写入命令行、镜像或日志。权重不默认打入镜像，避免镜像分发与模型许可证混同。下载失败退出非零，重试有上限。

## 容器与 Compose

对外发布一个 jevserve 镜像；镜像内包含固定的 Python、vLLM、适配器和 CLI。prepare/check/benchmark 复用同一镜像的子命令。镜像 build 不需 GPU，release 验证需要真实 GPU。

在线模式：可写持久模型缓存挂载 /var/lib/jevserve，编译缓存也按兼容性 key 隔离。离线模式：模型包只读挂载 /models，可写编译缓存独立挂载。以非 root 用户运行，并测试 Docker 命名卷初始权限及 host bind mount 的显式 uid/gid 使用方法。

Docker ENTRYPOINT 使用 exec 形式指向 jevserve。CMD 是 serve；docker run IMAGE serve --model MODEL 和 docker run IMAGE --model MODEL 均由 CLI 定义并测试。SIGTERM 先停止准入、等待在途请求，最多 30 秒后 abort、关闭 engine 并退出。Compose stop_grace_period 45s。

默认发布到宿主机 127.0.0.1:8000；容器内部监听 0.0.0.0。对外暴露须显式配置并建议 API key/TLS 反向代理。初版 API key 只覆盖简单单租户需求，不宣传为多租户系统。

设置专属 shm_size，不能为方便默认使用 privileged 或 host IPC。GPU device_ids 默认仅 0，管理员可改；内存规格与驱动最低要求由实测 profile 确定。

Compose 默认仅 inference 服务，监控 profile 增加 Prometheus/Grafana。prepare 和 validation 使用 tools profile 按需运行。启动时不 pip install，不依赖宿主项目源码挂载。restart on-failure 限制重启次数，致命 engine 故障使进程退出；healthcheck 仅报告状态，不声称 unhealthy 会自动重启。

## 运维和正确性

livez 表示 API 事件循环可响应；readyz 只有模型加载、预热和 smoke inference 通过且 engine 健康时返回成功。healthcheck 有足够 start_period；长期准备可用容器状态和日志区分 downloading/loading/warming/ready/failed。doctor 输出驱动/GPU/磁盘/权限问题，不输出 secrets。

Prometheus 记录端到端延迟、准入等待、活动/等待请求、错误码、取消和模型状态。仅能可靠测量的阶段发布阶段耗时；不把宿主计时误称 GPU kernel 时间。request_id、输入、任意 label 不作为 metric 标签。

校准不是首版必需功能；保留校准标识且默认 null。可选阈值策略的版本必须出现在响应中。不宣称跨硬件位级一致。所有稳定性结论绑定模型、GPU、精度、引擎、模板与批处理配置。

## 发布分级

v0.1.0-alpha：完整 repo、可构建容器、至少一个真模型在指定 NVIDIA GPU 上一条命令启动、协议/参考一致性/重启/离线检查通过。公开 GPU 报告和限制，不能用 CPU mock 替代。

v0.1.0-rc：增加开放式负载测试、24 小时长稳、故障和回滚演练。

v0.1.0：72 小时长稳及全新主机安装演练通过，发布固定 digest 和支持矩阵。仍只承诺已认证配置，不能称为与 vLLM 等成熟。

## 来源和需验证事项

- https://github.com/mode-io/vllm-jev ：模型适配参考及公开一致性限制。
- https://raw.githubusercontent.com/mode-io/vllm-jev/main/pyproject.toml ：2026-10-12 查阅到 v0.3.6、vLLM 0.29.0 与 Python 3.12+ 的依赖声明；它们仅是 P0 验证候选，不等于我们已验证。
- https://docs.vllm.ai/en/latest/design/plugin_system/ ：上游插件接入。
- https://docs.vllm.ai/en/latest/deployment/docker/ ：官方容器路径。
- https://docs.vllm.ai/en/latest/serving/online_serving/structured_decisions/ ：标签 token 决策路径及限制；latest 文档能力需与实际固定发行版核对。

硬件型号、权重许可证、上游 tag/commit 和可获取的镜像 digest 必须在 T00 中生成机器可读锁文件。未验证的 profile 不进入支持矩阵。GPU 不可用时可以完成 CPU 工作，但不能完成 GPU 门禁或发布认证。
