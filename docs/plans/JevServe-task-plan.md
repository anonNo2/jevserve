# JevServe Implementation Plan

> **For agentic workers:** Execute task-by-task; checkbox items are the tracking unit. Read the repository root AGENTS.md and docs/development/HANDOFF.md first. If available, use superpowers:executing-plans for sequential work; delegation requires the active session to allow it. Missing optional skills do not block the concrete plan. This document plans work; no inference implementation has been performed.

**Goal:** 交付完善的开源 repo 雏形，让用户通过一个 Docker 镜像或 Docker Compose 极简启动经过验证的开源 Jev 类决策模型。

**Architecture:** 单容器对外暴露 API，内含一个 vLLM engine 及其 worker；模型适配、协议、请求治理与模型包管理分层。默认一个容器一个模型一张 GPU，不强制独立 gateway，不实现第二套调度器。

**Tech Stack:** Python 3.12、vLLM、FastAPI/Pydantic、Typer、httpx、pytest、Prometheus、Docker/Compose、GitHub Actions。精确 engine/CUDA/PyTorch 组合由 T00 验证并锁定。

**Spec:** [JevServe-design.md](JevServe-design.md)

## Global Constraints

- 项目品牌为 JevServe；仓库、Python 包、CLI、镜像基础名称为 jevserve。用户已接受与 JevServe-Bench 近名；README 明确独立于该项目及 TypeSafe AI。包名与发布 namespace 可用性在发布前核实。
- Linux x86_64 + NVIDIA、单机 Docker/Compose；不包含 Kubernetes。
- 一个发布镜像必须支持 docker run 与 Compose 两条同等有效路径；宿主无需 Python。
- 主线先实现 vLLM。Triton 不在本计划执行范围，不生成伪实现。
- 首发默认目标 ZefanCai/Open-Jev-2B；至少一个真实 checkpoint 完成 GPU 验证才可交付 alpha。
- 只支持显式 registry/manifest 中的模型；官方 TypeSafe Jev 权重不属于交付。
- 所有建议命令中的本地镜像名 jevserve:dev 是未来实现目标；计划文件本身不提供可运行镜像。
- Body 上限 1 MiB，最多 16 问题，默认 deadline 30s，默认最多等待 64 请求、已准入 16 请求；模型 token/候选预算另行约束。
- 首版整体结果、无自动重试/换模型/截断/换精度，默认 calibration_id=null。
- drain 上限 30s、Compose stop_grace_period=45s；不得以 unhealthy 自动重启作为恢复机制。
- 每个任务结束运行相关检查并提交一次逻辑完整的 commit；写代码后进行独立代码审查，处理输入/鉴权部分进行安全审查。用户选择执行方式前不启动实现。

## Review Focus

1. 中断下载、并发准备和磁盘耗尽：不允许产生可被误用的半成品模型包（T02、T08）。
2. 单请求/批处理/缓存变化：标签顺序、mask、概率归一化和 near-tie 翻转可追踪（T03、T04、T10）。
3. 取消、deadline、worker 崩溃：无配额泄漏、孤儿请求或永远 ready 状态（T05、T06、T08）。
4. 从空缓存启动及离线启动：无需宿主 Python，卷权限正确，secret 不出现在日志中（T02、T07、T08）。
5. 上游升级改变模板/head/API：锁文件和对照测试挡住语义回归，不只检查 JSON 格式（T00、T04、T11）。

## 最终用户体验：作为验收脚本实现

以下命令为仓库建成后的本地构建验收流程，不代表镜像已存在：

```bash
docker build -t jevserve:dev .
docker run --rm --gpus 'device=0' --shm-size=2g \
  -p 127.0.0.1:8000:8000 \
  -v jevserve-cache:/var/lib/jevserve \
  jevserve:dev serve --model ZefanCai/Open-Jev-2B
```

正式 release 的 README 以已发布的 OCI registry 地址替换本地镜像，不要求用户构建。提供默认模型时也支持省略 serve/model 参数。发布前必须先获得实际 registry namespace，禁止印出虚构可用地址。

```bash
docker compose up -d
curl -fsS http://127.0.0.1:8000/readyz
curl -fsS http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"state":"I was charged twice.","questions":{"team":{"type":"choice","instructions":"Which team should handle this?","criteria":{"billing":"Payments and refunds","technical":"Technical problems"}}}}'
```

只有 readiness 通过才调用推理。自动化 quickstart 脚本必须轮询并有超时，不能靠固定 sleep。认证 fixture 应使用参考模型确认结果，不能仅凭示例语义断言模型必答 billing。

## 仓库地图

```text
README.md / README.zh-CN.md / LICENSE / NOTICE / SECURITY.md
CONTRIBUTING.md / CODE_OF_CONDUCT.md / CHANGELOG.md
pyproject.toml / uv.lock / Makefile / Dockerfile / .dockerignore
compose.yaml / compose.offline.yaml / .env.example
src/jevserve/
  cli.py / config.py / lifecycle.py
  contracts/{requests,responses,errors,capabilities}.py
  artifacts/{manifest,registry,prepare,verify}.py
  adapters/{base,open_jev,laya}.py
  runtime/{base,vllm_backend}.py
  server/{app,routes,auth,admission,execution,metrics}.py
  sdk/client.py
models/registry.yaml / models/locks/
compat/engine-lock.json / compat/systemone.openapi.json
examples/{request.json,python_client.py}
scripts/{quickstart_check,offline_check,upgrade,rollback}.py
benchmarks/{load,compare,report}.py
validation/{reference,parity,soak}.py
tests/{unit,contract,integration,gpu,container}/
deploy/prometheus.yml / deploy/grafana/
docs/{architecture,quickstart,models,configuration,offline,operations,adapter-guide,benchmark-methodology}.md
.github/workflows/{ci,image,gpu-validation,release}.yml
.github/ISSUE_TEMPLATE/ / .github/PULL_REQUEST_TEMPLATE.md
```

T06 的测例工具 FakeRuntime 放在 tests/conftest.py，仅用于 CPU 生命周期测试，绝不进入发布包或运行时 fallback。laya.py 仅在 T09 真正实现时创建。

## 接口约定

- `DecisionRequest`：state(str|dict)、questions(dict[str, Question])、deadline_ms(int|None)。Question 为 ChoiceQuestion、NoulQuestion、ScoreQuestion 的 discriminated union。
- `DecisionResponse`：answers、request_id、model_revision、adapter_revision、execution_profile、probability_kind、calibration_id。校准为空也显式返回。
- `PreparedModel`：不可变 manifest 和已验证的本地文件路径。
- `ModelAdapter.capabilities() -> Capabilities`；`encode(request) -> EncodedDecision`；`decode(raw: RawDecision) -> Answers`。
- `EncodedDecision`：有序问题 ID、引擎输入、readout metadata、输入 token 数及候选数；保留候选顺序，禁止全局字典排序改变语义。
- `Runtime.start(model: PreparedModel) -> None`；`infer(encoded: EncodedDecision, request_id: str, deadline: float) -> RawDecision`；`abort(request_id: str) -> None`；`healthy() -> bool`；`close() -> None`。start/infer/abort/close 为 async。
- `deadline` 使用进程 monotonic clock 的绝对值，不能把墙上时间或跨进程 monotonic 值直接转发。
- `create_app(runtime: Runtime, adapter: ModelAdapter, config: ServerConfig) -> FastAPI`。
- `DecisionClient.decide(request: DecisionRequest) -> DecisionResponse`；客户端默认不重试。

## 任务依赖和里程碑

```text
T00 → T01 → T02 → T03 → T04 → T05 → T06 → T07 → T08
                                              ├→ T09
                                              └→ T10
T08 + T10 → T11 → T12
```

T09 第二模型不阻塞 alpha，但不得用它填充未经验证的支持矩阵。T10 的 alpha 子集先完成，24/72 小时检查分别用于 rc/stable。默认顺序执行；如以后选择并行，T09 与 T10 可独立进行。

## T00 — 冻结参考实现、版本与 GPU profile（2–3 工程日）

**文件：** compat/engine-lock.json、models/locks/open-jev-2b.json、docs/models.md、validation/reference.py、tests/contract/test_release_lock.py。

**输入：** 设计、上游仓库与模型卡。**输出：** 锁定的版本组合、参考数据与模型许可证结论。

- [x] 查验模型卡/许可证与 tokenizer/head 实现，记录 upstream commit，禁止用 main 作为 release revision。
- [ ] 验证 vLLM 候选发行版及官方镜像可获取性，记录 digest、Python/PyTorch/CUDA 与最低驱动要求；vllm-jev 0.3.6/vLLM 0.29.0 仅作候选。
- [ ] 在实际 NVIDIA 机器记录 GPU 型号、显存、驱动；运行模型作者参考实现，建立至少 100 条合成/许可允许的用例，覆盖三种 question、中文、空值/边界和近并列选项。
- [x] 锁文件检查先失败，再实现检查器；命令 `pytest tests/contract/test_release_lock.py -q`。

```python
def test_release_lock_pins_artifacts(release_lock):
    assert release_lock.model_revision != "main"
    assert release_lock.engine_digest.startswith("sha256:")
    assert release_lock.tokenizer_digest
    assert release_lock.reference_report_sha256
```

- [ ] 若目标模型无法在候选引擎实现原始 head 语义，记录阻塞原因并修正适配策略；不能改用普通 LLM JSON 输出冒充完成。
- [ ] 提交基线和机器可读报告。GPU 暂缺时任务保持未完成，可先进行后续 CPU 独立工作。

**验收：** 一个真实模型的参考执行可复现；每个发行依赖有可追溯来源。

## T01 — 工程骨架、协议与 CPU 测试基础（2 工程日）

**文件：** pyproject.toml、uv.lock、src/jevserve/contracts/*、config.py、tests/conftest.py、tests/contract/test_requests.py、tests/unit/test_config.py。

**输入：** T00 锁定 Python/引擎边界。**输出：** 上述公共类型和可安装包，核心类型不导入 CUDA。

- [x] 创建包结构和 dev 依赖、ruff、类型检查、pytest markers；GPU 依赖和纯 CPU 合约测试分离。
- [x] 先写下列契约测试，再实现 discriminated union、配置和错误模型。

```python
@pytest.mark.parametrize("count", [0, 17])
def test_question_count_rejected(make_request, count):
    with pytest.raises(ValidationError):
        DecisionRequest.model_validate(make_request(question_count=count))

def test_choice_order_is_preserved(valid_request):
    req = DecisionRequest.model_validate(valid_request)
    assert list(req.questions["team"].criteria) == ["billing", "technical"]
```

- [x] 补测非有限数值、空 instructions、重复/非法字段、Score 至少两级、超过 1 MiB 在 HTTP 层处理的约定；明确定义允许的空 state 行为为拒绝。
- [x] 执行 `pytest tests/contract/test_requests.py tests/unit/test_config.py -q`、`ruff check .` 和类型检查，记录红/绿结果后提交。

**验收：** 模型能力与协议结构分开，CPU CI 可运行且不下载权重。

## T02 — 不可变模型包与安全准备流程（3 工程日）

**文件：** artifacts/*、models/registry.yaml、tests/unit/test_prepare.py、tests/unit/test_manifest.py。

**接口：** `prepare_model(model_id: str, cache_dir: Path, offline: bool) -> PreparedModel`；下载函数可注入测试替身；manifest 只允许相对包内路径。

- [x] 先写缓存命中、错误 digest、离线缺失、并发锁、中断及目录穿越测试。

```python
def test_interrupted_download_is_not_ready(preparer, model_id, cache_dir):
    preparer.fail_after_bytes = 1024
    with pytest.raises(DownloadError):
        preparer.prepare(model_id, cache_dir)
    assert not preparer.ready_path(model_id, cache_dir).exists()

def test_offline_missing_never_downloads(preparer, model_id, cache_dir):
    with pytest.raises(ArtifactMissing):
        preparer.prepare(model_id, cache_dir, offline=True)
    assert preparer.download_calls == 0
```

- [x] 实现锁、staging、hash、原子发布与 bounded retry；staging/final 在同一文件系统；中断只清理自己拥有的 staging。
- [x] 权重身份绑定 manifest；读取令牌文件；拒绝包外路径与符号链接逃逸；实际磁盘不足返回可读错误。
- [x] 执行 `pytest tests/unit/test_prepare.py tests/unit/test_manifest.py -q` 并提交。

**验收：** 准备与首次 serve 复用同一路径；半成品不能 ready，离线无隐式网络。

## T03 — Open-Jev 模型适配器与输出语义（3–5 工程日）

**文件：** adapters/base.py、adapters/open_jev.py、contracts/capabilities.py、tests/unit/test_open_jev.py、tests/gpu/test_reference_encoding.py。

**接口：** ModelAdapter、EncodedDecision、RawDecision、Answers；字段结构遵循接口约定。

- [ ] 冻结参考 token 序列/特殊标记、question order 与 readout 位置；先测试 encode，再实现。

```python
def test_encoding_matches_reference(adapter, reference_case):
    encoded = adapter.encode(reference_case.request)
    assert encoded.engine_inputs == reference_case.engine_inputs
    assert encoded.readout_metadata == reference_case.readout_metadata

def test_nonfinite_scores_fail(adapter, nan_raw):
    with pytest.raises(InvalidModelOutput):
        adapter.decode(nan_raw)
```

- [ ] 实现专用 head/readout 必需的模型注册；共享上游实现时保留 attribution。softmax 如需执行，在适当精度下计算；若上游已经返回概率，不能重复 softmax。
- [ ] 补测 label 正负映射、Score 期望、不同候选长度与 mask；不得静默短名单裁剪候选。
- [ ] 执行 CPU 单元检查及 GPU 编码对照，提交。

**验收：** 编码与作者参考策略一致；capabilities 描述真实上限；readout 语义有文档。

## T04 — vLLM 执行桥与真实推理闭环（4–6 工程日）

**文件：** runtime/base.py、runtime/vllm_backend.py、validation/parity.py、tests/gpu/test_runtime.py。

**接口：** Runtime 的五个方法；版本差异全部限制在 runtime 模块。

- [ ] 为 start/infer/abort/close 写接口测试，禁止测试环境自动 fallback 到 mock。
- [ ] 接入冻结版本的公开插件/engine API；请求 ID 映射到全部子问题并在 abort 时清理；禁止把本机 Python 锁当成 GPU batching。
- [ ] 先执行参考对比测试观察失败，再修正模型映射直到通过。

```python
@pytest.mark.gpu
async def test_runtime_matches_reference(runtime, adapter, reference_case):
    raw = await runtime.infer(adapter.encode(reference_case.request), "parity-1", monotonic() + 30)
    answers = adapter.decode(raw)
    np.testing.assert_allclose(answers.vector(), reference_case.probabilities,
                               atol=reference_case.atol, rtol=reference_case.rtol)
```

- [ ] 容差必须在参考测试前写入 profile，并报告绝对误差及标签翻转，不能为了通过而不断放宽。近并列样本单独统计。
- [ ] 命令 `pytest tests/gpu/test_runtime.py -m gpu -q`；提供真实 GPU 成功报告后提交。

**验收：** 直接调用 runtime 可以执行三种问题类型，退出释放 worker；至少默认 checkpoint 可用。

## T05 — API、准入、取消与健康生命周期（4–5 工程日）

**文件：** server/*、lifecycle.py、tests/contract/test_http.py、tests/unit/test_admission.py、tests/integration/test_lifecycle.py、compat/systemone.openapi.json。

**接口：** create_app；async admission lease 在任何退出路径释放；一个 API worker 对一个 engine。

- [ ] 先用 FakeRuntime 写错误码、排队超时、断连、初始化失败与退出测试。

```python
async def test_timeout_releases_capacity(client, fake_runtime, admission):
    fake_runtime.delay_s = 1
    response = await client.post("/v1/decisions", json=short_deadline_request())
    assert response.status_code == 504
    assert admission.active == 0
    assert fake_runtime.aborted_ids

async def test_readiness_tracks_engine_failure(client, fake_runtime):
    fake_runtime.is_healthy = False
    assert (await client.get("/readyz")).status_code == 503
```

- [ ] 实现 body limit、鉴权、模型能力拒绝、有界等待和 deadline 全链路预算；不无限等待 semaphore。
- [ ] 实现三种问题结果及兼容响应；认证和非认证错误都带 request_id，日志不包含原始输入和 key。
- [ ] 使用冻结官方 SDK 做真实兼容测试；不能仅以自行创建的 schema 宣称兼容。
- [ ] 实现阶段状态机 downloading/loading/warming/ready/draining/failed；fatal engine failure 退出非零；SIGTERM drain/abort/close。
- [ ] 执行 contract/unit/integration 测试，进行输入与鉴权安全审查并提交。

**验收：** HTTP 真推理闭环；取消不泄漏容量；健康状态与 engine 一致。

## T06 — CLI、SDK 与可诊断启动（2–3 工程日）

**文件：** cli.py、sdk/client.py、examples/*、tests/unit/test_cli.py、tests/contract/test_sdk.py。

**接口：** serve/prepare/models list/doctor/check/benchmark；DecisionClient 使用稳定主接口。

- [ ] 用 CliRunner 测试默认 serve、显式 serve、首参数为 --model 时的兼容归一化与未知参数报错。

```python
def test_unknown_model_exits_before_engine(cli_runner, engine_factory):
    result = cli_runner.invoke(app, ["serve", "--model", "unknown/model"])
    assert result.exit_code != 0
    engine_factory.assert_not_called()
```

- [ ] serve 首次自动调用 prepare，允许 `--offline --model-path /models/default`；没有已认证默认模型时拒绝构建 release，不从互联网上猜测默认项。
- [ ] doctor 检查 GPU 可见性、驱动兼容、缓存可写和磁盘空间；不能承诺仅检查显存就证明可运行。
- [ ] check 等待 ready 并执行三种真实问题，失败非零；SDK 传播 error.code/request_id，默认不重试。
- [ ] 运行 CLI/SDK 测试与一个真实端到端 check 后提交。

**验收：** 用户可以读日志定位下载、权限、显存不足和模型不支持等问题。

## T07 — 单镜像、一条 docker run（3 工程日）

**文件：** Dockerfile、.dockerignore、tests/container/test_image.py、scripts/quickstart_check.py。

- [ ] 先定义容器验收用例：空命名卷、非 root、无宿主 Python、首次下载、重启复用、SIGTERM、无 GPU 的清晰失败。

```python
@pytest.mark.container
@pytest.mark.gpu
def test_fresh_container_serves_real_model(container_runner):
    with container_runner.fresh_cache(image="jevserve:dev") as service:
        service.wait_ready(timeout_s=1800)
        result = service.post_fixture("choice")
        assert result.status_code == 200
        assert service.exec_check().returncode == 0
```

- [ ] 基于 T00 digest 构建镜像；固定依赖并执行安装时版本检查；禁止构建后再升级 PyTorch 破坏 ABI。
- [ ] 配置 exec ENTRYPOINT、默认 CMD、非 root UID、cache 目录所有权、healthcheck、共享内存需求；不默认 privileged/host IPC。
- [ ] 测试三个入口：默认启动；serve --model；--model；测试命名卷与 host bind mount 权限说明。
- [ ] 运行 `docker build -t jevserve:dev .`、`pytest tests/container/test_image.py -m 'container and gpu' -q`。
- [ ] 记录首次下载/冷启动/热启动的独立时间和峰值显存后提交。

**验收：** 默认模型可一条 docker run 启动；不是仅 build 成功。

## T08 — Compose、离线与升级回滚（3–4 工程日）

**文件：** compose.yaml、compose.offline.yaml、.env.example、scripts/offline_check.py、scripts/upgrade.py、scripts/rollback.py、tests/container/test_compose.py、docs/offline.md、docs/operations.md。

- [ ] 默认 inference 单服务；GPU 0、loopback 端口、命名卷、shm_size、45s stop grace；tools/observability profile 非默认。
- [ ] compose 中的镜像地址允许环境覆盖，本地开发默认 jevserve:dev；release 工具生成实际发布镜像的示例配置。
- [ ] 测试 `docker compose config`，再测全新空卷真实启动；不要求用户先执行 prepare。

```python
@pytest.mark.container
@pytest.mark.gpu
def test_offline_restart_without_network(offline_service):
    offline_service.start(network_disabled=True)
    offline_service.wait_ready(timeout_s=600)
    assert offline_service.exec_check().returncode == 0

def test_rollback_restores_full_release(deployment):
    old = deployment.release_identity()
    deployment.upgrade_to_bad_release()
    deployment.rollback()
    assert deployment.release_identity() == old
```

- [ ] 离线工具导出镜像、模型 manifest、checksum 和配置；模型许可证允许才导出权重，且不自动对外上传。
- [ ] 更新脚本记录完整 release identity（镜像、模型、适配器、策略），单卡默认停止旧服务后升级，失败恢复旧配置/镜像；缓存清理为显式独立命令，不随回滚删除数据。
- [ ] 测试损坏权重、下载中断、引擎崩溃、restart 上限；不把 Compose healthcheck 当 watchdog。
- [ ] 提交 Compose 和演练记录。

**验收：** docker run 与 Compose 同一语义；离线真推理成功，回滚恢复完整版本。

## T09 — 第二模型与适配器开发指南（可选 alpha，3–5 工程日）

**文件：** adapters/laya.py、models/locks/laya.json、tests/gpu/test_laya.py、docs/adapter-guide.md。

- [ ] 先生成 Laya 作者参考结果与许可证记录，测试 variable candidate count 和 mask，再实现适配。

```python
@pytest.mark.gpu
async def test_candidate_count_does_not_mix_requests(laya_runtime, mixed_cases):
    outputs = await run_mixed_cases(laya_runtime, mixed_cases)
    for case, output in zip(mixed_cases, outputs):
        assert set(output.probabilities) == set(case.labels)
        assert abs(sum(output.probabilities.values()) - 1) < 1e-5
```

- [ ] `run_mixed_cases` 在 tests/gpu/helpers.py 实现为并发提交带唯一 request_id 的调用并按 ID 还原顺序，不能依赖 engine 返回顺序。
- [ ] 执行与 T04 同规格参考/批处理验证；失败时不添加 validated 标签。
- [ ] 文档解释 encode/readout/能力/fixtures/认证流程，包含一个可运行的现有适配器示例；提交。

**验收：** 新模型无需修改 API 或 CLI；不能只有空类或 NotImplementedError。

## T10 — 监控、比较基准与长稳门禁（3–5 工程日 + 测试运行时间）

**文件：** server/metrics.py、benchmarks/*、validation/soak.py、deploy/*、tests/unit/test_metrics.py、docs/benchmark-methodology.md。

- [ ] 先测试 metric 无高基数/敏感标签，取消及错误也计入分母，再实现指标。

```python
def test_metrics_do_not_include_request_content(metrics_text):
    assert "request_id=" not in metrics_text
    assert "customer_message=" not in metrics_text
```

- [ ] 实现开放式到达负载和闭环并发两种模式；输入组合固定 128/512/2048 token 档、2/4/16 候选（超模型能力的档明确标 N/A），1/4/16 问题；并发 1/4/16/32。
- [ ] 固定同一 GPU、checkpoint、模板、精度、输入和缓存策略，对比作者实现、vllm-jev 固定 commit、JevServe；如果执行语义不同，不能作等价性能优劣结论。
- [ ] 输出成功/失败/拒绝/超时/弃权比例、端到端 P50/P95/P99、有效 decisions/s、输入 tokens/s、峰值显存；原始结果与硬件信息一起保存。
- [ ] 概率误差与标签翻转分别汇总，近并列样本单列；吞吐提升不能掩盖质量下降。服务层额外延迟建议预算为同机同负载 P95 增量 ≤ max(5ms, 直连P95的10%)，正式阈值在压测前记录，不作未经测量的承诺。
- [ ] alpha：至少 1 小时混合真实负载和故障检查；rc：24 小时；stable：72 小时。预热后按窗口记录内存，调查持续增长，不把正常缓存上涨自动判为泄漏。
- [ ] 监控 profile 只作为可选依赖；提交报告生成器和已实测结果，缺 GPU 数据显式标记未执行。

**验收：** 基准可复现；未经测量不写“比 vllm-jev 更快/更稳定”。

## T11 — CI、镜像发布与兼容矩阵（2–3 工程日）

**文件：** .github/workflows/*、tests/contract/test_release_manifest.py、docs/models.md、SECURITY.md、NOTICE。

- [ ] PR CI 执行 lint、类型、CPU unit/contract、package build、Compose config；GPU 测试独立工作流。
- [ ] GPU runner 仅运行已批准的可信代码，不把 fork PR 的未审查代码送入带 secrets/内网权限的机器。
- [ ] image job 构建待测镜像；GPU job 验证准确的 image digest；release job 只晋级被验证的同一 digest，不能重新构建另一个镜像。

```python
def test_release_is_exact_tested_image(release, gpu_report):
    assert release.image_digest == gpu_report.image_digest
    assert release.model_revision == gpu_report.model_revision
    assert gpu_report.required_checks_failed == 0
```

- [ ] 生成 SBOM、签名、第三方声明和 release manifest；关键漏洞发布前修复或给出具体适用性判断及期限。
- [ ] 兼容矩阵逐行包含 checkpoint revision、GPU、dtype、image digest、问题类型、测试等级、限制；engine 支持不等于模型支持。
- [ ] release 发布需要实际远端 repo/registry 权限；无权限时完成本地镜像/归档和发布演练，明确尚未发布，不伪造链接。
- [ ] 提交 CI 与验证报告。

**验收：** 有 GPU 证据才能标 validated；打 tag 不会绕过门禁。

## T12 — 开源完善度与全新主机验收（2 工程日）

**文件：** README.md、README.zh-CN.md、LICENSE、CONTRIBUTING.md、CODE_OF_CONDUCT.md、CHANGELOG.md、docs/*、issue/PR templates。

- [ ] README 前屏使用 JevServe 品牌和 “Serve Jev-style models. One command.” 标语，注明独立项目身份；给出支持的默认模型、宿主前提、真实镜像地址、docker run、Compose、curl、模型限制和许可证说明。
- [ ] 提供架构图、添加模型指南、开发安装、API schema、配置表、错误码、离线/升级/回滚和故障手册。
- [ ] 将 quickstart 从空缓存到第一条真实决策完整运行一遍，记录安装环境和耗时；检查至少“驱动不匹配、显存不足、下载失败、缓存无权限”四类诊断。

```python
@pytest.mark.release
@pytest.mark.gpu
def test_documented_quickstart_on_clean_host(clean_host):
    result = clean_host.run_documented_quickstart()
    assert result.container_exit_error is None
    assert result.ready
    assert result.real_decisions_passed == 3
```

- [ ] clean_host 是外部干净 Linux GPU runner fixture，在 tests/container/conftest.py 连接预置 runner 并使用唯一命名卷；只清理测试自身资源，不删除用户 Docker 数据。
- [ ] 运行整个 alpha 必需门禁；按已完成证据选择 alpha/rc/stable 标记；独立审查源码和发布内容。
- [ ] 交付 repo 源码、Git 历史、可运行镜像/归档、Compose、锁文件、报告、文档；若授权并具备权限再公开发布。

**验收：** 新用户照 README 可启动真实模型；repo 无伪实现、无默认 secret、无虚构 benchmark。

## 工期与资源

单人顺序约 30–45 工程日（不含可选 T09），以 T00 的适配难度修正；2 名有相关经验的工程师约 4–6 周可争取 alpha，生产候选需要额外测试与修复。长稳 wall-clock 另计，不能通过“计划完成”替代。

需要：至少一台可运行默认 checkpoint 的 Linux NVIDIA GPU 主机、GPU runner 时间、镜像存储、可访问模型源；若要公开发布还需要 repo 和 registry 的写权限。建议先用单张 24GB 或更大显存卡建立 profile，这只是资源预算假设，最低支持显存由实测决定。未取得这些资源时，CPU 任务可完成，GPU 和发布任务保持未完成。

## 总验收清单

- [ ] 一个真 checkpoint 在认证 GPU 上通过 docker run 和 Compose 两条路径。
- [ ] 首次下载和缓存重启均通过；无宿主 Python 依赖。
- [ ] 三种问题类型正确；官方 SDK 兼容范围有测试；无伪推理 fallback。
- [ ] 参考结果、并发一致性、取消、过载、崩溃和 SIGTERM 测试通过。
- [ ] 离线可运行；损坏包失败；完整版本可回滚。
- [ ] 镜像、权重、模板、适配器均锁定，许可证清晰。
- [ ] 有公开可复现的验证报告，支持矩阵只包含实测组合。
- [ ] 文档、贡献指南、安全报告渠道、CI 与 release 路径完整。
- [ ] 最终交付明确是 alpha 雏形、rc 还是稳定版，不以仓库体量冒充成熟度。

## 方案自审结论

已将用户的单机 NVIDIA / Compose 约束贯穿设计；默认单容器替代早期强制多服务部署；首次自动准备与离线只读模式不冲突；GPU 门禁明确不允许 mock 替代；Triton 和集群工作不进入首版。需要在实施 T00 实测生成的版本/硬件锁定值被设置为前置交付，未虚构 digest、镜像地址和 benchmark。此计划完成不意味着实现或 GPU 验证完成。
