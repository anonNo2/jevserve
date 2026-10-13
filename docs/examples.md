# Business examples / 业务示例

These three synthetic examples call a running JevServe model through the native
Python SDK. They return real responses, including model revision, adapter,
execution profile and calibration metadata; no expected prediction is hardcoded.
They propose support decisions for human inspection and do not modify a ticket,
remove a post, or page an on-call engineer.

三个示例分别使用 Choice（工单分流）、Noul（内容审核）和 Score（优先级评分）。
输入为公开的合成文本；输出完全来自真实服务。示例只展示决策建议，不执行工单转派、
删帖或告警，也不声称这些模型已通过业务准确率评测。

## Start the service and client

From a repository checkout, first inspect GPU occupancy with `nvidia-smi`.
Follow [Quick Start](quickstart.md) for host prerequisites or an existing mounted
model/Compose deployment. To start a dedicated default Open-Jev-2B instance:

```bash
./scripts/quickstart.sh \
  --image anonwx/jevserve@sha256:4b758ad1000f4d38501fd83f0a9c2b32f0cb91968d427b170a2c2b0d9a613aff \
  --name jevserve-examples --volume jevserve-examples-cache --port 18000
```

The helper waits for readiness and leaves the service running. First startup
needs network access for pinned weights; weights are separate from the image.
If a service is already running, reuse its URL instead of launching another
model on the same GPU. These examples require text support only.

The optional Python client needs the repository's Python 3.12 CPU environment;
no local Torch/CUDA installation is needed. With `uv` installed, from the root:

```bash
uv sync --locked
export JEVSERVE_URL=http://127.0.0.1:18000
.venv/bin/python examples/business_examples.py ticket-routing
.venv/bin/python examples/business_examples.py content-moderation
.venv/bin/python examples/business_examples.py priority-scoring
```

`--url http://127.0.0.1:18000` overrides `JEVSERVE_URL`. Use an origin URL without
credentials, a path, query string or fragment. For a service configured with
API-key authentication, append `--api-key-file /path/to/readable/key-file`.
The key is read from the file; never put its contents in a command or URL.
The default loopback example has no authentication configured. Remote access
should use the TLS/authentication setup described in [Quick Start](quickstart.md#authentication).
The client makes one request with a 30-second server deadline and a 35-second
HTTP timeout (adjustable with `--timeout`). It does not retry or change models.

Without a Python client environment, send any fixture directly from the checkout
to the default unauthenticated local instance:

```bash
curl --fail-with-body --max-time 35 \
  http://127.0.0.1:18000/v1/decisions \
  -H 'Content-Type: application/json' \
  --data-binary @examples/business/ticket-routing.json
```

Replace the filename with `content-moderation.json` or `priority-scoring.json`.
For authenticated access, prefer the Python client's key-file option.

## Interpret each response

| Command | Input and rubric | Where to read the result | Business use |
|---|---|---|---|
| `ticket-routing` | Duplicate subscription charge; billing, technical, account teams in fixed order | `response.answers.team.choice` and `.probabilities` | Present the proposed team and full candidate distribution to the support agent before assignment |
| `content-moderation` | A forum disagreement; only targeted insults or threats violate this example's policy | `response.answers.policy_violation.noul` | Show the probability of **true = policy violation** to a reviewer; do not interpret it as a safe/unsafe label or auto-remove a post |
| `priority-scoring` | Checkout outage; Routine, Soon, Urgent, Immediate in ascending order | `response.answers.priority.score`, `.probabilities`, `.legend` | Show the ordinal expected score and level distribution to the incident reviewer |

For Score, the four levels have indices **0, 1, 2, 3**. The score is
`sum(index * probability)` and can be fractional; it is not an SLA in minutes,
a predicted loss or a probability. Do not round it into an automatic response
policy without separately validating that policy. Changing candidate order,
rubrics or text changes the request and can change results.

All probabilities are relative model outputs, **not calibrated accuracy**.
The native Noul value describes the true proposition; Choice distributions are
conditional on the supplied candidates. `calibration_id: null` means no calibration
is identified. No universal acceptance/abstention threshold is established here.
Evaluate representative labeled examples and human-review requirements before
integrating a business action. These three requests are smoke examples, not a
quality benchmark, and no particular answer is guaranteed.

概率不等于准确率。Noul 数值对应“违反本例政策”为真的概率；Score 是从 0 开始的
有序等级期望，不能直接转换成分钟或损失金额。当前 Open-Jev/JPT 数值对照仍有失败；
NeoHorse/Kev 仅通过小型合成对照。所有模型仍为未认证的开发预览。

## Errors and exit status

The Python runner prints JSON only after the SDK validates the response and the
runner matches its questions, candidate order and Score rubric to the request.
It exits **0** on a valid response, **1** on service/transport failure and **2**
on invalid configuration, unreadable files or malformed/mismatched responses.
Failure messages omit input text, credentials and remote error bodies. No default
answer is returned on failure.

| Failure | Action before an explicit rerun |
|---|---|
| Connection failure | Check the URL, container state and readiness; first startup may still be loading |
| HTTP 400 / 413 / 422 | Inspect the request schema, body size and selected model's capability bounds; do not truncate silently |
| HTTP 401 / 403 | Check the service/proxy authentication and key-file access |
| HTTP 429 | Reduce concurrent demand or inspect admission limits |
| HTTP 503 | Inspect readiness and runtime health; an operator may need to restart the failed instance |
| HTTP 504 / client timeout | Inspect request cost and deadline; a short timeout does not establish a result |
| Invalid response | Check server/client version and protocol compatibility; do not consume a partial result |

Active cancellation of a Torch pointer request can terminate its worker and
require a service restart; do not use very short deadlines as a readiness probe.
The example does not log request text, but output includes the request ID and
business labels: keep real inputs/results out of public logs and Git.

## Licensing and cleanup

Example code and synthetic inputs follow this repository's Apache-2.0 license.
The runtime license does not grant rights to every model. Consult the pinned
[model/license matrix](models.md) before selecting a checkpoint: Open-Jev,
NeoHorse and Kev list Apache-2.0, while JPT lists **CC-BY-NC-4.0 (noncommercial)**.
A runnable JPT demonstration is not authorization for commercial deployment.
JevServe is independent of TypeSafe AI and JevServe-Bench.

Stop and remove only the example container when finished; retain the cache:

```bash
docker stop --timeout 65 jevserve-examples
docker rm jevserve-examples
```

If you reused an existing service, leave its lifecycle to its owner. For CPU-only
contract/error checks, use `.venv/bin/pytest tests/unit/test_examples.py -q`.
Real GPU execution evidence, when recorded, belongs in [STATUS](development/STATUS.md)
and does not turn these scenarios into accuracy or numerical certification.

## Recorded execution

All three examples executed successfully on 2026-10-13 against JPT-0.8B in the
immutable dev1 container on the development RTX A6000 host. These are execution
examples, not correct-answer labels or quality measurements. The ticket-routing
example also executed after an intentional container crash and explicit cached
restart. Recorded synthetic responses are in
[`validation/reports/hardening-workload`](../validation/reports/hardening-workload).
JPT probability parity remains failed and its separate noncommercial license
still applies.
