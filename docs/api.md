# HTTP API reference

JevServe exposes a synchronous, text-only Choice/Noul/Score API for one configured
model per process. It has no streaming, image input, multi-model routing, automatic
fallback, or general OpenAI API compatibility. Probabilities are candidate softmax
values, not calibrated accuracy. This remains a development preview; API shape
compatibility does not establish model numerical parity.

The versioned [full OpenAPI contract](../compat/openapi-native.json) covers all six
endpoints. The [SystemOne contract](../compat/openapi-systemone.json) covers that
compatibility endpoint and shared health/model/metrics routes. Both are OpenAPI
3.1 / JSON Schema 2020-12, generated without FastAPI or GPU imports. Live
`/openapi.json`, `/docs` and `/redoc` are disabled. These files describe this source
revision; changing source does not update the already-published dev1 container.

## Routes and authentication

| Method and path | Success | Authentication |
|---|---|---|
| `GET /livez` | `200 {"status":"live"}` | Public |
| `GET /readyz` | `200 {"status":"ready"}`; otherwise `503 {"status":"not_ready"}` | Public |
| `GET /metrics` | Prometheus text exposition | Conditional bearer |
| `GET /v1/models` | Single served model ID, revision and capabilities | Conditional bearer |
| `POST /v1/decisions` | Native answers and execution metadata | Conditional bearer |
| `POST /v1/systemone` | SystemOne subset answers and token usage | Conditional bearer |

If the administrator configures `api_key_file`, protected routes require the exact
`Authorization: Bearer <key>` value. Otherwise those routes accept unauthenticated
requests. The OpenAPI security alternatives describe these deployment modes; they
do not mean authentication can be bypassed on an authenticated instance. Keep a
service on localhost or behind an appropriately secured TLS reverse proxy.

Every handled HTTP response carries a server-generated `x-request-id`. Native
success and structured errors also put that ID in the body. `/livez` means the API
process responds; use `/readyz` before inference. Model metadata is available
independently of readiness and does not claim numerical certification.

## Native request

```json
{
  "state": "A customer was charged twice and wants a refund.",
  "questions": {
    "team": {
      "type": "choice",
      "instructions": "Choose the team responsible.",
      "criteria": {"billing": "Payments and refunds", "technical": "Software faults"}
    },
    "urgent": {"type": "noul", "instructions": "Does this require urgent action?"},
    "severity": {
      "type": "score",
      "instructions": "Rate the impact.",
      "criteria": ["Low", "Medium", "High"]
    }
  },
  "deadline_ms": 5000
}
```

- `state` is a nonblank string or nonempty JSON object. Nested JSON values are
  allowed in that object; this is structured context, not an image input format.
- `questions` is an object of 1–16 nonblank question IDs. Each question has a
  nonblank `instructions` string and the lowercase discriminator `type`.
- Choice `criteria` is a nonempty object mapping nonblank candidate IDs to a
  nonblank description or `null`. Candidate insertion order is preserved; the
  first maximum wins ties. A model may require at least two candidates.
- Noul `criteria` may be omitted or `null`, or an object whose only allowed keys
  are `true` and `false` with nonblank descriptions. The parser permits an empty
  or partial criteria object; do not assume it requires both keys.
- Score `criteria` is an ordered list of at least two nonblank level descriptions.
- `deadline_ms` may be omitted, `null`, or an integer from 1 to 30,000. Booleans
  and numeric strings are rejected. Administrator limits may be lower.

Unknown top-level/question fields, duplicate JSON object keys (at any depth),
nonfinite numeric values and unsupported question types are rejected. JSON Schema
captures shapes and bounds, but cannot capture duplicate keys, insertion order,
or all parser/model invariants. JSON Schema also treats an integral JSON number
such as `1.0` as an integer, whereas the strict request parser requires `1` for
`deadline_ms`. The server reads a bounded raw JSON body; send
`Content-Type: application/json` even though media-type enforcement is not used as
the parser gate.

`GET /v1/models` returns `{"models":[{"id":"…","revision":"…","capabilities":{…}}]}`.
Use its capability object for supported question types, minimum/maximum Choice
candidates, maximum Score levels, candidate-sequence budget, token budget per
candidate, total tokens, optional padded token budget and Noul candidate cost.
The deployment can further reduce question/body/deadline limits. Requests are
rejected rather than truncated or silently reassigned.

## Native response

The `answers` object uses the original question IDs:

| Type | Answer fields | Meaning |
|---|---|---|
| Choice | `type`, `choice`, `probabilities` | Candidate ID and normalized distribution over original candidate IDs |
| Noul | `type`, `noul` | Probability of the affirmative answer in `[0,1]` |
| Score | `type`, `score`, `probabilities`, `legend` | Expected ordinal value; distribution/legend keys are ordered strings `"0"` through `"n-1"` |

The response also contains `request_id`, the pinned 40-hex `model_revision`,
`adapter_revision`, `execution_profile`, `probability_kind: "candidate_softmax"`,
and `calibration_id` (currently `null`). Choice/Score probabilities sum to one
within the contract tolerance. Score is `sum(level_index * probability)` and is
not a generated rating string. These arithmetic invariants are validated in the
Python contracts but are not expressible in the static JSON Schema.

## SystemOne compatibility

`POST /v1/systemone` implements the **typesafe-sdk 0.7.4 text-question subset**.
Use the same request fields plus required `model`, exactly matching the sole ID
from `/v1/models`. Other models return `422 unsupported_capability`; no routing or
substitution occurs. The native endpoint does not accept a `model` field.

A successful compatibility response contains `model`, `answers`, and
`usage: {"input_tokens": ..., "output_tokens": ...}`. Answer fields match native
answers except that Choice/Score also include `confidence`. Noul has no added
confidence field. Compatibility success omits native revision/request metadata
from the body: use `x-request-id`, `x-model-revision`, and `x-execution-profile`
headers. Errors use the same JevServe structured error envelope as native requests.

Confidence follows the selected adapter family: Open-Jev Choice uses normalized
excess over uniform probability; Open-Jev Score uses distance from the modal level
relative to uniform; token-readout families use normalized entropy; pointer Score
uses modal distance relative to the level span. These are concentration measures,
not calibrated correctness. Token-readout usage counts candidate engine inputs
as output tokens; pooling/pointer profiles report zero output tokens. Neither
field is a billing promise or generated-text length. See the [model matrix](models.md)
for actual family support and unresolved probability parity.

## Errors, deadlines and cancellation

Structured errors have this shape (messages contain no raw request content):

```json
{"error":{"code":"invalid_request","message":"invalid request"},"request_id":"..."}
```

| HTTP status | Error codes |
|---|---|
| 400 | `invalid_request` |
| 401 | `unauthorized` |
| 413 | `body_too_large` |
| 422 | `unsupported_capability`, `context_too_long` |
| 429 | `overloaded` |
| 503 | `not_ready`, `engine_failed`, `invalid_model_output` |
| 504 | `deadline_exceeded` |

A readiness `503` is the health object above, not this error envelope. Unknown
routes/methods use framework responses and are outside the six-route contract.

The default body bound is 1 MiB, with administrator-selectable lower limits. The
server deadline starts at request entry and includes upload, admission/queueing,
encoding and inference. A request deadline can shorten it, never extend it; a
value above the administrator maximum is invalid. Slow upload can exhaust the
server deadline before the request-specific deadline is parsed. There is no
implicit retry. On a timeout the server aborts engine work and returns `504`.

Disconnect after parsing cancels the inference operation and invokes engine
abort. The implementation uses an empty `499` response if it observes disconnect;
a disconnected caller normally cannot receive it. An encoding thread already
running in Python cannot be forcibly stopped: after timeout/disconnect its work
is contained, readiness stays false until it exits, and new inference is refused.
Shutdown closes admission and drains active work within the configured grace
period before aborting remaining requests. These behaviors do not promise that
an accelerator or third-party tokenizer can be interrupted at an arbitrary instruction.

## Updating the contract

From a Python 3.12 development checkout:

```bash
.venv/bin/python -m validation.api_schema
.venv/bin/python -m validation.api_schema --check
.venv/bin/pytest -q tests/contract/test_api_schema.py
```

Commit both generated files with a protocol change. The tests compare the exports
with the generator, verify local references and schema syntax, compare registered
routes, validate representative real CPU HTTP responses, and check parser/schema
acceptance and rejection examples. They do not certify GPU numerical behavior.
