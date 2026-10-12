# Compatibility boundary

The initial profile fixes Open-Jev-2B revision
`0c7aa498b1627be8da4acf34c863ff0ee0a92785`, Qwen3.5-2B revision
`15852e8c16360a2fea060d615a32b45270f8a8fc`, BF16 backbone, independent FP32 LoRA,
FP32 scalar head and checkpoint temperature `1.518796342858676`. Candidate order
and last non-padding token readout are preserved. Prefix caching, quantization,
dynamic LoRA, retry and fallback are disabled.

The GPU reference and serving environments are intentionally separate.
`reference-requirements.txt` freezes the author environment; `runtime-requirements.txt`
freezes the candidate vLLM environment. Both include wheel hashes.
`parity-profile.json` was frozen before measurement. It has NOT been relaxed:
latest pretokenized vLLM parity fails 38/113 cases (initial run: 33/113); native HTTP parity fails
36/110 cases, with no observed label flips. This profile remains uncertified.
Batch shape can affect BF16 numerical results; exact batch invariance is not claimed.

## System One subset

The actual `typesafe-sdk==0.7.4` client is exercised in CPU ASGI contract tests
and against the real GPU HTTP service. Primary source inspected:
[typesafe-sdk-python at e70bf99](https://github.com/typesafe-ai/typesafe-sdk-python/tree/e70bf99a89f27102247613bcd5d1aa8923a1cda9).
The SDK is a development dependency, not a server dependency.

Supported `POST /v1/systemone` fields: `model` must be exactly
`ZefanCai/Open-Jev-2B`; `state`, `questions` and optional `deadline_ms` follow the
native schema. Nonempty text/object state; text instructions/descriptions;
Choice null descriptions; optional textual true/false Noul descriptions;
Score 2–10 text levels. Object/array instructions, list state, empty state,
missing instructions, model aliases and hosted billing/account APIs are not supported.
Unknown fields and duplicate JSON keys are rejected. `/v1/models` is the native
capability endpoint, not a promise of SDK model-list compatibility.

Compatibility responses contain `model`, `answers` and `usage`; revision and
execution profile are response headers. `input_tokens` counts actual candidate
sequences (including repeated context); `output_tokens=0` because the scalar
head generates no tokens. Confidence formulas follow the author's MIT-licensed
implementation and describe concentration, not calibrated correctness.

```python
from typesafe_sdk import TypeSafeClient, Choice, RetryPolicy

with TypeSafeClient(
    base_url='http://127.0.0.1:8000',
    api_key='local-development',  # With auth enabled, load the real key from a file.
    model='ZefanCai/Open-Jev-2B',
    retry=RetryPolicy(max_retries=0),
) as client:
    answer = client.system_one(
        state='I was charged twice.',
        questions={'team': Choice(instructions='Choose a team',
                    criteria={'billing': 'Payments', 'technical': 'Software'})},
    )
```

Softmax probabilities and confidence must not be interpreted as measured accuracy.
