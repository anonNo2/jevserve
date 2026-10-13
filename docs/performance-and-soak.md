# Performance and soak measurements

The public `validation.workload` runner sends synthetic requests to an already
running, explicitly selected service. It measures HTTP execution, not probability
parity, model accuracy or release certification. It does not start/stop containers.
Use a dedicated validation deployment and preserve its immutable image digest.

```bash
.venv/bin/python -m validation.workload \
  --url http://127.0.0.1:18793 \
  --image-digest sha256:4b758ad1000f4d38501fd83f0a9c2b32f0cb91968d427b170a2c2b0d9a613aff \
  --kind choice --candidates 2 --concurrency 1 --requests 100 \
  --duration 300 --observe-gpu --output performance.json
```

Output must be a new path. The runner performs one separately timed warmup,
then requests in bounded concurrent batches, with no inference retries. Every response must match the valid model revision
observed at startup; a mid-run revision change fails the run. It stops
at the first failed batch, records HTTP/protocol errors separately and exits
nonzero. Every completed batch atomically updates the report. SIGTERM or Ctrl-C
interrupts the run; completed duration gates are never inferred from a process
being started. Credentials can be read from `--api-key-file`. The URL must be an HTTP(S)
origin without user credentials, a path, query or fragment. Errors never echo
exception messages, key contents or remote error bodies. The CLI exits 1 for a
recorded workload/preflight failure, 2 for invalid configuration or report I/O
failure, and 130 on interruption; stderr contains a stable `error_code`.

After configuration validation, an initial report is written before authentication,
readiness, model metadata and warmup. Failures include `stage` and `error_code`,
zero workload requests and `duration_gate_passed=false`. An invalid configuration
or unwritable output location cannot produce a report. Successful requests must
satisfy the complete native response contract and match the requested questions,
Choice candidate order and Score rubric; this is protocol validation, not accuracy.
The report stores metadata and a hash of the fixed synthetic input, not payloads,
credentials, endpoint URLs or individual response bodies.

For a small repeatable matrix, measure concurrency 1 and 4 with 2 and 255 Choice
candidates, each in its own invocation/report. Check `/v1/models` before testing
other candidate counts or question types. `--kind score` interprets candidate
count as score levels; Noul always has two outcomes. The fixed synthetic text,
question type/count, request hash, concurrency and model metadata are recorded.

P50/P95/P99 use nearest rank over completed HTTP latencies. Throughput includes
failed attempts; inspect successful request and error counts. This closed-loop
batch workload is not an open-loop saturation or capacity benchmark. GPU samples
are GPU 0 device totals taken after warmup and each batch, including engine reservation
and any other process; they can miss transient peaks and are not the model's
minimum memory requirement. Cold-start time needs a separate container-start to
readiness measurement. Always record whether the device was shared.

## Long-duration runs

A low-rate 24-hour stability probe can use:

```bash
.venv/bin/python -m validation.workload \
  --url http://127.0.0.1:18793 --image-digest sha256:ACTUAL_TESTED_DIGEST \
  --kind choice --candidates 2 --concurrency 1 --requests 1000000 \
  --duration 86400 --interval 10 --observe-gpu --output soak-24h.json
```

The digest placeholder is intentionally invalid: replace it with the tested
image. For a 72-hour gate use `--duration 259200`. Keep the exact invocation and
container lifecycle logs. `status=passed` alone means the bounded workload
finished without observed errors; **`duration_gate_passed=true` and observed
elapsed time must also meet the requested duration**. The workload clock starts
after warmup and baseline GPU sampling; preflight/warmup never count toward the
duration. Final elapsed time and throughput include the last sampling interval.
`termination_reason=request_limit` identifies an early request-cap stop and
cannot pass the duration gate, even with a nonzero probe interval. Hitting the request limit
early never passes a duration gate. These low-rate tests do not establish
sustained-load capacity or clean-host installation.

Do not start an indefinite job or parallel model runs on a shared GPU. Reserve
the device, set an explicit duration and retain the exact container name for
cleanup. A fresh-cache run on an existing machine is not a clean-host test.

## Observed development-host matrix (2026-10-13)

JPT-0.8B BF16, dev1 digest above, RTX A6000 48 GiB, dedicated idle GPU
at test start, 100 successful requests per cell, no observed HTTP/protocol errors.
The fixed requests encode to 105 input tokens for 2 candidates and 3,946 for
255 candidates, measured offline with the pinned tokenizer after the timed run.
These measurements include coordinator sampling overhead and are not a capacity claim.
The device reservation is not minimum required VRAM.

| Choice candidates | Concurrency | P50 ms | P95 ms | P99 ms | Attempts/s | Sampled peak MiB |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 1 | 74.6 | 87.8 | 95.1 | 10.10 | 24222 |
| 2 | 4 | 139.8 | 156.1 | 161.3 | 23.92 | 24222 |
| 255 | 1 | 117.1 | 138.1 | 143.4 | 6.96 | 24528 |
| 255 | 4 | 337.5 | 441.2 | 460.6 | 8.81 | 24592 |

Fresh runtime-cache container startup through readiness and one real inference took
248.997 seconds; model weights were already verified on the host, and image layers
were cached. This is not download time or a clean-host installation.
[Raw measurements and lifecycle evidence](../validation/reports/hardening-workload).


A 24-hour NeoHorse-Jev-4B low-rate run started on 2026-10-13 at approximately
02:51 UTC after container startup. The [start snapshot](../validation/reports/soak-24h-start.json)
confirms successful requests, but is **not a completed soak result**.
The local live report is `.local/soak/live.json`; consult STATUS for the next
review and final-report publication step. Other GPU validation remains serialized.
