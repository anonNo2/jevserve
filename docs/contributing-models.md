# Contributing a model integration

Start with the [model matrix](models.md) and [contribution workflow](../CONTRIBUTING.md).
Seven pinned profiles currently have GPU/container execution evidence, but none
is certified. Open-Jev/JPT probability parity fails; NeoHorse/Kev only pass a small
synthetic suite. A new catalog entry must remain a development profile requiring
`--allow-uncertified` until its own certification gates pass.

JevServe runs one model on one GPU per container, using one generic runtime image
and separately mounted weights. Current requests are text Choice/Noul/Score.
Do not advertise vision support because a checkpoint has a vision encoder.
An additional modality needs its own API, preprocessing, resource bounds and
real reference/container evidence. Ordinary JSON generation is not a substitute
for a Jev-style decision head.

## 1. Establish provenance and semantics

Open a [model request](https://github.com/anonNo2/jevserve/issues/new?template=model_request.md)
with the checkpoint and author implementation links. Before implementation:

- Verify checkpoint availability and a full immutable revision; separately pin
  any base model, tokenizer, template, custom code and dependency versions.
- Record source URLs, file sizes and SHA-256 hashes. Check checkpoint, base-model
  and code licenses independently, including commercial-use and redistribution
  restrictions. Preserve attribution in [NOTICE](../NOTICE) and `licenses/` when
  reusing code; the runtime license does not relicense weights.
- Describe exact tokens, template, question/candidate order, masks, readout
  positions, temperature and head/backbone precision from the author's code.
  State whether a readout returns logits or probabilities and how Noul labels,
  Score expectation, ties and confidence are defined.
- Obtain a runnable, pinned author reference on a real GPU. Record any reference
  configuration correction explicitly. An inaccessible upstream or unknown license
  is an unresolved prerequisite, not successful verification.

Use synthetic, publishable fixtures with representative bounds: Chinese and
structured state, mixed question types, reordered/unequal candidate counts, ties,
null descriptions, boundary markers and maximum permitted work. Reject unsupported
inputs explicitly. Freeze comparison rules before measuring; preserve the
[existing parity tolerances](../compat/parity-profile.json).

## 2. Implement the bounded profile

The following files describe the current integration seams; there is no automatic
third-party model plugin discovery. Keep model differences out of public request
schemas unless a separately reviewed capability requires a schema change.

| Concern | Existing implementation and required change |
|---|---|
| Immutable inventory | Add a profile under [`src/jevserve/data`](../src/jevserve/data), following an existing family; register it in [`artifacts/catalog.py`](../src/jevserve/artifacts/catalog.py). Include every required checkpoint/base file. |
| Verified preparation | Use [`ModelManifest`](../src/jevserve/artifacts/manifest.py) and existing preparation/verification. Preserve staging, hashes, atomic publication, corruption rejection and network-free offline behavior. Never ship weights in the image. |
| Encoding and decoding | Implement [`ModelAdapter`](../src/jevserve/adapters/base.py): `capabilities()`, `encode()` and `decode()`. Preserve all `EncodedDecision` order/readout metadata and `RawDecision` meaning. |
| Work bounds | Declare actual limits in [`Capabilities`](../src/jevserve/contracts/capabilities.py), including candidate, input and padded-token costs where applicable. Enforce model-specific state/context bounds without truncation. |
| Execution | Select an explicit profile in [`cli.py`](../src/jevserve/cli.py). Reuse the matching runtime or implement a bounded backend with start/infer/abort/close/health behavior. New architectures may require [`registration.py`](../src/jevserve/runtime/registration.py) and model loading changes. |
| Reference and comparison | Extend the family-aware [`matrix harness`](../validation/model_matrix.py) and [`coordinator`](../validation/run.py), or the Open-Jev reference path, with real reference execution and checked report schemas. An unknown family must fail closed. |

Use [Open-Jev](../src/jevserve/adapters/open_jev.py) for scalar candidate pooling,
[JPT](../src/jevserve/adapters/jpt.py) for masked token readout, and
[NeoHorse/Kev](../src/jevserve/adapters/neohorse.py) for pointer readout as concrete
examples. These families have different limits and probability semantics; copying
a profile name alone does not establish compatibility.

Keep GPU imports out of CPU contracts and test collection. The server owns one
engine; additional web workers must not create extra model copies. Propagate
deadlines, cancellation and fatal failures. Never fall back to a different model,
backend, precision or mock output; never retry inference implicitly. Do not apply
softmax twice. Softmax confidence is not calibrated accuracy.

## 3. Verify and report each gate

Run CPU checks from the repository root in the project Python 3.12 environment:

```bash
.venv/bin/pytest -q
.venv/bin/ruff check .
.venv/bin/mypy
.venv/bin/python -m build --no-isolation
```

Add meaningful CPU tests for encoding metadata, order, masks, decode semantics,
nonfinite values, capabilities and failure boundaries. Tokenizer/reference replay
that needs external artifacts is a separate check; CPU tests must not download
weights or import GPU packages. Use existing adapter tests as executable examples.

Follow [Reproduce model validation](validation.md) for exact reference setup,
immutable-image commands, report fields and exit codes. Run `--plan` first; extend
the coordinator for a new family before claiming it can validate that family.
Inspect GPU/Docker use before execution and run GPU stages serially. Clean up only
your own containers/processes/volumes. Do not change the shared driver or daemon.

The coordinator compares author reference and container HTTP/SDK responses; it
does **not** cover Compose, network isolation, performance or full certification.
Use the [deployment guide](quickstart.md) for both container paths and record
additional checks separately. Keep failed probability comparisons visible even
when labels agree. For diagnosis, compare tokens/masks, readout logits, temperature,
precision and batching individually; see [numerical diagnosis](numerical-diagnosis.md).

Copy this checklist into the PR and write **passed / failed / not run / not
applicable (reason)** beside each item, with a command and evidence link. A checked
box alone is insufficient evidence:

- [ ] Immutable checkpoint/base/tokenizer/author revisions, hashes, licenses and
  attribution verified; staged preparation and offline integrity checks covered.
- [ ] CPU contract, adapter, limits and failure-path tests passed without GPU
  dependencies or downloads; lint, types and package build recorded.
- [ ] Real author reference and encoding/readout comparison recorded with exact
  versions, dtypes and any reference modifications.
- [ ] Real runtime probabilities compared at frozen tolerances, including absolute
  errors, all label flips, near ties and batch/candidate-order consistency.
- [ ] Native HTTP and supported official SDK subset exercised for all declared
  question types; cancellation, overload, deadlines and fatal-worker behavior checked.
- [ ] Same immutable generic image ran real inference via both `docker run` and
  Compose, with external weights; cold preparation, restart reuse, offline/network
  isolation, permissions and graceful shutdown recorded separately.
- [ ] Public reports bind source revision/dirty state, model and image identity,
  runtime/GPU/precision, fixture scope and failures; private paths, request data,
  credentials and host identifiers removed before publication.
- [ ] [Model matrix](models.md), both README capability/status summaries and
  [STATUS](development/STATUS.md) agree; material changes recorded in
  [DECISIONS](development/DECISIONS.md). Remaining gates have exact next actions.

Development integration and release certification are separate decisions. Keep
[`models/registry.yaml`](../models/registry.yaml) uncertified until the required
evidence exists. Alpha also needs the real-inference/container and mixed-load/fault
gates; RC adds an actual 24-hour soak, stable adds 72 hours and a clean-host install.
Rollback must be exercised. Small synthetic comparisons and successful builds
cannot stand in for those checks. Follow the [release workflow](releasing.md) for
evidence and promotion; a model PR does not authorize publishing an image.
