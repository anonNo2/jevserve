# Numerical certification and new-source image acceptance

The user prioritizes these two remaining audit gates on 2026-10-13. The default
Open-Jev checkpoint remains fixed, and JPT-0.8B is the smaller diagnostic subject
for the shared Qwen3.5 backbone. Frozen probability tolerances, weights, templates,
precision and candidate semantics may not be relaxed to obtain a pass.

1. Inspect and preserve the old dev1 soak outcome; reclaim only its project-owned
   resources for the newly prioritized GPU work. The old soak failed during
   project-daemon loss before an attempted stop signal was sent. Record the actual
   failure and recover only the project-local Docker instance.
2. Capture corresponding HF/vLLM normalized backbone boundaries with identical
   unpadded synthetic token inputs and verified weights. Do not compare vLLM's
   separate residual return with HF's summed layer output as equivalent tensors.
3. Test a single source-supported arithmetic hypothesis at a time. Diagnostic
   counterfactuals are not production fixes or certification. Adopt a correction
   only after real reference/parity evidence at the unchanged tolerances.
4. Build the actual source Dockerfile, first CPU/no-network smoke, then sequential
   real GPU native/SDK, Docker run, Compose, offline and lifecycle regression.
   Bind local candidate checks to the immutable Docker local image ID and label
   that identity accurately; it is not a registry manifest digest or release.
5. Repeat affected reference/parity cases in the final candidate, review evidence,
   update status/decisions and commit. A failure remains a failure; no substituted
   model, HF fallback, changed reference baseline, or guessed release identity.

GPU execution is coordinated by the primary agent only. Independent workers own
layer diagnostics, source/arithmetic investigation and image/validator preparation.
No remote image push or release is authorized by this local acceptance task.
