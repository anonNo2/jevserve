# Current status

As of 2026-10-13, JevServe is a **development preview**, not an alpha/stable release
or a certified inference implementation. Source and the pinned dev1 container are
public; no Python package has been published.

| Area | Observed state |
|---|---|
| Text API | Choice, Noul and Score; native client and typesafe-sdk 0.7.4 text subset |
| Models | Seven fixed model profiles have real GPU/container execution evidence |
| Numerical acceptance | All seven profiles pass initial `bounded-drift-v1` on 270 recorded requests / 285 questions; historical strict Open-Jev/JPT comparisons still fail |
| Deployment | Plain Docker and Compose execution verified; clean-host installation still pending |
| New source image | Unpublished `13eed9b` candidate passed seven-model GPU/SDK and two-model offline Compose execution; its recorded corpus passes the bounded policy |
| Security | dev1 SBOM advisory scan completed; findings remain under triage; private reporting still disabled |
| Vision | No image inputs; VL investigation does not establish vision support |
| Release endurance | Old dev1 24-hour run failed after 16,271.64 seconds; no completed 24/72-hour acceptance |

The initial `bounded-drift-v1` policy requires absolute error ≤0.03 for every
probability (3 percentage points, not 3% relative error) and unchanged argmax for
every distribution. All 285 recorded questions meet both checks. This is a
[reassessment of existing GPU evidence](../validation/reports/source-13eed9b-bounded-drift.json),
not new GPU execution or an arithmetic correction. The original strict reports
remain unchanged. The result applies to this finite synthetic corpus, not unseen
inputs, measured accuracy, a certified registry entry or an alpha release.
Further strict operator-by-operator root-cause research is optional and no longer
blocks this corpus's numerical acceptance under the new policy. Operational and
release gates, including completed soak and clean-host installation, remain open.

Use [Quick Start](quickstart.md) to deploy and the [model matrix](models.md) for
individual licenses, text capabilities, evidence and limits. See [API reference](api.md)
for HTTP behavior and [CONTRIBUTING](../CONTRIBUTING.md) to work on CPU contracts.

The detailed [maintainer status](development/STATUS.md) records commands and failed,
unrun or outstanding gates. Historical plans describe intent; they are not evidence
that a release gate passed. New source changes do not alter the published dev1 image.
The new candidate is bound to a local Docker ID, not a published registry digest;
see [image/GPU evidence](../validation/reports/source-13eed9b-image-gpu.json).

[Security scan and remaining maintenance actions](security-maintenance.md).
