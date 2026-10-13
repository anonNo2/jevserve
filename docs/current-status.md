# Current status

As of 2026-10-13, JevServe is a **development preview**, not an alpha/stable release
or a certified inference implementation. Source and the pinned dev1 container are
public; no Python package has been published.

| Area | Observed state |
|---|---|
| Text API | Choice, Noul and Score; native client and typesafe-sdk 0.7.4 text subset |
| Models | Seven fixed model profiles have real GPU/container execution evidence |
| Numerical parity | Open-Jev/JPT still fail; NeoHorse/Kev pass only a small synthetic suite |
| Deployment | Plain Docker and Compose execution verified; clean-host installation still pending |
| New source image | Unpublished `13eed9b` candidate passed seven-model GPU/SDK and two-model offline Compose execution; strict numerical gate still fails |
| Security | dev1 SBOM advisory scan completed; findings remain under triage; private reporting still disabled |
| Vision | No image inputs; VL investigation does not establish vision support |
| Release endurance | Old dev1 24-hour run failed after 16,271.64 seconds; no completed 24/72-hour acceptance |

Use [Quick Start](quickstart.md) to deploy and the [model matrix](models.md) for
individual licenses, text capabilities, evidence and limits. See [API reference](api.md)
for HTTP behavior and [CONTRIBUTING](../CONTRIBUTING.md) to work on CPU contracts.

The detailed [maintainer status](development/STATUS.md) records commands and failed,
unrun or outstanding gates. Historical plans describe intent; they are not evidence
that a release gate passed. New source changes do not alter the published dev1 image.
The new candidate is bound to a local Docker ID, not a published registry digest;
see [image/GPU evidence](../validation/reports/source-13eed9b-image-gpu.json).

[Security scan and remaining maintenance actions](security-maintenance.md).
