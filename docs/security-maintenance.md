# Security maintenance and dev1 advisory assessment

This is a package-advisory assessment of the **published dev1 image**, not a
scan of the newly edited source wheel or proof of exploitability. No vulnerable
package was silently replaced, no advisory was suppressed, and no new runtime
image or GPU compatibility claim follows from this scan.

## Observed scan — 2026-10-13

Grype 0.120.1 scanned the existing [digest-bound SPDX inventory](sbom.md), exited
0, and reported 2,123 package/advisory matches. Exit 0 means the scan completed;
no severity-failure threshold was supplied. Multiple packages and aliases may
refer to the same underlying vulnerability; these are not unique CVE counts.

| Grype severity | Matches |
|---|---:|
| Critical | 3 |
| High | 56 |
| Medium | 1,816 |
| Low | 225 |
| Negligible | 19 |
| Unknown | 4 |

The sanitized [complete match inventory](../artifacts/security/dev1-grype-matches.json.gz)
retains package/version/type/PURL, advisory identity/source, severity and fix
metadata. It omits local paths and scanner configuration; the private original
report's hash is retained for provenance. Its gzip SHA-256 is
`344a3e9dcbc0600e3743c5294a16da428cfe709f0554b1082380d89ac05015a9`.
The database was valid, schema v6.1.10, built `2026-10-12T06:45:22Z`;
its download URL and checksum are in the report. Future databases can change
matches without any image changes.

Grype warned that the Go packages lack function symbols in this SPDX input.
It therefore matched at module granularity, with possible false positives.
The fix-state counts are 370 `fixed`, 1,221 `not-fixed`, 516 `wont-fix` and 16
unspecified. `fixed` means the advisory database lists a fixed version, not that
this image contains it or that a compatible replacement has been tested.

## Initial triage, not a clean bill of health

All three Critical matches identify Go components. The original native Syft
inventory locates `golang.org/x/net v0.55.0` and `stdlib go1.25.10` inside
`mooncake/libetcd_wrapper.so` in the image's Python packages:

- [GO-2026-6604](https://pkg.go.dev/vuln/GO-2026-6604) concerns Windows junction
  handling. The documented JevServe deployment is Linux; that described Windows
  trigger does not apply to this deployment. The raw match remains in the report.
- [GO-2026-6612](https://pkg.go.dev/vuln/GO-2026-6612) appears for both the standard
  library and x/net and concerns HTTP/2 server flow control. Whether an affected
  Go server is reachable through the JevServe execution profile is **unverified**.
  A package's presence is insufficient to claim either exposure or safety.

The seven vLLM 0.29.0 matches (six Medium, one Low) list 0.30.0 as a fixed version.
They still require per-advisory configuration/reachability review. For example,
[GHSA-2823-qmq8-rwvj](https://github.com/vllm-project/vllm/security/advisories/GHSA-2823-qmq8-rwvj)
concerns LMCache-MP configuration; this Dockerfile uninstalls `lmcache` and the
native request parser rejects undeclared fields. Those are relevant deployment
facts, not a blanket disposition of the remaining engine advisories.

Prioritize the remaining High findings in OpenSSL, FFmpeg, Go modules and GnuPG,
as well as the actual engine-facing advisories. Record affected configuration,
reachable path, upstream fix and regression evidence for each disposition. Do
not remove packages or upgrade the pinned engine solely to reduce a scan count:
first verify whether they are required by the runtime, then build a separate
candidate and re-run the affected CPU, GPU, Docker and Compose gates. No complete
vulnerability triage, exploitability analysis or remediation is claimed here.

## Reproduce

Use the official [Grype v0.120.1 release](https://github.com/anchore/grype/releases/tag/v0.120.1).
The Linux amd64 archive matched its official checksum list:

```text
0a9ee97ef5ae2ee953b0a80098105052e846cdbe319a57d808b519c33cd1343d
```

Install the verified executable project-locally as `.local/bin/grype`. No global
package, Docker service, running container or model needs to be modified:

```bash
mkdir -p .local/audit-security
gzip -dc artifacts/sbom/dev1.spdx.json.gz > .local/audit-security/dev1.spdx.json
GRYPE_CHECK_FOR_APP_UPDATE=false \
GRYPE_DB_CACHE_DIR="$PWD/.local/audit-security/grype-db" \
XDG_CACHE_HOME="$PWD/.local/audit-security/cache" \
  .local/bin/grype sbom:.local/audit-security/dev1.spdx.json \
  -o json --file .local/audit-security/dev1-grype.json
```

This command downloads vulnerability data, not model weights; it is not an offline
scan. To repeat with the already recorded database, set `GRYPE_DB_AUTO_UPDATE=false`
and retain its identity. Preserve scanner warnings and inspect the database
validity and age. Review raw reports for private paths before sharing. The public
artifact is a field-allowlisted extract, not a byte-identical raw Grype report.

## Private reporting remains an external action

The repository's public reporting-status API was rechecked on 2026-10-13 and
returned `enabled: false`. No authenticated GitHub API administration route was
available in this session, so the setting was not changed and no private email
address was invented. A repository administrator can enable **Settings → Advanced
Security → Private vulnerability reporting**, then verify the public endpoint and
update [SECURITY.md](../SECURITY.md) with the working route. See the
[official configuration instructions](https://docs.github.com/en/code-security/how-tos/report-and-fix-vulnerabilities/configure-vulnerability-reporting/configure-for-a-repository).
Until verified, use the existing contact-request policy and do not post sensitive
vulnerability details publicly.
