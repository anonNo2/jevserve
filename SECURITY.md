# Security reporting

JevServe is a development preview. There is no stable security-maintenance branch,
guaranteed response time or published long-term support policy. Reports should
identify the affected source revision and immutable image digest; a development
tag or model integration is not a security certification.

## Reporting channel

As of 2026-10-13, this project has no verified private reporting channel documented
here. The public [GitHub reporting-status endpoint](https://api.github.com/repos/anonNo2/jevserve/private-vulnerability-reporting)
returned `{"enabled": false}`; private vulnerability reporting was disabled at
that check, and no security email address is published here. Do not send sensitive details to an
address guessed from commits or contributor identities.

You can open a [public issue](https://github.com/anonNo2/jevserve/issues/new)
titled “Security contact request” containing only a request for a private channel.
Do **not** include exploit details, credentials, affected private deployments,
request bodies or logs in that public issue. Wait for a maintainer to provide and
confirm a private route before sharing sensitive material. This is a public
coordination route, not a confidential vulnerability-reporting service. If public
contact itself is unsafe, retain the report until a private channel is available.

Once a private route is confirmed, useful report details include the affected
commit/image digest, relevant model revision and configuration, impact, a minimal
synthetic reproduction and any mitigation. Redact host/user/network identifiers
and avoid sending model weights or real user data.

## Dependencies and deployment scope

The [engine lock](compat/engine-lock.json),
[runtime requirements](compat/runtime-requirements.txt),
[build requirements](compat/build-requirements.txt) and [uv lock](uv.lock) record
different dependency layers. Model inventories are in
[src/jevserve/data](src/jevserve/data); model licenses are listed in
[the model matrix](docs/models.md). These lock files are not a complete image SBOM
or a vulnerability-scan result. The separate [dev1 image inventory](docs/sbom.md)
provides a Syft-generated SPDX report covering discoverable packages in the final
image filesystem, including base-image OS packages. Its scope, tool version,
hashes and detection limits are documented; no vulnerability scan or independent
completeness audit is claimed. See [release documentation](docs/releasing.md) for
the actual evidence required by the release process.

Use the [deployment guide](docs/quickstart.md) for the current loopback-bound
configuration, external weights and credentials. Review custom upstream code and
dependencies before executing them. A pinned revision provides identity, not a
security audit. Keep private prompts, authentication tokens and high-cardinality
identifiers out of public logs and reports.
