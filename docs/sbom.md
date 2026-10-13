# Image software inventory

JevServe's dependency locks describe selected build/runtime inputs. An image SBOM
also inventories discoverable operating-system and language packages inherited
from the base image. Neither is a vulnerability scan, a license audit or proof
that every embedded binary component has been identified.

## Recorded dev1 scan

On 2026-10-13, the scan of
`anonwx/jevserve@sha256:4b758ad1000f4d38501fd83f0a9c2b32f0cb91968d427b170a2c2b0d9a613aff`
completed with exit code 0 and no scanner warnings in approximately 427 seconds
(process start through final output write). The public artifact is
[`dev1.spdx.json.gz`](../artifacts/sbom/dev1.spdx.json.gz), containing unmodified
SPDX 2.3 JSON compressed with gzip, an empty stored filename and `mtime=0`.

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| Published gzip | 2,995,089 | `7317e2198137a947735cad880fb037d8ffe67e8d62fe3fcb2b6713f27be858e1` |
| Decompressed SPDX JSON | 49,361,510 | `a992d755033f3bdf8b210314a41644855982ebab05897b68540202e8781d48cb` |

Syft reported 1,868 package records: 482 Debian packages, 285 Python packages,
1,066 Rust crates, 19 Go modules, 15 binary records and one safetensors record.
SPDX contains 1,869 package entries because it also describes the source image.
These are discovered records, not necessarily unique project names. The
safetensors record is the installed `compressed_tensors` package's
`transform/utils/hadamards.safetensors` utility data, not a mounted Jev checkpoint.

The public SPDX passed JSON parsing, package-ID uniqueness, image-digest binding,
gzip round-trip and checks for known local paths, usernames and Docker socket
identifiers. Its source name/digest were already public; no report fields were
rewritten. The larger native Syft report contains scanner-local configuration and
is kept private. No vulnerability scanner or independent completeness audit was
run, and this inventory does not add a model or release certification.

## Scanner and scope

The dev1 inventory uses [Syft v1.54.1](https://github.com/anchore/syft/releases/tag/v1.54.1),
commit `b254e6d92f28c3868a755f62fb3ca8f26e9fee76`, for Linux amd64. The official
`syft_1.54.1_linux_amd64.tar.gz` matched the release checksum:

```text
c069905b391cc4c20a5ba65ad5c10be2a7ba074f8ea6ad203e24d14e303dad47
```

This checks the downloaded artifact against the official checksum; no separate
signature verification is claimed. Install it in a project-local tool directory,
without changing system packages. The [official documentation](https://github.com/anchore/syft)
describes supported package ecosystems and output formats.

The scan selects `docker:` explicitly, reading the already-present immutable
image through an explicit local Docker socket. It does not run the image, start
GPU inference, change tags or modify the daemon. `--scope squashed` inventories
the final effective image filesystem, including base-image OS packages. Deleted
files from earlier layers and separately mounted model weights are outside this
scope. The default image catalogers are used with two workers; remote enrichment
is not enabled and the update check is disabled. Package-manager metadata and
binary identification can be incomplete, especially for vendored components.

## Reproduction

Obtain the scanner archive and checksum list from the fixed official release,
verify the archive before extracting the `syft` executable to `.local/bin`, then
run from the repository root. Adjust the socket to your own local Docker daemon;
the specified image must already be present there.

```bash
mkdir -p .local/sbom/tmp .local/sbom/cache
export SBOM_DOCKER_HOST=unix:///var/run/docker.sock
export SBOM_IMAGE=anonwx/jevserve@sha256:4b758ad1000f4d38501fd83f0a9c2b32f0cb91968d427b170a2c2b0d9a613aff
SYFT_CHECK_FOR_APP_UPDATE=false \
DOCKER_HOST="$SBOM_DOCKER_HOST" \
TMPDIR="$PWD/.local/sbom/tmp" \
XDG_CACHE_HOME="$PWD/.local/sbom/cache" \
  .local/bin/syft scan "docker:$SBOM_IMAGE" --scope squashed --parallelism 2 \
  -o syft-json=.local/sbom/jevserve-0.1.0-dev1.syft.json \
  -o spdx-json=.local/sbom/jevserve-0.1.0-dev1.spdx.json \
  -o cyclonedx-json=.local/sbom/jevserve-0.1.0-dev1.cdx.json
sha256sum .local/sbom/jevserve-0.1.0-dev1.*.json
```

Review exit status, scanner warnings, source image identity and package counts
before accepting output. Record scanner version, scope, command, output sizes and
SHA-256 hashes with the scan. Generated timestamps/identifiers may change between
runs, so equal package inventories do not imply byte-identical files. Syft JSON
retains scanner-specific metadata; SPDX and CycloneDX enable downstream tooling.

Keep raw reports in `.local/sbom` until reviewed for private paths, Docker socket
locations, credentials and image configuration metadata. A public scan summary
should link immutable reports and their hashes; do not publish raw host paths or
claim that an SBOM proves absence of vulnerabilities. See [SECURITY.md](../SECURITY.md)
for reporting and [releasing](releasing.md) for release evidence boundaries.
