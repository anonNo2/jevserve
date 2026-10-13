# Fault and recovery checks

These are bounded development checks on dedicated resources. They do not certify
power-loss safety, network-interruption recovery or a clean-host installation.
Current evidence is in
[`artifact-corruption-drill.json`](../validation/reports/artifact-corruption-drill.json)
and the [workload reports](../validation/reports/hardening-workload).

## Corrupted prepared artifacts

On the development host, a filesystem reflink copy of the verified JPT-0.8B
source inventory was created under a new project-local directory. One newline was
appended to the copy's `config.json`. `load_model_directory(copy,
source_manifest("kirp/jpt-0.8b"))` raised `ArtifactCorrupt` before any engine
startup. Original weight bytes were unchanged. The experiment changes a copied
configuration file, not a real upstream download or model weight file.

To reproduce, copy a verified source directory into a fresh test directory with
`cp --reflink=always -a SOURCE COPY` on a supporting filesystem, change only the
copy's configuration, and invoke the same verifier from the project Python
environment. Do not use hard links: modifying a hard-linked file also changes its
source. Keep the original prepared inventory immutable.

## Missing cache in a network-isolated container

Using the explicitly selected project Docker socket:

```bash
docker --host "$VALIDATION_DOCKER_HOST" run --rm --pull=never \
  --name jevserve-offline-missing-check --network none --read-only --tmpfs /tmp \
  --entrypoint jevserve \
  anonwx/jevserve@sha256:4b758ad1000f4d38501fd83f0a9c2b32f0cb91968d427b170a2c2b0d9a613aff \
  prepare --offline --cache-dir /tmp/missing
```

Observed result: exit 1 with `ArtifactMissing`. This is the expected refusal,
not a successful model startup. There is no usable partial package or implicit
online fallback.

## Abrupt container termination and explicit recovery

The JPT workload service used a fixed dev1 digest, separately mounted verified
weights, and a persistent project-scoped cache volume. After its examples and
performance requests completed, the dedicated container received
`docker kill --signal KILL CONTAINER`. The drill checks stopped-container exit
137, zero automatic restarts, unavailable readiness and released GPU resources.
It then removes only that stopped container, explicitly recreates the same image
with the same mounts, waits for `jevserve check --wait-seconds 1800`, and runs the
real ticket-routing example again. The final graceful stop is recorded separately
and must exit zero; intentional exit 137 is not described as graceful shutdown.

Keep exact names and mounts in private execution records. This cached restart
exercise does not establish image-version rollback; that separate three-stage
drill is documented in [rollback.md](rollback.md). Do not run a destructive crash
check against a service carrying other users' requests.
