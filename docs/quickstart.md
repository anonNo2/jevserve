# Running JevServe

JevServe serves Open-Jev-2B as a text decision API. This is a **development
preview**: GPU inference works, but numerical certification is incomplete.
The development image explicitly enables this uncertified execution profile.
See [current verification results](development/STATUS.md).

## Requirements

- Linux x86_64, Docker Engine with Compose, and NVIDIA Container Toolkit.
- An NVIDIA GPU available to your container. Development inference was tested
  on an RTX A6000 with 48 GiB VRAM; a minimum-VRAM claim has not been validated.
- Internet access on first startup to download the pinned model. Reserve disk
  space for the large vLLM image, approximately 4.6 GB of source model files,
  and approximately 3.5 GB of compiled model files.
- A repository checkout for Compose, local builds and the optional helper; plain Docker needs no checkout.

Host Python and CUDA Toolkit are unnecessary. Check your host's GPU occupancy
before starting; the default runtime reserves 50% of GPU memory. Follow
[NVIDIA's installation instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
when setting up a new host. Do not change a shared server's runtime or restart
its Docker service without its administrator's agreement.

## One-command startup

```bash
git clone https://github.com/anonNo2/jevserve.git
cd jevserve
./scripts/quickstart.sh --image anonwx/jevserve:0.1.0-dev0
```

The script pulls `anonwx/jevserve:0.1.0-dev0`, creates container `jevserve`, attaches named
volume `jevserve-cache`, waits up to 30 minutes for readiness, and makes one
real request with Choice, Noul, and Score questions. The API remains running at
`http://127.0.0.1:8000`. It does not replace existing containers or delete caches.
Health checks may be repeated; decision requests are never automatically retried.

To customize the local instance:

```bash
./scripts/quickstart.sh --image anonwx/jevserve:0.1.0-dev0 --name jevserve-demo \
  --volume jevserve-demo-cache --port 18000 --gpu 0
```

The development preview is published on [Docker Hub](https://hub.docker.com/r/anonwx/jevserve)
for Linux amd64. To build from your checkout instead, run
`./scripts/quickstart.sh --build`. See [supported models and test results](models.md)
for the tested checkpoint and the outstanding numerical certification gate.

## Pin the published image

For reproducible deployments, replace the tag in any example with the immutable
registry digest:

```text
anonwx/jevserve@sha256:f27c15c1da40d0c02139672a426d41d5370e6ea061d209b1d3fc5aef53b73396
```

This is the same development image exercised by the recorded container tests.
See [publication and verification evidence](development/STATUS.md).

## Plain Docker

These are the commands wrapped by the helper, with the readiness check separated:

```bash
docker run -d --name jevserve --gpus device=0 --shm-size=1g \
  -p 127.0.0.1:8000:8000 \
  --mount type=volume,source=jevserve-cache,target=/cache \
  anonwx/jevserve:0.1.0-dev0
docker exec jevserve jevserve check --wait-seconds 1800
```

Model preparation verifies pinned source hashes, publishes complete downloads
atomically, and reuses the named cache. Weights are not baked into the image.
The container runs as UID/GID `10001:10001`; use a named volume to avoid bind-mount
ownership problems.

## Docker Compose

Use this instead of the plain Docker instance on port 8000:

```bash
JEVSERVE_IMAGE=anonwx/jevserve:0.1.0-dev0 docker compose up --pull always --no-build -d
docker compose exec jevserve jevserve check --wait-seconds 1800
docker compose logs --tail 100 jevserve
```

Compose uses a read-only root filesystem, a writable cache volume and `/tmp`,
and GPU device `0`. Change `ports` or `device_ids` in `compose.yaml` for another
port or GPU. `JEVSERVE_IMAGE` selects the image; use `--no-build` when starting
an image you have already pulled. `docker compose down` removes this project's
containers and network while preserving its model cache.

## Send a request

```bash
curl -fsS http://127.0.0.1:8000/readyz
curl -fsS http://127.0.0.1:8000/v1/decisions \
  -H 'Content-Type: application/json' \
  --data-binary @examples/decision.json
```

The example asks which team should handle a message, whether it is urgent, and
its priority. Each answer has its own typed result and distribution. These are
model probabilities, not a guarantee of correctness or calibrated accuracy.
The response records the model revision and execution profile.

## Offline reuse

First prepare the cache with a successful online startup. Stop that container,
then reuse the same cache and explicitly disable model network access:

```bash
docker stop --timeout 65 jevserve
docker run -d --name jevserve-offline --gpus device=0 --shm-size=1g \
  -p 127.0.0.1:8000:8000 \
  --mount type=volume,source=jevserve-cache,target=/cache \
  anonwx/jevserve:0.1.0-dev0 serve --allow-uncertified --offline
docker exec jevserve-offline jevserve check --wait-seconds 1800
```

Missing or corrupt cached files fail startup in offline mode. There is no
network fallback. For fully disconnected operation, transfer the image and
verified cache ahead of time; Docker cannot pull a missing image offline.

## Authentication

The examples bind to loopback. Before exposing the API, configure authentication
and terminate TLS at your reverse proxy. Store a strong key in a local file,
mount it read-only, and pass `serve --allow-uncertified --api-key-file /run/secrets/jevserve-key`.
The file must be readable by the container's UID 10001; do not embed its contents
in Docker commands, images, Git, or logs. Mount only the intended key file.
The native Python client accepts `api_key_file=Path("/path/to/key")`; the CLI
check accepts `--api-key-file /run/secrets/jevserve-key` inside the container.

## Troubleshooting and shutdown

- **Still loading:** `docker logs --tail 100 jevserve`. First startup downloads
  and compiles the pinned model before opening the API. The helper preserves
  a failed container and its cache for inspection.
- **GPU unavailable:** check `nvidia-smi` and your NVIDIA Container Toolkit
  setup. Hosts configured for native CDI can use `./scripts/quickstart.sh --image anonwx/jevserve:0.1.0-dev0 --cdi`
  or replace `--gpus device=0` with `--device nvidia.com/gpu=0`.
  See [NVIDIA CDI documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html).
- **Name already exists:** inspect `docker ps -a --filter name=jevserve`, restart
  your existing container, or choose another `--name`. The helper never deletes it.
- **Out of memory:** stop this instance and check other GPU workloads. There is
  no automatic precision reduction, truncation or alternate model. Runtime
  options are listed by `docker run --rm anonwx/jevserve:0.1.0-dev0 serve --help`.
- **Stop:** `docker stop --timeout 65 jevserve`. Remove only your stopped container
  with `docker rm jevserve`. Its named model cache remains for the next start.

Avoid `docker system prune` on shared hosts. Delete a project cache volume only
when you intentionally want to discard its downloaded model.
