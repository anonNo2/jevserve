# Running JevServe

JevServe serves pinned Open-Jev, JPT, NeoHorse and Kev models as a text decision API. This is a **development
preview**: GPU inference works, but numerical certification is incomplete.
The development image explicitly enables this uncertified execution profile.
See [current verification results](development/STATUS.md).

## Requirements

- Linux x86_64, Docker Engine with Compose, and NVIDIA Container Toolkit.
- An NVIDIA GPU available to your container. Development inference was tested
  on an RTX A6000 with 48 GiB VRAM; a minimum-VRAM claim has not been validated.
- Internet access on first startup to download the pinned model. Reserve disk
  space for the large vLLM image and separate model/runtime caches. The default
  Open-Jev-2B needs approximately 4.6 GB of source files and 3.5 GB of compiled
  model files; other profiles have different [download sizes](models.md).
- A repository checkout for Compose, local builds and the optional helper; plain Docker needs no checkout.

Host Python and CUDA Toolkit are unnecessary. Check your host's GPU occupancy
before starting. The default GPU memory budget is 50%: vLLM reserves its pool,
while the Torch profiles enforce an allocator limit. This is not minimum VRAM. Follow
[NVIDIA's installation instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
when setting up a new host. Do not change a shared server's runtime or restart
its Docker service without its administrator's agreement.

## One-command startup

```bash
git clone https://github.com/anonNo2/jevserve.git
cd jevserve
./scripts/quickstart.sh --image anonwx/jevserve:0.1.0-dev1
```

The script pulls `anonwx/jevserve:0.1.0-dev1`, creates container `jevserve`, attaches named
volume `jevserve-cache`, waits up to 30 minutes for readiness, and makes one
real request with Choice, Noul, and Score questions. The API remains running at
`http://127.0.0.1:8000`. It does not replace existing containers or delete caches.
Health checks may be repeated; decision requests are never automatically retried.

To customize the local instance:

```bash
./scripts/quickstart.sh --image anonwx/jevserve:0.1.0-dev1 --name jevserve-demo \
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
anonwx/jevserve@sha256:4b758ad1000f4d38501fd83f0a9c2b32f0cb91968d427b170a2c2b0d9a613aff
```

This digest identifies `0.1.0-dev1`, the generic seven-profile runtime image.
Its anonymous registry manifest and digest pull were verified; weights remain
separate mounts or cache-volume files.
See [publication and verification evidence](development/STATUS.md).

## Plain Docker

These are the commands wrapped by the helper, with the readiness check separated:

```bash
docker run -d --name jevserve --gpus device=0 --shm-size=1g \
  -p 127.0.0.1:8000:8000 \
  --mount type=volume,source=jevserve-cache,target=/cache \
  anonwx/jevserve:0.1.0-dev1
docker exec jevserve jevserve check --wait-seconds 1800
```

Model preparation verifies pinned source hashes, publishes complete downloads
atomically, and reuses the named cache. Weights are not baked into the image.
The container runs as UID/GID `10001:10001`; use a named volume to avoid bind-mount
ownership problems.

## Docker Compose

Use this instead of the plain Docker instance on port 8000:

```bash
JEVSERVE_IMAGE=anonwx/jevserve:0.1.0-dev1 docker compose up --pull always --no-build -d
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
  anonwx/jevserve:0.1.0-dev1 serve --allow-uncertified --offline
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
  setup. Hosts configured for native CDI can use `./scripts/quickstart.sh --image anonwx/jevserve:0.1.0-dev1 --cdi`
  or replace `--gpus device=0` with `--device nvidia.com/gpu=0`.
  See [NVIDIA CDI documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html).
- **Name already exists:** inspect `docker ps -a --filter name=jevserve`, restart
  your existing container, or choose another `--name`. The helper never deletes it.
- **Out of memory:** stop this instance and check other GPU workloads. There is
  no automatic precision reduction, truncation or alternate model. Runtime
  options are listed by `docker run --rm anonwx/jevserve:0.1.0-dev1 serve --help`.
- **Stop:** `docker stop --timeout 65 jevserve`. Remove only your stopped container
  with `docker rm jevserve`. Its named model cache remains for the next start.

Avoid `docker system prune` on shared hosts. Delete a project cache volume only
when you intentionally want to discard its downloaded model.

## Mounted models

The image contains the runtime and adapters, **not model weights**. Use one image
for every supported model; each container owns one engine and one GPU. The model
ID must be in `jevserve models list`. See [licenses and actual test results](models.md)
before selecting it. Model names containing “VL” do not enable image inputs in
this text API.

Prepare a public model into a host directory, using your host UID so the files
remain manageable. This command needs network access but no GPU:

```bash
mkdir -p ./model-cache
docker run --rm --user "$(id -u):$(id -g)" \
  --mount "type=bind,source=$PWD/model-cache,target=/cache" \
  anonwx/jevserve:0.1.0-dev1 prepare --model kirp/jpt-0.8b
```

The result prints `/cache/sources/<identity>`. Its host location is
`$PWD/model-cache/sources/<identity>`. The identity is derived from the pinned
manifest. Assign the actual printed directory and make these public weights
readable by the serving UID (`10001`):

```bash
export JEVSERVE_MODEL=kirp/jpt-0.8b
export JEVSERVE_MODEL_DIR="$PWD/model-cache/sources/<identity>"
chmod -R a+rX "$JEVSERVE_MODEL_DIR"
./scripts/quickstart.sh --image anonwx/jevserve:0.1.0-dev1 \
  --model "$JEVSERVE_MODEL" --model-dir "$JEVSERVE_MODEL_DIR"
```

`<identity>` is a placeholder, not a literal directory name. The serving command
uses `--model-dir /model --offline`: it verifies the full pinned inventory locally,
mounts weights read-only and does not download a substitute. Writable compilation
and kernel caches remain under `/cache`. Never mount credentials or private data
in the model directory. A prepared directory must contain its complete verified
inventory; arbitrary Hub snapshots with extra cache metadata are not accepted.

Plain Docker equivalent:

```bash
docker run -d --name jevserve --gpus device=0 --shm-size=1g \
  -p 127.0.0.1:8000:8000 \
  --mount "type=bind,source=$JEVSERVE_MODEL_DIR,target=/model,readonly" \
  --mount type=volume,source=jevserve-runtime,target=/cache \
  anonwx/jevserve:0.1.0-dev1 serve --allow-uncertified \
  --model "$JEVSERVE_MODEL" --model-dir /model --offline
docker exec jevserve jevserve check --wait-seconds 1800
```

Mounted Compose uses the same model ID and directory variables:

```bash
export JEVSERVE_IMAGE=anonwx/jevserve:0.1.0-dev1
docker compose -f compose.yaml -f compose.mount.yaml up --no-build -d
docker compose exec jevserve jevserve check --wait-seconds 1800
```

Changing a model requires recreating its container with the new ID and directory;
there is no hot swap or automatic fallback. Leave enough free VRAM before starting.
`--gpu-memory-utilization` (Compose: `JEVSERVE_GPU_MEMORY_UTILIZATION`) controls the
selected runtime's memory budget; it does not shrink the model weights. CPU
weight offload is an explicit JPT-only option and is not part of the tested default
profiles. Quantization and vision serving are not implemented.
