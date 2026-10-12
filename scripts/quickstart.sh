#!/usr/bin/env bash
# Requires Bash and Docker; all Python/model work runs inside the image.
set -euo pipefail

image="${JEVSERVE_IMAGE:-jevserve:dev}"
name="jevserve"
volume="jevserve-cache"
port=8000
gpu=0
build=false
cdi=false
while (($#)); do
  case "$1" in
    --build) build=true; shift ;;
    --cdi) cdi=true; shift ;;
    --image|--name|--volume|--port|--gpu)
      if (($# < 2)); then echo "Missing value for $1" >&2; exit 2; fi
      option="$1"; value="$2"; shift 2
      case "$option" in
        --image) image="$value" ;;
        --name) name="$value" ;;
        --volume) volume="$value" ;;
        --port) port="$value" ;;
        --gpu) gpu="$value" ;;
      esac ;;
    -h|--help)
      cat <<'HELP'
Usage: scripts/quickstart.sh [--build] [--image REPOSITORY:TAG]
       [--name NAME] [--volume NAME] [--port PORT] [--gpu INDEX] [--cdi]

Build or use an image, start one GPU container, wait for readiness, then run
one real Choice/Noul/Score request. The service stays running on success.
Use --cdi on a Docker host configured for native NVIDIA CDI (e.g. rootless).
Existing containers/volumes are never deleted or replaced by this script.
HELP
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
[[ "$name" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]] || { echo 'Invalid container name' >&2; exit 2; }
[[ "$volume" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]] || { echo 'Use a named cache volume' >&2; exit 2; }
[[ "$port" =~ ^[0-9]{1,5}$ ]] && ((10#$port >= 1 && 10#$port <= 65535)) || { echo 'Invalid port' >&2; exit 2; }
[[ "$gpu" =~ ^[0-9]+$ ]] || { echo 'GPU must be a numeric device index' >&2; exit 2; }
[[ -n "$image" && "$image" != -* && "$image" != *[[:space:]]* ]] || { echo 'Invalid image reference' >&2; exit 2; }
if docker container inspect "$name" >/dev/null 2>&1; then
  echo "Container $name already exists; inspect it or choose --name. Nothing was changed." >&2
  exit 1
fi
if $build; then
  repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
  docker build --tag "$image" "$repo_dir"
elif ! docker image inspect "$image" >/dev/null 2>&1; then
  if [[ "$image" == jevserve:dev ]]; then
    echo 'Build first with --build, or specify a published --image.' >&2
    exit 1
  fi
  docker pull "$image"
fi

gpu_args=(--gpus "device=$gpu")
if $cdi; then gpu_args=(--device "nvidia.com/gpu=$gpu"); fi
echo 'Starting JevServe development preview. Model numerical certification is incomplete.'
docker run --detach --name "$name" "${gpu_args[@]}" --shm-size=1g \
  --publish "127.0.0.1:$port:8000" --mount "type=volume,source=$volume,target=/cache" \
  "$image"
echo "Preparing the model and warming the GPU. Progress: docker logs -f $name"
if ! docker exec "$name" jevserve check --wait-seconds 1800; then
  echo "Startup check failed. Inspect: docker logs $name" >&2
  echo "The container and cache were preserved. Stop with: docker stop --timeout 65 $name" >&2
  exit 1
fi
echo "Ready: http://127.0.0.1:$port/v1/decisions"
echo "Stop: docker stop --timeout 65 $name"
