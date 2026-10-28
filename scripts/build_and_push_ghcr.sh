#!/usr/bin/env bash
# build_and_push_ghcr.sh
# Build the MuJoCo + Warp Holosoma image and push it to GHCR so that
# scripts/build_sif.sh can pull it on the ACES cluster.
#
# The repo's own docker/build_and_push.sh publishes to ECR; this is the GHCR
# counterpart for a personal/public container package.
#
# Usage:
#   GHCR_TOKEN=<pat> bash scripts/build_and_push_ghcr.sh
#   TAG=1.0.0 bash scripts/build_and_push_ghcr.sh
#   TAG_LATEST=1 bash scripts/build_and_push_ghcr.sh
#   PUSH=0 bash scripts/build_and_push_ghcr.sh          # build only
#   DRY_RUN=1 bash scripts/build_and_push_ghcr.sh       # print commands only
#
# Overridable env vars:
#   IMAGE_REPO, TAG, DOCKERFILE, BUILD_ARGS, PLATFORM, TAG_LATEST, PUSH,
#   NO_CACHE, GHCR_USER, GHCR_TOKEN, DRY_RUN
#
# Notes:
#   * Base is nvcr.io/nvidia/isaac-sim:5.1.0 (large); the first build pulls it.
#     No GPU is needed at build time (setup_mujoco.sh --skip-driver-check).
#   * The image bakes a COPY of the repo, but the sbatch scripts bind-mount the
#     live repo over /workspace/holosoma, so rebuild only when the Dockerfile or
#     scripts/setup_mujoco.sh dependency layer changes.
set -uo pipefail

IMAGE_REPO="${IMAGE_REPO:-ghcr.io/noahleegithub/safefall-mujoco}"
TAG="${TAG:-$(date -u +%Y_%m%d_%H%M)}"
DOCKERFILE="${DOCKERFILE:-docker/mujoco.Dockerfile}"
BUILD_ARGS="${BUILD_ARGS:-WARP=true}"
PLATFORM="${PLATFORM:-linux/amd64}"
TAG_LATEST="${TAG_LATEST:-0}"
PUSH="${PUSH:-1}"
NO_CACHE="${NO_CACHE:-0}"
GHCR_USER="${GHCR_USER:-noahleegithub}"
GHCR_TOKEN="${GHCR_TOKEN:-}"
DRY_RUN="${DRY_RUN:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --latest)   TAG_LATEST=1 ;;
    --no-push)  PUSH=0 ;;
    --no-cache) NO_CACHE=1 ;;
    --dry-run)  DRY_RUN=1 ;;
    --help|-h)  sed -n '2,/^$/ { s/^# \?//; p }' "$0"; exit 0 ;;
    -*)         echo "Unknown flag: $1" >&2; exit 1 ;;
    *)          TAG="$1" ;;
  esac
  shift
done

green(){ printf '\033[32m%s\033[0m' "$1"; }
red(){   printf '\033[31m%s\033[0m' "$1"; }
hdr(){   printf '\n=== %s ===\n' "$1"; }
ok(){    printf '  [%s] %s\n' "$(green PASS)" "$1"; }
warn(){  printf '  [%s] %s\n' "$(red WARN)" "$1"; }
no(){    printf '  [%s] %s\n' "$(red FAIL)" "$1"; }

run(){
  echo "+ $*"
  if [[ "$DRY_RUN" != "1" ]]; then "$@"; fi
}

ROOT_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REGISTRY="${IMAGE_REPO%%/*}"          # ghcr.io
SHORT_SHA="$(git -C "$ROOT_REPO" rev-parse --short HEAD 2>/dev/null || echo nogit)"

# ---- preflight ----
hdr "Context"
echo "repo        : $ROOT_REPO"
echo "image_repo  : $IMAGE_REPO"
echo "tag         : $TAG  (sha-$SHORT_SHA$( [[ "$TAG_LATEST" == "1" ]] && echo ", latest"))"
echo "dockerfile  : $DOCKERFILE"
echo "build_args  : ${BUILD_ARGS:-<none>}"
echo "platform    : $PLATFORM"
echo "push        : $PUSH   dry_run: $DRY_RUN"

if ! command -v docker >/dev/null 2>&1; then
  no "docker not found"
  exit 1
fi
if [[ "$DRY_RUN" == "1" ]]; then
  ok "docker present (dry-run)"
else
  ok "docker ${DOCKER_VER:-$(docker version --format '{{.Server.Version}}' 2>/dev/null || echo '?')}"
fi

if [[ ! -f "$ROOT_REPO/$DOCKERFILE" ]]; then
  no "Dockerfile not found: $ROOT_REPO/$DOCKERFILE"
  exit 1
fi
ok "Dockerfile present"

if [[ "$DOCKERFILE" == *"mujoco"* && "$BUILD_ARGS" != *"WARP=true"* ]]; then
  warn "mujoco image built without WARP=true -> the hsmujoco env will lack mujoco_warp"
fi

# ---- tags ----
REFS=("${IMAGE_REPO}:${TAG}" "${IMAGE_REPO}:sha-${SHORT_SHA}")
[[ "$TAG_LATEST" == "1" ]] && REFS+=("${IMAGE_REPO}:latest")
TAGS=()
for r in "${REFS[@]}"; do TAGS+=(-t "$r"); done

# ---- registry login ----
if [[ "$PUSH" == "1" ]]; then
  hdr "Registry login"
  if [[ -n "$GHCR_TOKEN" ]]; then
    if [[ "$DRY_RUN" == "1" ]]; then
      echo "+ printf '%s' <redacted> | docker login $REGISTRY -u $GHCR_USER --password-stdin"
    else
      if printf '%s' "$GHCR_TOKEN" | docker login "$REGISTRY" -u "$GHCR_USER" --password-stdin; then
        ok "logged in to $REGISTRY as $GHCR_USER"
      else
        no "docker login failed"
        exit 1
      fi
    fi
  else
    warn "GHCR_TOKEN unset; assuming docker is already logged in to $REGISTRY"
  fi
fi

# ---- build ----
hdr "Build"
BUILD_CMD=(env DOCKER_BUILDKIT=1 docker build --platform "$PLATFORM")
[[ "$NO_CACHE" == "1" ]] && BUILD_CMD+=(--no-cache)
for ba in $BUILD_ARGS; do BUILD_CMD+=(--build-arg "$ba"); done
BUILD_CMD+=("${TAGS[@]}" -f "$DOCKERFILE" "$ROOT_REPO")

if run "${BUILD_CMD[@]}"; then
  ok "built ${IMAGE_REPO}:${TAG}"
else
  no "build failed"
  exit 1
fi

# ---- push ----
if [[ "$PUSH" == "1" ]]; then
  hdr "Push"
  for ref in "${REFS[@]}"; do
    if run docker push "$ref"; then
      ok "pushed $ref"
    else
      no "push failed: $ref"
      exit 1
    fi
  done
else
  hdr "Push"
  warn "PUSH=0; skipped push"
fi

# ---- report ----
hdr "Summary"
echo "  image       : ${IMAGE_REPO}:${TAG}"
if [[ "$PUSH" == "1" && "$DRY_RUN" != "1" ]] && command -v docker >/dev/null 2>&1; then
  digest="$(docker buildx imagetools inspect "${IMAGE_REPO}:${TAG}" 2>/dev/null | awk '/^Digest:/ {print $2; exit}')"
  [[ -n "$digest" ]] && echo "  digest      : ${IMAGE_REPO}@${digest}"
fi
echo
echo "Pull it on the cluster:"
echo "  IMAGE_URI=docker://${IMAGE_REPO}:${TAG} bash scripts/build_sif.sh"
echo
echo "First push only: make the GHCR package public so anonymous pulls work --"
echo "  https://github.com/users/${GHCR_USER}/packages/container/${IMAGE_REPO##*/}/settings"
