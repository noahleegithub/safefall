#!/usr/bin/env bash
# build_sif.sh
# Pull a Docker/OCI image URI into a single-file SIF ready for the MJWarp
# train/bench/eval sbatch scripts on ACES.
#
# Reuses the probe_holosoma_*.sh environment fixes:
#   * keep ALL temp/cache on node-local /tmp (never /scratch, never $HOME) --
#     /scratch is Lustre with a 250k-file quota, and /tmp is xfs NVMe.
#   * best-effort `module load WebProxy` under Slurm, since compute nodes have
#     no direct egress and the pull needs the network.
#
# Runnable on a login node, inside `srun --pty`, or as an sbatch job.
#
# Usage:
#   bash scripts/build_sif.sh
#   IMAGE_URI=docker://ghcr.io/you/img:tag SIF=/somewhere/img.sif bash scripts/build_sif.sh
#   FORCE=1 bash scripts/build_sif.sh          # re-pull even if the SIF exists
#   VERIFY=0 bash scripts/build_sif.sh         # skip the in-image env check
#
# Overridable env vars:
#   IMAGE_URI, SIF, FORCE, VERIFY, CONTAINER, ENV_PREFIX, MIN_FREE_GB,
#   CLEAN_SCRATCH, DOCKER_USERNAME, DOCKER_PASSWORD
set -u

IMAGE_URI="${IMAGE_URI:-docker://ghcr.io/noahleegithub/safefall-mujoco:1.0.0}"
SIF="${SIF:-${SCRATCH:-/scratch}/images/safefall-mujoco.sif}"
FORCE="${FORCE:-0}"
VERIFY="${VERIFY:-1}"
ENV_PREFIX="${ENV_PREFIX:-/root/.holosoma_deps/miniconda3/envs/hsmujoco}"
MIN_FREE_GB="${MIN_FREE_GB:-70}"
CLEAN_SCRATCH="${CLEAN_SCRATCH:-0}"

green(){ printf '\033[32m%s\033[0m' "$1"; }
red(){   printf '\033[31m%s\033[0m' "$1"; }
hdr(){   printf '\n=== %s ===\n' "$1"; }
ok(){    printf '  [%s] %s\n' "$(green PASS)" "$1"; }
warn(){  printf '  [%s] %s\n' "$(red WARN)" "$1"; }
no(){    printf '  [%s] %s\n' "$(red FAIL)" "$1"; }

# ---- force node-local temp/cache (never /scratch, never $HOME) ----
NODE_TMP="${TMPDIR:-/tmp}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-$NODE_TMP/ap-tmp.$$}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$NODE_TMP/ap-cache.$$}"
export SINGULARITY_TMPDIR="$APPTAINER_TMPDIR"
export SINGULARITY_CACHEDIR="$APPTAINER_CACHEDIR"
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"

# ---- resolve container runtime ----
CONTAINER="${CONTAINER:-$(command -v apptainer || command -v singularity || true)}"

hdr "Context"
echo "host        : $(hostname)"
echo "SLURM job   : ${SLURM_JOB_ID:-<none>}"
command -v nvidia-smi >/dev/null && \
  echo "gpu         : $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1)"
echo "container   : ${CONTAINER:-<none>} $([[ -n "$CONTAINER" ]] && "$CONTAINER" --version 2>&1 | head -1)"
echo "image       : $IMAGE_URI"
echo "sif         : $SIF"
echo "temp        : $APPTAINER_TMPDIR"
df -T "$APPTAINER_TMPDIR" 2>/dev/null | tail -1

if [[ -z "$CONTAINER" ]]; then
  no "no apptainer/singularity on PATH (set CONTAINER=/path/to/singularity)"
  exit 1
fi

# ---- network: compute nodes need the proxy module ----
if type module >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
  hdr "Load WebProxy (best effort)"
  if module load WebProxy 2>/dev/null; then ok "module load WebProxy"; else warn "module load WebProxy failed; assuming direct egress"; fi
fi

# ---- optional scratch cleanup (leftover extractions eat the file quota) ----
if [[ "$CLEAN_SCRATCH" == "1" ]]; then
  hdr "Reclaim scratch: removing leftover rootfs-* dirs"
  if [[ -n "${SCRATCH:-}" ]]; then
    find "$SCRATCH" -maxdepth 2 -type d -name 'rootfs-*' -print -exec rm -rf {} + 2>/dev/null \
      && ok "cleaned" || echo "  (none found / not permitted)"
  else
    warn "SCRATCH unset; skipping"
  fi
fi

# ---- preflight: destination dir + free space ----
hdr "Preflight"
SIF_DIR="$(dirname "$SIF")"
if ! mkdir -p "$SIF_DIR"; then
  no "cannot create $SIF_DIR"
  exit 1
fi
ok "destination dir: $SIF_DIR"

free_gb="$(df -Pk "$SIF_DIR" 2>/dev/null | awk 'NR==2 {printf "%d", $4/1024/1024}')"
if [[ -n "$free_gb" ]]; then
  echo "  free on dest : ${free_gb} GB (need >= ${MIN_FREE_GB} GB)"
  if (( free_gb < MIN_FREE_GB )); then
    no "not enough free space on $SIF_DIR"
    exit 1
  fi
  ok "free space"
else
  warn "could not determine free space on $SIF_DIR"
fi

# ---- idempotency: reuse a valid existing SIF unless FORCE=1 ----
PULL=1
if [[ -f "$SIF" && "$FORCE" != "1" ]]; then
  if "$CONTAINER" inspect "$SIF" >/dev/null 2>&1; then
    ok "valid SIF already present: $SIF (FORCE=1 to re-pull)"
    PULL=0
  else
    warn "existing SIF is invalid; will re-pull"
  fi
fi

# ---- atomic pull: stage next to the target (same FS) so a partial pull is never
#      mistaken for a usable image by the sbatch scripts ----
if [[ "$PULL" == "1" ]]; then
  hdr "Pull image -> SIF"
  TMP_SIF="${SIF}.partial.$$.sif"
  cleanup(){ [[ -e "$TMP_SIF" ]] && rm -f "$TMP_SIF"; }
  trap cleanup EXIT INT TERM

  PULL_ARGS=(pull --force --tmpdir "$APPTAINER_TMPDIR")
  if [[ -n "${DOCKER_USERNAME:-}" && -n "${DOCKER_PASSWORD:-}" ]]; then
    # Non-interactive registry auth: both runtimes honor these env vars.
    ok "using registry credentials for $DOCKER_USERNAME"
    export APPTAINER_DOCKER_USERNAME="$DOCKER_USERNAME" APPTAINER_DOCKER_PASSWORD="$DOCKER_PASSWORD"
    export SINGULARITY_DOCKER_USERNAME="$DOCKER_USERNAME" SINGULARITY_DOCKER_PASSWORD="$DOCKER_PASSWORD"
    PULL_ARGS+=(--docker-login)
  fi

  "$CONTAINER" "${PULL_ARGS[@]}" "$TMP_SIF" "$IMAGE_URI"
  pull_rc=$?

  if [[ $pull_rc -ne 0 || ! -f "$TMP_SIF" ]]; then
    no "pull failed (rc=$pull_rc)"
    exit 1
  fi
  mv -f "$TMP_SIF" "$SIF"
  trap - EXIT INT TERM
  ok "pulled $SIF"
fi

# ---- verify ----
hdr "Verify"
if "$CONTAINER" inspect "$SIF" >/dev/null 2>&1; then
  ok "SIF inspect OK"
else
  no "SIF inspect failed: $SIF"
  exit 1
fi

if [[ "$VERIFY" == "1" ]]; then
  # Mirror the sbatch invocation (--fakeroot --no-home) when possible; fall
  # back to a plain exec if fakeroot is unavailable but the tree is readable.
  if "$CONTAINER" exec --fakeroot --no-home "$SIF" ls "$ENV_PREFIX" >/dev/null 2>&1 \
     || "$CONTAINER" exec "$SIF" ls "$ENV_PREFIX" >/dev/null 2>&1; then
    ok "conda env present: $ENV_PREFIX"
  else
    no "conda env not readable at $ENV_PREFIX"
    exit 1
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    if "$CONTAINER" exec --nv --fakeroot --no-home "$SIF" \
          bash -lc 'source scripts/source_mujoco_setup.sh >/dev/null 2>&1 && python -c "import torch,mujoco_warp; print(torch.cuda.get_device_name(0))"'; then
      ok "torch + mujoco_warp import on GPU"
    else
      warn "GPU import check failed (image may still be usable; inspect manually)"
    fi
  else
    warn "no GPU on this node; skipped torch/mujoco_warp import check"
  fi
fi

hdr "Summary"
echo "  sif     : $SIF"
ls -lh "$SIF" 2>/dev/null | sed 's/^/  /'
echo "  image   : $IMAGE_URI"
echo "  env     : $ENV_PREFIX"
echo
echo "Ready. The MJWarp sbatch scripts default to this SIF; next:"
echo "  ALGO=ppo sbatch scripts/train_mjwarp_full.sbatch"
echo "  CHECKPOINT=... sbatch scripts/eval_mjwarp_video.sbatch"
