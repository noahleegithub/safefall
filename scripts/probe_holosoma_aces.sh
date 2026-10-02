#!/usr/bin/env bash
# probe_holosoma_aces_v2.sh
# Apptainer/Singularity probe that avoids two ACES pitfalls:
#   * FUSE "allow_other" is disabled in /etc/fuse.conf -> SIF mounts fall back to
#     full extraction.
#   * Extraction temp defaults land on /scratch (Lustre, 250k-file quota) -> quota.
# Fix: keep ALL temp/cache on node-local /tmp (xfs NVMe) and run from a sandbox.
#
# Usage:
#   srun --partition=gpu_debug --gres=gpu:h100:1 --cpus-per-task=8 --mem=0 \
#        --time=02:00:00 --pty bash -i
#   bash probe_holosoma_aces_v2.sh 2>&1 | tee "$SCRATCH/probe_v2.$(date +%s).log"
#
# Overridable: SIF, RUNS, SANDBOX, IMAGE_URI, CLEAN_SCRATCH (set 1 to delete leftover rootfs-*)
set -u

IMAGE_URI="${IMAGE_URI:-docker://ghcr.io/noahleegithub/safefall-mujoco:1.0.0}"
SIF="${SIF:-$SCRATCH/images/safefall-mujoco.sif}"
RUNS="${RUNS:-$SCRATCH/runs}"
SANDBOX="${SANDBOX:-${TMPDIR:-/tmp}/safefall-rootfs}"
ENV_PREFIX=/root/.holosoma_deps/miniconda3/envs/hsmujoco
CLEAN_SCRATCH="${CLEAN_SCRATCH:-0}"

# ---- force node-local temp/cache (never /scratch, never $HOME) ----
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${TMPDIR:-/tmp}/ap-tmp.$$}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${TMPDIR:-/tmp}/ap-cache.$$}"
export SINGULARITY_TMPDIR="$APPTAINER_TMPDIR"
export SINGULARITY_CACHEDIR="$APPTAINER_CACHEDIR"
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR" "$RUNS"

green(){ printf '\033[32m%s\033[0m' "$1"; }
red(){   printf '\033[31m%s\033[0m' "$1"; }
hdr(){   printf '\n=== %s ===\n' "$1"; }
ok(){    printf '  [%s] %s\n' "$(green PASS)" "$1"; }
no(){    printf '  [%s] %s\n' "$(red FAIL)" "$1"; }

hdr "Context"
echo "host        : $(hostname)"
echo "SLURM job   : ${SLURM_JOB_ID:-<none>}  mem/node=${SLURM_MEM_PER_NODE:-<unset>} cpus=${SLURM_CPUS_PER_TASK:-<unset>}"
command -v nvidia-smi >/dev/null && \
  echo "gpu         : $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
echo "TMPDIR      : $APPTAINER_TMPDIR"
df -T "$APPTAINER_TMPDIR" 2>/dev/null | tail -1
echo "sandbox     : $SANDBOX"

CONTAINER="$(command -v apptainer || command -v singularity || true)"
[[ -n "$CONTAINER" ]] || { echo "FATAL: no apptainer/singularity"; exit 1; }
echo "container   : $CONTAINER ($("$CONTAINER" --version 2>&1 | head -1))"

if [[ "$CLEAN_SCRATCH" == "1" ]]; then
  hdr "Reclaim scratch: removing leftover rootfs-* dirs"
  find "$SCRATCH" -maxdepth 2 -type d -name 'rootfs-*' -print -exec rm -rf {} + 2>/dev/null \
    && ok "cleaned" || echo "  (none found / not permitted)"
else
  hdr "Leftover scratch extractions (manual cleanup if quota is tight)"
  ls -ld "$SCRATCH"/rootfs-* 2>/dev/null || echo "  none found"
fi

hdr "Ensure SIF exists"
if [[ -f "$SIF" ]]; then
  ok "SIF present: $SIF"
else
  if type module >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then module load WebProxy || true; fi
  if "$CONTAINER" pull --tmpdir "$APPTAINER_TMPDIR" "$SIF" "$IMAGE_URI"; then
    ok "pulled $SIF"
  else
    no "pull failed"; exit 1
  fi
fi

hdr "Materialize sandbox on node-local disk (avoids FUSE + scratch quota)"
if [[ -d "$SANDBOX" ]]; then
  ok "sandbox already exists: $SANDBOX"
elif "$CONTAINER" build --sandbox "$SANDBOX" "$SIF"; then
  ok "built sandbox: $SANDBOX"
else
  no "sandbox build failed; falling back to the SIF (each exec will re-extract)"
  SANDBOX="$SIF"
fi
TARGET="$SANDBOX"
echo "  target      : $TARGET"
ls -ld "$TARGET/root" 2>/dev/null && echo "  ^ /root ownership/perms inside target"

hdr "Fakeroot availability (retested with temp on /tmp)"
FR=0
if "$CONTAINER" exec --fakeroot "$TARGET" id 2>&1 | grep -q 'uid=0(root)'; then
  ok "--fakeroot yields uid 0"; FR=1
else
  no "--fakeroot unavailable"; "$CONTAINER" exec --fakeroot "$TARGET" id 2>&1 | sed 's/^/      /' | head -5
fi

hdr "Read image conda tree ($ENV_PREFIX)"
ENV_READ=0
if "$CONTAINER" exec "$TARGET" ls "$ENV_PREFIX" >/dev/null 2>&1; then
  ok "$ENV_PREFIX readable"; ENV_READ=1
else
  no "$ENV_PREFIX not readable as plain user"
fi

hdr "HOME handling -> pick a working invocation"
HOME_ARGS=()
if [[ "$ENV_READ" == 1 || "$FR" == 1 ]]; then
  for variant in "" "--cleanenv --env HOME=/root" "--home /root" \
                 "--fakeroot --cleanenv --env HOME=/root" "--fakeroot --home /root"; do
    if [[ -n "$variant" ]]; then read -r -a a <<< "$variant"; else a=(); fi
    printf '  %-42s HOME=' "${variant:-<default>}"
    "$CONTAINER" exec "${a[@]}" "$TARGET" bash -lc 'echo "${HOME:-<unset>}"' 2>/dev/null || echo "(error)"
  done
  for variant in "--cleanenv --env HOME=/root" "--home /root" \
                 "--fakeroot --cleanenv --env HOME=/root" "" "--fakeroot"; do
    if [[ -n "$variant" ]]; then read -r -a a <<< "$variant"; else a=(); fi
    if "$CONTAINER" exec "${a[@]}" "$TARGET" bash -lc '[ "$HOME" = /root ] && [ -d /root/.holosoma_deps ]' 2>/dev/null; then
      HOME_ARGS=("${a[@]}"); break
    fi
  done
fi

hdr "End-to-end: activate env + GPU"
IMPORT_OK=0
if [[ ${#HOME_ARGS[@]} -gt 0 ]]; then
  echo "  using HOME args: ${HOME_ARGS[*]}"
  if "$CONTAINER" exec --nv "${HOME_ARGS[@]}" -B "$RUNS:/workspace/holosoma/logs" \
        --pwd /workspace/holosoma "$TARGET" bash -lc \
        'source scripts/source_mujoco_setup.sh && python -c "import torch,mujoco_warp; print(torch.cuda.get_device_name(0))"'; then
    ok "source_mujoco_setup.sh + imports + GPU"; IMPORT_OK=1
  else
    no "env activation/imports failed"
  fi
else
  echo "  no HOME-forcing variant; trying direct-python fallback:"
  if "$CONTAINER" exec --nv --fakeroot --env MUJOCO_GL=egl \
        --env LD_LIBRARY_PATH="$ENV_PREFIX/lib" --pwd /workspace/holosoma "$TARGET" \
        "$ENV_PREFIX/bin/python" -c "import torch,mujoco_warp; print(torch.cuda.get_device_name(0))"; then
    ok "direct-python fallback + GPU"; IMPORT_OK=1
  else
    no "direct-python fallback failed"
  fi
fi

hdr "Writable logs bind"
if "$CONTAINER" exec -B "$RUNS:/workspace/holosoma/logs" "$TARGET" \
      bash -lc 'touch /workspace/holosoma/logs/.probe && rm /workspace/holosoma/logs/.probe' 2>/dev/null; then
  ok "$RUNS writable from container"
else
  no "logs bind not writable"
fi

hdr "Summary"
echo "  fakeroot    : $([[ "$FR" == 1 ]] && echo yes || echo NO)"
echo "  env readable: $([[ "$ENV_READ" == 1 ]] && echo yes || echo NO)"
echo "  home args   : ${HOME_ARGS[*]:-<none>}"
echo "  gpu+imports : $([[ "$IMPORT_OK" == 1 ]] && echo yes || echo NO)"
echo "  target      : $TARGET"
echo
echo "If gpu+imports=yes -> reuse 'target' + 'home args' in the SLURM script."
echo "If NO and 'root' isn't readable -> the /root-based image can't run unprivileged;"
echo "  the durable fix is the lean /opt-based mujoco-hpc.Dockerfile rebuild."