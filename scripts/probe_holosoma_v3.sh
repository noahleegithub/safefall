#!/usr/bin/env bash
# probe_holosoma_aces_v3.sh
# Focus: (1) confirm a real GPU is allocated, (2) find the conda env using
# --no-home so Singularity doesn't shadow the image's /root, (3) run imports.
#
# Launch with a REAL GPU:
#   srun --partition=gpu_debug --gres=gpu:a30:1 --cpus-per-task=8 --mem=0 --time=02:00:00 --pty bash -i
#   (or --partition=gpu --gres=gpu:h100:1)
set -u

SIF="${SIF:-$SCRATCH/images/safefall-mujoco.sif}"
RUNS="${RUNS:-$SCRATCH/runs}"

export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${TMPDIR:-/tmp}/ap-tmp.$$}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${TMPDIR:-/tmp}/ap-cache.$$}"
export SINGULARITY_TMPDIR="$APPTAINER_TMPDIR" SINGULARITY_CACHEDIR="$APPTAINER_CACHEDIR"
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR" "$RUNS"

green(){ printf '\033[32m%s\033[0m' "$1"; }
red(){ printf '\033[31m%s\033[0m' "$1"; }
hdr(){ printf '\n=== %s ===\n' "$1"; }
ok(){ printf '  [%s] %s\n' "$(green PASS)" "$1"; }
no(){ printf '  [%s] %s\n' "$(red FAIL)" "$1"; }

hdr "Context"
echo "host=$HOSTNAME job=${SLURM_JOB_ID:-none} gres=${SLURM_GRES:-<unset>} cpus=${SLURM_CPUS_PER_TASK:-<unset>} mem/node=${SLURM_MEM_PER_NODE:-<unset>}"
echo "GPUs: $(nvidia-smi -L 2>&1 | tr '\n' ' ')"
if nvidia-smi -L >/dev/null 2>&1; then ok "host sees a GPU"; else
  no "host does NOT see a GPU -> relaunch with --gres=gpu:a30:1 (gpu_debug) or --gres=gpu:h100:1 (gpu)"; fi
echo "temp=$APPTAINER_TMPDIR"; df -T "$APPTAINER_TMPDIR" 2>/dev/null | tail -1

CONTAINER="$(command -v apptainer || command -v singularity)"
[[ -f "$SIF" ]] || { no "missing SIF: $SIF"; exit 1; }
echo "container=$($CONTAINER --version 2>&1 | head -1)"

hdr "Diagnose: --fakeroot --no-home (single extraction)"
"$CONTAINER" exec --nv --fakeroot --no-home --pwd /workspace/holosoma "$SIF" bash -lc '
  echo "id: $(id)"
  echo "HOME=${HOME:-<unset>}"
  echo "--- /root ---"; ls -la /root 2>&1 | head -40
  echo "--- find hsmujoco ---"; find / -maxdepth 6 -type d -name hsmujoco 2>/dev/null
  echo "--- find .holosoma_deps ---"; find / -maxdepth 5 -type d -name .holosoma_deps 2>/dev/null
  echo "--- find miniconda3 ---"; find / -maxdepth 5 -type d -name miniconda3 2>/dev/null
'
RC=$?; [[ $RC -eq 0 ]] && ok "diagnostic exec ran" || no "diagnostic exec rc=$RC"

hdr "PATH-B -- try old /root path directly with --no-home"
"$CONTAINER" exec --nv --fakeroot --no-home \
  --env MUJOCO_GL=egl \
  --env LD_LIBRARY_PATH=/root/.holosoma_deps/miniconda3/envs/hsmujoco/lib \
  --pwd /workspace/holosoma "$SIF" \
  /root/.holosoma_deps/miniconda3/envs/hsmujoco/bin/python \
  -c "import torch,mujoco_warp; print(torch.cuda.get_device_name(0))" \
  && ok "imports + GPU OK at /root path" || no "imports failed at /root path"

hdr "Optional: reusable sandbox from the registry (avoids per-exec extraction)"
SANDBOX="${SANDBOX:-${TMPDIR:-/tmp}/safefall-rootfs}"
if [[ -d "$SANDBOX" ]]; then ok "sandbox exists: $SANDBOX"; else
  if type module >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then module load WebProxy || true; fi
  if "$CONTAINER" build --sandbox --force "$SANDBOX" docker://ghcr.io/noahleegithub/safefall-mujoco:1.0.0; then
    ok "built sandbox: $SANDBOX"
  else
    no "sandbox build from registry failed (SIF path still works, just slower)"
  fi
fi