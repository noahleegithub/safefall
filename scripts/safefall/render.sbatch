#!/usr/bin/env bash
#SBATCH --job-name=safefall-render
#SBATCH --account=158685321433
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/safefall-render-%j.out
#
# Render selected benchmark trials with the mjwarp camera path (no EGL/GL).
#
# Example:
#   CHECKPOINT=$SCRATCH/runs/<train>/.../model_025000.pt \
#     TRIAL_IDS="3 17 42" BASELINES="nominal damping" \
#     sbatch scripts/safefall/render.sbatch
#
# Overridable: CHECKPOINT, TRIAL_IDS, BASELINES, RUN_ROOT, EXTRA_ARGS, REPO, SIF.
set -euo pipefail

CHECKPOINT="${CHECKPOINT:-}"
TRIAL_IDS="${TRIAL_IDS:-0}"
BASELINES="${BASELINES:-nominal}"
RUN_ROOT="${RUN_ROOT:-$SCRATCH/runs/${SLURM_JOB_ID}}"
REPO="${REPO:-$SCRATCH/safefall}"
SIF="${SIF:-$SCRATCH/images/safefall-mujoco.sif}"
VIDEO_DIR="$RUN_ROOT/safefall/videos"
CACHE="$RUN_ROOT/cache"

[[ -n "$CHECKPOINT" ]] || { echo "ERROR: CHECKPOINT is required" >&2; exit 2; }
[[ -f "$SIF" ]] || { echo "ERROR: image not found: $SIF" >&2; exit 1; }
[[ -d "$REPO" ]] || { echo "ERROR: repo checkout not found: $REPO" >&2; exit 1; }
mkdir -p "$VIDEO_DIR" "$CACHE/warp" "$CACHE/matplotlib" "$CACHE/torch" "$CACHE/triton" "$CACHE/cuda"

if [[ "$CHECKPOINT" == wandb://* ]]; then
  CKPT_ARG="$CHECKPOINT"
else
  mkdir -p "$RUN_ROOT/checkpoint"
  cp "$CHECKPOINT" "$RUN_ROOT/checkpoint/"
  CKPT_ARG="/bench/checkpoint/$(basename "$CHECKPOINT")"
fi

export SINGULARITY_TMPDIR="${TMPDIR:-/tmp}/singularity-tmp.$$"
export APPTAINER_TMPDIR="$SINGULARITY_TMPDIR"
export SINGULARITY_CACHEDIR="${TMPDIR:-/tmp}/singularity-cache.$$"
export APPTAINER_CACHEDIR="$SINGULARITY_CACHEDIR"
mkdir -p "$SINGULARITY_TMPDIR" "$SINGULARITY_CACHEDIR"

JOBS=()
for tid in $TRIAL_IDS; do
  for baseline in $BASELINES; do
    JOBS+=("$tid:$baseline")
  done
done

cat > "$RUN_ROOT/run_render.sh" <<EOF
set -euo pipefail
source scripts/source_mujoco_setup.sh
for job in ${JOBS[*]}; do
  tid="\${job%%:*}"
  baseline="\${job##*:}"
  echo "=== render trial=\$tid baseline=\$baseline ==="
  python -u scripts/safefall/benchmark.py \\
    --checkpoint=${CKPT_ARG} \\
    --output-dir=/bench/safefall \\
    --num-envs=1 \\
    --seed=${SEED:-0} \\
    --render-trial-id="\$tid" \\
    --render-baseline="\$baseline" \\
    --render-dir=/bench/safefall/videos \\
    --render-playback-rate=${PLAYBACK_RATE:-0.25} \\
    ${EXTRA_ARGS:-}
done
EOF

echo "Rendering: ${JOBS[*]}"
set +e
singularity exec \
  --nv --fakeroot --no-home \
  --bind "$REPO:/workspace/holosoma" \
  --bind "$RUN_ROOT:/bench" \
  --env WARP_CACHE_PATH="$CACHE/warp" \
  --env XDG_CACHE_HOME="$CACHE" \
  --env MPLCONFIGDIR="$CACHE/matplotlib" \
  --env TORCH_HOME="$CACHE/torch" \
  --env TRITON_CACHE_DIR="$CACHE/triton" \
  --env TORCHINDUCTOR_CACHE_DIR="$CACHE/torchinductor" \
  --env CUDA_CACHE_PATH="$CACHE/cuda" \
  --pwd /workspace/holosoma \
  "$SIF" bash -lc 'bash /bench/run_render.sh' 2>&1 | tee "$RUN_ROOT/render.log"
RC=${PIPESTATUS[0]}
set -e
echo "Videos: $VIDEO_DIR"
ls -lh "$VIDEO_DIR" || true
exit "$RC"
