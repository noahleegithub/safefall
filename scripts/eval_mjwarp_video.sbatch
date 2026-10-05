#!/usr/bin/env bash
#SBATCH --job-name=holosoma-eval
#SBATCH --account=158685321433
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/holosoma-eval-%j.out
#
# Evaluate a trained MJWarp checkpoint and render a video of the policy.
#
# Rendering uses the MJWarp GPU camera path (mujoco_warp mjw.render) via the `viz-record`
# plugin, so it does NOT need EGL/GL and works headless. Do not set MUJOCO_GL here.
#
# Submit:
#   CHECKPOINT=$SCRATCH/runs/<train_job>/<project>/<ts>-.../model_025000.pt \
#     sbatch scripts/eval_mjwarp_video.sbatch
#
# Overridable env vars:
#   CHECKPOINT (required; local .pt or wandb://...), CAMERA, EVAL_STEPS,
#   CAM_W, CAM_H, PLAYBACK_RATE, EXTRA_ARGS, REPO, SIF
#
# Notes:
#   * Video saved to $RUNS/eval_videos/ (host: $SCRATCH/runs/$SLURM_JOB_ID/).
#   * eval_agent closes the simulator after evaluation, which fires the CLOSE hook the
#     camera recorder uses to encode/save the video.
set -euo pipefail

CHECKPOINT="${CHECKPOINT:-}"
CAMERA="${CAMERA:-overview}"
EVAL_STEPS="${EVAL_STEPS:-1000}"
CAM_W="${CAM_W:-640}"
CAM_H="${CAM_H:-480}"
PLAYBACK_RATE="${PLAYBACK_RATE:-1.0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
REPO="${REPO:-$SCRATCH/safefall}"
SIF="${SIF:-$SCRATCH/images/safefall-mujoco.sif}"

[[ -n "$CHECKPOINT" ]] || { echo "ERROR: CHECKPOINT is required" >&2; exit 2; }
[[ -f "$SIF" ]] || { echo "ERROR: image not found: $SIF" >&2; exit 1; }
[[ -d "$REPO" ]] || { echo "ERROR: repo checkout not found: $REPO" >&2; exit 1; }

RUNS="$SCRATCH/runs/$SLURM_JOB_ID"
CACHE="$RUNS/cache"
VIDEO_DIR="$RUNS/eval_videos"
LOG="$RUNS/eval.log"

mkdir -p "$RUNS" "$VIDEO_DIR" "$CACHE/warp" "$CACHE/matplotlib" "$CACHE/torch" \
         "$CACHE/triton" "$CACHE/torchinductor" "$CACHE/cuda"

# Copy the checkpoint into the writable run dir: eval writes <ckpt_dir>/exported/.
mkdir -p "$RUNS/checkpoint"
if [[ "$CHECKPOINT" == wandb://* ]]; then
  CKPT_ARG="$CHECKPOINT"
else
  [[ -f "$CHECKPOINT" ]] || { echo "ERROR: checkpoint not found: $CHECKPOINT" >&2; exit 1; }
  cp "$CHECKPOINT" "$RUNS/checkpoint/"
  CKPT_ARG="/bench/checkpoint/$(basename "$CHECKPOINT")"
fi

# Keep Singularity's temp extraction and cache on node-local NVMe.
export SINGULARITY_TMPDIR="$TMPDIR/singularity-tmp"
export APPTAINER_TMPDIR="$SINGULARITY_TMPDIR"
export SINGULARITY_CACHEDIR="$TMPDIR/singularity-cache"
export APPTAINER_CACHEDIR="$SINGULARITY_CACHEDIR"
mkdir -p "$SINGULARITY_TMPDIR" "$SINGULARITY_CACHEDIR"

cat > "$RUNS/run_eval.sh" <<EOF
set -euo pipefail
source scripts/source_mujoco_setup.sh
python -u src/holosoma/holosoma/eval_agent.py \\
    --checkpoint=${CKPT_ARG} \\
    --training.headless=True \\
    --training.max-eval-steps=${EVAL_STEPS} \\
    --training.export-onnx=False \\
    sensor.cam:${CAMERA} \\
    plugin.rec:viz-record \\
    --sensor.cam.width=${CAM_W} \\
    --sensor.cam.height=${CAM_H} \\
    --plugin.rec.save-dir=/bench/eval_videos \\
    --plugin.rec.playback-rate=${PLAYBACK_RATE} ${EXTRA_ARGS}
EOF

echo "=== EVAL CONFIG ==="
echo "checkpoint=$CKPT_ARG camera=$CAMERA eval_steps=$EVAL_STEPS res=${CAM_W}x${CAM_H} playback_rate=$PLAYBACK_RATE"
echo "host=$(hostname) job=$SLURM_JOB_ID sif=$SIF repo=$REPO runs=$RUNS"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

set +e
singularity exec \
    --nv \
    --fakeroot \
    --no-home \
    --bind "$REPO:/workspace/holosoma" \
    --bind "$RUNS:/workspace/holosoma/logs" \
    --bind "$RUNS:/bench" \
    --env WARP_CACHE_PATH="$CACHE/warp" \
    --env XDG_CACHE_HOME="$CACHE" \
    --env MPLCONFIGDIR="$CACHE/matplotlib" \
    --env TORCH_HOME="$CACHE/torch" \
    --env TRITON_CACHE_DIR="$CACHE/triton" \
    --env TORCHINDUCTOR_CACHE_DIR="$CACHE/torchinductor" \
    --env CUDA_CACHE_PATH="$CACHE/cuda" \
    --pwd /workspace/holosoma \
    "$SIF" bash -lc 'bash /bench/run_eval.sh' 2>&1 | tee "$LOG"
SING_RC=${PIPESTATUS[0]}
set -e

echo "Eval done (singularity rc=$SING_RC)."
echo "Videos: $VIDEO_DIR"
ls -lh "$VIDEO_DIR" || true
exit "$SING_RC"
