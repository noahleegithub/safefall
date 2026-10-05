#!/usr/bin/env bash
#SBATCH --job-name=holosoma-train
#SBATCH --account=158685321433
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=logs/holosoma-train-%j.out
#
# Full MJWarp training run for PPO or FastSAC on one H100 (G1 29-DoF locomotion).
#
# Submit (override walltime per algo as needed):
#   ALGO=ppo      sbatch --time=12:00:00 scripts/train_mjwarp_full.sbatch
#   ALGO=fast_sac sbatch --time=04:00:00 scripts/train_mjwarp_full.sbatch
#
# Overridable env vars:
#   ALGO (ppo|fast_sac), NUM_ENVS, SEED, LOGGER, SCALE_FACTOR, TERRAIN,
#   PROJECT, RUN_NAME, GROUP, SAVE_INTERVAL, TOTAL_ITERS, CHECKPOINT, VIDEO,
#   EXTRA_ARGS, REPO, SIF
#
# Notes:
#   * MJWarp mixed terrain requires --terrain.terrain-term.scale-factor=0.5;
#     defaulted here (see src/holosoma/README.md).
#   * Bench estimate at 4096 envs, scale 0.5: PPO ~9 GPU-h, FastSAC ~1-3 GPU-h.
#   * Resume/continue: CHECKPOINT=/path/to/model_XXXXX.pt (or wandb://...).
#   * Video off by default; VIDEO=1 enables periodic recordings (uses EGL).
set -euo pipefail

ALGO="${ALGO:-ppo}"                        # ppo | fast_sac
NUM_ENVS="${NUM_ENVS:-4096}"
SEED="${SEED:-42}"
TERRAIN="${TERRAIN:-terrain:terrain-locomotion-mix}"
SCALE_FACTOR="${SCALE_FACTOR:-0.5}"
LOGGER="${LOGGER:-logger:disabled}"        # logger:disabled | logger:wandb_offline | logger:wandb
PROJECT="${PROJECT:-}"
RUN_NAME="${RUN_NAME:-}"
GROUP="${GROUP:-}"
CHECKPOINT="${CHECKPOINT:-}"
VIDEO="${VIDEO:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
REPO="${REPO:-$SCRATCH/safefall}"
SIF="${SIF:-$SCRATCH/images/safefall-mujoco.sif}"

case "$ALGO" in
  ppo)
    TOTAL_ITERS="${TOTAL_ITERS:-25000}"
    SAVE_INTERVAL="${SAVE_INTERVAL:-500}"
    EXP="exp:g1-29dof"
    ;;
  fast_sac)
    TOTAL_ITERS="${TOTAL_ITERS:-50000}"
    SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
    EXP="exp:g1-29dof-fast-sac"
    ;;
  *)
    echo "ERROR: ALGO must be 'ppo' or 'fast_sac' (got '$ALGO')" >&2
    exit 2
    ;;
esac

RUNS="$SCRATCH/runs/$SLURM_JOB_ID"
CACHE="$RUNS/cache"
LOG="$RUNS/train_${ALGO}.log"

[[ -f "$SIF" ]] || { echo "ERROR: image not found: $SIF" >&2; exit 1; }
[[ -d "$REPO" ]] || { echo "ERROR: repo checkout not found: $REPO" >&2; exit 1; }

mkdir -p "$RUNS" "$CACHE/warp" "$CACHE/matplotlib" "$CACHE/torch" \
         "$CACHE/triton" "$CACHE/torchinductor" "$CACHE/cuda"

# Keep Singularity's temp extraction and cache on node-local NVMe.
export SINGULARITY_TMPDIR="$TMPDIR/singularity-tmp"
export APPTAINER_TMPDIR="$SINGULARITY_TMPDIR"
export SINGULARITY_CACHEDIR="$TMPDIR/singularity-cache"
export APPTAINER_CACHEDIR="$SINGULARITY_CACHEDIR"
mkdir -p "$SINGULARITY_TMPDIR" "$SINGULARITY_CACHEDIR"

# Assemble flags (array so nothing is word-split incorrectly).
FLAGS=(
  "--algo.config.num-learning-iterations=${TOTAL_ITERS}"
  "--algo.config.save-interval=${SAVE_INTERVAL}"
  "--terrain.terrain-term.scale-factor=${SCALE_FACTOR}"
  "--training.headless=True"
  "--training.num-envs=${NUM_ENVS}"
  "--training.seed=${SEED}"
)
if [[ "$VIDEO" == "1" ]]; then
  FLAGS+=("--logger.headless-recording=True" "--logger.video.enabled=True")
else
  FLAGS+=("--logger.video.enabled=False")
fi
if [[ "$LOGGER" != "logger:disabled" ]]; then
  [[ -n "$PROJECT" ]] && FLAGS+=("--logger.project=${PROJECT}")
  [[ -n "$RUN_NAME" ]] && FLAGS+=("--logger.name=${RUN_NAME}")
  [[ -n "$GROUP" ]] && FLAGS+=("--logger.group=${GROUP}")
fi
[[ -n "$CHECKPOINT" ]] && FLAGS+=("--training.checkpoint=${CHECKPOINT}")
if [[ -n "$EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206
  FLAGS+=($EXTRA_ARGS)
fi

cat > "$RUNS/run_in_container.sh" <<EOF
set -euo pipefail
source scripts/source_mujoco_setup.sh
export MUJOCO_GL=egl
echo "TRAIN_PY_START \$(date +%s.%N)"
python -u src/holosoma/holosoma/train_agent.py \\
    ${EXP} simulator:mjwarp ${LOGGER} ${TERRAIN} \\
    ${FLAGS[*]}
echo "TRAIN_PY_END \$(date +%s.%N)"
EOF

echo "=== TRAIN CONFIG ==="
echo "algo=$ALGO exp=$EXP num_envs=$NUM_ENVS total_iters=$TOTAL_ITERS seed=$SEED"
echo "terrain=$TERRAIN scale_factor=$SCALE_FACTOR save_interval=$SAVE_INTERVAL"
echo "logger=$LOGGER project=${PROJECT:-<default>} run_name=${RUN_NAME:-<default>} group=${GROUP:-<default>}"
echo "checkpoint=${CHECKPOINT:-<none>} video=$VIDEO"
echo "host=$(hostname) job=$SLURM_JOB_ID"
echo "sif=$SIF  repo=$REPO  runs=$RUNS"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

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
    "$SIF" bash -lc 'bash /bench/run_in_container.sh' 2>&1 | tee "$LOG"
SING_RC=${PIPESTATUS[0]}
set -e

echo "Training done (singularity rc=$SING_RC). Log: $LOG"
echo "Checkpoints/metrics: $RUNS/<project>/<timestamp>-.../"
exit "$SING_RC"
