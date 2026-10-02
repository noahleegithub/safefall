#!/usr/bin/env bash
#SBATCH --job-name=holosoma-smoke
#SBATCH --account=158685321433
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=holosoma-smoke-%j.out

set -euo pipefail

SIF="$SCRATCH/images/safefall-mujoco.sif"
RUNS="$SCRATCH/runs/$SLURM_JOB_ID"
CACHE="$RUNS/cache"

[[ -f "$SIF" ]] || { echo "ERROR: image not found: $SIF" >&2; exit 1; }
mkdir -p "$RUNS" "$CACHE/warp" "$CACHE/matplotlib" "$CACHE/torch"

# Keep Singularity's temporary extraction and cache on node-local NVMe.
export SINGULARITY_TMPDIR="$TMPDIR/singularity-tmp"
export APPTAINER_TMPDIR="$SINGULARITY_TMPDIR"
export SINGULARITY_CACHEDIR="$TMPDIR/singularity-cache"
export APPTAINER_CACHEDIR="$SINGULARITY_CACHEDIR"
mkdir -p "$SINGULARITY_TMPDIR" "$SINGULARITY_CACHEDIR"

echo "Host: $(hostname)  Job: $SLURM_JOB_ID"
echo "Image: $SIF"
echo "Results: $RUNS"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

singularity exec \
    --nv \
    --fakeroot \
    --no-home \
    --bind "$SCRATCH/safefall:/workspace/holosoma" \
    --bind "$RUNS:/workspace/holosoma/logs" \
    --env WARP_CACHE_PATH="$CACHE/warp" \
    --env XDG_CACHE_HOME="$CACHE" \
    --env MPLCONFIGDIR="$CACHE/matplotlib" \
    --env TORCH_HOME="$CACHE/torch" \
    --pwd /workspace/holosoma \
    "$SIF" \
    bash -lc '
        set -euo pipefail
        source scripts/source_mujoco_setup.sh
        export MUJOCO_GL=egl

        python src/holosoma/holosoma/train_agent.py \
            exp:g1-29dof \
            simulator:mjwarp \
            logger:disabled \
            terrain:terrain-locomotion-plane \
            --training.num-envs=256 \
            --algo.config.num-learning-iterations=5 \
            --training.headless=True
    '

echo "Smoke test finished. Results: $RUNS"