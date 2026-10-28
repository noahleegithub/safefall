#!/usr/bin/env bash
#SBATCH --job-name=safefall-bench
#SBATCH --account=158685321433
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/safefall-bench-%A_%a.out
#
# SafeFall baseline benchmark, one shard per array task.
#
# Submit:
#   CHECKPOINT=$SCRATCH/runs/<train>/.../model_025000.pt \
#     TRIALS=5000 SHARDS=8 NUM_ENVS=256 \
#     sbatch --array=0-7 scripts/safefall/bench.sbatch
#
# Overridable env vars:
#   CHECKPOINT (required; local .pt or wandb://...), TRIALS (total across shards),
#   SHARDS (must match the array size), NUM_ENVS, SEED, EXTRA_ARGS, REPO, SIF, RUN_ROOT
#
# Notes:
#   * Each shard gets trials [shard, shard+SHARDS, ...], so the scenario set is
#     deterministic and disjoint.
#   * Results: $RUN_ROOT/$SLURM_ARRAY_JOB_ID/safefall/trials_shard_<shard>.csv
#     and scenarios_shard_<shard>.jsonl.  Run scripts/safefall/report.sh after.
#   * The repo is bind-mounted live; no SIF rebuild for Python changes.
set -euo pipefail

CHECKPOINT="${CHECKPOINT:-}"
TRIALS="${TRIALS:-200}"
SHARDS="${SHARDS:-8}"
NUM_ENVS="${NUM_ENVS:-256}"
SEED="${SEED:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
REPO="${REPO:-$SCRATCH/safefall}"
SIF="${SIF:-$SCRATCH/images/safefall-mujoco.sif}"
RUN_ROOT="${RUN_ROOT:-$SCRATCH/runs/${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}}"
OUT="$RUN_ROOT/safefall"
CACHE="$RUN_ROOT/cache"
LOG="$RUN_ROOT/bench_${SLURM_ARRAY_TASK_ID:-0}.log"

[[ -n "$CHECKPOINT" ]] || { echo "ERROR: CHECKPOINT is required" >&2; exit 2; }
[[ -f "$SIF" ]] || { echo "ERROR: image not found: $SIF" >&2; exit 1; }
[[ -d "$REPO" ]] || { echo "ERROR: repo checkout not found: $REPO" >&2; exit 1; }
mkdir -p "$OUT" "$CACHE/warp" "$CACHE/matplotlib" "$CACHE/torch" \
         "$CACHE/triton" "$CACHE/torchinductor" "$CACHE/cuda"

mkdir -p "$RUN_ROOT/checkpoint"
if [[ "$CHECKPOINT" == wandb://* ]]; then
  CKPT_ARG="$CHECKPOINT"
else
  [[ -f "$CHECKPOINT" ]] || { echo "ERROR: checkpoint not found: $CHECKPOINT" >&2; exit 1; }
  cp "$CHECKPOINT" "$RUN_ROOT/checkpoint/"
  CKPT_ARG="/bench/checkpoint/$(basename "$CHECKPOINT")"
fi

# Keep Singularity temp/cache off Lustre.
export SINGULARITY_TMPDIR="${TMPDIR:-/tmp}/singularity-tmp.$$"
export APPTAINER_TMPDIR="$SINGULARITY_TMPDIR"
export SINGULARITY_CACHEDIR="${TMPDIR:-/tmp}/singularity-cache.$$"
export APPTAINER_CACHEDIR="$SINGULARITY_CACHEDIR"
mkdir -p "$SINGULARITY_TMPDIR" "$SINGULARITY_CACHEDIR"

SHARD="${SLURM_ARRAY_TASK_ID:-0}"

cat > "$RUN_ROOT/run_bench_${SHARD}.sh" <<EOF
set -euo pipefail
source scripts/source_mujoco_setup.sh
python -u scripts/safefall/benchmark.py \\
    --checkpoint=${CKPT_ARG} \\
    --output-dir=/bench/safefall \\
    --trials=${TRIALS} \\
    --num-envs=${NUM_ENVS} \\
    --seed=${SEED} \\
    --shard=${SHARD} \\
    --num-shards=${SHARDS} \\
    ${EXTRA_ARGS}
EOF

echo "=== SAFEFALL BENCH ==="
echo "job=$SLURM_ARRAY_JOB_ID shard=$SHARD/$SHARDS trials=$TRIALS envs=$NUM_ENVS seed=$SEED"
echo "checkpoint=$CKPT_ARG repo=$REPO runs=$RUN_ROOT"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

set +e
singularity exec \
    --nv \
    --fakeroot \
    --no-home \
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
    "$SIF" bash -lc "bash /bench/run_bench_${SHARD}.sh" 2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
set -e

echo "Shard done (rc=$RC)."
echo "Results: $OUT"
ls -lh "$OUT" || true
exit "$RC"
