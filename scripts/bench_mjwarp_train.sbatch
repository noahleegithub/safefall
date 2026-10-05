#!/usr/bin/env bash
#SBATCH --job-name=holosoma-bench
#SBATCH --account=158685321433
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=logs/holosoma-bench-%j.out
#
# Benchmark a Holosoma MJWarp training run at production scale to estimate
# full-run GPU-hours for PPO and FastSAC on one H100.
#
# Submit (Slurm forwards the submitting environment by default):
#   ALGO=ppo      ITERS=100 sbatch scripts/bench_mjwarp_train.sbatch
#   ALGO=fast_sac ITERS=400 sbatch scripts/bench_mjwarp_train.sbatch
#
# Overridable env vars:
#   ALGO (ppo|fast_sac), ITERS, NUM_ENVS, SEED, LOGGER, TOTAL_ITERS, REPO, SIF,
#   TERRAIN, SCALE_FACTOR
#
# Notes:
#   * MJWarp mixed terrain requires --terrain.terrain-term.scale-factor=0.5 to
#     avoid training instabilities (see src/holosoma/README.md); defaulted here.
#   * Headless: --training.headless=True and --logger.video.enabled=False.
#   * FastSAC's num_learning_iterations counts env steps, so ITERS=400 gives
#     ~4 logged intervals (100 steps each); the first is dropped as warmup.
set -euo pipefail

ALGO="${ALGO:-ppo}"                       # ppo | fast_sac
NUM_ENVS="${NUM_ENVS:-4096}"              # production scale
SEED="${SEED:-42}"
LOGGER="${LOGGER:-logger:disabled}"       # or logger:wandb_offline
TERRAIN="${TERRAIN:-terrain:terrain-locomotion-mix}"
SCALE_FACTOR="${SCALE_FACTOR:-0.5}"       # MJWarp mixed-terrain remedy (README)
REPO="${REPO:-$SCRATCH/safefall}"
SIF="${SIF:-$SCRATCH/images/safefall-mujoco.sif}"

case "$ALGO" in
  ppo)
    ITERS="${ITERS:-100}"
    TOTAL_ITERS="${TOTAL_ITERS:-25000}"
    EXP="exp:g1-29dof"
    ;;
  fast_sac)
    ITERS="${ITERS:-400}"
    TOTAL_ITERS="${TOTAL_ITERS:-50000}"
    EXP="exp:g1-29dof-fast-sac"
    ;;
  *)
    echo "ERROR: ALGO must be 'ppo' or 'fast_sac' (got '$ALGO')" >&2
    exit 2
    ;;
esac

RUNS="$SCRATCH/runs/$SLURM_JOB_ID"
CACHE="$RUNS/cache"
LOG="$RUNS/bench_${ALGO}.log"
CLEAN="$RUNS/bench_${ALGO}.clean.log"

[[ -f "$SIF" ]] || { echo "ERROR: image not found: $SIF" >&2; exit 1; }
[[ -d "$REPO" ]] || { echo "ERROR: repo checkout not found: $REPO" >&2; exit 1; }

mkdir -p "$RUNS" "$CACHE/warp" "$CACHE/matplotlib" "$CACHE/torch" \
         "$CACHE/triton" "$CACHE/torchinductor" "$CACHE/cuda"

# Keep Singularity's temporary extraction and cache on node-local NVMe.
export SINGULARITY_TMPDIR="$TMPDIR/singularity-tmp"
export APPTAINER_TMPDIR="$SINGULARITY_TMPDIR"
export SINGULARITY_CACHEDIR="$TMPDIR/singularity-cache"
export APPTAINER_CACHEDIR="$SINGULARITY_CACHEDIR"
mkdir -p "$SINGULARITY_TMPDIR" "$SINGULARITY_CACHEDIR"

# In-container training command, written to the bound RUNS dir to avoid nested
# quoting. Outer vars are expanded now; runtime command substitutions escaped.
cat > "$RUNS/run_in_container.sh" <<EOF
set -euo pipefail
source scripts/source_mujoco_setup.sh
export MUJOCO_GL=egl
echo "BENCH_PY_START \$(date +%s.%N)"
python -u src/holosoma/holosoma/train_agent.py \\
    ${EXP} simulator:mjwarp ${LOGGER} ${TERRAIN} \\
    --terrain.terrain-term.scale-factor=${SCALE_FACTOR} \\
    --training.headless=True \\
    --training.num-envs=${NUM_ENVS} \\
    --training.seed=${SEED} \\
    --algo.config.num-learning-iterations=${ITERS} \\
    --logger.video.enabled=False
echo "BENCH_PY_END \$(date +%s.%N)"
EOF

echo "=== BENCH CONFIG ==="
echo "algo=$ALGO exp=$EXP num_envs=$NUM_ENVS bench_iters=$ITERS total_iters=$TOTAL_ITERS seed=$SEED logger=$LOGGER terrain=$TERRAIN scale_factor=$SCALE_FACTOR"
echo "host=$(hostname) job=$SLURM_JOB_ID"
echo "sif=$SIF"
echo "repo=$REPO"
echo "runs=$RUNS"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

T0=$(date +%s.%N)

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

T1=$(date +%s.%N)

# Strip ANSI color codes for parsing.
sed -r 's/\x1B\[[0-9;]*[mK]//g' "$LOG" > "$CLEAN" || cp "$LOG" "$CLEAN"

if command -v python3 >/dev/null 2>&1; then
    python3 - "$CLEAN" "$ALGO" "$ITERS" "$TOTAL_ITERS" "$NUM_ENVS" "$T0" "$T1" <<'PY'
import re
import statistics
import sys

clean, algo, iters, total, num_envs, t0, t1 = (
    sys.argv[1],
    sys.argv[2],
    int(sys.argv[3]),
    int(sys.argv[4]),
    int(sys.argv[5]),
    float(sys.argv[6]),
    float(sys.argv[7]),
)
text = open(clean, encoding="utf-8", errors="replace").read()

# FastSAC logs once per `logging_interval` control steps, so the console box's
# "Iteration time" spans that many steps. Assumes the default (100).
FAST_SAC_LOG_INTERVAL = 100

m0 = re.search(r"BENCH_PY_START ([\d.]+)", text)
m1 = re.search(r"BENCH_PY_END ([\d.]+)", text)
train_wall = (float(m1.group(1)) - float(m0.group(1))) if (m0 and m1) else None
outer_wall = t1 - t0

it_times = [float(x) for x in re.findall(r"Iteration time:\s*([\d.]+)s", text)]
fps = [float(x) for x in re.findall(r"Computation:\s*([\d.]+) steps/s", text)]

print()
print("=== BENCH SUMMARY ===")
print(f"algorithm            : {algo}")
print(f"bench iters          : {iters}")
print(f"total iters (preset) : {total}")
if train_wall is None:
    print("training wall (s)    : <no BENCH_PY markers found>")
else:
    print(f"training wall (s)    : {train_wall:.1f}")
print(f"outer wall (s)       : {outer_wall:.1f}  (incl. SIF extraction + startup)")
if train_wall is not None:
    print(f"startup overhead (s) : {outer_wall - train_wall:.1f}")

if algo == "fast_sac":
    # One control step = one env.step across all envs; `total` counts control steps.
    per_step = None
    method = ""
    if it_times:
        steady = it_times[1:] if len(it_times) > 2 else it_times
        per_step = statistics.mean(steady) / FAST_SAC_LOG_INTERVAL
        method = f"iteration_time / {FAST_SAC_LOG_INTERVAL}"
    elif fps:
        # fps is env-steps/s, so one control step takes num_envs / fps.
        steady = fps[1:] if len(fps) > 2 else fps
        per_step = num_envs / statistics.mean(steady)
        method = "num_envs / steps_per_s"
    if per_step:
        print(f"per control-step (s) : {per_step:.5f}  ({method})")
        print(f"EST full-run GPU-h   : {total * per_step / 3600:.2f}")
    else:
        print("no timing found in log; inspect", clean)
elif it_times:
    steady = it_times[1:] if len(it_times) > 3 else it_times
    mean_it = statistics.mean(steady)
    print(f"iterations logged    : {len(it_times)}")
    print(f"first iter (s)       : {it_times[0]:.4f}")
    print(f"per-iter (s, steady) : {mean_it:.4f}")
    print(f"EST full-run GPU-h   : {total * mean_it / 3600:.2f}")
else:
    print("no timing found in log; inspect", clean)
PY
else
    echo "python3 not found on host; inspect $CLEAN manually"
fi

echo "Bench done (singularity rc=$SING_RC). Log: $LOG"
exit "$SING_RC"
