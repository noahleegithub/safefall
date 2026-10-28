#!/usr/bin/env bash
# Generate the SafeFall report (tables + plots + HTML gallery) from a bench run.
#
# Usage (login node or any node with the SIF):
#   RUN_ROOT=$SCRATCH/runs/<array_job_id> bash scripts/safefall/report.sh
#
# Overridable: RUN_ROOT, REPO, SIF, OUT_DIR.
set -euo pipefail

RUN_ROOT="${RUN_ROOT:-}"
REPO="${REPO:-$SCRATCH/safefall}"
SIF="${SIF:-$SCRATCH/images/safefall-mujoco.sif}"
OUT_DIR="${OUT_DIR:-/bench/safefall/report}"

[[ -n "$RUN_ROOT" ]] || { echo "ERROR: RUN_ROOT is required (e.g. \$SCRATCH/runs/123456)" >&2; exit 2; }
[[ -d "$RUN_ROOT/safefall" ]] || { echo "ERROR: no safefall dir under $RUN_ROOT" >&2; exit 1; }
[[ -f "$SIF" ]] || { echo "ERROR: image not found: $SIF" >&2; exit 1; }

exec singularity exec \
    --fakeroot \
    --no-home \
    --bind "$REPO:/workspace/holosoma" \
    --bind "$RUN_ROOT:/bench" \
    --pwd /workspace/holosoma \
    "$SIF" bash -lc "python -u scripts/safefall/report.py --run-dir=/bench/safefall --out-dir=$OUT_DIR"
