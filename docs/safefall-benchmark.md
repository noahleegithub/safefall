# SafeFall baseline benchmark

This is the measurement + baseline-evaluation half of a SafeFall replication
(Meng et al., [arXiv:2511.18509](https://arxiv.org/abs/2511.18509)). It does **not**
train the fall predictor or the protective policy yet. It runs walking + perturbation
fall trials on a trained nominal locomotion policy and evaluates the paper's three
baselines from the same trigger state:

| baseline | controller after `T - 300 ms` |
|---|---|
| `nominal` | the locomotion policy continues through the fall |
| `default` | hold the default joint targets with nominal PD gains |
| `damping` | `Kp ≈ 0`, `Kd = 10`, constant target, zero target velocity |

## Metrics

Implemented in `holosoma/simulator/mujoco/impact_metrics.py` and sampled at every
physics substep:

| metric | signal | definition |
|---|---|---|
| `max_joint_torque` | `qfrc_constraint` | peak `|τ|` over the 29 robot joints (constraint torque, excluding motor output) |
| `max_torque_ratio` | `qfrc_constraint` | peak `|τ| / τ_max` using `dof_effort_limit_list` |
| `max_joint_reaction` | `cfrc_int` | peak force transmitted through each joint's child body |
| `max_contact_force` | per-contact solver force | peak per-link **ground** contact force (non-robot body only) |
| `max_impulse` | per-contact force | max per-physics-step ground reaction impulse, `|ΣF| dt` |
| `cumulative_impulse` | per-contact force | sum of per-step impulses over the trial window |
| `illegal_contact` | per-contact body pair | 1 if a protected link (head / hand) contacted ground or a non-adjacent link |
| `max_limit_violations` | `qpos` vs hard limits | max number of joints outside position limits at one substep |

`cfrc_int` and `cfrc_ext` are `[torque(3), force(3)]`; the code uses the force half.

## Fall scenarios

A trial is: walk 1-2 s with the nominal policy, then apply 1-3 of:

- **push** — horizontal base velocity perturbation, 1.5-3.5 m/s
- **slip** — lower the foot sliding friction to `[0.02, 0.15]` for 0.1-0.3 s
- **trip** — 150-400 N horizontal force on the swing foot for ~50 ms
- **freeze** — motor-off (zero torque) or brake-lock (hold the failure angle) on 1-2 joints

The first non-foot ground contact after the perturbation is `T`. The oracle trigger is
`T - 300 ms`; the evaluated window is 40 control steps (0.8 s at 50 Hz). The nominal
baseline runs continuously; `default`/`damping` branch from the saved trigger state.

The benchmark disables domain randomization and termination so all baselines are paired
and the post-impact sequence is not cut short.

## HPRC / ACES workflow

```bash
# 1. Pilot (one shard, 200 trials)
CHECKPOINT=$SCRATCH/runs/<train>/.../model_025000.pt \
  TRIALS=200 SHARDS=1 NUM_ENVS=64 SEED=0 \
  sbatch --array=0 scripts/safefall/bench.sbatch

# 2. Full run (8 shards, 5000 trials)
CHECKPOINT=$SCRATCH/runs/<train>/.../model_025000.pt \
  TRIALS=5000 SHARDS=8 NUM_ENVS=256 SEED=0 \
  sbatch --array=0-7 scripts/safefall/bench.sbatch

# 3. Report (login node or CPU node)
RUN_ROOT=$SCRATCH/runs/<array_job_id> bash scripts/safefall/report.sh

# 4. Optional videos for selected trials
CHECKPOINT=$SCRATCH/runs/<train>/.../model_025000.pt \
  TRIAL_IDS="3 17 42" BASELINES="nominal default damping" \
  RUN_ROOT=$SCRATCH/runs/<render_job_id> \
  sbatch scripts/safefall/render.sbatch
```

Outputs (under `$SCRATCH/runs/<array_job_id>/safefall/`):

```
trials_shard_<s>.csv       one row per (trial, baseline)
scenarios_shard_<s>.jsonl  scenario spec + T/trigger/outcome
report/summary.csv         Table-IV-shaped overall summary
report/summary_by_mechanism.csv
report/paired_reductions.csv
report/metric_*.png/pdf    overall bars
report/mechanism_*.png/pdf per-mechanism bars
report/dist_*.png          distributions
report/index.html          gallery (plots + videos)
videos/trial_<id>_<baseline>.mp4
```

`TRIALS` is the total across shards; each shard takes `trial_id % shards == shard`.
`SHARDS` must match the Slurm array size.

## Important caveats

- **Check the force extraction before trusting numbers.** In `mujoco_warp`,
  `cfrc_ext`/`cfrc_int` are Warp `spatial_vector`s with the angular (torque) component
  first. The pre-existing `WarpBackend.compute_contact_forces()` returns
  `cfrc_t[..., :3]`, which is likely the **torque**, not the force. This benchmark uses
  the new explicit `[..., 3:6]` force path and per-contact forces, so its metrics are
  unaffected; the legacy `contact_forces` used by termination/rewards should be fixed
  separately with an A/B test.
- Absolute force/torque values depend on the contact model (`solref`/`solimp`),
  collision geometry, timestep, and friction. Reproduce relative baseline ordering and
  reductions first; absolute matching to the paper requires calibration.
- The paper's `J` (whole-body impulse) is ambiguous in the text; both max per-step and
  cumulative impulse are recorded.
- The paper's trigger lead is ~0.3-0.4 s (the 100 ms value is the predictor label
  boundary, not the activation time). The oracle uses `T - 300 ms` by default.
- The paper's failure set includes sensor noise, system delay, and dynamics mismatch;
  this benchmark implements the physical factors (push/slip/trip/freeze) first.
- Recommended pilot checks: per-foot force ≈ mg/2 at rest, baseline ordering
  `nominal > default ≳ damping` on joint torque/joint reaction/illegal contacts, and
  classic-vs-warp agreement on a handful of trials.
