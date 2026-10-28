#!/usr/bin/env python3
"""SafeFall baseline benchmark.

Runs walking + perturbation fall trials on a trained nominal policy and evaluates
the paper's three baselines from the same trigger state:

* ``nominal``  -- the locomotion policy continues through the fall
* ``default``  -- hold the default joint targets with nominal PD gains
* ``damping``  -- Kp ~= 0, Kd = 10, constant target

The fall is induced by 1-3 of: torso push, friction slip, foot push (trip), and
actuator failure (motor-off or brake-lock).  The impact window starts at
``T - 300 ms`` (oracle trigger), where ``T`` is the first non-foot ground contact
after the perturbation.  Metrics are the SafeFall Table IV metrics implemented in
``holosoma.simulator.mujoco.impact_metrics``.

Usage (single shard, pilot)::

    python -m scripts.safefall.benchmark \
        --checkpoint /scratch/.../model_025000.pt \
        --output-dir /scratch/.../safefall_eval \
        --trials 200 --num-envs 64 --seed 0 --shard 0 --num-shards 1

See ``scripts/safefall/bench.sh`` for the Slurm wrapper.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from holosoma.config_types.plugin import CameraVizPluginConfig
from holosoma.config_types.randomization import RandomizationManagerCfg
from holosoma.config_types.simulator import MujocoWarpConfig
from holosoma.config_types.termination import TerminationManagerCfg
from holosoma.config_values.logger import disabled as disabled_logger
from holosoma.config_values.sensor import CAMERA_REGISTRY
from holosoma.config_values.simulator import mjwarp as mjwarp_preset
from holosoma.simulator.mujoco.impact_metrics import ImpactMonitor
from holosoma.utils.eval_utils import (
    CheckpointConfig,
    init_eval_logging,
    load_checkpoint,
    load_saved_experiment_config,
)
from holosoma.utils.helpers import get_class
from holosoma.utils.safe_torch_import import torch
from holosoma.utils.sim_utils import close_simulation_app, setup_simulation_environment
from loguru import logger

MECHANISMS = ("push", "slip", "trip", "freeze")
BASELINES = ("nominal", "default", "damping")

METRIC_FIELDS = (
    "max_joint_torque",
    "max_torque_ratio",
    "max_joint_reaction",
    "max_contact_force",
    "max_impulse",
    "cumulative_impulse",
    "illegal_contact",
    "max_limit_violations",
)


@dataclass
class Scenario:
    trial_id: int
    seed: int
    mechanisms: list[str]
    command: tuple[float, float, float]
    t_pert: int
    push_dv: tuple[float, float, float] | None = None
    slip_mu: float | None = None
    slip_steps: int = 0
    trip_force: float | None = None
    trip_steps: int = 0
    freeze_mode: str | None = None
    freeze_dofs: list[int] = field(default_factory=list)

    @property
    def primary(self) -> str:
        return self.mechanisms[0]


@dataclass
class PreFallResult:
    valid: torch.Tensor
    recovery: torch.Tensor
    t_step: torch.Tensor
    trigger_step: torch.Tensor
    post_end: torch.Tensor
    nominal: dict[str, torch.Tensor]
    trigger_root: torch.Tensor
    trigger_dof: torch.Tensor
    trigger_cmd: torch.Tensor
    trigger_friction: torch.Tensor
    trigger_freeze_mode: torch.Tensor
    trigger_freeze_mask: torch.Tensor
    trigger_freeze_q: torch.Tensor
    trigger_slip_remaining: torch.Tensor
    trigger_trip_remaining: torch.Tensor
    trigger_trip_force: torch.Tensor
    trigger_trip_body: torch.Tensor
    friction_baseline: torch.Tensor


class FallBenchmark:
    """Build the environment/policy once, then run batches of fall scenarios."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(args.output_dir).expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._build_config()
        self._build_env()
        self._setup_monitor()
        self._install_freeze_override()

    # ------------------------------------------------------------------ setup

    def _build_config(self) -> None:
        saved_cfg, wandb_path = load_saved_experiment_config(CheckpointConfig(checkpoint=self.args.checkpoint))
        base = saved_cfg.get_eval_config()
        render = self.args.render_trial_id >= 0
        num_envs = 1 if render else self.args.num_envs
        overrides = {
            "sensor": {},
            "plugin": {},
        }
        # Switch the backend to mjwarp but keep the checkpoint's physics/control rates.
        sim_preset = dataclasses.replace(
            mjwarp_preset,
            config=dataclasses.replace(
                mjwarp_preset.config,
                sim=base.simulator.config.sim,
                mujoco_warp=MujocoWarpConfig(nconmax_per_env=self.args.nconmax_per_env),
            ),
        )
        cfg = dataclasses.replace(base, simulator=sim_preset, logger=disabled_logger, **overrides)
        # No termination: the benchmark needs the full fall and post-impact window.
        # No domain randomization: evaluation scenarios vary the fall, not the robot
        # parameters, and this makes all baselines exactly paired.
        cfg = dataclasses.replace(
            cfg,
            termination=TerminationManagerCfg(terms={}),
            randomization=RandomizationManagerCfg(),
            training=dataclasses.replace(
                cfg.training,
                num_envs=num_envs,
                headless=True,
                seed=self.args.seed,
            ),
        )
        if render:
            cfg = dataclasses.replace(cfg, sensor={"cam": CAMERA_REGISTRY[self.args.render_camera]})
        else:
            # No camera rendering in the batch benchmark.
            cfg = dataclasses.replace(cfg, sensor={})
        self.config = cfg
        self.saved_wandb_path = wandb_path
        (self.output_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "checkpoint": self.args.checkpoint,
                    "wandb_path": wandb_path,
                    "args": vars(self.args),
                    "git_commit": _git_commit(),
                    "sif": _env("SIF"),
                },
                indent=2,
                default=str,
            )
        )

    def _build_env(self) -> None:
        self.env, self.device, self.simulation_app = setup_simulation_environment(self.config, device=self.device)
        self.sim = self.env.simulator
        self.control_hz = 1.0 / float(self.env.dt)
        self.ndof = int(self.env.num_dof)
        self.nbody = int(self.sim.root_model.nbody)
        self.num_envs = int(self.env.num_envs)

        algo_cls = get_class(self.config.algo._target_)
        if "ppo" not in str(self.config.algo._target_).lower():
            raise RuntimeError(
                f"This benchmark currently supports PPO checkpoints, got {self.config.algo._target_}. "
                "The SafeFall paper uses PPO."
            )
        algo = algo_cls(
            device=self.device,
            env=self.env,
            config=self.config.algo.config,
            log_dir=str(self.output_dir / "algo"),
            multi_gpu_cfg=None,
        )
        algo.setup()
        ckpt_path = load_checkpoint(self.args.checkpoint, str(self.output_dir / "checkpoint"))
        algo.load(str(ckpt_path))
        algo._eval_mode()
        algo.eval_policy = algo.get_inference_policy()
        self.algo = algo

        self.env.set_is_evaluating()
        self._zeros_actions = torch.zeros(self.num_envs, self.ndof, device=self.device)

    def _setup_monitor(self) -> None:

        model = self.sim.root_model
        foot_body_ids = [
            int(self.sim.body_ids[i])
            for i, name in enumerate(self.sim.body_names)
            if "foot" in name.lower() or "ankle" in name.lower()
        ]
        if not foot_body_ids:
            raise RuntimeError("No foot/ankle bodies found; cannot identify ground impact bodies.")
        self.friction_geom_ids = [g for g in range(int(model.ngeom)) if int(model.geom_bodyid[g]) in set(foot_body_ids)]
        if not self.friction_geom_ids:
            raise RuntimeError("No geoms attached to foot/ankle bodies; cannot model friction slip.")
        # MuJoCo combines contact friction with ``max(friction1, friction2)``, so lowering only
        # the foot is ineffective when the ground is grippier.  Include every non-robot geom
        # (world/terrain) in the slip write; this simulates a slippery patch under the robot.
        robot_bodies = {int(b) for b in self.sim.body_ids}
        ground_geom_ids = [g for g in range(int(model.ngeom)) if int(model.geom_bodyid[g]) not in robot_bodies]
        self.friction_geom_ids = list(dict.fromkeys(self.friction_geom_ids + ground_geom_ids))
        self.foot_body_ids = foot_body_ids
        self.foot_holosoma_ids = [
            i for i, name in enumerate(self.sim.body_names) if "foot" in name.lower() or "ankle" in name.lower()
        ]

        window = max(48, self.args.post_steps + 8)
        self.base_friction = self.sim.get_geom_friction(
            self.friction_geom_ids, torch.arange(self.num_envs, device=self.device)
        )
        self.monitor = ImpactMonitor(self.env, self.sim, window_steps=window)
        self.monitor.reset()

    def _install_freeze_override(self) -> None:
        self.freeze_mode = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.freeze_mask = torch.zeros(self.num_envs, self.ndof, dtype=torch.bool, device=self.device)
        self.freeze_q = torch.zeros(self.num_envs, self.ndof, device=self.device)
        self._kp_lock = float(self.args.brake_kp)
        self._kd_lock = float(self.args.brake_kd)

        term = self._joint_term()
        original = self.env._apply_force_in_physics_step

        def patched() -> None:
            original()
            if not bool(self.freeze_mode.any()):
                return
            q = self.sim.dof_pos[:]
            qd = self.sim.dof_vel[:]
            tau = term.torques
            motor = self.freeze_mode == 1
            brake = self.freeze_mode == 2
            if bool(motor.any()):
                tau[motor] = torch.where(self.freeze_mask[motor], torch.zeros_like(tau[motor]), tau[motor])
            if bool(brake.any()):
                desired = self._kp_lock * (self.freeze_q - q) - self._kd_lock * qd
                tau[brake] = torch.where(self.freeze_mask[brake], desired[brake], tau[brake])
            tau.clamp_(-self.env.torque_limits, self.env.torque_limits)
            self.sim.apply_torques_at_dof(tau)

        self.env._apply_force_in_physics_step = patched

    def _joint_term(self):
        for _, term in self.env.action_manager.iter_terms():
            if hasattr(term, "p_gains") and hasattr(term, "torques"):
                return term
        raise RuntimeError("No joint_control term with PD gains found.")

    # ------------------------------------------------------------------ scenarios

    def make_scenario(self, trial_id: int) -> Scenario:
        args = self.args
        rng = np.random.default_rng(args.seed * 100003 + trial_id * 7919 + 17)
        primary = MECHANISMS[trial_id % len(MECHANISMS)]
        n_factors = int(rng.choice([1, 2, 3], p=[0.6, 0.3, 0.1]))
        mechanisms = [primary]
        remaining = [m for m in MECHANISMS if m != primary]
        if n_factors > 1:
            k = min(n_factors - 1, len(remaining))
            mechanisms.extend(rng.choice(remaining, size=k, replace=False).tolist())

        command = (
            float(rng.uniform(0.5, 1.5)),
            float(rng.uniform(-0.4, 0.4)),
            float(rng.uniform(-0.5, 0.5)),
        )
        warmup_s = float(rng.uniform(args.warmup_min, args.warmup_max))
        t_pert = round(warmup_s * self.control_hz)
        t_pert = max(1, min(t_pert, args.max_pre_steps - 60))

        scenario = Scenario(
            trial_id=trial_id,
            seed=int(args.seed + trial_id),
            mechanisms=mechanisms,
            command=command,
            t_pert=t_pert,
        )
        if "push" in mechanisms:
            angle = float(rng.uniform(0.0, 2.0 * math.pi))
            mag = float(rng.uniform(args.push_min, args.push_max))
            scenario.push_dv = (mag * math.cos(angle), mag * math.sin(angle), 0.0)
        if "slip" in mechanisms:
            scenario.slip_mu = float(rng.uniform(args.slip_min, args.slip_max))
            scenario.slip_steps = max(
                1, round(rng.uniform(args.slip_duration_min, args.slip_duration_max) * self.control_hz)
            )
        if "trip" in mechanisms:
            scenario.trip_force = float(rng.uniform(args.trip_force_min, args.trip_force_max))
            scenario.trip_steps = max(1, round(args.trip_duration * self.control_hz))
        if "freeze" in mechanisms:
            scenario.freeze_mode = str(rng.choice(["motor_off", "brake_lock"]))
            leg_idx = [
                i for i, n in enumerate(self.sim.dof_names) if any(k in n.lower() for k in ("hip", "knee", "ankle"))
            ]
            arm_idx = [
                i
                for i, n in enumerate(self.sim.dof_names)
                if any(k in n.lower() for k in ("shoulder", "elbow", "wrist"))
            ]
            n_joints = int(rng.choice([1, 2], p=[0.7, 0.3]))
            pool = leg_idx if (not arm_idx or rng.random() < 0.9) else arm_idx
            scenario.freeze_dofs = sorted(rng.choice(pool, size=min(n_joints, len(pool)), replace=False).tolist())
        return scenario

    # ------------------------------------------------------------------ rollout

    def _reset(self, commands: torch.Tensor) -> dict:
        # Start every batch from a clean model/controller state.
        self.freeze_mode.zero_()
        self.freeze_mask.zero_()
        self._clear_all_applied()
        self.sim.set_geom_friction(
            self.friction_geom_ids, self.base_friction, torch.arange(self.num_envs, device=self.device)
        )
        term = self._joint_term()
        term._kp_scale[:] = 1.0
        term._kd_scale[:] = 1.0

        self.env.reset_all()
        self.env.command_manager.commands[: commands.shape[0]] = commands.to(self.device)
        self.env.observation_manager.reset()
        self.env._compute_observations()
        self.env._clip_observations()
        self.monitor.reset()
        return self.env.obs_buf_dict

    def _clear_all_applied(self) -> None:
        af = self.sim.applied_forces
        if hasattr(af, "zero_"):
            af.zero_()
        else:
            af.fill(0.0)

    def _actor_obs(self, obs: dict) -> torch.Tensor:
        return torch.cat([obs[k] for k in self.algo.actor_obs_keys], dim=1)

    def _policy_actions(self, obs: dict) -> torch.Tensor:
        return self.algo.eval_policy({"actor_obs": self._actor_obs(obs)})

    def _dof_state_3d(self) -> torch.Tensor:
        ds = self.sim.dof_state
        shape = tuple(ds.shape)
        arr = ds[:] if len(shape) == 2 else ds
        if len(shape) == 2 and shape[0] == self.num_envs * self.ndof:
            return arr.view(self.num_envs, self.ndof, 2)
        return arr.view(1, self.ndof, 2)

    def _set_applied(self, env_ids: torch.Tensor, body_ids: torch.Tensor, forces: torch.Tensor) -> None:
        af = self.sim.applied_forces
        if af.ndim == 3:
            af[env_ids, body_ids, 0:3] = forces
        else:
            # Classic single-env backend (numpy view).
            af[body_ids[0].item(), 0:3] = forces[0].detach().cpu().numpy()

    def _clear_applied(self, env_ids: torch.Tensor, body_ids: torch.Tensor) -> None:
        zero = torch.zeros(len(env_ids), 3, device=self.device)
        self._set_applied(env_ids, body_ids, zero)

    def _apply_factors(
        self,
        scenarios: list[Scenario],
        env_ids: torch.Tensor,
        *,
        slip_end: torch.Tensor,
        slip_mu: torch.Tensor,
        slip_baseline: torch.Tensor,
        trip_end: torch.Tensor,
        trip_force: torch.Tensor,
        trip_body: torch.Tensor,
    ) -> None:
        """Apply the selected factors to ``env_ids`` (parallel to ``scenarios``)."""
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        # push
        push_local = [j for j, s in enumerate(scenarios) if s.push_dv is not None]
        if push_local:
            idx = env_ids[push_local]
            root = self.sim.robot_root_states.clone()
            dv = torch.tensor([scenarios[j].push_dv for j in push_local], device=self.device, dtype=root.dtype)
            root[idx, 7:10] += dv
            self.sim.set_actor_root_state_tensor_robots(idx, root[idx])

        # friction slip
        slip_local = [j for j, s in enumerate(scenarios) if s.slip_mu is not None]
        if slip_local:
            idx = env_ids[slip_local]
            base = self.sim.get_geom_friction(self.friction_geom_ids, idx)
            slip_baseline[idx] = base
            mu = torch.tensor([scenarios[j].slip_mu for j in slip_local], device=self.device, dtype=base.dtype)
            values = base.clone()
            values[:] = mu.unsqueeze(1)
            self.sim.set_geom_friction(self.friction_geom_ids, values, idx)
            steps = torch.tensor([scenarios[j].slip_steps for j in slip_local], device=self.device, dtype=torch.long)
            t_pert = torch.tensor([scenarios[j].t_pert for j in slip_local], device=self.device, dtype=torch.long)
            slip_end[idx] = t_pert + steps
            slip_mu[idx] = mu

        # trip: push the swing foot (lower normal force)
        trip_local = [j for j, s in enumerate(scenarios) if s.trip_force is not None]
        if trip_local:
            idx = env_ids[trip_local]
            foot_holo = torch.as_tensor(self.foot_holosoma_ids, device=self.device, dtype=torch.long)
            foot_raw = torch.as_tensor(self.foot_body_ids, device=self.device, dtype=torch.long)
            normal = self.sim.contact_forces[:, foot_holo, 2]
            swing_local = normal[idx].argmin(dim=1)
            body = foot_raw[swing_local]
            vel = self.sim._rigid_body_vel[idx][:, foot_holo[swing_local], :2]
            speed = torch.linalg.vector_norm(vel, dim=-1, keepdim=True)
            direction = -vel / speed.clamp_min(1e-6)
            random_dir = torch.nn.functional.normalize(torch.randn(len(idx), 2, device=self.device), dim=-1)
            direction = torch.where(speed > 0.1, direction, random_dir)
            mag = torch.tensor([scenarios[j].trip_force for j in trip_local], device=self.device, dtype=torch.float32)
            forces = torch.cat([direction * mag.unsqueeze(1), (0.15 * mag).unsqueeze(1)], dim=1)
            self._set_applied(idx, body, forces)
            trip_body[idx] = body
            trip_force[idx] = forces
            steps = torch.tensor([scenarios[j].trip_steps for j in trip_local], device=self.device, dtype=torch.long)
            t_pert = torch.tensor([scenarios[j].t_pert for j in trip_local], device=self.device, dtype=torch.long)
            trip_end[idx] = t_pert + steps

        # actuator failure
        for j, s in enumerate(scenarios):
            if s.freeze_mode is None:
                continue
            i = int(env_ids[j].item())
            self.freeze_mode[i] = 1 if s.freeze_mode == "motor_off" else 2
            self.freeze_mask[i] = False
            if s.freeze_dofs:
                self.freeze_mask[i, s.freeze_dofs] = True
            self.freeze_q[i] = self.sim.dof_pos[i].clone()

        # First non-foot ground contact is now meaningful (ignore walking contacts).
        self.monitor.clear_trial_start(env_ids)

    def run_prefall(self, scenarios: list[Scenario], *, nominal_metrics: bool = True) -> PreFallResult:
        n = len(scenarios)
        args = self.args
        N = self.num_envs
        k_state = max(32, args.oracle_lead_steps + 4)

        commands = torch.tensor([s.command for s in scenarios], device=self.device, dtype=torch.float32)
        obs = self._reset(commands)

        t_pert = torch.tensor([s.t_pert for s in scenarios], device=self.device, dtype=torch.long)
        phase = torch.zeros(n, dtype=torch.long, device=self.device)  # 0=pre,1=falling,2=post,3=done,4=recovery
        t_step = torch.full((n,), -1, dtype=torch.long, device=self.device)
        trigger_step = torch.full((n,), -1, dtype=torch.long, device=self.device)
        post_end = torch.full((n,), -1, dtype=torch.long, device=self.device)
        valid = torch.zeros(n, dtype=torch.bool, device=self.device)
        recovery = torch.zeros(n, dtype=torch.bool, device=self.device)
        done = torch.zeros(n, dtype=torch.bool, device=self.device)

        trigger_root = torch.zeros(n, 13, device=self.device)
        trigger_dof = torch.zeros(n, self.ndof, 2, device=self.device)
        trigger_cmd = commands.clone()
        trigger_friction = torch.zeros(n, len(self.friction_geom_ids), device=self.device)
        trigger_freeze_mode = torch.zeros(n, dtype=torch.long, device=self.device)
        trigger_freeze_mask = torch.zeros(n, self.ndof, dtype=torch.bool, device=self.device)
        trigger_freeze_q = torch.zeros(n, self.ndof, device=self.device)
        trigger_slip_remaining = torch.zeros(n, dtype=torch.long, device=self.device)
        trigger_trip_remaining = torch.zeros(n, dtype=torch.long, device=self.device)
        trigger_trip_force = torch.zeros(n, 3, device=self.device)
        trigger_trip_body = torch.zeros(n, dtype=torch.long, device=self.device)

        slip_end = torch.full((n,), -1, dtype=torch.long, device=self.device)
        slip_mu = torch.zeros(n, device=self.device)
        slip_baseline = torch.zeros(n, len(self.friction_geom_ids), device=self.device)
        trip_end = torch.full((n,), -1, dtype=torch.long, device=self.device)
        trip_force = torch.zeros(n, 3, device=self.device)
        trip_body = torch.zeros(n, dtype=torch.long, device=self.device)

        state_root = torch.zeros(k_state, N, 13, device=self.device)
        state_dof = torch.zeros(k_state, N, self.ndof, 2, device=self.device)
        state_root[0] = self.sim.robot_root_states.clone()
        state_dof[0] = self._dof_state_3d().clone()
        slip_baseline[:] = self.sim.get_geom_friction(self.friction_geom_ids, torch.arange(n, device=self.device))

        nominal: dict[str, torch.Tensor] = {key: torch.zeros(n, device=self.device) for key in METRIC_FIELDS}

        max_steps = args.max_pre_steps + args.post_steps
        s = 1
        while s <= max_steps:
            state_root[s % k_state] = self.sim.robot_root_states.clone()
            state_dof[s % k_state] = self._dof_state_3d().clone()

            # Start perturbations / restore slip and trip.
            start_ids = ((phase == 0) & (s == t_pert)).nonzero(as_tuple=False).flatten()
            if start_ids.numel() > 0:
                self._apply_factors(
                    [scenarios[i] for i in start_ids.tolist()],
                    start_ids,
                    slip_end=slip_end,
                    slip_mu=slip_mu,
                    slip_baseline=slip_baseline,
                    trip_end=trip_end,
                    trip_force=trip_force,
                    trip_body=trip_body,
                )
                phase[start_ids] = 1

            restore_slip = (slip_end == s).nonzero(as_tuple=False).flatten()
            if restore_slip.numel() > 0:
                self.sim.set_geom_friction(self.friction_geom_ids, slip_baseline[restore_slip], restore_slip)
                # Keep slip_end: a trigger detected at T > s still needs to know
                # whether slip was active at its (earlier) trigger step.
            clear_trip = (trip_end == s).nonzero(as_tuple=False).flatten()
            if clear_trip.numel() > 0:
                self._clear_applied(clear_trip, trip_body[clear_trip])
                # Keep trip_end/trip_force/trip_body for the same reason.

            actions = self._policy_actions(obs)
            obs, _, _, _ = self.env.step({"actions": actions, "obs": obs})

            # Impact detection: T = first non-foot ground contact after perturbation.
            newly = (
                ((phase == 1) & (t_step < 0) & (self.monitor.first_contact_step[:n] >= 0))
                .nonzero(as_tuple=False)
                .flatten()
            )
            if newly.numel() > 0:
                tvals = self.monitor.first_contact_step[newly]
                trig = torch.maximum(t_pert[newly] + 1, tvals - self.args.oracle_lead_steps)
                trig = torch.minimum(trig, torch.full_like(trig, s))
                for j, i in enumerate(newly.tolist()):
                    ts = int(trig[j].item())
                    t_step[i] = tvals[j]
                    trigger_step[i] = ts
                    post_end[i] = ts + self.args.post_steps - 1
                    valid[i] = True
                    idx = ts % k_state
                    trigger_root[i] = state_root[idx, i]
                    trigger_dof[i] = state_dof[idx, i]
                    trigger_freeze_mode[i] = self.freeze_mode[i]
                    trigger_freeze_mask[i] = self.freeze_mask[i]
                    trigger_freeze_q[i] = self.freeze_q[i]
                    slip_end_i = int(slip_end[i].item())
                    trip_end_i = int(trip_end[i].item())
                    trigger_slip_remaining[i] = max(0, slip_end_i - ts) if slip_end_i >= 0 else 0
                    trigger_trip_remaining[i] = max(0, trip_end_i - ts) if trip_end_i >= 0 else 0
                    trigger_trip_force[i] = trip_force[i]
                    trigger_trip_body[i] = trip_body[i]
                    if slip_end_i >= 0 and ts < slip_end_i:
                        trigger_friction[i] = slip_mu[i]
                    else:
                        trigger_friction[i] = slip_baseline[i]
                phase[newly] = 2

            # Snapshot completed nominal windows.
            completed = ((phase == 2) & (s >= post_end) & ~done).nonzero(as_tuple=False).flatten()
            if completed.numel() > 0:
                if nominal_metrics:
                    res = self.monitor.results(completed, trigger_step[completed], post_end[completed])
                    for key in METRIC_FIELDS:
                        nominal[key][completed] = getattr(res, key).to(torch.float32)
                done[completed] = True
                phase[completed] = 3

            # Give up on scenarios that never fell.
            lost = ((phase == 0) | (phase == 1)) & (s >= self.args.max_pre_steps)
            if lost.any():
                recovery[lost] = True
                phase[lost] = 4

            if bool(((phase == 3) | (phase == 4)).all()):
                break
            s += 1

        return PreFallResult(
            valid=valid,
            recovery=recovery,
            t_step=t_step,
            trigger_step=trigger_step,
            post_end=post_end,
            nominal=nominal,
            trigger_root=trigger_root,
            trigger_dof=trigger_dof,
            trigger_cmd=trigger_cmd,
            trigger_friction=trigger_friction,
            trigger_freeze_mode=trigger_freeze_mode,
            trigger_freeze_mask=trigger_freeze_mask,
            trigger_freeze_q=trigger_freeze_q,
            trigger_slip_remaining=trigger_slip_remaining,
            trigger_trip_remaining=trigger_trip_remaining,
            trigger_trip_force=trigger_trip_force,
            trigger_trip_body=trigger_trip_body,
            friction_baseline=slip_baseline,
        )

    def _prepare_branch(self, pre: PreFallResult, baseline: str) -> torch.Tensor:
        env_ids = pre.valid.nonzero(as_tuple=False).flatten()
        if env_ids.numel() == 0:
            return env_ids
        n = len(pre.valid)
        all_ids = torch.arange(n, device=self.device)
        # Reset any residual perturbation/failure state before the env's internal
        # reset step, so it cannot leak into the branch.
        term = self._joint_term()
        self.freeze_mode.zero_()
        self.freeze_mask.zero_()
        self._clear_applied(all_ids, pre.trigger_trip_body)
        self.sim.set_geom_friction(self.friction_geom_ids, pre.friction_baseline, all_ids)
        term._kp_scale[:] = 1.0
        term._kd_scale[:] = 1.0

        self.env.reset_all()
        self.env.command_manager.commands[env_ids] = pre.trigger_cmd[env_ids]
        self.sim.set_actor_root_state_tensor_robots(env_ids, pre.trigger_root[env_ids])
        dof = self._dof_state_3d()
        dof[env_ids] = pre.trigger_dof[env_ids]
        self.sim.set_dof_state_tensor_robots(env_ids, dof.reshape(-1, 2))
        self.env.action_manager.reset(env_ids)
        self.env.observation_manager.reset(env_ids)

        # Restore friction at the trigger.
        self.sim.set_geom_friction(self.friction_geom_ids, pre.trigger_friction[env_ids], env_ids)
        # Restore actuator-failure state.
        self.freeze_mode[env_ids] = pre.trigger_freeze_mode[env_ids]
        self.freeze_mask[env_ids] = pre.trigger_freeze_mask[env_ids]
        self.freeze_q[env_ids] = pre.trigger_freeze_q[env_ids]
        # Restore any residual trip force.
        residual = pre.trigger_trip_remaining[env_ids] > 0
        if residual.any():
            rid = env_ids[residual]
            self._set_applied(rid, pre.trigger_trip_body[rid], pre.trigger_trip_force[rid])
        # Controller gains.
        if baseline == "damping":
            term._kp_scale[env_ids] = (1e-5 / term.p_gains.clamp_min(1e-6)).unsqueeze(0).expand(len(env_ids), -1)
            term._kd_scale[env_ids] = (10.0 / term.d_gains.clamp_min(1e-6)).unsqueeze(0).expand(len(env_ids), -1)
        else:
            term._kp_scale[env_ids] = 1.0
            term._kd_scale[env_ids] = 1.0
        self.monitor.reset()
        return env_ids

    def run_branch(self, pre: PreFallResult, baseline: str) -> dict[str, torch.Tensor]:
        args = self.args
        env_ids = self._prepare_branch(pre, baseline)
        n = len(pre.valid)
        out: dict[str, torch.Tensor] = {key: torch.zeros(n, device=self.device) for key in METRIC_FIELDS}
        if env_ids.numel() == 0:
            return out

        slip_left = pre.trigger_slip_remaining[env_ids].clone()
        trip_left = pre.trigger_trip_remaining[env_ids].clone()
        baseline_fric = pre.friction_baseline[env_ids]

        for s in range(1, args.post_steps + 1):
            restore = (slip_left == s - 1) & (slip_left > 0)
            if restore.any():
                rid = env_ids[restore]
                self.sim.set_geom_friction(self.friction_geom_ids, baseline_fric[restore], rid)
            clear = (trip_left == s - 1) & (trip_left > 0)
            if clear.any():
                cid = env_ids[clear]
                self._clear_applied(cid, pre.trigger_trip_body[cid])
            self.env.step({"actions": self._zeros_actions})

        res = self.monitor.results(env_ids, torch.ones_like(env_ids), torch.full_like(env_ids, args.post_steps))
        for key in METRIC_FIELDS:
            value = getattr(res, key).to(torch.float32)
            out[key][env_ids] = value

        # Cleanup so the next baseline starts from a nominal model.
        self.sim.set_geom_friction(self.friction_geom_ids, pre.friction_baseline[env_ids], env_ids)
        self._clear_applied(env_ids, pre.trigger_trip_body[env_ids])
        self.freeze_mode[env_ids] = 0
        term = self._joint_term()
        term._kp_scale[env_ids] = 1.0
        term._kd_scale[env_ids] = 1.0
        return out

    # ------------------------------------------------------------------ driver

    def render_trial(self) -> None:
        """Replay one scenario with the camera plugin and save an mp4.

        Two passes: first find ``T``/trigger (without rendering), then replay the
        full scenario from the start with the selected controller switch at the
        trigger.  The replay is deterministic (no domain randomization, fixed seed).
        """
        args = self.args
        if self.num_envs != 1:
            raise RuntimeError("--render-trial-id requires --num-envs 1")
        scenario = self.make_scenario(args.render_trial_id)
        logger.info(f"Render trial {scenario.trial_id}: {scenario.mechanisms} {args.render_baseline}")
        pre = self.run_prefall([scenario], nominal_metrics=False)
        if not bool(pre.valid[0]):
            raise RuntimeError(f"Trial {scenario.trial_id} did not fall; nothing to render.")

        # Clear anything left by the pre-fall pass.
        ids = torch.tensor([0], device=self.device, dtype=torch.long)
        term = self._joint_term()
        self.freeze_mode.zero_()
        self.freeze_mask.zero_()
        self._clear_applied(ids, pre.trigger_trip_body)
        self.sim.set_geom_friction(self.friction_geom_ids, pre.friction_baseline, ids)
        term._kp_scale[:] = 1.0
        term._kd_scale[:] = 1.0

        obs = self._reset(torch.tensor([scenario.command], device=self.device))
        self.monitor.reset()

        out_dir = Path(args.render_dir).expanduser() if args.render_dir else (self.output_dir / "videos")
        out_dir.mkdir(parents=True, exist_ok=True)
        video_cfg = CameraVizPluginConfig(
            record_video=True,
            save_dir=str(out_dir),
            env_ids=[0],
            cameras=["cam"],
            playback_rate=float(args.render_playback_rate),
        )
        video_cfg.get_cls()(video_cfg, self.sim)

        slip_end = torch.full((1,), -1, dtype=torch.long, device=self.device)
        slip_mu = torch.zeros(1, device=self.device)
        slip_baseline = self.sim.get_geom_friction(self.friction_geom_ids, ids)
        trip_end = torch.full((1,), -1, dtype=torch.long, device=self.device)
        trip_force = torch.zeros(1, 3, device=self.device)
        trip_body = torch.zeros(1, dtype=torch.long, device=self.device)

        trigger = int(pre.trigger_step[0])
        end = int(pre.post_end[0])
        switched = False
        for s in range(1, end + 1):
            if s == scenario.t_pert:
                self._apply_factors(
                    [scenario],
                    ids,
                    slip_end=slip_end,
                    slip_mu=slip_mu,
                    slip_baseline=slip_baseline,
                    trip_end=trip_end,
                    trip_force=trip_force,
                    trip_body=trip_body,
                )
            if slip_end[0] >= 0 and s == int(slip_end[0].item()):
                self.sim.set_geom_friction(self.friction_geom_ids, slip_baseline, ids)
                slip_end[0] = -1
            if trip_end[0] >= 0 and s == int(trip_end[0].item()):
                self._clear_applied(ids, trip_body)
                trip_end[0] = -1

            if not switched and s >= trigger and args.render_baseline != "nominal":
                if args.render_baseline == "damping":
                    term._kp_scale[0] = 1e-5 / term.p_gains.clamp_min(1e-6)
                    term._kd_scale[0] = 10.0 / term.d_gains.clamp_min(1e-6)
                else:
                    term._kp_scale[0] = 1.0
                    term._kd_scale[0] = 1.0
                switched = True

            if args.render_baseline == "nominal" or not switched:
                actions = self._policy_actions(obs)
            else:
                actions = self._zeros_actions
            obs, _, _, _ = self.env.step({"actions": actions, "obs": obs})

        self.env.simulator.close()  # CLOSE hook encodes the mp4
        # Rename the timestamped file to a stable, report-friendly name.
        videos = sorted(out_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
        if videos:
            target = out_dir / f"trial_{scenario.trial_id}_{args.render_baseline}.mp4"
            videos[-1].replace(target)
            logger.info(f"Wrote {target}")
        else:
            logger.warning("Camera plugin produced no video (check camera/sensor config).")

    def run(self) -> None:
        args = self.args
        trial_ids = list(range(args.shard, args.trials, args.num_shards))
        if args.limit:
            trial_ids = trial_ids[: args.limit]
        logger.info(f"Shard {args.shard}/{args.num_shards}: {len(trial_ids)} trials")

        csv_path = self.output_dir / f"trials_shard_{args.shard}.csv"
        jsonl_path = self.output_dir / f"scenarios_shard_{args.shard}.jsonl"
        write_header = not csv_path.exists()
        with csv_path.open("a", newline="") as f_csv, jsonl_path.open("a") as f_json:
            writer = csv.DictWriter(f_csv, fieldnames=self._csv_fields())
            if write_header:
                writer.writeheader()

            b = 0
            while b < len(trial_ids):
                batch_ids = trial_ids[b : b + self.num_envs]
                scenarios = [self.make_scenario(tid) for tid in batch_ids]
                logger.info(f"Batch {b // self.num_envs}: trials {batch_ids[0]}..{batch_ids[-1]}")
                pre = self.run_prefall(scenarios)
                branches = {
                    "default": self.run_branch(pre, "default"),
                    "damping": self.run_branch(pre, "damping"),
                }

                for i, scenario in enumerate(scenarios):
                    row_meta = self._scenario_meta(scenario, pre, i)
                    f_json.write(json.dumps(row_meta) + "\n")
                    if not bool(pre.valid[i]):
                        continue
                    for baseline in BASELINES:
                        values = pre.nominal if baseline == "nominal" else branches[baseline]
                        row = {
                            "trial_id": scenario.trial_id,
                            "baseline": baseline,
                            "mechanisms": "+".join(scenario.mechanisms),
                            "primary": scenario.primary,
                            "command_vx": scenario.command[0],
                            "command_vy": scenario.command[1],
                            "command_yaw": scenario.command[2],
                            "t_pert_step": scenario.t_pert,
                            "t_step": int(pre.t_step[i]),
                            "trigger_step": int(pre.trigger_step[i]),
                            "post_end": int(pre.post_end[i]),
                            "push_dv_x": scenario.push_dv[0] if scenario.push_dv else 0.0,
                            "push_dv_y": scenario.push_dv[1] if scenario.push_dv else 0.0,
                            "slip_mu": scenario.slip_mu or 0.0,
                            "trip_force": scenario.trip_force or 0.0,
                            "freeze_mode": scenario.freeze_mode or "",
                            "freeze_dofs": ";".join(map(str, scenario.freeze_dofs)),
                        }
                        for key in METRIC_FIELDS:
                            row[key] = float(values[key][i].item())
                        writer.writerow(row)
                f_csv.flush()
                b += len(batch_ids)

        logger.info(f"Wrote {csv_path} and {jsonl_path}")

    @staticmethod
    def _csv_fields() -> list[str]:
        return [
            "trial_id",
            "baseline",
            "mechanisms",
            "primary",
            "command_vx",
            "command_vy",
            "command_yaw",
            "t_pert_step",
            "t_step",
            "trigger_step",
            "post_end",
            "push_dv_x",
            "push_dv_y",
            "slip_mu",
            "trip_force",
            "freeze_mode",
            "freeze_dofs",
            *METRIC_FIELDS,
        ]

    def _scenario_meta(self, scenario: Scenario, pre: PreFallResult, i: int) -> dict:
        return {
            "trial_id": scenario.trial_id,
            "seed": scenario.seed,
            "mechanisms": scenario.mechanisms,
            "primary": scenario.primary,
            "command": scenario.command,
            "t_pert_step": scenario.t_pert,
            "push_dv": scenario.push_dv,
            "slip_mu": scenario.slip_mu,
            "slip_steps": scenario.slip_steps,
            "trip_force": scenario.trip_force,
            "trip_steps": scenario.trip_steps,
            "freeze_mode": scenario.freeze_mode,
            "freeze_dofs": scenario.freeze_dofs,
            "t_step": int(pre.t_step[i]),
            "trigger_step": int(pre.trigger_step[i]),
            "post_end": int(pre.post_end[i]),
            "valid": bool(pre.valid[i]),
            "recovery": bool(pre.recovery[i]),
        }

    def close(self) -> None:
        try:
            self.env.simulator.close()
        finally:
            if self.simulation_app:
                close_simulation_app(self.simulation_app)


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def _env(name: str) -> str | None:
    return os.environ.get(name)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SafeFall baseline fall benchmark")
    p.add_argument("--checkpoint", required=True, help="Local .pt or wandb://... checkpoint of the nominal policy")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--trials", type=int, default=200, help="Total trials across all shards")
    p.add_argument("--limit", type=int, default=0, help="Optional cap on this shard's trials (for debugging)")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--nconmax-per-env", type=int, default=192, help="mjwarp contact capacity per env")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)

    p.add_argument("--warmup-min", type=float, default=1.0, help="Walking warmup before perturbation (s)")
    p.add_argument("--warmup-max", type=float, default=2.0)
    p.add_argument("--max-pre-steps", type=int, default=150, help="Max control steps before giving up (3 s at 50 Hz)")
    p.add_argument("--post-steps", type=int, default=40, help="Post-trigger window (40 = 0.8 s at 50 Hz)")
    p.add_argument("--oracle-lead-steps", type=int, default=15, help="Oracle trigger lead (15 = 300 ms at 50 Hz)")

    p.add_argument("--push-min", type=float, default=1.5)
    p.add_argument("--push-max", type=float, default=3.5)
    p.add_argument("--slip-min", type=float, default=0.02)
    p.add_argument("--slip-max", type=float, default=0.15)
    p.add_argument("--slip-duration-min", type=float, default=0.10)
    p.add_argument("--slip-duration-max", type=float, default=0.30)
    p.add_argument("--trip-force-min", type=float, default=150.0)
    p.add_argument("--trip-force-max", type=float, default=400.0)
    p.add_argument("--trip-duration", type=float, default=0.05)
    p.add_argument("--brake-kp", type=float, default=100.0)
    p.add_argument("--brake-kd", type=float, default=10.0)

    p.add_argument("--render-trial-id", type=int, default=-1, help="Render one trial instead of running the benchmark")
    p.add_argument("--render-baseline", choices=("nominal", "default", "damping"), default="nominal")
    p.add_argument("--render-camera", default="g1-chase", help="CAMERA_REGISTRY preset for the render camera")
    p.add_argument("--render-dir", default=None, help="Video output dir (default: <output-dir>/videos)")
    p.add_argument("--render-playback-rate", type=float, default=0.25)

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    init_eval_logging()
    args = parse_args(argv)
    bench = FallBenchmark(args)
    try:
        if args.render_trial_id >= 0:
            bench.render_trial()
        else:
            bench.run()
    finally:
        bench.close()


if __name__ == "__main__":
    main()
