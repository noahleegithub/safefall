"""Unit tests for the SafeFall impact metric core (no simulator required)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from holosoma.simulator.base_simulator.hooks import HookRegistry, Phase

# Load the module directly: importing the ``holosoma.simulator.mujoco`` package would
# import MuJoCo, which is not needed for this pure metric test.
_MODULE_PATH = Path(__file__).resolve().parents[1] / "holosoma/simulator/mujoco/impact_metrics.py"
_spec = importlib.util.spec_from_file_location("_impact_metrics_test", _MODULE_PATH)
assert _spec is not None and _spec.loader is not None
_impact_metrics = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impact_metrics)
ImpactMonitor = _impact_metrics.ImpactMonitor


class _FakeBackend:
    def __init__(self, num_envs: int, nbody: int, nv: int) -> None:
        self.num_envs = num_envs
        self.nbody = nbody
        self.nv = nv
        self.contact_force = torch.zeros(0, 3)
        self.body1 = torch.zeros(0, dtype=torch.long)
        self.body2 = torch.zeros(0, dtype=torch.long)
        self.worldid = torch.zeros(0, dtype=torch.long)
        self.valid = torch.zeros(0, dtype=torch.bool)
        self.cfrc_int = torch.zeros(num_envs, nbody, 6)
        self.qfrc_constraint = torch.zeros(num_envs, nv)

    def get_contact_forces_world(self):
        return self.contact_force, self.body1, self.body2, self.worldid, self.valid

    def get_internal_wrenches(self):
        return self.cfrc_int

    def get_constraint_forces(self):
        return self.qfrc_constraint


def _make_monitor(num_envs: int = 2, nbody: int = 4, nv: int = 7):
    # body 0 world, 1 torso, 2 head, 3 foot.  Robot dof address 6 -> joint 0 -> body 1.
    root_model = SimpleNamespace(
        nbody=nbody,
        body_parentid=np.array([0, 0, 1, 1], dtype=np.int32),
        dof_jntid=np.zeros(nv, dtype=np.int32),
        jnt_bodyid=np.array([1], dtype=np.int32),
    )
    backend = _FakeBackend(num_envs, nbody, nv)
    sim = SimpleNamespace(
        backend=backend,
        root_model=root_model,
        sim_device="cpu",
        num_envs=num_envs,
        num_dof=1,
        sim_dt=0.005,
        body_ids=[1, 2, 3],
        body_names=["torso", "head", "foot"],
        dof_names=["joint0"],
        dof_qvel_addrs=[6],
        dof_pos=torch.zeros(num_envs, 1),
        hooks=HookRegistry(base_rates={Phase.FRAME_BEGIN: 50.0, Phase.FRAME_END: 50.0, Phase.POST_STEP: 200.0}),
    )
    env = SimpleNamespace(
        dof_pos_limits=torch.tensor([[-1.0, 1.0]]),
        hard_dof_pos_limits=torch.tensor([[-1.0, 1.0]]),
        torque_limits=torch.tensor([100.0]),
    )
    monitor = ImpactMonitor(env, sim, window_steps=8)
    return monitor, backend, sim


def _step(monitor, sim, *, torque, reaction, contact):
    sim.hooks.emit(Phase.FRAME_BEGIN)
    backend = sim.backend
    backend.qfrc_constraint = torch.zeros(2, 7)
    backend.qfrc_constraint[:, 6] = torch.tensor(torque, dtype=torch.float32)
    backend.cfrc_int = torch.zeros(2, 4, 6)
    backend.cfrc_int[:, 1, 5] = torch.tensor(reaction, dtype=torch.float32)
    force = torch.tensor([[0.0, 0.0, v] for v in contact], dtype=torch.float32)
    backend.contact_force = force
    backend.body1 = torch.tensor([1, 2], dtype=torch.long)
    backend.body2 = torch.tensor([0, 0], dtype=torch.long)
    backend.worldid = torch.tensor([0, 1], dtype=torch.long)
    backend.valid = torch.tensor([True, True])
    sim.hooks.emit(Phase.POST_STEP)
    sim.hooks.emit(Phase.FRAME_END)


def test_peak_metrics():
    monitor, _backend, sim = _make_monitor()
    _step(monitor, sim, torque=[10.0, 20.0], reaction=[50.0, 80.0], contact=[100.0, 200.0])
    res = monitor.results(torch.tensor([0, 1]), torch.tensor([1, 1]), torch.tensor([1, 1]))
    assert torch.allclose(res.max_joint_torque, torch.tensor([10.0, 20.0]))
    assert torch.allclose(res.max_torque_ratio, torch.tensor([0.1, 0.2]))
    assert torch.allclose(res.max_joint_reaction, torch.tensor([50.0, 80.0]))
    assert torch.allclose(res.max_contact_force, torch.tensor([100.0, 200.0]))
    assert torch.allclose(res.max_impulse, torch.tensor([0.5, 1.0]))
    assert torch.equal(res.illegal_contact, torch.tensor([False, True]))
    assert torch.equal(res.max_limit_violations, torch.tensor([0.0, 0.0]))
    assert torch.equal(monitor.first_contact_step[:2], torch.tensor([1, 1]))


def test_ring_window_and_reset():
    monitor, _backend, sim = _make_monitor()
    _step(monitor, sim, torque=[10.0, 20.0], reaction=[50.0, 80.0], contact=[100.0, 200.0])
    _step(monitor, sim, torque=[5.0, 200.0], reaction=[10.0, 20.0], contact=[50.0, 500.0])
    # Full window keeps the per-step max; the narrow window sees only step 2.
    full = monitor.results(torch.tensor([0, 1]), torch.tensor([1, 1]), torch.tensor([2, 2]))
    narrow = monitor.results(torch.tensor([0, 1]), torch.tensor([2, 2]), torch.tensor([2, 2]))
    assert torch.allclose(full.max_joint_torque, torch.tensor([10.0, 200.0]))
    assert torch.allclose(narrow.max_joint_torque, torch.tensor([5.0, 200.0]))
    assert torch.allclose(full.max_contact_force, torch.tensor([100.0, 500.0]))

    monitor.reset(torch.tensor([0]))
    assert monitor.first_contact_step[0].item() == -1
    # env 1 is untouched
    assert monitor.first_contact_step[1].item() == 1
