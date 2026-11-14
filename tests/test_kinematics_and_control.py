import math

import mujoco
import numpy as np
import pytest

from robot_sim import config as C
from robot_sim.kinematics import UnreachableError, forward_kinematics, inverse_kinematics


def test_fk_matches_mujoco_at_home(robot):
    sim, *_ = robot
    assert np.allclose(forward_kinematics(C.HOME_Q), sim.tcp_pos(), atol=1e-3)


def test_ik_matches_mujoco_across_workspace(robot):
    """Our trigonometry and the physics engine must agree everywhere."""
    sim, *_ = robot
    for r in np.linspace(C.WORKSPACE_R_MIN, C.WORKSPACE_R_MAX, 6):
        for a in np.linspace(-C.WORKSPACE_ANGLE_MAX, C.WORKSPACE_ANGLE_MAX, 7):
            for z in np.linspace(C.WORKSPACE_Z_MIN, C.WORKSPACE_Z_MAX, 5):
                x, y = r * math.cos(a), r * math.sin(a)
                q = inverse_kinematics(x, y, z)
                assert np.allclose(forward_kinematics(q), [x, y, z], atol=1e-9)
                sim.data.qpos[sim._arm_qpos] = q
                mujoco.mj_forward(sim.model, sim.data)
                assert np.allclose(sim.tcp_pos(), [x, y, z], atol=1e-6)
                rot = sim.data.site_xmat[sim._tcp_id].reshape(3, 3)
                assert rot[2, 2] == pytest.approx(-1, abs=1e-6)        # pointing down
                assert abs(rot[1, 1]) == pytest.approx(1, abs=1e-6)    # fingers square to y


def test_ik_rejects_unreachable():
    with pytest.raises(UnreachableError):
        inverse_kinematics(0.9, 0.0, 0.1)


def test_controller_reaches_targets(robot):
    sim, ctrl, *_ = robot
    for target in [(0.30, 0.15, 0.20), (0.18, -0.15, 0.10), (0.34, 0.0, 0.22)]:
        ctrl.move_tcp(*target)
        assert np.linalg.norm(sim.tcp_pos() - target) < 0.002     # within 2 mm


def test_arm_holds_still_without_gravity_compensation(robot):
    """With the feedforward term off, the PID terms alone must hold the arm."""
    sim, ctrl, *_ = robot
    ctrl.gravity_compensation = False
    ctrl.hold(2.0)
    assert np.linalg.norm(sim.tcp_pos() - forward_kinematics(C.HOME_Q)) < 0.003
