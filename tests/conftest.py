"""Shared test setup.  pytest runs every function named test_* in this folder."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robot_sim.controller import ArmController  # noqa: E402
from robot_sim.executor import SafeExecutor  # noqa: E402
from robot_sim.runlog import RunLog  # noqa: E402
from robot_sim.scene import Sim  # noqa: E402
from robot_sim.skills import Skills  # noqa: E402
from robot_sim.validator import Validator  # noqa: E402


@pytest.fixture
def robot(tmp_path):
    """A fresh robot (all layers below the planner) for each test."""
    sim = Sim()
    ctrl = ArmController(sim)
    skills = Skills(sim, ctrl)
    log = RunLog(sim, tmp_path, echo=False, snapshots=False)
    executor = SafeExecutor(skills, Validator(), log)
    ctrl.hold(0.3)
    yield sim, ctrl, skills, executor
    log.close()
    sim.close()
