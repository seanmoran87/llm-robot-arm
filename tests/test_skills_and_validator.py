import numpy as np

from robot_sim import config as C
from robot_sim.perception import CameraPerception
from robot_sim.validator import Validator

HALF = C.BLOCK_SIZE / 2


def test_grasp_lifts_block(robot):
    sim, ctrl, skills, _ = robot
    result = skills.grasp("red")
    assert result["success"] and skills.holding == "red"
    assert sim.block_pos("red")[2] > 0.15                 # really in the air


def test_grasp_reports_a_miss(robot):
    """If perception is wrong the fingers close on air - and the skill says so."""
    sim, ctrl, skills, _ = robot
    skills.perception = lambda: {"red": np.array([0.30, -0.15, HALF])}   # red is not there
    result = skills.grasp("red")
    assert not result["success"] and skills.holding is None


def test_stack_three_blocks(robot):
    sim, ctrl, skills, _ = robot
    bx, by, _ = sim.block_pos("blue")
    assert skills.grasp("red")["success"]
    assert skills.place(bx, by, HALF + C.BLOCK_SIZE)["success"]
    assert skills.grasp("green")["success"]
    assert skills.place(bx, by, HALF + 2 * C.BLOCK_SIZE)["success"]
    skills.home()
    ctrl.hold(1.0)
    for name, level in (("blue", 0), ("red", 1), ("green", 2)):
        pos = sim.block_pos(name)
        assert abs(pos[2] - (HALF + level * C.BLOCK_SIZE)) < 0.004
        assert np.hypot(pos[0] - bx, pos[1] - by) < 0.008
        assert sim.block_upright_error(name) < 0.05


def test_camera_perception_matches_ground_truth(robot):
    sim, *_ = robot
    seen = CameraPerception(sim).locate_blocks()
    assert set(seen) == set(sim.block_names())
    for name, pos in seen.items():
        assert np.linalg.norm(pos - sim.block_pos(name)) < 0.003     # within 3 mm


def _scene(holding=None, **blocks):
    base = {"red": [0.24, -0.12, HALF], "green": [0.30, 0.0, HALF], "blue": [0.24, 0.12, HALF]}
    base.update(blocks)
    return {"blocks": base, "holding": holding, "gripper_position": [0.22, 0.0, 0.22]}


def test_validator_rules():
    v = Validator()
    ok = lambda tool, args, scene: v.check(tool, args, scene).ok   # noqa: E731

    assert ok("grasp", {"block": "red"}, _scene())
    assert not ok("grasp", {"block": "purple"}, _scene())                    # unknown block
    assert not ok("grasp", {"block": "red"}, _scene(holding="green"))        # hands full
    assert not ok("launch_missiles", {}, _scene())                           # unknown tool
    # red sits under green -> must not be pulled out
    assert not ok("grasp", {"block": "red"}, _scene(green=[0.24, -0.12, HALF + C.BLOCK_SIZE]))

    held = _scene(holding="red")
    assert ok("place", {"x": 0.24, "y": 0.12, "z": HALF + C.BLOCK_SIZE}, held)   # on blue
    assert ok("place", {"x": 0.20, "y": -0.20, "z": HALF}, held)                 # on the table
    assert not ok("place", {"x": 0.24, "y": 0.12, "z": HALF}, held)              # inside blue
    assert not ok("place", {"x": 0.20, "y": -0.20, "z": 0.15}, held)             # mid-air
    assert not ok("place", {"x": 0.20, "y": -0.20, "z": -0.05}, held)            # under the table
    assert not ok("place", {"x": 0.60, "y": 0.0, "z": HALF}, held)               # out of reach
    assert not ok("place", {"x": 0.26, "y": 0.12, "z": HALF + C.BLOCK_SIZE}, held)  # overhang
    assert not ok("place", {"x": float("nan"), "y": 0.0, "z": HALF}, held)
    assert not ok("place", {"x": "0.2", "y": 0.0, "z": HALF}, held)              # text, not a number
    assert not ok("place", {"x": 0.2, "y": -0.2, "z": HALF}, _scene())           # holding nothing

    assert ok("move_to", {"x": 0.30, "y": 0.10, "z": 0.20}, _scene())
    assert not ok("move_to", {"x": 0.30, "y": 0.0, "z": 0.03}, _scene())         # down onto green
    assert not ok("move_to", {"x": 0.05, "y": 0.0, "z": 0.20}, _scene())         # into the base
    assert not ok("move_to", {"x": 0.30, "y": 0.0, "z": 0.60}, _scene())         # too high


def test_rejected_action_does_not_move_robot(robot):
    sim, ctrl, skills, executor = robot
    executor.start_command("test", "test")
    before = sim.tcp_pos().copy()
    t0 = sim.time
    result = executor.execute("move_to", {"x": 0.9, "y": 0.0, "z": 0.1})
    assert result["rejected"] and not result["success"]
    assert sim.time == t0 and np.allclose(sim.tcp_pos(), before)
