"""STAGE 6 - a taste of the sim-to-real gap.

In simulation the robot knows exactly where each block is.  A real camera is
off by millimetres, sometimes centimetres.  Here we add random error to the
block positions the robot is told (the blocks themselves do not move) and
count how often "stack red on blue" still works.

    python stages/06_sim_to_real.py                   (takes about a minute)
    python stages/06_sim_to_real.py --noise-mm 0 3 6 12 --trials 10

PREDICT: the blocks are 40 mm wide and the open gripper is 70 mm wide.  At
what position error (in mm) do you expect stacking to start failing?
"""
import argparse

import numpy as np
import _common  # noqa: F401

from robot_sim import config as C
from robot_sim.controller import ArmController
from robot_sim.scene import Sim
from robot_sim.skills import Skills

parser = argparse.ArgumentParser()
parser.add_argument("--noise-mm", type=float, nargs="+", default=[0, 2, 5, 10, 15, 20])
parser.add_argument("--trials", type=int, default=10)
args = parser.parse_args()


def one_trial(noise_m: float, rng) -> tuple[bool, str]:
    sim = Sim()
    ctrl = ArmController(sim)

    def noisy_perception():
        """Ground truth plus a random horizontal error (a made-up 'bad camera')."""
        seen = {}
        for name in sim.block_names():
            pos = sim.block_pos(name)
            pos[:2] += rng.normal(0.0, noise_m, size=2)
            seen[name] = pos
        return seen

    skills = Skills(sim, ctrl, noisy_perception)
    blue = noisy_perception()["blue"]
    result = skills.grasp("red")
    if not result["success"]:
        sim.close()
        return False, "grasp failed"
    skills.place(blue[0], blue[1], C.BLOCK_SIZE * 1.5)
    skills.home()
    ctrl.hold(1.0)
    # Judge success with the TRUE positions: is red really sitting on blue?
    red, blue_true = sim.block_pos("red"), sim.block_pos("blue")
    on_top = abs(red[2] - blue_true[2] - C.BLOCK_SIZE) < 0.006 and \
        np.hypot(red[0] - blue_true[0], red[1] - blue_true[1]) < C.BLOCK_SIZE / 2
    sim.close()
    return bool(on_top), "ok" if on_top else "red not on blue"


print(f"{'position error (std dev)':<28}{'stacks that worked':<22}what went wrong")
for noise_mm in args.noise_mm:
    rng = np.random.default_rng(1)
    outcomes = [one_trial(noise_mm / 1000, rng) for _ in range(args.trials)]
    wins = sum(ok for ok, _ in outcomes)
    problems = {}
    for ok, why in outcomes:
        if not ok:
            problems[why] = problems.get(why, 0) + 1
    detail = ", ".join(f"{n}x {why}" for why, n in problems.items()) or "-"
    print(f"{noise_mm:>6.0f} mm{'':<19}{wins:>2}/{args.trials:<19}{detail}")
