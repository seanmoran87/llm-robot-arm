"""STAGE 3 - the skill library, called directly from Python (no LLM yet).

This is what the LLM will later do for you: decide which skills to call, in
which order, with which numbers.  Here YOU are the planner.

    python stages/03_skills.py           (add --viewer to watch it live)

PREDICT: the last call asks to place a block while holding nothing.  What
will the skill return?  Will the robot move?
"""
import argparse

from _common import attach_outputs

from robot_sim import config as C
from robot_sim.controller import ArmController
from robot_sim.scene import Sim
from robot_sim.skills import Skills

parser = argparse.ArgumentParser()
parser.add_argument("--viewer", action="store_true")
args = parser.parse_args()

sim = Sim()
finish = attach_outputs(sim, "03_skills", args.viewer)
skills = Skills(sim, ArmController(sim))

scene = skills.get_scene()
print("get_scene ->", scene, "\n")

bx, by, bz = scene["blocks"]["blue"]
print("grasp('red')  ->", skills.grasp("red"))
print("place on blue ->", skills.place(bx, by, bz + C.BLOCK_SIZE))
print("move_to       ->", skills.move_to(0.30, -0.10, 0.15))
print("home          ->", skills.home())
print("place (empty) ->", skills.place(0.20, -0.20, C.BLOCK_SIZE / 2))
finish()
sim.close()
