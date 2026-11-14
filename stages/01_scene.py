"""STAGE 1 - the physics scene on its own.

Nothing here is controlling the robot.  We build the world, send the motors
ZERO torque, and let physics run for two seconds.

PREDICT before you run: what will the arm do with its motors switched off?

    python stages/01_scene.py            (add --viewer to watch it live)
"""
import argparse

from _common import attach_outputs

from robot_sim import config as C
from robot_sim.scene import Sim

parser = argparse.ArgumentParser()
parser.add_argument("--viewer", action="store_true")
args = parser.parse_args()

sim = Sim()
finish = attach_outputs(sim, "01_scene", args.viewer)

print("Model summary")
print(f"  bodies: {sim.model.nbody}   joints: {sim.model.njnt}   motors: {sim.model.nu}")
print(f"  timestep: {sim.model.opt.timestep} s  ->  {1 / sim.model.opt.timestep:.0f} physics steps per second")
print(f"  arm joint angles at start (rad): {sim.arm_q().round(3)}")
print(f"  gripper (TCP) position at start: {sim.tcp_pos().round(3)}")
for name in sim.block_names():
    print(f"  {name:<5} block at {sim.block_pos(name).round(3)}")

print("\nRunning 2 seconds with all motor torques = 0 ...")
for i in range(int(2.0 / C.TIMESTEP)):
    sim.step()                       # no controller: ctrl stays at zero
    if i % 250 == 0:
        print(f"  t = {sim.time:4.2f} s   gripper height z = {sim.tcp_pos()[2]:.3f} m")

print(f"\nFinal gripper position: {sim.tcp_pos().round(3)}")
print("The arm collapsed under its own weight.  Motors do not 'hold position' by")
print("themselves - something has to keep choosing torques.  That is stage 2.")
finish()
sim.close()
