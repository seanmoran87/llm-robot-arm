"""STAGE 2 - the low-level controller.

The PID controller holds the arm up and moves the gripper through a square.
The table printed at the end shows how closely the arm tracked each target.

    python stages/02_controller.py
    python stages/02_controller.py --no-gravity-comp     (PID has to do it all)
    python stages/02_controller.py --kp-scale 0.1        (a weak, lazy controller)
    python stages/02_controller.py --kd-scale 0          (no damping term)
    python stages/02_controller.py --ki-scale 0 --no-gravity-comp
    python stages/02_controller.py --step                (see below)

Part 1 moves along smooth trajectories, which are kind to any controller.
Part 2 (--step) is the harsh test engineers actually use to judge a
controller: the elbow's target JUMPS by 0.5 rad in an instant, and we measure
  overshoot      how far past the target the joint swings
  settling time  how long until it stays within 2% of the jump

PREDICT before each variation: will the error get bigger or smaller?  Will
the arm sag, overshoot, or wobble?
"""
import argparse

import numpy as np
from _common import attach_outputs

from robot_sim.controller import ArmController
from robot_sim.scene import Sim

parser = argparse.ArgumentParser()
parser.add_argument("--viewer", action="store_true")
parser.add_argument("--no-gravity-comp", action="store_true")
parser.add_argument("--kp-scale", type=float, default=1.0)
parser.add_argument("--ki-scale", type=float, default=1.0)
parser.add_argument("--kd-scale", type=float, default=1.0)
parser.add_argument("--step", action="store_true", help="run the step-response test instead")
args = parser.parse_args()

sim = Sim({"red": (0.16, -0.30), "green": (0.20, -0.30), "blue": (0.24, -0.30)})  # blocks moved aside
finish = attach_outputs(sim, "02_controller", args.viewer)
ctrl = ArmController(sim, gravity_compensation=not args.no_gravity_comp)
ctrl.kp *= args.kp_scale
ctrl.ki *= args.ki_scale
ctrl.kd *= args.kd_scale

# Measure the worst gap between target and reality during each move.
worst = {"value": 0.0}
def track(sim):
    from robot_sim.kinematics import forward_kinematics
    gap = np.linalg.norm(forward_kinematics(ctrl.q_target) - sim.tcp_pos())
    worst["value"] = max(worst["value"], gap)
sim.step_listeners.append(track)

print(f"gravity compensation: {ctrl.gravity_compensation}   kp x{args.kp_scale}  ki x{args.ki_scale}  kd x{args.kd_scale}\n")
ctrl.hold(1.0)

if args.step:
    from robot_sim import config as C
    JOINT, JUMP = 2, -0.5                  # elbow; negative = straighten the arm upwards
    start = ctrl.q_target[JOINT]
    ctrl.q_target[JOINT] += JUMP           # the instant jump: no trajectory
    history = []
    for _ in range(int(2.0 / C.TIMESTEP)):
        ctrl.step()
        history.append((sim.arm_q()[JOINT] - start) / JUMP)   # 0 = start, 1 = target
    history = np.array(history)
    overshoot = max(0.0, history.max() - 1.0) * 100
    outside = np.nonzero(np.abs(history - 1.0) > 0.02)[0]
    settle = (outside[-1] + 1) * C.TIMESTEP if len(outside) else 0.0
    print("STEP RESPONSE of the elbow joint (target jumps 0.5 rad)")
    print("  time (s)   progress towards target (1.00 = exactly on target)")
    for t in (0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 2.0):
        value = history[int(t / C.TIMESTEP) - 1]
        print(f"   {t:5.2f}     {value:5.2f}  " + "#" * int(max(0.0, min(value, 1.6)) * 40))
    print(f"\n  overshoot:      {overshoot:5.1f} %")
    print(f"  settling time:  {'never (within 2 s)' if settle >= 1.99 else f'{settle:.2f} s'}")
    print(f"  final error:    {abs(history[-1] - 1.0) * abs(JUMP) * 1000:5.2f} mrad")
    finish()
    sim.close()
    raise SystemExit

print(f"{'target (x, y, z)':<24}{'reached (x, y, z)':<26}{'final error':>12}{'worst during move':>20}")
corners = [(0.20, -0.10, 0.10), (0.32, -0.10, 0.10), (0.32, 0.10, 0.10), (0.20, 0.10, 0.10), (0.22, 0.0, 0.22)]
for target in corners:
    worst["value"] = 0.0
    ctrl.move_tcp(*target)
    reached = sim.tcp_pos()
    err = np.linalg.norm(reached - target)
    print(f"{str(target):<24}{str(tuple(round(float(v), 3) for v in reached)):<26}{err * 1000:>9.2f} mm{worst['value'] * 1000:>17.2f} mm")
finish()
sim.close()
