"""STAGE 0 - kinematics by itself (pure geometry, no physics, runs instantly).

    python stages/00_kinematics.py 0.30 0.10 0.05      (a target x y z, metres)
    python stages/00_kinematics.py 0.50 0.00 0.05      (too far - see what happens)

Prints the joint angles that put the gripper at the target (inverse
kinematics), then feeds those angles back through forward kinematics to
check that we land where we wanted.
"""
import math
import sys

import _common  # noqa: F401  (sets up the import path)

from robot_sim import config as C
from robot_sim.kinematics import UnreachableError, forward_kinematics, inverse_kinematics

x, y, z = (float(v) for v in sys.argv[1:4]) if len(sys.argv) >= 4 else (0.30, 0.10, 0.05)
print(f"Target gripper position: x={x}  y={y}  z={z}   (metres)\n")
try:
    q = inverse_kinematics(x, y, z)
except UnreachableError as e:
    raise SystemExit(f"Inverse kinematics says NO: {e}")

print("Inverse kinematics -> joint angles")
for name, angle in zip(C.ARM_JOINTS, q):
    print(f"  {name:<12} {angle:7.3f} rad  = {math.degrees(angle):7.1f} deg")
print(f"\n  shoulder + elbow + wrist_pitch = {math.degrees(q[1] + q[2] + q[3]):.1f} deg "
      f"(always 180: gripper straight down)")
back = forward_kinematics(q)
print(f"\nForward kinematics of those angles -> ({back[0]:.4f}, {back[1]:.4f}, {back[2]:.4f})")
print(f"Round-trip error: {1000 * math.dist(back, (x, y, z)):.6f} mm")
