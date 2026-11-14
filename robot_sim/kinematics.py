"""KINEMATICS: the geometry linking joint angles to gripper position.

  * FORWARD kinematics (FK):  joint angles  ->  where is the gripper?
        Easy: just add up the links, like following directions on a map.
  * INVERSE kinematics (IK):  gripper position  ->  which joint angles?
        Harder: like being told the destination and having to work out the
        directions.  There may be several answers, or none (out of reach).

This arm always works with the gripper pointing STRAIGHT DOWN (a "top-down
grasp").  That restriction makes IK solvable with school trigonometry:

  1. Looking from above, the arm is just a line from the base to the target.
     The "pan" joint turns that line to the right compass direction.
  2. Looking from the side, the upper arm and forearm form a triangle with
     the line from shoulder to wrist.  We know all three side lengths, so the
     law of cosines gives the elbow angle.
  3. The wrist then bends by whatever is left to make the gripper vertical.

Angle convention: the shoulder angle is measured from "straight up", and
positive angles lean forward.  The elbow and wrist angles are measured
relative to the previous link.
"""

from __future__ import annotations

import math

import numpy as np

from . import config as C


class UnreachableError(ValueError):
    """Raised when no joint angles can put the gripper at the requested spot."""


def forward_kinematics(q) -> np.ndarray:
    """Joint angles (5 numbers) -> TCP position (x, y, z) in the world frame."""
    pan, shoulder, elbow, wrist_pitch, _roll = q   # roll does not move the TCP
    # Direction of each link in the side view, measured from straight up.
    a1 = shoulder                      # upper arm
    a2 = shoulder + elbow              # forearm
    a3 = shoulder + elbow + wrist_pitch  # tool (pi = pointing straight down)
    # r = horizontal distance from the base axis; z = height.
    r = (C.UPPER_ARM_LEN * math.sin(a1) + C.FOREARM_LEN * math.sin(a2)
         + C.TOOL_LEN * math.sin(a3))
    z = (C.SHOULDER_HEIGHT + C.UPPER_ARM_LEN * math.cos(a1)
         + C.FOREARM_LEN * math.cos(a2) + C.TOOL_LEN * math.cos(a3))
    return np.array([r * math.cos(pan), r * math.sin(pan), z])


def inverse_kinematics(x: float, y: float, z: float) -> np.ndarray:
    """TCP position -> joint angles, with the gripper pointing straight down
    and the fingers kept square to the world's x/y axes.

    Raises UnreachableError if the point is out of reach or breaks a joint limit.
    """
    r = math.hypot(x, y)               # horizontal distance from the base
    if r < 1e-6:
        raise UnreachableError("target is directly above the base")

    # Step 1 (top view): turn to face the target.
    pan = math.atan2(y, x)

    # Step 2 (side view): the gripper points down, so the WRIST joint must sit
    # exactly TOOL_LEN above the target.
    wrist_r = r
    wrist_h = z + C.TOOL_LEN - C.SHOULDER_HEIGHT   # height above the shoulder
    d = math.hypot(wrist_r, wrist_h)               # shoulder -> wrist distance

    l1, l2 = C.UPPER_ARM_LEN, C.FOREARM_LEN
    # Law of cosines.  cos_elbow outside [-1, 1] means the triangle cannot
    # be formed: the wrist point is too far away (or too close).
    cos_elbow = (d * d - l1 * l1 - l2 * l2) / (2 * l1 * l2)
    if not -1.0 <= cos_elbow <= 1.0:
        raise UnreachableError(
            f"({x:.3f}, {y:.3f}, {z:.3f}) is out of reach "
            f"(needs {d:.3f} m from the shoulder, arm spans {l1 + l2:.3f} m)")
    elbow = math.acos(cos_elbow)       # the positive answer = "elbow up" pose

    # Angle from straight-up to the shoulder->wrist line, minus the corner of
    # the triangle at the shoulder.
    shoulder = math.atan2(wrist_r, wrist_h) - math.atan2(
        l2 * math.sin(elbow), l1 + l2 * math.cos(elbow))

    # Step 3: the three side-view angles must add up to pi (straight down).
    wrist_pitch = math.pi - shoulder - elbow

    # Step 4: turning the base also turns the fingers.  Rolling the wrist by
    # the same angle cancels that out, so the fingers always close along the
    # world's y axis and meet square blocks face-on.
    roll = pan

    q = np.array([pan, shoulder, elbow, wrist_pitch, roll])
    for name, value in zip(C.ARM_JOINTS, q):
        lo, hi = C.JOINT_LIMITS[name]
        if not lo <= value <= hi:
            raise UnreachableError(
                f"({x:.3f}, {y:.3f}, {z:.3f}) needs joint '{name}' at "
                f"{value:.2f} rad, outside its limits [{lo}, {hi}]")
    return q
