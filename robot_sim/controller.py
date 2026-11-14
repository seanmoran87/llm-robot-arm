"""LAYER (b): THE LOW-LEVEL CONTROLLER.

The physics layer accepts motor torques.  Humans (and the layers above) think
in positions: "put the gripper THERE".  This file bridges that gap:

    target gripper position
        -> inverse kinematics   (kinematics.py)  -> target joint angles
        -> trajectory           (a smooth path over time, not a sudden jump)
        -> PID controller       (runs 500x per second)  -> motor torques

PID in one paragraph.  For each joint, 500 times a second, compare where the
joint IS with where it SHOULD be.  The difference is the "error".  The torque
we apply is the sum of three terms:

    P (proportional): kp * error            -> push harder the further off you are
    I (integral):     ki * sum of past error -> lean in if you have been off for a while
    D (derivative):   kd * rate of change    -> ease off as you approach, to avoid overshoot

Analogy: steering a car back to the centre of a lane.  P turns the wheel more
the further you have drifted.  D notices you are already swinging back and
straightens up early.  I notices a constant crosswind pushing you off and
holds a slight permanent correction.

We also add "gravity compensation": the known torque needed just to hold the
arm's own weight (a FEEDFORWARD term - computed from a model in advance,
rather than as a reaction to error).  Try switching it off in
stages/02_controller.py to see what the PID terms then have to cope with.
"""

from __future__ import annotations

import math

import numpy as np

from . import config as C
from .kinematics import forward_kinematics, inverse_kinematics
from .scene import Sim


def min_jerk(s: float) -> float:
    """Smooth 0 -> 1 ramp.  Input s is the fraction of time elapsed (0..1).

    A straight ramp would start and stop instantly, jerking the arm.  This
    curve (10 s^3 - 15 s^4 + 6 s^5) starts and ends with zero speed and zero
    acceleration, the way your own hand moves when you reach for a cup.
    """
    s = min(max(s, 0.0), 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


class ArmController:
    """Drives the arm joints with PID and the gripper with a simple squeeze."""

    def __init__(self, sim: Sim, gravity_compensation: bool = True):
        self.sim = sim
        self.gravity_compensation = gravity_compensation

        # PID gains, one per arm joint (pan, shoulder, elbow, wrist_pitch,
        # wrist_roll).  Bigger joints carry more weight so they get stronger
        # gains.  These were tuned by trial and error in simulation - which is
        # how gains are normally found in practice too.
        self.kp = np.array([120.0, 220.0, 160.0, 60.0, 12.0])
        self.ki = np.array([40.0, 80.0, 60.0, 20.0, 4.0])
        self.kd = np.array([8.0, 14.0, 10.0, 3.0, 0.4])
        self.integral_limit = 0.5      # cap on the I term's memory ("anti-windup")
        self.integral_zone = 0.05      # only integrate within this many radians of target

        self.q_target = sim.arm_q()            # where the joints should be
        self.qvel_target = np.zeros(5)         # how fast they should be moving
        self._integral = np.zeros(5)
        self.gripper_closed = False

    # ---------------------------------------------------------------- core
    def step(self) -> None:
        """One control cycle: read sensors, compute torques, advance physics."""
        q = self.sim.arm_q()
        qvel = self.sim.arm_qvel()

        error = self.q_target - q
        # Accumulate error over time for the I term - but only for joints
        # that are already close to their target, and with a cap on the total.
        # Otherwise a joint that is far away (or blocked) would build up a
        # huge "debt" and then overshoot badly while paying it back.  That
        # failure is called integral windup.
        close = np.abs(error) < self.integral_zone
        self._integral = np.where(close, self._integral + error * C.TIMESTEP, self._integral)
        self._integral = np.clip(self._integral, -self.integral_limit, self.integral_limit)

        torque = (self.kp * error
                  + self.ki * self._integral
                  + self.kd * (self.qvel_target - qvel))
        if self.gravity_compensation:
            torque += self.sim.gravity_torques()
        self.sim.set_arm_torques(torque)

        # Gripper: push with constant force towards open or closed.  When the
        # fingers meet a block they simply stop, squeezing it with GRIP_FORCE.
        # (A small spring-like term slows the fingers near their open stop.)
        if self.gripper_closed:
            # Stop pushing once the pads have met in the middle (nothing to grip).
            fully_shut = self.sim.finger_pos() <= C.FINGER_CLOSED
            self.sim.set_finger_force(0.0 if fully_shut else -C.GRIP_FORCE)
        else:
            gap_error = C.FINGER_OPEN - self.sim.finger_pos()
            self.sim.set_finger_force(float(np.clip(400.0 * gap_error, -C.GRIP_FORCE, C.GRIP_FORCE)))

        self.sim.step()

    def hold(self, seconds: float) -> None:
        """Keep the current targets for a while (lets things settle)."""
        for _ in range(int(round(seconds / C.TIMESTEP))):
            self.step()

    # ------------------------------------------------------------ movement
    def move_joints(self, q_goal, duration: float) -> None:
        """Move all joints smoothly from the current target to q_goal."""
        q_start = self.q_target.copy()
        q_goal = np.asarray(q_goal, dtype=float)
        steps = max(1, int(round(duration / C.TIMESTEP)))
        previous = q_start
        for i in range(1, steps + 1):
            self.q_target = q_start + min_jerk(i / steps) * (q_goal - q_start)
            self.qvel_target = (self.q_target - previous) / C.TIMESTEP
            previous = self.q_target
            self.step()
        self.qvel_target = np.zeros(5)

    def move_tcp(self, x: float, y: float, z: float, speed: float = 0.20,
                 settle: float = 0.25) -> None:
        """Move the gripper (pointing down) to (x, y, z).

        The path is planned in "cylindrical" coordinates - distance from the
        base (r), compass angle around the base, and height - because those
        match how this arm is built.  Blending r, angle and height smoothly
        guarantees the path never cuts across the robot's own base, which a
        straight line between two points on opposite sides would do.
        Purely vertical moves (same x, y) come out as exact straight lines.

        Raises kinematics.UnreachableError BEFORE moving if any point on the
        path cannot be reached.
        """
        start = forward_kinematics(self.q_target)
        r0, a0, z0 = math.hypot(start[0], start[1]), math.atan2(start[1], start[0]), start[2]
        r1, a1, z1 = math.hypot(x, y), math.atan2(y, x), z

        # Rough path length sets how long the move should take.
        length = math.sqrt((r1 - r0) ** 2 + (max(r0, r1) * (a1 - a0)) ** 2 + (z1 - z0) ** 2)
        duration = max(0.4, 1.875 * length / speed)   # 1.875 = min-jerk peak/average speed
        steps = max(1, int(round(duration / C.TIMESTEP)))

        # Plan the whole path first, so an unreachable point is discovered
        # before the arm has moved at all.
        path = []
        for i in range(1, steps + 1):
            s = min_jerk(i / steps)
            r, a, zz = r0 + s * (r1 - r0), a0 + s * (a1 - a0), z0 + s * (z1 - z0)
            path.append(inverse_kinematics(r * math.cos(a), r * math.sin(a), zz))

        previous = self.q_target
        for q in path:
            self.q_target = q
            self.qvel_target = (q - previous) / C.TIMESTEP
            previous = q
            self.step()
        self.qvel_target = np.zeros(5)
        self.hold(settle)

    # ------------------------------------------------------------- gripper
    def set_gripper(self, closed: bool, wait: float = 0.6) -> None:
        self.gripper_closed = closed
        self.hold(wait)

    def gripper_gap(self) -> float:
        """Distance between the two finger pads (metres)."""
        return 2.0 * self.sim.finger_pos()
