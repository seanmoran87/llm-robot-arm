"""LAYER (c): THE SKILL LIBRARY.

A "skill" is a small, reliable, named behaviour built out of controller
commands - the robot equivalent of a verb.  The three core skills are:

    move_to(x, y, z)    put the (empty or loaded) gripper at a point
    grasp(block)        pick up a named block and lift it
    place(x, y, z)      set the held block down with its centre at a point

Why bother with this layer?  Because the LLM above is good at deciding WHAT
to do ("red goes on blue") and bad at the split-second HOW (500 torque values
per second).  Skills are the vocabulary that lets it stay in its lane.

Every skill returns a plain dictionary describing what ACTUALLY happened,
measured from the simulator's sensors - never what was merely intended.
That honest report is what the LLM reads to decide its next step, and what
ends up in the plan-vs-actual log.
"""

from __future__ import annotations

import math

import numpy as np

from . import config as C
from .controller import ArmController
from .kinematics import UnreachableError
from .scene import Sim


def _round(v, digits: int = 3) -> list[float]:
    """Turn a numpy vector into a short list of plain rounded floats."""
    return [round(float(a), digits) for a in v]


class Skills:
    def __init__(self, sim: Sim, controller: ArmController, perception=None):
        self.sim = sim
        self.ctrl = controller
        # `perception` is any function that returns {block_name: (x, y, z)}.
        # By default we read the simulator's ground truth; perception.py
        # offers a camera-based replacement with the same shape of output.
        self.perception = perception or self._ground_truth
        self.holding: str | None = None        # name of the block in the gripper

    def _ground_truth(self) -> dict[str, np.ndarray]:
        return {name: self.sim.block_pos(name) for name in self.sim.block_names()}

    def _look(self) -> dict[str, np.ndarray]:
        """Get block positions.  If we are using a real camera, first swing
        the arm out of the camera's line of sight - you cannot see the table
        through your own arm."""
        owner = getattr(self.perception, "__self__", None)   # the object the function belongs to
        if getattr(owner, "needs_clear_view", False):
            lx = C.LOOK_POSE_R * math.cos(C.LOOK_POSE_ANGLE)
            ly = C.LOOK_POSE_R * math.sin(C.LOOK_POSE_ANGLE)
            tcp = self.sim.tcp_pos()
            if np.linalg.norm(tcp - [lx, ly, C.TRANSIT_Z]) > 0.005:
                if tcp[2] < C.TRANSIT_Z - 0.01:
                    self.ctrl.move_tcp(tcp[0], tcp[1], C.TRANSIT_Z)
                self.ctrl.move_tcp(lx, ly, C.TRANSIT_Z, speed=0.35)
        return self.perception()

    # ---------------------------------------------------------------- sense
    def get_scene(self) -> dict:
        """Report what the robot knows about the world right now."""
        blocks = {name: _round(pos) for name, pos in self._look().items()}
        return {
            "success": True,
            "blocks": blocks,                       # centre of each block (m)
            "holding": self.holding,
            "gripper_position": _round(self.sim.tcp_pos()),
            "block_size": C.BLOCK_SIZE,
            "table_z": 0.0,
        }

    # ----------------------------------------------------------------- move
    def move_to(self, x: float, y: float, z: float) -> dict:
        """Move the gripper to (x, y, z) and report how close it really got."""
        try:
            self.ctrl.move_tcp(x, y, z)
        except UnreachableError as e:
            return {"success": False, "error": str(e),
                    "gripper_position": _round(self.sim.tcp_pos())}
        actual = self.sim.tcp_pos()
        error = float(np.linalg.norm(actual - [x, y, z]))
        return {"success": error < 0.005, "gripper_position": _round(actual),
                "position_error_mm": round(error * 1000, 2)}

    def home(self) -> dict:
        """Return to the home pose (clear of the camera's view of the table)."""
        home_xyz = (C.FOREARM_LEN, 0.0,
                    C.SHOULDER_HEIGHT + C.UPPER_ARM_LEN - C.TOOL_LEN)
        tcp = self.sim.tcp_pos()
        if tcp[2] < C.TRANSIT_Z - 0.01:            # go up first, then across
            self.ctrl.move_tcp(tcp[0], tcp[1], C.TRANSIT_Z)
        return self.move_to(*home_xyz)

    # ---------------------------------------------------------------- grasp
    def grasp(self, block: str) -> dict:
        """Pick up `block`: hover above it, descend, close, check, lift."""
        if self.holding is not None:
            return {"success": False, "error": f"already holding {self.holding}"}
        seen = self._look()
        if block not in seen:
            return {"success": False, "error": f"cannot see a block called '{block}'"}
        bx, by, bz = (float(v) for v in seen[block])

        try:
            self.ctrl.set_gripper(closed=False, wait=0.3)
            tcp = self.sim.tcp_pos()
            # 1. Rise to a safe travelling height, then fly to above the block.
            if tcp[2] < C.TRANSIT_Z - 0.01:
                self.ctrl.move_tcp(tcp[0], tcp[1], C.TRANSIT_Z)
            self.ctrl.move_tcp(bx, by, C.TRANSIT_Z)
            # 2. Straight down until the block's centre is between the pads.
            self.ctrl.move_tcp(bx, by, max(bz, C.WORKSPACE_Z_MIN), speed=0.12)
            # 3. Squeeze.
            self.ctrl.set_gripper(closed=True)
            # 4. Straight back up.
            self.ctrl.move_tcp(bx, by, C.TRANSIT_Z, speed=0.15)
        except UnreachableError as e:
            return {"success": False, "error": str(e)}

        # 5. Did it work?  We check with a sensor a real gripper also has:
        #    how far the fingers closed.  If they stopped about one block-width
        #    apart, something is between them.  If they closed fully, we missed.
        gap = self.ctrl.gripper_gap()
        if gap > C.BLOCK_SIZE * 0.6:
            self.holding = block
            return {"success": True, "holding": block, "finger_gap_mm": round(gap * 1000, 1),
                    "gripper_position": _round(self.sim.tcp_pos())}
        self.ctrl.set_gripper(closed=False, wait=0.3)
        return {"success": False, "error": "fingers closed on nothing - grasp missed",
                "finger_gap_mm": round(gap * 1000, 1)}

    # ---------------------------------------------------------------- place
    def place(self, x: float, y: float, z: float) -> dict:
        """Set the held block down so its CENTRE ends up at (x, y, z).

        For a block resting on the table, z = block_size / 2.
        For a block resting on another block, z = that block's z + block_size.
        """
        if self.holding is None:
            return {"success": False, "error": "not holding anything"}
        block = self.holding
        try:
            tcp = self.sim.tcp_pos()
            if tcp[2] < C.TRANSIT_Z - 0.01:
                self.ctrl.move_tcp(tcp[0], tcp[1], C.TRANSIT_Z)
            self.ctrl.move_tcp(x, y, C.TRANSIT_Z)
            # Descend slowly and stop 2 mm high: letting go just above the
            # surface is gentler than pressing the block into it.
            self.ctrl.move_tcp(x, y, max(z + 0.002, C.WORKSPACE_Z_MIN), speed=0.10)
            self.ctrl.set_gripper(closed=False)
            self.holding = None
            self.ctrl.move_tcp(x, y, C.TRANSIT_Z, speed=0.15)
        except UnreachableError as e:
            return {"success": False, "error": str(e), "holding": self.holding}

        # Check where the block really ended up (it may have slid or toppled).
        self.ctrl.hold(0.3)
        actual = self._look().get(block)
        if actual is None:
            return {"success": False, "error": f"lost sight of {block} after placing"}
        error = float(np.linalg.norm(np.asarray(actual) - [x, y, z]))
        return {"success": error < 0.015, "placed": block,
                "block_position": _round(actual),
                "placement_error_mm": round(error * 1000, 1)}
