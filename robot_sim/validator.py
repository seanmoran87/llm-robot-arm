"""THE SAFETY VALIDATOR: a gate between the LLM and the robot.

An LLM is a brilliant guesser, not a guaranteed-correct machine.  It can
misread a number, forget what it is holding, or invent a block that does not
exist.  On a real robot those slips break things.  So NOTHING the LLM asks
for reaches the skills until this file has approved it.

Key design choice: the validator is ordinary, boring, predictable code.
It does not "understand" the goal.  It only knows hard facts about what is
physically safe:

    * Is this a tool we actually have, with sensible numbers?
    * Is the target inside the allowed workspace, and reachable?
    * Would the gripper or the held block plough through another block?
    * Is the robot in the right state (holding / not holding)?
    * Would a placed block have something solid underneath it?

If a check fails, the action is NOT executed and the reason is sent back to
the LLM in plain English, so it can try a better plan.

Analogy: a spell-checker cannot write your essay, but it reliably catches a
class of mistakes no matter how confident the author was.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from . import config as C
from .kinematics import UnreachableError, inverse_kinematics

HALF = C.BLOCK_SIZE / 2
FINGER_REACH = C.FINGER_OPEN + 0.012     # how far the open fingers stick out sideways


@dataclass
class Verdict:
    """The validator's answer: allowed or not, and why."""
    ok: bool
    reason: str = "ok"


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


class Validator:
    KNOWN_TOOLS = {"get_scene", "move_to", "grasp", "place", "go_home"}

    def check(self, tool: str, args: dict, scene: dict) -> Verdict:
        """Decide whether `tool(**args)` is safe given the current `scene`.

        `scene` is the dictionary returned by Skills.get_scene().
        """
        if tool not in self.KNOWN_TOOLS:
            return Verdict(False, f"unknown tool '{tool}'")
        if not isinstance(args, dict):
            return Verdict(False, "arguments must be an object")
        if tool in ("get_scene", "go_home"):
            return Verdict(True)                 # always safe
        if tool == "move_to":
            return self._check_move_to(args, scene)
        if tool == "grasp":
            return self._check_grasp(args, scene)
        return self._check_place(args, scene)

    # ------------------------------------------------------------- helpers
    def _check_xyz(self, args: dict) -> Verdict | None:
        for key in ("x", "y", "z"):
            if not _is_number(args.get(key)):
                return Verdict(False, f"'{key}' must be a finite number, got {args.get(key)!r}")
        return None

    def _check_workspace(self, x: float, y: float, z: float, what: str) -> Verdict | None:
        """Is the point inside the allowed region and reachable by the arm?"""
        r = math.hypot(x, y)
        angle = math.atan2(y, x)
        if not C.WORKSPACE_R_MIN <= r <= C.WORKSPACE_R_MAX:
            return Verdict(False, f"{what} is {r:.3f} m from the base; allowed range is "
                                  f"{C.WORKSPACE_R_MIN}-{C.WORKSPACE_R_MAX} m")
        if abs(angle) > C.WORKSPACE_ANGLE_MAX:
            return Verdict(False, f"{what} is too far round to the side/behind the robot")
        if not C.WORKSPACE_Z_MIN <= z <= C.WORKSPACE_Z_MAX:
            return Verdict(False, f"{what} height z={z:.3f} m is outside the allowed "
                                  f"{C.WORKSPACE_Z_MIN}-{C.WORKSPACE_Z_MAX} m "
                                  f"(z=0 is the table surface)")
        try:
            inverse_kinematics(x, y, z)
        except UnreachableError as e:
            return Verdict(False, f"{what} is not reachable: {e}")
        return None

    def _others(self, scene: dict) -> dict:
        """Blocks lying in the world (everything except the one being held)."""
        return {n: p for n, p in scene["blocks"].items() if n != scene.get("holding")}

    # -------------------------------------------------------------- move_to
    def _check_move_to(self, args: dict, scene: dict) -> Verdict:
        bad = self._check_xyz(args)
        if bad:
            return bad
        x, y, z = args["x"], args["y"], args["z"]
        bad = self._check_workspace(x, y, z, "target")
        if bad:
            return bad

        # Check the whole PATH, not just the destination.  We sample points
        # along the same cylindrical path the controller will follow and make
        # sure the lowest part of the gripper (or of the held block) clears
        # every block it passes over by at least 1 cm.
        sx, sy, sz = scene["gripper_position"]
        r0, a0 = math.hypot(sx, sy), math.atan2(sy, sx)
        r1, a1 = math.hypot(x, y), math.atan2(y, x)
        lowest_offset = HALF if scene.get("holding") else C.FINGERTIP_BELOW_TCP
        for i in range(1, 41):
            s = i / 40
            r, a, zz = r0 + s * (r1 - r0), a0 + s * (a1 - a0), sz + s * (z - sz)
            px, py = r * math.cos(a), r * math.sin(a)
            for name, (bx, by, bz) in self._others(scene).items():
                near = abs(px - bx) < HALF + 0.02 and abs(py - by) < HALF + FINGER_REACH
                if near and zz - lowest_offset < bz + HALF + 0.01:
                    return Verdict(False, f"path would hit the {name} block; travel at "
                                          f"z >= {C.TRANSIT_Z} to pass over blocks")
        return Verdict(True)

    # ---------------------------------------------------------------- grasp
    def _check_grasp(self, args: dict, scene: dict) -> Verdict:
        block = args.get("block")
        if not isinstance(block, str) or block not in scene["blocks"]:
            return Verdict(False, f"no block called {block!r}; known blocks: "
                                  f"{sorted(scene['blocks'])}")
        if scene.get("holding"):
            return Verdict(False, f"gripper is already holding {scene['holding']}; "
                                  f"place it first")
        bx, by, bz = scene["blocks"][block]
        bad = self._check_workspace(bx, by, max(bz, C.WORKSPACE_Z_MIN), f"the {block} block")
        if bad:
            return bad
        for name, (ox, oy, oz) in scene["blocks"].items():
            if name == block:
                continue
            # Something sitting on top of it?  Lifting it would topple the pile.
            if abs(ox - bx) < C.BLOCK_SIZE and abs(oy - by) < C.BLOCK_SIZE and oz > bz + HALF:
                return Verdict(False, f"the {name} block is on top of {block}; move {name} first")
            # A neighbour so close that the open fingers would land on it?
            if (abs(oz - bz) < C.BLOCK_SIZE - 0.006 and abs(ox - bx) < HALF + 0.015
                    and abs(oy - by) < HALF + FINGER_REACH):
                return Verdict(False, f"the {name} block is too close beside {block} "
                                      f"for the fingers to fit")
        return Verdict(True)

    # ---------------------------------------------------------------- place
    def _check_place(self, args: dict, scene: dict) -> Verdict:
        bad = self._check_xyz(args)
        if bad:
            return bad
        if not scene.get("holding"):
            return Verdict(False, "gripper is not holding anything; grasp a block first")
        x, y, z = args["x"], args["y"], args["z"]
        bad = self._check_workspace(x, y, max(z, C.WORKSPACE_Z_MIN), "place target")
        if bad:
            return bad

        bottom = z - HALF                      # underside of the block being placed
        if bottom < -0.003:
            return Verdict(False, f"z={z:.3f} would push the block into the table; a block "
                                  f"resting on the table has z={HALF}")

        supported = abs(bottom) <= 0.006       # resting on the table?
        for name, (ox, oy, oz) in self._others(scene).items():
            dx, dy, dz = abs(x - ox), abs(y - oy), z - oz
            # Would the two blocks occupy the same space?
            if dx < C.BLOCK_SIZE - 0.002 and dy < C.BLOCK_SIZE - 0.002 and abs(dz) < C.BLOCK_SIZE - 0.006:
                return Verdict(False, f"that spot is occupied by the {name} block")
            # Resting squarely on top of this block?
            if abs(dz - C.BLOCK_SIZE) <= 0.006 and dx <= 0.012 and dy <= 0.012:
                supported = True
            elif abs(dz - C.BLOCK_SIZE) <= 0.006 and dx < C.BLOCK_SIZE and dy < C.BLOCK_SIZE:
                return Verdict(False, f"placement overhangs the {name} block by "
                                      f"{max(dx, dy) * 1000:.0f} mm and would be unstable; "
                                      f"use x={ox}, y={oy}")
            # A neighbour at the same level where the fingers need to open?
            if abs(dz) < C.BLOCK_SIZE - 0.006 and dx < HALF + 0.015 and dy < HALF + FINGER_REACH:
                return Verdict(False, f"too close to the {name} block; the fingers "
                                      f"need room to open")
        if not supported:
            return Verdict(False, f"nothing to rest on at z={z:.3f}: the block would be "
                                  f"dropped from mid-air. On the table z={HALF}; on top of "
                                  f"a block z = that block's z + {C.BLOCK_SIZE}")
        return Verdict(True)
