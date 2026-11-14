"""LAYER (a): THE PHYSICS SCENE.

This file answers one question: "what exists in the world, and how does it
move when forces act on it?"  It knows nothing about goals, plans or language.

We use MuJoCo, a physics engine ("engine" = a library that computes how rigid
bodies move, collide and rub against each other).  MuJoCo needs two things:

  * a MODEL  - the fixed description of the world: shapes, masses, joints,
               motors.  It is written in an XML dialect called MJCF.
  * the DATA - everything that changes over time: joint angles, velocities,
               contact forces, the clock.

Each call to `mujoco.mj_step(model, data)` advances the clock by one timestep
(2 ms here) using Newton's laws.  That is the whole simulation loop.

The arm is a chain of rigid "bodies" connected by "joints":

    table -> base -> [pan] -> turret -> [shoulder] -> upper arm -> [elbow]
          -> forearm -> [wrist_pitch] -> wrist -> [wrist_roll] -> hand
          -> two fingers that slide towards each other

A "hinge" joint rotates about an axis (like a door hinge); a "slide" joint
moves along an axis (like a drawer).  Each joint has a "motor" (MuJoCo calls
it an actuator) that we command with a torque.  Choosing those torques is the
controller's job, in controller.py.
"""

from __future__ import annotations

import os
import sys

import numpy as np

# MuJoCo draws images with OpenGL.  On Linux machines without a screen (cloud
# servers), it must be told to use "EGL", which renders without a window.
# On Windows and macOS the default works, so we leave it alone there.
if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402  (must come after the environment variable above)

from . import config as C


def build_mjcf(block_positions: dict[str, tuple[float, float]] | None = None) -> str:
    """Return the MJCF (XML) text describing the whole world.

    We generate the XML from Python so that every length comes from config.py.
    Reading tips for the XML below:
      pos="x y z"      position relative to the PARENT body
      size             half-sizes for boxes; radius (and half-length) otherwise
      rgba             colour: red green blue opacity
      contype/conaffinity   collision "groups" (explained at the <default> tag)
    """
    blocks = dict(C.BLOCK_START)
    if block_positions:
        blocks.update(block_positions)

    half = C.BLOCK_SIZE / 2
    block_xml = ""
    for name, (bx, by) in blocks.items():
        r, g, b, a = C.BLOCK_COLORS[name]
        # <freejoint/> lets a body move and rotate freely: the block is not
        # attached to anything, it just lies on the table under gravity.
        block_xml += f"""
    <body name="block_{name}" pos="{bx} {by} {half}">
      <freejoint name="block_{name}_joint"/>
      <geom name="block_{name}_geom" class="object" type="box"
            size="{half} {half} {half}" mass="{C.BLOCK_MASS}" rgba="{r} {g} {b} {a}"/>
    </body>"""

    lim = C.JOINT_LIMITS
    tq = C.TORQUE_LIMITS
    # Where the finger bodies and the TCP sit, measured from the hand body.
    wrist_stub = 0.03                       # short link between wrist and hand
    tcp_z = C.TOOL_LEN - wrist_stub         # hand origin -> TCP
    finger_half_len = 0.0275
    finger_z = tcp_z + C.FINGERTIP_BELOW_TCP - finger_half_len

    return f"""
<mujoco model="llm_robot_arm">
  <!-- Solver settings.  timestep: how far the clock advances per step.
       cone/impratio: settings that make friction contacts firmer, which
       keeps a grasped block from slowly sliding out of the fingers. -->
  <option timestep="{C.TIMESTEP}" integrator="implicitfast" cone="elliptic" impratio="10"/>
  <compiler angle="radian" autolimits="true"/>

  <visual>
    <global offwidth="1280" offheight="960"/>
    <headlight ambient="0.40 0.40 0.40" diffuse="0.35 0.35 0.35" specular="0 0 0"/>
    <quality shadowsize="2048"/>
  </visual>

  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.86 0.86 0.84"
             rgb2="0.80 0.80 0.78" width="256" height="256"/>
    <material name="floor_mat" texture="grid" texrepeat="6 6" reflectance="0.05"/>
    <material name="wood" rgba="0.72 0.58 0.42 1"/>
    <material name="arm_light" rgba="0.92 0.92 0.94 1"/>
    <material name="arm_dark" rgba="0.18 0.18 0.20 1"/>
    <material name="joint_mat" rgba="0.30 0.31 0.34 1"/>
    <texture type="skybox" builtin="gradient" rgb1="0.80 0.87 0.95" rgb2="0.97 0.97 0.98" width="64" height="64"/>
  </asset>

  <default>
    <!-- Collision groups.  Two shapes collide if one's "contype" shares a bit
         with the other's "conaffinity".
           objects (table, blocks): contype 1, conaffinity 1 -> hit each other
           arm parts:               contype 2, conaffinity 1 -> hit objects,
                                    but NOT other arm parts (keeps things simple
                                    and stops neighbouring links from jamming). -->
    <default class="object">
      <geom contype="1" conaffinity="1" friction="1.0 0.005 0.0001" condim="4"/>
    </default>
    <default class="arm">
      <geom contype="2" conaffinity="1" material="arm_light"/>
      <!-- damping: friction inside the joint.  armature: the motor's own
           rotor inertia.  Both make the arm calmer and easier to control. -->
      <joint damping="0.4" armature="0.02"/>
    </default>
    <default class="finger">
      <geom contype="2" conaffinity="1" friction="2.0 0.01 0.0001" condim="4"
            material="arm_dark"/>
      <joint type="slide" damping="8" armature="0.002" range="0 0.04"/>
    </default>
  </default>

  <worldbody>
    <light pos="0.3 -0.4 1.2" dir="-0.1 0.3 -1" diffuse="0.45 0.45 0.45" specular="0.1 0.1 0.1" castshadow="true"/>
    <geom name="floor" type="plane" pos="0 0 -0.40" size="2 2 0.1" material="floor_mat"
          contype="1" conaffinity="1"/>

    <!-- The table.  Its TOP surface is at z = 0, which is why "height above
         the table" and "z" are the same number everywhere in this project. -->
    <geom name="table_top" class="object" type="box" pos="0.20 0 -0.015"
          size="0.34 0.40 0.015" material="wood"/>
    <geom type="box" pos="-0.10 -0.36 -0.215" size="0.02 0.02 0.185" material="wood" contype="0" conaffinity="0"/>
    <geom type="box" pos="-0.10  0.36 -0.215" size="0.02 0.02 0.185" material="wood" contype="0" conaffinity="0"/>
    <geom type="box" pos=" 0.50 -0.36 -0.215" size="0.02 0.02 0.185" material="wood" contype="0" conaffinity="0"/>
    <geom type="box" pos=" 0.50  0.36 -0.215" size="0.02 0.02 0.185" material="wood" contype="0" conaffinity="0"/>

    <!-- Cameras.  "overview" is for humans to watch.  "topdown" looks straight
         down and is used by perception.py to find the blocks. -->
    <camera name="overview" pos="0.72 -0.50 0.42" xyaxes="0.60 0.80 0 -0.38 0.285 0.88" fovy="45"/>
    <camera name="topdown" pos="0.25 0 0.90" fovy="50"/>
    <camera name="side" pos="0.22 -0.95 0.16" xyaxes="1 0 0 0 0.12 1" fovy="40"/>

    <!-- ============================ THE ARM ============================ -->
    <body name="base" pos="0 0 0" childclass="arm">
      <geom type="cylinder" pos="0 0 0.025" size="0.055 0.025" material="arm_dark"/>

      <body name="turret" pos="0 0 0.05">
        <!-- JOINT 1 "pan": spins the whole arm left/right about the vertical axis. -->
        <joint name="pan" type="hinge" axis="0 0 1" range="{lim['pan'][0]} {lim['pan'][1]}"/>
        <geom type="cylinder" pos="0 0 0.015" size="0.04 0.015" mass="0.3"/>
        <geom type="box" pos="0 0 0.04" size="0.022 0.03 0.018" mass="0.1"/>

        <body name="upper_arm" pos="0 0 {C.SHOULDER_HEIGHT - 0.05}">
          <!-- JOINT 2 "shoulder": tilts the upper arm forward/back.
               Angle 0 = pointing straight up; positive = leaning forward (+x). -->
          <joint name="shoulder" type="hinge" axis="0 1 0" range="{lim['shoulder'][0]} {lim['shoulder'][1]}"/>
          <geom type="cylinder" fromto="0 -0.032 0 0 0.032 0" size="0.022" material="joint_mat" mass="0.05"/>
          <geom type="capsule" fromto="0 0 0 0 0 {C.UPPER_ARM_LEN}" size="0.018" mass="0.25"/>

          <body name="forearm" pos="0 0 {C.UPPER_ARM_LEN}">
            <!-- JOINT 3 "elbow": bends the forearm relative to the upper arm. -->
            <joint name="elbow" type="hinge" axis="0 1 0" range="{lim['elbow'][0]} {lim['elbow'][1]}"/>
            <geom type="cylinder" fromto="0 -0.028 0 0 0.028 0" size="0.020" material="joint_mat" mass="0.05"/>
            <geom type="capsule" fromto="0 0 0 0 0 {C.FOREARM_LEN}" size="0.015" mass="0.18"/>

            <body name="wrist" pos="0 0 {C.FOREARM_LEN}">
              <!-- JOINT 4 "wrist_pitch": tilts the gripper up/down. -->
              <joint name="wrist_pitch" type="hinge" axis="0 1 0" range="{lim['wrist_pitch'][0]} {lim['wrist_pitch'][1]}"/>
              <geom type="cylinder" fromto="0 -0.022 0 0 0.022 0" size="0.016" material="joint_mat" mass="0.03"/>
              <geom type="capsule" fromto="0 0 0 0 0 {wrist_stub}" size="0.012" mass="0.05"/>

              <body name="hand" pos="0 0 {wrist_stub}">
                <!-- JOINT 5 "wrist_roll": twists the gripper about its own axis. -->
                <joint name="wrist_roll" type="hinge" axis="0 0 1" range="{lim['wrist_roll'][0]} {lim['wrist_roll'][1]}" damping="0.1" armature="0.005"/>
                <geom name="palm" type="box" pos="0 0 0.01" size="0.016 0.05 0.008" material="arm_dark" mass="0.08"/>

                <!-- A "site" is a marker with no mass or shape: a named point
                     we can ask MuJoCo the position of.  This one is the TCP. -->
                <site name="tcp" pos="0 0 {tcp_z}" size="0.004" rgba="1 1 0 0.8"/>

                <!-- The two fingers.  Each slides along the hand's y axis.
                     Joint value = distance of the finger's inner face from the
                     centre line, so gap between fingers = 2 x joint value. -->
                <body name="finger_left" pos="0 0 {finger_z}" childclass="finger">
                  <joint name="finger_left" axis="0 1 0"/>
                  <geom name="pad_left" type="box" pos="0 0.004 0" size="0.011 0.004 {finger_half_len}" mass="0.02"/>
                </body>
                <body name="finger_right" pos="0 0 {finger_z}" childclass="finger">
                  <joint name="finger_right" axis="0 -1 0"/>
                  <geom name="pad_right" type="box" pos="0 -0.004 0" size="0.011 0.004 {finger_half_len}" mass="0.02"/>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>

    <!-- ============================ BLOCKS ============================= -->{block_xml}
  </worldbody>

  <!-- Force the two fingers to mirror each other, like a real parallel
       gripper where one motor drives both jaws through a gear. -->
  <equality>
    <joint joint1="finger_right" joint2="finger_left" polycoef="0 1 0 0 0"/>
  </equality>

  <!-- Motors.  ctrl (the number we send) is a torque in N.m for the hinge
       joints and a force in N for the fingers; ctrlrange is the motor limit. -->
  <actuator>
    <motor name="m_pan" joint="pan" ctrlrange="-{tq['pan']} {tq['pan']}"/>
    <motor name="m_shoulder" joint="shoulder" ctrlrange="-{tq['shoulder']} {tq['shoulder']}"/>
    <motor name="m_elbow" joint="elbow" ctrlrange="-{tq['elbow']} {tq['elbow']}"/>
    <motor name="m_wrist_pitch" joint="wrist_pitch" ctrlrange="-{tq['wrist_pitch']} {tq['wrist_pitch']}"/>
    <motor name="m_wrist_roll" joint="wrist_roll" ctrlrange="-{tq['wrist_roll']} {tq['wrist_roll']}"/>
    <motor name="m_finger_left" joint="finger_left" ctrlrange="-{C.GRIP_FORCE} {C.GRIP_FORCE}"/>
    <motor name="m_finger_right" joint="finger_right" ctrlrange="-{C.GRIP_FORCE} {C.GRIP_FORCE}"/>
  </actuator>
</mujoco>
"""


class Sim:
    """A thin, friendly wrapper around MuJoCo's model + data.

    Everything above this layer talks to the world ONLY through this class:
    read sensors (joint angles, positions), write motor commands, step time.
    That is exactly the interface a real robot gives you, which is what makes
    it possible to later swap the simulator for real hardware.
    """

    def __init__(self, block_positions: dict[str, tuple[float, float]] | None = None):
        self.model = mujoco.MjModel.from_xml_string(build_mjcf(block_positions))
        self.data = mujoco.MjData(self.model)

        # MuJoCo stores all joint angles in one long array, data.qpos, and all
        # joint velocities in data.qvel.  These lines look up WHERE in those
        # arrays each of our joints lives, once, so later code can be simple.
        self._arm_qpos = [self.model.joint(n).qposadr[0] for n in C.ARM_JOINTS]
        self._arm_qvel = [self.model.joint(n).dofadr[0] for n in C.ARM_JOINTS]
        self._finger_qpos = self.model.joint("finger_left").qposadr[0]
        self._finger_qvel = self.model.joint("finger_left").dofadr[0]
        self._tcp_id = self.model.site("tcp").id

        self._renderers: dict[tuple[int, int], mujoco.Renderer] = {}
        # Functions to call after every physics step (used to record video or
        # refresh the live viewer).  Each receives this Sim object.
        self.step_listeners: list = []

        self.reset()

    # ------------------------------------------------------------------ time
    def reset(self) -> None:
        """Put the arm in the home pose with the gripper open; restart the clock."""
        mujoco.mj_resetData(self.model, self.data)
        for adr, q in zip(self._arm_qpos, C.HOME_Q):
            self.data.qpos[adr] = q
        self.data.qpos[self._finger_qpos] = C.FINGER_OPEN
        self.data.qpos[self.model.joint("finger_right").qposadr[0]] = C.FINGER_OPEN
        # mj_forward computes everything that depends on the joint angles
        # (positions of all bodies etc.) WITHOUT advancing time.
        mujoco.mj_forward(self.model, self.data)

    def step(self) -> None:
        """Advance physics by one timestep (config.TIMESTEP seconds)."""
        mujoco.mj_step(self.model, self.data)
        for listener in self.step_listeners:
            listener(self)

    @property
    def time(self) -> float:
        return float(self.data.time)

    # --------------------------------------------------------------- sensing
    def arm_q(self) -> np.ndarray:
        """Current angles of the 5 arm joints (radians)."""
        return self.data.qpos[self._arm_qpos].copy()

    def arm_qvel(self) -> np.ndarray:
        """Current angular velocities of the 5 arm joints (radians/second)."""
        return self.data.qvel[self._arm_qvel].copy()

    def finger_pos(self) -> float:
        """Distance of each finger from the centre line; gap = 2 x this."""
        return float(self.data.qpos[self._finger_qpos])

    def finger_vel(self) -> float:
        return float(self.data.qvel[self._finger_qvel])

    def tcp_pos(self) -> np.ndarray:
        """Where the gripper's tool centre point is right now, world frame."""
        return self.data.site_xpos[self._tcp_id].copy()

    def gravity_torques(self) -> np.ndarray:
        """The torque each arm joint needs just to hold still against gravity.

        MuJoCo computes this for free ("qfrc_bias").  A real robot would get it
        from a mathematical model of itself.  The controller can add it to its
        output so the PID part only has to fix what is left over.
        """
        return self.data.qfrc_bias[self._arm_qvel].copy()

    def block_names(self) -> list[str]:
        return list(C.BLOCK_COLORS)

    def block_pos(self, name: str) -> np.ndarray:
        """TRUE position of a block's centre.  This is "ground truth": perfect
        knowledge that only a simulator can give.  A real robot has to estimate
        it with a camera (see perception.py)."""
        return self.data.body(f"block_{name}").xpos.copy()

    def block_upright_error(self, name: str) -> float:
        """How tilted a block is, in radians (0 = lying flat)."""
        zaxis = self.data.body(f"block_{name}").xmat.reshape(3, 3)[:, 2]
        return float(np.arccos(np.clip(abs(zaxis[2]), -1.0, 1.0)))

    # --------------------------------------------------------------- acting
    def set_arm_torques(self, tau: np.ndarray) -> None:
        """Send one torque per arm joint to the motors (clipped to motor limits)."""
        self.data.ctrl[:5] = tau   # MuJoCo itself clips to each ctrlrange

    def set_finger_force(self, force: float) -> None:
        """Positive pushes the fingers apart (open), negative squeezes (close)."""
        self.data.ctrl[5] = force
        self.data.ctrl[6] = force

    # ------------------------------------------------------------- rendering
    def _renderer(self, width: int, height: int) -> mujoco.Renderer:
        key = (width, height)
        if key not in self._renderers:
            self._renderers[key] = mujoco.Renderer(self.model, height=height, width=width)
        return self._renderers[key]

    def render(self, camera: str = "overview", width: int = 480, height: int = 360) -> np.ndarray:
        """Take a picture.  Returns an array of shape (height, width, 3), 0-255."""
        r = self._renderer(width, height)
        r.update_scene(self.data, camera=camera)
        return r.render().copy()

    def render_depth(self, camera: str = "topdown", width: int = 320, height: int = 320) -> np.ndarray:
        """A "depth image": each pixel holds the distance (m) from the camera
        to whatever it sees, instead of a colour."""
        r = self._renderer(width, height)
        r.enable_depth_rendering()
        r.update_scene(self.data, camera=camera)
        depth = r.render().copy()
        r.disable_depth_rendering()
        return depth

    def close(self) -> None:
        for r in self._renderers.values():
            r.close()
        self._renderers.clear()
