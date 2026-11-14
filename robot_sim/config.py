"""All the numbers that describe the robot and its world, in one place.

Why a separate file?  Several layers need the SAME numbers.  The physics scene
needs the link lengths to build the arm, and the kinematics code needs the same
lengths to work out where the gripper is.  If the two ever disagreed, the robot
would confidently reach for the wrong spot.  Keeping one copy prevents that.

UNITS (the same everywhere in this project, and in MuJoCo):
    length  -> metres       (0.04 means 4 cm)
    angle   -> radians      (3.1416 rad = 180 degrees)
    mass    -> kilograms
    time    -> seconds
    force   -> newtons,  torque (turning force) -> newton-metres

COORDINATE FRAME ("frame" = a choice of origin and axis directions):
    The "world frame" has its origin on the table top, directly under the
    robot's base.   +x points forward (away from the robot),
                    +y points to the robot's left,
                    +z points up.   So z = 0 is the table surface.
"""

import math

# --------------------------------------------------------------------------
# Arm geometry
# --------------------------------------------------------------------------
SHOULDER_HEIGHT = 0.10   # height of the shoulder joint above the table
UPPER_ARM_LEN = 0.22     # shoulder joint -> elbow joint
FOREARM_LEN = 0.22       # elbow joint    -> wrist joint
TOOL_LEN = 0.10          # wrist joint    -> TCP

# TCP = "tool centre point": the single point we steer around.  Here it is the
# spot midway between the two finger pads, i.e. where a grasped block's centre
# ends up.
FINGERTIP_BELOW_TCP = 0.015   # the fingers extend this far past the TCP

# Joint names, in order from the base outwards.  "DOF" = degree of freedom =
# one independent way the robot can move.  This arm has 5 DOF plus a gripper,
# the same layout as the LeRobot SO-101.
ARM_JOINTS = ["pan", "shoulder", "elbow", "wrist_pitch", "wrist_roll"]

# Joint limits (radians): how far each joint is allowed to turn.
JOINT_LIMITS = {
    "pan":         (-2.0, 2.0),
    "shoulder":    (-1.6, 1.8),
    "elbow":       (0.0, 2.8),
    "wrist_pitch": (-0.6, 3.0),
    "wrist_roll":  (-2.0, 2.0),
}

# Largest torque each motor can produce (newton-metres).  Real motors have
# limits too; the controller's output gets clipped to these.
TORQUE_LIMITS = {
    "pan": 6.0, "shoulder": 10.0, "elbow": 6.0,
    "wrist_pitch": 3.0, "wrist_roll": 1.0,
}

# "Home" pose: upper arm straight up, forearm pointing forward, gripper
# pointing straight down.  A safe, known pose to start and finish in.
HOME_Q = [0.0, 0.0, math.pi / 2, math.pi / 2, 0.0]

# --------------------------------------------------------------------------
# Gripper
# --------------------------------------------------------------------------
FINGER_OPEN = 0.035       # each finger sits 3.5 cm from centre -> 7 cm gap
FINGER_CLOSED = 0.0
GRIP_FORCE = 6.0          # newtons each finger can squeeze with

# --------------------------------------------------------------------------
# Blocks and table
# --------------------------------------------------------------------------
BLOCK_SIZE = 0.04         # edge length of each cube
BLOCK_MASS = 0.05         # 50 grams
BLOCK_COLORS = {          # name -> (red, green, blue, opacity), each 0..1
    "red":   (0.85, 0.15, 0.15, 1.0),
    "green": (0.15, 0.70, 0.20, 1.0),
    "blue":  (0.15, 0.30, 0.90, 1.0),
}
BLOCK_START = {           # name -> (x, y) of the block centre on the table
    "red":   (0.24, -0.12),
    "green": (0.30, 0.00),
    "blue":  (0.24, 0.12),
}

# --------------------------------------------------------------------------
# Workspace: the region the gripper is ALLOWED to go.  The safety validator
# enforces this.  It is deliberately a bit smaller than what the arm can
# physically reach, the same way a speed limit is below a car's top speed.
# --------------------------------------------------------------------------
WORKSPACE_R_MIN = 0.14    # closest to the base (horizontal distance from it)
WORKSPACE_R_MAX = 0.36    # farthest from the base
WORKSPACE_Z_MIN = 0.018   # lowest TCP height (finger tips just clear the table)
WORKSPACE_Z_MAX = 0.24    # highest TCP height
WORKSPACE_ANGLE_MAX = 1.75  # max sideways angle from straight ahead (~100 deg)

TRANSIT_Z = 0.20          # height the gripper travels at when carrying things

# Where the arm parks (off to the left side) when the camera needs a clear
# view of the table.  Only used with --perception camera.
LOOK_POSE_R = 0.16
LOOK_POSE_ANGLE = 1.70

# --------------------------------------------------------------------------
# Simulation timing
# --------------------------------------------------------------------------
TIMESTEP = 0.002          # physics advances 2 ms per step = 500 steps/second
