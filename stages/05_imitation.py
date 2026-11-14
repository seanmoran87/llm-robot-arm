"""STAGE 5 - imitation learning in miniature (no LLM, no GPU, ~20 seconds).

So far every motion came from maths we wrote (inverse kinematics).  The other
way to get robot behaviour is to LEARN it from examples: show the robot what
to do ("demonstrations") and let it copy.  That is imitation learning, and it
is how VLA models (vision-language-action models) are trained - at vastly
larger scale.

The task: move the gripper to a target point above the table.

  1. An "expert" (our own controller) performs the reach many times.  We
     record pairs of  OBSERVATION (where the gripper is, where the target is)
     and ACTION (which way the expert moved next).
  2. The learned "policy" is the simplest one possible: faced with a new
     observation, find the most similar recorded observations and copy the
     average of their actions ("nearest neighbours").  No neural network -
     but the same idea, and the same weaknesses.
  3. We test it on targets like the ones it saw (LEFT half of the table) and
     on targets it never saw (RIGHT half).

    python stages/05_imitation.py
    python stages/05_imitation.py --demos 3        (try 1, 3, 10, 50)

PREDICT: how many demonstrations until it succeeds on the left?  And what
happens on the right half, which no demonstration ever visited?
"""
import argparse

import numpy as np
import _common  # noqa: F401

from robot_sim import config as C
from robot_sim.controller import ArmController
from robot_sim.kinematics import UnreachableError, forward_kinematics, inverse_kinematics
from robot_sim.scene import Sim

parser = argparse.ArgumentParser()
parser.add_argument("--demos", type=int, default=200)
parser.add_argument("--trials", type=int, default=20)
args = parser.parse_args()

rng = np.random.default_rng(0)        # fixed seed: same "random" numbers every run
STEP = 0.01                           # the policy moves the target 1 cm per decision
HOME = forward_kinematics(C.HOME_Q)
FAR_AWAY = {"red": (0.10, -0.36), "green": (0.14, -0.36), "blue": (0.18, -0.36)}


def random_target(side: str) -> np.ndarray:
    """A random reachable point; side 'left' has y > 0, 'right' has y < 0."""
    y = rng.uniform(0.04, 0.22) * (1 if side == "left" else -1)
    return np.array([rng.uniform(0.18, 0.30), y, rng.uniform(0.06, 0.18)])


def expert_action(tcp: np.ndarray, target: np.ndarray) -> np.ndarray:
    """What the expert does: step straight towards the target, 1 cm at a time."""
    to_go = target - tcp
    dist = np.linalg.norm(to_go)
    return to_go if dist < STEP else to_go / dist * STEP


# ---- 1. collect demonstrations (left half of the table only) --------------
observations, actions = [], []
for _ in range(args.demos):
    target, tcp = random_target("left"), HOME.copy()
    for _ in range(80):
        a = expert_action(tcp, target)
        observations.append(np.concatenate([tcp, target]))
        actions.append(a)
        tcp = tcp + a
observations, actions = np.array(observations), np.array(actions)
print(f"Recorded {args.demos} demonstrations = {len(observations)} (observation, action) pairs, "
      f"all with targets on the LEFT (y > 0).\n")


# ---- 2. the learned policy -------------------------------------------------
def policy(tcp: np.ndarray, target: np.ndarray, k: int = 5) -> np.ndarray:
    """Copy the average action of the k most similar recorded situations."""
    obs = np.concatenate([tcp, target])
    distance = np.linalg.norm(observations - obs, axis=1)
    nearest = np.argsort(distance)[:k]
    return actions[nearest].mean(axis=0)


# ---- 3. run the policy on the simulated robot ------------------------------
def run_trial(target: np.ndarray) -> float:
    """Let the policy drive the real (simulated) arm; return final miss in metres."""
    sim = Sim(FAR_AWAY)
    ctrl = ArmController(sim)
    goal = HOME.copy()                       # the point we ask the controller to track
    for _ in range(80):
        goal = goal + policy(sim.tcp_pos(), target)
        try:
            ctrl.q_target = inverse_kinematics(*goal)
        except UnreachableError:
            break                            # the policy wandered out of reach
        ctrl.hold(0.04)
    miss = float(np.linalg.norm(sim.tcp_pos() - target))
    sim.close()
    return miss


for side, label in (("left", "LEFT  (like the demonstrations)"), ("right", "RIGHT (never demonstrated)   ")):
    misses = np.array([run_trial(random_target(side)) for _ in range(args.trials)])
    ok = (misses < 0.02).sum()
    print(f"{label}: reached within 2 cm in {ok:>2}/{args.trials} trials;   "
          f"typical miss {np.median(misses) * 1000:6.1f} mm")
