# Code walkthrough

This guide explains the simulator in the order that makes it easiest to
understand: from the bottom (physics) up to the top (language). Each section
names the file, says what problem it solves, and points at the lines worth
reading slowly. Terms in **bold** are defined in the [glossary](#glossary).

Every command and number in this guide was produced by actually running the code.

## 1. The big picture

A robot program is a stack of layers. Each layer turns a *vaguer, slower*
instruction into a *more precise, faster* one for the layer below.

```
  You:  "stack red on blue"
            │
            ▼
┌──────────────────────────────────────────────────────────────┐
│ (d) PLANNER             llm_planner.py                        │  decides WHAT
│     LLM (or offline stand-in) picks the next skill to call    │  ~1 decision / few seconds
└──────────────────────────────────────────────────────────────┘
            │  tool call:  place(x=0.24, y=0.12, z=0.06)       ▲ result: what was measured
            ▼                                                  │
┌──────────────────────────────────────────────────────────────┐
│ EXECUTOR                executor.py                           │
│   1. get_scene()   ─────────────►  where is everything now?   │
│   2. VALIDATOR     validator.py    is this safe?  ── no ──► rejected, reason sent back up
│   3. run the skill (only if approved)                         │
│   4. RUN LOG       runlog.py       plan | verdict | actual    │
└──────────────────────────────────────────────────────────────┘
            │  skill call
            ▼
┌──────────────────────────────────────────────────────────────┐
│ (c) SKILLS              skills.py                             │  decides HOW, in steps
│     place = fly over → descend → open → rise → check          │  ~5 motions / skill
└──────────────────────────────────────────────────────────────┘
            │  move_tcp(x, y, z), set_gripper(closed)
            ▼
┌──────────────────────────────────────────────────────────────┐
│ (b) CONTROLLER          controller.py  (+ kinematics.py)      │  decides torques
│     target point → joint angles (IK) → smooth path → PID      │  500 times / second
└──────────────────────────────────────────────────────────────┘
            │  7 motor torques              ▲ joint angles, speeds
            ▼                               │
┌──────────────────────────────────────────────────────────────┐
│ (a) PHYSICS SCENE       scene.py   [MuJoCo]                   │  what really happens
│     arm, table, 3 blocks, gravity, friction, cameras          │  2 ms per step
└──────────────────────────────────────────────────────────────┘
            │ camera images (optional)
            └────────►  perception.py  ────►  block positions for get_scene()
```

Two things to notice:

- **Information flows both ways.** Commands go down; measurements come back up.
  Every layer checks what really happened rather than assuming success.
- **The LLM is at the top and far from the motors.** It makes a few decisions
  per task. The controller makes 500 per second. Section 7 explains why they
  must not be swapped.

`run_demo.py` wires this up in one function, `build_robot()`. Read it first: it
is twenty lines and it *is* the diagram above.

## 2. `config.py` — the numbers

Start here because every other file imports it. It holds link lengths, joint
limits, block size, the allowed workspace, and the timestep.

Read the docstring at the top for two conventions used everywhere:

- **Units**: metres, radians, kilograms, seconds.
- **World frame**: origin on the table under the robot's base; +x forward, +y to
  the robot's left, +z up. So `z = 0` is the table top, and a 4 cm block lying
  on the table has its centre at `z = 0.02`.

## 3. `scene.py` — layer (a), the physics scene

**Problem it solves:** describe a world and let physics run in it.

- `build_mjcf()` returns a long XML string in **MJCF**, MuJoCo's scene format.
  Read the `<worldbody>` part top to bottom: it is a tree. The table is in the
  world; the arm's `base` is in the world; `turret` is inside `base`;
  `upper_arm` inside `turret`; and so on to the fingers. Moving a parent moves
  everything nested inside it, which is exactly how an arm behaves.
- Each `<joint>` line is one **degree of freedom**. There are five hinge joints
  in the arm plus two sliding fingers.
- The `<actuator>` block gives each joint a motor that accepts a **torque**.
- The `<default>` block explains collision groups: arm parts collide with blocks
  and the table but not with each other.
- The `Sim` class is the only thing the upper layers touch. Its methods fall into
  three groups: *sensing* (`arm_q`, `tcp_pos`, `block_pos`), *acting*
  (`set_arm_torques`, `set_finger_force`), and *time* (`step`). A real robot's
  driver offers the same three groups, which is why the layers above could later
  be pointed at hardware.

Run it alone:

```powershell
python stages/01_scene.py --viewer
```

With all motors at zero torque the gripper falls from z = 0.220 m to 0.016 m in
under half a second. Nothing holds a robot up except its controller.

## 4. `kinematics.py` — where is the gripper?

**Problem it solves:** translate between joint angles and gripper position.

- `forward_kinematics(q)` adds up the three links like directions on a map.
- `inverse_kinematics(x, y, z)` goes the other way, in four commented steps:
  face the target (pan), solve a triangle for the elbow (law of cosines), tilt
  the wrist so the gripper points straight down, and roll the wrist to keep the
  fingers square to the blocks.
- It raises `UnreachableError` when the triangle cannot be formed. Callers above
  rely on this to refuse impossible moves *before* moving.

```powershell
python stages/00_kinematics.py 0.30 0.10 0.05
```

prints the five joint angles (pan 18.4°, shoulder 37.7°, elbow 86.6°, wrist
pitch 55.7°, roll 18.4°) and confirms the round trip lands back on the target.
Note that shoulder + elbow + wrist pitch is always 180°: that is the "gripper
points down" rule.

## 5. `controller.py` — layer (b), the low-level controller

**Problem it solves:** the motors want torques; everyone else wants positions.

- `ArmController.step()` is the heart: read angles, compute the error, compute
  **PID** torques, add **gravity compensation**, send to motors, step physics.
  It is about fifteen lines. Everything else in the project ultimately calls it.
- `min_jerk()` is the smooth 0-to-1 ramp that turns "be there" into "get there
  gently".
- `move_tcp()` plans the whole path first (so an unreachable point is found
  before the arm moves), then feeds one target per timestep to `step()`.
- The gripper is simpler: push with a constant force. When the pads meet a
  block they stop; the squeeze is the grip.

```powershell
python stages/02_controller.py --step
```

## 6. `skills.py` — layer (c), the skill library

**Problem it solves:** give the planner a small vocabulary of reliable verbs.

- `grasp(block)`: open → rise to travel height → fly above the block → descend →
  close → rise → **check the finger gap**. If the fingers stopped about one
  block-width apart, something is in them. If they closed fully, the grasp
  missed, and the skill says so.
- `place(x, y, z)`: fly over → descend to 2 mm above → open → rise → look at
  where the block really ended up and report the error in millimetres.
- Every skill returns a dictionary with `success` and measurements. Nothing
  returns "done" on faith.

```powershell
python stages/03_skills.py --viewer
```

## 7. `validator.py` and `executor.py` — the safety gate

**Problem it solves:** an LLM can be confidently wrong, and wrong on a robot
breaks things.

`executor.py` is short. `SafeExecutor.execute()` is the only path from a planner
to a skill, and it always does the same four things: read the scene, ask the
validator, run the skill if approved, write the log.

`validator.py` holds the rules. They are plain `if` statements about physical
facts, with no intelligence:

| Rule | Example it blocks |
|---|---|
| Known tool, finite numbers | `launch_missiles()`, `x = NaN` |
| Inside the workspace and reachable | `place(x=0.6, …)` |
| Whole path clears every block | `move_to(0.30, 0.0, 0.03)` over the green block |
| Right state | `place` while holding nothing; `grasp` while holding |
| Nothing on top of the block to grasp | grasping blue while red sits on it |
| Supported and unoccupied place target | releasing in mid-air; placing inside a block |
| Action budget | a plan that loops forever (30 actions max) |

A rejection is returned to the planner as an error message in plain English, so
an LLM can read it and try something better.

**Why the LLM plans but does not control.** An LLM reply takes around a second
and is not guaranteed to be right. The controller needs a correct answer every
2 milliseconds. So the LLM chooses among a few safe verbs, slowly, and
ordinary deterministic code does everything fast or dangerous.

## 8. `llm_planner.py` — layer (d), the LLM

**Problem it solves:** turn a sentence into skill calls.

- `TOOLS` is the menu the LLM sees: five tools with descriptions and an exact
  argument shape (**JSON Schema**). The descriptions are the LLM's only manual,
  so they state units and how to compute a stacking height.
- `SYSTEM_PROMPT` gives the standing instructions: the coordinate frame, that a
  validator exists, and to trust measurements over its own plan.
- `AnthropicPlanner.run()` is the **function-calling** loop: send the
  conversation → the model answers with tool requests → run each through the
  executor → send the results back → repeat until the model says it is finished.
  Each action tool has a `reason` field, so the log records *why* the model
  chose each step.
- `RuleBasedPlanner` is the offline stand-in: pattern matching on a few fixed
  phrasings. It exists so you can run everything without a key, and as a
  contrast. It fails on any rewording and stops at the first problem.

## 9. `perception.py`, `runlog.py`, `viz.py`

- `perception.py` finds blocks from a top-down colour + depth image: pick pixels
  by colour, keep the highest ones (the top face), convert pixels to metres.
  Measured accuracy here: about 0.3 mm from the truth.
- `runlog.py` writes the plan-vs-actual table to the terminal, `log.jsonl`, and
  `report.html`.
- `viz.py` has the live window and the GIF recorder. Both simply listen to
  `sim.step()`; the control code does not know it is being watched.

## 10. One command, traced end to end

`python run_demo.py "stack red on blue"`

1. `build_robot()` creates Sim → ArmController → Skills → SafeExecutor → planner.
2. The planner calls `get_scene`. Result: red at (0.24, −0.12, 0.02), blue at
   (0.24, 0.12, 0.02), holding nothing.
3. The planner calls `grasp(block=red)`. Validator: red exists, hand is empty,
   nothing is on red, it is reachable → approved. The skill runs four motions
   and a squeeze; the controller runs a few thousand PID cycles. Measured: fingers
   stopped 37.8 mm apart → holding red.
4. The planner calls `place(x=0.24, y=0.12, z=0.06)`. That z is blue's 0.02 plus
   one block height, 0.04. Validator: holding something, the spot is free, and
   blue is squarely underneath → approved. Measured afterwards: red at
   (0.242, 0.120, 0.060), 2.2 mm from the target.
5. `go_home`, then the planner's closing message.
6. `report.html` shows those rows with a snapshot beside each.

## 11. Five experiments

Do them in order. For each, write down your prediction first.

**1. Rephrase the command.**
```powershell
python run_demo.py "please pile the red one onto the blue one"
```
Offline planner: *"Sorry, I only understand a few fixed phrasings."* With an API
key set, run the same line again. This gap is the reason to use an LLM.

**2. Ask for something unsafe.**
```powershell
python run_demo.py "put red at 0.6 0.0"
```
```powershell
python run_demo.py "move to 0.30 0.0 0.03"
```
The first is rejected at the `place` step (0.600 m from the base; allowed
0.14–0.36 m). The second is rejected because the path would hit the green block.
In both the robot does not move for the rejected step. Then open
`robot_sim/validator.py`, find `_check_workspace`, and read which line said no.

**3. Break the controller on purpose.**
```powershell
python stages/02_controller.py --step
```
```powershell
python stages/02_controller.py --step --kd-scale 0
```
```powershell
python stages/02_controller.py --step --kp-scale 0.1
```
```powershell
python stages/02_controller.py --step --ki-scale 0 --no-gravity-comp
```
Measured results:

| Setting | Overshoot | Settling time | Final error |
|---|---|---|---|
| default | 0.2 % | 0.26 s | 0.62 mrad |
| no D term (`--kd-scale 0`) | 29.2 % | 0.72 s | 0.32 mrad |
| weak P (`--kp-scale 0.1`) | 0 % | 1.92 s | 3.21 mrad |
| no I and no gravity compensation | 0 % | 0.30 s | 3.66 mrad |

Without D the joint swings past the target. With weak P it is sluggish. Without
I or gravity compensation it stops short and stays there, sagging under its own
weight.

**4. Weaken the gripper.** In `robot_sim/config.py` change `GRIP_FORCE = 6.0` to
`GRIP_FORCE = 0.1`, then run `python stages/03_skills.py`. The block slips out
as the arm lifts, the fingers close fully, and `grasp` reports *"fingers closed
on nothing - grasp missed"*. Try `0.3`: it holds. Friction force = grip force ×
friction coefficient, and it has to beat the block's weight (0.05 kg × 9.81 ≈
0.5 N). Put the value back to `6.0` afterwards.

**5. Give the robot bad eyes.**
```powershell
python stages/06_sim_to_real.py
```
Adds random error to the block positions the robot is told. Measured:

| Position error | Stacks that worked |
|---|---|
| 0 mm | 10/10 |
| 2 mm | 10/10 |
| 5 mm | 9/10 |
| 10 mm | 6/10 |
| 15 mm | 5/10 |
| 20 mm | 4/10 |

Nothing about the arm changed. Only what it believed about the world.

## 12. Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `python` is not recognised | Python is not on PATH. Reinstall with "Add python.exe to PATH" ticked, or use `py` instead of `python`. |
| `Activate.ps1 cannot be loaded because running scripts is disabled` | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate again. |
| `ModuleNotFoundError: No module named 'mujoco'` | The virtual environment is not active (no `(.venv)` in the prompt). Run `.\.venv\Scripts\Activate.ps1`. |
| `ModuleNotFoundError: No module named 'robot_sim'` | You are not in the project folder. `cd` into `llm-robot-arm` first. |
| `--viewer` window does not appear, or an OpenGL/GLFW error | Update your graphics driver. Over Remote Desktop, OpenGL is often unavailable; run without `--viewer` and open `run.gif` instead. |
| Program ends with a crash message after the report is written | Shutdown-order problem with the window. Results are already saved. Please note what you ran so it can be fixed. |
| "ANTHROPIC_API_KEY is not set" note | Expected without a key; the offline planner is used. See README, "Using the real LLM". |
| `The Anthropic API rejected the API key` | The key was mistyped, or set in a different PowerShell window. Set `$env:ANTHROPIC_API_KEY` again in this window. |
| `Anthropic API error 400` mentioning `fallbacks` or `betas` | Your account does not accept the optional fallback feature. In `robot_sim/llm_planner.py`, function `_ask`, delete the two lines `betas=[…]` and `fallbacks="default"`, and change `self.client.beta.messages.create` to `self.client.messages.create`. |
| A step shows `REJECTED` | Working as designed. The reason is printed underneath; the matching rule is in `validator.py`. |
| `grasp` reports "fingers closed on nothing" | The block was not where the robot believed (try `--perception truth`), or `GRIP_FORCE` / `BLOCK_MASS` in `config.py` were changed. |
| `--perception camera` says "cannot see a block" | Something is covering it, or its colour in `config.py` is no longer clearly red, green or blue. |
| The arm shakes | `TIMESTEP` in `config.py` was raised (at 0.01 the arm misses targets by about 8 mm), or the PID gains in `controller.py` were changed. Restore the defaults. |
| Everything is odd after experimenting | Restore the original files: re-extract the project, or `git checkout .` if you use git. |

## Glossary

- **Actuator** — a motor: the part that produces force or torque.
- **API key** — a password that lets a program use a paid online service.
- **Degree of freedom (DOF)** — one independent way a robot can move; usually one per joint.
- **Depth image** — a picture whose pixels store distance from the camera instead of colour.
- **End effector** — whatever is at the end of the arm; here, the gripper.
- **Feedback** — using measurements of what happened to correct what you do next.
- **Feedforward** — a correction computed in advance from a model, such as gravity compensation.
- **Forward kinematics (FK)** — computing the gripper position from the joint angles.
- **Frame (coordinate frame)** — an origin plus axis directions that positions are measured in.
- **Function calling (tool use)** — an LLM replying with a structured request to run a named function, instead of prose.
- **Gravity compensation** — adding the torque needed to hold the arm's own weight.
- **Ground truth** — the exact true state, available only in simulation.
- **Imitation learning** — learning a behaviour by copying demonstrations.
- **Inverse kinematics (IK)** — computing joint angles that put the gripper at a wanted position.
- **Joint** — a connection that lets two rigid parts move relative to each other (hinge = rotate, slide = translate).
- **JSON Schema** — a standard way to describe the shape of structured data.
- **LLM** — large language model; a program trained on text that predicts useful continuations.
- **MJCF** — MuJoCo's XML file format for describing a scene.
- **MuJoCo** — the physics engine used here.
- **Perception** — turning raw sensor data into facts about the world.
- **PID** — a controller that sums terms proportional to the error, its accumulated history (integral), and its rate of change (derivative).
- **Policy** — any rule that maps what the robot observes to what it does next.
- **Radian** — unit of angle; π radians = 180°.
- **Sim-to-real gap** — the differences between simulation and reality that make simulated success fail on hardware.
- **Skill** — a named, reusable behaviour such as `grasp`.
- **TCP (tool centre point)** — the single point on the gripper that is steered around.
- **Timestep** — how much simulated time one physics step covers (2 ms here).
- **Torque** — turning force, in newton-metres.
- **Trajectory** — a path with timing: where to be at each moment.
- **Validator** — deterministic code that approves or rejects each planned action.
- **VLA (vision-language-action model)** — a neural network that takes camera images and a text instruction and outputs robot actions directly.
- **Workspace** — the region the gripper can (or is allowed to) reach.
