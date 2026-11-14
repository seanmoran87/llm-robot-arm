# Robotics from zero: an 8-module intro course

Built around the simulator in this folder. Each module takes **1–2 hours** and
follows the same shape:

- **Hook** — a question worth an hour of your time
- **Analogy** — the idea in everyday terms
- **Core ideas** — the minimum to know, with jargon defined on first use
- **Predict, then run it** — write your prediction down *before* running. Being
  wrong on paper first is what makes the result stick.
- **Check yourself** — three questions; answers at the end of each module

At about 5 hours a week this is a 3–4 week course. You need the simulator
installed (see `README.md`). No GPU is needed for anything here. All commands
are run from the project folder with the virtual environment active, and every
result quoted below came from actually running that command.

| # | Module | Time | You will run |
|---|---|---|---|
| 1 | Robot basics: sense–think–act | 1 h | `stages/01_scene.py` |
| 2 | Kinematics and coordinate frames | 1.5 h | `stages/00_kinematics.py` |
| 3 | Physics simulation | 1.5 h | `stages/03_skills.py` with edited `config.py` |
| 4 | Control: PID and trajectories | 2 h | `stages/02_controller.py --step` |
| 5 | Perception | 1.5 h | `run_demo.py --perception camera` |
| 6 | Where LLMs fit: planner vs. controller | 1.5 h | `run_demo.py` |
| 7 | VLA models and imitation learning | 2 h | `stages/05_imitation.py` |
| 8 | Safety, failure modes, sim-to-real | 1.5 h | `stages/06_sim_to_real.py` |

---

## Module 1 — Robot basics: sense, think, act

**Hook.** A robot arm is switched on but nobody is controlling it. What does it do?

**Analogy.** Catching a ball. Your eyes track it (sense), your brain predicts
where it will be (think), your arm moves (act), and your eyes immediately check
how that went. A robot is this loop built from parts, running many times a second.

**Core ideas.**
- A robot is a machine that **senses** the world, **decides**, and **acts** on
  it, in a loop. Take away any one of the three and it is something else: a
  camera, a calculator, or a motor.
- **Sensors** measure: joint angle sensors (called *encoders*), cameras.
  **Actuators** move: motors. The **controller** is the program between them.
- **Feedback** means using the measurement of what just happened to choose what
  to do next. Without it ("open loop") a robot is doing choreography with its
  eyes shut.
- A **joint** lets two rigid parts move relative to each other. Each independent
  motion is a **degree of freedom (DOF)**. This arm has 5 DOF plus a gripper:
  the same layout as the SO-101 you are aiming for.
- The loop exists at several speeds at once. In this project: the LLM loop turns
  once every few seconds, the skill loop a few times per task, the motor loop
  500 times a second. Open `run_demo.py` and read `build_robot()` to see them
  stacked.

**Predict, then run it.**
The script builds the world and runs physics for 2 seconds with every motor
torque set to zero. Predict: (a) what happens to the arm, (b) roughly how long
it takes, (c) what happens to the blocks.

```powershell
python stages/01_scene.py --viewer
```

<details><summary>What happened</summary>
The gripper drops from z = 0.220 m to 0.016 m in under 0.5 s and lies on the
table; the blocks do not move (nothing touched them). Motors do not hold a pose
by themselves. "Staying still" is an active job done 500 times a second.
</details>

**Check yourself.**
1. Name the sensor, the "think" step and the actuator in a room thermostat.
2. Why is a dishwasher on a fixed timer not using feedback?
3. In `robot_sim/scene.py`, which three groups of methods does the `Sim` class offer?

<details><summary>Answers</summary>
1. Thermometer; compare with the set temperature; heater switch.
2. It never measures whether the dishes are clean; it just runs its schedule.
3. Sensing (`arm_q`, `tcp_pos`, …), acting (`set_arm_torques`, …), time (`step`).
</details>

---

## Module 2 — Kinematics and coordinate frames

**Hook.** You want the gripper exactly here. The robot only understands "turn
joint 2 by this many degrees". How do you translate?

**Analogy.** Directions versus an address. "Walk 200 m north, turn right, walk
100 m" (joint angles) always leads to one place, which is easy to work out. An
address (a target position) may be reachable by several routes or by none.
Going from directions to place is easy; going from place to directions is the
hard direction.

**Core ideas.**
- A **coordinate frame** is an origin plus three axis directions. "The block is
  at (0.24, 0.12, 0.02)" means nothing until you say in which frame. Ours, the
  *world frame*, is defined at the top of `robot_sim/config.py`. A camera has
  its own frame; so does the gripper. Much of robotics is converting between frames.
- Angles are in **radians** (π rad = 180°). Lengths are in metres.
- **Forward kinematics (FK)**: joint angles → gripper position. Always one
  answer. Read `forward_kinematics()` in `robot_sim/kinematics.py`: three links,
  three sines and cosines.
- **Inverse kinematics (IK)**: gripper position → joint angles. May have several
  answers (elbow up or down) or none (out of reach). Ours is solved with a
  triangle; read the four commented steps in `inverse_kinematics()`.
- The **workspace** is the set of points the gripper can reach.
- The **TCP** (tool centre point) is the one point on the gripper we steer.

**Predict, then run it.**
The home pose has the upper arm vertical and the forearm horizontal, putting the
TCP at (0.22, 0, 0.22). Link lengths: upper arm 0.22 m, forearm 0.22 m.

1. Predict the five joint angles for the target (0.22, 0, 0.22).
2. Predict which angles change, and how, for (0.30, 0.10, 0.05) versus (0.30, −0.10, 0.05).
3. The arm spans 0.44 m from its shoulder. Predict whether (0.50, 0, 0.05) works.

```powershell
python stages/00_kinematics.py 0.22 0 0.22
```
```powershell
python stages/00_kinematics.py 0.30 0.10 0.05
```
```powershell
python stages/00_kinematics.py 0.30 -0.10 0.05
```
```powershell
python stages/00_kinematics.py 0.50 0 0.05
```

<details><summary>What happened</summary>
1. pan 0°, shoulder 0°, elbow 90°, wrist pitch 90°, roll 0°.<br>
2. Only pan and roll flip sign (+18.4° → −18.4°). Shoulder 37.7°, elbow 86.6°
and wrist pitch 55.7° are identical: the side view of the arm is the same, it is
just turned the other way.<br>
3. Refused: "out of reach (needs 0.502 m from the shoulder, arm spans 0.440 m)".
</details>

**Check yourself.**
1. Why do shoulder + elbow + wrist pitch always sum to 180° here?
2. A point 0.30 m in front of the robot is reachable at z = 0.05. Is it
   necessarily reachable at every height? Why not?
3. Why does `inverse_kinematics` set roll equal to pan?

<details><summary>Answers</summary>
1. That sum is the gripper's direction measured from straight up; 180° is straight down.
2. No. Reach is a distance from the shoulder; going higher or lower changes that distance.
3. Turning the base also turns the fingers; rolling the wrist by the same angle
   cancels it so the fingers stay square to the blocks.
</details>

---

## Module 3 — Physics simulation

**Hook.** How can a program know that a block will slip out of a gripper?

**Analogy.** A flip-book. Each page is the world a moment later, drawn from the
previous page using fixed rules. Flip fast and it looks continuous. Draw too few
pages and the motion turns jerky and wrong.

**Core ideas.**
- A **physics engine** (here MuJoCo) repeats one step: from the current
  positions, velocities and forces, compute where everything is a **timestep**
  later (2 ms for us). Read `Sim.step()` in `robot_sim/scene.py`: it is one line.
- The world is described in a **model** (fixed: shapes, masses, joints) and
  **data** (changing: angles, velocities, time). Read `build_mjcf()`.
- **Contact** and **friction** are the hard part. Friction can resist up to
  (friction coefficient × squeezing force). A grasp holds while that exceeds
  the weight. Ours: 2 pads × 6 N × coefficient 2.0 = 24 N available, against a
  block weight of 0.05 kg × 9.81 ≈ 0.5 N.
- Simulation is an approximation. Smaller timesteps are more accurate and
  slower. Too large a timestep and things go unstable.
- Why simulate at all: it is free, fast (this one runs several times faster than
  real time), safe, and repeatable.

**Predict, then run it.** Edit `robot_sim/config.py`, run, and put the value
back each time. The script grasps red and places it on blue.

| Change | Your prediction |
|---|---|
| `BLOCK_MASS = 1.0` (20× heavier) | |
| `BLOCK_MASS = 2.5` | |
| `GRIP_FORCE = 0.1` | |
| `TIMESTEP = 0.01` (5× bigger steps) | |

```powershell
python stages/03_skills.py --viewer
```

<details><summary>What happened</summary>
<code>BLOCK_MASS = 1.0</code>: still works (weight 9.8 N &lt; 24 N of friction).<br>
<code>BLOCK_MASS = 2.5</code>: weight 24.5 N beats the friction; the block stays on
the table and <code>grasp</code> reports "fingers closed on nothing".<br>
<code>GRIP_FORCE = 0.1</code>: friction available is 0.4 N, less than the 0.5 N
weight; the grasp fails the same way.<br>
<code>TIMESTEP = 0.01</code>: the arm visibly shakes and the grasp fails. Running
<code>python stages/02_controller.py</code> with this setting shows position errors
of 4–8 mm instead of about 0.1 mm. The controller now acts only 100 times a
second with gains tuned for 500.
</details>

**Check yourself.**
1. With `GRIP_FORCE = 6.0`, what is the heaviest block the friction estimate says can be held?
2. Why not always use a tiny timestep such as 0.00001 s?
3. Name one thing that is true in this simulation and false on a real table.

<details><summary>Answers</summary>
1. 24 N / 9.81 ≈ 2.4 kg, which matches: 1.0 kg held, 2.5 kg did not.
2. 200× more computation for the same simulated second.
3. Examples: blocks are perfectly cubic and identical; friction is uniform; the
   robot knows block positions exactly; motors respond instantly.
</details>

---

## Module 4 — Control: PID and trajectories

**Hook.** You tell a motor "go to 90°". Why might it overshoot, wobble, or stop short?

**Analogy.** Steering a car back to the centre of a lane. Turn the wheel in
proportion to how far off you are (**P**). Straighten early because you can see
you are already swinging back (**D**). Hold a small permanent correction against
a steady crosswind (**I**).

**Core ideas.**
- The **error** is target minus measurement. A **PID controller** outputs
  `kp × error + ki × (sum of past error) + kd × (rate of change of error)`.
  The three numbers `kp`, `ki`, `kd` are called *gains*. Read
  `ArmController.step()` in `robot_sim/controller.py`.
- **P** does most of the work. Too low is sluggish; too high buzzes.
- **D** is a brake that prevents **overshoot** (swinging past the target).
- **I** removes **steady-state error**: a small leftover that P alone tolerates,
  such as sag under gravity. It needs limits, or it over-corrects (*integral
  windup*; see the comment in `step()`).
- **Gravity compensation** is **feedforward**: instead of waiting for the arm to
  sag and then reacting, add the torque you already know gravity demands.
- A **trajectory** is a path with timing. `min_jerk()` produces a smooth ramp so
  the target glides to its destination instead of jumping. Smooth targets are
  easy for any controller to follow, which is why engineers test with a harsh
  *step* (an instant jump) instead.

**Predict, then run it.** The elbow's target jumps by 0.5 rad. For each row
predict: overshoot (yes/no), faster or slower to settle, any leftover error.

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

<details><summary>What happened</summary>

| Setting | Overshoot | Settling time | Final error |
|---|---|---|---|
| default | 0.2 % | 0.26 s | 0.62 mrad |
| no D | 29.2 % | 0.72 s | 0.32 mrad |
| P at one tenth | 0 % | 1.92 s | 3.21 mrad |
| no I, no gravity compensation | 0 % | 0.30 s | 3.66 mrad |

Then run without `--step` and with `--kd-scale 0`: on smooth trajectories the
missing D term makes almost no difference (about 0.1 mm error either way). A
gentle test hides a bad controller.
</details>

**Check yourself.**
1. Which term would you raise if the arm reaches the target but slowly sags below it under load?
2. Why does removing D matter for a step but hardly at all for a min-jerk trajectory?
3. Gravity compensation here reads the answer from the simulator. What would a real robot need instead?

<details><summary>Answers</summary>
1. I (or add gravity compensation).
2. A step demands a violent correction that needs braking; a smooth trajectory
   never lets the error get large.
3. A mathematical model of its own link masses and lengths. Hobby servos such as
   the SO-101's skip this and rely on stiff position control inside each servo.
</details>

---

## Module 5 — Perception

**Hook.** The simulator has been telling the robot exactly where each block is.
Who tells a real robot?

**Analogy.** Finding your keys on a cluttered desk. You are not handed
coordinates. You scan for a colour and shape, estimate where they are, and
reach. If your own arm blocks the view, you move it.

**Core ideas.**
- **Perception** turns raw sensor data (millions of pixel values) into facts
  ("red block at x, y, z").
- **Ground truth** is the exact state, available only in simulation. Code that
  uses it is "cheating" in a useful way: it lets you build and test everything
  else first. `Skills` uses it by default.
- A classic pipeline, in `robot_sim/perception.py`: **segmentation** (which
  pixels belong to the object, here by colour), then **localisation** (where
  that is in metres, using a **depth image** and the *pinhole camera model*).
- Converting pixels to metres requires knowing where the camera is and how wide
  it sees. Finding those numbers on real hardware is called *calibration*.
- **Occlusion**: you cannot see through things, including your own arm. In
  camera mode the skills park the arm to the side before every look
  (`Skills._look()`).
- Modern systems replace hand-written colour rules with trained neural networks,
  but the job (pixels in, facts out) and the failure modes are the same.

**Predict, then run it.**

1. Predict how far the camera's estimates will be from ground truth, and how the
   robot's motion will differ from the default mode.

```powershell
python run_demo.py "stack green on red on blue" --perception camera
```

2. In `robot_sim/config.py`, change red's colour to purple:
   `"red":   (0.60, 0.10, 0.80, 1.0),` Predict what `pick up red` does. Restore it afterwards.

```powershell
python run_demo.py "pick up red" --perception camera
```

<details><summary>What happened</summary>
1. The stack succeeds with placements 1–3 mm off, the same as with ground truth
(the camera's own error is about 0.3 mm). The visible difference: before each
action the arm swings out to the left so the camera can see.<br>
2. <code>get_scene</code> reports only green and blue. The grasp is REJECTED:
"no block called 'red'; known blocks: ['blue', 'green']". The block is on the
table in plain view; the robot's rules simply have no category for that colour.
</details>

**Check yourself.**
1. Why does `locate_blocks` keep only the pixels closest to the camera?
2. Give two ways this colour-based method would fail in a real kitchen.
3. With `--perception truth` the purple block is picked up fine. Why?

<details><summary>Answers</summary>
1. From above, an off-centre block also shows a side face; the top face is the
   highest part and gives the true centre.
2. Lighting changes the colours; two objects share a colour; shiny or
   transparent objects; clutter.
3. Ground truth reads positions from the physics engine and ignores appearance.
</details>

---

## Module 6 — Where LLMs fit: planner vs. controller

**Hook.** If an LLM is smart enough to write code, why not let it drive the motors directly?

**Analogy.** A ship's captain and the engine room. The captain says "take us
alongside the east pier", a few decisions per hour drawn from broad knowledge.
The engine governor adjusts fuel many times a second and knows nothing about
piers. Swap them and the ship is lost either way.

**Core ideas.**
- An **LLM** (large language model) is excellent at turning loose human language
  into structured intent, and at common-sense sequencing ("clear the top block
  first").
- It is a poor fit for motor control for three reasons: it is **slow** (around a
  second per reply; the motor loop needs 500 per second), it is **not
  guaranteed correct**, and it has **no direct sense of physical dynamics**.
- **Function calling** (tool use) is the bridge. We give the model a menu of
  tools with exact argument shapes; it replies with a structured request; our
  code runs it and returns the result; repeat. Read `TOOLS`, `SYSTEM_PROMPT`
  and `AnthropicPlanner.run()` in `robot_sim/llm_planner.py`.
- The LLM never touches the robot. Every request passes the **validator**
  (`robot_sim/validator.py`), deterministic code that enforces physical rules.
- Results flow back up, so the model can react to a rejection or a failed grasp.
  This is the sense–think–act loop again, at the slowest timescale.
- The **plan-vs-actual log** answers the first debugging question in robotics:
  was the plan wrong, or was the execution wrong?

**Predict, then run it.** These work without an API key, using the offline
stand-in planner. Predict the outcome of each line of this session:

```powershell
python run_demo.py --interactive --viewer
```
```
stack red on blue
pick up blue
please pile the red one onto the blue one
put red at 0.6 0.0
```

<details><summary>What happened</summary>
1. Done: red placed on blue, about 2 mm off.<br>
2. REJECTED: "the red block is on top of blue; move red first".<br>
3. Offline planner: "Sorry, I only understand a few fixed phrasings."<br>
4. Red is grasped, then <code>place</code> is REJECTED: 0.600 m from the base,
allowed 0.14–0.36 m. The stand-in planner gives up while still holding red.<br><br>
If you have an API key, set it (see README) and repeat lines 2–4 with
<code>--planner llm</code>. Compare the two <code>report.html</code> files: how does
each planner respond to a rejection? (The LLM path was not run live while this
course was written, so this comparison is yours to make.)
</details>

**Check yourself.**
1. Give one task for the LLM layer and one it must never do.
2. Why are the validator's rules plain `if` statements instead of a second LLM?
3. The LLM is told to trust tool results over its own plan. Why does that matter?

<details><summary>Answers</summary>
1. Decide the order of pick-and-place steps; never compute motor torques.
2. They must be predictable, fast, and testable: the same input always gives the same verdict.
3. The world may not match the plan (a block slipped). Measurements are the truth.
</details>

---

## Module 7 — VLA models and imitation learning

**Hook.** Nobody can write the equations for folding a towel. How do robots learn it?

**Analogy.** Learning to cook by watching, rather than from a recipe. You copy
what the cook did in situations like the one in front of you. It works well for
dishes you have watched, and badly the moment something happens that you never saw.

**Core ideas.**
- Everything so far was *programmed*: we wrote the kinematics and the skills.
  The alternative is to *learn* behaviour from data.
- A **policy** is any rule mapping what the robot observes to what it does next.
  Our skills are hand-written policies.
- **Imitation learning** (also *behaviour cloning*): record a human operating
  the robot (*teleoperation*), collecting many (observation, action) pairs, then
  train a model to predict the action from the observation. With the SO-101 you
  record by moving a "leader" arm by hand while the "follower" arm copies it.
- A **VLA (vision-language-action model)** is a large neural network taking
  camera images plus a text instruction and outputting motor actions directly,
  many times a second. It merges perception, planning and control into one
  learned model. Compare with our design: the LLM picks from a menu of
  hand-written skills; a VLA *is* the skill.
- Trade-offs. Our modular stack is inspectable (you can see which layer failed),
  needs no training data, and only does what its skills cover. A learned policy
  can handle things nobody can program (cloth, clutter), but needs many
  demonstrations, is hard to debug, and fails unpredictably in unfamiliar situations.
- **Distribution shift**: a learned policy is only reliable in situations
  resembling its training data. Small errors push it into unfamiliar states,
  where it errs more, and the errors compound.
- The two approaches combine well: an LLM planner on top, learned skills beneath.

**Predict, then run it.** The script records an expert reaching for targets on
the **left** half of the table only. The "learned" policy copies the action from
the most similar recorded situations. It is then tested on left-side and on
never-seen right-side targets, 20 trials each, success = within 2 cm.

Fill in your predicted success counts first:

| Demonstrations | Left (seen) | Right (never seen) |
|---|---|---|
| 1 | | |
| 10 | | |
| 50 | | |
| 200 | | |

```powershell
python stages/05_imitation.py --demos 1
```
```powershell
python stages/05_imitation.py --demos 10
```
```powershell
python stages/05_imitation.py --demos 50
```
```powershell
python stages/05_imitation.py --demos 200
```

<details><summary>What happened</summary>

| Demonstrations | Left (seen) | Right (never seen) |
|---|---|---|
| 1 | 0/20 | 0/20 |
| 10 | 2/20 | 0/20 |
| 50 | 10/20 | 0/20 |
| 200 | 18/20 | 0/20 |

(600 demonstrations: 20/20 and still 0/20.) More data steadily fixes the
situations the data covers and does nothing for those it does not. The
right-side misses stay around 150–300 mm. Compare with Module 2: the IK formula
reaches any target exactly, with zero demonstrations, because it encodes the
geometry rather than examples of it.
</details>

**Check yourself.**
1. In `stages/05_imitation.py`, what is the observation and what is the action?
2. Why would collecting 10,000 more left-side demonstrations not fix the right side?
3. When would you choose a learned policy over a hand-written skill?

<details><summary>Answers</summary>
1. Observation: gripper position and target position (6 numbers). Action: the next small move (3 numbers).
2. The policy copies similar situations, and no recorded situation resembles a right-side target.
3. When nobody can write the rules (deformable objects, clutter, contact-rich tasks) and demonstrations are obtainable.
</details>

---

## Module 8 — Safety, failure modes, and the sim-to-real gap

**Hook.** The simulated robot stacked the blocks 10 times out of 10. What will
the real one do?

**Analogy.** Flight simulators. Pilots train in them for good reasons, and still
nobody goes straight from the simulator to carrying passengers. The simulator
is never wrong about the wind.

**Core ideas.**
- The **sim-to-real gap** is everything the simulation gets wrong or leaves out:
  sensor noise, camera calibration error, motor backlash and delay, friction
  that varies, objects that differ from their models, lighting.
- **Failure modes** in this project, by layer. Planner: wrong block, wrong
  height, invented objects. Perception: occlusion, unknown colours, position
  error. Skills: missed or slipping grasps, toppled stacks. Controller:
  overshoot, sag, instability. Knowing the layer is most of the diagnosis.
- **Defence in depth**: several independent safeguards, since any single one
  fails sometimes. Here: strict tool schemas, the validator, IK reachability
  checks, the finger-gap check after grasping, the position check after placing,
  motor torque limits, an action budget.
- On real hardware, add layers that do not depend on software at all: a
  physical emergency stop, low speed and torque limits, a clear workspace, and
  you standing out of reach.
- Engineers narrow the gap with **domain randomisation** (train and test under
  deliberately varied conditions, as below) and by measuring the real system
  and correcting the simulation (*system identification*).
- A validator catches *unsafe* actions. It does not catch *wrong but safe* ones,
  such as stacking the wrong colour. Safety and correctness are different properties.

**Predict, then run it.** The blocks are 40 mm wide; the open gripper is 70 mm
wide, so there are 15 mm of clearance each side. The script adds random error
to the block positions the robot is *told* and tries "stack red on blue" 10
times per noise level. Predict the success count at each level.

```powershell
python stages/06_sim_to_real.py
```

<details><summary>What happened</summary>

| Position error (std. dev.) | Stacks that worked | Failures |
|---|---|---|
| 0 mm | 10/10 | – |
| 2 mm | 10/10 | – |
| 5 mm | 9/10 | 1 red not on blue |
| 10 mm | 6/10 | 1 grasp failed, 3 red not on blue |
| 15 mm | 5/10 | 3 grasp failed, 2 red not on blue |
| 20 mm | 4/10 | 3 grasp failed, 3 red not on blue |

Two lessons. Performance does not fall off a cliff at 15 mm; it erodes from
5 mm, because errors in the grasp and in the placement add up. And "grasp
failed" is at least *detected* by the finger-gap check, whereas "red not on
blue" is a stack that looked fine to a robot using the same faulty positions
to check its own work.
</details>

**Check yourself.**
1. Name one failure the validator would catch and one it would not.
2. Why does error in perceiving the *blue* block matter when only red is being moved?
3. List three non-software safety measures for a first real arm.

<details><summary>Answers</summary>
1. Catches: placing outside the workspace. Misses: stacking red on green when asked for blue.
2. Blue's believed position is the placement target.
3. Emergency stop or easy power cut; low torque and speed limits; clear, bounded
   workspace; eye protection; never reaching in while it is powered.
</details>

---

## What to learn next: a roadmap to real hardware

You now have the map. Here is a sensible order for the territory, aimed at a
low-cost arm such as the **LeRobot SO-101** (a 5-joint arm plus gripper driven
by serial-bus hobby servos, usually built as a leader/follower pair for
teleoperation, and supported by Hugging Face's open-source `lerobot` library).
Check current kit options and prices before buying; they change.

**Stage 1 — Strengthen the foundations (2–4 weeks, still simulation)**
- Python fluency: write small programs yourself, not just read them. Good
  drills in this project: add a fourth block; add a `rotate` skill; make the
  rule-based planner understand "swap red and blue".
- NumPy basics (arrays, indexing), since every robotics library speaks it.
- Rotations in 3D: rotation matrices and quaternions. We avoided them by always
  pointing the gripper straight down. Real grasps do not.
- Extend this simulator: replace the analytic IK with a numerical one using the
  Jacobian (MuJoCo's `mj_jacSite`), which generalises to any arm.

**Stage 2 — Simulate the real arm (1–2 weeks)**
- Load a published SO-101 model into MuJoCo and point this project's layers at
  it. The interface to change is the `Sim` class and the lengths in `config.py`.
- Note what differs: position-controlled servos (you send target angles, not
  torques), joint limits, a gripper that is not a parallel jaw.

**Stage 3 — First hardware (2–4 weeks)**
- Assemble and calibrate the arm following the LeRobot documentation.
- Run teleoperation before any autonomy: leader arm moves, follower copies.
- Safety habits from day one: power switch within reach, low torque limits,
  clear table, never reach in while powered.
- Reproduce Module 4 on a real joint: command a step, plot the response.

**Stage 4 — Real perception (2–3 weeks)**
- Add a camera; learn camera calibration and hand–eye calibration (where the
  camera is relative to the arm) with OpenCV.
- Redo Module 5 for real, and expect lighting to break the colour rules. Then
  try a pretrained object detector.

**Stage 5 — Learning from demonstration (4+ weeks)**
- Record a small dataset by teleoperation with `lerobot`, train a policy such as
  ACT (Action Chunking with Transformers) or a diffusion policy, and evaluate it.
  This is the real version of Module 7, including its failure outside the data.
- This stage wants a GPU. Without one, use a rented cloud GPU or a free notebook
  service for training; running a small trained policy afterwards is far lighter.
- Then try fine-tuning a small open VLA model on your data.

**Stage 6 — Put the layers back together**
- Use an LLM planner (this project's layer d) to sequence *learned* skills, with
  a validator in between. That is the architecture of much current research.

**Worth knowing exists, for later:** ROS 2 (the standard robotics middleware,
needed for bigger systems), motion planning with obstacle avoidance (MoveIt,
OMPL), reinforcement learning, and GPU-parallel simulation (MuJoCo MJX, Isaac Lab).

**Free references**
- MuJoCo documentation: <https://mujoco.readthedocs.io/>
- LeRobot: <https://github.com/huggingface/lerobot>
- *Modern Robotics* (Lynch and Park), free textbook and video lectures, for the
  mathematics of Modules 2 and 4: <https://modernrobotics.org/>
- Anthropic tool-use documentation, for Module 6: <https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview>
