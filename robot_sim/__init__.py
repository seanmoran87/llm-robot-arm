"""robot_sim: a small LLM-controlled robot-arm simulator built for learning.

Read the files in this order (it is also the order the layers stack up):

    config.py      -> every number that describes the robot and the world
    scene.py       -> layer (a): the physics scene (MuJoCo)
    kinematics.py  -> the geometry that links joint angles to gripper position
    controller.py  -> layer (b): the low-level controller (PID + trajectories)
    skills.py      -> layer (c): the skill library (move_to, grasp, place)
    validator.py   -> the safety check that sits between the LLM and the robot
    llm_planner.py -> layer (d): the LLM turns a sentence into skill calls
    perception.py  -> optional: find the blocks with a camera instead of cheating
    runlog.py      -> the "plan vs. what actually happened" log and HTML report
"""
