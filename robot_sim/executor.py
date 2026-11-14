"""THE EXECUTOR: the single doorway every planned action must pass through.

    planner (LLM or rules)  --tool call-->  SafeExecutor
                                              1. look at the scene
                                              2. ask the Validator
                                              3. if approved, run the Skill
                                              4. write both to the log
                                           <--result--

Because BOTH planners (the real LLM and the offline rule-based stand-in) go
through this same doorway, the safety checks and the log are identical
whichever one is driving.
"""

from __future__ import annotations

from .runlog import RunLog
from .skills import Skills
from .validator import Validator

MAX_ACTIONS_PER_COMMAND = 30    # a runaway plan gets cut off here


class SafeExecutor:
    def __init__(self, skills: Skills, validator: Validator, log: RunLog):
        self.skills = skills
        self.validator = validator
        self.log = log
        self.actions_used = 0

    def start_command(self, command: str, planner_name: str) -> None:
        self.actions_used = 0
        self.log.start_command(command, planner_name)

    def execute(self, tool: str, args: dict, reason: str = "") -> dict:
        """Validate, then (only if approved) run one tool call.

        Returns the result dictionary that goes back to the planner.  A
        rejected call returns {"success": False, "rejected": True, ...}.
        """
        args = {k: v for k, v in (args or {}).items() if k != "reason"}

        self.actions_used += 1
        if self.actions_used > MAX_ACTIONS_PER_COMMAND:
            verdict_ok, why = False, f"action budget of {MAX_ACTIONS_PER_COMMAND} used up"
        else:
            scene = self.skills.get_scene()
            verdict = self.validator.check(tool, args, scene)
            verdict_ok, why = verdict.ok, verdict.reason

        if not verdict_ok:
            result = {"success": False, "rejected": True,
                      "error": f"REJECTED by safety validator: {why}"}
        else:
            # Each tool name maps to exactly one skill method.
            if tool == "get_scene":
                result = self.skills.get_scene()
            elif tool == "move_to":
                result = self.skills.move_to(args["x"], args["y"], args["z"])
            elif tool == "grasp":
                result = self.skills.grasp(args["block"])
            elif tool == "place":
                result = self.skills.place(args["x"], args["y"], args["z"])
            else:  # go_home
                result = self.skills.home()

        self.log.record(tool, args, reason, verdict_ok, why, result)
        return result
