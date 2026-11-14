"""Tests for the planning layer.

The real Anthropic API is NOT called here (tests must run offline and for
free).  Instead `FakeClient` plays back a scripted conversation shaped like
real API responses, which checks our loop: tool calls go through the
validator, results go back with the right ids, errors are flagged.
"""
import json
from types import SimpleNamespace

import numpy as np

from robot_sim import config as C
from robot_sim.llm_planner import TOOLS, AnthropicPlanner, RuleBasedPlanner

HALF = C.BLOCK_SIZE / 2


def test_rule_planner_stacks(robot):
    sim, ctrl, skills, executor = robot
    assert RuleBasedPlanner(executor).run("Stack red on blue") == "Done."
    red, blue = sim.block_pos("red"), sim.block_pos("blue")
    assert abs(red[2] - blue[2] - C.BLOCK_SIZE) < 0.004
    assert np.hypot(red[0] - blue[0], red[1] - blue[1]) < 0.008


def test_stack_then_unstack(robot):
    sim, ctrl, skills, executor = robot
    planner = RuleBasedPlanner(executor)
    assert planner.run("stack red on blue") == "Done."
    assert "on top of blue" in planner.run("pick up blue")          # blocked: red is on it
    assert planner.run("put red on the table") == "Done."
    red = sim.block_pos("red")
    assert abs(red[2] - HALF) < 0.004
    assert np.hypot(red[0] - C.BLOCK_START["red"][0], red[1] - C.BLOCK_START["red"][1]) < 0.008


def test_rule_planner_unsafe_request_is_blocked(robot):
    sim, ctrl, skills, executor = robot
    summary = RuleBasedPlanner(executor).run("put red at 0.6 0.0")
    assert "REJECTED" in summary
    entries = [e for e in executor.log.entries if "plan" in e]
    assert entries[-1]["validator"]["approved"] is False


def test_rule_planner_admits_what_it_cannot_parse(robot):
    *_, executor = robot
    assert "only understand" in RuleBasedPlanner(executor).run("juggle the blocks")


def test_tool_schemas_are_strict():
    for tool in TOOLS:
        schema = tool["input_schema"]
        assert tool["strict"] is True and schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])


def _tool_use(id_, name, **args):
    return SimpleNamespace(type="tool_use", id=id_, name=name, input=args)


class FakeClient:
    """Stands in for anthropic.Anthropic(): replays scripted model replies."""

    def __init__(self, script):
        self.script = list(script)
        self.requests = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(json.loads(json.dumps(kwargs, default=lambda o: vars(o))))
        return self.script.pop(0)


def test_llm_loop_with_scripted_model(robot):
    sim, ctrl, skills, executor = robot
    bx, by, bz = (round(float(v), 3) for v in sim.block_pos("blue"))
    script = [
        SimpleNamespace(stop_reason="tool_use", content=[_tool_use("t1", "get_scene")]),
        SimpleNamespace(stop_reason="tool_use", content=[
            _tool_use("t2", "grasp", block="red", reason="red must move onto blue")]),
        # A deliberate mistake: placing INSIDE blue.  The validator must stop it.
        SimpleNamespace(stop_reason="tool_use", content=[
            _tool_use("t3", "place", x=bx, y=by, z=bz, reason="put red on blue")]),
        SimpleNamespace(stop_reason="tool_use", content=[
            _tool_use("t4", "place", x=bx, y=by, z=bz + C.BLOCK_SIZE, reason="corrected height")]),
        SimpleNamespace(stop_reason="tool_use", content=[_tool_use("t5", "go_home")]),
        SimpleNamespace(stop_reason="end_turn", content=[
            SimpleNamespace(type="text", text="Red is now on blue.")]),
    ]
    client = FakeClient(script)
    summary = AnthropicPlanner(executor, client=client).run("stack red on blue")

    assert summary == "Red is now on blue."
    steps = [e for e in executor.log.entries if "plan" in e]
    assert [s["validator"]["approved"] for s in steps] == [True, True, False, True, True]
    assert abs(sim.block_pos("red")[2] - (HALF + C.BLOCK_SIZE)) < 0.004

    # The rejected call must have been reported back to the model as an error,
    # tied to the right tool_use id.
    feedback = client.requests[3]["messages"][-1]["content"][0]
    assert feedback["tool_use_id"] == "t3" and feedback["is_error"] is True
    assert "REJECTED" in feedback["content"]
    # Request shape we rely on.
    assert client.requests[0]["model"] == "claude-opus-5-5"
    assert client.requests[0]["thinking"] == {"type": "adaptive"}


def test_llm_refusal_is_handled(robot):
    *_, executor = robot
    client = FakeClient([SimpleNamespace(stop_reason="refusal", content=[])])
    assert "declined" in AnthropicPlanner(executor, client=client).run("anything")
