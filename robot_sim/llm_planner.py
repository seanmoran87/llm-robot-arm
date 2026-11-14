"""LAYER (d): THE LLM PLANNER.

This layer turns a sentence such as "stack red on blue" into a sequence of
skill calls.  It uses FUNCTION CALLING (also called "tool use"):

  1. We send the LLM the user's sentence plus a menu of tools - each with a
     name, a description, and the exact shape of its arguments.
  2. Instead of replying with prose, the LLM replies with a structured
     request: "call grasp with block = red".
  3. OUR code runs that request (through the safety validator!) and sends the
     result back.
  4. The LLM reads the result and decides the next call, or says it is done.

Important: the LLM never touches the robot.  It only ever produces a request.
Whether that request runs is decided by executor.py and validator.py.

Two planners live here and both speak the same tool language:

  * AnthropicPlanner  - a real LLM (Claude) via the Anthropic API.
  * RuleBasedPlanner  - a tiny hand-written stand-in that understands a few
                        fixed phrasings.  It needs no API key or internet, so
                        you can run and test every other layer offline.  It is
                        also a useful contrast: see how brittle it is compared
                        with the LLM once you rephrase a command.
"""

from __future__ import annotations

import json
import re

from . import config as C
from .executor import SafeExecutor

# ---------------------------------------------------------------------------
# The tool menu.  This is exactly what the LLM sees.  The descriptions matter:
# they are the LLM's only instruction manual for each skill.
# "input_schema" is JSON Schema, a standard way of describing the shape of
# data.  "strict": True asks the API to guarantee the arguments match it.
# ---------------------------------------------------------------------------
_REASON = {"type": "string",
           "description": "One short sentence: why this step, in plain English."}
_XYZ = {
    "x": {"type": "number", "description": "metres, forward from the robot base"},
    "y": {"type": "number", "description": "metres, to the robot's left"},
    "z": {"type": "number", "description": "metres above the table surface"},
}

TOOLS = [
    {
        "name": "get_scene",
        "description": "Look at the table. Returns the centre position (x, y, z in metres) "
                       "of every block, which block (if any) the gripper holds, the "
                       "gripper position, and the block size. Call this first, and "
                       "again whenever you need fresh positions.",
        "strict": True,
        "input_schema": {"type": "object", "properties": {}, "required": [],
                         "additionalProperties": False},
    },
    {
        "name": "grasp",
        "description": "Pick up the named block from above and lift it to a safe "
                       "travel height. Fails if the gripper is already holding "
                       "something or another block sits on top of it.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"block": {"type": "string", "enum": list(C.BLOCK_COLORS)},
                           "reason": _REASON},
            "required": ["block", "reason"], "additionalProperties": False},
    },
    {
        "name": "place",
        "description": "Set the held block down so that its CENTRE ends at (x, y, z). "
                       "On the table, z = block_size / 2. On top of another block, use "
                       "that block's x and y, and z = its z + block_size. The target "
                       "must be supported and unoccupied.",
        "strict": True,
        "input_schema": {
            "type": "object", "properties": {**_XYZ, "reason": _REASON},
            "required": ["x", "y", "z", "reason"], "additionalProperties": False},
    },
    {
        "name": "move_to",
        "description": "Move the gripper (pointing straight down) to (x, y, z) without "
                       "opening or closing it. Not needed for ordinary pick-and-place: "
                       "grasp and place do their own moving.",
        "strict": True,
        "input_schema": {
            "type": "object", "properties": {**_XYZ, "reason": _REASON},
            "required": ["x", "y", "z", "reason"], "additionalProperties": False},
    },
    {
        "name": "go_home",
        "description": "Return the arm to its home pose, clear of the table.",
        "strict": True,
        "input_schema": {"type": "object", "properties": {}, "required": [],
                         "additionalProperties": False},
    },
]

SYSTEM_PROMPT = f"""You plan actions for a small robot arm that moves coloured blocks on a table.

World: coordinates are in metres. The origin is on the table under the robot base;
+x is forward, +y is to the robot's left, +z is up, and z = 0 is the table surface.
The blocks are {C.BLOCK_SIZE} m cubes named {", ".join(C.BLOCK_COLORS)}. The gripper can reach points
between {C.WORKSPACE_R_MIN} and {C.WORKSPACE_R_MAX} m (horizontal distance) from the base.

You act only by calling tools, one physical action at a time. Every action is checked by
a safety validator before it runs; a rejected action comes back as an error explaining
why, and nothing moves. Read each result before choosing the next step: results are
sensor measurements and they, not your plan, are the truth about the world.

Start by calling get_scene. When placing a block on the table, keep its centre at least
0.08 m from other blocks. If the request cannot be done safely, or is not something
this robot can do, say so instead of improvising. When finished, call go_home and then
reply with one or two plain sentences saying what was done and anything that went wrong."""

DEFAULT_MODEL = "claude-opus-5-5"


class AnthropicPlanner:
    """Drives the robot with a real LLM through the Anthropic Messages API."""

    name = "llm"

    def __init__(self, executor: SafeExecutor, model: str = DEFAULT_MODEL, client=None,
                 max_turns: int = 40):
        self.executor = executor
        self.model = model
        self.max_turns = max_turns
        if client is None:
            # Imported here so the offline planner works even without the package.
            import anthropic
            # Reads the API key from the ANTHROPIC_API_KEY environment variable.
            client = anthropic.Anthropic()
        self.client = client

    def run(self, command: str) -> str:
        """Carry out one natural-language command.  Returns the LLM's closing message."""
        self.executor.start_command(command, f"llm:{self.model}")
        # The conversation so far.  We resend all of it on every request: the
        # API is "stateless", meaning the LLM remembers nothing between calls.
        messages = [{"role": "user", "content": command}]
        summary = "(planner stopped without a closing message)"

        for _ in range(self.max_turns):
            response = self._ask(messages)

            # stop_reason tells us WHY the LLM stopped writing.
            if response.stop_reason == "refusal":
                summary = "The model declined this request."
                break
            if response.stop_reason == "max_tokens":
                summary = "The model's reply was cut off (max_tokens); stopping."
                break

            # Keep the LLM's full reply in the conversation, exactly as received.
            messages.append({"role": "assistant", "content": response.content})
            text = " ".join(b.text for b in response.content if b.type == "text").strip()

            if response.stop_reason != "tool_use":       # "end_turn": it is finished
                summary = text or summary
                break

            # Run every tool the LLM asked for and collect the results.
            results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                args = dict(block.input) if isinstance(block.input, dict) else {}
                result = self.executor.execute(block.name, args, reason=str(args.get("reason", text)))
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,            # ties the result to the request
                    "content": json.dumps(result),
                    "is_error": not result.get("success", False),
                })
            # All results go back together in ONE user message.
            messages.append({"role": "user", "content": results})
        else:
            summary = f"Stopped after {self.max_turns} turns without finishing."

        self.executor.log.finish_command(summary)
        return summary

    def _ask(self, messages: list):
        """One request to the API, with errors turned into readable messages."""
        import anthropic
        try:
            # client.beta.messages is used (instead of client.messages) only
            # so that we can pass `fallbacks`: if a safety filter on the main
            # model declines a request, the API retries it on another model
            # instead of handing us a refusal.
            return self.client.beta.messages.create(
                model=self.model,
                max_tokens=16000,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
                thinking={"type": "adaptive"},          # let the model reason when useful
                output_config={"effort": "low"},        # short plans: no need to think hard
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except anthropic.AuthenticationError as e:
            raise SystemExit("The Anthropic API rejected the API key. Check ANTHROPIC_API_KEY.") from e
        except anthropic.RateLimitError as e:
            raise SystemExit("Rate limited by the Anthropic API. Wait a minute and retry.") from e
        except anthropic.APIStatusError as e:
            raise SystemExit(f"Anthropic API error {e.status_code}: {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise SystemExit("Could not reach the Anthropic API. Check your internet connection.") from e


# ---------------------------------------------------------------------------
# The offline stand-in.
# ---------------------------------------------------------------------------
_COLOR = "|".join(C.BLOCK_COLORS)
_NUM = r"(-?\d+(?:\.\d+)?)"


class RuleBasedPlanner:
    """A deliberately simple planner: pattern-matching, no intelligence.

    Understands (case-insensitive):
        stack red on blue            put green on top of red
        stack green on red on blue   (builds from the bottom up)
        put red at 0.30 0.10         (a spot on the table: x y)
        put red on the table
        pick up green
        move to 0.25 0.0 0.15        (gripper x y z)
        home
    Anything else: it gives up - which is exactly the gap an LLM fills.
    Unlike the LLM it also cannot recover: at the first failure it stops.
    """

    name = "rules"

    def __init__(self, executor: SafeExecutor):
        self.executor = executor

    def run(self, command: str) -> str:
        self.executor.start_command(command, "rules (offline stand-in)")
        summary = self._plan_and_run(command.lower().strip())
        self.executor.log.finish_command(summary)
        return summary

    def _do(self, tool: str, args: dict, reason: str) -> dict:
        return self.executor.execute(tool, args, reason=reason)

    def _plan_and_run(self, text: str) -> str:
        half = C.BLOCK_SIZE / 2

        if text in ("home", "go home"):
            self._do("go_home", {}, "asked to go home")
            return "Went home."

        m = re.search(rf"move (?:the )?(?:gripper )?to {_NUM}[,\s]+{_NUM}[,\s]+{_NUM}", text)
        if m:
            x, y, z = (float(v) for v in m.groups())
            r = self._do("move_to", {"x": x, "y": y, "z": z}, "asked to move the gripper there")
            return "Moved." if r["success"] else f"Could not move: {r['error']}"

        scene = self._do("get_scene", {}, "need block positions before planning")
        blocks = scene["blocks"]

        m = re.search(rf"pick up (?:the )?({_COLOR})", text)
        if m:
            r = self._do("grasp", {"block": m.group(1)}, "asked to pick it up")
            return f"Holding {m.group(1)}." if r["success"] else f"Could not: {r['error']}"

        # "put red at X Y"  /  "put red on the table"
        m = re.search(rf"(?:put|place|move) (?:the )?({_COLOR})(?: block)? (?:at|to) {_NUM}[,\s]+{_NUM}", text)
        table = re.search(rf"(?:put|place|move) (?:the )?({_COLOR})(?: block)? (?:back )?on (?:to )?the table", text)
        if m or table:
            if m:
                name, x, y = m.group(1), float(m.group(2)), float(m.group(3))
            else:
                name = table.group(1)
                x, y = C.BLOCK_START[name]            # its original parking spot
            steps = [("grasp", {"block": name}, f"pick up {name}"),
                     ("place", {"x": x, "y": y, "z": half}, f"set {name} on the table")]
            return self._run_steps(steps)

        # "stack A on B [on C ...]" -> work from the bottom of the pile upwards.
        if re.search(r"\b(stack|put|place)\b", text) and " on " in text:
            names = re.findall(rf"\b({_COLOR})\b", text)
            if len(names) >= 2 and len(set(names)) == len(names):
                steps = []
                pos = {n: list(p) for n, p in blocks.items()}   # predicted positions
                for upper, lower in reversed(list(zip(names, names[1:]))):
                    lx, ly, lz = pos[lower]
                    target = {"x": lx, "y": ly, "z": round(lz + C.BLOCK_SIZE, 4)}
                    steps.append(("grasp", {"block": upper}, f"pick up {upper}"))
                    steps.append(("place", target, f"set {upper} on top of {lower}"))
                    pos[upper] = [target["x"], target["y"], target["z"]]
                return self._run_steps(steps)

        return ("Sorry, I only understand a few fixed phrasings (see RuleBasedPlanner). "
                "Use --planner llm for free-form commands.")

    def _run_steps(self, steps: list) -> str:
        for tool, args, reason in steps:
            result = self._do(tool, args, reason)
            if not result.get("success"):
                if self.executor.skills.holding is None:
                    self._do("go_home", {}, "giving up after a failure")
                return f"Stopped at {tool}: {result.get('error', 'target not met')}"
        self._do("go_home", {}, "task finished")
        return "Done."
