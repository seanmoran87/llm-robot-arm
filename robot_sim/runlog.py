"""THE RUN LOG: the plan next to what the robot actually did.

For every step we store two things side by side:

    PLAN    what the planner asked for (tool, numbers, and its stated reason)
    ACTUAL  what the validator decided, and what the sensors measured after

Comparing the two columns is the single most useful debugging habit in
robotics: when something goes wrong, the first question is always "was the
plan wrong, or was the plan fine and the execution wrong?"

Output files (in the run folder):
    log.jsonl     one JSON object per step - easy to load in Python later
    report.html   the same thing as a table with a snapshot per step
"""

from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path

from PIL import Image

from .scene import Sim


def format_call(tool: str, args: dict) -> str:
    """'place', {x: 0.24, ...} -> 'place(x=0.24, y=0.12, z=0.06)'."""
    inner = ", ".join(f"{k}={v}" for k, v in args.items())
    return f"{tool}({inner})"


def describe_result(tool: str, ok: bool, result: dict) -> str:
    """One short human-readable line about what really happened."""
    if not ok:
        return "NOT EXECUTED"
    if not result.get("success"):
        return f"FAILED: {result.get('error', 'target not met')}"
    if tool == "get_scene":
        blocks = ", ".join(f"{n}@({p[0]:.2f},{p[1]:.2f},{p[2]:.2f})" for n, p in result["blocks"].items())
        return f"saw {blocks}; holding {result['holding']}"
    if tool in ("move_to", "go_home"):
        p = result["gripper_position"]
        return f"gripper at ({p[0]:.3f},{p[1]:.3f},{p[2]:.3f}), {result['position_error_mm']} mm off"
    if tool == "grasp":
        return f"holding {result['holding']} (finger gap {result['finger_gap_mm']} mm)"
    p = result["block_position"]
    return f"{result['placed']} now at ({p[0]:.3f},{p[1]:.3f},{p[2]:.3f}), {result['placement_error_mm']} mm off"


class RunLog:
    def __init__(self, sim: Sim, out_dir: str | Path, echo: bool = True, snapshots: bool = True):
        self.sim = sim
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.echo = echo
        self.snapshots = snapshots
        self.entries: list[dict] = []
        self._jsonl = open(self.out_dir / "log.jsonl", "w", encoding="utf-8")
        self._command = ""
        self._planner = ""
        self._step = 0

    def _print(self, text: str) -> None:
        if self.echo:
            print(text, flush=True)

    def start_command(self, command: str, planner_name: str) -> None:
        self._command, self._planner, self._step = command, planner_name, 0
        self._print(f"\nCOMMAND: {command!r}    (planner: {planner_name})")
        self._print(f"{'#':>2}  {'PLAN (what was asked for)':<44} {'CHECK':<9} ACTUAL (what was measured)")
        self._print("-" * 118)

    def record(self, tool: str, args: dict, reason: str, ok: bool, why: str, result: dict) -> None:
        self._step += 1
        entry = {
            "command": self._command, "planner": self._planner, "step": self._step,
            "sim_time_s": round(self.sim.time, 2),
            "plan": {"tool": tool, "args": args, "reason": reason},
            "validator": {"approved": ok, "reason": why},
            "actual": result,
            # Ground truth, recorded for the human reader only (the planner
            # never sees this field).
            "true_block_positions": {n: [round(float(v), 4) for v in self.sim.block_pos(n)]
                                     for n in self.sim.block_names()},
        }
        self._jsonl.write(json.dumps(entry) + "\n")
        self._jsonl.flush()
        if self.snapshots:
            entry["_image"] = self.sim.render("overview", 320, 240)
        self.entries.append(entry)

        check = "approved" if ok else "REJECTED"
        self._print(f"{self._step:>2}  {format_call(tool, args):<44} {check:<9} {describe_result(tool, ok, result)}")
        if not ok:
            self._print(f"    {'':<44} {'':<9} reason: {why}")

    def finish_command(self, summary: str) -> None:
        entry = {"command": self._command, "planner": self._planner, "summary": summary}
        self._jsonl.write(json.dumps(entry) + "\n")
        self._jsonl.flush()
        self.entries.append(entry)
        self._print(f"PLANNER SAYS: {summary}")

    # ------------------------------------------------------------- report
    def write_html(self, title: str = "Robot run report", video: str | None = None) -> Path:
        """Write report.html: one row per step, plan on the left, reality on the right."""
        rows, last_command = [], None
        for e in self.entries:
            if "summary" in e:
                rows.append(f'<tr class="summary"><td colspan="5"><b>Planner\'s closing message:</b> '
                            f'{html.escape(e["summary"])}</td></tr>')
                continue
            if e["command"] != last_command:
                last_command = e["command"]
                rows.append(f'<tr class="cmd"><td colspan="5">Command: &ldquo;{html.escape(e["command"])}&rdquo; '
                            f'<span class="dim">planner: {html.escape(e["planner"])}</span></td></tr>')
            plan, val = e["plan"], e["validator"]
            img = ""
            if "_image" in e:
                buf = io.BytesIO()
                Image.fromarray(e["_image"]).save(buf, format="PNG")
                img = f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}" width="200">'
            verdict = ('<span class="ok">approved</span>' if val["approved"]
                       else f'<span class="bad">REJECTED</span><br><small>{html.escape(val["reason"])}</small>')
            outcome = describe_result(plan["tool"], val["approved"], e["actual"])
            cls = "good" if e["actual"].get("success") else "fail"
            rows.append(
                f'<tr><td>{e["step"]}</td>'
                f'<td><code>{html.escape(format_call(plan["tool"], plan["args"]))}</code>'
                f'<br><small class="dim">{html.escape(plan["reason"])}</small></td>'
                f'<td>{verdict}</td>'
                f'<td class="{cls}">{html.escape(outcome)}<br><small class="dim">t = {e["sim_time_s"]} s</small></td>'
                f'<td>{img}</td></tr>')
        video_html = f'<p><img src="{html.escape(video)}" alt="recording of the run"></p>' if video else ""
        page = f"""<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>
 body{{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:24px;color:#1c1c1c;background:#fafafa}}
 table{{border-collapse:collapse;width:100%;max-width:1150px;background:#fff}}
 th,td{{border:1px solid #d8d8d8;padding:8px 10px;vertical-align:top;text-align:left;font-size:14px}}
 th{{background:#eef1f5}} code{{font-size:13px}}
 tr.cmd td{{background:#1f2a44;color:#fff;font-weight:600}} tr.summary td{{background:#f3f6fa}}
 .dim{{color:#777;font-weight:400}} tr.cmd .dim{{color:#b9c3d8;margin-left:12px}}
 .ok{{color:#1a7f37;font-weight:600}} .bad{{color:#c62828;font-weight:700}}
 td.fail{{background:#fdecea}} td.good{{background:#f1f8f2}}
</style></head><body>
<h1>{html.escape(title)}</h1>
<p>Left: what the planner asked for. Middle: the safety validator's decision.
Right: what the robot's sensors measured afterwards.</p>
{video_html}
<table><tr><th>#</th><th>Plan (from the planner)</th><th>Safety check</th>
<th>Actual (measured)</th><th>Snapshot after the step</th></tr>
{''.join(rows)}
</table></body></html>"""
        path = self.out_dir / "report.html"
        path.write_text(page, encoding="utf-8")
        return path

    def close(self) -> None:
        self._jsonl.close()
