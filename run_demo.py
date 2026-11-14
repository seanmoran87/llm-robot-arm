"""Run the LLM-controlled robot arm.

Examples (run from the project folder, with the virtual environment active):

    python run_demo.py "stack red on blue"
    python run_demo.py "stack green on red on blue" --viewer
    python run_demo.py --interactive --viewer
    python run_demo.py "put the red block on blue" --planner llm

Each run creates a folder under runs/ containing:
    report.html   plan-vs-actual table with a snapshot per step
    log.jsonl     the same data, machine-readable
    run.gif       a recording of the whole run
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from robot_sim.controller import ArmController
from robot_sim.executor import SafeExecutor
from robot_sim.llm_planner import DEFAULT_MODEL, AnthropicPlanner, RuleBasedPlanner
from robot_sim.runlog import RunLog
from robot_sim.scene import Sim
from robot_sim.skills import Skills
from robot_sim.validator import Validator


def build_robot(args, out_dir: Path):
    """Assemble the layers, bottom to top.  This function IS the architecture."""
    sim = Sim()                                         # (a) physics scene
    controller = ArmController(sim)                     # (b) low-level controller

    perception = None                                   # None = simulator ground truth
    if args.perception == "camera":
        from robot_sim.perception import CameraPerception
        perception = CameraPerception(sim).locate_blocks
    skills = Skills(sim, controller, perception)        # (c) skill library

    # The live window and the offscreen camera both use the graphics card,
    # and sharing it between them is unreliable.  So with --viewer we skip
    # the per-step snapshots (and the GIF, in main() below).
    log = RunLog(sim, out_dir, snapshots=not args.viewer)
    executor = SafeExecutor(skills, Validator(), log)   # safety gate + log

    planner_kind = args.planner
    if planner_kind == "auto":
        planner_kind = "llm" if os.environ.get("ANTHROPIC_API_KEY") else "rules"
        if planner_kind == "rules":
            print("NOTE: ANTHROPIC_API_KEY is not set, so the offline rule-based planner "
                  "is being used.\n      It understands only a few fixed phrasings. "
                  "See README.md, 'Using the real LLM'.")
    if planner_kind == "llm":
        planner = AnthropicPlanner(executor, model=args.model)   # (d) LLM planner
    else:
        planner = RuleBasedPlanner(executor)
    return sim, controller, log, planner


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", nargs="?", help='what the robot should do, e.g. "stack red on blue"')
    p.add_argument("--planner", choices=["auto", "llm", "rules"], default="auto",
                   help="llm = Claude via the Anthropic API; rules = offline stand-in; "
                        "auto = llm if ANTHROPIC_API_KEY is set, otherwise rules")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Anthropic model id for --planner llm")
    p.add_argument("--perception", choices=["truth", "camera"], default="truth",
                   help="truth = read block positions from the simulator; "
                        "camera = estimate them from the top-down camera image")
    p.add_argument("--viewer", action="store_true",
                   help="open the live 3D window (needs a screen); turns off the GIF and snapshots")
    p.add_argument("--interactive", action="store_true", help="type commands one after another")
    p.add_argument("--no-gif", action="store_true", help="skip recording run.gif (faster)")
    p.add_argument("--out", default=None, help="output folder (default: runs/<date-time>)")
    args = p.parse_args()
    if not args.command and not args.interactive:
        p.error('give a command, e.g.  python run_demo.py "stack red on blue"   (or use --interactive)')

    out_dir = Path(args.out) if args.out else Path("runs") / time.strftime("%Y%m%d-%H%M%S")
    sim, controller, log, planner = build_robot(args, out_dir)

    recorder = viewer = None
    if args.viewer and args.perception == "camera":
        p.error("--viewer cannot be combined with --perception camera (both need the graphics card)")
    if not args.no_gif and not args.viewer:
        from robot_sim.viz import GifRecorder
        recorder = GifRecorder(sim)
    if args.viewer:
        from robot_sim.viz import LiveViewer
        viewer = LiveViewer(sim)

    controller.hold(0.5)                      # let the scene settle before starting
    try:
        if args.command:
            planner.run(args.command)
        if args.interactive:
            print('\nType a command and press Enter. Empty line or "quit" to finish.')
            while True:
                try:
                    command = input("robot> ").strip()
                except EOFError:
                    break
                if command in ("", "quit", "exit"):
                    break
                planner.run(command)
        controller.hold(0.5)
    finally:
        gif = recorder.save(out_dir / "run.gif") if recorder else None
        report = log.write_html(video=gif.name if gif else None)
        log.close()
        if viewer:
            viewer.close()
        sim.close()
        print(f"\nReport: {report}")
        if gif:
            print(f"Video:  {gif}")


if __name__ == "__main__":
    sys.exit(main())
