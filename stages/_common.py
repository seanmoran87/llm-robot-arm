"""Small helpers shared by the stage scripts."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))          # lets "import robot_sim" work from this folder
OUT = ROOT / "runs" / "stages"
OUT.mkdir(parents=True, exist_ok=True)


def attach_outputs(sim, name: str, viewer: bool):
    """Record a GIF, or with --viewer open the live window instead.  Returns a function
    to call at the end that saves the GIF and prints where it went."""
    from robot_sim.viz import GifRecorder, LiveViewer
    # One or the other: the live window and the offscreen recorder do not
    # share the graphics card reliably.
    live = LiveViewer(sim) if viewer else None
    recorder = None if viewer else GifRecorder(sim)

    def finish():
        if live:
            live.close()
        if recorder:
            print(f"\nSaved recording: {recorder.save(OUT / f'{name}.gif')}")
    return finish
