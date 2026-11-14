"""VISUALISATION: watch the robot live, or record a GIF.

Both tools work the same way: they register a "listener" - a function the
Sim calls after every physics step.  The control code does not know or care
that it is being watched.
"""

from __future__ import annotations

import time
from pathlib import Path

from PIL import Image

from . import config as C
from .scene import Sim


class GifRecorder:
    """Grabs a camera frame several times per simulated second."""

    def __init__(self, sim: Sim, fps: int = 10, camera: str = "overview",
                 width: int = 400, height: int = 300):
        self.fps, self.camera, self.width, self.height = fps, camera, width, height
        self.frames: list[Image.Image] = []
        self._every = max(1, round(1.0 / (fps * C.TIMESTEP)))   # physics steps per frame
        self._count = 0
        sim.step_listeners.append(self._on_step)

    def _on_step(self, sim: Sim) -> None:
        self._count += 1
        if self._count % self._every == 0:
            # Reducing to a small colour palette right away keeps memory low.
            frame = Image.fromarray(sim.render(self.camera, self.width, self.height))
            self.frames.append(frame.quantize(colors=128, method=Image.Quantize.FASTOCTREE,
                                              dither=Image.Dither.NONE))

    def save(self, path: str | Path, speedup: float = 1.5) -> Path | None:
        """Write the GIF.  speedup > 1 plays it faster than real time."""
        if not self.frames:
            return None
        path = Path(path)
        self.frames[0].save(path, save_all=True, append_images=self.frames[1:],
                            duration=int(1000 / (self.fps * speedup)), loop=0, optimize=False)
        return path


class LiveViewer:
    """Opens MuJoCo's interactive 3D window and keeps it in step with the sim.

    Mouse: left-drag rotates, right-drag pans, scroll zooms.
    Needs a real screen (your own PC).  On macOS, run the script with
    `mjpython` instead of `python`.
    """

    def __init__(self, sim: Sim, realtime: bool = True):
        import mujoco.viewer
        self.viewer = mujoco.viewer.launch_passive(sim.model, sim.data)
        self.realtime = realtime
        self._wall_start = time.perf_counter()
        self._sim_start = sim.time
        self._count = 0
        sim.step_listeners.append(self._on_step)

    def _on_step(self, sim: Sim) -> None:
        self._count += 1
        if self._count % 8:                # refresh the window ~60 times a second
            return
        if not self.viewer.is_running():   # the user closed the window
            return
        self.viewer.sync()
        if self.realtime:
            # The simulation computes far faster than real life; wait so that
            # one simulated second takes one real second to watch.
            ahead = (sim.time - self._sim_start) - (time.perf_counter() - self._wall_start)
            if ahead > 0:
                time.sleep(ahead)

    def close(self) -> None:
        """Close the window and WAIT until it has really gone.  The window
        runs in its own background thread; exiting Python while that thread
        is still tidying up can crash, so we give it a moment."""
        self.viewer.close()
        deadline = time.perf_counter() + 3.0
        while self.viewer.is_running() and time.perf_counter() < deadline:
            time.sleep(0.02)
        time.sleep(0.3)
