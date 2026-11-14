"""PERCEPTION: finding the blocks with a camera instead of asking the simulator.

By default the skills read block positions straight out of the physics engine
("ground truth").  A real robot cannot do that - it has to work positions out
from sensor data.  This file does it the simplest way that actually works:

  1. Take a colour picture AND a depth picture from a camera looking straight
     down at the table.  (A depth picture stores, for every pixel, how far
     away the surface is.  Cameras that produce both are called RGB-D.)
  2. For each block colour, find the pixels of that colour ("segmentation").
  3. Keep only the highest of those pixels - the block's top face.
  4. Take their average pixel position and convert pixels to metres using
     the camera's geometry (the "pinhole camera model").

Run it:   python run_demo.py "stack red on blue" --perception camera

Limits worth noticing (they are the classic ones):
  * If the arm is in the way, the camera cannot see ("occlusion").  So the
    skills first move the arm aside whenever they need to look.
  * It only knows three colours.  A purple block is invisible to it.
  * It finds position only, not rotation.
"""

from __future__ import annotations

import math

import numpy as np

from . import config as C
from .scene import Sim


class CameraPerception:
    needs_clear_view = True       # tells the skills to move the arm aside first

    def __init__(self, sim: Sim, camera: str = "topdown", size: int = 320):
        self.sim, self.camera, self.size = sim, camera, size
        cam = sim.model.camera(camera)
        self.cam_pos = cam.pos.copy()                       # camera position in the world
        fovy = math.radians(float(cam.fovy[0]))             # vertical field of view
        # Focal length in pixels: how many pixels one metre spans at 1 m distance.
        self.focal = (size / 2) / math.tan(fovy / 2)

    def locate_blocks(self) -> dict[str, np.ndarray]:
        """Return {block_name: (x, y, z) of its centre} for every block seen."""
        rgb = self.sim.render(self.camera, self.size, self.size).astype(float)
        depth = self.sim.render_depth(self.camera, self.size, self.size)
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

        # Colour rules: a pixel is "red" if red clearly dominates, and so on.
        masks = {
            "red":   (r > 90) & (r > 1.8 * g) & (r > 1.8 * b),
            "green": (g > 80) & (g > 1.5 * r) & (g > 1.5 * b),
            "blue":  (b > 90) & (b > 1.5 * r) & (b > 1.4 * g),
        }
        found = {}
        for name, mask in masks.items():
            if mask.sum() < 20:                 # too few pixels: not visible
                continue
            # The top face is the part closest to the camera.
            top = mask & (depth < depth[mask].min() + 0.006)
            rows, cols = np.nonzero(top)
            u, v = cols.mean() + 0.5, rows.mean() + 0.5     # average pixel (column, row)
            d = float(depth[top].mean())                    # distance to the top face

            # Pinhole model: a point d metres away that appears u pixels right
            # of the image centre is  d * u / focal  metres to the right.
            # The camera looks straight down with image-right = world +x and
            # image-up = world +y (rows count downwards, hence the minus).
            x = self.cam_pos[0] + (u - self.size / 2) * d / self.focal
            y = self.cam_pos[1] - (v - self.size / 2) * d / self.focal
            z_top = self.cam_pos[2] - d
            found[name] = np.array([x, y, z_top - C.BLOCK_SIZE / 2])
        return found
