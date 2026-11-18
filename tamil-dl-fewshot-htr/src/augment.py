"""Batched, tensor-only augmentation for TRAIN episodes.

Operates on float images in [0, 1] (ink bright). Deliberately has NO flips and NO 90-degree
rotations (Tamil letters are not symmetric) and no CutMix.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Optional

import torch
import torch.nn.functional as F


class Augmenter:
    """Random affine + elastic warp, stroke thickness, blur, noise, brightness/contrast, erasing."""

    def __init__(self, cfg: Mapping[str, Any], seed: int = 0, device: Optional[torch.device] = None) -> None:
        self.cfg = cfg
        self.device = device or torch.device("cpu")
        self.gen = torch.Generator(device="cpu")
        self.gen.manual_seed(seed)

    def state_dict(self) -> dict:
        return {"gen": self.gen.get_state()}

    def load_state_dict(self, state: dict) -> None:
        self.gen.set_state(state["gen"])

    def _u(self, n: int, lo: float, hi: float) -> torch.Tensor:
        return (torch.rand(n, generator=self.gen) * (hi - lo) + lo).to(self.device)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        b, _, h, w = x.shape
        # --- affine: rotation, shear, zoom, shift (grid maps output -> input coords) ---
        rot = self._u(b, -c["rotate_deg"], c["rotate_deg"]) * math.pi / 180
        shear = self._u(b, -c["shear_deg"], c["shear_deg"]) * math.pi / 180
        sx = 1.0 + self._u(b, -c["scale"], c["scale"])
        sy = 1.0 + self._u(b, -c["scale"], c["scale"])
        tx = self._u(b, -c["translate"], c["translate"]) * 2
        ty = self._u(b, -c["translate"], c["translate"]) * 2
        cos, sin = torch.cos(rot), torch.sin(rot)
        a11, a12 = cos / sx, (-sin + torch.tan(shear) * cos) / sx
        a21, a22 = sin / sy, (cos + torch.tan(shear) * sin) / sy
        theta = torch.stack([torch.stack([a11, a12, tx], 1), torch.stack([a21, a22, ty], 1)], 1)
        grid = F.affine_grid(theta, list(x.shape), align_corners=False)
        # --- elastic: smooth random displacement field ---
        if c["elastic_alpha"] > 0:
            g = int(c["elastic_grid"])
            disp = (torch.rand(b, 2, g, g, generator=self.gen) * 2 - 1).to(self.device)
            disp = F.interpolate(disp, size=(h, w), mode="bicubic", align_corners=False)
            grid = grid + disp.permute(0, 2, 3, 1) * (2.0 * c["elastic_alpha"] / w)
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        # --- stroke thickness: dilate or erode a random subset ---
        r = torch.rand(b, generator=self.gen).to(self.device)
        thick = (r < c["stroke_p"] / 2).view(b, 1, 1, 1)
        thin = ((r >= c["stroke_p"] / 2) & (r < c["stroke_p"])).view(b, 1, 1, 1)
        if bool(thick.any()) or bool(thin.any()):
            dil = F.max_pool2d(x, 3, 1, 1)
            ero = -F.max_pool2d(-x, 3, 1, 1)
            x = torch.where(thick, dil, torch.where(thin, 0.5 * (ero + x), x))
        # --- blur ---
        blur = (torch.rand(b, generator=self.gen) < c["blur_p"]).to(self.device).view(b, 1, 1, 1)
        if bool(blur.any()):
            k = torch.tensor([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]], device=x.device).view(1, 1, 3, 3) / 16
            x = torch.where(blur, F.conv2d(x, k, padding=1), x)
        # --- brightness / contrast / noise ---
        con = 1.0 + self._u(b, -c["contrast"], c["contrast"]).view(b, 1, 1, 1)
        bri = self._u(b, -c["brightness"], c["brightness"]).view(b, 1, 1, 1)
        x = (x - 0.5) * con + 0.5 + bri * 0.5
        if c["noise_std"] > 0:
            sd = self._u(b, 0.0, c["noise_std"]).view(b, 1, 1, 1)
            x = x + torch.randn(x.shape, generator=self.gen).to(self.device) * sd
        # --- small random erasing ---
        if c["erase_p"] > 0:
            er = torch.rand(b, generator=self.gen) < c["erase_p"]
            eh = (torch.rand(b, generator=self.gen) * c["erase_frac"] * h).long() + 2
            ew = (torch.rand(b, generator=self.gen) * c["erase_frac"] * w).long() + 2
            ey = (torch.rand(b, generator=self.gen) * (h - eh)).long()
            ex = (torch.rand(b, generator=self.gen) * (w - ew)).long()
            yy = torch.arange(h).view(1, h, 1)
            xx = torch.arange(w).view(1, 1, w)
            m = er.view(b, 1, 1) & (yy >= ey.view(b, 1, 1)) & (yy < (ey + eh).view(b, 1, 1)) \
                & (xx >= ex.view(b, 1, 1)) & (xx < (ex + ew).view(b, 1, 1))
            x = x.masked_fill(m.unsqueeze(1).to(self.device), 0.0)
        return x.clamp_(0.0, 1.0)
