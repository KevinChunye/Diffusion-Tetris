"""render.py - GIF frames from the repo's own renderer (TetrisGym.render(mode="rgb_array")) plus a
caption strip, side-by-side composition, and experiments.video_utils.save_video for writing."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from experiments.video_utils import save_video

INK = (11, 11, 11)
SURFACE = (252, 252, 251)


def _font(size: int):
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def env_frame(env, info=None) -> np.ndarray:
    """One frame from the existing matplotlib renderer (board, last placement, current/next piece, score)."""
    return env.render(info=info, mode="rgb_array")


def caption(frame: np.ndarray, text: str, height: int = 34) -> np.ndarray:
    img = Image.fromarray(frame)
    out = Image.new("RGB", (img.width, img.height + height), SURFACE)
    out.paste(img, (0, height))
    ImageDraw.Draw(out).text((10, 8), text, fill=INK, font=_font(16))
    return np.asarray(out)


def side_by_side(columns: Sequence[List[np.ndarray]], scale: float = 0.6) -> List[np.ndarray]:
    """Compose per-bot frame lists into one GIF; shorter episodes hold their last frame."""
    n = max(len(c) for c in columns)
    out = []
    for i in range(n):
        tiles = [Image.fromarray(c[min(i, len(c) - 1)]) for c in columns]
        if scale != 1.0:
            tiles = [t.resize((int(t.width * scale), int(t.height * scale))) for t in tiles]
        canvas = Image.new("RGB", (sum(t.width for t in tiles), max(t.height for t in tiles)), SURFACE)
        x = 0
        for t in tiles:
            canvas.paste(t, (x, 0))
            x += t.width
        out.append(np.asarray(canvas))
    return out


def write_gif(frames: List[np.ndarray], path: str, fps: int = 4) -> str:
    return save_video(frames, path, fmt="gif", fps=fps)
