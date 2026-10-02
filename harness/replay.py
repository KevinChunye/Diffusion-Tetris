"""replay.py - turn logged LLM episodes (steps.csv from llm.run_pilot) into GIFs, side by side.

Each logged episode is replayed exactly: the episode seed fixes the piece sequence and `used_id` is
the action that was executed, so the GIF shows precisely what each model/arm did on the same game.

  python -m harness.replay --steps runs/explore/iter01/steps.csv --seed 1000 --out runs/explore/iter01/replay_seed1000.gif
  python -m harness.replay --steps runs/explore/iter01/steps.csv --seed 1000 --arms stateless,append --max_frames 120
"""

from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np
import pandas as pd
from PIL import Image

from harness.render import caption, env_frame, side_by_side, write_gif
from llm.llm_policy import board_to_str
from TetrisGym_updated import TetrisGym


def replay_frames(episode_seed: int, steps: pd.DataFrame, label: str, max_frames: int = 400) -> List[np.ndarray]:
    """steps: one episode's rows sorted by turn (needs used_id; board is checked when present)."""
    env = TetrisGym(max_steps=None)
    env.reset(seed=episode_seed)
    lines = 0
    frames = [caption(env_frame(env), f"{label} | piece 0 | lines 0 | score 0")]
    for turn, row in enumerate(steps.itertuples()):
        if "board" in steps.columns and board_to_str(env.game.board) != row.board:
            raise RuntimeError(f"{label}: replay diverged from the log at turn {turn}")
        _, _, done, info = env.step(int(row.used_id))
        lines += int(info["lines_cleared"])
        tag = "  TOP OUT" if env.game.game_over else ""
        if len(frames) < max_frames:
            frames.append(caption(env_frame(env, info), f"{label} | piece {turn + 1} | lines {lines} | score {env.game.score}{tag}"))
        if done:
            break
    return frames


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--arms", default="", help="comma-separated arm names (default: all, in log order)")
    ap.add_argument("--out", default="")
    ap.add_argument("--fps", type=int, default=4)
    ap.add_argument("--max_frames", type=int, default=150)
    ap.add_argument("--scale", type=float, default=0.6)
    ap.add_argument("--ncols", type=int, default=0, help="tiles per row (default: one row)")
    ap.add_argument("--snapshot", type=int, default=-1, help="also save the side-by-side frame at this piece as PNG")
    args = ap.parse_args()

    steps = pd.read_csv(args.steps)
    steps = steps[steps["episode_seed"] == args.seed]
    arms = args.arms.split(",") if args.arms else list(dict.fromkeys(steps["arm"]))
    columns = []
    for arm in arms:
        ep = steps[steps["arm"] == arm].sort_values("turn")
        model = str(ep["model"].iloc[0]).split("/")[-1] if "model" in ep.columns and len(ep) else ""
        columns.append(replay_frames(args.seed, ep, f"{arm} ({model})" if model else arm, args.max_frames))
    out = args.out or os.path.join(os.path.dirname(args.steps), f"replay_seed{args.seed}.gif")
    frames = side_by_side(columns, scale=args.scale, ncols=args.ncols)
    print("GIF:", write_gif(frames, out, fps=args.fps))
    if args.snapshot >= 0:
        png = os.path.splitext(out)[0] + f"_piece{args.snapshot}.png"
        Image.fromarray(frames[min(args.snapshot, len(frames) - 1)]).save(png)
        print("PNG:", png)


if __name__ == "__main__":
    main()
