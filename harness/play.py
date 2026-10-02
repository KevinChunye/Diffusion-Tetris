"""play.py - deploy a bot on seeded episodes, print metrics, and optionally record a GIF.

  python -m harness.play --bot beam --seeds 0,1,2 --pieces 100 --gif runs/play/beam.gif
  python -m harness.play --bot greedy --seeds 0 --pieces 200 --gif runs/play/greedy.gif
  python -m harness.play --bot dqn:runs/train/<run>/checkpoint.pt --seeds 0 --gif runs/play/dqn.gif
  python -m harness.play --bot diffusion:runs/train/diffusion/checkpoints/ckpt.pt --seeds 0 --gif runs/play/diff.gif
  python -m harness.play --bot llm:openai/gpt-oss-20b --history stateless --seeds 1000 --pieces 50 --gif runs/play/oss20b.gif

Same episode seed = same piece sequence for every bot, so GIFs/metrics compare bots on identical games.
Add --compare to put several bots side by side in one GIF (e.g. --bot greedy,beam,llm:openai/gpt-oss-20b).
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from harness.bots import Bot, make_bot
from harness.render import caption, env_frame, side_by_side, write_gif
from llm.tetris_tools import safe_step
from TetrisGym_updated import TetrisGym


def play_episode(bot: Bot, episode_seed: int, max_pieces: int, record: bool = False,
                 max_frames: int = 400) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[np.ndarray]]:
    env = TetrisGym(max_steps=max_pieces)
    env.reset(seed=episode_seed)
    bot.reset(episode_seed)
    frames: List[np.ndarray] = []
    rows: List[Dict[str, Any]] = []
    lines = turn = 0
    done = False

    def snap(info=None, tag=""):
        if record and len(frames) < max_frames:
            text = f"{bot.name} | seed {episode_seed} | piece {turn} | lines {lines} | score {env.game.score}{tag}"
            frames.append(caption(env_frame(env, info), text))

    snap()
    t_start = time.perf_counter()
    while not done and turn < max_pieces and env.get_valid_action_ids():
        t0 = time.perf_counter()
        proposed = bot.act(env, turn)
        decide_s = time.perf_counter() - t0
        _, done, info, used, reason = safe_step(env, proposed)
        bot.record(used, info)
        lines += int(info["lines_cleared"])
        turn += 1
        row = {"bot": bot.name, "episode_seed": episode_seed, "turn": turn - 1, "proposed_id": proposed,
               "used_id": used, "fallback_reason": reason, "lines_cleared": int(info["lines_cleared"]),
               "score": int(env.game.score), "decide_s": decide_s}
        row.update(bot.step_info())
        rows.append(row)
        snap(info, "  TOP OUT" if env.game.game_over else "")
    summary = {"bot": bot.name, "episode_seed": episode_seed, "pieces": turn, "lines": lines,
               "score": float(env.game.score), "topped_out": bool(env.game.game_over),
               "fallbacks": int(sum(1 for r in rows if r["fallback_reason"])),
               "sec_per_decision": (time.perf_counter() - t_start) / max(1, turn),
               "cost_usd": float(sum(r.get("cost_usd", 0.0) or 0.0 for r in rows))}
    return summary, rows, frames


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bot", required=True, help="random | greedy | beam[:HxW] | dqn:CKPT | diffusion:CKPT | llm:MODEL "
                                                 "(comma-separate several with --compare)")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--pieces", type=int, default=100)
    ap.add_argument("--gif", default="", help="record the first seed to this GIF")
    ap.add_argument("--compare", action="store_true", help="one side-by-side GIF for all bots in --bot")
    ap.add_argument("--fps", type=int, default=4)
    ap.add_argument("--max_frames", type=int, default=400)
    ap.add_argument("--csv", default="", help="per-step log")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--history", default="stateless")
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--compact_every", type=int, default=16)
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--num_candidates", type=int, default=16)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--mock", action="store_true", help="LLM bots use the offline mock client")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    specs = args.bot.split(",") if args.compare else [args.bot]
    kw = dict(device=args.device, history=args.history, window=args.window, compact_every=args.compact_every,
              max_tokens=args.max_tokens, num_candidates=args.num_candidates, horizon=args.horizon, mock=args.mock)
    summaries, all_rows, gif_columns = [], [], []
    for spec in specs:
        bot = make_bot(spec, **kw)
        for i, seed in enumerate(seeds):
            record = bool(args.gif) and i == 0
            s, rows, frames = play_episode(bot, seed, args.pieces, record=record, max_frames=args.max_frames)
            summaries.append(s)
            all_rows.extend(rows)
            print(f"{s['bot']:>28s} seed {seed}: pieces {s['pieces']:4d} lines {s['lines']:4d} score {s['score']:6.0f} "
                  f"fallbacks {s['fallbacks']:3d}  {s['sec_per_decision']:.3f}s/decision  ${s['cost_usd']:.4f}", flush=True)
            if record:
                gif_columns.append(frames)
                if not args.compare:
                    print("GIF:", write_gif(frames, args.gif, fps=args.fps))
    if args.compare and args.gif and gif_columns:
        print("GIF:", write_gif(side_by_side(gif_columns), args.gif, fps=args.fps))
    table = pd.DataFrame(summaries).groupby("bot", sort=False).agg(
        episodes=("episode_seed", "size"), pieces=("pieces", "mean"), lines=("lines", "mean"), score=("score", "mean"),
        topped_out=("topped_out", "mean"), fallbacks=("fallbacks", "sum"), sec_per_decision=("sec_per_decision", "mean"),
        cost_usd=("cost_usd", "sum"))
    print(table.round(3).to_string())
    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        pd.DataFrame(all_rows).to_csv(args.csv, index=False)


if __name__ == "__main__":
    main()
