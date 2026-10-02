"""reference_regret.py - score non-LLM reference bots with the same regret oracle as the LLM pilots,
so LLM regret numbers have a yardstick (what does a greedy or beam-search policy score?).

  python -m llm.reference_regret --seeds 1000,1001,1002 --pieces 100 --out runs/explore/reference_regret.csv
"""

from __future__ import annotations

import argparse

import pandas as pd

from harness.bots import make_bot
from harness.play import play_episode
from llm.llm_policy import board_to_str
from llm.oracle import compute_regrets
from TetrisGym_updated import TetrisGym


def bot_steps(spec: str, seed: int, pieces: int) -> pd.DataFrame:
    """Replay a bot's episode, logging the visible state before each decision (as the LLM logs do)."""
    _, rows, _ = play_episode(make_bot(spec), seed, pieces)
    env = TetrisGym(max_steps=None)
    env.reset(seed=seed)
    out = []
    for r in rows:
        out.append({"arm": spec, "episode_seed": seed, "turn": r["turn"], "board": board_to_str(env.game.board),
                    "curr": env.game.current_piece[0], "next": env.game.next_piece[0], "used_id": r["used_id"]})
        _, _, done, _ = env.step(r["used_id"])
        if done:
            break
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1000,1001,1002")
    ap.add_argument("--pieces", type=int, default=100)
    ap.add_argument("--bots", default="greedy,beam")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="runs/explore/reference_regret.csv")
    args = ap.parse_args()
    steps = pd.concat([bot_steps(b, int(s), args.pieces) for b in args.bots.split(",") for s in args.seeds.split(",")],
                      ignore_index=True)
    q = compute_regrets(steps, workers=args.workers, n_samples=2, depth=4, beam_h=2, beam_w=4, beam_samples=1)
    q["top1_beam"] = (q["rank_beam"] == 1).astype(float)
    summ = q.groupby("arm").agg(decisions=("turn", "size"), regret_beam=("regret_beam", "mean"),
                                regret_rollout=("regret_rollout", "mean"), top1_beam=("top1_beam", "mean")).reset_index()
    summ.to_csv(args.out, index=False)
    print(summ.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
