"""oracle.py

Dense per-step decision quality for logged decisions (offline, CPU only).

For a logged state (board, current piece, next-piece preview), every legal placement `a` is valued
with non-clairvoyant clones: pieces after the preview are sampled from independent RNGs, with common
random numbers across actions so the comparison is fair.
  q_rollout(a): place a, then play the greedy 1-ply heuristic policy for `depth` more pieces;
                heuristic score of the final board (TOPOUT added if the game ends).
  q_beam(a):    place a, then run beam_search_planner(horizon, width) from there (the first piece it
                places is the known preview); best leaf heuristic score.
regret = max_a q(a) - q(chosen) >= 0 (rollouts averaged over `n_samples` sampled futures).
"""

from __future__ import annotations

import os
import zlib
from multiprocessing import Pool
from typing import Dict, List

import numpy as np
import pandas as pd

from TetrisGame_updated import TetrisGame
from TetrisGym_updated import TetrisGym
from baselines.beam_search_planner import BeamCfg, BeamSearchPlanner
from diffusion.diffusion_utils_updated import heuristic_score_board
from llm.tetris_tools import placement_outcomes

TOPOUT = -100.0


def env_from_state(board_str: str, curr: str, nxt: str) -> TetrisGym:
    env = TetrisGym(seed=0)
    env.game.board = np.array([[1 if ch == "#" else 0 for ch in row] for row in board_str.split("/")], dtype=int)
    env.game.current_piece = (curr, TetrisGame.TETROMINOES[curr])
    env.game.next_piece = (nxt, TetrisGame.TETROMINOES[nxt])
    env.game.game_over = False
    env.game.score = 0
    env.valid_actions = env.game.get_valid_actions()
    return env


def _greedy(env: TetrisGym) -> int:
    best = max((o.heuristic, -o.action_id) for o in placement_outcomes(env).values())
    return -best[1]


def _board_value(env: TetrisGym, done: bool) -> float:
    return heuristic_score_board(env.game.board) + (TOPOUT if (done and env.game.game_over) else 0.0)


def action_values(env: TetrisGym, n_samples: int = 2, depth: int = 4, beam_h: int = 2, beam_w: int = 4,
                  beam_samples: int = 1, seed: int = 0) -> pd.DataFrame:
    rows = []
    for aid in env.get_valid_action_ids():
        qr, qb = [], []
        for k in range(max(n_samples, beam_samples)):
            s = seed * 1000 + k
            post = env.clone_for_simulation(3 * s)
            _, _, done, _ = post.step(aid)
            if done:
                v = _board_value(post, True)
                qr.append(v)
                qb.append(v)
                continue
            if k < n_samples:
                roll = post.clone_for_simulation(3 * s + 1)
                d = False
                for _ in range(depth):
                    if not roll.get_valid_action_ids():
                        break
                    _, _, d, _ = roll.step(_greedy(roll))
                    if d:
                        break
                qr.append(_board_value(roll, d))
            if k < beam_samples:
                _, best = BeamSearchPlanner(BeamCfg(horizon=beam_h, beam_width=beam_w), sim_seed=3 * s + 2).plan(post)
                qb.append(best if best > -1e17 else _board_value(post, True))
        rows.append({"action_id": aid, "q_rollout": float(np.mean(qr)), "q_beam": float(np.mean(qb))})
    return pd.DataFrame(rows)


def step_quality(board: str, curr: str, nxt: str, chosen: int, n_samples: int = 2, depth: int = 4,
                 beam_h: int = 2, beam_w: int = 4, beam_samples: int = 1, seed: int = 0) -> Dict[str, float]:
    env = env_from_state(board, curr, nxt)
    q = action_values(env, n_samples=n_samples, depth=depth, beam_h=beam_h, beam_w=beam_w,
                      beam_samples=beam_samples, seed=seed)
    out: Dict[str, float] = {"n_legal": len(q)}
    for col, tag in [("q_rollout", "rollout"), ("q_beam", "beam")]:
        ranked = q.sort_values([col, "action_id"], ascending=[False, True]).reset_index(drop=True)
        chosen_row = ranked[ranked["action_id"] == chosen]
        best = float(ranked[col].iloc[0])
        val = float(chosen_row[col].iloc[0]) if not chosen_row.empty else float("nan")
        out[f"q_{tag}_best"] = best
        out[f"q_{tag}_chosen"] = val
        out[f"regret_{tag}"] = best - val
        out[f"rank_{tag}"] = float(chosen_row.index[0] + 1) if not chosen_row.empty else float("nan")
        out[f"best_id_{tag}"] = int(ranked["action_id"].iloc[0])
    return out


def _work(args) -> Dict[str, float]:
    idx, board, curr, nxt, chosen, kw = args
    # Seed by state content: identical states (e.g. the same seed's opening in every arm) get identical
    # sampled futures, so paired arm comparisons don't pick up oracle noise.
    seed = zlib.crc32(f"{board}|{curr}|{nxt}".encode()) % 1_000_000
    out = step_quality(board, curr, nxt, int(chosen), seed=seed, **kw)
    out["row_idx"] = idx
    return out


def compute_regrets(steps: pd.DataFrame, workers: int = 4, **kw) -> pd.DataFrame:
    """Adds oracle columns to a step log (one row per decision; needs board/curr/next/used_id)."""
    tasks = [(i, r.board, r.curr, r.next, r.used_id, kw) for i, r in zip(steps.index, steps.itertuples())]
    if workers > 1:
        with Pool(workers) as pool:
            res = pool.map(_work, tasks, chunksize=4)
    else:
        res = [_work(t) for t in tasks]
    q = pd.DataFrame(res).set_index("row_idx")
    return steps.join(q)


if __name__ == "__main__":
    import time

    env = TetrisGym(seed=0)
    env.reset(seed=3)
    for _ in range(15):
        env.step(_greedy(env))
    from llm.llm_policy import board_to_str

    t = time.perf_counter()
    print(step_quality(board_to_str(env.game.board), env.game.current_piece[0], env.game.next_piece[0],
                       env.get_valid_action_ids()[0]))
    print(f"{time.perf_counter() - t:.2f}s per state", os.cpu_count(), "cpus")
