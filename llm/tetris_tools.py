"""tetris_tools.py

Env helpers shared by LLM agents, reference policies and the regret oracle:
  - placement_outcomes(env): what each legal placement does (lines, height, holes...), no env mutation
  - fallback_action(env):    lowest-resulting-height legal placement
  - safe_step(env, aid):     env.step that never raises on illegal/unparseable actions
  - run_reference_episode:   random-legal and beam-search references for normalized score
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

from TetrisGym_updated import TetrisGym
from agent.value_dqn import board_props
from diffusion.diffusion_utils_updated import heuristic_score_board

SCORE_TABLE = {0: 0, 1: 2, 2: 5, 3: 15, 4: 60}


@dataclass
class Outcome:
    action_id: int
    rot: int
    x: int
    lines: int
    max_height: int
    agg_height: int
    holes: int
    bumpiness: int
    heuristic: float
    board: np.ndarray


def placement_outcomes(env: TetrisGym) -> Dict[int, Outcome]:
    """Board after each legal placement of the current piece (lines cleared), without touching env."""
    game = env.game
    _, rotations = game.current_piece
    tops = game._column_tops()
    out: Dict[int, Outcome] = {}
    for (rot, x) in env.valid_actions:
        piece = rotations[rot]
        h, w = piece.shape
        y = game._find_drop_height(piece, x, tops)
        if y is None:
            continue
        board = game.board.copy()
        board[y:y + h, x:x + w] += piece
        full = np.all(board == 1, axis=1)
        lines = int(full.sum())
        if lines:
            board = np.vstack([np.zeros((lines, game.width), dtype=board.dtype), board[~full]])
        feats = board_props(board.astype(np.uint8))  # [lines(0 here), max_h, min_h, total_h, max_bump, total_bump, holes]
        aid = env.action_to_id[(rot, x)]
        out[aid] = Outcome(aid, rot, x, lines, int(feats[1]), int(feats[3]), int(feats[6]), int(feats[5]),
                           float(heuristic_score_board(board)), board)
    return out


def fallback_action(env: TetrisGym, outcomes: Optional[Dict[int, Outcome]] = None) -> int:
    """Lowest resulting max height; ties -> lower aggregate height, fewer holes, smaller action id."""
    outcomes = outcomes if outcomes is not None else placement_outcomes(env)
    return int(min((o.max_height, o.agg_height, o.holes, o.action_id) for o in outcomes.values())[-1])


def safe_step(env: TetrisGym, action_id: Optional[int]):
    """env.step that replaces an illegal/unparseable action with the fallback instead of raising.

    Returns (obs, done, info, used_action_id, fallback_reason) where fallback_reason is "" when the
    proposed action was legal, else "unparseable" or "illegal"."""
    valid = env.get_valid_action_ids()
    reason = ""
    if action_id is None:
        reason = "unparseable"
    elif int(action_id) not in valid:
        reason = "illegal"
    aid = fallback_action(env) if reason else int(action_id)
    obs, _, done, info = env.step(aid)
    return obs, done, info, aid, reason


# ---------------------------------------------------------------------------------------------
# Reference policies for normalized score (0 = random legal policy, 1 = beam_search default).
# ---------------------------------------------------------------------------------------------

def run_reference_episode(policy: str, episode_seed: int, max_pieces: int, sim_seed: int = 0) -> Dict[str, float]:
    from baselines.beam_search_planner import BeamCfg, BeamSearchPlanner

    env = TetrisGym(max_steps=max_pieces)
    env.reset(seed=episode_seed)
    if policy == "random":
        rng = random.Random(10_000 + episode_seed)
        act: Callable[[TetrisGym], int] = lambda e: rng.choice(e.get_valid_action_ids())
    elif policy == "beam":
        planner = BeamSearchPlanner(BeamCfg(horizon=3, beam_width=16), sim_seed=sim_seed + episode_seed)
        act = lambda e: planner.plan(e)[0]
    else:
        raise ValueError(policy)
    lines = pieces = 0
    done = False
    while not done and pieces < max_pieces:
        if not env.get_valid_action_ids():
            break
        _, _, done, info = env.step(act(env))
        lines += int(info["lines_cleared"])
        pieces += 1
    return {"policy": policy, "episode_seed": episode_seed, "max_pieces": max_pieces, "score": float(env.game.score),
            "lines_cleared": lines, "pieces_placed": pieces, "topped_out": bool(env.game.game_over)}


def reference_table(episode_seeds: List[int], max_pieces: int, cache_csv: str) -> "pd.DataFrame":
    """Random + beam reference results on the given seeds, cached on disk (computed once)."""
    import os

    import pandas as pd

    have = pd.read_csv(cache_csv) if os.path.exists(cache_csv) else pd.DataFrame(
        columns=["policy", "episode_seed", "max_pieces", "score", "lines_cleared", "pieces_placed", "topped_out"])
    rows = []
    for seed in episode_seeds:
        for policy in ["random", "beam"]:
            hit = have[(have["policy"] == policy) & (have["episode_seed"] == seed) & (have["max_pieces"] == max_pieces)]
            if hit.empty:
                rows.append(run_reference_episode(policy, seed, max_pieces))
    if rows:
        have = pd.concat([have, pd.DataFrame(rows)], ignore_index=True)
        os.makedirs(os.path.dirname(os.path.abspath(cache_csv)), exist_ok=True)
        have.to_csv(cache_csv, index=False)
    sel = have[have["episode_seed"].isin(episode_seeds) & (have["max_pieces"] == max_pieces)]
    return sel.reset_index(drop=True)


def normalized_scores(episodes: "pd.DataFrame", refs: "pd.DataFrame", value: str = "score") -> "pd.DataFrame":
    """Adds norm_<value> = (x - random) / (beam - random) using the same episode seeds."""
    piv = refs.pivot_table(index="episode_seed", columns="policy", values=value, aggfunc="mean")
    piv = piv.rename(columns={"random": f"ref_random_{value}", "beam": f"ref_beam_{value}"}).reset_index()
    out = episodes.merge(piv, on="episode_seed", how="left")
    denom = (out[f"ref_beam_{value}"] - out[f"ref_random_{value}"]).replace(0, np.nan)
    out[f"norm_{value}"] = (out[value] - out[f"ref_random_{value}"]) / denom
    return out
