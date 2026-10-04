"""decision_model.py - a Tetris agent backed by a decision model instead of a text generator.

Decision models score a caller-supplied set of options in one forward pass and return a probability
for each, with no text to generate or parse. TypeSafe's Jev is the proprietary, API-only example;
Intelif (github.com/SkAndMl/intelif, Qwen3-4B + LoRA + linear scorer, weights on Hugging Face) is an
open model that is wire-compatible with Jev's /v1/systemone format. We self-host Intelif here.

Each legal placement of the current piece is one option. The state and question follow Intelif's own
Tetris example (board rows as '#'/'.', the piece, an instruction to clear lines and avoid holes), and
each option's description gives the same simulated outcome our annotated LLM prompts show (lines
cleared, new holes, stack height change, surface change), plus the next piece.

Run inside the Python 3.12 environment that has Intelif installed (CPU here, no GPU):
  /home/user/intelif-venv/bin/python -m llm.decision_model --seeds 1000,1001,1002 --pieces 100 \\
      --out runs/explore/scaleup/intelif_seeds1000
Writes calls.jsonl (one raw record per call: every option with its description and probability, latency,
input tokens), steps.csv in the same schema as LLM game logs (so llm.reference_regret and harness.compare_gif
work unchanged), episodes.csv and manifest.json.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from TetrisGym_updated import TetrisGym

INSTRUCTIONS = ("You are playing Tetris. Where should the {piece} piece land? "
                "Clear lines, avoid holes and keep the stack low and flat.")


def _stats(board: np.ndarray) -> Tuple[int, int, int]:
    h, w = board.shape
    filled = board != 0
    heights = [h - int(np.argmax(filled[:, c])) if filled[:, c].any() else 0 for c in range(w)]
    holes = sum(int((~filled[h - heights[c]:, c]).sum()) for c in range(w))
    bump = sum(abs(a - b) for a, b in zip(heights, heights[1:]))
    return max(heights), holes, bump


def _after(env: TetrisGym, rot: int, x: int) -> Tuple[np.ndarray, int]:
    game = env.game
    piece = game.current_piece[1][rot]
    ph, pw = piece.shape
    y = game._find_drop_height(piece, x)
    board = game.board.copy()
    board[y:y + ph, x:x + pw] += piece
    full = np.all(board == 1, axis=1)
    lines = int(full.sum())
    if lines:
        board = np.vstack([np.zeros((lines, game.width), dtype=board.dtype), board[~full]])
    return board, lines


def describe(before: np.ndarray, after: np.ndarray, lines: int) -> str:
    dh, dholes, dbump = (a - b for a, b in zip(_stats(after), _stats(before)))
    return ", ".join([
        f"clears {lines} line{'s' * (lines > 1)}" if lines else "clears no lines",
        f"creates {dholes} new hole{'s' * (dholes > 1)}" if dholes > 0 else "no new holes",
        f"raises the stack by {dh}" if dh > 0 else "keeps the stack low",
        "surface gets bumpier" if dbump > 0 else "surface stays flat"])


def question(env: TetrisGym) -> Tuple[Dict, Dict[str, str], str, Dict[str, int]]:
    """(state, criteria, instructions, option key -> action id) for the current position."""
    board = env.game.board
    piece, nxt = env.game.current_piece[0], env.game.next_piece[0]
    criteria, ids = {}, {}
    for aid in env.get_valid_action_ids():
        rot, x = env.id_to_action[aid]
        after, lines = _after(env, rot, x)
        key = f"rotation {rot}, column {x}"
        criteria[key] = describe(board, after, lines)
        ids[key] = aid
    state = {"board": "\n".join("".join("#" if c else "." for c in row) for row in board), "piece": piece,
             "next piece": nxt}
    return state, criteria, INSTRUCTIONS.format(piece=piece), ids


def play(model, seed: int, pieces: int, arm: str, calls_log: Path = None) -> Tuple[List[Dict], Dict]:
    from llm.llm_policy import board_to_str

    env = TetrisGym(max_steps=None)
    env.reset(seed=seed)
    rows, lines = [], 0
    for turn in range(pieces):
        if env.game.game_over or not env.get_valid_action_ids():
            break
        state, criteria, instr, ids = question(env)
        board_s, curr, nxt = board_to_str(env.game.board), env.game.current_piece[0], env.game.next_piece[0]
        t0 = time.perf_counter()
        resp = model.system_one(state, {"move": {"type": "choice", "criteria": criteria, "instructions": instr}})
        latency = time.perf_counter() - t0
        ans = resp.choices["move"]
        aid = ids[ans.choice]
        if calls_log is not None:  # raw record of every call: full distribution over options, kept for later analysis
            probs = {k: float(v) for k, v in ans.probabilities.items()}
            rec = {"event": "call", "arm": arm, "episode_seed": seed, "turn": turn, "t_start_unix": time.time() - latency,
                   "latency_s": latency, "input_tokens": int(resp.usage.input_tokens), "choice": ans.choice,
                   "confidence": float(ans.confidence), "options": {k: {"action_id": ids[k], "description": criteria[k],
                                                                       "p": probs.get(k)} for k in criteria},
                   "state": state, "instructions": instr}
            with open(calls_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        _, _, done, info = env.step(aid)
        lines += int(info["lines_cleared"])
        rows.append({"arm": arm, "model": "intelif-qwen3-4b@v0.1", "episode_seed": seed, "turn": turn, "board": board_s,
                     "curr": curr, "next": nxt, "legal_ids": json.dumps(sorted(ids.values())), "proposed_id": aid,
                     "used_id": aid, "fallback_reason": "", "lines_cleared": int(info["lines_cleared"]),
                     "score_after": int(env.game.score), "done": bool(done), "n_legal": len(ids),
                     "top_prob": float(ans.confidence), "input_tokens": int(resp.usage.input_tokens),
                     "latency_s": latency, "ok": True})
        print(f"[intelif] seed {seed} piece {turn + 1}: {ans.choice} p={ans.confidence:.2f} "
              f"{latency:.1f}s lines {lines} score {env.game.score}", flush=True)
        if done:
            break
    ep = {"arm": arm, "episode_seed": seed, "pieces": len(rows), "lines": lines, "score": int(env.game.score),
          "topped_out": bool(env.game.game_over), "latency_p50_s": float(np.median([r["latency_s"] for r in rows])),
          "input_tokens_mean": float(np.mean([r["input_tokens"] for r in rows]))}
    return rows, ep


def main() -> None:
    import pandas as pd
    import torch
    from intelif import Intelif

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--pieces", type=int, default=100)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    out = Path(args.out)
    if any((out / f).exists() for f in ("steps.csv", "calls.jsonl", "manifest.json")):
        raise FileExistsError(f"fresh directory required: {out}")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    t_load = time.perf_counter()
    model = Intelif.from_pretrained(device="cpu", dtype=args.dtype)
    load_s = time.perf_counter() - t_load
    manifest = {"model": "UserMoonlight/intelif-qwen3-4b", "revision": "v0.1", "device": "cpu", "dtype": args.dtype,
                "threads": args.threads, "cpu": platform.processor() or platform.machine(), "torch": torch.__version__,
                "load_s": load_s, "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "seeds": args.seeds, "pieces": args.pieces, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    eps = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        rows, ep = play(model, seed, args.pieces, "intelif-qwen3-4b/decision", out / "calls.jsonl")
        pd.DataFrame(rows).to_csv(out / "steps.csv", mode="a", header=not (out / "steps.csv").exists(), index=False)
        eps.append(ep)
        pd.DataFrame(eps).to_csv(out / "episodes.csv", index=False)
    print(pd.DataFrame(eps).to_string(index=False))


if __name__ == "__main__":
    main()
