"""compare_gif.py - one animated GIF, several agents, the same Tetris game.

Every agent plays the same episode seed. The piece sequence depends only on that seed, so all agents
receive identical pieces in identical order. This script checks it at every move: the current and
next piece must match across all agents still playing, and logged episodes must reproduce their
logged boards exactly.

Logged LLM episodes (steps.csv from llm.run_pilot) are replayed move for move. Reference bots
(greedy, beam) play live on the same seed, with no API calls. Frames come from the gym's colored
renderer (TetrisGym.enable_visual / tetris_render). Animation, all boards in lockstep:
- each piece falls from the top toward its landing ghost and locks;
- full rows flash and collapse;
- a header shows the piece everyone is getting next;
- a footer keeps a live leaderboard;
- an agent that tops out shows GAME OVER while the others continue.

  python -m harness.compare_gif --steps runs/explore/iter06/steps.csv --seed 1001 \\
      --arms gemma-4-31B/direct,gpt-oss-20b/low,DeepSeek-V4-Flash/direct --bots beam
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from llm.llm_policy import board_to_str
from TetrisGym_updated import TetrisGym
from tetris_render import (BG, BOARD_BG, GOOD, MUTED, PANEL, TEXT, Move, PanelState, draw_mini, draw_panel, font,
                           step_states, ui_colors)

ACCENTS = [(110, 160, 255), (255, 170, 80), (120, 220, 160), (235, 120, 205), (200, 200, 90), (150, 130, 255)]


@dataclass
class AgentRun:
    name: str
    accent: Tuple[int, int, int]
    first: Tuple[str, str]           # (current, next) before the first move
    moves: List[Move]
    pieces: List[Tuple[str, str]]    # (current, next) faced at each move

    @property
    def final(self) -> Move:
        return self.moves[-1]


def record_agent(seed: int, used_ids: List[int], name: str, accent, logged_boards: Optional[List[str]] = None) -> AgentRun:
    env = TetrisGym(max_steps=None).enable_visual()
    env.reset(seed=seed)
    first = (env.game.current_piece[0], env.game.next_piece[0])
    moves, pieces, lines = [], [], 0
    for k, action in enumerate(used_ids):
        if logged_boards is not None and board_to_str(env.game.board) != logged_boards[k]:
            raise RuntimeError(f"{name}: replay diverged from the log at piece {k}")
        curr, nxt = env.game.current_piece[0], env.game.next_piece[0]
        score_before, lines_before = int(env.game.score), lines
        _, _, done, info = env.step(int(action))
        lines += int(info["lines_cleared"])
        over = bool(env.game.game_over)
        moves.append(Move(piece=curr, before=env.visual.before, locked=env.visual.locked, after=env.visual.cells.copy(),
                          mask_cells=[(int(r), int(c)) for r, c in zip(*np.nonzero(info["placement_mask"]))],
                          cleared_rows=list(info["cleared_rows"]), next_piece=nxt,
                          next_after=None if over else env.game.next_piece[0], score_before=score_before,
                          score_after=int(env.game.score), lines_before=lines_before, lines_after=lines,
                          pieces_after=k + 1, over=over))
        pieces.append((curr, nxt))
        if done:
            break
    return AgentRun(name, accent, first, moves, pieces)


def check_same_pieces(runs: List[AgentRun]) -> int:
    """At every move index, all agents still playing face the same (current, next) piece.
    Returns the number of (agent, move) positions compared."""
    checked = 0
    for k in range(max(len(r.pieces) for r in runs)):
        faced = {r.pieces[k] for r in runs if k < len(r.pieces)}
        if len(faced) != 1:
            raise RuntimeError(f"agents face different pieces at move {k}: {faced}")
        checked += sum(1 for r in runs if k < len(r.pieces))
    return checked


# ---- composition ----------------------------------------------------------------------------------

def _header(width: int, seed: int, k: int, n: int, curr: Optional[str], nxt: Optional[str]) -> Image.Image:
    img = Image.new("RGB", (width, 92), BG)
    d = ImageDraw.Draw(img)
    d.text((16, 10), "Same game, same pieces", fill=TEXT, font=font(24))
    d.text((16, 44), f"seed {seed}  ·  piece {min(k, n)} of {n}", fill=MUTED, font=font(16, False))
    d.text((16, 66), "identical piece sequence for every agent, checked at every move", fill=MUTED, font=font(13, False))
    x = width - 16
    for label, piece in (("next", nxt), ("everyone gets", curr)):
        mini = draw_mini(piece, cell=16, box=(76, 44), bg=BOARD_BG)
        x -= mini.width
        img.paste(mini, (x, 26))
        tw = d.textlength(label, font=font(14))
        d.text((x + (mini.width - tw) / 2, 6), label, fill=MUTED, font=font(14))
        x -= 18
    return img


def _footer(width: int, states: List[PanelState], scale_max: int) -> Image.Image:
    rows = sorted(states, key=lambda s: (-s.score, -s.lines))
    h = 28 + 24 * len(rows)
    img = Image.new("RGB", (width, h), BG)
    d = ImageDraw.Draw(img)
    d.text((16, 6), "LEADERBOARD", fill=MUTED, font=font(13))
    name_w = max(d.textlength(f"{i + 1}. {s.name}", font=font(14)) for i, s in enumerate(rows))
    labels = [f"{s.score} pts · {s.lines} lines" + ("  · game over" if s.over else "") for s in rows]
    text_w = max(d.textlength("9999 pts · 999 lines  · game over", font=font(13, False)), 0)
    bar_x = int(30 + name_w + 16)
    bar_w = max(60, int(width - bar_x - text_w - 26))
    for i, (s, label) in enumerate(zip(rows, labels)):
        y = 26 + 24 * i
        d.rectangle([16, y + 4, 22, y + 16], fill=s.accent)
        d.text((30, y), f"{i + 1}. {s.name}", fill=TEXT if i == 0 else MUTED, font=font(14))
        d.rounded_rectangle([bar_x, y + 3, bar_x + bar_w, y + 17], radius=4, fill=PANEL)
        fill = int(bar_w * min(1.0, s.score / max(1, scale_max)))
        if fill > 0:
            d.rounded_rectangle([bar_x, y + 3, bar_x + fill, y + 17], radius=4, fill=s.accent)
        d.text((bar_x + bar_w + 10, y), label, fill=GOOD if i == 0 else MUTED, font=font(13, False))
    return img


def _canvas(states: List[PanelState], ncols: int, seed: int, k: int, n: int, curr, nxt, scale_max: int, cell: int) -> Image.Image:
    tiles = [draw_panel(s, cell) for s in states]
    gap = 10
    tw, th = tiles[0].size
    nrows = -(-len(tiles) // ncols)
    width = ncols * tw + (ncols + 1) * gap
    head, foot = _header(width, seed, k, n, curr, nxt), _footer(width, states, scale_max)
    img = Image.new("RGB", (width, head.height + nrows * (th + gap) + foot.height), BG)
    img.paste(head, (0, 0))
    for i, t in enumerate(tiles):
        img.paste(t, (gap + (i % ncols) * (tw + gap), head.height + (i // ncols) * (th + gap)))
    img.paste(foot, (0, head.height + nrows * (th + gap)))
    return img


def build_frames(runs: List[AgentRun], seed: int, ncols: int = 2, cell: int = 20, n_drop: int = 3, n_flash: int = 3,
                 ms_drop: int = 50, ms_lock: int = 110, ms_flash: int = 90, ms_settle: int = 220,
                 ms_end: int = 3500) -> Tuple[List[Image.Image], List[int]]:
    n = max(len(r.moves) for r in runs)
    scale_max = max(r.final.score_after for r in runs)
    frames, durations = [], []

    def add(states, k, curr, nxt, ms):
        frames.append(_canvas(states, ncols, seed, k, n, curr, nxt, scale_max, cell))
        durations.append(ms)

    empty = [PanelState(r.name, np.full_like(r.moves[0].before, ""), r.first[1], accent=r.accent) for r in runs]
    add(empty, 0, runs[0].first[0], runs[0].first[1], 900)
    for k in range(n):
        alive = [r for r in runs if k < len(r.moves)]
        curr, nxt = alive[0].pieces[k]
        clear = any(r.moves[k].cleared_rows for r in alive)
        per_agent = []
        for r in runs:
            if k < len(r.moves):
                per_agent.append(step_states(r.moves[k], r.name, r.accent, n_drop, n_flash))
            else:  # finished earlier: hold its final board
                phases = step_states(r.final, r.name, r.accent, n_drop, n_flash)
                fin = phases["settle"][0]
                fin.badge = ""
                per_agent.append({p: [fin] * len(v) for p, v in phases.items()})
        for j in range(n_drop):
            add([a["drop"][j] for a in per_agent], k + 1, curr, nxt, ms_drop)
        add([a["lock"][0] for a in per_agent], k + 1, curr, nxt, ms_lock)
        if clear:
            for j in range(n_flash):
                add([a["flash"][j] for a in per_agent], k + 1, curr, nxt, ms_flash)
            add([a["settle"][0] for a in per_agent], k + 1, curr, nxt, ms_settle)
        elif any(r.moves[k].over for r in alive):
            add([a["settle"][0] for a in per_agent], k + 1, curr, nxt, ms_settle)
    durations[-1] = ms_end
    return frames, durations


TRANSPARENT = 255


def save_gif(frames: List[Image.Image], durations: List[int], path: str, colors: int = 160) -> str:
    """One shared palette for all frames (no color flicker). From the second frame on, pixels that did
    not change are written as a transparent index over the kept previous frame (disposal 1), so long
    runs of unchanged board compress to almost nothing."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    sample = frames[:: max(1, len(frames) // 24)] + [frames[-1]]
    w, h = sample[0].size
    sheet = Image.new("RGB", (w, h * len(sample)))
    for i, f in enumerate(sample):
        sheet.paste(f, (0, i * h))
    fixed = ui_colors(ACCENTS)  # exact piece/UI colors first, so no piece color is ever merged away
    learned = sheet.quantize(colors=min(colors, 255 - len(fixed)), method=Image.Quantize.MEDIANCUT)
    lp = learned.getpalette()[: 3 * min(colors, 255 - len(fixed))]
    extra = [tuple(lp[i:i + 3]) for i in range(0, len(lp), 3)]
    table = (fixed + [c for c in extra if c not in fixed])[:255]
    pal = [v for c in table for v in c]
    palette = Image.new("P", (1, 1))
    palette.putpalette(pal + [0] * (768 - len(pal)))
    idx = [np.asarray(f.quantize(palette=palette, dither=Image.Dither.NONE)) for f in frames]
    out = []
    for i, a in enumerate(idx):
        b = a.copy()
        if i:
            b[a == idx[i - 1]] = TRANSPARENT
        im = Image.fromarray(b.astype(np.uint8), mode="P")
        im.putpalette(pal + [0] * (768 - len(pal)))
        out.append(im)
    out[0].save(path, save_all=True, append_images=out[1:], duration=durations, loop=0, optimize=False,
                disposal=1, transparency=TRANSPARENT)
    return path


def check_gif(path: str, frames: List[Image.Image], positions: List[int]) -> float:
    """Largest per-pixel color error between decoded GIF frames and the rendered frames (palette
    rounding only); raises if any checked frame differs structurally."""
    worst = 0.0
    with Image.open(path) as g:
        for i in positions:
            g.seek(i)
            got = np.asarray(g.convert("RGB"), dtype=int)
            want = np.asarray(frames[i], dtype=int)
            err = np.abs(got - want).max(axis=2)
            if (err > 40).mean() > 0.001:
                raise RuntimeError(f"GIF frame {i} does not match the rendered frame")
            worst = max(worst, float(np.percentile(err, 99.9)))
    return worst


def main() -> None:
    from llm.reference_regret import bot_steps

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", required=True, help="steps.csv of a logged run (llm.run_pilot)")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--arms", default="", help="comma-separated logged arms (default: all)")
    ap.add_argument("--bots", default="", help="comma-separated reference bots played live on the same seed (greedy,beam)")
    ap.add_argument("--pieces", type=int, default=0, help="bot episode length (default: longest logged episode)")
    ap.add_argument("--out", default="")
    ap.add_argument("--ncols", type=int, default=2)
    ap.add_argument("--cell", type=int, default=20, help="board cell size in pixels")
    ap.add_argument("--snapshot", type=int, default=-1, help="also save a PNG of the board after this piece")
    args = ap.parse_args()

    steps = pd.read_csv(args.steps)
    steps = steps[steps["episode_seed"] == args.seed]
    arms = args.arms.split(",") if args.arms else list(dict.fromkeys(steps["arm"]))
    runs = []
    for i, arm in enumerate(arms):
        ep = steps[steps["arm"] == arm].sort_values("turn")
        if ep.empty:
            raise SystemExit(f"arm {arm!r} has no episode for seed {args.seed}")
        model = str(ep["model"].iloc[0]).split("/")[-1] if "model" in ep else arm
        runs.append(record_agent(args.seed, ep["used_id"].tolist(), model, ACCENTS[i % len(ACCENTS)], ep["board"].tolist()))
    pieces = args.pieces or max(len(r.moves) for r in runs)
    for bot in filter(None, args.bots.split(",")):
        ep = bot_steps(bot, args.seed, pieces)
        runs.append(record_agent(args.seed, ep["used_id"].tolist(), f"{bot} search bot",
                                 ACCENTS[len(runs) % len(ACCENTS)], ep["board"].tolist()))
    print(f"same pieces verified at {check_same_pieces(runs)} (agent, move) positions across {len(runs)} agents")
    frames, durations = build_frames(runs, args.seed, ncols=args.ncols, cell=args.cell)
    out = args.out or os.path.join(os.path.dirname(args.steps), f"compare_seed{args.seed}.gif")
    save_gif(frames, durations, out)
    probe = list(range(len(frames)))
    print(f"decoded GIF matches rendered frames (checked all {len(probe)} frames, 99.9th pct color error "
          f"{check_gif(out, frames, probe):.0f}/255)")
    print(f"GIF: {out} ({len(frames)} frames, {sum(durations) / 1000:.0f} s, {os.path.getsize(out) / 1e6:.1f} MB)")
    for r in runs:
        f = r.final
        print(f"  {r.name}: {f.pieces_after} pieces, {f.lines_after} lines, score {f.score_after}"
              f"{', topped out' if f.over else ''}")
    if args.snapshot >= 0:
        k = min(args.snapshot, max(len(r.moves) for r in runs)) - 1
        states = [step_states(r.moves[min(k, len(r.moves) - 1)], r.name, r.accent, 1, 1)["settle"][0] for r in runs]
        alive = [r for r in runs if k + 1 < len(r.pieces)] or runs
        curr, nxt = alive[0].pieces[min(k + 1, len(alive[0].pieces) - 1)]
        png = os.path.splitext(out)[0] + f"_piece{k + 1}.png"
        _canvas(states, args.ncols, args.seed, k + 1, max(len(r.moves) for r in runs), curr, nxt,
                max(r.final.score_after for r in runs), args.cell).save(png)
        print("PNG:", png)


if __name__ == "__main__":
    main()
