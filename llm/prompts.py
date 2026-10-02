"""prompts.py

Prompt construction for LLM Tetris agents. Static content (rules, piece shapes, output format) goes
FIRST so it forms a byte-identical prefix across calls (prefix-cache friendly); the board, pieces and
legal placement ids come after it.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional

import numpy as np

from TetrisGame_updated import TetrisGame

PIECES = TetrisGame.TETROMINOES_TYPES


def _shape_lines(arr: np.ndarray) -> List[str]:
    return ["".join("#" if v else "." for v in row) for row in arr]


def piece_shapes_text() -> str:
    lines = []
    for p in PIECES:
        rots = TetrisGame.TETROMINOES[p]
        parts = []
        for r, arr in enumerate(rots):
            parts.append(f"rot {r}: " + "/".join(_shape_lines(arr)))
        lines.append(f"{p}: " + "; ".join(parts))
    return "\n".join(lines)


def system_prompt(annotated: bool = True, json_output: bool = True) -> str:
    listing = ("Each legal placement is listed as `id: rot R, cols A-B -> lines L, height H, holes N`, where "
               "L/H/N describe the board right after that drop (lines cleared, tallest column, covered empty cells)."
               if annotated else
               "Each legal placement is listed as `id: rot R, cols A-B` (rotation index, leftmost and rightmost column).")
    out = ('Reply with only a JSON object: {"action_id": <id>} using one of the listed ids.' if json_output
           else "Reply with only the id of your placement.")
    return (
        "You are playing Tetris on a board 10 columns wide and 20 rows tall.\n"
        "Each turn you choose where the CURRENT piece goes. A placement is a rotation plus a column offset; "
        "the piece is then dropped straight down from the top until it lands.\n"
        "Full rows are cleared. Points per placement: 1 line = 2, 2 lines = 5, 3 lines = 15, 4 lines = 60. "
        "The game ends when a block reaches the top row, so survival matters more than anything else.\n"
        "Good play keeps the stack low and flat, avoids holes (empty cells with a block above them), and keeps "
        "one column open so I pieces can clear several lines at once.\n"
        "Board legend: '#' filled, '.' empty; the first board line is the top row, the last is the bottom row; "
        "columns are numbered 0-9 from left to right.\n"
        "Piece shapes by rotation index (rows separated by '/', '#' = block):\n"
        f"{piece_shapes_text()}\n"
        f"{listing}\n"
        f"{out}"
    )


def render_board(board: np.ndarray) -> str:
    return "\n".join("".join("#" if v else "." for v in row) for row in board)


def render_state(turn: int, board: np.ndarray, curr: str, nxt: str, score: int, lines: int,
                 legal: List[Dict], annotated: bool = True) -> str:
    """`legal`: list of dicts with action_id, rot, x, w (+ lines, max_height, holes if annotated)."""
    rows = []
    for a in legal:
        s = f"{a['action_id']}: rot {a['rot']}, cols {a['x']}-{a['x'] + a['w'] - 1}"
        if annotated:
            s += f" -> lines {a['lines']}, height {a['max_height']}, holes {a['holes']}"
        rows.append(s)
    return (
        f"Turn {turn}. Score {score}, lines cleared {lines}.\n"
        f"Current piece: {curr}. Next piece: {nxt}.\n"
        f"Board:\n{render_board(board)}\n"
        f"Legal placements:\n" + "\n".join(rows)
    )


def render_summary(first_turn: int, last_turn: int, pieces: int, lines: int, score_gain: int,
                   h0: int, h1: int, holes0: int, holes1: int) -> str:
    return (f"Summary of turns {first_turn}-{last_turn}: placed {pieces} pieces, cleared {lines} lines "
            f"(+{score_gain} points); max height {h0} -> {h1}; holes {holes0} -> {holes1}.")


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
_INT = re.compile(r"-?\d+")


def parse_action(text: str, legal_ids: List[int]) -> Optional[int]:
    """JSON {"action_id": N} (fences tolerated), else the last bare integer. None if unparseable.
    The returned id may still be illegal; legality is checked by safe_step."""
    if not text:
        return None
    t = _FENCE.sub("", text.strip()).strip()
    try:
        obj = json.loads(t)
        if isinstance(obj, dict) and "action_id" in obj:
            return int(obj["action_id"])
        if isinstance(obj, int):
            return int(obj)
    except (ValueError, TypeError):
        pass
    m = re.search(r'"?action_id"?\s*[:=]\s*(-?\d+)', t)
    if m:
        return int(m.group(1))
    ints = _INT.findall(t)
    return int(ints[-1]) if ints else None
