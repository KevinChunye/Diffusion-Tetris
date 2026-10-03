"""tetris_render.py - a readable, colored renderer for TetrisGym (PIL only, no matplotlib).

The game board stores occupancy only (0/1). For color, `ColorBoard` remembers which tetromino each
cell came from. It is updated from the step info (`placement_mask`, `locked_board`,
`cleared_rows`) and checked against the real board after every move, so the picture can never
drift from the game.

Drawing primitives:
- `draw_panel`: one agent's view, with the board, a falling piece and its landing ghost, rows
  flashing as they clear, a NEXT box, score / lines / pieces, and a GAME OVER overlay.
- `draw_mini`: a single tetromino, for headers.
- `step_states`: turns one move into animation states (drop, lock, flash, settle) that callers can
  align across agents.

Opt in from the gym:

    env = TetrisGym().enable_visual()
    env.reset(seed=1); obs, _, done, info = env.step(env.get_valid_action_ids()[0])
    frame = env.render(info, mode="pretty")   # numpy RGB array
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PIECE_COLORS: Dict[str, Tuple[int, int, int]] = {
    "I": (0, 200, 235), "O": (245, 205, 0), "T": (170, 80, 240), "S": (60, 200, 95),
    "Z": (240, 65, 80), "J": (55, 115, 245), "L": (245, 145, 35)}
BG = (16, 18, 24)
PANEL = (24, 27, 35)
BOARD_BG = (12, 13, 18)
GRID = (34, 38, 48)
TEXT = (236, 238, 243)
MUTED = (148, 155, 170)
GOOD = (90, 225, 130)
BAD = (255, 95, 95)
FLASH = (255, 255, 255)


def font(size: int, bold: bool = True):
    for name in (("DejaVuSans-Bold.ttf",) if bold else ()) + ("DejaVuSans.ttf",):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _shade(c, f):
    return tuple(max(0, min(255, int(v * f))) for v in c)


def _mix(c, d, t):
    return tuple(int(a + (b - a) * t) for a, b in zip(c, d))


# ---- color tracking -----------------------------------------------------------------------------

class ColorBoard:
    """Piece letter per cell ('' = empty), kept in lockstep with the game board."""

    def __init__(self, height: int, width: int):
        self.height, self.width = height, width
        self.reset()

    def reset(self) -> None:
        self.cells = np.full((self.height, self.width), "", dtype="<U1")
        self.before = self.cells.copy()
        self.locked = self.cells.copy()

    def place(self, piece: str, info: Dict, board_after: np.ndarray) -> None:
        self.before = self.cells.copy()
        locked = self.cells.copy()
        locked[np.asarray(info["placement_mask"]) != 0] = piece
        if not np.array_equal(locked != "", np.asarray(info["locked_board"]) != 0):
            raise RuntimeError("color board does not match the locked board")
        self.locked = locked
        rows = list(info.get("cleared_rows") or [])
        if rows:
            keep = [r for r in range(self.height) if r not in set(rows)]
            locked = np.vstack([np.full((len(rows), self.width), "", dtype="<U1"), locked[keep]])
        self.cells = locked
        if not np.array_equal(self.cells != "", np.asarray(board_after) != 0):
            raise RuntimeError("color board drifted from the game board")


# ---- drawing ------------------------------------------------------------------------------------

def _block(draw: ImageDraw.ImageDraw, x: int, y: int, s: int, color, ghost: bool = False) -> None:
    if ghost:
        draw.rectangle([x + 2, y + 2, x + s - 3, y + s - 3], outline=_shade(color, 0.85), width=2)
        return
    b = max(2, s // 7)
    draw.rectangle([x + 1, y + 1, x + s - 2, y + s - 2], fill=color)
    draw.polygon([(x + 1, y + 1), (x + s - 2, y + 1), (x + s - 2 - b, y + 1 + b), (x + 1 + b, y + 1 + b),
                  (x + 1 + b, y + s - 2 - b), (x + 1, y + s - 2)], fill=_mix(color, (255, 255, 255), 0.35))
    draw.polygon([(x + s - 2, y + s - 2), (x + 1, y + s - 2), (x + 1 + b, y + s - 2 - b),
                  (x + s - 2 - b, y + s - 2 - b), (x + s - 2 - b, y + 1 + b), (x + s - 2, y + 1)],
                 fill=_shade(color, 0.62))


def draw_board(cells: np.ndarray, cell: int = 20, falling: Sequence[Tuple[int, int]] = (), falling_type: str = "",
               ghost: Sequence[Tuple[int, int]] = (), flash_rows: Sequence[int] = (), flash_on: bool = True) -> Image.Image:
    h, w = cells.shape
    img = Image.new("RGB", (w * cell + 4, h * cell + 4), GRID)
    d = ImageDraw.Draw(img)
    d.rectangle([2, 2, w * cell + 1, h * cell + 1], fill=BOARD_BG)
    for c in range(1, w):
        d.line([(2 + c * cell, 2), (2 + c * cell, 2 + h * cell)], fill=(22, 24, 31))
    for r in range(1, h):
        d.line([(2, 2 + r * cell), (2 + w * cell, 2 + r * cell)], fill=(22, 24, 31))
    flash = set(flash_rows)
    for r in range(h):
        for c in range(w):
            if cells[r, c]:
                color = FLASH if (r in flash and flash_on) else PIECE_COLORS[cells[r, c]]
                _block(d, 2 + c * cell, 2 + r * cell, cell, color)
    if falling_type:
        for r, c in ghost:
            _block(d, 2 + c * cell, 2 + r * cell, cell, PIECE_COLORS[falling_type], ghost=True)
        for r, c in falling:
            if r >= 0:
                _block(d, 2 + c * cell, 2 + r * cell, cell, PIECE_COLORS[falling_type])
    return img


def piece_shape(piece: str) -> np.ndarray:
    from TetrisGame_updated import TetrisGame

    return TetrisGame.TETROMINOES[piece][0]


def draw_mini(piece: Optional[str], cell: int = 14, box: Tuple[int, int] = (0, 0), bg=PANEL) -> Image.Image:
    shape = piece_shape(piece) if piece else np.zeros((1, 1), int)
    bw, bh = box if box != (0, 0) else (shape.shape[1] * cell, shape.shape[0] * cell)
    img = Image.new("RGB", (bw, bh), bg)
    if piece:
        d = ImageDraw.Draw(img)
        ox, oy = (bw - shape.shape[1] * cell) // 2, (bh - shape.shape[0] * cell) // 2
        for r, c in zip(*np.nonzero(shape)):
            _block(d, ox + c * cell, oy + r * cell, cell, PIECE_COLORS[piece])
    return img


@dataclass
class PanelState:
    name: str
    cells: np.ndarray
    next_piece: Optional[str] = None
    score: int = 0
    lines: int = 0
    pieces: int = 0
    falling: List[Tuple[int, int]] = field(default_factory=list)
    falling_type: str = ""
    ghost: List[Tuple[int, int]] = field(default_factory=list)
    flash_rows: List[int] = field(default_factory=list)
    flash_on: bool = True
    badge: str = ""
    badge_color: Tuple[int, int, int] = GOOD
    over: bool = False
    accent: Tuple[int, int, int] = MUTED


def draw_panel(st: PanelState, cell: int = 20) -> Image.Image:
    board = draw_board(st.cells, cell, st.falling, st.falling_type, st.ghost, st.flash_rows, st.flash_on)
    side_w, pad, head = 118, 12, 38
    W, H = pad + board.width + pad + side_w + pad, head + board.height + pad
    img = Image.new("RGB", (W, H), PANEL)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W - 1, head - 6], fill=_shade(PANEL, 1.35))
    d.rectangle([0, 0, 5, head - 6], fill=st.accent)
    d.text((pad + 2, 8), st.name, fill=TEXT, font=font(17))
    img.paste(board, (pad, head))
    if st.over:
        ov = Image.new("RGB", board.size, (0, 0, 0))
        img.paste(Image.blend(board, ov, 0.55), (pad, head))
        msg = "GAME OVER"
        tw = d.textlength(msg, font=font(26))
        d.text((pad + (board.width - tw) / 2, head + board.height / 2 - 30), msg, fill=BAD, font=font(26))
        sub = f"after {st.pieces} pieces"
        sw = d.textlength(sub, font=font(15, False))
        d.text((pad + (board.width - sw) / 2, head + board.height / 2 + 4), sub, fill=TEXT, font=font(15, False))
    x = pad + board.width + pad
    y = head
    d.text((x, y), "NEXT", fill=MUTED, font=font(14))
    nb = draw_mini(st.next_piece, cell=18, box=(side_w, 64), bg=BOARD_BG)
    img.paste(nb, (x, y + 20))
    y += 100
    for label, value in (("SCORE", st.score), ("LINES", st.lines), ("PIECES", st.pieces)):
        d.text((x, y), label, fill=MUTED, font=font(14))
        d.text((x, y + 18), str(value), fill=TEXT, font=font(30))
        y += 66
    if st.badge:
        bw = side_w
        d.rounded_rectangle([x, y + 4, x + bw, y + 40], radius=8, fill=_shade(st.badge_color, 0.35),
                            outline=st.badge_color, width=2)
        tw = d.textlength(st.badge, font=font(15))
        d.text((x + (bw - tw) / 2, y + 13), st.badge, fill=st.badge_color, font=font(15))
    return img


def ui_colors(accents: Sequence[Tuple[int, int, int]] = ()) -> List[Tuple[int, int, int]]:
    """Every flat color the renderer paints (pieces with bevel shades and ghosts, UI, accents, and the
    darkened GAME OVER versions), so GIF palettes can reserve them exactly."""
    base = [BG, PANEL, BOARD_BG, GRID, (22, 24, 31), TEXT, MUTED, GOOD, BAD, FLASH, _shade(PANEL, 1.35),
            _shade(GOOD, 0.35), _shade(BAD, 0.35), *accents]
    for c in list(PIECE_COLORS.values()) + [FLASH]:
        base += [c, _mix(c, (255, 255, 255), 0.35), _shade(c, 0.62), _shade(c, 0.85)]
    dark = [tuple(int(v * 0.45) for v in c) for c in base]
    out = []
    for c in base + dark:
        if c not in out:
            out.append(c)
    return out


# ---- one move as animation states -----------------------------------------------------------------

CLEAR_NAMES = {1: "SINGLE", 2: "DOUBLE", 3: "TRIPLE", 4: "TETRIS!"}


@dataclass
class Move:
    """Everything needed to animate one placement (from ColorBoard + the step info)."""
    piece: str
    before: np.ndarray
    locked: np.ndarray
    after: np.ndarray
    mask_cells: List[Tuple[int, int]]
    cleared_rows: List[int]
    next_piece: Optional[str]          # preview while this piece falls
    next_after: Optional[str]          # preview once it has landed
    score_before: int
    score_after: int
    lines_before: int
    lines_after: int
    pieces_after: int
    over: bool


def step_states(m: Move, name: str, accent, n_drop: int, n_flash: int) -> Dict[str, List[PanelState]]:
    """Phases: 'drop' (n_drop states), 'lock' (1), 'flash' (n_flash), 'settle' (1). A move without a
    line clear repeats its locked board in 'flash'/'settle' so several agents stay aligned."""
    top = min(r for r, _ in m.mask_cells)
    drop = []
    for j in range(n_drop):
        off = int(round(top * (1 - j / max(1, n_drop))))  # last drop state is one step above landing
        drop.append(PanelState(name, m.before, m.next_piece, m.score_before, m.lines_before, m.pieces_after - 1,
                               falling=[(r - off, c) for r, c in m.mask_cells], falling_type=m.piece,
                               ghost=m.mask_cells, accent=accent))
    n = len(m.cleared_rows)
    badge = f"+{n} {CLEAR_NAMES.get(n, 'LINES')}" if n else ""
    lock = PanelState(name, m.locked, m.next_after, m.score_after, m.lines_after, m.pieces_after,
                      flash_rows=m.cleared_rows, flash_on=bool(n), badge=badge, accent=accent)
    flash = [PanelState(**{**lock.__dict__, "flash_on": bool(n) and j % 2 == 1}) for j in range(n_flash)]
    settle = PanelState(name, m.after, m.next_after, m.score_after, m.lines_after, m.pieces_after,
                        badge=badge, over=m.over, accent=accent)
    if m.over:
        lock.over = True
        flash = [PanelState(**{**f.__dict__, "over": True}) for f in flash]
    return {"drop": drop, "lock": [lock], "flash": flash, "settle": [settle]}


def render_env(env, info=None, name: str = "", cell: int = 20) -> np.ndarray:
    """Static pretty frame of the gym's current state (used by TetrisGym.render(mode="pretty"))."""
    vis = env.visual
    cells = vis.locked if info is not None else vis.cells
    rows = list(info.get("cleared_rows") or []) if info is not None else []
    st = PanelState(name or "Tetris", cells, env.game.next_piece[0] if env.game.next_piece else None,
                    int(env.game.score), 0, int(env.step_count), flash_rows=rows,
                    badge=f"+{len(rows)} {CLEAR_NAMES.get(len(rows), 'LINES')}" if rows else "",
                    over=bool(env.game.game_over))
    return np.asarray(draw_panel(st, cell))
