"""The O(width) drop-height computation must match the original top-down collision scan exactly."""

from __future__ import annotations

import numpy as np

from TetrisGame_updated import TetrisGame


def _scan_drop_height(game: TetrisGame, piece: np.ndarray, x: int):
    """Original implementation: move down from y=0 until the first invalid position."""
    h, _ = piece.shape
    last_valid_y = None
    for y in range(game.height - h + 1):
        if not game._valid_position(piece, (y, x)):
            break
        last_valid_y = y
    return last_valid_y


def _scan_valid_actions(game: TetrisGame):
    _, rotations = game.current_piece
    out = []
    for rot_idx in range(min(4, len(rotations))):
        piece = rotations[rot_idx]
        for x in range(game.width - piece.shape[1] + 1):
            y = _scan_drop_height(game, piece, x)
            if y is not None and game._valid_position(piece, (y, x)):
                out.append((rot_idx, x))
    return out


def test_drop_height_matches_scan_on_random_boards():
    rng = np.random.default_rng(0)
    game = TetrisGame(seed=0)
    for trial in range(400):
        p_fill = rng.uniform(0.05, 0.8)
        board = (rng.random((game.height, game.width)) < p_fill).astype(int)
        board[: rng.integers(0, game.height)] = 0  # random empty band on top, overhangs/holes below
        game.board = board
        for ptype in game.TETROMINOES_TYPES:
            for piece in game.TETROMINOES[ptype]:
                fresh = piece.copy()  # distinct object, as after a deepcopy of the env
                for x in range(-2, game.width + 2):
                    assert game._find_drop_height(piece, x) == _scan_drop_height(game, piece, x), (trial, ptype, x)
                    assert game._find_drop_height(fresh, x) == _scan_drop_height(game, piece, x), (trial, ptype, x)
            game.current_piece = (ptype, game.TETROMINOES[ptype])
            assert game.get_valid_actions() == _scan_valid_actions(game)
