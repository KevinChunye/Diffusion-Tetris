import random

import numpy as np
import pytest
from PIL import Image

from TetrisGame_updated import TetrisGame
from TetrisGym_updated import TetrisGym


def _game_with_rows(full_rows, hole_col=0, piece="I"):
    g = TetrisGame(seed=0)
    g.reset_board()
    for r in full_rows:
        g.board[r, :] = 1
        g.board[r, hole_col] = 0
    g.current_piece = (piece, TetrisGame.TETROMINOES[piece])
    return g


# ---- the game itself runs correctly ---------------------------------------------------------------

@pytest.mark.parametrize("n, points", [(1, 2), (2, 5), (3, 15), (4, 60)])
def test_vertical_i_into_a_well_clears_rows_and_scores(n, points):
    rows = list(range(20 - n, 20))
    g = _game_with_rows(rows)
    before = g.board.copy()
    info = g.update_board(1, 0)  # vertical I dropped into column 0
    assert info["lines_cleared"] == n and g.score == points
    assert info["cleared_rows"] == rows
    assert np.array_equal(info["pre_clear_board"], before)  # unchanged meaning (agents rely on it)
    assert np.array_equal(info["locked_board"], before + info["placement_mask"])
    assert info["locked_board"][rows].all()
    # cleared rows disappear and everything above (here: the rest of the I) moves down by n
    expected = np.vstack([np.zeros((n, 10), int), np.delete(info["locked_board"], rows, axis=0)])
    assert np.array_equal(g.board, expected)


def test_rows_above_a_clear_fall_down_intact():
    g = _game_with_rows([19])
    g.board[18, 3] = 1
    g.board[17, 3] = 1
    g.update_board(1, 0)  # I fills (16..19, 0): clears row 19
    assert g.board[19, 3] == 1 and g.board[18, 3] == 1 and g.board[17, 3] == 0
    assert g.board[17:20, 0].tolist() == [1, 1, 1]


def test_game_over_when_blocks_reach_the_top_and_when_no_move_fits():
    g = TetrisGame(seed=0)
    g.reset_board()
    g.board[0, 4] = 1
    g.check_game_over()
    assert g.game_over
    env = TetrisGym(max_steps=None)
    env.reset(seed=3)
    done, n = False, 0
    while not done:  # stacking everything in one column must end the game
        ids = env.get_valid_action_ids()
        _, _, done, _ = env.step(ids[0])
        n += 1
        assert n < 200
    assert env.game.game_over
    with pytest.raises(RuntimeError):
        env.step(0)


def test_same_seed_same_pieces_whatever_the_moves():
    a, b = TetrisGym(max_steps=None), TetrisGym(max_steps=None)
    a.reset(seed=42)
    b.reset(seed=42)
    rng = random.Random(1)
    for _ in range(40):
        assert (a.game.current_piece[0], a.game.next_piece[0]) == (b.game.current_piece[0], b.game.next_piece[0])
        ida, idb = a.get_valid_action_ids(), b.get_valid_action_ids()
        _, _, da, _ = a.step(ida[0])
        _, _, db, _ = b.step(idb[rng.randrange(len(idb))])
        if da or db:
            break


# ---- colored renderer stays in lockstep with the game ----------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_color_board_tracks_the_real_board_over_whole_games(seed):
    from harness.bots import make_bot

    for spec in ("random", "greedy"):
        env = TetrisGym(max_steps=None).enable_visual()
        env.reset(seed=seed)
        bot = make_bot(spec)
        bot.reset(seed)
        lines = 0
        for turn in range(150):
            _, _, done, info = env.step(bot.act(env, turn))
            lines += info["lines_cleared"]
            assert np.array_equal(env.visual.cells != "", env.game.board != 0)
            assert set(np.unique(env.visual.cells)) <= set("IJLOSZT") | {""}
            if done:
                break
        frame = env.render(info, mode="pretty")
        assert frame.ndim == 3 and frame.shape[2] == 3
        if spec == "greedy":
            assert lines > 0  # the game really clears lines


def test_visual_is_opt_in_and_never_leaks_into_lookahead():
    plain = TetrisGym()
    assert plain.visual is None
    with pytest.raises(RuntimeError):
        plain.render(None, mode="pretty")
    env = TetrisGym(max_steps=None).enable_visual()
    env.reset(seed=5)
    env.step(env.get_valid_action_ids()[0])
    snapshot = env.visual.cells.copy()
    sim = env.clone_for_simulation(sim_seed=9)
    assert sim.visual is None
    for _ in range(5):
        _, _, done, _ = sim.step(sim.get_valid_action_ids()[-1])
        if done:
            break
    assert np.array_equal(env.visual.cells, snapshot)


# ---- comparison GIF -------------------------------------------------------------------------------

def _random_ids(seed, n, pick):
    env = TetrisGym(max_steps=None)
    env.reset(seed=seed)
    ids = []
    for _ in range(n):
        valid = env.get_valid_action_ids()
        ids.append(valid[pick(len(valid))])
        _, _, done, _ = env.step(ids[-1])
        if done:
            break
    return ids


def test_comparison_gif_aligns_agents_and_round_trips(tmp_path):
    from harness.compare_gif import build_frames, check_gif, check_same_pieces, record_agent, save_gif

    rng = random.Random(0)
    runs = [record_agent(11, _random_ids(11, 12, lambda n: 0), "left", (200, 100, 100)),
            record_agent(11, _random_ids(11, 12, lambda n: rng.randrange(n)), "random", (100, 200, 100))]
    assert check_same_pieces(runs) > 0
    frames, durations = build_frames(runs, 11, ncols=2, n_drop=2, n_flash=2)
    assert len(frames) == len(durations) and len({f.size for f in frames}) == 1
    path = save_gif(frames, durations, str(tmp_path / "cmp.gif"))
    check_gif(path, frames, list(range(len(frames))))
    with Image.open(path) as g:
        assert g.n_frames == len(frames)
    other = record_agent(12, _random_ids(12, 12, lambda n: 0), "other seed", (100, 100, 200))
    with pytest.raises(RuntimeError):
        check_same_pieces(runs + [other])
