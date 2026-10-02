"""Harness: bots deploy through one interface; GIFs come from the repo renderer; logged episodes replay exactly."""

from __future__ import annotations

import pandas as pd
from PIL import Image

from harness.bots import make_bot
from harness.play import play_episode
from harness.render import side_by_side, write_gif
from harness.replay import replay_frames
from llm.run_pilot import run


def test_play_records_gif_for_several_bots(tmp_path):
    columns = []
    for spec in ["random", "greedy", "beam:1x4", "llm:openai/gpt-oss-20b"]:
        summary, rows, frames = play_episode(make_bot(spec, mock=True), episode_seed=4, max_pieces=6, record=True)
        assert summary["pieces"] == len(rows) == 6
        assert len(frames) == 7  # initial board + one per piece
        columns.append(frames)
    path = write_gif(side_by_side(columns, scale=0.3), str(tmp_path / "cmp.gif"), fps=4)
    assert Image.open(path).n_frames == 7


def test_replay_reproduces_logged_episode_exactly(tmp_path):
    cfg = {"iteration": 0, "name": "replay", "model": "openai/gpt-oss-20b", "seeds": [11], "max_pieces": 6,
           "budget_usd": 1.0, "reference_csv": str(tmp_path / "refs.csv"), "oracle": None,
           "arms": {"append": {"history": "append"}}}
    run(cfg, str(tmp_path / "p"), mock=True, workers=1, skip_oracle=True)
    steps = pd.read_csv(tmp_path / "p" / "steps.csv")
    frames = replay_frames(11, steps.sort_values("turn"), "append", max_frames=50)  # raises if boards diverge
    assert len(frames) == len(steps) + 1
