"""Offline tests for the LLM harness: no network calls (MockClient)."""

from __future__ import annotations

import os

import pandas as pd

from TetrisGym_updated import TetrisGym
from llm.prompts import parse_action
from llm.run_pilot import run
from llm.tensormesh_client import call_cost_usd
from llm.tetris_tools import fallback_action, placement_outcomes, safe_step


def test_parse_action_formats():
    legal = [3, 7, 12]
    assert parse_action('{"action_id": 7}', legal) == 7
    assert parse_action('```json\n{"action_id": 12}\n```', legal) == 12
    assert parse_action('I pick action_id: 3', legal) == 3
    assert parse_action('42', legal) == 42  # parsed; legality is checked by safe_step
    assert parse_action('no idea', legal) is None
    assert parse_action('', legal) is None


def test_safe_step_replaces_illegal_and_unparseable_with_lowest_height():
    env = TetrisGym()
    env.reset(seed=5)
    expected = fallback_action(env)
    outs = placement_outcomes(env)
    assert outs[expected].max_height == min(o.max_height for o in outs.values())
    _, _, _, used, reason = safe_step(env, 999)
    assert (used, reason) == (expected, "illegal")
    env.reset(seed=5)
    _, _, _, used, reason = safe_step(env, None)
    assert (used, reason) == (expected, "unparseable")
    env.reset(seed=5)
    legal = env.get_valid_action_ids()[-1]
    _, _, _, used, reason = safe_step(env, legal)
    assert (used, reason) == (legal, "")


def test_cost_uses_cached_price_only_when_listed():
    free_cache = {"input": 1.0, "output": 2.0, "cached": 0.0}
    no_cache_price = {"input": 1.0, "output": 2.0, "cached": None}
    assert call_cost_usd(free_cache, 1_000_000, 900_000, 0) == 0.1
    assert call_cost_usd(no_cache_price, 1_000_000, 900_000, 0) == 1.0


def test_dry_run_pilot_end_to_end(tmp_path):
    cfg = {
        "iteration": 0, "name": "smoke", "model": "openai/gpt-oss-20b", "seeds": [7], "max_pieces": 8,
        "budget_usd": 1.0, "concurrency": 4, "reference_csv": str(tmp_path / "refs.csv"),
        "oracle": {"n_samples": 1, "depth": 1, "beam_h": 1, "beam_w": 2, "beam_samples": 1},
        "defaults": {"annotated": True, "max_tokens": 64},
        "arms": {"stateless": {"history": "stateless"}, "append": {"history": "append"},
                 "window2": {"history": "window", "window": 2}, "compact3": {"history": "compact", "compact_every": 3}},
    }
    out = run(cfg, str(tmp_path / "pilot"), mock=True, workers=1)
    assert out["calls"] == 4 * 8
    steps = pd.read_csv(tmp_path / "pilot" / "steps.csv")
    eps = pd.read_csv(tmp_path / "pilot" / "episodes.csv")
    assert set(eps["arm"]) == set(cfg["arms"])
    assert {"regret_beam", "regret_rollout", "norm_score", "cost_usd"} <= set(eps.columns)
    assert (steps["regret_beam"] >= -1e-9).all()
    frac = steps.groupby("arm")["cached_tokens"].sum() / steps.groupby("arm")["prompt_tokens"].sum()
    assert frac["append"] > frac["stateless"] > 0
    hist = steps.groupby("arm")["history_turns"].max()
    assert hist["stateless"] == 0 and hist["window2"] == 2 and hist["compact3"] <= 2 and hist["append"] == 7
    assert os.path.exists(tmp_path / "pilot" / "calls.jsonl")
