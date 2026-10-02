"""The real piece sequence must depend only on the episode seed, never on how much an agent simulates."""

from __future__ import annotations

import random

import pytest

from TetrisGym_updated import TetrisGym
from baselines.beam_search_planner import BeamCfg, BeamSearchPlanner


def _piece_trace(env: TetrisGym, policy, episode_seed: int, n_steps: int) -> list[str]:
    env.reset(seed=episode_seed)
    trace = [env.game.current_piece[0], env.game.next_piece[0]]
    for _ in range(n_steps):
        aid = policy(env)
        _, _, done, _ = env.step(aid)
        if done:
            break
        trace.append(env.game.next_piece[0])
    return trace


def _random_policy(seed: int):
    rng = random.Random(seed)
    return lambda env: rng.choice(env.get_valid_action_ids())


def test_agents_on_same_seed_see_identical_pieces():
    """Beam search clones the env ~1000x per decision; a random policy never does. Same pieces anyway."""
    beam = BeamSearchPlanner(BeamCfg(horizon=2, beam_width=4), sim_seed=123)
    n = 25
    t_beam = _piece_trace(TetrisGym(seed=0), lambda env: beam.plan(env)[0], episode_seed=7, n_steps=n)
    t_rand = _piece_trace(TetrisGym(seed=99), _random_policy(1), episode_seed=7, n_steps=n)
    t_first = _piece_trace(TetrisGym(), lambda env: env.get_valid_action_ids()[0], episode_seed=7, n_steps=n)
    k = min(len(t_beam), len(t_rand), len(t_first))
    assert k >= 10
    assert t_beam[:k] == t_rand[:k] == t_first[:k]


def test_episode_seed_fixes_sequence_and_differs_across_seeds():
    env = TetrisGym()
    policy = lambda e: e.get_valid_action_ids()[0]
    a = _piece_trace(env, policy, episode_seed=3, n_steps=15)
    b = _piece_trace(env, policy, episode_seed=3, n_steps=15)
    c = _piece_trace(env, policy, episode_seed=4, n_steps=15)
    assert a == b
    assert a != c


def test_clone_does_not_advance_real_stream():
    env, ref = TetrisGym(), TetrisGym()
    env.reset(seed=11)
    ref.reset(seed=11)
    sim = env.clone_for_simulation(sim_seed=5)
    for _ in range(10):
        valid = sim.get_valid_action_ids()
        if not valid:
            break
        _, _, done, _ = sim.step(valid[0])
        if done:
            break
    for _ in range(10):
        _, _, d1, _ = env.step(env.get_valid_action_ids()[0])
        _, _, d2, _ = ref.step(ref.get_valid_action_ids()[0])
        assert env.game.current_piece[0] == ref.game.current_piece[0]
        assert env.game.next_piece[0] == ref.game.next_piece[0]
        assert d1 == d2
        if d1:
            break


def _future(env: TetrisGym, n: int) -> list[str]:
    out = []
    for _ in range(n):
        valid = env.get_valid_action_ids()
        if not valid:
            break
        _, _, done, _ = env.step(valid[0])
        if done:
            break
        out.append(env.game.next_piece[0])
    return out


def test_clone_copies_visible_state_but_not_the_future():
    env = TetrisGym()
    env.reset(seed=21)
    for _ in range(3):
        env.step(env.get_valid_action_ids()[0])
    sims = [env.clone_for_simulation(sim_seed=s) for s in range(5)]
    for sim in sims:
        assert (sim.game.board == env.game.board).all()
        assert sim.game.current_piece[0] == env.game.current_piece[0]
        assert sim.game.next_piece[0] == env.game.next_piece[0]
        assert sim.game.score == env.game.score
    # Same sim seed -> same simulated future; different seeds -> (almost surely) different futures.
    f0a = _future(env.clone_for_simulation(sim_seed=1), 12)
    f0b = _future(env.clone_for_simulation(sim_seed=1), 12)
    assert f0a == f0b
    futures = {tuple(_future(env.clone_for_simulation(sim_seed=s), 12)) for s in range(5)}
    assert len(futures) > 1
    # The clone's future is not the real future (real stream continues from the env's own RNG).
    probe = TetrisGym()
    probe.reset(seed=21)
    for _ in range(3):
        probe.step(probe.get_valid_action_ids()[0])
    truth = _future(probe, 12)
    assert any(list(f) != truth for f in futures)


def test_beam_search_near_step_limit_does_not_step_illegal_moves():
    """Regression: clones inherited max_steps, so lookahead past the episode budget used stale actions."""
    env = TetrisGym(max_steps=12)
    env.reset(seed=1000)
    beam = BeamSearchPlanner(BeamCfg(horizon=3, beam_width=16), sim_seed=0)
    done = False
    while not done:
        _, _, done, _ = env.step(beam.plan(env)[0])
    assert env.step_count == 12
    assert env.clone_for_simulation(0).max_steps is None


def test_clone_has_independent_rng_object():
    env = TetrisGym(seed=1)
    sim = env.clone_for_simulation(sim_seed=2)
    assert sim.game.rng is not env.game.rng
    assert sim.render_mode == "skip"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
