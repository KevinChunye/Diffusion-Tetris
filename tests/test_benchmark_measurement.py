import json

import pandas as pd
import pytest

from llm.tensormesh_client import CallResult, TensormeshClient


def test_missing_cache_usage_is_not_a_measured_miss():
    client = TensormeshClient(api_key='test-no-network')
    absent = CallResult(model='test')
    client._read_usage({'prompt_tokens': 10, 'completion_tokens': 2}, absent)
    assert absent.usage_reported and not absent.cache_usage_reported
    zero = CallResult(model='test')
    client._read_usage({'prompt_tokens': 10, 'completion_tokens': 2,
                        'prompt_tokens_details': {'cached_tokens': 0}}, zero)
    assert zero.cache_usage_reported and zero.cached_tokens == 0


def test_retries_include_backoff_and_clear_partial_usage(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr('llm.tensormesh_client.time.perf_counter', lambda: clock[0])
    monkeypatch.setattr('llm.tensormesh_client.time.sleep', lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    monkeypatch.setattr('llm.tensormesh_client.random.uniform', lambda *args: 1.0)
    client = TensormeshClient(api_key='test-no-network', max_retries=1, verbose_retries=False)
    attempts = []
    def request(body, res, t0):
        clock[0] += 2
        attempts.append(1)
        if len(attempts) == 1:
            res.cached_tokens = 99
            res.ttft_token_s = .1
            res.usage_reported = res.cache_usage_reported = True
            return 503, 'retry'
        assert res.cached_tokens == 0 and res.ttft_token_s is None
        assert not res.usage_reported and not res.cache_usage_reported
        return 200, ''
    monkeypatch.setattr(client, '_do_stream', request)
    res = client.chat('openai/gpt-oss-20b', [{'role': 'user', 'content': 'test'}])
    assert res.retries == 1 and res.latency_s == 2 and res.end_to_end_s == 5
    assert not res.cost_estimate_known


def test_cache_replay_pairs_exact_warm_request_and_preserves_data(tmp_path):
    from TetrisGym_updated import TetrisGym
    from llm.llm_policy import board_to_str
    from llm.cache_replay import run
    env = TetrisGym()
    env.reset(seed=4)
    source = tmp_path / 'steps.csv'
    pd.DataFrame([{'board': board_to_str(env.game.board), 'curr': env.game.current_piece[0],
                   'next': env.game.next_piece[0], 'arm': 'reference', 'episode_seed': 4,
                   'turn': 0, 'used_id': env.get_valid_action_ids()[0]}]).to_csv(source, index=False)
    out = tmp_path / 'probe'
    result = run(source, out, 'openai/gpt-oss-20b', pairs=1, mock=True)
    assert result['calls'] == 3 and result['mock']
    rows = pd.read_csv(out / 'measurements.csv').set_index('phase')
    assert rows.loc['prime', 'request_sha256'] == rows.loc['warm_repeat', 'request_sha256']
    assert rows.loc['fresh_prefix', 'request_sha256'] != rows.loc['prime', 'request_sha256']
    assert rows.prompt_sha256.nunique() == 1
    assert rows.loc['warm_repeat', 'cached_tokens'] > rows.loc['fresh_prefix', 'cached_tokens']
    assert json.loads((out / 'manifest.json').read_text())['mock']
    with pytest.raises(FileExistsError):
        run(source, out, 'openai/gpt-oss-20b', pairs=1, mock=True)


def test_model_accuracy_does_not_credit_fallback(tmp_path):
    from llm.llm_policy import LLMPolicy, PolicyCfg, run_episode
    from llm.mock_client import MockClient
    rows = []
    run_episode(LLMPolicy(MockClient(illegal_every=1), PolicyCfg()), 1, 2, {}, rows)
    assert len(rows) == 2
    assert all(not r['model_action_accepted'] and r['fallback_reason'] == 'illegal' for r in rows)


def test_report_pairs_seeds_and_exposes_missing_telemetry(tmp_path):
    from llm.benchmark_report import report
    (tmp_path / 'manifest.json').write_text(json.dumps({'mock': True}))
    episodes = []
    steps = []
    for arm, offset in [('base', 0), ('candidate', 1)]:
        for seed in [1, 2]:
            episodes.append(dict(arm=arm, episode_seed=seed, stop_reason='',
                                 lines_cleared=seed + offset, score=2 * (seed + offset), pieces_placed=10))
            steps.append(dict(arm=arm, episode_seed=seed, model_action_accepted=False,
                              cache_usage_reported=False, prompt_tokens=10, cached_tokens=0,
                              ok=True, cost_usd=0, cost_estimate_known=False, retries=0,
                              end_to_end_s=1, ttft_token_s=.5, model_regret_beam=float('nan')))
    pd.DataFrame(episodes).to_csv(tmp_path / 'episodes.csv', index=False)
    pd.DataFrame(steps).to_csv(tmp_path / 'steps.csv', index=False)
    result = report(tmp_path, 'base')
    assert result['mock']
    assert result['paired_differences']['candidate']['mean'] == 1
    assert result['paired_differences']['candidate']['n'] == 2
    arm = result['arms']['candidate']
    assert arm['reported_cached_fraction'] is None
    assert arm['fallback_rate'] == 1 and arm['accepted_action_regret']['n'] == 0
    assert arm['unknown_cost_calls'] == 2
