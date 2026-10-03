"""Episode-level benchmark summaries; mocks are visibly labeled, failures retained."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def mean_ci(values, seed=42):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {'n': 0, 'mean': None, 'ci95': None}
    interval = None
    if len(values) >= 2:
        rng = np.random.default_rng(seed)
        boot = rng.choice(values, (2000, len(values)), replace=True).mean(axis=1)
        interval = np.quantile(boot, [.025, .975]).tolist()
    return {'n': len(values), 'mean': float(values.mean()), 'ci95': interval}


def report(directory, baseline=None):
    root = Path(directory)
    manifest = json.loads((root / 'manifest.json').read_text())
    eps = pd.read_csv(root / 'episodes.csv')
    steps = pd.read_csv(root / 'steps.csv')
    result = {'mock': manifest['mock'], 'inference': 'episode bootstrap; pilot intervals are exploratory', 'arms': {}, 'paired_differences': {}}
    if eps.duplicated(['arm', 'episode_seed']).any():
        raise ValueError('Repeated seeds need a run/block identifier before aggregation')
    for arm, episodes in eps.groupby('arm'):
        calls = steps[steps.arm == arm]
        accepted = calls.model_action_accepted.fillna(False).astype(bool)
        observed = calls.cache_usage_reported.fillna(False).astype(bool)
        cache_den = calls.loc[observed, 'prompt_tokens'].sum()
        complete = episodes.stop_reason.fillna('').eq('')
        result['arms'][arm] = {
            'episodes': len(episodes), 'incomplete_episodes': int((~complete).sum()),
            'lines_completed_episodes': mean_ci(episodes.loc[complete, 'lines_cleared']),
            'score_completed_episodes': mean_ci(episodes.loc[complete, 'score']),
            'pieces_completed_episodes': mean_ci(episodes.loc[complete, 'pieces_placed']),
            'decisions': len(calls), 'fallback_rate': float((~accepted).mean()),
            'api_error_rate': float((~calls.ok.astype(bool)).mean()),
            'cache_telemetry_coverage': float(observed.mean()),
            'reported_cached_fraction': float(calls.loc[observed, 'cached_tokens'].sum() / cache_den) if cache_den else None,
            'estimated_cost_usd': float(calls.cost_usd.sum()),
            'unknown_cost_calls': int((~calls.cost_estimate_known.astype(bool)).sum()),
            'retried_calls': int((calls.retries > 0).sum()),
            'end_to_end_p50_s': float(calls.end_to_end_s.median()),
            'end_to_end_p95_s': float(calls.end_to_end_s.quantile(.95)),
            'first_output_p50_s': float(calls.ttft_token_s.median()) if calls.ttft_token_s.notna().any() else None,
        }
        if 'model_regret_beam' in calls:
            # Average within episode before resampling, not over correlated turns.
            result['arms'][arm]['accepted_action_regret'] = mean_ci(calls.groupby('episode_seed').model_regret_beam.mean())
    if baseline:
        if baseline not in set(eps.arm):
            raise ValueError(f'Unknown baseline: {baseline}')
        base = eps[(eps.arm == baseline) & eps.stop_reason.fillna('').eq('')].set_index('episode_seed')
        for arm, episodes in eps.groupby('arm'):
            if arm == baseline:
                continue
            other = episodes[episodes.stop_reason.fillna('').eq('')].set_index('episode_seed')
            paired = other[['lines_cleared']].join(base[['lines_cleared']], lsuffix='_arm', rsuffix='_base', how='inner')
            result['paired_differences'][arm] = mean_ci(paired.lines_cleared_arm - paired.lines_cleared_base)
    (root / 'benchmark_report.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dir', required=True)
    p.add_argument('--baseline')
    args = p.parse_args()
    print(json.dumps(report(args.dir, args.baseline), indent=2, allow_nan=False))
