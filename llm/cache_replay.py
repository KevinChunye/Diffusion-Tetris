"""Paired cache probes on frozen Tetris decisions, with separate priming accounting.

This manipulates prefix reuse, not a provider-side cache-disable switch. A fresh
nonce is a cold candidate, never proof that all server caches were flushed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import uuid
from pathlib import Path

import pandas as pd

from llm.mock_client import MockClient
from llm.oracle import env_from_state
from llm.prompts import parse_action, render_state, system_prompt
from llm.tensormesh_client import TensormeshClient, load_model_settings
from llm.tetris_tools import placement_outcomes


def state_message(row, annotated=False):
    env = env_from_state(row['board'], row['curr'], row['next'])
    outcomes = placement_outcomes(env)
    rotations = env.game.current_piece[1]
    legal = [dict(action_id=a, rot=o.rot, x=o.x, w=rotations[o.rot].shape[1],
                  lines=o.lines, max_height=o.max_height, holes=o.holes)
             for a, o in sorted(outcomes.items())]
    # Score is deliberately excluded as a trajectory-specific conditioning signal.
    text = render_state(int(row['turn']), env.game.board, row['curr'], row['next'], 0, 0, legal, annotated)
    return text, list(outcomes)


def frozen_messages(frame, index, history_turns=0, annotated=False):
    row = frame.loc[index]
    prior = frame[(frame.arm == row.arm) & (frame.episode_seed == row.episode_seed) & (frame.turn < row.turn)]
    prior = prior.sort_values('turn').tail(history_turns) if history_turns else prior.iloc[:0]
    messages = [{'role': 'system', 'content': system_prompt(annotated)}]
    for _, old in prior.iterrows():
        text, _ = state_message(old, annotated)
        messages.extend([{'role': 'user', 'content': text},
                         {'role': 'assistant', 'content': json.dumps({'action_id': int(old.used_id)})}])
    current, legal = state_message(row, annotated)
    messages.append({'role': 'user', 'content': current})
    return messages, legal


def with_namespace(messages, namespace):
    result = [dict(m) for m in messages]
    result[0]['content'] = f'Experiment identifier (ignore): {namespace}\n' + result[0]['content']
    return result


def run(steps_path, out_dir, model, pairs=6, history_turns=8, mock=False,
        budget_usd=1.0, max_tokens=128, random_seed=42):
    if pairs < 1 or history_turns < 0 or max_tokens < 1 or budget_usd <= 0:
        raise ValueError('pairs/max_tokens/budget must be positive and history_turns nonnegative')
    frame = pd.read_csv(steps_path)
    required = {'board', 'curr', 'next', 'arm', 'episode_seed', 'turn', 'used_id'}
    if not required <= set(frame) or frame.empty:
        raise ValueError(f'Nonempty frozen steps file must contain {sorted(required)}')
    if frame.duplicated(['arm', 'episode_seed', 'turn']).any():
        raise ValueError('Duplicate trajectory/turn keys in frozen corpus')
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Use a fresh output directory to preserve prior results')
    # No retries: an unavailable call is a measured failure, not a hidden slow success.
    client = MockClient() if mock else TensormeshClient(max_retries=0, timeout_s=60)
    if model not in client.pricing:
        raise ValueError('Model needs an explicit price card before spending')
    catalog = [] if mock else client.list_models()
    if not mock and model not in {m['id'] for m in catalog}:
        raise ValueError(f'Model unavailable in current catalog: {model}')
    price = client.pricing[model]
    if not all(isinstance(price.get(k), (int, float)) and price[k] >= 0 for k in ('input', 'output')):
        raise ValueError('Input/output prices must be nonnegative numbers')
    # Conservative request reservation from the advertised full context, not a
    # character/token heuristic. This is a price-card estimate, not a billing cap.
    context_limit = 32768 if mock else next(m.get('max_model_len') for m in catalog if m['id'] == model)
    if not isinstance(context_limit, int) or context_limit < 1:
        raise ValueError('Catalog must expose max_model_len for budget reservation')
    reserve = (context_limit * max(price['input'], price.get('cached') or 0) + max_tokens * price['output']) / 1e6
    out.mkdir(parents=True, exist_ok=True)
    client.log_path = str(out / 'calls.jsonl')
    rng = random.Random(random_seed)
    indices = list(frame.index)
    rng.shuffle(indices)
    selected = indices[:pairs]
    settings = load_model_settings().get(model, {})
    manifest = dict(schema_version=1, mock=mock, model=model, catalog=catalog,
                    corpus_sha256=hashlib.sha256(Path(steps_path).read_bytes()).hexdigest(),
                    git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    git_dirty=bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip()),
                    requested_pairs=pairs, selected_indices=selected, history_turns=history_turns,
                    random_seed=random_seed, budget_usd=budget_usd, max_tokens=max_tokens,
                    price_card=price, model_settings=settings, cost_definition='historical price-card estimate, not invoice',
                    condition_definition='fresh namespace vs exact-request repeat; server cache state uncontrolled')
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    rows = []
    stop_reason = 'completed'
    for pair_id, idx in enumerate(selected):
        if client.total_cost_usd + 3 * reserve > budget_usd:
            stop_reason = 'budget_reservation'
            break
        messages, legal = frozen_messages(frame, idx, history_turns)
        digest = hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()
        warm = with_namespace(messages, uuid.uuid4().hex)
        cold = with_namespace(messages, uuid.uuid4().hex)
        order = ['warm_repeat', 'fresh_prefix']
        rng.shuffle(order)
        # Prime before the two randomized measurement calls. Include its tokens,
        # latency, failures and cost in raw data and total experiment spending.
        for phase, request in [('prime', warm)] + [(phase, warm if phase == 'warm_repeat' else cold) for phase in order]:
            with (out / 'requests.jsonl').open('a') as f:
                f.write(json.dumps(dict(pair_id=pair_id, phase=phase, messages=request)) + '\n')
            result = client.chat(model, request, max_tokens=max_tokens, temperature=0, reasoning_effort=settings.get('reasoning_effort'),
                                 extra_body=settings.get('extra_body'), response_format={'type': 'json_object'},
                                 seed=random_seed,
                                 meta=dict(pair_id=pair_id, phase=phase, corpus_index=int(idx), prompt_sha256=digest))
            action = parse_action(result.text, legal) if result.ok else None
            row = result.to_row()
            row.update(pair_id=pair_id, phase=phase, corpus_index=int(idx),
                       episode_seed=int(frame.loc[idx, 'episode_seed']), source_arm=frame.loc[idx, 'arm'],
                       proposed_id=action, legal_action=action in legal, prompt_sha256=digest,
                       request_sha256=hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest())
            rows.append(row)
            # Persist each result; do not lose paid observations on interruption.
            with (out / 'measurements.jsonl').open('a') as f:
                f.write(json.dumps(row) + '\n')
            if not result.ok or not result.cost_estimate_known:
                stop_reason = 'failed_call_or_unknown_usage'
                break
        if stop_reason != 'completed':
            break
    pd.DataFrame(rows).to_csv(out / 'measurements.csv', index=False)
    paired = []
    for pair_id in sorted({r['pair_id'] for r in rows}):
        group = {r['phase']: r for r in rows if r['pair_id'] == pair_id}
        if not all(k in group and group[k]['ok'] for k in ('prime', 'warm_repeat', 'fresh_prefix')):
            continue
        warm, cold = group['warm_repeat'], group['fresh_prefix']
        delta = None
        if cold['ttft_token_s'] is not None and warm['ttft_token_s'] is not None:
            delta = cold['ttft_token_s'] - warm['ttft_token_s']
        paired.append(dict(pair_id=pair_id, episode_seed=warm['episode_seed'],
                           first_output_s_saved=delta,
                           total_s_saved=cold['end_to_end_s'] - warm['end_to_end_s'],
                           warm_cached_tokens=warm['cached_tokens'] if warm['cache_usage_reported'] else None,
                           fresh_cached_tokens=cold['cached_tokens'] if cold['cache_usage_reported'] else None,
                           same_action=warm['proposed_id'] == cold['proposed_id'],
                           both_legal=warm['legal_action'] and cold['legal_action']))
    pd.DataFrame(paired).to_csv(out / 'paired.csv', index=False)
    summary = dict(mock=mock, calls=len(rows), measured_pairs=len(paired),
                   estimated_cost_usd=client.total_cost_usd, stop_reason=stop_reason)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--model', default='openai/gpt-oss-20b')
    parser.add_argument('--pairs', type=int, default=6)
    parser.add_argument('--history_turns', type=int, default=8)
    parser.add_argument('--max_tokens', type=int, default=128)
    parser.add_argument('--budget_usd', type=float, default=1.0)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--mock', action='store_true')
    args = vars(parser.parse_args())
    args['steps_path'] = args.pop('steps')
    print(json.dumps(run(**args), indent=2))


if __name__ == '__main__':
    main()
