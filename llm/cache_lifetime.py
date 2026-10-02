"""cache_lifetime.py

How long does a Tetris agent's KV-cached prefix survive an idle gap on serverless inference?

For each (model, gap, trial) we build a unique append-history Tetris prompt (a nonce comes first, so the
first call is a guaranteed miss): system prompt + K played turns (state, reply). We send it once
(warm-up), sleep `gap` seconds, then send the same history plus the next turn. Calls are non-streaming
with max_tokens=1, so latency ≈ queue + prefill + network, and the response carries `cached_tokens`
plus, on deployments that fill it, the vLLM-GPU vs LMCache split. Trials start staggered so warm-ups
don't arrive as a burst.

  python -m llm.cache_lifetime --config configs/explore/iter03.yaml [--mock]
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

import pandas as pd
import yaml

from TetrisGym_updated import TetrisGym
from harness.bots import GreedyBot
from llm.mock_client import MockClient
from llm.prompts import render_state, system_prompt
from llm.run_pilot import TOTAL_BUDGET_USD, record_spend, spent_so_far
from llm.tensormesh_client import TensormeshClient, load_model_settings
from llm.tetris_tools import placement_outcomes


def tetris_history(turns: int, episode_seed: int = 7) -> List[Dict[str, str]]:
    """Messages of a real append-history game (greedy bot) with `turns` past turns + the next state."""
    env = TetrisGym()
    env.reset(seed=episode_seed)
    bot = GreedyBot()
    msgs: List[Dict[str, str]] = []
    lines = 0
    for t in range(turns + 1):
        outs = placement_outcomes(env)
        _, rotations = env.game.current_piece
        legal = [{"action_id": a, "rot": o.rot, "x": o.x, "w": rotations[o.rot].shape[1], "lines": o.lines,
                  "max_height": o.max_height, "holes": o.holes} for a, o in sorted(outs.items())]
        msgs.append({"role": "user", "content": render_state(t, env.game.board, env.game.current_piece[0],
                                                             env.game.next_piece[0], int(env.game.score), lines, legal)})
        if t == turns:
            break
        aid = bot.act(env, t)
        msgs.append({"role": "assistant", "content": json.dumps({"action_id": aid})})
        _, _, done, info = env.step(aid)
        lines += int(info["lines_cleared"])
        if done:
            raise RuntimeError("greedy topped out while building the prefix; use fewer turns")
    return msgs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--out_dir", default="")
    args = ap.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    out_dir = args.out_dir or os.path.join("runs/explore", f"iter{int(cfg['iteration']):02d}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    log = os.path.join(out_dir, "calls.jsonl")
    if os.path.exists(log):
        os.remove(log)
    client = MockClient(log_path=log) if args.mock else TensormeshClient(log_path=log)
    already = 0.0 if args.mock else spent_so_far()
    cap = min(float(cfg.get("budget_usd", 5.0)), TOTAL_BUDGET_USD - already)
    print(f"[lifetime] spent so far ${already:.3f}; cap ${cap:.3f}")

    history = tetris_history(int(cfg["history_turns"]))
    final_user = history[-1]
    past = history[:-1]
    settings = load_model_settings()
    experiments = cfg.get("experiments") or [{"model": m, "gaps_s": cfg["gaps_s"], "trials": cfg["trials"]}
                                             for m in cfg["models"]]
    rewarm = bool(cfg.get("rewarm_connection", False))
    # A "lane" runs its jobs one after another (sequential experiments); other jobs get a lane each.
    lanes: List[List[tuple]] = []
    for e in experiments:
        jobs_e = [(e["model"], float(g), r, e.get("tag", ""), e.get("keepalive_s"))
                  for r in range(int(e.get("trials", 3))) for g in e["gaps_s"]]
        if e.get("sequential"):
            lanes.append(jobs_e)
        else:
            lanes.extend([[j] for j in jobs_e])
    rows: List[Dict[str, Any]] = []
    lock = threading.Lock()
    stagger = float(cfg.get("stagger_s", 1.0))

    def trial(job) -> None:
        model, gap, trial_idx, tag, keepalive_s = job
        if client.total_cost_usd >= cap:
            return
        nonce = uuid.uuid4().hex
        sys_msg = {"role": "system", "content": f"Session {nonce}.\n" + system_prompt()}
        kw = dict(settings.get(model, {}))
        meta = {"model": model, "gap_s": gap, "trial": trial_idx, "tag": tag}
        warm = client.chat(model, [sys_msg] + past, max_tokens=1, stream=False, meta=dict(meta, call="warmup"), **kw)
        t_warm_end = time.time()
        pings, ping_cost = 0, 0.0
        if keepalive_s:
            while time.time() - t_warm_end + float(keepalive_s) < gap:
                time.sleep(float(keepalive_s))
                pr = client.chat(model, [sys_msg] + past, max_tokens=1, stream=False, meta=dict(meta, call="keepalive"), **kw)
                pings += 1
                ping_cost += pr.cost_usd
        time.sleep(max(0.0, gap - (time.time() - t_warm_end)))
        if rewarm and hasattr(client, "list_models"):
            try:
                client.list_models()  # re-open the pooled HTTP connection so the probe measures the server
            except Exception:
                pass
        probe = client.chat(model, [sys_msg] + past + [final_user], max_tokens=1, stream=False,
                            meta=dict(meta, call="after_gap"), **kw)
        row = dict(meta, keepalive_s=keepalive_s or 0, pings=pings, ping_cost_usd=ping_cost,
                   prompt_warm=warm.prompt_tokens, cached_warm=warm.cached_tokens, latency_warm=warm.latency_s,
                   prompt_probe=probe.prompt_tokens, cached_probe=probe.cached_tokens, latency_probe=probe.latency_s,
                   vllm_cached_probe=probe.vllm_cached_tokens, lmcache_cached_probe=probe.lmcache_cached_tokens,
                   actual_gap_s=probe.t_start - t_warm_end, ok=warm.ok and probe.ok,
                   error=(warm.error or probe.error)[:200], cost_usd=warm.cost_usd + probe.cost_usd + ping_cost)
        with lock:
            rows.append(row)
        print(f"[lifetime] {model.split('/')[-1]:>22s} {tag:>12s} gap {gap:6.0f}s trial {trial_idx}: warm {warm.latency_s:5.2f}s "
              f"(cached {warm.cached_tokens}) -> probe {probe.latency_s:5.2f}s cached {probe.cached_tokens}/{probe.prompt_tokens} "
              f"[vllm {probe.vllm_cached_tokens} lmc {probe.lmcache_cached_tokens}] pings {pings}", flush=True)

    def lane(i_lane):
        i, jobs_l = i_lane
        time.sleep(i * stagger)
        for job in jobs_l:
            trial(job)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=len(lanes)) as ex:
        list(ex.map(lane, list(enumerate(lanes))))
    cost = float(client.total_cost_usd)
    if not args.mock:
        record_spend(cfg["iteration"], cfg.get("name", ""), cost, int(sum(2 + r["pings"] for r in rows)))
    pd.DataFrame(rows).sort_values(["model", "tag", "gap_s", "trial"]).to_csv(os.path.join(out_dir, "lifetime.csv"), index=False)
    print(f"[lifetime] {len(rows)} trials, {time.time() - t0:.0f}s, ${cost:.4f}")


if __name__ == "__main__":
    main()
