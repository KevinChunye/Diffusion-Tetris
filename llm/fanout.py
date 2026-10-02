"""fanout.py

Fan-out over a shared prefix: an agent with a long context (a ~15k-token append-history Tetris prompt)
asks K parallel questions that differ only in a short suffix: "rate candidate placement i", the
pattern of an LLM reranker over K beam-search candidates.

Strategies (each trial uses a fresh nonce, so the shared prefix starts cold):
  cold_fanout   K requests sent at the same time
  primed_fanout 1 request first (waited for), then K-1 at the same time
  sequential    K requests one after another
Measured per trial: prefill tokens actually computed (sum of prompt - cached), makespan, $ billed,
per-request latency. Calls are non-streaming with max_tokens=1, so latency ≈ queue + prefill.

  python -m llm.fanout --config configs/explore/iter05.yaml [--mock]
"""

from __future__ import annotations

import argparse
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

import pandas as pd
import yaml

from llm.cache_lifetime import tetris_history
from llm.mock_client import MockClient
from llm.prompts import system_prompt
from llm.run_pilot import TOTAL_BUDGET_USD, record_spend, spent_so_far
from llm.tensormesh_client import TensormeshClient, load_model_settings


def candidate_question(i: int) -> Dict[str, str]:
    return {"role": "user", "content": f"Candidate {i}: rate placement id {i} from 0 (terrible) to 9 (best). "
                                       f"Reply with one digit."}


def run_trial(client, model: str, k: int, strategy: str, past: List[Dict[str, str]], kw: Dict[str, Any],
              meta: Dict[str, Any]) -> Dict[str, Any]:
    sys_msg = {"role": "system", "content": f"Session {uuid.uuid4().hex}.\n" + system_prompt()}

    def call(i: int):
        return client.chat(model, [sys_msg] + past + [candidate_question(i)], max_tokens=1, stream=False,
                           meta=dict(meta, i=i), **kw)

    t0 = time.time()
    if strategy == "sequential":
        res = [call(i) for i in range(k)]
    elif strategy == "cold_fanout":
        with ThreadPoolExecutor(max_workers=k) as ex:
            res = list(ex.map(call, range(k)))
    elif strategy == "primed_fanout":
        first = call(0)
        with ThreadPoolExecutor(max_workers=max(1, k - 1)) as ex:
            res = [first] + list(ex.map(call, range(1, k)))
    else:
        raise ValueError(strategy)
    makespan = time.time() - t0
    lat = pd.Series([r.latency_s for r in res])
    return dict(meta, k=k, strategy=strategy, ok=all(r.ok for r in res), makespan_s=makespan,
                prompt_tokens=sum(r.prompt_tokens for r in res), cached_tokens=sum(r.cached_tokens for r in res),
                computed_tokens=sum(r.prompt_tokens - r.cached_tokens for r in res),
                prompt_per_request=res[0].prompt_tokens, latency_p50=float(lat.median()), latency_max=float(lat.max()),
                retries=sum(r.retries for r in res), cost_usd=sum(r.cost_usd for r in res),
                errors=";".join(r.error[:60] for r in res if not r.ok))


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
    print(f"[fanout] spent so far ${already:.3f}; cap ${cap:.3f}")
    history = tetris_history(int(cfg["history_turns"]))
    past = history[:-1]
    settings = load_model_settings()
    rows: List[Dict[str, Any]] = []
    lock = threading.Lock()

    def model_lane(model: str) -> None:
        # One trial at a time per model, so trials don't share load; models run side by side.
        kw = dict(settings.get(model, {}))
        for trial in range(int(cfg["trials"])):
            for k in cfg["k_values"]:
                for strategy in cfg["strategies"]:
                    if client.total_cost_usd >= cap:
                        return
                    row = run_trial(client, model, int(k), strategy, past, kw, {"model": model, "trial": trial})
                    with lock:
                        rows.append(row)
                    print(f"[fanout] {model.split('/')[-1]:>20s} K={k:2d} {strategy:>13s} trial {trial}: makespan "
                          f"{row['makespan_s']:6.2f}s computed {row['computed_tokens']:7d}/{row['prompt_tokens']:7d} tok "
                          f"p50 {row['latency_p50']:5.2f}s ${row['cost_usd']:.4f} retries {row['retries']}", flush=True)
                    time.sleep(float(cfg.get("pause_s", 2.0)))

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=len(cfg["models"])) as ex:
        list(ex.map(model_lane, cfg["models"]))
    cost = float(client.total_cost_usd)
    if not args.mock:
        record_spend(cfg["iteration"], cfg.get("name", ""), cost, int(sum(r["k"] for r in rows)))
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "fanout.csv"), index=False)
    print(f"[fanout] {len(rows)} trials in {time.time() - t0:.0f}s, ${cost:.4f}")


if __name__ == "__main__":
    main()
