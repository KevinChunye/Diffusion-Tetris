"""capacity_sweep.py

How many concurrent agent contexts does a serverless deployment keep in its prefix cache?

For each model, a round with load N starts N agents at once. Each agent has its own Tetris history
(a unique nonce first, so nothing is shared and the first call is a guaranteed miss). Each agent:
  1. warm-up: sends its ~6.6k-token history (non-streaming, max_tokens=1): a cold prefill under load N
  2. idles `gap_s` seconds from the end of its own warm-up
  3. probe: sends the same history plus one more turn; reported cached_tokens says whether its
     prefix survived. Latency is kept too, because some deployments do not report hits.
Rounds run N in a shuffled order per repetition, with a cooldown between rounds. Models run in
parallel lanes, but a global semaphore caps in-flight requests across all lanes. No retries: errors
(e.g. HTTP 429) are recorded as outcomes. Spend is reserved per round from the price card
(fail-closed), every row is appended to rows.jsonl as soon as it completes, and the spend ledger is
updated even if the run is interrupted.

  python -m llm.capacity_sweep --config configs/scaleup/capacity.yaml --out runs/explore/scaleup/capacity
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

import yaml

from llm.cache_lifetime import tetris_history
from llm.prompts import system_prompt
from llm.run_pilot import TOTAL_BUDGET_USD, record_spend, spent_so_far
from llm.tensormesh_client import load_model_settings


def round_reservation(price: Dict, prefix_tokens: int, n: int) -> float:
    """Upper bound for one round: 2 calls per agent, every prompt token billed at full input price."""
    if not price or price.get("input") is None:
        raise ValueError("missing price card: refusing to spend (fail-closed)")
    return 2 * n * (prefix_tokens + 700) * float(price["input"]) / 1e6 + 2 * n * float(price.get("output") or 0) / 1e6


def plan(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Per model: reps x shuffled N order (seeded, recorded). With `resume_from`, rounds already completed
    in that run directory are skipped; `extra` appends rounds (e.g. repeats of rounds that were confounded)."""
    rng = random.Random(int(cfg.get("seed", 0)))
    done = set()
    if cfg.get("resume_from"):
        for line in open(Path(cfg["resume_from"]) / "rows.jsonl", encoding="utf-8"):
            r = json.loads(line)
            done.add((r["model"], r["rep"], r["load"]))
    lanes = []
    for m in cfg["models"]:
        rounds = []
        for rep in range(int(m["reps"])):
            ns = [n for n in cfg["loads"] if n <= int(m.get("max_load", max(cfg["loads"])))]
            rng.shuffle(ns)
            rounds += [{"model": m["id"], "rep": rep, "load": n} for n in ns if (m["id"], rep, n) not in done]
        rounds += [{"model": m["id"], "rep": int(e["rep"]), "load": int(e["load"])} for e in cfg.get("extra", [])
                   if e["model"] == m["id"]]
        if rounds:
            lanes.append({"model": m["id"], "rounds": rounds})
    return lanes


def run(cfg: Dict[str, Any], out: str, client=None, sleep=time.sleep) -> Path:
    d = Path(out)
    if d.exists() and any(d.iterdir()):
        raise FileExistsError(f"fresh directory required: {d}")
    d.mkdir(parents=True)
    mock = client is not None
    if client is None:
        from llm.tensormesh_client import TensormeshClient

        client = TensormeshClient(log_path=str(d / "calls.jsonl"), verbose_retries=False, max_retries=0,
                                  timeout_s=float(cfg.get("timeout_s", 120)))
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    lanes = plan(cfg)
    (d / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (d / "manifest.json").write_text(json.dumps({
        "live": not mock, "mock": mock, "git_commit": commit, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config_sha256": hashlib.sha256(yaml.safe_dump(cfg, sort_keys=True).encode()).hexdigest(),
        "plan": lanes}, indent=1))
    history = tetris_history(int(cfg["history_turns"]))
    past, final_user = history[:-1], history[-1]
    settings = load_model_settings()
    cap = float(cfg["budget_usd"]) if mock else min(float(cfg["budget_usd"]), TOTAL_BUDGET_USD - spent_so_far())
    gate = threading.Semaphore(int(cfg.get("max_inflight", 64)))
    # Rounds at or above `exclusive_load` never overlap each other, so one model's high-load round cannot
    # trip a shared gateway or account limit while another model is being measured.
    big_lock = threading.Lock()
    exclusive = int(cfg.get("exclusive_load", 10**9))
    lock = threading.Lock()
    state = {"spent": 0.0, "reserved": 0.0, "stopped": "", "calls": 0}
    est_prefix = int(len(json.dumps(past).encode()) / 2.4) + 800

    def call(model, messages, meta):
        with gate:
            return client.chat(model, messages, max_tokens=1, stream=False, meta=meta,
                               **dict(settings.get(model, {})))

    def agent(model, rnd, idx, results):
        nonce = uuid.uuid4().hex
        msgs = [{"role": "system", "content": f"Session {nonce}.\n" + system_prompt()}] + past
        meta = {"model": model, "rep": rnd["rep"], "load": rnd["load"], "agent": idx}
        t_start = time.time()
        w = call(model, msgs, dict(meta, call="warmup"))
        t_warm = time.time()
        sleep(max(0.0, float(cfg["gap_s"]) - (time.time() - t_warm)))
        p = call(model, msgs + [final_user], dict(meta, call="probe"))
        row = dict(meta, nonce=nonce, t_start=t_start, gap_actual_s=p.t_start - t_warm,
                   warm_ok=w.ok, warm_status=w.http_status, warm_error=w.error[:160], warm_latency_s=w.latency_s,
                   warm_prompt_tokens=w.prompt_tokens, warm_cached_tokens=w.cached_tokens,
                   probe_ok=p.ok, probe_status=p.http_status, probe_error=p.error[:160], probe_latency_s=p.latency_s,
                   probe_prompt_tokens=p.prompt_tokens, probe_cached_tokens=p.cached_tokens,
                   cache_usage_reported=p.cache_usage_reported, vllm_cached=p.vllm_cached_tokens,
                   lmcache_cached=p.lmcache_cached_tokens,
                   cost_usd=(w.cost_usd if w.cost_estimate_known else None, p.cost_usd if p.cost_estimate_known else None),
                   live=not mock)
        results.append(row)

    def lane(spec, i):
        sleep(min(i, int(cfg.get("parallel_lanes", 1))) * float(cfg.get("lane_stagger_s", 3)))
        price = client.pricing.get(spec["model"])
        for rnd in spec["rounds"]:
            reserve = round_reservation(price, est_prefix, rnd["load"])
            with lock:
                if state["stopped"] or state["spent"] + state["reserved"] + reserve > cap:
                    state["stopped"] = state["stopped"] or "budget"
                    return
                state["reserved"] += reserve
            results: List[Dict[str, Any]] = []
            big = big_lock if rnd["load"] >= exclusive else None
            if big:
                big.acquire()
            try:
                with ThreadPoolExecutor(max_workers=rnd["load"]) as ex:
                    list(ex.map(lambda k: agent(spec["model"], rnd, k, results), range(rnd["load"])))
            finally:
                if big:
                    big.release()
            spent = 0.0
            for r in results:
                w_cost, p_cost = r["cost_usd"]
                spent += (w_cost if w_cost is not None else reserve / (2 * rnd["load"]))
                spent += (p_cost if p_cost is not None else reserve / (2 * rnd["load"]))
                r["cost_usd"] = (w_cost or 0.0) + (p_cost or 0.0)
            with lock:
                state["reserved"] -= reserve
                state["spent"] += spent
                state["calls"] += 2 * len(results)
                with open(d / "rows.jsonl", "a", encoding="utf-8") as f:
                    for r in results:
                        f.write(json.dumps(r, default=str) + "\n")
                hits = sum(1 for r in results if r["probe_prompt_tokens"] and r["probe_cached_tokens"] >= 0.5 * r["warm_prompt_tokens"])
                errs = sum(1 for r in results if not (r["warm_ok"] and r["probe_ok"]))
                print(f"[capacity] {spec['model'].split('/')[-1]:>20s} rep {rnd['rep']} N={rnd['load']:>2d}: "
                      f"reported hits {hits}/{len(results)} errors {errs} spent ${state['spent']:.3f}", flush=True)
            sleep(float(cfg.get("cooldown_s", 10)))

    try:
        with ThreadPoolExecutor(max_workers=int(cfg.get("parallel_lanes", len(lanes)))) as ex:
            list(ex.map(lambda a: lane(*a), [(s, i) for i, s in enumerate(lanes)]))
    finally:
        if not mock:
            record_spend(cfg["iteration"], f"{cfg['name']} ({d.name})", state["spent"] + state["reserved"], state["calls"])
        (d / "spend.json").write_text(json.dumps({"estimated_cost_usd": state["spent"], "unsettled_reservation_usd": state["reserved"],
                                                  "calls": state["calls"], "cap_usd": cap,
                                                  "stopped": state["stopped"] or "completed"}, indent=1))
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    print(run(cfg, args.out))


if __name__ == "__main__":
    main()
