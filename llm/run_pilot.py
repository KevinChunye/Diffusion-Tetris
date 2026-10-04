"""run_pilot.py

Run one exploration pilot: every arm plays the same episode seeds (paired), concurrently so arms are
interleaved in time; then score decisions offline (regret oracle) and add normalized scores.

  python -m llm.run_pilot --config configs/explore/iter01.yaml            # real API
  python -m llm.run_pilot --config configs/explore/iter01.yaml --mock     # offline dry run

Outputs (out_dir, default runs/explore/iterNN): calls.jsonl, steps.csv (one row per decision with call
metrics + oracle columns), episodes.csv (score, lines, pieces, normalized score, $), config.yaml.
Spend is appended to runs/explore/spend.csv; the run stops early at the iteration/total budget.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

import pandas as pd
import yaml

from llm.llm_policy import LLMPolicy, PolicyCfg, cfg_to_dict, run_episode
from llm.mock_client import MockClient
from llm.oracle import compute_regrets
from llm.tensormesh_client import TensormeshClient, load_model_settings
from llm.tetris_tools import normalized_scores, reference_table

EXPLORE_DIR = "runs/explore"
SPEND_CSV = os.path.join(EXPLORE_DIR, "spend.csv")
TOTAL_BUDGET_USD = 20.0  # whole research program (user-set budget, 2026-10-04)


def spent_so_far() -> float:
    if not os.path.exists(SPEND_CSV):
        return 0.0
    return float(pd.read_csv(SPEND_CSV)["cost_usd"].sum())


def record_spend(iteration: Any, label: str, cost: float, n_calls: int) -> None:
    os.makedirs(EXPLORE_DIR, exist_ok=True)
    row = pd.DataFrame([{"iteration": iteration, "label": label, "cost_usd": round(cost, 6), "n_calls": n_calls,
                         "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}])
    row.to_csv(SPEND_CSV, mode="a", header=not os.path.exists(SPEND_CSV), index=False)


def build_arms(cfg: Dict[str, Any]) -> Dict[str, PolicyCfg]:
    settings = load_model_settings()
    arms = {}
    for name, spec in cfg["arms"].items():
        merged = dict(cfg.get("defaults", {}))
        merged.update(spec or {})
        model = merged.pop("model", cfg["model"])  # an arm may override the model
        chat_kwargs = dict(settings.get(model, {}))
        chat_kwargs.update(merged.pop("chat_kwargs", {}) or {})
        arms[name] = PolicyCfg(model=model, chat_kwargs=chat_kwargs, **merged)
    return arms


def run(cfg: Dict[str, Any], out_dir: str, mock: bool, workers: int, skip_oracle: bool = False) -> Dict[str, Any]:
    if any(os.path.exists(os.path.join(out_dir, f)) for f in ("calls.jsonl", "steps.csv", "manifest.json")):
        raise FileExistsError(f"Choose a fresh output directory; existing results at {out_dir}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    calls_log = os.path.join(out_dir, "calls.jsonl")
    if os.path.exists(calls_log):
        os.remove(calls_log)
    client = MockClient(log_path=calls_log) if mock else TensormeshClient(log_path=calls_log, **cfg.get("client", {}))

    already = 0.0 if mock else spent_so_far()
    cap = min(float(cfg.get("budget_usd", 5.0)), TOTAL_BUDGET_USD - already)
    if cap <= 0:
        raise SystemExit(f"Budget exhausted: ${already:.2f} of ${TOTAL_BUDGET_USD:.2f} already spent")
    print(f"[pilot] spent so far ${already:.3f}; this run capped at ${cap:.3f}")

    arms = build_arms(cfg)
    manifest = {"schema_version": 2, "mock": mock, "latency_definition": "latency_s=final attempt; end_to_end_s=all attempts and backoff",
                "cost_definition": "price-card estimate for final attempt; retry billing may be unobserved",
                "resolved_arms": {name: cfg_to_dict(arm) for name, arm in arms.items()},
                "config_sha256": hashlib.sha256(yaml.safe_dump(cfg, sort_keys=True).encode()).hexdigest(),
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())}
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    seeds = [int(s) for s in cfg["seeds"]]
    max_pieces = int(cfg["max_pieces"])
    jobs = [(arm, seed) for seed in seeds for arm in arms]
    steps: List[Dict[str, Any]] = []
    episodes: List[Dict[str, Any]] = []
    lock = threading.Lock()
    stop = lambda: client.total_cost_usd >= cap

    def one(job):
        arm, seed = job
        rows: List[Dict[str, Any]] = []
        meta = {"arm": arm, "model": arms[arm].model, "episode_seed": seed}
        try:
            ep = run_episode(LLMPolicy(client, arms[arm]), seed, max_pieces, meta, rows, should_stop=stop)
        except Exception as exc:  # keep the other episodes' (paid) data; mark this one as crashed
            ep = dict(meta, score=float("nan"), lines_cleared=0, pieces_placed=len(rows), topped_out=False,
                      stop_reason=f"crash: {type(exc).__name__}: {str(exc)[:200]}")
            print(f"[pilot] {arm} seed {seed} crashed: {exc}", flush=True)
        with lock:
            steps.extend(rows)
            episodes.append(ep)
        print(f"[pilot] {arm:>14s} seed {seed}: pieces {ep['pieces_placed']:3d} lines {ep['lines_cleared']:3d} "
              f"score {ep['score']:5.0f}  run ${client.total_cost_usd:.4f}", flush=True)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=int(cfg.get("concurrency", len(jobs)))) as ex:
        list(ex.map(one, jobs))
    wall = time.time() - t0
    cost = float(client.total_cost_usd)
    n_calls = len(steps)
    if not mock:
        record_spend(cfg.get("iteration", "?"), cfg.get("name", ""), cost, n_calls)
    print(f"[pilot] {n_calls} calls in {wall:.0f}s, cost ${cost:.4f}")

    eps_df = pd.DataFrame(episodes).sort_values(["arm", "episode_seed"]).reset_index(drop=True)
    if not steps:
        eps_df.to_csv(os.path.join(out_dir, "episodes_raw.csv"), index=False)
        raise RuntimeError("No decisions completed; inspect episodes_raw.csv and calls.jsonl")
    steps_df = pd.DataFrame(steps).sort_values(["arm", "episode_seed", "turn"]).reset_index(drop=True)
    # Persist the paid-for data before any post-processing.
    steps_df.to_csv(os.path.join(out_dir, "steps_raw.csv"), index=False)
    eps_df.to_csv(os.path.join(out_dir, "episodes_raw.csv"), index=False)
    cost_ep = steps_df.groupby(["arm", "episode_seed"], as_index=False).agg(
        cost_usd=("cost_usd", "sum"), calls=("turn", "size"), prompt_tokens=("prompt_tokens", "sum"),
        cached_tokens=("cached_tokens", "sum"), completion_tokens=("completion_tokens", "sum"))
    eps_df = eps_df.merge(cost_ep, on=["arm", "episode_seed"], how="left")
    refs = reference_table(seeds, max_pieces, cfg.get("reference_csv", os.path.join(EXPLORE_DIR, "reference_scores.csv")))
    eps_df = normalized_scores(eps_df, refs, "score")
    eps_df = normalized_scores(eps_df, refs, "lines_cleared")

    if not skip_oracle and cfg.get("oracle", {}) is not None:
        okw = dict(cfg.get("oracle") or {})
        t1 = time.time()
        steps_df = compute_regrets(steps_df, workers=workers, **okw)
        steps_df["top1_beam"] = (steps_df["rank_beam"] == 1).astype(float)
        # Executed-action regret includes the fallback policy. Never label it model accuracy.
        accepted = steps_df["model_action_accepted"]
        steps_df["model_regret_beam"] = steps_df["regret_beam"].where(accepted)
        steps_df["model_top1_beam"] = steps_df["top1_beam"].where(accepted)
        steps_df["model_epsilon_best_beam"] = (steps_df["regret_beam"] <= 1e-9).astype(float).where(accepted)
        print(f"[pilot] oracle on {len(steps_df)} states in {time.time() - t1:.0f}s")
        reg = steps_df.groupby(["arm", "episode_seed"], as_index=False).agg(
            regret_beam=("regret_beam", "mean"), regret_rollout=("regret_rollout", "mean"),
            top1_beam=("top1_beam", "mean"))
        eps_df = eps_df.merge(reg, on=["arm", "episode_seed"], how="left")

    steps_df.to_csv(os.path.join(out_dir, "steps.csv"), index=False)
    eps_df.to_csv(os.path.join(out_dir, "episodes.csv"), index=False)
    for raw in ("steps_raw.csv", "episodes_raw.csv"):  # superseded by the full files
        os.remove(os.path.join(out_dir, raw))
    return {"cost": cost, "calls": n_calls, "wall_s": wall, "out_dir": out_dir}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--skip_oracle", action="store_true")
    ap.add_argument("--seeds", default="", help="override seeds, comma separated")
    ap.add_argument("--max_pieces", type=int, default=0)
    args = ap.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.seeds:
        cfg["seeds"] = [int(s) for s in args.seeds.split(",")]
    if args.max_pieces:
        cfg["max_pieces"] = args.max_pieces
    out_dir = args.out_dir or os.path.join(EXPLORE_DIR, f"iter{int(cfg.get('iteration', 0)):02d}")
    print(run(cfg, out_dir, mock=args.mock, workers=args.workers, skip_oracle=args.skip_oracle))


if __name__ == "__main__":
    main()
