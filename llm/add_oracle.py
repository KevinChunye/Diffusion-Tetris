"""add_oracle.py - oracle regret for a game run made with `run_pilot --skip_oracle`.

Applies exactly the post-processing `llm.run_pilot` would have applied (same oracle settings, read from the
run's config.yaml), so the API phase and the CPU-heavy oracle can run at different times.

  python -m llm.add_oracle runs/explore/scaleup/ladder --workers 4
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import yaml

from llm.oracle import compute_regrets


def add_oracle(run_dir: str, workers: int) -> None:
    d = Path(run_dir)
    steps = pd.read_csv(d / "steps.csv")
    eps = pd.read_csv(d / "episodes.csv")
    if "regret_beam" in steps.columns:
        print(f"[oracle] {d}: already has regret columns; nothing to do")
        return
    cfg = yaml.safe_load((d / "config.yaml").read_text())
    okw = dict(cfg.get("oracle") or {})
    t1 = time.time()
    steps = compute_regrets(steps, workers=workers, **okw)
    steps["top1_beam"] = (steps["rank_beam"] == 1).astype(float)
    if "model_action_accepted" in steps.columns:
        accepted = steps["model_action_accepted"].astype(bool)
        steps["model_regret_beam"] = steps["regret_beam"].where(accepted)
        steps["model_top1_beam"] = steps["top1_beam"].where(accepted)
        steps["model_epsilon_best_beam"] = (steps["regret_beam"] <= 1e-9).astype(float).where(accepted)
    print(f"[oracle] {d}: {len(steps)} states in {time.time() - t1:.0f}s", flush=True)
    reg = steps.groupby(["arm", "episode_seed"], as_index=False).agg(
        regret_beam=("regret_beam", "mean"), regret_rollout=("regret_rollout", "mean"), top1_beam=("top1_beam", "mean"))
    eps = eps.merge(reg, on=["arm", "episode_seed"], how="left")
    steps.to_csv(d / "steps.csv", index=False)
    eps.to_csv(d / "episodes.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    for d in args.run_dirs:
        add_oracle(d, args.workers)


if __name__ == "__main__":
    main()
