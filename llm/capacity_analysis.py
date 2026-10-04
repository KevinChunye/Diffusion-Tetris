"""capacity_analysis.py - summarize a capacity sweep (llm/capacity_sweep.py).

Per model and load N:
- reported hit rate: probe cached_tokens >= 50% of the warm-up prompt, with a Wilson 95% interval;
- inferred hit rate: probe latency < 0.35 x the model's median cold warm-up latency at N = 1. This is
  for deployments that do not report hits; its agreement with reported hits is measured on the
  models that do report;
- cold-prefill latency (warm-up, median), and aggregate prefill throughput = N x prompt tokens / the
  slowest warm-up in the round;
- errors (any non-200 warm-up or probe).
Capacity is the largest N such that the hit rate is >= 0.9 at that N and at every smaller N tested.

  python -m llm.capacity_analysis runs/explore/scaleup/capacity
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

INFER_RATIO = 0.35


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def load(run_dir: str) -> pd.DataFrame:
    df = pd.DataFrame([json.loads(l) for l in open(Path(run_dir) / "rows.jsonl", encoding="utf-8")])
    df["ok"] = df["warm_ok"].astype(bool) & df["probe_ok"].astype(bool)
    df["hit_reported"] = df["ok"] & (df["probe_cached_tokens"] >= 0.5 * df["warm_prompt_tokens"])
    cold = df[(df["load"] == 1) & df["ok"]].groupby("model")["warm_latency_s"].median()
    df["cold_ref_s"] = df["model"].map(cold)
    df["hit_inferred"] = df["ok"] & (df["probe_latency_s"] < INFER_RATIO * df["cold_ref_s"])
    df["reports_hits"] = df["model"].map(df.groupby("model")["probe_cached_tokens"].max() > 0)
    df["hit"] = np.where(df["reports_hits"], df["hit_reported"], df["hit_inferred"])
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (m, n), g in df.groupby(["model", "load"]):
        k, tot = int(g["hit"].sum()), len(g)
        lo, hi = wilson(k, tot)
        per_round = g.groupby("rep").agg(slowest=("warm_latency_s", "max"), tokens=("warm_prompt_tokens", "sum"))
        rows.append({"model": m, "load": n, "agents": tot, "rounds": g["rep"].nunique(), "hit_rate": k / tot,
                     "hit_lo": lo, "hit_hi": hi, "errors": int((~g["ok"]).sum()),
                     "warm_p50_s": float(g["warm_latency_s"].median()), "probe_p50_s": float(g["probe_latency_s"].median()),
                     "prefill_tok_per_s": float((per_round["tokens"] / per_round["slowest"]).median()),
                     "hit_basis": "reported" if bool(g["reports_hits"].iloc[0]) else "inferred from latency"})
    return pd.DataFrame(rows)


def capacity(summary: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    out = []
    for m, g in summary.sort_values("load").groupby("model"):
        cap = 0
        for _, r in g.iterrows():
            if r["hit_rate"] >= threshold:
                cap = int(r["load"])
            else:
                break
        tested = int(g["load"].max())
        out.append({"model": m, "capacity": cap, "max_tested": tested, "censored": cap == tested,
                    "cold_p50_s": float(g[g["load"] == 1]["warm_p50_s"].iloc[0]) if (g["load"] == 1).any() else np.nan})
    return pd.DataFrame(out)


def inference_agreement(df: pd.DataFrame) -> dict:
    rep = df[df["reports_hits"] & df["ok"]]
    return {"agents": int(len(rep)), "agreement": float((rep["hit_reported"] == rep["hit_inferred"]).mean()) if len(rep) else np.nan}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    args = ap.parse_args()
    df = load(args.run_dir)
    s = summarize(df)
    c = capacity(s)
    d = Path(args.run_dir)
    s.to_csv(d / "summary.csv", index=False)
    c.to_csv(d / "capacity.csv", index=False)
    agree = inference_agreement(df)
    (d / "inference_check.json").write_text(json.dumps(agree, indent=1))
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(s.round(3).to_string(index=False))
        print(c.round(3).to_string(index=False))
    print("latency rule vs reported hits:", agree)


if __name__ == "__main__":
    main()
