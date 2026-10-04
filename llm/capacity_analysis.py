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

Several run directories (parts of one sweep) can be pooled. Part 2 of the sweep never ran two rounds with
N >= 32 at once, because in part 1 such overlaps drew gateway-wide "Server is overloaded" 429s. The same
rule is applied to all parts after the fact: any round that was in flight while two N >= 32 rounds
overlapped is excluded (`contended`), and the number of excluded rounds is reported.

  python -m llm.capacity_analysis runs/explore/scaleup/capacity runs/explore/scaleup/capacity_part2 \
      --out runs/explore/scaleup/capacity_analysis
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


BIG = 32


def _contended_rounds(df: pd.DataFrame) -> set:
    """Rounds (part, model, rep, load) that were in flight while two rounds with load >= BIG overlapped."""
    df = df.assign(t_end=df["t_start"] + df["warm_latency_s"] + df["gap_actual_s"] + df["probe_latency_s"])
    out = set()
    for part, g in df.groupby("part"):
        r = g.groupby(["model", "rep", "load"]).agg(t0=("t_start", "min"), t1=("t_end", "max")).reset_index()
        big = r[r["load"] >= BIG].to_dict("records")
        windows = [(max(a["t0"], b["t0"]), min(a["t1"], b["t1"])) for i, a in enumerate(big) for b in big[i + 1:]
                   if max(a["t0"], b["t0"]) < min(a["t1"], b["t1"])]
        for x in r.itertuples():
            if any(x.t0 < w1 and w0 < x.t1 for w0, w1 in windows):
                out.add((part, x.model, x.rep, x.load))
    return out


def load(run_dirs) -> pd.DataFrame:
    run_dirs = [run_dirs] if isinstance(run_dirs, (str, Path)) else list(run_dirs)
    parts = []
    for i, d in enumerate(run_dirs):
        p = pd.DataFrame([json.loads(l) for l in open(Path(d) / "rows.jsonl", encoding="utf-8")])
        parts.append(p.assign(part=i))
    df = pd.concat(parts, ignore_index=True)
    bad = _contended_rounds(df)
    df["contended"] = [(a, b, c, d) in bad for a, b, c, d in zip(df["part"], df["model"], df["rep"], df["load"])]
    df["ok"] = df["warm_ok"].astype(bool) & df["probe_ok"].astype(bool)
    df["hit_reported"] = df["ok"] & (df["probe_cached_tokens"] >= 0.5 * df["warm_prompt_tokens"])
    cold = df[(df["load"] == 1) & df["ok"] & ~df["contended"]].groupby("model")["warm_latency_s"].median()
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
        ok = g[g["ok"]]
        per_round = g.groupby("rep").agg(slowest=("warm_latency_s", "max"), tokens=("warm_prompt_tokens", "sum"))
        rows.append({"model": m, "load": n, "agents": tot, "rounds": g["rep"].nunique(), "hit_rate": k / tot,
                     "hit_lo": lo, "hit_hi": hi, "errors": int((~g["ok"]).sum()),
                     "warm_p50_s": float(ok["warm_latency_s"].median()), "probe_p50_s": float(ok["probe_latency_s"].median()),
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
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    raw = load(args.run_dirs)
    df = raw[~raw["contended"]]
    s = summarize(df)
    c = capacity(s)
    d = Path(args.out or args.run_dirs[0])
    d.mkdir(parents=True, exist_ok=True)
    raw.to_csv(d / "agents.csv", index=False)
    s.to_csv(d / "summary.csv", index=False)
    c.to_csv(d / "capacity.csv", index=False)
    excl = raw[raw["contended"]].groupby(["part", "model", "rep", "load"]).size().reset_index(name="agents")
    excl.to_csv(d / "excluded_rounds.csv", index=False)
    print(f"excluded {len(excl)} contended rounds ({int(excl['agents'].sum())} agents); "
          f"errors in excluded rounds: {int((~raw[raw['contended']]['ok']).sum())}, in kept rounds: {int((~df['ok']).sum())}")
    agree = inference_agreement(df)
    (d / "inference_check.json").write_text(json.dumps(agree, indent=1))
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(s.round(3).to_string(index=False))
        print(c.round(3).to_string(index=False))
    print("latency rule vs reported hits:", agree)


if __name__ == "__main__":
    main()
