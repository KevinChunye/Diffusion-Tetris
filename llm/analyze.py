"""analyze.py

Per-iteration analysis of a pilot directory (steps.csv, episodes.csv): one summary table and one
figure. All aggregation is pandas groupby / sort_values.

  python -m llm.analyze --dir runs/explore/iter01 --kind history
"""

from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np
import pandas as pd
import yaml

from llm import plotstyle

SUMMARY_COLS = ["arm", "episodes", "pieces", "lines", "score", "norm_score", "norm_lines", "regret_beam",
                "regret_rollout", "top1_beam", "illegal_rate", "prompt_tok", "cached_frac", "uncached_tok",
                "ttft_p50", "ttft_p90", "latency_p50", "completion_tok", "usd_per_episode", "usd_per_100_pieces",
                "peak_prompt_tok"]


def load(d: str):
    steps = pd.read_csv(os.path.join(d, "steps.csv"))
    eps = pd.read_csv(os.path.join(d, "episodes.csv"))
    with open(os.path.join(d, "config.yaml"), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    steps["uncached_tokens"] = steps["prompt_tokens"] - steps["cached_tokens"]
    steps["illegal"] = steps["fallback_reason"].isin(["illegal", "unparseable"]).astype(float)
    steps["fallback"] = steps["fallback_reason"].fillna("").ne("").astype(float)
    return steps, eps, cfg


def arm_order(cfg) -> List[str]:
    return list(cfg["arms"].keys())


def arm_summary(steps: pd.DataFrame, eps: pd.DataFrame, order: List[str]) -> pd.DataFrame:
    s = steps.groupby("arm").agg(
        prompt_tok=("prompt_tokens", "mean"), cached=("cached_tokens", "sum"), prompt_sum=("prompt_tokens", "sum"),
        uncached_tok=("uncached_tokens", "mean"), ttft_p50=("ttft_s", "median"),
        ttft_p90=("ttft_s", lambda x: x.quantile(0.9)), latency_p50=("latency_s", "median"),
        completion_tok=("completion_tokens", "mean"), illegal_rate=("illegal", "mean"),
        peak_prompt_tok=("prompt_tokens", "max"), cost=("cost_usd", "sum"), decisions=("turn", "size"))
    if "regret_beam" in steps:
        r = steps.groupby("arm").agg(regret_beam=("regret_beam", "mean"), regret_rollout=("regret_rollout", "mean"),
                                     top1_beam=("top1_beam", "mean"))
        s = s.join(r)
    e = eps.groupby("arm").agg(episodes=("episode_seed", "size"), pieces=("pieces_placed", "mean"),
                               lines=("lines_cleared", "mean"), score=("score", "mean"),
                               norm_score=("norm_score", "mean"), norm_lines=("norm_lines_cleared", "mean"))
    out = e.join(s)
    out["cached_frac"] = out["cached"] / out["prompt_sum"]
    out["usd_per_episode"] = out["cost"] / out["episodes"]
    out["usd_per_100_pieces"] = 100.0 * out["cost"] / out["decisions"]
    out = out.reset_index()
    out["arm"] = pd.Categorical(out["arm"], categories=order, ordered=True)
    out = out.sort_values("arm")
    return out[[c for c in SUMMARY_COLS if c in out.columns]]


def paired(eps: pd.DataFrame, steps: pd.DataFrame, baseline: str, metrics: List[str]) -> pd.DataFrame:
    """Per-seed difference of each arm vs `baseline`; mean, standard error, and sign agreement."""
    per_ep = eps.set_index(["arm", "episode_seed"])
    ep_steps = steps.groupby(["arm", "episode_seed"]).agg(
        ttft_p50=("ttft_s", "median"), cached_frac=("cached_tokens", "sum"), prompt_sum=("prompt_tokens", "sum"))
    ep_steps["cached_frac"] = ep_steps["cached_frac"] / ep_steps["prompt_sum"]
    per_ep = per_ep.join(ep_steps[["ttft_p50", "cached_frac"]])
    rows = []
    base = per_ep.xs(baseline, level="arm")
    for arm in per_ep.index.get_level_values("arm").unique():
        if arm == baseline:
            continue
        cur = per_ep.xs(arm, level="arm")
        for m in metrics:
            if m not in cur.columns:
                continue
            d = (cur[m] - base[m]).dropna()
            if d.empty:
                continue
            se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
            rows.append({"arm": arm, "metric": m, "vs": baseline, "mean_diff": float(d.mean()), "se": se,
                         "n_seeds": int(len(d)), "same_sign": int(max((d > 0).sum(), (d < 0).sum())),
                         "t": float(d.mean() / se) if se and se > 0 else float("nan")})
    return pd.DataFrame(rows)


def _end_labels(ax, ends):
    """Direct labels at line ends, skipped where they would collide (the legend carries identity)."""
    lo, hi = ax.get_ylim()
    min_gap = 0.045 * (hi - lo)
    placed: List[float] = []
    for y, x, text in sorted(ends):
        if all(abs(y - p) >= min_gap for p in placed):
            ax.annotate(text, (x, y), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8,
                        color=plotstyle.INK_2)
            placed.append(y)


def plot_history(steps: pd.DataFrame, order: List[str], title: str, out_png: str) -> None:
    """2x2 small multiples vs turn: prompt tokens, cached fraction, TTFT, cumulative $ per episode."""
    plotstyle.setup()
    import matplotlib.pyplot as plt

    colors = plotstyle.colors_for(order)
    steps = steps.copy()
    steps["cached_frac"] = steps["cached_tokens"] / steps["prompt_tokens"]
    steps = steps.sort_values(["arm", "episode_seed", "turn"])
    steps["cum_usd"] = steps.groupby(["arm", "episode_seed"])["cost_usd"].cumsum()
    per_turn = steps.groupby(["arm", "turn"]).agg(prompt=("prompt_tokens", "mean"), cached=("cached_frac", "mean"),
                                                 ttft=("ttft_s", "median"), cum_usd=("cum_usd", "mean")).reset_index()
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), constrained_layout=True)
    panels = [("prompt", "Prompt tokens per call", "tokens (thousands)", 1e-3),
              ("cached", "Share of prompt served from KV cache", "cached / prompt tokens", 1.0),
              ("ttft", "Time to first token (median over seeds)", "seconds", 1.0),
              ("cum_usd", "Cumulative cost per episode", "USD", 1.0)]
    for ax, (col, ttl, ylab, scale) in zip(axes.flat, panels):
        ends = []
        for arm in order:
            d = per_turn[per_turn["arm"] == arm].sort_values("turn")
            if d.empty:
                continue
            y = d[col] * scale
            if col == "ttft":
                y = y.rolling(5, min_periods=1, center=True).median()
            ax.plot(d["turn"], y, color=colors[arm], label=arm)
            ends.append((float(y.iloc[-1]), float(d["turn"].iloc[-1]), arm))
        if col == "cached":
            ax.set_ylim(-0.02, 1.02)
        _end_labels(ax, ends)
        ax.set_title(ttl, loc="left")
        ax.set_xlabel("turn (piece index)")
        ax.set_ylabel(ylab)
        ax.margins(x=0.12)
    axes.flat[0].legend(loc="upper left")
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_models(steps: pd.DataFrame, order: List[str], title: str, out_png: str) -> None:
    """Arms named '<model>/<history>': rows = cached share, $ per decision, TTFT; columns = model;
    color = history policy (same color for the same policy in every column)."""
    plotstyle.setup()
    import matplotlib.pyplot as plt

    steps = steps.copy()
    steps["model_tag"] = steps["arm"].str.split("/").str[0]
    steps["history"] = steps["arm"].str.split("/").str[1]
    steps["cached_frac"] = steps["cached_tokens"] / steps["prompt_tokens"]
    models = list(dict.fromkeys(a.split("/")[0] for a in order))
    histories = list(dict.fromkeys(a.split("/")[1] for a in order))
    colors = plotstyle.colors_for(histories)
    per_turn = steps.groupby(["model_tag", "history", "turn"]).agg(
        cached=("cached_frac", "mean"), usd=("cost_usd", "mean"), ttft=("ttft_s", "median"),
        n=("episode_seed", "size")).reset_index()
    per_turn = per_turn[per_turn["n"] >= 2]  # at least two seeds still alive at this turn
    rows = [("cached", "Share of prompt served from cache", "cached / prompt", False),
            ("usd", "Cost per decision", "USD (log scale)", True),
            ("ttft", "Time to first token (rolling median)", "seconds", False)]
    fig, axes = plt.subplots(len(rows), len(models), figsize=(4.2 * len(models), 9.2), sharey="row",
                             constrained_layout=True, squeeze=False)
    for j, m in enumerate(models):
        for i, (col, ttl, ylab, logy) in enumerate(rows):
            ax = axes[i][j]
            for h in histories:
                d = per_turn[(per_turn["model_tag"] == m) & (per_turn["history"] == h)].sort_values("turn")
                if d.empty:
                    continue
                y = d[col]
                if col == "ttft":
                    y = y.rolling(7, min_periods=1, center=True).median()
                ax.plot(d["turn"], y, color=colors[h], label=h)
            if logy:
                ax.set_yscale("log")
            if i == 0:
                ax.set_title(f"{m}\n{ttl}", loc="left")
                ax.set_ylim(-0.02, 1.02)
            else:
                ax.set_title(ttl, loc="left")
            ax.set_xlabel("turn (piece index)")
            if j == 0:
                ax.set_ylabel(ylab)
    axes[0][0].legend(loc="lower right")
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def write_tables(d: str, summary: pd.DataFrame, pairs: pd.DataFrame) -> str:
    md = "### Per-arm summary\n\n" + summary.round(4).to_markdown(index=False)
    if not pairs.empty:
        md += "\n\n### Paired differences vs baseline (per-seed diffs; t = mean/SE)\n\n" + pairs.round(4).to_markdown(index=False)
    summary.to_csv(os.path.join(d, "summary.csv"), index=False)
    pairs.to_csv(os.path.join(d, "paired.csv"), index=False)
    with open(os.path.join(d, "summary.md"), "w", encoding="utf-8") as f:
        f.write(md + "\n")
    return md


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--kind", default="history")
    ap.add_argument("--baseline", default="")
    args = ap.parse_args()
    steps, eps, cfg = load(args.dir)
    order = arm_order(cfg)
    summary = arm_summary(steps, eps, order)
    metrics = ["norm_score", "lines_cleared", "pieces_placed", "regret_beam", "regret_rollout", "cost_usd",
               "ttft_p50", "cached_frac"]
    if all("/" in a for a in order):  # '<model>/<policy>' arms: pair within each model
        base_policy = args.baseline or order[0].split("/")[1]
        parts = []
        for m in dict.fromkeys(a.split("/")[0] for a in order):
            keep = [a for a in order if a.startswith(m + "/")]
            parts.append(paired(eps[eps["arm"].isin(keep)], steps[steps["arm"].isin(keep)], f"{m}/{base_policy}", metrics))
        pairs = pd.concat(parts, ignore_index=True)
    else:
        pairs = paired(eps, steps, args.baseline or order[0], metrics)
    print(write_tables(args.dir, summary, pairs))
    title = f"Iteration {cfg.get('iteration')}: {cfg.get('name')} ({cfg.get('model')}, {len(cfg['seeds'])} seeds x {cfg['max_pieces']} pieces)"
    if args.kind == "history":
        plot_history(steps, order, title, os.path.join(args.dir, "figure.png"))
    elif args.kind == "models":
        plot_models(steps, order, title, os.path.join(args.dir, "figure.png"))


if __name__ == "__main__":
    main()
