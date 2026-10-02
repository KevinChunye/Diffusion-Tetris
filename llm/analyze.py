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
    order = pd.DataFrame(ends, columns=["y", "x", "text"]).sort_values("y")
    for y, x, text in order.itertuples(index=False):
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


def lifetime_summary(d: str) -> pd.DataFrame:
    lt = pd.read_csv(os.path.join(d, "lifetime.csv"))
    lt = lt[lt["ok"]].copy()
    lt["cached_frac"] = lt["cached_probe"] / lt["prompt_probe"]
    lt["speedup"] = lt["latency_warm"] / lt["latency_probe"]
    g = lt.groupby(["model", "gap_s"]).agg(
        trials=("trial", "size"), prompt=("prompt_probe", "median"), cached_frac=("cached_frac", "median"),
        vllm_cached=("vllm_cached_probe", "median"), lmcache_cached=("lmcache_cached_probe", "median"),
        cold_latency=("latency_warm", "median"), probe_latency=("latency_probe", "median"),
        speedup=("speedup", "median"), actual_gap=("actual_gap_s", "median"), cost=("cost_usd", "sum")).reset_index()
    return g.sort_values(["model", "gap_s"])


def plot_lifetime(g: pd.DataFrame, title: str, out_png: str) -> None:
    """Two panels vs idle gap (log x): prefill latency of the next turn, and share of it served from cache."""
    plotstyle.setup()
    import matplotlib.pyplot as plt

    models = list(dict.fromkeys(g["model"]))
    colors = plotstyle.colors_for(models)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
    hidden = []
    for m in models:
        d = g[g["model"] == m].sort_values("gap_s")
        x = d["gap_s"].clip(lower=1)  # 0 s gap drawn at 1 s on the log axis
        label = m.split("/")[-1]
        axes[0].plot(x, d["probe_latency"], color=colors[m], marker="o", markersize=5, label=label)
        axes[0].scatter([0.6], [d["cold_latency"].median()], color=colors[m], marker="x", s=36)
        if d["cached_frac"].max() <= 0 and d["probe_latency"].min() < 0.2 * d["cold_latency"].median():
            hidden.append(label)  # fast warm turns but 0 cached tokens reported: hits are not reported
            continue
        axes[1].plot(x, d["cached_frac"], color=colors[m], marker="o", markersize=5, label=label)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks([0.6, 1, 5, 30, 120, 600])
        ax.set_xticklabels(["cold", "0 s", "5 s", "30 s", "2 min", "10 min"])
        ax.set_xlabel("idle gap before the next turn")
    axes[0].set_title("Latency of the next turn (max_tokens=1 ≈ prefill); x = cold", loc="left")
    axes[0].set_ylabel("seconds (log scale, median of trials)")
    axes[0].set_yscale("log")
    if hidden:
        axes[1].text(0.02, 0.04, f"not shown (reports 0 cached tokens even on hits): {', '.join(hidden)}",
                     transform=axes[1].transAxes, fontsize=8, color=plotstyle.INK_2)
    axes[1].set_title("Share of the next turn's prompt served from cache", loc="left")
    axes[1].set_ylabel("cached / prompt tokens")
    axes[1].set_ylim(-0.02, 1.02)
    axes[0].legend(loc="upper right")
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def capacity_tables(d: str):
    """Iteration-4 style: subdirs of lifetime runs (load1..16, keepalive, isolated, glm, map)."""
    import glob

    parts = []
    for f in pd.Series(glob.glob(os.path.join(d, "*", "lifetime.csv")), dtype=str).sort_values():
        x = pd.read_csv(f)
        x["phase"] = os.path.basename(os.path.dirname(f))
        parts.append(x)
    lt = pd.concat(parts, ignore_index=True)
    lt["reported_hit"] = lt["cached_probe"] > 0.5 * lt["prompt_probe"]
    # Use reported cached tokens wherever a model reports them. Only for models that never report
    # (GLM) fall back to latency vs that model's sequential cold prefill, because queued warm-ups
    # make warm latency a bad baseline under load.
    reports = lt.groupby("model")["cached_probe"].max().gt(0).rename("reports_cache")
    lt = lt.join(reports, on="model")
    cold_ref = lt[lt["phase"].isin(["glm", "isolated", "load1"])].groupby("model")["latency_warm"].median().rename("cold_ref")
    lt = lt.join(cold_ref, on="model")
    lt["hit"] = lt["reported_hit"].where(lt["reports_cache"], lt["latency_probe"] < 0.3 * lt["cold_ref"])
    # One context at a time (the isolated control at the same 30 s gap) is the N=1 point too.
    load = lt[lt["phase"].str.startswith("load") | ((lt["phase"] == "isolated") & (lt["gap_s"] == 30))].copy()
    load["contexts"] = load["phase"].str.replace("load", "").replace("isolated", "1").astype(int)
    cap = load.groupby(["model", "contexts"]).agg(
        trials=("trial", "size"), hit_rate=("hit", "mean"), probe_p50=("latency_probe", "median"),
        warm_p50=("latency_warm", "median"), warm_max=("latency_warm", "max")).reset_index()
    other = lt[~lt["phase"].str.startswith("load")].groupby(["phase", "model", "tag", "gap_s"]).agg(
        trials=("trial", "size"), hit_rate=("hit", "mean"), reported_hit_rate=("reported_hit", "mean"),
        cold_p50=("latency_warm", "median"), probe_p50=("latency_probe", "median"), pings=("pings", "mean"),
        cost=("cost_usd", "sum")).reset_index()
    return lt, cap, other


def plot_capacity(cap: pd.DataFrame, title: str, out_png: str) -> None:
    plotstyle.setup()
    import matplotlib.pyplot as plt

    models = list(dict.fromkeys(cap["model"]))
    colors = plotstyle.colors_for(models)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
    for m in models:
        d = cap[cap["model"] == m].sort_values("contexts")
        label = m.split("/")[-1]
        axes[0].plot(d["contexts"], d["hit_rate"], color=colors[m], marker="o", markersize=6, label=label)
        axes[1].plot(d["contexts"], d["probe_p50"], color=colors[m], marker="o", markersize=6, label=label)
        axes[1].plot(d["contexts"], d["warm_p50"], color=colors[m], linestyle=(0, (1, 2)), linewidth=1.5)
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 4, 8, 16])
        ax.set_xticklabels(["1", "4", "8", "16"])
        ax.set_xlabel("concurrent distinct 15k-token agent contexts (ours)")
    axes[0].set_title("Next turn hits the cache after 30 s idle", loc="left")
    axes[0].set_ylabel("hit rate")
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].legend(loc="center right")
    axes[1].set_title("Next-turn latency (solid) vs cold prefill (dotted)", loc="left")
    axes[1].set_ylabel("seconds (log scale, median)")
    axes[1].set_yscale("log")
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def fanout_summary(d: str) -> pd.DataFrame:
    import yaml as _yaml

    fo = pd.read_csv(os.path.join(d, "fanout.csv"))
    pricing = _yaml.safe_load(open("configs/pricing.yaml"))["models"]
    fo["billed_full_price_tokens"] = [
        r.prompt_tokens if pricing[r.model]["cached"] is None else r.computed_tokens for r in fo.itertuples()]
    g = fo.groupby(["model", "k", "strategy"]).agg(
        trials=("trial", "size"), prefix_tokens=("prompt_per_request", "median"),
        computed_tokens=("computed_tokens", "median"), billed_full_price_tokens=("billed_full_price_tokens", "median"),
        makespan_s=("makespan_s", "median"), latency_p50=("latency_p50", "median"), cost_usd=("cost_usd", "median"),
        retries=("retries", "sum")).reset_index()
    g["prefills_computed"] = g["computed_tokens"] / g["prefix_tokens"]
    g["prefills_billed"] = g["billed_full_price_tokens"] / g["prefix_tokens"]
    return g


def plot_fanout(g: pd.DataFrame, title: str, out_png: str) -> None:
    """Left: prefills computed vs billed at full price (K=8). Right: makespan by strategy (K=8)."""
    plotstyle.setup()
    import matplotlib.pyplot as plt
    import numpy as np

    k = int(g["k"].max())
    d = g[g["k"] == k]
    models = list(dict.fromkeys(d["model"]))
    strategies = ["cold_fanout", "primed_fanout", "sequential"]
    colors = plotstyle.colors_for(strategies)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    x = np.arange(len(models))
    width = 0.24
    for i, (col, label) in enumerate([("prefills_computed", "prefills computed by the server"),
                                      ("prefills_billed", "prefills billed at full input price")]):
        vals = d[d["strategy"] == "cold_fanout"].set_index("model").loc[models, col]
        bars = axes[0].bar(x + (i - 0.5) * (width + 0.04), vals, width, color=plotstyle.SERIES[i], label=label)
        for b, v in zip(bars, vals):
            axes[0].annotate(f"{v:.1f}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 3), textcoords="offset points",
                             ha="center", fontsize=8, color=plotstyle.INK_2)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([m.split("/")[-1] for m in models])
    axes[0].set_ylabel(f"multiples of the 15k-token prefix (K={k} requests)")
    axes[0].set_title(f"Cold fan-out of K={k}: computed once, billed K times?", loc="left")
    axes[0].legend(loc="upper right")
    for j, st in enumerate(strategies):
        vals = d[d["strategy"] == st].set_index("model").loc[models, "makespan_s"]
        axes[1].bar(x + (j - 1) * (width + 0.03), vals, width, color=colors[st], label=st.replace("_", " "))
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([m.split("/")[-1] for m in models])
    axes[1].set_ylabel("seconds until all K answers (median)")
    axes[1].set_title(f"Makespan by strategy (K={k})", loc="left")
    axes[1].legend(loc="upper right")
    for ax in axes:
        ax.grid(axis="x", visible=False)
        ax.set_axisbelow(True)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _pareto(points: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    """Non-dominated points (lower x, higher y), sorted by x."""
    pts = points.sort_values([x, y], ascending=[True, False])
    best = -np.inf
    keep = []
    for idx, row in pts.iterrows():
        if row[y] > best:
            keep.append(idx)
            best = row[y]
    return pts.loc[keep]


def plot_ladder(summary: pd.DataFrame, eps: pd.DataFrame, cached_price: dict, title: str, out_png: str) -> None:
    """Normalized score vs $ / 100 decisions and vs median decision latency; Pareto front dashed."""
    plotstyle.setup()
    import matplotlib.pyplot as plt

    spread = eps.groupby("arm")["norm_score"].std().rename("norm_sd")
    pts = summary.set_index("arm").join(spread).reset_index()
    pts["arm"] = pts["arm"].astype(str)
    pts["cached_free"] = pts["arm"].map(cached_price)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), constrained_layout=True)
    for ax, xcol, xlab in [(axes[0], "usd_per_100_pieces", "USD per 100 decisions (log scale)"),
                           (axes[1], "latency_p50", "median decision latency, s (log scale)")]:
        for free, color, marker, label in [(True, plotstyle.SERIES[0], "o", "cached input billed $0"),
                                           (False, plotstyle.SERIES[1], "s", "no cached price (full input price)")]:
            d = pts[pts["cached_free"] == free]
            ax.errorbar(d[xcol], d["norm_score"], yerr=d["norm_sd"], fmt=marker, color=color, ms=8,
                        mec=plotstyle.SURFACE, mew=1.5, elinewidth=1, capsize=0, label=label)
        front = _pareto(pts, xcol, "norm_score")
        ax.plot(front[xcol], front["norm_score"], linestyle=(0, (4, 3)), color=plotstyle.INK_2, linewidth=1, zorder=0)
        for r in pts.itertuples():
            ax.annotate(r.arm, (getattr(r, xcol), r.norm_score), xytext=(6, 4), textcoords="offset points",
                        fontsize=8, color=plotstyle.INK_2)
        ax.set_xscale("log")
        ax.set_xlabel(xlab)
        ax.set_ylabel("normalized score (0 = random, 1 = beam search)")
        ax.axhline(0, color=plotstyle.GRID, linewidth=1)
    axes[0].set_title("Quality vs cost (dashed: Pareto front)", loc="left")
    axes[1].set_title("Quality vs latency", loc="left")
    axes[0].legend(loc="upper left")
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
    if args.kind == "fanout":
        g = fanout_summary(args.dir)
        md = "### Fan-out over a shared ~14k-token prefix (medians of 3 trials)\n\n" + g.round(3).to_markdown(index=False)
        g.to_csv(os.path.join(args.dir, "summary.csv"), index=False)
        with open(os.path.join(args.dir, "summary.md"), "w", encoding="utf-8") as f:
            f.write(md + "\n")
        print(md)
        plot_fanout(g, "Iteration 5: K parallel questions over one long Tetris context (LLM-reranker pattern)",
                    os.path.join(args.dir, "figure.png"))
        return
    if args.kind == "capacity":
        lt, cap, other = capacity_tables(args.dir)
        md = ("### Cache hit after 30 s idle vs concurrent contexts\n\n" + cap.round(3).to_markdown(index=False)
              + "\n\n### Other phases (isolated control, keep-alive, GLM sequential, catalog map)\n\n"
              + other.round(3).to_markdown(index=False))
        cap.to_csv(os.path.join(args.dir, "summary_capacity.csv"), index=False)
        other.to_csv(os.path.join(args.dir, "summary_other.csv"), index=False)
        with open(os.path.join(args.dir, "summary.md"), "w", encoding="utf-8") as f:
            f.write(md + "\n")
        print(md)
        plot_capacity(cap, "Iteration 4: how many agent contexts does a deployment keep warm? (30 s idle, 15k tokens each)",
                      os.path.join(args.dir, "figure.png"))
        return
    if args.kind == "lifetime":
        with open(os.path.join(args.dir, "config.yaml"), "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        g = lifetime_summary(args.dir)
        md = "### KV-cache lifetime (medians over trials)\n\n" + g.round(3).to_markdown(index=False)
        g.to_csv(os.path.join(args.dir, "summary.csv"), index=False)
        with open(os.path.join(args.dir, "summary.md"), "w", encoding="utf-8") as f:
            f.write(md + "\n")
        print(md)
        plot_lifetime(g, f"Iteration {cfg['iteration']}: KV-cache lifetime under idle gaps "
                         f"(~{int(g['prompt'].median()):,}-token Tetris history, {int(g['trials'].max())} trials)",
                      os.path.join(args.dir, "figure.png"))
        return
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
            base = f"{m}/{base_policy}" if f"{m}/{base_policy}" in keep else keep[0]
            if len(keep) > 1:
                parts.append(paired(eps[eps["arm"].isin(keep)], steps[steps["arm"].isin(keep)], base, metrics))
        pairs = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    else:
        pairs = paired(eps, steps, args.baseline or order[0], metrics)
    print(write_tables(args.dir, summary, pairs))
    models_used = list(dict.fromkeys(a.split("/")[0] for a in order)) if all("/" in a for a in order) else [cfg.get("model")]
    title = (f"Iteration {cfg.get('iteration')}: {cfg.get('name')} ({', '.join(models_used)}; "
             f"{len(cfg['seeds'])} seeds x {cfg['max_pieces']} pieces)")
    if args.kind == "ladder":
        pricing = yaml.safe_load(open("configs/pricing.yaml"))["models"]
        arm_model = steps.groupby("arm")["model"].first()
        cached_free = {a: pricing[m]["cached"] is not None for a, m in arm_model.items()}
        plot_ladder(summary, eps, cached_free, f"Iteration {cfg.get('iteration')}: model ladder, stateless + constrained "
                    f"output ({len(cfg['seeds'])} seeds x {cfg['max_pieces']} pieces)", os.path.join(args.dir, "figure.png"))
        return
    if args.kind == "history":
        plot_history(steps, order, title, os.path.join(args.dir, "figure.png"))
    elif args.kind == "models":
        plot_models(steps, order, title, os.path.join(args.dir, "figure.png"))


if __name__ == "__main__":
    main()
