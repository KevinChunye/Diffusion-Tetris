"""figures.py - every figure in the paper, built from the committed raw logs under runs/explore/.

    python paper/figures.py                 # writes paper/figures/*.pdf and paper/figures/facts.json
    FIG_PREVIEW=/some/dir python paper/figures.py   # also writes PNG previews there

Design rules, applied to every figure:
- the panel title states the finding, not just the variable;
- the series the finding is about carry color and everything else is gray context;
- series are labeled where they end and reference lines where they are drawn;
- one y-axis per panel, hairline grid, thin marks, 2-pt surface ring on dots.
Model colors are fixed across the paper. The four hues (aqua, violet, blue, orange) pass a color-vision
deficiency check on all pairs; any further model is drawn in neutral gray and labeled by name.
Sizes match a two-column page: COL = 3.25 in, FULL = 6.75 in, text at 6.5-8 pt.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "explore"
OUT = Path(__file__).resolve().parent / "figures"
sys.path.insert(0, str(ROOT))

COL, FULL = 3.25, 6.75
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#a3a29d", "#e6e5e1"
CONTEXT = "#c4c3bd"          # gray for context marks
NEUTRAL = "#52514e"          # a fifth highlighted series, labeled by name
RED = "#e34948"              # the problem condition in E4
MODEL = {"gemma-4-31B": "#2a78d6", "DeepSeek-V4-Flash": "#eb6834", "gpt-oss-20b": "#1baf7a", "gpt-oss-120b": "#4a3aa7"}
SHORT = {"openai/gpt-oss-20b": "gpt-oss-20b", "openai/gpt-oss-120b": "gpt-oss-120b", "google/gemma-4-31B-it": "gemma-4-31B",
         "deepseek-ai/DeepSeek-V4-Flash": "DeepSeek-V4-Flash", "MiniMaxAI/MiniMax-M2.5": "MiniMax-M2.5",
         "Qwen/Qwen3.8-27B-FP8": "Qwen3.8-27B", "Qwen/Qwen3.5-397B-A17B-FP8": "Qwen3.5-397B",
         "moonshotai/Kimi-K2.7-Code": "Kimi-K2.7", "lukealonso/GLM-5.2-NVFP4": "GLM-5.2"}
ARM_LABEL = {"gpt-oss-20b/low": "gpt-oss-20b", "gpt-oss-120b/low": "gpt-oss-120b", "gpt-oss-120b/medium": "gpt-oss-120b (medium)",
             "gemma-4-31B/direct": "gemma-4-31B", "DeepSeek-V4-Flash/direct": "DeepSeek-V4-Flash",
             "Qwen3.8-27B/direct": "Qwen3.8-27B", "Qwen3.5-397B/direct": "Qwen3.5-397B", "Kimi-K2.7/direct": "Kimi-K2.7",
             "GLM-5.2/direct": "GLM-5.2"}

for _ttf in sorted((Path(__file__).resolve().parent / "fonts").glob("SourceSans3-*.ttf")):
    font_manager.fontManager.addfont(str(_ttf))

plt.rcParams.update({
    "font.family": "Source Sans 3", "font.size": 7.5, "axes.titlesize": 8, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.titlepad": 6, "axes.labelsize": 7.5, "axes.labelcolor": INK2,
    "axes.edgecolor": GRID, "axes.linewidth": 0.7, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "xtick.major.size": 0, "ytick.major.size": 0,
    "xtick.minor.size": 0, "ytick.minor.size": 0, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "legend.fontsize": 7, "pdf.fonttype": 42, "figure.dpi": 100, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round"})

FACTS: dict = {}


def _save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    preview = os.environ.get("FIG_PREVIEW")
    if preview:
        fig.savefig(Path(preview) / f"{name}.png", dpi=200)
    plt.close(fig)


def _dot(ax, x, y, color, size=5.0, marker="o", z=3, hollow=False):
    if hollow:
        ax.plot([x], [y], marker, ms=size, mfc="white", mec=color, mew=1.1, zorder=z)
    else:
        ax.plot([x], [y], marker, ms=size, color=color, mec="white", mew=1.0, zorder=z)


def _label(ax, x, y, text, dx=4, dy=0, ha=None, color=INK, size=6.8, weight=None):
    ha = ha or ("left" if dx > 0 else "right" if dx < 0 else "center")
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center", fontsize=size,
                color=color, fontweight=weight)


def _log_ticks(ax, axis, ticks, fmt):
    lo, hi = (ax.get_xlim() if axis == "x" else ax.get_ylim())
    tk = [t for t in ticks if lo <= t <= hi]
    if axis == "x":
        ax.set_xticks(tk)
        ax.set_xticklabels([fmt(t) for t in tk])
        ax.xaxis.set_minor_formatter(NullFormatter())
    else:
        ax.set_yticks(tk)
        ax.set_yticklabels([fmt(t) for t in tk])
        ax.yaxis.set_minor_formatter(NullFormatter())


def _arm_color(arm: str):
    fam = arm.split("/")[0]
    return MODEL.get(fam)


def _pool(g: pd.DataFrame) -> float:
    return float((g["score"] - g["ref_random_score"]).sum() / (g["ref_beam_score"] - g["ref_random_score"]).sum())


# ---- E1: price and size do not buy decisions -----------------------------------------------------------

def fig_ladder() -> None:
    d = RUNS / "scaleup" / "ladder"
    s = pd.read_csv(d / "summary_scaleup.csv")
    an = json.loads((d / "analysis.json").read_text())
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(FULL, 2.35), sharey=True, gridspec_kw={"wspace": 0.08})
    for a in (ax, ax2):
        a.set_xscale("log")
        a.axhline(1.0, color=INK2, lw=0.7, zorder=1)
        a.set_ylim(-0.03, 1.1)
    off_a = {"gpt-oss-120b/medium": (0, 11), "GLM-5.2/direct": (-5, 0), "Qwen3.5-397B/direct": (5, -5),
             "Kimi-K2.7/direct": (5, 4), "Qwen3.8-27B/direct": (5, 3)}
    off_b = {"gpt-oss-120b/low": (5, 6), "GLM-5.2/direct": (-5, 0), "Kimi-K2.7/direct": (-5, 0),
             "DeepSeek-V4-Flash/direct": (-5, 0), "Qwen3.5-397B/direct": (5, 5), "Qwen3.8-27B/direct": (5, -5)}
    for _, r in s.iterrows():
        col = _arm_color(r["arm"])
        hl = col is not None
        for a, x, off in ((ax, r["usd_per_100_stated"], off_a), (ax2, r["total_b"], off_b)):
            a.plot([x, x], [r["norm_score_lo"], r["norm_score_hi"]], color=col or CONTEXT, lw=1.1,
                   alpha=0.55 if hl else 0.9, zorder=2)
            _dot(a, x, r["norm_score"], col or MUTED, size=5.5 if hl else 4.5)
            if a is ax2 and r["arm"] == "gpt-oss-120b/medium":
                continue  # same model and size as gpt-oss-120b; labeled once in panel b
            dx, dy = off.get(r["arm"], (5, 0))
            text = ARM_LABEL[r["arm"]]
            if a is ax and r["arm"] == "GLM-5.2/direct":
                text = "GLM-5.2: priciest, plays worst"
            if a is ax and r["arm"] == "gpt-oss-120b/low":
                text = "gpt-oss-120b: best at $0.015"
            _label(a, x, r["norm_score"], text, dx, dy, color=INK if hl else INK2, size=6.6)
    _log_ticks(ax, "x", [0.005, 0.01, 0.02, 0.05, 0.1, 0.2], lambda t: f"${t:g}")
    _log_ticks(ax2, "x", [20, 50, 100, 200, 500, 1000], lambda t: f"{t:g}B")
    sp = an["spearman_vs_score"]
    for a, key, name in ((ax, "price_stated", "price"), (ax2, "total_params", "size")):
        a.text(0.02, 0.04, f"rank correlation with {name}: {sp[key]['rho']:+.2f}\n95% interval [{sp[key]['lo']:+.2f}, {sp[key]['hi']:+.2f}]",
               transform=a.transAxes, fontsize=6.6, color=INK2, va="bottom")
        a.text(0.98, 1.0, "beam-search bot", transform=a.get_yaxis_transform(), ha="right", va="bottom", fontsize=6.6,
               color=INK2)
    ax.set_xlabel("$ per 100 decisions (log)")
    ax2.set_xlabel("total parameters (log)")
    ax.set_ylabel("score vs. beam-search bot")
    ax.set_title("a  Paying more did not buy better play")
    ax2.set_title("b  Nor did a larger model")
    _save(fig, "ladder")
    FACTS["ladder"] = {"summary": s.round(4).to_dict("records"), "analysis": an}


# ---- E2: what history costs, and what it does to play -------------------------------------------------

def fig_memory() -> None:
    d = RUNS / "scaleup" / "memory"
    s = pd.read_csv(d / "summary_scaleup.csv")
    an = json.loads((d / "analysis.json").read_text())
    order = ["stateless", "append", "window8"]
    names = {"stateless": "none", "append": "all turns", "window8": "last 8"}
    color = {**MODEL, "Qwen3.8-27B": NEUTRAL}
    models = ["gpt-oss-20b", "gpt-oss-120b", "DeepSeek-V4-Flash", "Qwen3.8-27B"]
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.2), gridspec_kw={"wspace": 0.42})
    panels = ((axes[0], "uncached_per_decision", "a  Appended history is cached", "uncached tokens per move (log)", True),
              (axes[1], "usd_per_100_stated", "b  Its cost hinges on pricing", "$ per 100 moves (log)", True),
              (axes[2], "regret", "c  But it plays worse", "oracle regret per move", False))
    for ax, col, title, ylab, log in panels:
        if log:
            ax.set_yscale("log")
        for m in models:
            g = s[s["model"] == m].set_index("policy").reindex(order)
            xs = [i for i, p in enumerate(order) if pd.notna(g.loc[p, col])]
            ax.plot(xs, g[col].iloc[xs], color=color[m], lw=1.6)
            for i in xs:
                _dot(ax, i, g[col].iloc[i], color[m], size=4.8)
            if col == "usd_per_100_stated":
                for i in xs:
                    v, u = g[col].iloc[i], g["usd_per_100_full"].iloc[i]
                    if u > v * 1.05:
                        _dot(ax, i, u, color[m], size=4.3, hollow=True)
                        ax.plot([i, i], [v, u], color=color[m], lw=0.6, alpha=0.5, zorder=1)
        ax.set_xticks(range(3))
        ax.set_xticklabels([names[p] for p in order])
        ax.set_xlim(-0.3, 2.3)
        ax.set_title(title)
        ax.set_ylabel(ylab)
        ax.set_xlabel("history resent each move")
        ax.grid(axis="x", visible=False)
    _log_ticks(axes[0], "y", [300, 1000, 3000, 10000], lambda t: f"{t / 1000:g}k" if t >= 1000 else f"{t:g}")
    _log_ticks(axes[1], "y", [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5], lambda t: f"${t:g}")
    axes[1].annotate("if cached input were\nbilled in full: 11–12×", (1, 0.53), xytext=(14, -2), textcoords="offset points",
                     fontsize=6.6, color=INK2, ha="left", va="center")
    handles = [plt.Line2D([], [], color=color[m], lw=1.6, marker="o", ms=4.3, mec="white") for m in models]
    handles.append(plt.Line2D([], [], ls="", marker="o", ms=4.3, mfc="white", mec=INK2, mew=1.1))
    fig.legend(handles, models + ["cached input billed in full"], loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.98),
               handlelength=1.4, columnspacing=1.2, handletextpad=0.4)
    _save(fig, "memory")
    FACTS["memory"] = {"summary": s.round(4).to_dict("records"), "paired": an["paired"]}


# ---- E3: how many agents a deployment keeps warm ------------------------------------------------------

CAP_DIRS = [d for d in (RUNS / "scaleup" / f"capacity{x}" for x in ("", "_part2", "_part3")) if (d / "rows.jsonl").exists()]
CAP_MAIN = ["openai/gpt-oss-20b", "MiniMaxAI/MiniMax-M2.5", "google/gemma-4-31B-it", "deepseek-ai/DeepSeek-V4-Flash"]


def _cap_color(m: str):
    return MODEL.get(SHORT[m]) or (NEUTRAL if m in CAP_MAIN else None)


def _capacity_data():
    from llm.capacity_analysis import capacity, load, summarize
    from llm.model_arch import PARAMS_B, kv_bytes

    raw = load(CAP_DIRS)
    df = raw[~raw["contended"]]
    s = summarize(df)
    s["err_rate"] = s["errors"] / s["agents"]
    c = capacity(s)
    ctx = float(df["warm_prompt_tokens"][df["warm_ok"].astype(bool)].median())
    c["kv_mb_per_ctx"] = [kv_bytes(m, int(ctx)) / 2**20 for m in c["model"]]
    c["active_b"] = [PARAMS_B[m][1] for m in c["model"]]
    c["peak"] = c["model"].map(s.groupby("model")["prefill_tok_per_s"].max())
    return raw, df, s, c, ctx


def fig_capacity() -> None:
    raw, df, s, c, ctx = _capacity_data()
    fig, axes = plt.subplots(1, 4, figsize=(FULL, 2.05), gridspec_kw={"wspace": 0.5})
    ax, ax2, ax3, ax4 = axes
    for m in CAP_MAIN:
        g = s[s["model"] == m].sort_values("load")
        col = _cap_color(m)
        ax.plot(g["load"], g["hit_rate"] * 100, color=col, lw=1.6)
        ax2.plot(g["load"], g["prefill_tok_per_s"] / 1000, color=col, lw=1.6)
        for x, y, z in zip(g["load"], g["hit_rate"] * 100, g["prefill_tok_per_s"] / 1000):
            _dot(ax, x, y, col, size=4.0)
            _dot(ax2, x, z, col, size=4.0)
    for a in (ax, ax2):
        a.set_xscale("log", base=2)
        a.set_xticks([1, 4, 16, 64])
        a.set_xticklabels(["1", "4", "16", "64"])
        a.set_xlabel("concurrent agents")
    ax.set_ylim(-6, 112)
    ax.set_yticks([0, 50, 100])
    ax.set_yticklabels(["0%", "50%", "100%"])
    ax.set_ylabel("contexts still cached")
    ax.set_title("a  Warm after 30 s idle")
    gem = s[(s["model"] == "google/gemma-4-31B-it")].set_index("load")["hit_rate"]
    ax.text(1.05, 52, f"gemma:\n{gem[16]:.0%} at 16,\n{gem[32]:.0%} at 32", fontsize=6.3, color=INK2, va="center")
    ax.text(1.05, 18, "DeepSeek: 4", fontsize=6.3, color=INK2, va="center")
    ax2.set_ylabel("thousand tokens / s")
    ax2.set_title("b  Prefill throughput")
    ax2.set_ylim(0, 35)
    at8 = {m: s[(s["model"] == m) & (s["load"] == 8)]["prefill_tok_per_s"].iloc[0] / 1000 for m in CAP_MAIN}
    _label(ax2, 8, at8["openai/gpt-oss-20b"], "gpt-oss-20b", -4, 5, ha="right", size=6.3, color=INK2)
    _label(ax2, 8, at8["MiniMaxAI/MiniMax-M2.5"], "MiniMax", 0, 6, size=6.3, color=INK2)
    _label(ax2, 8, at8["google/gemma-4-31B-it"], "gemma, DeepSeek", 0, 7, size=6.3, color=INK2)
    ax2.set_xlim(0.8, 64 * 1.25)
    r = lambda x, y: x.rank().corr(y.rank())
    for a, col, xlabel, title, xs in ((ax3, "kv_mb_per_ctx", "KV per context, MB (log)", "c  Not set by KV size", 1),
                                      (ax4, "peak", "peak prefill, k tok/s (log)", "d  Set by throughput", 1000)):
        for _, row in c.iterrows():
            colr = _cap_color(row["model"])
            x = row[col] / xs
            _dot(a, x, row["capacity"], colr or MUTED, size=4.6 if colr else 4.0)
            if row["censored"]:
                a.annotate("", xy=(x, row["capacity"] * 1.6), xytext=(x, row["capacity"] * 1.1),
                           arrowprops=dict(arrowstyle="-|>", color=colr or MUTED, lw=0.8, mutation_scale=5))
        a.set_xscale("log")
        a.set_yscale("log", base=2)
        a.set_yticks([4, 8, 16, 32, 64])
        a.set_yticklabels(["4", "8", "16", "32", "64"])
        a.yaxis.set_minor_formatter(NullFormatter())
        a.set_ylim(2.8, 125)
        a.set_xlabel(xlabel)
        a.set_ylabel("contexts kept warm")
        a.set_title(title)
        rho = r(c["capacity"], c[col])
        a.text(0.04, 0.95, f"$\\rho$ = {rho:+.2f}", transform=a.transAxes, fontsize=7, color=INK, va="top")
    _log_ticks(ax3, "x", [20, 50, 100, 200, 500, 1000], lambda t: f"{t:g}")
    _log_ticks(ax4, "x", [2, 5, 10, 20], lambda t: f"{t:g}")
    for name, a, col, xs, dx, dy in (("DeepSeek-V4-Flash", ax3, "kv_mb_per_ctx", 1, 4, 0), ("MiniMax-M2.5", ax3, "kv_mb_per_ctx", 1, -5, 0),
                                     ("DeepSeek-V4-Flash", ax4, "peak", 1000, 4, 0), ("GLM-5.2", ax4, "peak", 1000, 0, 8)):
        row = c[c["model"].map(SHORT) == name].iloc[0]
        _label(a, row[col] / xs, row["capacity"], name.replace("-V4-Flash", "").replace("-M2.5", ""), dx, dy, size=6.3, color=INK2)
    handles = [plt.Line2D([], [], color=_cap_color(m), lw=1.6, marker="o", ms=4, mec="white") for m in CAP_MAIN]
    handles.append(plt.Line2D([], [], ls="", marker="o", ms=4, color=MUTED))
    fig.legend(handles, [SHORT[m] for m in CAP_MAIN] + ["other served models"], loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, 0.98), handlelength=1.4, columnspacing=1.2, handletextpad=0.4)
    _save(fig, "capacity")
    FACTS["capacity_rho"] = {"capacity_vs_kv": round(r(c["capacity"], c["kv_mb_per_ctx"]), 3),
                             "capacity_vs_peak_prefill": round(r(c["capacity"], c["peak"]), 3),
                             "cold_vs_active": round(r(c["cold_p50_s"], c["active_b"]), 3)}
    FACTS["capacity_table"] = c.round(3).to_dict("records")
    FACTS["capacity_summary"] = s.round(4).to_dict("records")
    FACTS["capacity_context_tokens"] = ctx
    FACTS["capacity_agents"] = {"kept": int(len(df)), "excluded_contended": int(raw["contended"].sum()), "calls": int(2 * len(raw))}


def fig_capacity_all() -> None:
    raw, df, s, c, ctx = _capacity_data()
    order = c.sort_values(["capacity", "cold_p50_s"], ascending=[False, True])["model"].tolist()
    fig, axes = plt.subplots(3, 3, figsize=(FULL, 4.2), sharex=True, sharey=True, gridspec_kw={"hspace": 0.5, "wspace": 0.1})
    for ax, m in zip(axes.flat, order):
        g = s[s["model"] == m].sort_values("load")
        ax.fill_between(g["load"], g["hit_lo"] * 100, g["hit_hi"] * 100, color=GRID, lw=0, zorder=1)
        ax.plot(g["load"], g["hit_rate"] * 100, color=INK, lw=1.4, zorder=3)
        ax.plot(g["load"], g["hit_rate"] * 100, "o", ms=3.2, color=INK, mec="white", mew=0.6, zorder=3)
        if g["errors"].sum():
            ax.plot(g["load"], g["err_rate"] * 100, color=RED, lw=1.1, zorder=2)
        cc = c[c["model"] == m].iloc[0]
        basis = "" if g["hit_basis"].iloc[0] == "reported" else " (hits from latency)"
        ax.set_title(f"{SHORT[m]}{basis}", fontsize=7.5)
        ax.text(0.04, 0.1, f"keeps {cc['capacity']}" + ("+" if cc["censored"] else ""), transform=ax.transAxes,
                fontsize=6.8, color=INK2)
    for ax in axes.flat:
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 4, 16, 64])
        ax.set_xticklabels(["1", "4", "16", "64"])
        ax.set_ylim(-5, 108)
        ax.set_yticks([0, 50, 100])
        ax.set_yticklabels(["0%", "50%", "100%"])
    for ax in axes[-1]:
        ax.set_xlabel("concurrent agents")
    fig.legend([plt.Line2D([], [], color=INK, lw=1.4, marker="o", ms=3.2), plt.Line2D([], [], color=RED, lw=1.1)],
               ["contexts still cached after 30 s (band: 95% interval)", "agents with a refused request (HTTP 429)"],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.965))
    _save(fig, "capacity_all")


# ---- E4: an agent's own long requests slow its short ones ---------------------------------------------

INTERF_DIRS = [RUNS / "interference" / "confirm_gemma", RUNS / "interference" / "confirm_deepseek"]


def _interf_load():
    rows, summ, ov, verdicts = [], [], [], {}
    for d in INTERF_DIRS:
        rows.append(pd.DataFrame([json.loads(l) for l in open(d / "requests.jsonl", encoding="utf-8")]))
        summ.append(pd.read_csv(d / "summary.csv"))
        ov.append(pd.read_csv(d / "overlap.csv"))
        verdicts[d.name] = json.loads((d / "verdicts.json").read_text())["verdicts"]
    return pd.concat(rows, ignore_index=True), pd.concat(summ, ignore_index=True), pd.concat(ov, ignore_index=True), verdicts


def fig_interference() -> None:
    rows, summ, ov, verdicts = _interf_load()
    dec = rows[rows["kind"] == "short"].copy()
    valid = dec["status"].eq("ok") & dec["valid_action_s"].notna()
    dec["ttva"] = np.where(valid, dec["valid_action_s"] - dec["arrival_s"], np.inf)
    fig, (ax, ax3) = plt.subplots(1, 2, figsize=(COL, 1.95), gridspec_kw={"wspace": 0.55, "width_ratios": [1.25, 1]})
    g = dec[dec["model"] == "google/gemma-4-31B-it"]
    style = {"short_only": ("alone", MUTED, 1.4), "fifo": ("FIFO", RED, 1.6), "defer_long": ("defer-long", INK, 1.6)}
    gs = summ[summ["model"] == "google/gemma-4-31B-it"].set_index("condition")
    for cond, (label, color, lw) in style.items():
        x = np.sort(g[g["condition"] == cond]["ttva"].to_numpy())
        y = np.arange(1, len(x) + 1) / len(x)
        ax.step(x, y, where="post", color=color, lw=lw)
        p95 = gs.loc[cond, "short_p95_ttva"]
        _dot(ax, p95, 0.95, color, size=3.8, z=4)
        k = list(style).index(cond)
        ax.plot([0.56], [0.36 - 0.12 * k], "o", ms=3.8, color=color, mec="white", mew=0.8, transform=ax.transAxes, clip_on=False)
        ax.text(0.61, 0.36 - 0.12 * k, f"{label}: {p95:.1f} s", transform=ax.transAxes, fontsize=6.3, color=INK2, va="center")
    ax.text(0.53, 0.48, "p95 time", transform=ax.transAxes, fontsize=6.3, color=INK2, va="center")
    ax.axhline(0.95, color=GRID, lw=0.8, zorder=0)
    ax.set_xscale("log")
    ax.set_xlim(0.4, 30)
    _log_ticks(ax, "x", [0.5, 1, 2, 5, 10, 20], lambda t: f"{t:g}")
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.5, 0.95])
    ax.set_yticklabels(["0%", "50%", "p95"])
    ax.set_xlabel("s to a valid move (log)")
    ax.set_title("a  gemma: tail 11× longer")
    for m, name in (("google/gemma-4-31B-it", "gemma-4-31B"), ("deepseek-ai/DeepSeek-V4-Flash", "DeepSeek-V4-Flash")):
        o = ov[(ov["model"] == m) & (ov["condition"] == "fifo")].sort_values("longs_prefilling")
        ax3.plot(o["longs_prefilling"], o["p50"], color=MODEL[name], lw=1.6)
        for x, y in zip(o["longs_prefilling"], o["p50"]):
            _dot(ax3, x, y, MODEL[name], size=4.3)
        y1 = o[o["longs_prefilling"] == 1]["p50"].iloc[0]
        _label(ax3, 1, y1, "DeepSeek" if "Deep" in name else "gemma", -5 if "Deep" in name else 5, 7 if "Deep" in name else -7,
               size=6.4, color=INK2)
    ax3.set_xticks([0, 1, 2])
    ax3.set_xticklabels(["0", "1", "2+"])
    ax3.set_xlim(-0.25, 2.25)
    ax3.set_ylim(0, 10.5)
    ax3.set_xlabel("long requests in prefill")
    ax3.set_ylabel("median s to first byte")
    ax3.set_title("b  inside the endpoint")
    ax3.grid(axis="x", visible=False)
    _save(fig, "interference")
    keep = ["model", "condition", "short_p50_ttva", "short_p95_ttva", "short_p95_queue_s", "useful_per_s",
            "long_p50_ttva", "long_p95_ttva", "short_invalid_rate", "legal_rate", "est_cost_usd"]
    FACTS["interference"] = summ[keep].replace([np.inf], "inf").round(4).to_dict("records")
    FACTS["overlap"] = ov.round(3).to_dict("records")
    FACTS["verdicts"] = verdicts


# ---- E5: decision models as served ---------------------------------------------------------------------

DECISION_RUNS = {"Intelif": RUNS / "scaleup" / "intelif_all", "Jev": RUNS / "scaleup" / "jev"}


def _decision_runs():
    out = {}
    for name, d in DECISION_RUNS.items():
        if (d / "steps.csv").exists() and (d / "episodes_scored.csv").exists():
            st = pd.read_csv(d / "steps.csv")
            if "regret_beam" in st:
                out[name] = (d, st, pd.read_csv(d / "episodes_scored.csv"))
    return out


def fig_decision() -> None:
    runs = _decision_runs()
    il_dir, st, ep = runs["Intelif"]
    seeds = sorted(ep["episode_seed"].unique())
    it6 = pd.read_csv(RUNS / "iter06" / "episodes.csv")
    it6 = it6[it6["episode_seed"].isin(seeds)]
    it6_steps = pd.read_csv(RUNS / "iter06" / "steps.csv")
    fig = plt.figure(figsize=(FULL, 2.0))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.25, 0.62, 0.8, 0.85], wspace=0.5)
    ax, ax2, ax3 = fig.add_subplot(gs[0]), fig.add_subplot(gs[2]), fig.add_subplot(gs[3])
    # (a) where a decision is computed sets its latency: measured rows solid, vendor/author figures hollow
    q = pd.read_csv(il_dir / "latency_quiet.csv")
    llm = it6_steps.groupby("arm")["latency_s"].median()
    rows = [("Intelif, 4-core CPU", q["latency_s"].min(), q["latency_s"].max(), q["latency_s"].median(), INK, False)]
    if "Jev" in runs:
        jst = runs["Jev"][1]
        rows.append(("Jev, TypeSafe API", jst["latency_s"].quantile(0.05), jst["latency_s"].quantile(0.95),
                     jst["latency_s"].median(), RED, False))
    rows += [("LLMs, serverless API", llm.min(), llm.max(), float(llm.median()), MODEL["gemma-4-31B"], False),
             ("Jev (vendor figure)", 0.07, 0.5, None, MUTED, True),
             ("Intelif, one GPU (reported)", 0.0165, 0.04, None, MUTED, True)]
    for i, (name, lo, hi, med, col, reported) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot([lo, hi], [y, y], color=col, lw=5 if not reported else 4, alpha=0.35 if not reported else 0.6,
                solid_capstyle="butt")
        if med is None:
            _label(ax, hi, y, f"{lo * 1000:.0f}–{hi * 1000:.0f} ms", 5, 0, size=6.4, color=INK2)
        if med is not None:
            _dot(ax, med, y, col, size=5.2, z=4)
            _label(ax, hi, y, f"{med:.2g} s" if med >= 1 else f"{med * 1000:.0f} ms" if med < 0.1 else f"{med:.2f} s",
                   5, 0, size=6.4, color=INK2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=6.6)
    ax.set_xscale("log")
    ax.set_xlim(0.01, 200)
    _log_ticks(ax, "x", [0.01, 0.1, 1, 10, 100], lambda t: f"{t:g}")
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("s per decision (log; bar: range, dot: median)")
    ax.set_title("a  Where it runs sets its speed")
    ax.grid(axis="y", visible=False)
    # (b) play on the same games
    per = it6.groupby("arm").apply(_pool).sort_values()
    pts = [(ARM_LABEL.get(a, a), v, _arm_color(a) or MUTED, "o") for a, v in per.items()]
    pts.append(("Intelif", _pool(ep), INK, "D"))
    if "Jev" in runs:
        pts.append(("Jev", _pool(runs["Jev"][2]), RED, "D"))
    pts.sort(key=lambda r: r[1])
    for i, (name, v, col, mk) in enumerate(pts):
        _dot(ax2, v, i, col, size=4.8 if mk == "o" else 4.6, marker=mk)
    ax2.set_yticks(range(len(pts)))
    ax2.set_yticklabels([p[0] for p in pts], fontsize=6.6)
    for lbl in ax2.get_yticklabels():
        if lbl.get_text() in ("Intelif", "Jev"):
            lbl.set_fontweight("bold")
    ax2.set_ylim(-0.7, len(pts) - 0.3)
    ax2.set_xlim(-0.04, 1.08)
    ax2.axvline(1.0, color=INK2, lw=0.7)
    ax2.set_xticks([0, 0.5, 1])
    ax2.set_xticklabels(["0", "0.5", "bot"])
    ax2.set_xlabel("score vs. beam-search bot")
    ax2.set_title("b  Same games: mid-pack")
    ax2.grid(axis="y", visible=False)
    # (c) confidence vs choosing the oracle's best move
    st = st.copy()
    st["pbin"] = pd.qcut(st["top_prob"], 4, duplicates="drop")
    g = st.groupby("pbin", observed=True).agg(p=("top_prob", "median"), regret=("regret_beam", "mean"),
                                               best=("top1_beam", "mean"), n=("turn", "size")).reset_index(drop=True)
    ax3.bar(range(len(g)), g["best"] * 100, color=[CONTEXT] * (len(g) - 1) + [INK], width=0.6)
    for i, v in enumerate(g["best"] * 100):
        ax3.text(i, v + 2, f"{v:.0f}%", ha="center", va="bottom", fontsize=6.6, color=INK2)
    ax3.set_xticks(range(len(g)))
    ax3.set_xticklabels([f"{x:.2f}" for x in g["p"]])
    ax3.set_ylim(0, 68)
    ax3.set_yticks([0, 20, 40, 60])
    ax3.set_yticklabels(["0%", "20%", "40%", "60%"])
    ax3.set_xlabel("p(choice), quartile median")
    ax3.set_ylabel("oracle's best move")
    ax3.set_title("c  Confidence is informative")
    ax3.grid(axis="x", visible=False)
    _save(fig, "decision")
    FACTS["decision_calibration"] = g.round(4).to_dict("records")
    FACTS["decision"] = {"intelif_quiet_p50_s": float(q["latency_s"].median()), "intelif_quiet_min_s": float(q["latency_s"].min()),
                         "intelif_quiet_max_s": float(q["latency_s"].max()),
                         "intelif_ms_per_token": float(1000 * (q["latency_s"] / q["input_tokens"]).median()),
                         "llm_api_median_s": float(llm.median()), "same_games_scores": {p[0]: round(p[1], 4) for p in pts},
                         "jev_present": "Jev" in runs}


# ---- calls and cost per experiment (appendix table) ----------------------------------------------------

EXPERIMENTS = {"E1": ["scaleup/ladder"], "E2": ["scaleup/memory"],
               "E3": ["scaleup/capacity", "scaleup/capacity_part2", "scaleup/capacity_part3"],
               "E4": ["interference/confirm_gemma", "interference/confirm_deepseek"]}


def facts_calls() -> None:
    from llm.tensormesh_client import load_pricing

    pricing = load_pricing(ROOT / "configs" / "pricing.yaml")
    out = {}
    for name, dirs in EXPERIMENTS.items():
        n = failed = 0
        stated = full = 0.0
        for d in dirs:
            f = RUNS / d / "calls.jsonl"
            if not f.exists():
                continue
            for line in open(f, encoding="utf-8"):
                r = json.loads(line)
                if r.get("event") != "call":
                    continue
                n += 1
                failed += 0 if r.get("ok") else 1
                pr = pricing.get(r["model"]) or {}
                pt, ct, ot = (int(r.get(k) or 0) for k in ("prompt_tokens", "cached_tokens", "completion_tokens"))
                pin, pout = float(pr.get("input") or 0), float(pr.get("output") or 0)
                stated += ((pt - ct) * pin + ot * pout) / 1e6
                full += ((pt if pr.get("cached") is None else pt - ct) * pin + ot * pout) / 1e6
        out[name] = {"calls": n, "failed": failed, "usd_stated": round(stated, 3), "usd_full": round(full, 3)}
    FACTS["calls_by_experiment"] = out
    spend = pd.read_csv(RUNS / "spend.csv")
    FACTS["spend_total_usd"] = round(float(spend["cost_usd"].sum()), 4)


def teaser() -> None:
    src = ROOT / "gallery" / "same_game_seed1002_piece60.png"
    if src.exists():
        OUT.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, OUT / "teaser.png")


def main() -> None:
    teaser()
    fig_ladder()
    fig_memory()
    fig_capacity()
    fig_capacity_all()
    fig_interference()
    fig_decision()
    facts_calls()
    (OUT / "facts.json").write_text(json.dumps(FACTS, indent=1, default=str))
    print("wrote", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
