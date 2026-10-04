"""figures.py - every figure in the paper, built from committed run data under runs/explore/.

    python paper/figures.py          # writes paper/figures/*.svg and paper/figures/facts.json

Each figure is drawn at its printed size (5.5 in wide, the ICLR text width) so text prints at the same point size in
every figure. Colors follow the entity (one color per model across the whole paper) and come from
a palette validated for color-vision deficiency; identity is always also given by a direct label.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "explore"
OUT = Path(__file__).resolve().parent / "figures"

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#a3a29d", "#e6e5e1"
MODEL = {"gemma-4-31B": "#2a78d6", "DeepSeek-V4-Flash": "#eb6834", "gpt-oss-20b": "#1baf7a", "gpt-oss-120b": "#4a3aa7"}
COND = {"alone": "#eda100", "fifo": "#e34948", "defer_long": "#4a3aa7"}
W = 5.5  # printed text width, inches
SHORT = {"openai/gpt-oss-20b": "gpt-oss-20b", "openai/gpt-oss-120b": "gpt-oss-120b", "google/gemma-4-31B-it": "gemma-4-31B",
         "deepseek-ai/DeepSeek-V4-Flash": "DeepSeek-V4-Flash", "MiniMaxAI/MiniMax-M2.5": "MiniMax-M2.5",
         "Qwen/Qwen3.8-27B-FP8": "Qwen3.8-27B", "Qwen/Qwen3.5-397B-A17B-FP8": "Qwen3.5-397B",
         "moonshotai/Kimi-K2.7-Code": "Kimi-K2.7", "lukealonso/GLM-5.2-NVFP4": "GLM-5.2"}

from matplotlib import font_manager  # noqa: E402

for _ttf in sorted((Path(__file__).resolve().parent / "fonts").glob("SourceSans3-*.ttf")):
    font_manager.fontManager.addfont(str(_ttf))  # same sans as the paper's headings and captions

plt.rcParams.update({
    "font.family": "Source Sans 3", "font.size": 8.5, "axes.titlesize": 9, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.labelsize": 8.5, "axes.labelcolor": INK2, "axes.edgecolor": GRID,
    "axes.linewidth": 0.8, "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "xtick.major.size": 0, "ytick.major.size": 0, "xtick.minor.size": 0, "ytick.minor.size": 0,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 8,
    "svg.fonttype": "path", "figure.dpi": 100, "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
    "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round"})

FACTS: dict = {}


def _save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.svg")
    preview = os.environ.get("FIG_PREVIEW")  # optional PNG previews for checking layout
    if preview:
        fig.savefig(Path(preview) / f"{name}.png", dpi=150)
    plt.close(fig)


def _dot(ax, x, y, color, size=7, **kw):
    ax.plot([x], [y], "o", ms=size, color=color, mec="white", mew=1.4, zorder=3, **kw)


def _short(arm: str) -> str:
    return arm.split("/")[0]


# ---- Figure: price and size do not buy decisions (iteration 6 model ladder) ----------------------

def fig_ladder() -> None:
    s = pd.read_csv(RUNS / "iter06" / "summary.csv")
    s["usd100"] = s["usd_per_100_pieces"]  # estimated $ per 100 placed pieces (= decisions)
    fig, ax = plt.subplots(figsize=(W, 3.0))
    ax.set_xscale("log")
    ax.axhline(1.0, color=INK2, lw=0.8, zorder=1)
    ax.text(0.0052, 1.0, "beam-search bot = 1.0", va="bottom", ha="left", fontsize=7.5, color=INK2)
    labels = {
        "gpt-oss-20b/low": ("gpt-oss-20b", (6, -11)), "gpt-oss-120b/low": ("gpt-oss-120b, low reasoning", (7, -3)),
        "gpt-oss-120b/medium": ("gpt-oss-120b, medium reasoning\n8.3 s per move", (7, -6)),
        "gemma-4-31B/direct": ("gemma-4-31B", (7, -3)), "DeepSeek-V4-Flash/direct": ("DeepSeek-V4-Flash", (7, -3)),
        "Qwen3.8-27B/direct": ("Qwen3.8-27B", (7, -3)), "Qwen3.5-397B/direct": ("Qwen3.5-397B (largest)", (7, -3)),
        "Kimi-K2.7/direct": ("Kimi-K2.7", (7, -3)), "GLM-5.2/direct": ("GLM-5.2 (priciest)", (-8, 6))}
    for _, r in s.iterrows():
        fam = _short(r["arm"]).replace("-direct", "")
        color = next((c for k, c in MODEL.items() if fam.startswith(k)), MUTED)
        _dot(ax, r["usd100"], r["norm_score"], color, size=7.5)
        text, off = labels[r["arm"]]
        ax.annotate(text, (r["usd100"], r["norm_score"]), xytext=off, textcoords="offset points",
                    fontsize=7.5, color=INK, ha="right" if off[0] < 0 else "left", va="center")
    ax.set_xlim(0.0045, 0.32)
    ax.set_ylim(-0.04, 1.08)
    ax.set_xticks([0.005, 0.01, 0.02, 0.05, 0.1, 0.2])
    ax.set_xticklabels(["$0.005", "$0.01", "$0.02", "$0.05", "$0.10", "$0.20"])
    ax.set_xlabel("estimated cost per 100 decisions (log scale)")
    ax.set_ylabel("game score relative to beam search")
    _save(fig, "ladder")
    FACTS["ladder"] = s[["arm", "usd100", "norm_score", "norm_lines", "regret_beam", "latency_p50", "pieces"]].round(4).to_dict("records")


# ---- Figure: the price card decides which memory is cheap (iteration 2) --------------------------

def fig_memory() -> None:
    s = pd.read_csv(RUNS / "iter02" / "summary.csv")
    s["model"] = s["arm"].str.split("/").str[0].map({"oss20b": "gpt-oss-20b", "oss120b": "gpt-oss-120b",
                                                     "dsv4flash": "DeepSeek-V4-Flash"})
    s["policy"] = s["arm"].str.split("/").str[1]
    order = ["stateless", "append", "window8"]
    names = {"stateless": "no history", "append": "append all turns", "window8": "last 8 turns"}
    note = {"gpt-oss-20b": "cached input $0", "gpt-oss-120b": "cached input $0", "DeepSeek-V4-Flash": "no cached price"}
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.65), gridspec_kw={"wspace": 0.55})
    for ax, col, title, ylab, log in (
            (axes[0], "usd_per_100_pieces", "a  Cost: the ranking flips with the price card", "$ per 100 decisions (log)", True),
            (axes[1], "regret_beam", "b  Quality: more history plays worse", "regret per decision (lower is better)", False)):
        if log:
            ax.set_yscale("log")
        for m in ["gpt-oss-20b", "gpt-oss-120b", "DeepSeek-V4-Flash"]:
            g = s[s["model"] == m].set_index("policy").loc[order]
            ax.plot(range(3), g[col], color=MODEL[m], lw=2)
            for i, v in enumerate(g[col]):
                _dot(ax, i, v, MODEL[m], size=7)
            if not log:
                ax.annotate(m, (2, g[col].iloc[-1]), xytext=(9, 0), textcoords="offset points", va="center",
                            fontsize=7.5, color=INK)
        ax.set_xticks(range(3))
        ax.set_xticklabels([names[p] for p in order])
        ax.set_xlim(-0.3, 2.3)
        ax.set_title(title, pad=8)
        ax.set_ylabel(ylab)
        ax.grid(axis="x", visible=False)
    axes[0].set_yticks([0.005, 0.01, 0.02, 0.05, 0.1, 0.2])
    axes[0].set_yticklabels(["$0.005", "$0.01", "$0.02", "$0.05", "$0.10", "$0.20"])
    axes[0].set_ylim(0.0045, 0.25)
    axes[1].set_ylim(0, 4.8)
    handles = [plt.Line2D([], [], color=MODEL[m], lw=2, marker="o", ms=5.5, mec=MODEL[m]) for m in note]
    fig.legend(handles, [f"{m} ({note[m]})" for m in note], loc="lower center", ncol=3, bbox_to_anchor=(0.5, 1.0),
               handlelength=1.6, columnspacing=1.6, fontsize=7.5)
    _save(fig, "memory")
    FACTS["memory"] = s[["arm", "usd_per_100_pieces", "regret_beam", "cached_frac", "prompt_tok"]].round(4).to_dict("records")


# ---- Figure: caching works everywhere, but billing and capacity differ (probe + iteration 4) -----

def fig_cache() -> None:
    c = pd.read_csv(RUNS / "probe" / "cache_16k.csv")
    c = c[c["stream"] == True]  # noqa: E712
    t = c.groupby(["model", "call_idx"])["ttft_s"].mean().unstack()
    rep = c[c["call_idx"] == 1].groupby("model")["cached_tokens"].mean()
    price = pd.read_csv(RUNS / "probe" / "models.csv") if (RUNS / "probe" / "models.csv").exists() else None
    t = t.sort_values(0)
    fig = plt.figure(figsize=(W, 2.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 1, 1], wspace=0.62)
    ax = fig.add_subplot(gs[0])
    ax.set_xscale("log")
    names = [m.split("/")[-1].replace("-it", "").replace("-FP8", "").replace("-NVFP4", "*").replace("-Code", "")
             .replace("-A17B", "") for m in t.index]
    for i, (m, row) in enumerate(t.iterrows()):
        ax.plot([row[1], row[0]], [i, i], color=MUTED, lw=2, zorder=1)
        ax.plot([row[0]], [i], "o", ms=7, mfc="white", mec=INK2, mew=1.4, zorder=2)
        ax.plot([row[1]], [i], "o", ms=7, color=INK, mec="white", mew=1.4, zorder=3)
    ax.set_yticks(range(len(t)))
    ax.set_yticklabels(names)
    ax.set_ylim(-0.6, len(t) - 0.4)
    ax.grid(axis="y", visible=False)
    ax.set_xticks([0.5, 1, 2, 5, 10])
    ax.set_xticklabels(["0.5", "1", "2", "5", "10"])
    ax.set_xlabel("time to first token, s (log)")
    ax.set_title("a  16k-token prefix, cold vs repeat", pad=8)
    ax.plot([], [], "o", ms=6, mfc="white", mec=INK2, mew=1.4, label="cold")
    ax.plot([], [], "o", ms=6, color=INK, label="repeat")
    ax.legend(loc="lower right", handletextpad=0.2, borderaxespad=0.2)

    cap = pd.read_csv(RUNS / "iter04" / "summary_capacity.csv")
    lab = {"deepseek-ai/DeepSeek-V4-Flash": "DeepSeek-V4-Flash", "openai/gpt-oss-20b": "gpt-oss-20b"}
    ax2, ax3 = fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
    for m, g in cap.groupby("model"):
        name = lab[m]
        ax2.plot(g["contexts"], g["hit_rate"] * 100, color=MODEL[name], lw=2)
        ax3.plot(g["contexts"], g["warm_p50"], color=MODEL[name], lw=2)
        for x, y, z in zip(g["contexts"], g["hit_rate"] * 100, g["warm_p50"]):
            _dot(ax2, x, y, MODEL[name], size=6.5)
            _dot(ax3, x, z, MODEL[name], size=6.5)
    for ax_ in (ax2, ax3):
        ax_.set_xscale("log", base=2)
        ax_.set_xticks([1, 4, 8, 16])
        ax_.set_xticklabels(["1", "4", "8", "16"])
        ax_.set_xlabel("concurrent 15k-token agents")
    ax2.set_ylim(-8, 112)
    ax2.set_yticks([0, 50, 100])
    ax2.set_yticklabels(["0%", "50%", "100%"])
    ax2.set_title("b  Still cached after 30 s", pad=8)
    ax2.text(16, 92, "gpt-oss-20b", ha="right", va="top", fontsize=7.5, color=INK)
    ax2.text(16, 25, "DeepSeek-\nV4-Flash", ha="right", va="bottom", fontsize=7.5, color=INK)
    ax3.set_title("c  Time to re-serve", pad=8)
    ax3.set_ylabel("median latency, s")
    ax3.set_ylim(0, 26)
    ax3.text(1.1, 17, "DeepSeek-\nV4-Flash", ha="left", va="bottom", fontsize=7.5, color=INK)
    ax3.text(16, 4.3, "gpt-oss-20b", ha="right", va="bottom", fontsize=7.5, color=INK)
    _save(fig, "cache")
    FACTS["cache_16k"] = {m: {"cold_s": round(float(r[0]), 3), "repeat_s": round(float(r[1]), 3),
                              "reported_cached": float(rep.get(m, np.nan))} for m, r in t.iterrows()}
    FACTS["capacity"] = cap.round(3).to_dict("records")


# ---- Figure: your own long requests slow your short ones (iteration 8 live pilot) ----------------

INTERF_DIRS = [Path(x) for x in os.environ.get("INTERF_DIRS", "").split(",") if x] or \
    [RUNS / "interference" / "confirm_gemma", RUNS / "interference" / "confirm_deepseek"]


def _interf_load(dirs):
    rows, summ, ov, verdicts = [], [], [], {}
    for d in dirs:
        rows.append(pd.DataFrame([json.loads(l) for l in open(d / "requests.jsonl", encoding="utf-8")]))
        summ.append(pd.read_csv(d / "summary.csv"))
        ov.append(pd.read_csv(d / "overlap.csv"))
        verdicts[d.name] = json.loads((d / "verdicts.json").read_text())["verdicts"]
    return pd.concat(rows, ignore_index=True), pd.concat(summ, ignore_index=True), pd.concat(ov, ignore_index=True), verdicts


def fig_interference() -> None:
    rows, summ, ov, verdicts = _interf_load(INTERF_DIRS)
    dec = rows[rows["kind"] == "short"].copy()
    valid = dec["status"].eq("ok") & dec["valid_action_s"].notna()
    dec["ttva"] = np.where(valid, dec["valid_action_s"] - dec["arrival_s"], np.inf)
    fig = plt.figure(figsize=(W, 2.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.15, 0.95], wspace=0.62)
    ax = fig.add_subplot(gs[0])
    g = dec[dec["model"] == "google/gemma-4-31B-it"]
    names = {"short_only": ("alone", COND["alone"]), "fifo": ("FIFO", COND["fifo"]),
             "defer_long": ("defer-long", COND["defer_long"])}
    for cond, (label, color) in names.items():
        x = np.sort(g[g["condition"] == cond]["ttva"].to_numpy())
        y = np.arange(1, len(x) + 1) / len(x)
        ax.step(x, y, where="post", color=color, lw=2, label=label)
    ax.axhline(0.95, color=INK2, lw=0.6)
    ax.text(0.42, 0.955, "p95", fontsize=7, color=INK2, va="bottom")
    ax.set_xscale("log")
    ax.set_xlim(0.4, 25)
    ax.set_xticks([0.5, 1, 2, 5, 10, 20])
    ax.set_xticklabels(["0.5", "1", "2", "5", "10", "20"])
    ax.set_ylim(0, 1.02)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["0%", "50%", "100%"])
    ax.set_xlabel("time to a valid move, s (log)")
    ax.set_title("a  Time to a move (gemma)", pad=8)
    ax.legend(loc="lower right", handlelength=1.4, borderaxespad=0.1)

    ax2 = fig.add_subplot(gs[1])
    conds = [("short_only", "alone"), ("fifo", "FIFO"), ("fifo_prio", "FIFO\n+prio"), ("defer_long", "defer\nlong")]
    for m, name in (("google/gemma-4-31B-it", "gemma-4-31B"), ("deepseek-ai/DeepSeek-V4-Flash", "DeepSeek-V4-Flash")):
        vals = [float(summ[(summ["model"] == m) & (summ["condition"] == c)]["useful_per_s"].iloc[0]) for c, _ in conds]
        ax2.plot(range(4), vals, color=MODEL[name], lw=2)
        for i, v in enumerate(vals):
            _dot(ax2, i, v, MODEL[name], size=6.5)
    ax2.set_xticks(range(4))
    ax2.set_xticklabels([n for _, n in conds], fontsize=7.5)
    ax2.set_xlim(-0.35, 3.35)
    ax2.set_ylim(0, 1.2)
    ax2.set_ylabel("valid moves within 2.5 s, per s")
    ax2.set_title("b  Useful decisions", pad=8)
    ax2.text(3.3, 0.74, "gemma-4-31B", ha="right", fontsize=7.5, color=INK)
    ax2.text(3.3, 0.06, "DeepSeek-V4-Flash", ha="right", fontsize=7.5, color=INK)
    ax2.grid(axis="x", visible=False)

    ax3 = fig.add_subplot(gs[2])
    for m, name in (("google/gemma-4-31B-it", "gemma-4-31B"), ("deepseek-ai/DeepSeek-V4-Flash", "DeepSeek-V4-Flash")):
        o = ov[(ov["model"] == m) & (ov["condition"] == "fifo")].sort_values("longs_prefilling")
        ax3.plot(o["longs_prefilling"], o["p50"], color=MODEL[name], lw=2)
        for x, y in zip(o["longs_prefilling"], o["p50"]):
            _dot(ax3, x, y, MODEL[name], size=6.5)
    ax3.set_xticks([0, 1, 2])
    ax3.set_xticklabels(["0", "1", "2+"])
    ax3.set_xlim(-0.25, 2.25)
    ax3.set_ylim(0, 10)
    ax3.set_xlabel("long requests still prefilling")
    ax3.set_ylabel("median s to first byte")
    ax3.set_title("c  Endpoint delay", pad=8)
    ax3.text(2.2, 9.0, "DeepSeek-V4-Flash", ha="right", fontsize=7.5, color=INK)
    ax3.text(2.2, 1.0, "gemma-4-31B", ha="right", fontsize=7.5, color=INK)
    ax3.grid(axis="x", visible=False)
    _save(fig, "interference")
    keep = ["model", "condition", "short_p50_ttva", "short_p95_ttva", "short_p95_queue_s", "useful_per_s",
            "long_p50_ttva", "long_p95_ttva", "short_invalid_rate", "legal_rate", "est_cost_usd"]
    FACTS["interference"] = summ[keep].replace([np.inf], "inf").round(4).to_dict("records")
    FACTS["overlap"] = ov.round(3).to_dict("records")
    FACTS["verdicts"] = verdicts


# ---- Figure: E1 model ladder at scale, with the self-hosted decision model ----------------------

LADDER_DIR = Path(os.environ.get("LADDER_DIR", RUNS / "scaleup" / "ladder"))
INTELIF_DIR = Path(os.environ.get("INTELIF_DIR", RUNS / "scaleup" / "intelif"))
ARM_LABEL = {"gpt-oss-20b/low": "gpt-oss-20b", "gpt-oss-120b/low": "gpt-oss-120b", "gpt-oss-120b/medium": "gpt-oss-120b, medium",
             "gemma-4-31B/direct": "gemma-4-31B", "DeepSeek-V4-Flash/direct": "DeepSeek-V4-Flash",
             "Qwen3.8-27B/direct": "Qwen3.8-27B", "Qwen3.5-397B/direct": "Qwen3.5-397B", "Kimi-K2.7/direct": "Kimi-K2.7",
             "GLM-5.2/direct": "GLM-5.2"}
LADDER_OFF = {}  # per-arm label offsets in points, tuned after viewing the data


def _arm_color(arm: str) -> str:
    fam = _short(arm)
    return next((c for k, c in MODEL.items() if fam == k), MUTED)


def fig_ladder2() -> None:
    s = pd.read_csv(LADDER_DIR / "summary_scaleup.csv")
    an = json.loads((LADDER_DIR / "analysis.json").read_text())
    il = json.loads((INTELIF_DIR / "analysis.json").read_text()) if (INTELIF_DIR / "analysis.json").exists() else None
    fig = plt.figure(figsize=(W, 2.7))
    gs = fig.add_gridspec(1, 2, wspace=0.32)
    ax, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1], sharey=None)
    for a in (ax, ax2):
        a.set_xscale("log")
        a.axhline(1.0, color=INK2, lw=0.7, zorder=1)
    for _, r in s.iterrows():
        col = _arm_color(r["arm"])
        for a, x in ((ax, r["usd_per_100_stated"]), (ax2, r["latency_p50_s"])):
            a.plot([x, x], [r["norm_score_lo"], r["norm_score_hi"]], color=col, lw=1.2, alpha=0.55, zorder=2)
            _dot(a, x, r["norm_score"], col, size=6)
        dx, dy = LADDER_OFF.get(r["arm"], (5, 0))
        ax.annotate(ARM_LABEL[r["arm"]], (r["usd_per_100_stated"], r["norm_score"]), xytext=(dx, dy), textcoords="offset points",
                    fontsize=6.8, color=INK, ha="left" if dx >= 0 else "right", va="center")
    if il is not None:
        ax2.plot([il["latency_p50_s"]], [il["norm_score"]], "D", ms=6, color=INK, mec="white", mew=1.2, zorder=4)
        if "norm_score_lo" in il:
            ax2.plot([il["latency_p50_s"]] * 2, [il["norm_score_lo"], il["norm_score_hi"]], color=INK, lw=1.2, alpha=0.55)
        ax2.annotate("Intelif decision model\n(self-hosted, 4-core CPU)", (il["latency_p50_s"], il["norm_score"]), xytext=(-6, 0),
                     textcoords="offset points", fontsize=6.8, ha="right", va="center", color=INK)
    from matplotlib.ticker import NullFormatter
    for a, ticks, fmt in ((ax, [0.005, 0.01, 0.02, 0.05, 0.1, 0.2], lambda t: f"${t:g}"),
                          (ax2, [0.5, 1, 2, 5, 10, 20, 50], lambda t: f"{t:g}")):
        lo, hi = a.get_xlim()
        tk = [t for t in ticks if lo <= t <= hi]
        a.set_xticks(tk)
        a.set_xticklabels([fmt(t) for t in tk])
        a.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("$ per 100 decisions (stated pricing, log)")
    ax.set_ylabel("score relative to beam-search bot")
    ax.set_title("a  Quality against cost", pad=7)
    ax2.set_xlabel("median seconds per decision (log)")
    ax2.set_title("b  Quality against time per move", pad=7)
    sp = an["spearman_vs_score"]
    ax.text(0.98, 0.03, f"Spearman vs price {sp['price_stated']['rho']:+.2f}", transform=ax.transAxes, ha="right",
            va="bottom", fontsize=6.8, color=INK2)
    _save(fig, "ladder")
    FACTS["ladder2"] = {"summary": s.round(4).to_dict("records"), "analysis": an, "intelif": il}



# ---- Figure: E2 memory policy at scale --------------------------------------------------------------

MEMORY_DIR = Path(os.environ.get("MEMORY_DIR", RUNS / "scaleup" / "memory"))
MEM_NAMES = {"oss20b": "gpt-oss-20b", "oss120b": "gpt-oss-120b", "dsv4flash": "DeepSeek-V4-Flash"}


def fig_memory2() -> None:
    s = pd.read_csv(MEMORY_DIR / "summary_scaleup.csv")
    s["model"] = s["model"].map(lambda m: MEM_NAMES.get(m, m))
    an = json.loads((MEMORY_DIR / "analysis.json").read_text())
    order = ["stateless", "append", "window8"]
    names = {"stateless": "none", "append": "append", "window8": "last 8"}
    models = [m for m in ["gpt-oss-20b", "gpt-oss-120b", "DeepSeek-V4-Flash", "Qwen3.8-27B"] if m in set(s["model"])]
    color = {**MODEL, "Qwen3.8-27B": "#eda100"}
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.45), gridspec_kw={"wspace": 0.62})
    panels = ((axes[0], "uncached_per_decision", "a  Uncached input", "tokens per decision (log)", True),
              (axes[1], "usd_per_100_stated", "b  Cost", "$ per 100 decisions (log)", True),
              (axes[2], "regret", "c  Regret", "oracle regret per decision", False))
    for ax, col, title, ylab, log in panels:
        if log:
            ax.set_yscale("log")
        for m in models:
            g = s[s["model"] == m].set_index("policy").loc[order]
            ax.plot(range(3), g[col], color=color[m], lw=1.8)
            for i, v in enumerate(g[col]):
                _dot(ax, i, v, color[m], size=5.5)
            if col == "usd_per_100_stated":
                for i, (v, u) in enumerate(zip(g[col], g["usd_per_100_full"])):
                    if u > v * 1.05:
                        ax.plot([i], [u], "o", ms=4.5, mfc="white", mec=color[m], mew=1.1, zorder=3)
        ax.set_xticks(range(3))
        ax.set_xticklabels([names[p] for p in order])
        ax.set_xlim(-0.3, 2.3)
        ax.set_title(title, pad=7)
        ax.set_ylabel(ylab)
        ax.set_xlabel("history sent")
        ax.grid(axis="x", visible=False)
    from matplotlib.ticker import NullFormatter
    for ax, ticks, fmt in ((axes[0], [100, 300, 1000, 3000, 10000, 30000], lambda t: f"{t / 1000:g}k" if t >= 1000 else f"{t:g}"),
                           (axes[1], [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5], lambda t: f"${t:g}")):
        lo, hi = ax.get_ylim()
        tk = [t for t in ticks if lo <= t <= hi]
        ax.set_yticks(tk)
        ax.set_yticklabels([fmt(t) for t in tk])
        ax.yaxis.set_minor_formatter(NullFormatter())
    handles = [plt.Line2D([], [], color=color[m], lw=1.8, marker="o", ms=5) for m in models]
    handles.append(plt.Line2D([], [], ls="", marker="o", ms=4.5, mfc="white", mec=INK2, mew=1.1))
    fig.legend(handles, models + ["cached input at full price"], loc="lower center", ncol=len(handles),
               bbox_to_anchor=(0.5, 0.97), handlelength=1.3, columnspacing=0.9, handletextpad=0.4, fontsize=6.8)
    _save(fig, "memory")
    FACTS["memory2"] = {"summary": s.round(4).to_dict("records"), "paired": an["paired"]}



# ---- Figures: how many agent contexts a deployment keeps warm (E3 capacity sweep) ---------------

CAP_DIRS = [d for d in (RUNS / "scaleup" / f"capacity{x}" for x in ("", "_part2", "_part3")) if (d / "rows.jsonl").exists()]


def _capacity_data():
    import sys
    sys.path.insert(0, str(ROOT))
    from llm.capacity_analysis import capacity, load, summarize
    from llm.model_arch import PARAMS_B, kv_bytes

    raw = load(CAP_DIRS)
    df = raw[~raw["contended"]]
    s = summarize(df)
    s["err_rate"] = s["errors"] / s["agents"]
    c = capacity(s)
    ctx = float(df["warm_prompt_tokens"].median())
    c["kv_mb_per_ctx"] = [kv_bytes(m, int(ctx)) / 2**20 for m in c["model"]]
    c["active_b"] = [PARAMS_B[m][1] for m in c["model"]]
    return raw, df, s, c, ctx


def fig_capacity_all() -> None:
    """Appendix: every model, warm-reuse rate and refusals against concurrent agents."""
    raw, df, s, c, ctx = _capacity_data()
    order = c.sort_values(["capacity", "cold_p50_s"], ascending=[False, True])["model"].tolist()
    fig, axes = plt.subplots(3, 3, figsize=(W, 4.6), sharex=True, sharey=True, gridspec_kw={"hspace": 0.55, "wspace": 0.12})
    for ax, m in zip(axes.flat, order):
        g = s[s["model"] == m].sort_values("load")
        ax.fill_between(g["load"], g["hit_lo"] * 100, g["hit_hi"] * 100, color=GRID, lw=0, zorder=1)
        ax.plot(g["load"], g["hit_rate"] * 100, color=INK, lw=1.6, zorder=3)
        ax.plot(g["load"], g["hit_rate"] * 100, "o", ms=3.5, color=INK, zorder=3)
        if g["errors"].sum():
            ax.plot(g["load"], g["err_rate"] * 100, color=COND["fifo"], lw=1.2, ls=(0, (3, 2)), zorder=2)
        cc = c[c["model"] == m].iloc[0]
        basis = "" if g["hit_basis"].iloc[0] == "reported" else " (hits from latency)"
        ax.set_title(f"{SHORT[m]}{basis}", fontsize=8, pad=4)
        cap = f"keeps {cc['capacity']}" + ("+" if cc["censored"] else "")
        ax.text(0.04, 0.08, cap, transform=ax.transAxes, ha="left", va="bottom", fontsize=7, color=INK2)
    for ax in axes.flat[len(order):]:
        ax.set_visible(False)
    for ax in axes.flat:
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 4, 16, 64])
        ax.set_xticklabels(["1", "4", "16", "64"])
        ax.set_ylim(-5, 108)
        ax.set_yticks([0, 50, 100])
        ax.set_yticklabels(["0%", "50%", "100%"])
    for ax in axes[-1]:
        ax.set_xlabel("concurrent agents (N)")
    fig.legend([plt.Line2D([], [], color=INK, lw=1.6, marker="o", ms=3.5),
                plt.Line2D([], [], color=COND["fifo"], lw=1.2, ls=(0, (3, 2)))],
               ["contexts still cached after 30 s idle (95% CI band)", "requests refused (429)"],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.9), handlelength=2.2)
    _save(fig, "capacity_all")
    FACTS["capacity_summary"] = s.round(4).to_dict("records")
    FACTS["capacity_table"] = c.round(3).to_dict("records")
    FACTS["capacity_context_tokens"] = ctx
    FACTS["capacity_agents"] = {"kept": int(len(df)), "excluded_contended": int(raw["contended"].sum()),
                                "calls_total": int(2 * len(raw))}



CAP_MAIN = ["openai/gpt-oss-20b", "MiniMaxAI/MiniMax-M2.5", "google/gemma-4-31B-it", "deepseek-ai/DeepSeek-V4-Flash"]  # chosen after viewing all
CAP_COLOR = {"gpt-oss-20b": MODEL["gpt-oss-20b"], "MiniMax-M2.5": "#c2408f", "gemma-4-31B": MODEL["gemma-4-31B"],
             "DeepSeek-V4-Flash": MODEL["DeepSeek-V4-Flash"]}


def fig_capacity() -> None:
    """Main text: the models whose curves show the pattern clearly, then capacity against KV size and against
    aggregate prefill throughput for all nine models."""
    raw, df, s, c, ctx = _capacity_data()
    c["peak"] = c["model"].map(s.groupby("model")["prefill_tok_per_s"].max())
    fig, axes = plt.subplots(2, 2, figsize=(W, 4.1), gridspec_kw={"wspace": 0.42, "hspace": 0.72})
    ax, ax2, ax3, ax4 = axes.flat
    for m in CAP_MAIN:
        g = s[s["model"] == m].sort_values("load")
        col = CAP_COLOR[SHORT[m]]
        ax.plot(g["load"], g["hit_rate"] * 100, color=col, lw=1.8)
        ax2.plot(g["load"], g["prefill_tok_per_s"] / 1000, color=col, lw=1.8)
        for x, y, z in zip(g["load"], g["hit_rate"] * 100, g["prefill_tok_per_s"] / 1000):
            _dot(ax, x, y, col, size=5)
            _dot(ax2, x, z, col, size=5)
    for a in (ax, ax2):
        a.set_xscale("log", base=2)
        a.set_xticks([1, 2, 4, 8, 16, 32, 64])
        a.set_xticklabels(["1", "2", "4", "8", "16", "32", "64"])
        a.set_xlabel("concurrent agents (N)")
    ax.set_ylim(-6, 112)
    ax.set_yticks([0, 50, 100])
    ax.set_yticklabels(["0%", "50%", "100%"])
    ax.set_ylabel("contexts still cached")
    ax.set_title("a  Warm after 30 s idle", pad=7)
    ax2.set_title("b  Aggregate prefill throughput", pad=7)
    ax2.set_ylabel("thousand tokens / s")
    ax2.set_xlim(0.8, 64 * 1.2)
    ax2.set_ylim(0, 34)
    handles = [plt.Line2D([], [], color=CAP_COLOR[SHORT[m]], lw=1.8, marker="o", ms=4.5) for m in CAP_MAIN]
    fig.legend(handles, [SHORT[m] for m in CAP_MAIN], loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.965),
               handlelength=1.6, columnspacing=1.4, fontsize=7.2)
    r = lambda x, y: x.rank().corr(y.rank())
    for a, col, xlabel, title, log in ((ax3, "kv_mb_per_ctx", f"est. KV cache per {ctx / 1000:.1f}k-token context, MB",
                                        "c  Capacity vs KV size", True),
                                       (ax4, "peak", "peak aggregate prefill, thousand tokens / s",
                                        "d  Capacity vs throughput", False)):
        for _, row in c.iterrows():
            name = SHORT[row["model"]]
            colr = CAP_COLOR.get(name, MUTED)
            x = row[col] / (1000 if col == "peak" else 1)
            _dot(a, x, row["capacity"], colr, size=5.5)
            if row["censored"]:
                a.annotate("", xy=(x, row["capacity"] * 1.55), xytext=(x, row["capacity"] * 1.08),
                           arrowprops=dict(arrowstyle="-|>", color=colr, lw=0.9, mutation_scale=6))
        a.set_yscale("log", base=2)
        a.set_yticks([4, 8, 16, 32, 64])
        a.set_yticklabels(["4", "8", "16", "32", "64"])
        a.set_ylim(2.8, 120)
        a.set_ylabel("contexts kept warm")
        a.set_xlabel(xlabel)
        a.set_title(title, pad=7)
        rho = r(c["capacity"], c[col])
        tx = (0.97, "right", 0.05, "bottom") if log else (0.03, "left", 0.95, "top")
        a.text(tx[0], tx[2], f"Spearman {rho:+.2f}, 9 models", transform=a.transAxes, fontsize=6.8, color=INK2, va=tx[3],
               ha=tx[1])
        if log:
            a.set_xscale("log")
            a.set_xticks([20, 50, 100, 200, 500, 1000])
            a.set_xticklabels(["20", "50", "100", "200", "500", "1000"])
            from matplotlib.ticker import NullFormatter
            a.xaxis.set_minor_formatter(NullFormatter())
    ax4.set_xscale("log")
    ax4.set_xticks([2, 5, 10, 20, 30])
    ax4.set_xticklabels(["2", "5", "10", "20", "30"])
    from matplotlib.ticker import NullFormatter
    ax4.xaxis.set_minor_formatter(NullFormatter())
    ax4.set_xlim(1.4, 45)
    lab = {"DeepSeek-V4-Flash": (5, 0, "left"), "GLM-5.2": (0, 8, "center"), "MiniMax-M2.5": (5, 0, "left"),
           "gpt-oss-20b": (0, -9, "center"), "gpt-oss-120b": (-5, 0, "right"), "Qwen3.5-397B": (-5, 0, "right")}
    for _, row in c.iterrows():
        name = SHORT[row["model"]]
        if name in lab:
            dx, dy, ha = lab[name]
            ax4.annotate(name, (row["peak"] / 1000, row["capacity"]), xytext=(dx, dy), textcoords="offset points",
                         fontsize=6.3, color=INK2, ha=ha, va="center")
    trio = c[c["model"].map(SHORT).isin(["Kimi-K2.7", "gemma-4-31B", "Qwen3.8-27B"])]
    ax4.annotate("Kimi, gemma,\nQwen3.8", (trio["peak"].min() / 1000, 16), xytext=(-5, 0), textcoords="offset points",
                 fontsize=6.3, color=INK2, ha="right", va="center")
    for name, (dx, dy, ha) in {"DeepSeek-V4-Flash": (5, 0, "left"), "MiniMax-M2.5": (0, 9, "center")}.items():
        row = c[c["model"].map(SHORT) == name].iloc[0]
        ax3.annotate(name, (row["kv_mb_per_ctx"], row["capacity"]), xytext=(dx, dy), textcoords="offset points",
                     fontsize=6.3, color=INK2, ha=ha, va="center")
    _save(fig, "capacity")
    FACTS["capacity_rho"] = {"capacity_vs_kv": round(r(c["capacity"], c["kv_mb_per_ctx"]), 3),
                             "capacity_vs_peak_prefill": round(r(c["capacity"], c["peak"]), 3),
                             "cold_vs_active": round(r(c["cold_p50_s"], c["active_b"]), 3)}
    FACTS["capacity_peak"] = c[["model", "capacity", "censored", "peak", "kv_mb_per_ctx", "active_b", "cold_p50_s"]].round(2).to_dict("records")



def main() -> None:
    fig_ladder()
    fig_memory()
    fig_cache()
    fig_interference()
    spend = pd.read_csv(RUNS / "spend.csv")
    FACTS["spend_total_usd"] = round(float(spend["cost_usd"].sum()), 4)
    FACTS["spend_by_iteration"] = spend.groupby("iteration")["cost_usd"].sum().round(4).to_dict()
    (OUT / "facts.json").write_text(json.dumps(FACTS, indent=1, default=str))
    print("wrote", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
