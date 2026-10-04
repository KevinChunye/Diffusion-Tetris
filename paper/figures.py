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


# ---- Figure: your own long requests slow your short ones (E4 confirmatory run) ------------------

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
    ax3.text(1.55, 1.4, "gemma-4-31B", ha="center", fontsize=7.5, color=INK)
    ax3.grid(axis="x", visible=False)
    _save(fig, "interference")
    keep = ["model", "condition", "short_p50_ttva", "short_p95_ttva", "short_p95_queue_s", "useful_per_s",
            "long_p50_ttva", "long_p95_ttva", "short_invalid_rate", "legal_rate", "est_cost_usd"]
    FACTS["interference"] = summ[keep].replace([np.inf], "inf").round(4).to_dict("records")
    FACTS["overlap"] = ov.round(3).to_dict("records")
    FACTS["verdicts"] = verdicts


# ---- Figure: E1 model ladder at scale --------------------------------------------------------------

LADDER_DIR = Path(os.environ.get("LADDER_DIR", RUNS / "scaleup" / "ladder"))
INTELIF_DIR = Path(os.environ.get("INTELIF_DIR", RUNS / "scaleup" / "intelif_all"))
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
    fig = plt.figure(figsize=(W, 2.7))
    gs = fig.add_gridspec(1, 2, wspace=0.3)
    ax, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    for a in (ax, ax2):
        a.set_xscale("log")
        a.axhline(1.0, color=INK2, lw=0.7, zorder=1)
        a.set_ylim(-0.03, 1.08)
    off_a = {"gpt-oss-120b/medium": (0, 11), "GLM-5.2/direct": (-6, 0), "Qwen3.5-397B/direct": (5, -4), "Kimi-K2.7/direct": (5, 3)}
    off_b = {"gpt-oss-120b/low": (5, 5), "GLM-5.2/direct": (-6, 0), "Kimi-K2.7/direct": (-6, 0), "DeepSeek-V4-Flash/direct": (-6, 0),
             "Qwen3.5-397B/direct": (5, 4), "Qwen3.8-27B/direct": (5, -4), "gemma-4-31B/direct": (5, 3)}
    for _, r in s.iterrows():
        col = _arm_color(r["arm"])
        for a, x, off in ((ax, r["usd_per_100_stated"], off_a), (ax2, r["total_b"], off_b)):
            a.plot([x, x], [r["norm_score_lo"], r["norm_score_hi"]], color=col, lw=1.2, alpha=0.55, zorder=2)
            _dot(a, x, r["norm_score"], col, size=6)
            dx, dy = off.get(r["arm"], (5, 0))
            label = ARM_LABEL[r["arm"]] if a is ax else ARM_LABEL[r["arm"]].replace(", medium", " (medium)")
            if a is ax2 and r["arm"] == "gpt-oss-120b/medium":
                continue  # same model and size as gpt-oss-120b (low); labeled once
            a.annotate(label, (x, r["norm_score"]), xytext=(dx, dy), textcoords="offset points", fontsize=6.8,
                       color=INK, ha="center" if dx == 0 else ("left" if dx > 0 else "right"), va="center")
    from matplotlib.ticker import NullFormatter
    for a, ticks, fmt in ((ax, [0.005, 0.01, 0.02, 0.05, 0.1, 0.2], lambda t: f"${t:g}"),
                          (ax2, [20, 50, 100, 200, 500, 1000], lambda t: f"{t:g}B")):
        lo, hi = a.get_xlim()
        tk = [t for t in ticks if lo <= t <= hi]
        a.set_xticks(tk)
        a.set_xticklabels([fmt(t) for t in tk])
        a.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("$ per 100 decisions (stated pricing, log)")
    ax.set_ylabel("score relative to beam-search bot")
    ax.set_title("a  Quality against price", pad=7)
    ax2.set_xlabel("total parameters (log)")
    ax2.set_title("b  Quality against size", pad=7)
    sp = an["spearman_vs_score"]
    for a, key in ((ax, "price_stated"), (ax2, "total_params")):
        a.text(0.03, 0.88, f"Spearman {sp[key]['rho']:+.2f} [{sp[key]['lo']:+.2f}, {sp[key]['hi']:+.2f}]", transform=a.transAxes,
               ha="left", va="top", fontsize=6.8, color=INK2)
    ax.text(ax.get_xlim()[0] * 1.05, 1.0, "beam-search bot = 1.0", va="bottom", ha="left", fontsize=6.8, color=INK2)
    _save(fig, "ladder")
    FACTS["ladder2"] = {"summary": s.round(4).to_dict("records"), "analysis": an}



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



# ---- Figure: the decision model as served (E5) ------------------------------------------------------

def fig_decision() -> None:
    d = INTELIF_DIR
    st = pd.read_csv(d / "steps.csv")
    ep = pd.read_csv(d / "episodes_scored.csv") if (d / "episodes_scored.csv").exists() else pd.read_csv(d / "episodes.csv")
    seeds = sorted(ep["episode_seed"].unique())
    it6 = pd.read_csv(RUNS / "iter06" / "episodes.csv")
    it6 = it6[it6["episode_seed"].isin(seeds)]
    fig = plt.figure(figsize=(W, 2.45))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 0.62, 0.8, 0.95], wspace=0.5)
    ax, ax2, ax3 = fig.add_subplot(gs[0]), fig.add_subplot(gs[2]), fig.add_subplot(gs[3])
    # (a) CPU latency against prompt length
    ax.plot(st["input_tokens"], st["latency_s"], "o", ms=2.0, color=MUTED, alpha=0.5, mec="none")
    q = pd.read_csv(d / "latency_quiet.csv") if (d / "latency_quiet.csv").exists() else None
    if q is not None:  # quiet-machine re-timing (the reported latency); in-game points are faint
        ax.plot(q["input_tokens"], q["latency_s"], "o", ms=3.2, color=INK, mec="white", mew=0.5, zorder=3)
        k = float(np.median(q["latency_s"] / q["input_tokens"]))
    else:
        k = float(np.median(st["latency_s"] / st["input_tokens"]))
    xs = np.array([300, 1400])
    ax.plot(xs, xs * k, color=INK2, lw=0.9)
    ax.text(1400, 1400 * k * 1.12, f"{k * 1000:.0f} ms / token", ha="right", va="bottom", fontsize=6.8, color=INK2)
    ax.axhline(0.04, color=MODEL["gpt-oss-120b"], lw=0.9, ls=(0, (3, 2)))
    ax.text(310, 0.047, "reported, one GPU: ~40 ms", fontsize=6.5, color=INK2, va="bottom")
    llm_p50 = float(pd.read_csv(RUNS / "iter06" / "steps.csv").query("arm == 'gemma-4-31B/direct'")["latency_s"].median())
    ax.axhline(llm_p50, color=MODEL["gemma-4-31B"], lw=0.9, ls=(0, (3, 2)))
    ax.text(310, llm_p50 * 1.18, f"gemma-4-31B, API: {llm_p50:.1f} s", fontsize=6.5, color=INK2, va="bottom")
    ax.set_yscale("log")
    ax.set_yticks([0.01, 0.1, 1, 10, 100])
    ax.set_yticklabels(["0.01", "0.1", "1", "10", "100"])
    ax.set_ylim(0.012, 150)
    ax.set_xlim(280, 1450)
    ax.set_xlabel("input tokens per decision")
    ax.set_ylabel("seconds per decision (log)")
    ax.set_title("a  Intelif on a 4-core CPU", pad=7)
    # (b) score on the same games
    pool = lambda g: float((g["score"] - g["ref_random_score"]).sum() / (g["ref_beam_score"] - g["ref_random_score"]).sum())
    per = it6.groupby("arm").apply(pool).sort_values()
    rows = [(ARM_LABEL.get(a, a), v, _arm_color(a)) for a, v in per.items()]
    rows.append(("Intelif (decision model)", pool(ep), INK))
    rows.sort(key=lambda r: r[1])
    for i, (name, v, col) in enumerate(rows):
        marker = "D" if name.startswith("Intelif") else "o"
        ax2.plot([v], [i], marker, ms=5.5 if marker == "o" else 5, color=col, mec="white", mew=1.0, zorder=3)
    ax2.set_yticks(range(len(rows)))
    ax2.set_yticklabels([r[0] for r in rows], fontsize=6.6)
    for lbl in ax2.get_yticklabels():
        if lbl.get_text().startswith("Intelif"):
            lbl.set_fontweight("bold")
    ax2.set_ylim(-0.7, len(rows) - 0.3)
    ax2.set_xlim(-0.03, 1.1)
    ax2.axvline(1.0, color=INK2, lw=0.7)
    ax2.set_xlabel("score relative to beam search")
    ax2.set_title(f"b  Same {len(seeds)} games", pad=7)
    ax2.grid(axis="y", visible=False)
    # (c) does the top option's probability track move quality?
    if "regret_beam" in st:
        st["pbin"] = pd.qcut(st["top_prob"], 4, duplicates="drop")
        g = st.groupby("pbin", observed=True).agg(p=("top_prob", "median"), regret=("regret_beam", "mean"),
                                                   best=("top1_beam", "mean"), n=("turn", "size"))
        ax3.plot(g["p"], g["regret"], color=INK, lw=1.6)
        for x, y in zip(g["p"], g["regret"]):
            _dot(ax3, x, y, INK, size=5)
        ax3.set_xlabel("probability of the chosen move")
        ax3.set_ylabel("oracle regret per move")
        ax3.set_ylim(bottom=0)
        ax3.set_title("c  Confidence vs quality", pad=7)
        FACTS["decision_calibration"] = g.reset_index(drop=True).round(4).to_dict("records")
    _save(fig, "decision")
    FACTS["decision"] = {"ms_per_token": round(k * 1000, 2), "latency_p50_s": float(st["latency_s"].median()),
                         "quiet": None if q is None else {"n": int(len(q)), "p50_s": float(q["latency_s"].median()),
                                                          "same_choice": float(q["same_choice"].mean()),
                                                          "min_s": float(q["latency_s"].min()), "max_s": float(q["latency_s"].max())},
                         "input_tokens_mean": float(st["input_tokens"].mean()), "decisions": int(len(st)),
                         "episodes": ep.round(4).to_dict("records"), "llm_same_seeds": per.round(4).to_dict(),
                         "gemma_api_latency_p50_s": llm_p50}



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



# ---- Calls and cost per experiment (Appendix A) -------------------------------------------------------

EXPERIMENTS = {"E1 model ladder": ["scaleup/ladder"], "E2 history policy": ["scaleup/memory"],
               "E3 cache capacity": ["scaleup/capacity", "scaleup/capacity_part2", "scaleup/capacity_part3"],
               "E4 interference": ["interference/confirm_gemma", "interference/confirm_deepseek"]}


def facts_calls() -> None:
    import sys
    sys.path.insert(0, str(ROOT))
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



def main() -> None:
    fig_ladder2()
    fig_memory2()
    fig_capacity_all()
    fig_capacity()
    fig_interference()
    fig_decision()
    facts_calls()
    spend = pd.read_csv(RUNS / "spend.csv")
    FACTS["spend_total_usd"] = round(float(spend["cost_usd"].sum()), 4)
    FACTS["spend_by_iteration"] = spend.groupby("iteration")["cost_usd"].sum().round(4).to_dict()
    (OUT / "facts.json").write_text(json.dumps(FACTS, indent=1, default=str))
    print("wrote", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
