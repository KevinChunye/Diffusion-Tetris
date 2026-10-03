"""interference_analysis.py

Paired analysis of an interference run directory (live or SIMULATED; the output says which).

Per decision request (kind short/long) the time-to-valid-action is
    ttva = valid_action_s - arrival_s   (scheduled arrival -> first strictly valid legal JSON action)
and is +inf when the request never produced one (HTTP error, exception, unsent, rejected, invalid or
illegal reply), so failures can only make tails worse, never disappear from them. A decision is
"useful" when ttva <= slo_s. Short-request latency is decomposed into client queueing
(admit - arrival), endpoint time to first SSE chunk (first_chunk - admit) and chunk-to-valid-action.
Quality uses the corpus oracle (q_beam): legal rate (lenient parse, as the game would) and regret
q_best - q[proposed] over legal answers.

Uncertainty: two-level paired bootstrap. Resample blocks (time blocks; one trace each) with
replacement, then requests within each sampled block with replacement; the same resampled requests
are used for both conditions (identical trace), and the statistic is recomputed on the pooled sample.

  python -m llm.interference_analysis runs/explore/interference/live_pilot
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
import yaml

from llm.mixed_workload import load_corpus

DECISION = ("short", "long")


def _q(x: np.ndarray, q: float) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return float("nan")
    return float(np.quantile(x, q, method="higher"))  # an order statistic; well defined with +inf


def load_rows(run_dir: str) -> pd.DataFrame:
    rows = [json.loads(l) for l in open(Path(run_dir) / "requests.jsonl", encoding="utf-8")]
    df = pd.DataFrame(rows)
    for col in ("valid_action_s", "first_chunk_s", "admit_s", "complete_s", "ping_s", "cost_usd", "cached_tokens",
                "prompt_tokens", "completion_tokens", "proposed_id"):
        if col not in df:
            df[col] = np.nan
    for col in ("legal", "cache_usage_reported"):
        if col not in df:
            df[col] = False
    return df


def add_metrics(df: pd.DataFrame, entries: Dict[str, Dict], slo_s: float) -> pd.DataFrame:
    df = df.copy()
    dec = df["kind"].isin(DECISION)
    ok = df["status"].eq("ok")
    valid = dec & ok & df["valid_action_s"].notna()
    df["ttva"] = np.where(valid, df["valid_action_s"] - df["arrival_s"], np.where(dec, np.inf, np.nan))
    df["useful"] = np.where(dec, df["ttva"] <= slo_s, np.nan)
    df["queue_s"] = df["admit_s"] - df["arrival_s"]
    df["endpoint_first_chunk_s"] = df["first_chunk_s"] - df["admit_s"]
    df["chunk_to_valid_s"] = df["valid_action_s"] - df["first_chunk_s"]
    legal, regret = [], []
    for rid, kind, status, prop, lg in zip(df["rid"], df["kind"], df["status"], df["proposed_id"], df["legal"]):
        if kind not in DECISION:
            legal.append(np.nan)
            regret.append(np.nan)
            continue
        is_legal = status == "ok" and bool(lg) and pd.notna(prop)
        legal.append(float(is_legal))
        e = entries.get(rid)
        if is_legal and e is not None and str(int(prop)) in e["q_beam"]:
            regret.append(float(e["q_best"]) - float(e["q_beam"][str(int(prop))]))
        else:
            regret.append(np.nan)
    df["legal_rate"] = legal
    df["regret"] = regret
    if "simulated" in df:  # the simulator models timing only; it has no replies to score
        sim = df["simulated"].fillna(False).astype(bool)
        df.loc[sim, ["legal_rate", "regret"]] = np.nan
    return df


def cell_summary(g: pd.DataFrame, duration_s: float) -> Dict[str, float]:
    s = g[g["kind"] == "short"]
    lg = g[g["kind"] == "long"]
    d = g[g["kind"].isin(DECISION)]
    p = g[(g["kind"] == "ping") & g["status"].eq("ok")]
    n_blocks = g["block"].nunique()
    out = {"n_short": len(s), "n_long": len(lg), "n_ping": len(g[g["kind"] == "ping"]),
           "short_p50_ttva": _q(s["ttva"], .5), "short_p95_ttva": _q(s["ttva"], .95), "short_p99_ttva": _q(s["ttva"], .99),
           "short_p95_queue_s": _q(s["queue_s"].dropna(), .95),
           "short_p50_endpoint_first_chunk_s": _q(s["endpoint_first_chunk_s"].dropna(), .5),
           "short_p95_endpoint_first_chunk_s": _q(s["endpoint_first_chunk_s"].dropna(), .95),
           "short_p95_chunk_to_valid_s": _q(s["chunk_to_valid_s"].dropna(), .95),
           "short_slo_miss_rate": float(1 - s["useful"].mean()) if len(s) else np.nan,
           "long_p50_ttva": _q(lg["ttva"], .5), "long_p95_ttva": _q(lg["ttva"], .95),
           "long_max_queue_s": float(lg["queue_s"].max()) if len(lg) else np.nan,
           "useful_per_s": float(d["useful"].sum() / (n_blocks * duration_s)) if len(d) else np.nan,
           "useful_short_per_s": float(s["useful"].sum() / (n_blocks * duration_s)) if len(s) else np.nan,
           "legal_rate": float(d["legal_rate"].mean()) if len(d) else np.nan,
           "mean_regret": float(d["regret"].mean()) if d["regret"].notna().any() else np.nan,
           "ping_p50_s": _q(p["ping_s"].dropna(), .5), "ping_p95_s": _q(p["ping_s"].dropna(), .95),
           "n_ok": int((d["status"] == "ok").sum()), "n_error": int(d["status"].isin(["error", "exception"]).sum()),
           "n_unsent_or_rejected": int(d["status"].str.startswith(("not_sent", "rejected")).sum()),
           "n_invalid_reply": int(((d["status"] == "ok") & ~np.isfinite(d["ttva"])).sum()),
           "est_cost_usd": float(d["cost_usd"].fillna(0).sum()),
           "prompt_tokens": float(d["prompt_tokens"].fillna(0).sum()),
           "completion_tokens": float(d["completion_tokens"].fillna(0).sum()),
           "n_cache_usage_reported": int(d["cache_usage_reported"].fillna(False).astype(bool).sum()),
           "n_cached_tokens_positive": int((d["cached_tokens"].fillna(0) > 0).sum()),
           "max_cached_tokens": float(d["cached_tokens"].fillna(0).max()) if len(d) else 0.0}
    return out


def summaries(df: pd.DataFrame, duration_s: float):
    per_block = pd.DataFrame([dict(model=m, block=b, condition=c, **cell_summary(g, duration_s))
                              for (m, b, c), g in df.groupby(["model", "block", "condition"])])
    pooled = pd.DataFrame([dict(model=m, condition=c, n_blocks=g["block"].nunique(), **cell_summary(g, duration_s))
                           for (m, c), g in df.groupby(["model", "condition"])])
    return per_block, pooled


# ---- paired two-level bootstrap -----------------------------------------------------------------

def _paired_arrays(df: pd.DataFrame, model: str, treat: str, ctrl: str, kinds, col: str):
    """Per block: aligned (treatment, control) values for the requests both conditions received."""
    out = {}
    m = df[(df["model"] == model) & df["kind"].isin(kinds)]
    for b, g in m.groupby("block"):
        t = g[g["condition"] == treat].set_index("rid")[col]
        c = g[g["condition"] == ctrl].set_index("rid")[col]
        rids = sorted(set(t.index) & set(c.index))
        if rids:
            out[b] = (t.loc[rids].to_numpy(float), c.loc[rids].to_numpy(float))
    return out


def paired_bootstrap(arrays: Dict, stat: Callable[[np.ndarray], float], n: int, seed: int,
                     relative: bool) -> Dict[str, float]:
    blocks = sorted(arrays)
    if not blocks:
        return {}

    def compare(tv, cv):
        a, b = stat(tv), stat(cv)
        if relative:
            return a / b - 1 if np.isfinite(a) and np.isfinite(b) and b != 0 else np.nan
        return a - b

    tv = np.concatenate([arrays[b][0] for b in blocks])
    cv = np.concatenate([arrays[b][1] for b in blocks])
    point = compare(tv, cv)
    rng = np.random.default_rng(seed)
    reps = np.empty(n)
    for i in range(n):
        tt, cc = [], []
        for b in rng.choice(blocks, size=len(blocks), replace=True):
            t, c = arrays[b]
            idx = rng.integers(0, len(t), len(t))
            tt.append(t[idx])
            cc.append(c[idx])
        reps[i] = compare(np.concatenate(tt), np.concatenate(cc))
    finite = reps[np.isfinite(reps)]
    lo, hi = (np.quantile(finite, [0.025, 0.975]) if finite.size else (np.nan, np.nan))
    per_block = [compare(*arrays[b]) for b in blocks]
    return {"point": float(point), "ci95_lo": float(lo), "ci95_hi": float(hi), "n_blocks": len(blocks),
            "n_requests": int(len(tv)), "boot_finite_frac": float(finite.size / n),
            "per_block": [round(float(x), 4) for x in per_block]}


def comparisons(df: pd.DataFrame, acfg: Dict, n: int, seed: int) -> pd.DataFrame:
    p95 = lambda x: _q(x, .95)  # noqa: E731
    p50 = lambda x: _q(x, .5)  # noqa: E731
    nanmean = lambda x: float(np.nanmean(x)) if np.isfinite(x).any() else np.nan  # noqa: E731
    specs = [("short_p95_ttva_rel", ("short",), "ttva", p95, True),
             ("short_p95_ttva_diff_s", ("short",), "ttva", p95, False),
             ("short_p50_ttva_diff_s", ("short",), "ttva", p50, False),
             ("short_useful_rate_diff", ("short",), "useful", np.mean, False),
             ("useful_rate_rel", DECISION, "useful", np.mean, True),
             ("long_p50_ttva_diff_s", ("long",), "ttva", p50, False),
             ("legal_rate_diff", DECISION, "legal_rate", np.mean, False),
             ("regret_diff", DECISION, "regret", nanmean, False),
             ("ping_p95_diff_s", ("ping",), "ping_s", lambda x: _q(x[np.isfinite(x)], .95), False)]
    rows = []
    for label in ("h1", "h2", "b2"):
        treat, ctrl = acfg[label]["treatment"], acfg[label]["control"]
        for model in sorted(df["model"].unique()):
            for name, kinds, col, fn, rel in specs:
                if label == "h1" and name in ("useful_rate_rel", "long_p50_ttva_diff_s", "legal_rate_diff", "regret_diff"):
                    continue  # the control has no long requests: different workload, not a paired comparison
                arrays = _paired_arrays(df, model, treat, ctrl, kinds, col)
                res = paired_bootstrap(arrays, fn, n, seed, rel)
                if res:
                    rows.append({"comparison": label, "model": model, "treatment": treat, "control": ctrl,
                                 "metric": name, **res})
    return pd.DataFrame(rows)


def verdicts(cmp: pd.DataFrame, acfg: Dict) -> Dict[str, Dict[str, str]]:
    ni = acfg["noninferiority"]
    out: Dict[str, Dict[str, str]] = {}

    def get(label, model, metric):
        r = cmp[(cmp.comparison == label) & (cmp.model == model) & (cmp.metric == metric)]
        return r.iloc[0] if len(r) else None

    for model in sorted(cmp["model"].unique()):
        v = {}
        rel, diff, ping = get("h1", model, "short_p95_ttva_rel"), get("h1", model, "short_p95_ttva_diff_s"), \
            get("h1", model, "ping_p95_diff_s")
        if rel is not None:
            above_path = ping is None or not np.isfinite(ping["point"]) or diff["point"] > abs(ping["point"])
            if rel["ci95_lo"] > 0 and above_path:
                v["H1"] = (f"supported: short p95 TTVA {rel['point']:+.0%} [{rel['ci95_lo']:+.0%}, {rel['ci95_hi']:+.0%}] "
                           f"with long requests mixed in (path-probe p95 shift {ping['point'] if ping is not None else float('nan'):+.2f} s)")
            elif rel["ci95_hi"] < 0:
                v["H1"] = f"refuted (opposite sign): {rel['point']:+.0%} [{rel['ci95_lo']:+.0%}, {rel['ci95_hi']:+.0%}]"
            else:
                v["H1"] = (f"not supported: {rel['point']:+.0%} [{rel['ci95_lo']:+.0%}, {rel['ci95_hi']:+.0%}]"
                           + ("" if above_path else "; shift not above path-probe variation"))
        h2 = get("h2", model, "short_p95_ttva_rel")
        if h2 is not None:
            need = -float(acfg["h2"]["min_p95_reduction"])
            ur, lr, rg = get("h2", model, "useful_rate_rel"), get("h2", model, "legal_rate_diff"), get("h2", model, "regret_diff")
            def guard(r, ok):  # True / False, or None when the metric was not measured (e.g. simulated)
                return None if r is None or not np.isfinite(r["ci95_lo"]) else bool(ok(r))

            useful_ok = guard(ur, lambda r: r["ci95_lo"] > -float(ni["useful_rate_rel"]))
            legal_ok = guard(lr, lambda r: r["ci95_lo"] > -float(ni["legal_rate_pp"]) / 100)
            regret_ok = guard(rg, lambda r: r["ci95_hi"] < float(ni["regret"]))
            effect = f"short p95 {h2['point']:+.0%} [{h2['ci95_lo']:+.0%}, {h2['ci95_hi']:+.0%}]"
            fmt = lambda ok: "n/a" if ok is None else str(ok)  # noqa: E731
            guards = (f"useful/s noninferior={fmt(useful_ok)}, legal-rate noninferior={fmt(legal_ok)}, "
                      f"regret noninferior={fmt(regret_ok)}")
            if h2["ci95_hi"] <= need and all(ok is not False for ok in (useful_ok, legal_ok, regret_ok)):
                v["H2"] = f"supported: {effect}; {guards}"
            elif h2["ci95_lo"] > need:
                v["H2"] = f"refuted (CI excludes a {-need:.0%} reduction): {effect}; {guards}"
            else:
                v["H2"] = f"inconclusive: {effect}; {guards}"
        b2 = get("b2", model, "short_p95_ttva_rel")
        if b2 is not None:
            v["B2"] = f"priority hints vs fifo: short p95 {b2['point']:+.0%} [{b2['ci95_lo']:+.0%}, {b2['ci95_hi']:+.0%}]"
        out[model] = v
    return out


def overlap_table(df: pd.DataFrame) -> pd.DataFrame:
    """Short-request endpoint time-to-first-chunk by how many long requests of the same replay were
    admitted but had not yet streamed a first chunk (a client-visible proxy for "long prefill in
    progress") when the short was sent. Observational: conditions differ in how often overlap happens."""
    out = []
    for (model, block, cond), g in df.groupby(["model", "block", "condition"]):
        longs = g[(g["kind"] == "long") & g["admit_s"].notna()]
        start = longs["admit_s"].to_numpy(float)
        end = longs["first_chunk_s"].fillna(longs["complete_s"]).to_numpy(float)
        for _, r in g[(g["kind"] == "short") & g["first_chunk_s"].notna()].iterrows():
            n = int(((start <= r["admit_s"]) & (r["admit_s"] < end)).sum())
            out.append({"model": model, "condition": cond, "longs_prefilling": min(n, 2),
                        "endpoint_first_chunk_s": r["endpoint_first_chunk_s"]})
    t = pd.DataFrame(out)
    if t.empty:
        return t
    return (t.groupby(["model", "condition", "longs_prefilling"])["endpoint_first_chunk_s"]
             .agg(n="size", p50="median", p95=lambda x: _q(x, .95)).reset_index())


def plot(df: pd.DataFrame, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = sorted(df["model"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(5.5 * len(models), 4), squeeze=False)
    for ax, model in zip(axes[0], models):
        s = df[(df["model"] == model) & (df["kind"] == "short")]
        for cond, g in s.groupby("condition"):
            x = np.sort(g["ttva"].to_numpy(float))
            y = np.arange(1, len(x) + 1) / len(x)
            fin = np.isfinite(x)
            ax.step(x[fin], y[fin], where="post", label=f"{cond} (n={len(x)}, invalid={int((~fin).sum())})")
        ax.axhline(0.95, color="grey", lw=0.6, ls=":")
        ax.set_xscale("log")
        ax.set_xlabel("short-request time-to-valid-action (s, from scheduled arrival)")
        ax.set_ylabel("CDF")
        ax.set_title(model.split("/")[-1], fontsize=10)
        ax.legend(fontsize=7, loc="lower right")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def analyze(run_dir: str, n_boot: int = None) -> Dict:
    d = Path(run_dir)
    cfg = yaml.safe_load((d / "config.yaml").read_text())
    manifest = json.loads((d / "manifest.json").read_text())
    acfg = cfg["analysis"]
    _, entries = load_corpus(cfg["corpus"])
    df = add_metrics(load_rows(run_dir), entries, float(acfg["slo_s"]))
    duration = float(cfg["trace"]["duration_s"])
    per_block, pooled = summaries(df, duration)
    n = int(n_boot or acfg["bootstrap"]["n"])
    cmp = comparisons(df, acfg, n, int(acfg["bootstrap"]["seed"]))
    verdict = verdicts(cmp, acfg)
    evidence = "SIMULATED" if manifest.get("simulated") else ("LIVE" if manifest.get("live") else "MOCK")
    per_block.to_csv(d / "summary_by_block.csv", index=False)
    pooled.to_csv(d / "summary.csv", index=False)
    cmp.to_csv(d / "paired.csv", index=False)
    overlap_table(df).to_csv(d / "overlap.csv", index=False)
    out = {"evidence": evidence, "run_dir": str(d), "n_bootstrap": n, "verdicts": verdict}
    (d / "verdicts.json").write_text(json.dumps(out, indent=2))
    plot(df, d / "short_ttva_cdf.png", f"{evidence}: short-request TTVA by condition ({d.name})")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--n_boot", type=int, default=None)
    args = ap.parse_args()
    out = analyze(args.run_dir, args.n_boot)
    print(json.dumps(out, indent=2))
    pooled = pd.read_csv(Path(args.run_dir) / "summary.csv")
    cols = ["model", "condition", "n_short", "short_p50_ttva", "short_p95_ttva", "short_p95_queue_s",
            "short_p95_endpoint_first_chunk_s", "long_p50_ttva", "long_max_queue_s", "useful_per_s", "legal_rate",
            "mean_regret", "ping_p95_s", "n_error", "n_invalid_reply", "est_cost_usd"]
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(pooled[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
