"""scaleup_analysis.py - analyses for the scale-up experiments (notes/scaleup_plan.md).

  python -m llm.scaleup_analysis ladder  runs/explore/scaleup/ladder     # E1
  python -m llm.scaleup_analysis memory  runs/explore/scaleup/memory     # E2
  python -m llm.scaleup_analysis intelif runs/explore/scaleup/intelif    # decision model (oracle regret, scores)

Cost is estimated two ways from reported token counts:
- `usd_stated`: the provider's stated policy (cached input tokens billed at $0 for every model);
- `usd_full`: an upper bound that bills cached tokens at the input price where a model's card lists no
  cached price.
Uncertainty is a bootstrap over seeds (games are the independent unit).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from llm.model_arch import PARAMS_B
from llm.tensormesh_client import load_pricing

B = 2000
RNG = np.random.default_rng(0)


def _usd(df: pd.DataFrame, pricing: dict) -> pd.DataFrame:
    df = df.copy()
    pin = df["model"].map(lambda m: float(pricing[m]["input"]))
    pout = df["model"].map(lambda m: float(pricing[m]["output"]))
    cached_listed = df["model"].map(lambda m: pricing[m].get("cached") is not None)
    unc = df["prompt_tokens"] - df["cached_tokens"].fillna(0)
    df["usd_stated"] = (unc * pin + df["completion_tokens"] * pout) / 1e6
    df["usd_full"] = np.where(cached_listed, df["usd_stated"], (df["prompt_tokens"] * pin + df["completion_tokens"] * pout) / 1e6)
    return df


def _boot_mean(values_by_seed: pd.Series):
    v = values_by_seed.to_numpy(float)
    if len(v) < 2:
        return (np.nan, np.nan)
    reps = [np.mean(RNG.choice(v, len(v), replace=True)) for _ in range(B)]
    return tuple(np.quantile(reps, [0.025, 0.975]))


def _spearman(x, y) -> float:
    return float(pd.Series(x).rank().corr(pd.Series(y).rank()))


def ladder(run_dir: str) -> dict:
    d = Path(run_dir)
    pricing = load_pricing("configs/pricing.yaml")
    ep = pd.read_csv(d / "episodes.csv")
    st = _usd(pd.read_csv(d / "steps.csv"), pricing)
    per = st.groupby(["arm", "episode_seed"]).agg(decisions=("turn", "size"), usd_stated=("usd_stated", "sum"),
                                                   usd_full=("usd_full", "sum"), latency_p50=("latency_s", "median")).reset_index()
    ep = ep.merge(per, on=["arm", "episode_seed"], how="left")
    rows = []
    for arm, g in ep.groupby("arm"):
        model = g["model"].iloc[0]
        lo, hi = _boot_mean(g["norm_score"])
        dec = g["decisions"].sum()
        rows.append({"arm": arm, "model": model, "seeds": len(g), "decisions": int(dec),
                     "norm_score": g["norm_score"].mean(), "norm_score_lo": lo, "norm_score_hi": hi,
                     "lines": g["lines_cleared"].mean(), "pieces": g["pieces_placed"].mean(),
                     "regret_beam": float(np.average(g["regret_beam"], weights=g["decisions"])),
                     "usd_per_100_stated": 100 * g["usd_stated"].sum() / dec, "usd_per_100_full": 100 * g["usd_full"].sum() / dec,
                     "latency_p50_s": float(st[st["arm"] == arm]["latency_s"].median()),
                     "total_b": PARAMS_B[model][0], "active_b": PARAMS_B[model][1]})
    s = pd.DataFrame(rows).sort_values("norm_score", ascending=False)
    # rank correlations, bootstrapped over seeds (same resampled seeds for every arm)
    seeds = sorted(ep["episode_seed"].unique())
    piv = ep.pivot_table(index="episode_seed", columns="arm", values="norm_score")
    arms = list(piv.columns)
    meta = s.set_index("arm").loc[arms]
    stats = {}
    for name, x in (("price_stated", meta["usd_per_100_stated"]), ("total_params", meta["total_b"]), ("active_params", meta["active_b"])):
        point = _spearman(x.values, piv.mean().values)
        reps = []
        for _ in range(B):
            pick = RNG.choice(seeds, len(seeds), replace=True)
            reps.append(_spearman(x.values, piv.loc[pick].mean().values))
        stats[name] = {"rho": point, "lo": float(np.quantile(reps, 0.025)), "hi": float(np.quantile(reps, 0.975))}
    s.to_csv(d / "summary_scaleup.csv", index=False)
    out = {"arms": len(s), "seeds": len(seeds), "decisions": int(s["decisions"].sum()), "spearman_vs_score": stats}
    (d / "analysis.json").write_text(json.dumps(out, indent=1))
    return out


def memory(run_dir: str) -> dict:
    d = Path(run_dir)
    pricing = load_pricing("configs/pricing.yaml")
    st = _usd(pd.read_csv(d / "steps.csv"), pricing)
    ep = pd.read_csv(d / "episodes.csv")
    st["policy"] = st["arm"].str.split("/").str[1]
    st["mname"] = st["arm"].str.split("/").str[0]
    st["uncached"] = st["prompt_tokens"] - st["cached_tokens"].fillna(0)
    per = st.groupby(["mname", "policy", "episode_seed"]).agg(
        decisions=("turn", "size"), uncached=("uncached", "mean"), prompt=("prompt_tokens", "mean"),
        cached_frac=("cached_tokens", lambda x: np.nan), regret=("regret_beam", "mean"),
        usd_stated=("usd_stated", "sum"), usd_full=("usd_full", "sum")).reset_index()
    tot = st.groupby(["mname", "policy", "episode_seed"])[["cached_tokens", "prompt_tokens"]].sum()
    per["cached_frac"] = (tot["cached_tokens"] / tot["prompt_tokens"]).values
    rows = []
    for (m, pol), g in per.groupby(["mname", "policy"]):
        e = ep[ep["arm"] == f"{m}/{pol}"]
        dec = g["decisions"].sum()
        rows.append({"model": m, "policy": pol, "seeds": len(g), "decisions": int(dec),
                     "uncached_per_decision": float(np.average(g["uncached"], weights=g["decisions"])),
                     "prompt_per_decision": float(np.average(g["prompt"], weights=g["decisions"])),
                     "cached_frac": float(np.average(g["cached_frac"], weights=g["decisions"])),
                     "regret": float(np.average(g["regret"], weights=g["decisions"])),
                     "usd_per_100_stated": 100 * g["usd_stated"].sum() / dec, "usd_per_100_full": 100 * g["usd_full"].sum() / dec,
                     "norm_score": float(e["norm_score"].mean()), "pieces": float(e["pieces_placed"].mean())})
    s = pd.DataFrame(rows)
    # paired by seed: regret(append) - regret(stateless); uncached(window8) / uncached(append)
    paired = {}
    for m, g in per.groupby("mname"):
        piv_r = g.pivot_table(index="episode_seed", columns="policy", values="regret")
        piv_u = g.pivot_table(index="episode_seed", columns="policy", values="uncached")
        res = {}
        for name, piv, fn in (("regret_append_minus_stateless", piv_r, lambda p: (p["append"] - p["stateless"]).mean()),
                              ("regret_window8_minus_stateless", piv_r, lambda p: (p["window8"] - p["stateless"]).mean()),
                              ("uncached_window8_over_append", piv_u, lambda p: p["window8"].mean() / p["append"].mean())):
            p = piv.dropna()
            if len(p) < 2:
                continue
            reps = [fn(p.iloc[RNG.integers(0, len(p), len(p))]) for _ in range(B)]
            res[name] = {"point": float(fn(p)), "lo": float(np.quantile(reps, 0.025)), "hi": float(np.quantile(reps, 0.975)), "seeds": len(p)}
        paired[m] = res
    s.to_csv(d / "summary_scaleup.csv", index=False)
    out = {"cells": len(s), "decisions": int(s["decisions"].sum()), "paired": paired}
    (d / "analysis.json").write_text(json.dumps(out, indent=1))
    return out


def intelif(run_dir: str, workers: int = 3) -> dict:
    from llm.oracle import compute_regrets
    from llm.tetris_tools import normalized_scores, reference_table

    d = Path(run_dir)
    st = pd.read_csv(d / "steps.csv")
    if "regret_beam" not in st:
        st = compute_regrets(st, workers=workers, n_samples=2, depth=4, beam_h=2, beam_w=4, beam_samples=1)
        st["top1_beam"] = (st["rank_beam"] == 1).astype(float)
        st.to_csv(d / "steps.csv", index=False)
    ep = pd.read_csv(d / "episodes.csv").rename(columns={"lines": "lines_cleared", "pieces": "pieces_placed"})
    refs = reference_table(sorted(ep["episode_seed"].unique()), 100, "runs/explore/reference_scores.csv")
    ep = normalized_scores(ep, refs, "score")
    reg = st.groupby("episode_seed").agg(regret_beam=("regret_beam", "mean"), top1_beam=("top1_beam", "mean"))
    ep = ep.merge(reg.reset_index(), on="episode_seed", how="left")
    ep.to_csv(d / "episodes_scored.csv", index=False)
    out = {"episodes": len(ep), "decisions": int(len(st)), "norm_score": float(ep["norm_score"].mean()),
           "top1_beam": float(st["top1_beam"].mean()),
           "regret_beam": float(st["regret_beam"].mean()), "latency_p50_s": float(st["latency_s"].median()),
           "input_tokens_mean": float(st["input_tokens"].mean()), "legal_rate": 1.0}
    (d / "analysis.json").write_text(json.dumps(out, indent=1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=["ladder", "memory", "intelif"])
    ap.add_argument("run_dir")
    args = ap.parse_args()
    out = {"ladder": ladder, "memory": memory, "intelif": intelif}[args.kind](args.run_dir)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
