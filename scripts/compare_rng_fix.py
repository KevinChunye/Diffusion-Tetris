"""compare_rng_fix.py - Phase 1 before/after for one v1 diffusion config.

"before" = original code (global `random` shared by the real game and every simulation deepcopy;
the eval runner never seeds it), "after" = per-episode seeded game RNG + independent simulation RNGs.
Copies both runs' summary/episode CSVs next to a comparison table.

  python scripts/compare_rng_fix.py --before <eval_before_dir> --after <eval_after_dir> \
      --out runs/explore/phase1_rng_fix
"""

from __future__ import annotations

import argparse
import os
import shutil

import numpy as np
import pandas as pd


def boot_ci(x: np.ndarray, n: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--out", default="runs/explore/phase1_rng_fix")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows = []
    for tag, d in [("before", args.before), ("after", args.after)]:
        shutil.copyfile(os.path.join(d, "summary.csv"), os.path.join(args.out, f"{tag}_summary.csv"))
        eps = pd.read_csv(os.path.join(d, "metrics.csv"))
        eps.to_csv(os.path.join(args.out, f"{tag}_episodes.csv"), index=False)
        row = {"run": tag, "episodes": len(eps)}
        for col in ["score", "steps", "lines_cleared", "regret", "mean_decision_ms"]:
            if col in eps.columns:
                x = eps[col].to_numpy(dtype=float)
                lo, hi = boot_ci(x)
                row[f"{col}_mean"] = float(x.mean())
                row[f"{col}_ci95"] = f"[{lo:.2f}, {hi:.2f}]"
        row["topped_out_before_cap"] = float((eps["steps"] < eps["steps"].max()).mean())
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(args.out, "comparison.csv"), index=False)
    md = table.to_markdown(index=False)
    with open(os.path.join(args.out, "comparison.md"), "w", encoding="utf-8") as f:
        f.write(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
