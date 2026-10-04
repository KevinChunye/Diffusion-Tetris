"""build_gallery.py - every animation in gallery/, rebuilt from the logged games.

    python scripts/build_gallery.py            # writes gallery/*.gif, gallery/*.png and gallery/README.md

Each GIF replays logged moves exactly (and checks that every agent received the same pieces); the
beam-search bot plays live on the same seed. Snapshots are taken at a fixed piece for the README.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "explore"
OUT = ROOT / "gallery"
TMP = OUT / ".steps"


def _combine(name: str, parts) -> Path:
    """parts: (steps.csv, arm, display label). Writes one steps file whose `model` column is the label."""
    cols = ["arm", "model", "episode_seed", "turn", "used_id", "board", "curr", "next"]
    rows = []
    for path, arm, label in parts:
        s = pd.read_csv(path)
        s = s[s["arm"] == arm].copy()
        s["arm"], s["model"] = label, label
        rows.append(s[cols])
    TMP.mkdir(parents=True, exist_ok=True)
    out = TMP / f"{name}.csv"
    pd.concat(rows).to_csv(out, index=False)
    return out


def _gif(name: str, steps: Path, seed: int, labels, ncols: int, snapshot: int) -> None:
    cmd = [sys.executable, "-m", "harness.compare_gif", "--steps", str(steps), "--seed", str(seed), "--arms", ",".join(labels),
           "--bots", "beam", "--ncols", str(ncols), "--snapshot", str(snapshot), "--out", str(OUT / f"{name}.gif")]
    print(" ".join(cmd[2:]), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def best_ladder_seed(arms) -> int:
    ep = pd.read_csv(RUNS / "scaleup" / "ladder" / "episodes.csv")
    ep = ep[ep["arm"].isin(arms)]
    overall = ep.groupby("arm")["score"].sum()
    scores = ep.pivot_table(index="episode_seed", columns="arm", values="score")
    rho = scores.apply(lambda r: r.rank().corr(overall[r.index].rank()), axis=1)
    return int(rho.idxmax())


def main() -> None:
    it6, il = RUNS / "iter06" / "steps.csv", RUNS / "scaleup" / "intelif_all" / "steps.csv"
    five = [(it6, "gemma-4-31B/direct", "gemma-4-31B"), (it6, "gpt-oss-20b/low", "gpt-oss-20b"),
            (it6, "DeepSeek-V4-Flash/direct", "DeepSeek-V4-Flash"), (il, "intelif-qwen3-4b/decision", "Intelif (decision model)")]
    steps = _combine("five_agents", five)
    for seed in (1000, 1001, 1002):
        _gif(f"same_game_seed{seed}", steps, seed, [p[2] for p in five], ncols=5, snapshot=60)

    mem = RUNS / "scaleup" / "memory" / "steps.csv"
    for model, seed in (("gpt-oss-20b", 5104), ("gpt-oss-120b", 5102)):
        parts = [(mem, f"{model}/stateless", f"{model} · no history"), (mem, f"{model}/append", f"{model} · all turns"),
                 (mem, f"{model}/window8", f"{model} · last 8 turns")]
        _gif(f"history_{model}_seed{seed}", _combine(f"history_{model}", parts), seed, [p[2] for p in parts], ncols=4, snapshot=40)

    lad = RUNS / "scaleup" / "ladder" / "steps.csv"
    arms = [("gpt-oss-120b/low", "gpt-oss-120b"), ("gemma-4-31B/direct", "gemma-4-31B"), ("gpt-oss-20b/low", "gpt-oss-20b"),
            ("Qwen3.5-397B/direct", "Qwen3.5-397B"), ("GLM-5.2/direct", "GLM-5.2")]
    seed = best_ladder_seed([a for a, _ in arms])
    parts = [(lad, a, l) for a, l in arms]
    _gif(f"ladder_seed{seed}", _combine("ladder", parts), seed, [p[2] for p in parts], ncols=3, snapshot=40)
    for f in TMP.glob("*.csv"):
        f.unlink()
    TMP.rmdir()


if __name__ == "__main__":
    main()
