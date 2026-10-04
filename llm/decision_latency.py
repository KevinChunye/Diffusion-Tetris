"""decision_latency.py - re-time the decision model alone on positions from its own games (quiet machine).

  /home/user/intelif-venv/bin/python -m llm.decision_latency runs/explore/scaleup/intelif --n 30 --threads 3
Writes latency_quiet.csv (one row per position: seed, turn, options, input tokens, seconds, same choice as
in the game) into the run directory.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    import huggingface_hub
    import torch
    from intelif import Intelif

    from llm.decision_model import compute_fp32

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--base_dir", default="/home/user/hf/Qwen3-4B")
    args = ap.parse_args()
    d = Path(args.run_dir)
    calls = [json.loads(l) for l in open(d / "calls.jsonl", encoding="utf-8")]
    pick = np.random.default_rng(0).choice(len(calls), size=min(args.n, len(calls)), replace=False)
    torch.set_num_threads(args.threads)
    huggingface_hub.snapshot_download = lambda *a, **k: args.base_dir
    model = Intelif.from_pretrained(device="cpu", dtype="bfloat16")
    compute_fp32(model)
    rows = []
    for i in sorted(pick):
        c = calls[i]
        criteria = {k: v["description"] for k, v in c["options"].items()}
        t0 = time.perf_counter()
        r = model.system_one(c["state"], {"move": {"type": "choice", "criteria": criteria, "instructions": c["instructions"]}})
        dt = time.perf_counter() - t0
        rows.append({"episode_seed": c["episode_seed"], "turn": c["turn"], "options": len(criteria),
                     "input_tokens": int(r.usage.input_tokens), "latency_s": dt, "in_game_latency_s": c["latency_s"],
                     "same_choice": r.choices["move"].choice == c["choice"]})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(d / "latency_quiet.csv", index=False)
    df = pd.DataFrame(rows)
    print("median s", df["latency_s"].median(), "ms/token", 1000 * (df["latency_s"] / df["input_tokens"]).median(),
          "same choice", df["same_choice"].mean())


if __name__ == "__main__":
    main()
