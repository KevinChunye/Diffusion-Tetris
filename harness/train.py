"""train.py - train a bot with the repo's existing trainers (CPU-friendly defaults), then deploy it
with harness.play.

  python -m harness.train dqn --episodes 300 --max_steps 500 --runs_dir runs/train
      -> train_updated.py (CNN-DQN); checkpoint at runs/train/<run>/checkpoint.pt
  python -m harness.train diffusion --episodes 100 --epochs 5 --out_dir runs/train/diffusion
      -> experiments.make_dataset (greedy heuristic teacher) + diffusion/train_diffusion_updated.py
         checkpoint at <out_dir>/checkpoints/ckpt.pt
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(cmd):
    print("[train]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=REPO)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="kind", required=True)
    d = sub.add_parser("dqn")
    d.add_argument("--episodes", type=int, default=300)
    d.add_argument("--max_steps", type=int, default=500)
    d.add_argument("--eval_episodes", type=int, default=20)
    d.add_argument("--device", default="cpu")
    d.add_argument("--seed", type=int, default=42)
    d.add_argument("--runs_dir", default="runs/train")
    f = sub.add_parser("diffusion")
    f.add_argument("--episodes", type=int, default=100, help="teacher episodes for the dataset")
    f.add_argument("--epochs", type=int, default=5)
    f.add_argument("--horizon", type=int, default=8)
    f.add_argument("--device", default="cpu")
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--out_dir", default="runs/train/diffusion")
    args = ap.parse_args()

    py = sys.executable
    if args.kind == "dqn":
        run([py, "train_updated.py", "--episodes", str(args.episodes), "--max_steps", str(args.max_steps),
             "--eval_episodes", str(args.eval_episodes), "--device", args.device, "--seed", str(args.seed),
             "--runs_dir", args.runs_dir])
        ckpts = sorted(glob.glob(os.path.join(REPO, args.runs_dir, "*", "checkpoint.pt")), key=os.path.getmtime)
        ckpt = os.path.relpath(ckpts[-1], REPO) if ckpts else "<checkpoint.pt>"
        print(f"\nDeploy it:\n  python -m harness.play --bot dqn:{ckpt} --seeds 0 --pieces 200 --gif runs/play/dqn.gif")
        return
    data_dir = os.path.join(args.out_dir, "dataset")
    run([py, "-m", "experiments.make_dataset", "--episodes", str(args.episodes), "--seed", str(args.seed),
         "--teacher", "heuristic", "--device", args.device, "--out_dir", data_dir,
         "--output_dir", os.path.join(args.out_dir, "dataset_run"), "--top_episode_pct", "0.5",
         "--advantage_quantile", "0.0", "--horizon", str(args.horizon)])
    run([py, "diffusion/train_diffusion_updated.py", "--dataset_path",
         os.path.join(data_dir, f"sequences_H{args.horizon}.npz"), "--epochs", str(args.epochs),
         "--device", args.device, "--seed", str(args.seed), "--output_dir", args.out_dir])
    ckpt = os.path.join(args.out_dir, "checkpoints", "ckpt.pt")
    print(f"\nDeploy it:\n  python -m harness.play --bot diffusion:{ckpt} --seeds 0 --pieces 100 --gif runs/play/diffusion.gif")


if __name__ == "__main__":
    main()
