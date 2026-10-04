"""decision_model.py - a Tetris agent backed by a decision model instead of a text generator.

Decision models score a caller-supplied set of options in one forward pass and return a probability
for each, with no text to generate or parse. Two backends share one request format (POST /v1/systemone):
- `intelif`: the open Intelif model (github.com/SkAndMl/intelif, Qwen3-4B + LoRA + linear scorer),
  self-hosted in this process (Python 3.12 environment with intelif installed);
- `jev`: TypeSafe's hosted Jev, called over HTTPS with the key in TYPESAFE_API_KEY. This backend needs
  only numpy/pandas/matplotlib, so it runs on a laptop or in deploy/jev/ (Docker).

Each legal placement of the current piece is one option. The state and question follow Intelif's own
Tetris example (board rows as '#'/'.', the piece, an instruction to clear lines and avoid holes), and
each option's description gives the same simulated outcome our annotated LLM prompts show (lines
cleared, new holes, stack height change, surface change), plus the next piece.

  python -m llm.decision_model --backend intelif --seeds 1000,1001,1002 --dtype fp32compute \\
      --out runs/explore/scaleup/intelif
  TYPESAFE_API_KEY=... python -m llm.decision_model --backend jev --seeds 1000,1001,1002 \\
      --out runs/explore/scaleup/jev
Writes calls.jsonl (one raw record per call: every option with its description and probability, latency,
input tokens), steps.csv in the same schema as LLM game logs (so llm.reference_regret and harness.compare_gif
work unchanged), episodes.csv and manifest.json.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np

from TetrisGym_updated import TetrisGym

INSTRUCTIONS = ("You are playing Tetris. Where should the {piece} piece land? "
                "Clear lines, avoid holes and keep the stack low and flat.")


def _stats(board: np.ndarray) -> Tuple[int, int, int]:
    h, w = board.shape
    filled = board != 0
    heights = [h - int(np.argmax(filled[:, c])) if filled[:, c].any() else 0 for c in range(w)]
    holes = sum(int((~filled[h - heights[c]:, c]).sum()) for c in range(w))
    bump = sum(abs(a - b) for a, b in zip(heights, heights[1:]))
    return max(heights), holes, bump


def _after(env: TetrisGym, rot: int, x: int) -> Tuple[np.ndarray, int]:
    game = env.game
    piece = game.current_piece[1][rot]
    ph, pw = piece.shape
    y = game._find_drop_height(piece, x)
    board = game.board.copy()
    board[y:y + ph, x:x + pw] += piece
    full = np.all(board == 1, axis=1)
    lines = int(full.sum())
    if lines:
        board = np.vstack([np.zeros((lines, game.width), dtype=board.dtype), board[~full]])
    return board, lines


def describe(before: np.ndarray, after: np.ndarray, lines: int) -> str:
    dh, dholes, dbump = (a - b for a, b in zip(_stats(after), _stats(before)))
    return ", ".join([
        f"clears {lines} line{'s' * (lines > 1)}" if lines else "clears no lines",
        f"creates {dholes} new hole{'s' * (dholes > 1)}" if dholes > 0 else "no new holes",
        f"raises the stack by {dh}" if dh > 0 else "keeps the stack low",
        "surface gets bumpier" if dbump > 0 else "surface stays flat"])


def question(env: TetrisGym) -> Tuple[Dict, Dict[str, str], str, Dict[str, int]]:
    """(state, criteria, instructions, option key -> action id) for the current position."""
    board = env.game.board
    piece, nxt = env.game.current_piece[0], env.game.next_piece[0]
    criteria, ids = {}, {}
    for aid in env.get_valid_action_ids():
        rot, x = env.id_to_action[aid]
        after, lines = _after(env, rot, x)
        key = f"rotation {rot}, column {x}"
        criteria[key] = describe(board, after, lines)
        ids[key] = aid
    state = {"board": "\n".join("".join("#" if c else "." for c in row) for row in board), "piece": piece,
             "next piece": nxt}
    return state, criteria, INSTRUCTIONS.format(piece=piece), ids


def board_to_str(board) -> str:  # same encoding as llm.llm_policy.board_to_str, without its torch imports
    return "/".join("".join("#" if v else "." for v in row) for row in board)


class JevClient:
    """Minimal client for TypeSafe's hosted Jev (POST /v1/systemone). Returns the same shape as Intelif's
    `system_one` (choices[name].choice / .confidence / .probabilities, usage.input_tokens), plus the raw
    response. The API key is read from the environment and is never logged."""

    def __init__(self, model: str = "jev-1.13.0", base_url: str = "", timeout: float = 60.0, retries: int = 4):
        self.key = os.environ.get("TYPESAFE_API_KEY", "").strip()
        if not self.key:
            raise SystemExit("TYPESAFE_API_KEY is not set (see deploy/jev/README.md)")
        self.model = model
        self.base_url = (base_url or os.environ.get("TYPESAFE_BASE_URL", "https://api.typesafe.ai")).rstrip("/")
        self.timeout, self.retries = timeout, retries
        self.name = model

    def system_one(self, state, questions):
        body = json.dumps({"model": self.model, "state": state, "questions": questions}).encode()
        for attempt in range(self.retries + 1):
            req = urllib.request.Request(f"{self.base_url}/v1/systemone", data=body, method="POST",
                                         headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    raw = json.loads(r.read())
                break
            except urllib.error.HTTPError as e:  # retry rate limits and server errors; fail on client errors
                if e.code in (429, 500, 502, 503, 504) and attempt < self.retries:
                    time.sleep(2 ** attempt)
                    continue
                raise SystemExit(f"TypeSafe API {e.code}: {e.read()[:300]!r}")
            except urllib.error.URLError:
                if attempt < self.retries:
                    time.sleep(2 ** attempt)
                    continue
                raise
        choices = {k: SimpleNamespace(choice=a["choice"], confidence=float(a.get("confidence", 0.0)),
                                      probabilities=a.get("probabilities", {}))
                   for k, a in raw["answers"].items() if a.get("type") == "choice"}
        usage = SimpleNamespace(input_tokens=int(raw.get("usage", {}).get("input_tokens", 0)))
        return SimpleNamespace(choices=choices, usage=usage, model=raw.get("model", self.model), raw=raw)


def play(model, seed: int, pieces: int, arm: str, calls_log: Path = None, model_label: str = "intelif-qwen3-4b@v0.1",
         tag: str = "intelif") -> Tuple[List[Dict], Dict]:
    env = TetrisGym(max_steps=None)
    env.reset(seed=seed)
    rows, lines = [], 0
    for turn in range(pieces):
        if env.game.game_over or not env.get_valid_action_ids():
            break
        state, criteria, instr, ids = question(env)
        board_s, curr, nxt = board_to_str(env.game.board), env.game.current_piece[0], env.game.next_piece[0]
        t0 = time.perf_counter()
        resp = model.system_one(state, {"move": {"type": "choice", "criteria": criteria, "instructions": instr}})
        latency = time.perf_counter() - t0
        ans = resp.choices["move"]
        aid = ids[ans.choice]
        probs = {k: float(v) for k, v in ans.probabilities.items()}
        top_prob = max(probs.values()) if probs else float(ans.confidence)
        if calls_log is not None:  # raw record of every call: full distribution over options, kept for later analysis
            rec = {"event": "call", "arm": arm, "episode_seed": seed, "turn": turn, "t_start_unix": time.time() - latency,
                   "latency_s": latency, "input_tokens": int(resp.usage.input_tokens), "choice": ans.choice,
                   "confidence": float(ans.confidence), "options": {k: {"action_id": ids[k], "description": criteria[k],
                                                                       "p": probs.get(k)} for k in criteria},
                   "state": state, "instructions": instr, "served_model": getattr(resp, "model", model_label)}
            with open(calls_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        _, _, done, info = env.step(aid)
        lines += int(info["lines_cleared"])
        rows.append({"arm": arm, "model": model_label, "episode_seed": seed, "turn": turn, "board": board_s,
                     "curr": curr, "next": nxt, "legal_ids": json.dumps(sorted(ids.values())), "proposed_id": aid,
                     "used_id": aid, "fallback_reason": "", "lines_cleared": int(info["lines_cleared"]),
                     "score_after": int(env.game.score), "done": bool(done), "n_legal": len(ids),
                     "top_prob": top_prob, "confidence": float(ans.confidence), "input_tokens": int(resp.usage.input_tokens),
                     "latency_s": latency, "ok": True})
        print(f"[{tag}] seed {seed} piece {turn + 1}: {ans.choice} p={top_prob:.2f} "
              f"{latency:.1f}s lines {lines} score {env.game.score}", flush=True)
        if done:
            break
    ep = {"arm": arm, "episode_seed": seed, "pieces": len(rows), "lines": lines, "score": int(env.game.score),
          "topped_out": bool(env.game.game_over), "latency_p50_s": float(np.median([r["latency_s"] for r in rows])),
          "input_tokens_mean": float(np.mean([r["input_tokens"] for r in rows]))}
    return rows, ep


def quantize_int8(model) -> None:
    """Dynamic int8 weights for every Linear layer of the base model (fp32 activations), for CPUs with int8
    dot-product instructions but no native bf16. Converted layer by layer to keep peak memory near the
    bf16 model's size. The 1-output scorer head stays in fp32."""
    import torch
    from torch import nn
    from torch.ao.nn.quantized.dynamic import Linear as QLinear
    from torch.ao.quantization import default_dynamic_qconfig

    import gc

    net = model.network
    parents = [m for m in net.base_model.modules() if type(m) is not nn.Linear]  # hold no Linear references
    for parent in parents:
        for name in [n for n, c in parent.named_children() if type(c) is nn.Linear]:
            child = getattr(parent, name)
            child.float()
            child.qconfig = default_dynamic_qconfig
            setattr(parent, name, QLinear.from_float(child))
            del child
        gc.collect()
    net.float()
    torch.set_grad_enabled(False)


def compute_fp32(model) -> None:
    """Keep the bf16 weights but compute in fp32: each Linear upcasts its weight per call, all other
    parameters are converted once. Same weights as bf16, higher arithmetic precision; on CPUs without
    native bf16 this is faster than bf16 matmuls."""
    import types

    import torch
    import torch.nn.functional as F
    from torch import nn

    def fwd(self, x):
        return F.linear(x, self.weight.float(), None if self.bias is None else self.bias.float())

    net = model.network
    for m in net.modules():
        if type(m) is nn.Linear and m is not net.scorer:
            m.forward = types.MethodType(fwd, m)
        else:
            for p in m.parameters(recurse=False):
                p.data = p.data.float()
    torch.set_grad_enabled(False)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return os.environ.get("GIT_COMMIT", "unknown")  # e.g. inside the Docker image


def load_intelif(args):
    import torch
    from intelif import Intelif

    torch.set_num_threads(args.threads)
    if args.base_dir:  # use base weights fetched ahead of time instead of the Hugging Face cache
        import huggingface_hub
        huggingface_hub.snapshot_download = lambda *a, **k: args.base_dir
    model = Intelif.from_pretrained(device="cpu", dtype="bfloat16" if args.dtype in ("int8", "fp32compute") else args.dtype)
    if args.dtype == "int8":
        quantize_int8(model)
    elif args.dtype == "fp32compute":
        compute_fp32(model)
    info = {"model": "UserMoonlight/intelif-qwen3-4b", "revision": "v0.1", "device": "cpu", "dtype": args.dtype,
            "threads": args.threads, "base_dir": args.base_dir, "torch": torch.__version__}
    return model, info, "intelif-qwen3-4b@v0.1", "intelif-qwen3-4b/decision"


def main() -> None:
    import pandas as pd

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["intelif", "jev"], default="intelif")
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--pieces", type=int, default=100)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="bfloat16", help="intelif: bfloat16, fp32compute (bf16 weights, fp32 arithmetic) or int8")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--stop_after_utc", default="", help="HH:MM; start no new game after this time (UTC, today)")
    ap.add_argument("--base_dir", default="", help="intelif: local folder with the base model's *.safetensors")
    ap.add_argument("--jev_model", default="jev-1.13.0", help="jev: pinned model id (jev-latest moves)")
    args = ap.parse_args()
    out = Path(args.out)
    if any((out / f).exists() for f in ("steps.csv", "calls.jsonl", "manifest.json")):
        raise FileExistsError(f"fresh directory required: {out}")
    t_load = time.perf_counter()
    if args.backend == "jev":
        model = JevClient(model=args.jev_model)
        info = {"model": args.jev_model, "endpoint": f"{model.base_url}/v1/systemone", "hosted_by": "TypeSafe"}
        label, arm = args.jev_model, "jev/decision"
    else:
        model, info, label, arm = load_intelif(args)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"backend": args.backend, **info, "load_s": time.perf_counter() - t_load,
                "client": platform.platform(), "cpu": platform.processor() or platform.machine(),
                "git_commit": _git_commit(), "seeds": args.seeds, "pieces": args.pieces,
                "stop_after_utc": args.stop_after_utc, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    eps = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        if args.stop_after_utc and time.strftime("%H:%M", time.gmtime()) >= args.stop_after_utc:
            print(f"[{args.backend}] {args.stop_after_utc} UTC reached; no new games started", flush=True)
            break
        rows, ep = play(model, seed, args.pieces, arm, out / "calls.jsonl", model_label=label, tag=args.backend)
        pd.DataFrame(rows).to_csv(out / "steps.csv", mode="a", header=not (out / "steps.csv").exists(), index=False)
        eps.append(ep)
        pd.DataFrame(eps).to_csv(out / "episodes.csv", index=False)
    print(pd.DataFrame(eps).to_string(index=False))


if __name__ == "__main__":
    main()
