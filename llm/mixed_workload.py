"""mixed_workload.py

Frozen Tetris request corpus + open-loop arrival traces for the interference study.

Corpus: decision states from several generators (greedy, beam search, and logged LLM play), split
into development and held-out episode seeds. Each corpus entry is one request with a defined task
(pick a legal placement id) and an oracle Q-table for scoring whatever the model answers.
  short  stateless raw-track prompt (system rules + current state), ~1k tokens
  long   the same kind of decision with `history_turns` prior turns (append memory), ~10-16k tokens
Messages are rebuilt deterministically from the stored states and verified by sha256, so the corpus
stays small and every condition replays byte-identical tasks.

Trace: seeded Poisson arrivals of shorts and longs (+ periodic zero-cost path probes) over a fixed
duration. The same trace is replayed under every condition of a block (paired design).

  python -m llm.mixed_workload corpus --out runs/explore/interference/corpus_dev --split dev
  python -m llm.mixed_workload trace --corpus runs/explore/interference/corpus_dev --seed 11 ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import zlib
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List

import pandas as pd

from llm.cache_replay import frozen_messages
from llm.oracle import action_values, env_from_state

SPLITS = {"dev": [3000, 3001, 3002, 3003], "heldout": [4000, 4001, 4002, 4003]}
LLM_LOG = "runs/explore/iter06/steps.csv"  # historical LLM-visited states (dev only; seeds 1000-1002)


def task_sha256(messages: List[Dict[str, str]]) -> str:
    return hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()


def generator_states(split: str, pieces: int) -> pd.DataFrame:
    from llm.reference_regret import bot_steps

    frames = [bot_steps(gen, seed, pieces) for gen in ("greedy", "beam") for seed in SPLITS[split]]
    if split == "dev" and os.path.exists(LLM_LOG):
        log = pd.read_csv(LLM_LOG)
        log = log[log["arm"].isin(["gpt-oss-120b/low", "gemma-4-31B/direct", "Kimi-K2.7/direct"])]
        frames.append(log[["arm", "episode_seed", "turn", "board", "curr", "next", "used_id"]])
    states = pd.concat(frames, ignore_index=True)
    return states.drop_duplicates(["arm", "episode_seed", "turn"]).reset_index(drop=True)


def _q_table(args):
    idx, board, curr, nxt = args
    seed = zlib.crc32(f"{board}|{curr}|{nxt}".encode()) % 1_000_000
    q = action_values(env_from_state(board, curr, nxt), n_samples=2, depth=4, beam_h=2, beam_w=4,
                      beam_samples=1, seed=seed)
    return idx, {int(a): float(v) for a, v in zip(q["action_id"], q["q_beam"])}


def build_corpus(out_dir: str, split: str, pieces: int, n_short: int, n_long: int, history_turns: int,
                 seed: int, workers: int) -> Dict:
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"fresh directory required: {out}")
    out.mkdir(parents=True)
    states = generator_states(split, pieces)
    states.to_csv(out / "states.csv", index=False)
    rng = random.Random(seed)
    long_pool = list(states.index[states["turn"] >= history_turns])
    short_pool = list(states.index)
    rng.shuffle(long_pool)
    rng.shuffle(short_pool)
    picks = [("long", i) for i in long_pool[:n_long]] + [("short", i) for i in short_pool[:n_short]]
    if len([p for p in picks if p[0] == "long"]) < n_long:
        raise ValueError("not enough long trajectories for the requested history length")
    with Pool(workers) as pool:
        qtabs = dict(pool.map(_q_table, [(i, states.loc[i, "board"], states.loc[i, "curr"], states.loc[i, "next"])
                                         for i in sorted({i for _, i in picks})]))
    entries = []
    for n, (kind, i) in enumerate(picks):
        h = history_turns if kind == "long" else 0
        messages, legal = frozen_messages(states, i, h, annotated=False)
        q = qtabs[i]
        best = max(q.values())
        entries.append({"rid": f"{kind[0]}{n:04d}", "kind": kind, "state_index": int(i), "history_turns": h,
                        "source_arm": states.loc[i, "arm"], "episode_seed": int(states.loc[i, "episode_seed"]),
                        "turn": int(states.loc[i, "turn"]), "legal_ids": sorted(int(a) for a in legal),
                        "bytes": len(json.dumps(messages).encode()), "task_sha256": task_sha256(messages),
                        "q_beam": {str(a): v for a, v in q.items()}, "q_best": best})
    with open(out / "corpus.jsonl", "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    manifest = {"split": split, "split_seeds": SPLITS[split], "pieces": pieces, "history_turns_long": history_turns,
                "n_short": n_short, "n_long": n_long, "sample_seed": seed, "generators": sorted(states["arm"].unique()),
                "states_sha256": hashlib.sha256((out / "states.csv").read_bytes()).hexdigest(),
                "corpus_sha256": hashlib.sha256((out / "corpus.jsonl").read_bytes()).hexdigest(),
                "prompt_track": "raw (no simulator annotations), stateless short / append-history long",
                "oracle": "llm.oracle q_beam (beam_h=2, beam_w=4, content-seeded); approximate, not ground truth"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def load_corpus(corpus_dir: str):
    d = Path(corpus_dir)
    states = pd.read_csv(d / "states.csv")
    entries = [json.loads(l) for l in open(d / "corpus.jsonl", encoding="utf-8")]
    return states, {e["rid"]: e for e in entries}


def request_messages(states: pd.DataFrame, entry: Dict) -> List[Dict[str, str]]:
    messages, _ = frozen_messages(states, entry["state_index"], entry["history_turns"], annotated=False)
    if task_sha256(messages) != entry["task_sha256"]:
        raise ValueError(f"corpus drift: rebuilt messages for {entry['rid']} do not match the stored hash")
    return messages


def make_trace(entries: Dict[str, Dict], duration_s: float, rate_short: float, rate_long: float, seed: int,
               ping_every_s: float = 0.0) -> Dict:
    """Seeded Poisson arrivals; requests drawn without replacement per kind within a trace."""
    rng = random.Random(seed)
    items = []
    for kind, rate in (("short", rate_short), ("long", rate_long)):
        pool = sorted(r for r, e in entries.items() if e["kind"] == kind)
        rng.shuffle(pool)
        t = rng.expovariate(rate) if rate > 0 else duration_s + 1
        k = 0
        while t < duration_s:
            if k >= len(pool):
                raise ValueError(f"corpus has too few {kind} requests for this trace")
            items.append({"rid": pool[k], "kind": kind, "offset_s": round(t, 4)})
            k += 1
            t += rng.expovariate(rate)
    if ping_every_s > 0:
        t = ping_every_s / 2
        n = 0
        while t < duration_s:
            items.append({"rid": f"ping{n:03d}", "kind": "ping", "offset_s": round(t, 4)})
            n += 1
            t += ping_every_s
    items = [{"rid": rid, "kind": kind, "offset_s": off}
             for off, rid, kind in sorted((i["offset_s"], i["rid"], i["kind"]) for i in items)]
    trace = {"trace_seed": seed, "duration_s": duration_s, "rate_short": rate_short, "rate_long": rate_long,
             "ping_every_s": ping_every_s, "items": items}
    trace["trace_sha256"] = hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest()
    return trace


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("corpus")
    c.add_argument("--out", required=True)
    c.add_argument("--split", default="dev", choices=sorted(SPLITS))
    c.add_argument("--pieces", type=int, default=100)
    c.add_argument("--n_short", type=int, default=160)
    c.add_argument("--n_long", type=int, default=40)
    c.add_argument("--history_turns", type=int, default=48)
    c.add_argument("--seed", type=int, default=7)
    c.add_argument("--workers", type=int, default=3)
    t = sub.add_parser("trace")
    t.add_argument("--corpus", required=True)
    t.add_argument("--duration_s", type=float, default=40)
    t.add_argument("--rate_short", type=float, default=1.0)
    t.add_argument("--rate_long", type=float, default=0.15)
    t.add_argument("--ping_every_s", type=float, default=2.0)
    t.add_argument("--seed", type=int, required=True)
    t.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.cmd == "corpus":
        print(json.dumps(build_corpus(args.out, args.split, args.pieces, args.n_short, args.n_long,
                                      args.history_turns, args.seed, args.workers), indent=2))
    else:
        _, entries = load_corpus(args.corpus)
        trace = make_trace(entries, args.duration_s, args.rate_short, args.rate_long, args.seed, args.ping_every_s)
        Path(args.out).write_text(json.dumps(trace, indent=2))
        print(len(trace["items"]), "items", trace["trace_sha256"][:12])


if __name__ == "__main__":
    main()
