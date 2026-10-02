"""probe_tensormesh.py

Phase 0 platform probe for Tensormesh serverless:
  1. which models respond, context limits, streaming, response_format support
  2. what `usage` returns (cached tokens?) and how to turn reasoning down
  3. minimal prefix-cache test: same ~4k-token prefix twice, different suffixes
  4. rough concurrency tolerance

Usage:
  python -m llm.probe_tensormesh --out_dir runs/explore/probe [--only cache,thinking,format,concurrency]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import uuid
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from llm.tensormesh_client import TensormeshClient, load_model_settings

WORDS = (
    "board piece column row stack hole line clear drop rotate left right tall flat well spin "
    "block grid cell shape height score level speed queue hold preview tetris single double triple"
).split()

THINKING_VARIANTS = {
    "default": {"reasoning_effort": None, "extra_body": None},
    "effort_low": {"reasoning_effort": "low", "extra_body": None},
    "enable_thinking_false": {"reasoning_effort": None, "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
    "thinking_false": {"reasoning_effort": None, "extra_body": {"chat_template_kwargs": {"thinking": False}}},
}


def filler_text(n_words: int, seed: int) -> str:
    rng = random.Random(seed)
    lines = []
    for i in range(0, n_words, 16):
        lines.append(f"{i // 16:04d}: " + " ".join(rng.choice(WORDS) for _ in range(16)))
    return "\n".join(lines)


def probe_thinking(client: TensormeshClient, model: str) -> list[dict]:
    rows = []
    msgs = [{"role": "user", "content": "What is 17*23? Answer with just the number."}]
    for name, v in THINKING_VARIANTS.items():
        r = client.chat(model, msgs, max_tokens=1024, stream=True, reasoning_effort=v["reasoning_effort"],
                        extra_body=v["extra_body"], meta={"probe": "thinking", "variant": name})
        rows.append({"model": model, "variant": name, "ok": r.ok, "status": r.http_status,
                     "error": r.error[:160], "completion_tokens": r.completion_tokens,
                     "reasoning_chars": len(r.reasoning), "answer": r.text.strip()[:40],
                     "correct": "391" in r.text, "ttft_s": r.ttft_s, "latency_s": r.latency_s,
                     "prompt_tokens": r.prompt_tokens, "cached_tokens": r.cached_tokens})
    return rows


def probe_format(client: TensormeshClient, model: str, settings: dict) -> list[dict]:
    rows = []
    schema = {"type": "object", "properties": {"action_id": {"type": "integer"}},
              "required": ["action_id"], "additionalProperties": False}
    msgs = [{"role": "user", "content": 'Legal action ids: [3, 7, 12]. Pick one. Respond as JSON {"action_id": <int>}.'}]
    for name, rf in [("none", None), ("json_object", {"type": "json_object"}),
                     ("json_schema", {"type": "json_schema", "json_schema": {"name": "act", "schema": schema, "strict": True}})]:
        r = client.chat(model, msgs, max_tokens=1024, stream=True, response_format=rf,
                        meta={"probe": "format", "variant": name}, **settings)
        parsed = None
        try:
            parsed = json.loads(r.text.strip())
        except Exception:
            pass
        rows.append({"model": model, "response_format": name, "ok": r.ok, "status": r.http_status,
                     "error": r.error[:160], "valid_json": isinstance(parsed, dict) and "action_id" in parsed,
                     "text": r.text.strip()[:60], "completion_tokens": r.completion_tokens, "latency_s": r.latency_s})
    return rows


def probe_cache(client: TensormeshClient, model: str, settings: dict, trials: int, n_words: int) -> list[dict]:
    """Same ~4k-token prefix twice back-to-back with different suffixes, then a third non-streaming
    call for the per-tier split. A unique nonce at the start makes the first call a guaranteed miss."""
    rows = []
    for t in range(trials):
        nonce = uuid.uuid4().hex
        prefix = (f"Session {nonce}. Below is a log of a puzzle game. Read it, then answer the question at the end.\n"
                  + filler_text(n_words, seed=t))
        for k, (suffix, stream) in enumerate([("Q: how many lines start with 00? Answer with one word.", True),
                                               ("Q: what is the first word of line 0003? Answer with one word.", True),
                                               ("Q: what is the last word of line 0001? Answer with one word.", False)]):
            msgs = [{"role": "system", "content": prefix}, {"role": "user", "content": suffix}]
            r = client.chat(model, msgs, max_tokens=8, stream=stream,
                            meta={"probe": "cache", "trial": t, "call_idx": k}, **settings)
            rows.append({"model": model, "trial": t, "call_idx": k, "stream": stream, "ok": r.ok,
                         "status": r.http_status, "error": r.error[:160], "prompt_tokens": r.prompt_tokens,
                         "cached_tokens": r.cached_tokens, "created_cache_tokens": r.created_cache_tokens,
                         "vllm_cached": r.vllm_cached_tokens, "lmcache_cached": r.lmcache_cached_tokens,
                         "ttft_s": r.ttft_s, "latency_s": r.latency_s, "cost_usd": r.cost_usd})
    return rows


def probe_concurrency(client: TensormeshClient, model: str, settings: dict, levels: list[int]) -> list[dict]:
    rows = []
    for n in levels:
        def one(i: int):
            msgs = [{"role": "user", "content": f"Request {uuid.uuid4().hex[:8]}: reply with the word ok."}]
            return client.chat(model, msgs, max_tokens=16, stream=True, meta={"probe": "concurrency", "level": n, "i": i}, **settings)
        with ThreadPoolExecutor(max_workers=n) as ex:
            res = list(ex.map(one, range(n)))
        for i, r in enumerate(res):
            rows.append({"model": model, "concurrency": n, "i": i, "ok": r.ok, "status": r.http_status,
                         "retries": r.retries, "ttft_s": r.ttft_s, "latency_s": r.latency_s})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="runs/explore/probe")
    ap.add_argument("--only", default="thinking,format,cache,concurrency")
    ap.add_argument("--models", default="")
    ap.add_argument("--cache_trials", type=int, default=3)
    ap.add_argument("--cache_words", type=int, default=2600)
    ap.add_argument("--cache_tag", default="", help="suffix for cache csv (e.g. 16k)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    client = TensormeshClient(log_path=os.path.join(args.out_dir, "probe_calls.jsonl"))

    models = client.list_models()
    pd.DataFrame(models)[["id", "max_model_len"]].to_csv(os.path.join(args.out_dir, "models.csv"), index=False)
    ids = [m["id"] for m in models]
    if args.models:
        ids = [i for i in ids if any(s in i for s in args.models.split(","))]
    settings = load_model_settings()
    only = set(args.only.split(","))

    def run_model(mid: str) -> dict[str, list[dict]]:
        s = settings.get(mid, {"reasoning_effort": "low"})
        out: dict[str, list[dict]] = {}
        if "thinking" in only:
            out["thinking"] = probe_thinking(client, mid)
        if "format" in only:
            out["format"] = probe_format(client, mid, s)
        if "cache" in only:
            out["cache"] = probe_cache(client, mid, s, args.cache_trials, args.cache_words)
        return out

    with ThreadPoolExecutor(max_workers=len(ids)) as ex:
        results = list(ex.map(run_model, ids))
    for kind in ["thinking", "format", "cache"]:
        rows = [r for res in results for r in res.get(kind, [])]
        if rows:
            name = f"{kind}_{args.cache_tag}" if (kind == "cache" and args.cache_tag) else kind
            pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, f"{name}.csv"), index=False)
    if "concurrency" in only:
        mid = "openai/gpt-oss-20b"
        rows = probe_concurrency(client, mid, settings.get(mid, {"reasoning_effort": "low"}), [1, 4, 16, 32])
        pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "concurrency.csv"), index=False)
    print(f"total probe cost: ${client.total_cost_usd:.4f}")


if __name__ == "__main__":
    main()


def summarize(out_dir: str) -> str:
    """Markdown tables for notes/tensormesh_probe.md (pandas groupby only)."""
    parts = []
    models = pd.read_csv(os.path.join(out_dir, "models.csv"))
    pricing = pd.DataFrame.from_dict(__import__("llm.tensormesh_client", fromlist=["load_pricing"]).load_pricing(), orient="index")
    pricing.index.name = "id"
    m = models.merge(pricing.reset_index(), on="id", how="left")
    parts.append("### Models\n\n" + m.to_markdown(index=False))
    for tag in ["", "_16k"]:
        path = os.path.join(out_dir, f"cache{tag}.csv")
        if not os.path.exists(path):
            continue
        c = pd.read_csv(path)
        c = c[c["stream"]]
        c["phase"] = c["call_idx"].map({0: "cold", 1: "warm"})
        g = c.groupby(["model", "phase"]).agg(prompt=("prompt_tokens", "median"), cached=("cached_tokens", "median"),
                                              ttft=("ttft_s", "median"), cost=("cost_usd", "median")).unstack("phase")
        g.columns = [f"{a}_{b}" for a, b in g.columns]
        g["ttft_speedup"] = g["ttft_cold"] / g["ttft_warm"]
        g["cost_ratio_warm_over_cold"] = g["cost_warm"] / g["cost_cold"]
        g = g[["prompt_cold", "cached_warm", "ttft_cold", "ttft_warm", "ttft_speedup", "cost_cold", "cost_warm",
               "cost_ratio_warm_over_cold"]].sort_values("ttft_cold")
        parts.append(f"### Prefix-cache test{tag or ' (~3.5k tokens)'}\n\n" + g.round(4).to_markdown())
    path = os.path.join(out_dir, "concurrency.csv")
    if os.path.exists(path):
        k = pd.read_csv(path)
        g = k.groupby("concurrency").agg(ok_rate=("ok", "mean"), retries=("retries", "sum"), ttft_median=("ttft_s", "median"),
                                         ttft_max=("ttft_s", "max"), latency_median=("latency_s", "median"))
        parts.append("### Concurrency (gpt-oss-20b, tiny prompts)\n\n" + g.round(3).to_markdown())
    return "\n\n".join(parts)
