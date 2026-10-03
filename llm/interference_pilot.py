"""interference_pilot.py

Drives the interference study from one config (configs/interference/pilot.yaml):

  sim        SIMULATED endpoints (llm/endpoint_sim.py); no network, no spend
  calibrate  LIVE: a few isolated fresh requests per model, to learn real prompt-token counts and
             unloaded latency before the pilot (charged to the same budget)
  live       LIVE pilot: per model and block, one seeded trace replayed under every condition in a
             randomized order, with a cooldown between replays

Every mode writes into a fresh run directory: config.yaml, manifest.json (git commit, dirty flag,
config/corpus/trace hashes, mock/simulated/live marker), blocks.jsonl (trace hash and condition
order per block), requests.jsonl (one row per trace item, appended as requests finish) and runs.jsonl.

  python -m llm.interference_pilot sim --out runs/explore/interference/sim_pilot
  python -m llm.interference_pilot calibrate --out runs/explore/interference/calibration
  python -m llm.interference_pilot live --out runs/explore/interference/live_pilot
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml

from llm.admission import Condition, make_policy
from llm.cache_replay import with_namespace
from llm.endpoint_sim import EndpointProfile, simulate
from llm.mixed_workload import load_corpus, make_trace, request_messages
from llm.run_pilot import SPEND_CSV, TOTAL_BUDGET_USD, record_spend, spent_so_far
from llm.trace_replay import first_valid_time, replay, reserve_usd

DEFAULT_CONFIG = "configs/interference/pilot.yaml"


def load_config(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def conditions(cfg: Dict[str, Any]) -> Dict[str, Condition]:
    out = {}
    for name, spec in cfg["conditions"].items():
        spec = dict(spec)
        kinds = spec.pop("kinds", ["short", "long", "ping"])
        policy = make_policy(spec)
        policy.name = name
        out[name] = Condition(name, policy, kinds)
    return out


def _git() -> Dict[str, Any]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--", "llm", "configs"], text=True).strip())
    return {"git_commit": commit, "git_dirty_llm_or_configs": dirty}


def fresh_run_dir(out: str, cfg: Dict[str, Any], mode: str, mock: bool = False) -> Path:
    d = Path(out)
    if d.exists() and any(d.iterdir()):
        raise FileExistsError(f"fresh directory required (prior artifacts are preserved): {d}")
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    corpus_manifest = json.loads((Path(cfg["corpus"]) / "manifest.json").read_text())
    manifest = {"mode": mode, "live": mode in ("live", "calibrate") and not mock, "simulated": mode == "sim", "mock": mock,
                "config_sha256": hashlib.sha256(yaml.safe_dump(cfg, sort_keys=True).encode()).hexdigest(),
                "corpus": cfg["corpus"], "corpus_sha256": corpus_manifest["corpus_sha256"],
                "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **_git(),
                "ttva_definition": "valid_action_s - arrival_s: scheduled arrival to the first streamed content "
                                   "prefix that parses as a complete JSON object with a legal action_id"}
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return d


def block_plan(cfg: Dict[str, Any], entries: Dict[str, Dict], models: List[str]) -> List[Dict[str, Any]]:
    """Blocks interleave models (block k of every model before block k+1) so slow drift hits all
    models alike; condition order inside each (model, block) is shuffled with a recorded seed."""
    plan = []
    names = list(cfg["conditions"])
    for k in range(int(cfg["blocks"])):
        seed = int(cfg["block_seed_base"]) + k
        trace = make_trace(entries, seed=seed, **cfg["trace"])
        rng = random.Random(seed)
        for model in rng.sample(models, len(models)):
            order = rng.sample(names, len(names))
            plan.append({"block": k, "trace_seed": seed, "model": model, "order": order, "trace": trace})
    return plan


def _slug(model: str) -> str:
    return model.split("/")[-1]


def run_sim(cfg: Dict[str, Any], out: str) -> Path:
    d = fresh_run_dir(out, cfg, "sim")
    _, entries = load_corpus(cfg["corpus"])
    sim = cfg["sim"]
    tokens = {rid: {"prompt": int(e["bytes"] / float(sim["bytes_per_token"])), "output": int(sim["output_tokens"][e["kind"]])}
              for rid, e in entries.items()}
    conds = conditions(cfg)
    profiles = {name: EndpointProfile(name, **p) for name, p in sim["profiles"].items()}
    with open(d / "requests.jsonl", "a", encoding="utf-8") as rows_f, open(d / "blocks.jsonl", "a") as blocks_f:
        for b in block_plan(cfg, entries, list(profiles)):
            blocks_f.write(json.dumps({k: v for k, v in b.items() if k != "trace"} |
                                      {"trace_sha256": b["trace"]["trace_sha256"]}) + "\n")
            for name in b["order"]:
                meta = {"run_id": f"{b['model']}_b{b['block']}_{name}", "model": b["model"], "block": b["block"],
                        "trace_seed": b["trace_seed"], "trace_sha256": b["trace"]["trace_sha256"], "simulated": True}
                for r in simulate(b["trace"], tokens, conds[name], profiles[b["model"]], seed=b["trace_seed"], run_meta=meta):
                    rows_f.write(json.dumps(r) + "\n")
    return d


def _client(cfg: Dict[str, Any], d: Path):
    from llm.tensormesh_client import TensormeshClient

    return TensormeshClient(log_path=str(d / "calls.jsonl"), verbose_retries=False, **cfg.get("client", {}))


def _cap(cfg: Dict[str, Any]) -> float:
    """The study budget covers calibration and every live replay of this iteration, not each call."""
    this_study = 0.0
    if os.path.exists(SPEND_CSV):
        ledger = pd.read_csv(SPEND_CSV)
        this_study = float(ledger.loc[ledger["iteration"].astype(str) == str(cfg["iteration"]), "cost_usd"].sum())
    return min(float(cfg["budget_usd"]) - this_study, TOTAL_BUDGET_USD - spent_so_far())


def run_calibrate(cfg: Dict[str, Any], out: str, models: List[str] = None) -> Path:
    """Isolated, sequential fresh requests: unloaded latency and real prompt-token counts per kind."""
    d = fresh_run_dir(out, cfg, "calibrate")
    states, entries = load_corpus(cfg["corpus"])
    client = _client(cfg, d)
    cal = cfg["calibration"]
    rng = random.Random(int(cal["seed"]))
    picks = []
    for kind, n in (("short", cal["n_short"]), ("long", cal["n_long"])):
        pool = sorted(r for r, e in entries.items() if e["kind"] == kind)
        picks += rng.sample(pool, int(n))
    cap, spent = _cap(cfg), 0.0
    try:
        with open(d / "requests.jsonl", "a", encoding="utf-8") as f:
            for model in models or list(cfg["models"]):
                mcfg = cfg["models"][model]
                for rid in picks:
                    e = entries[rid]
                    namespace = uuid.uuid4().hex
                    messages = with_namespace(request_messages(states, e), namespace)
                    max_tokens = int(mcfg["max_tokens_long" if e["kind"] == "long" else "max_tokens_short"])
                    prompt_bytes = len(json.dumps(messages).encode())
                    reserve = reserve_usd(client.pricing.get(model), prompt_bytes, max_tokens)
                    if spent + reserve > cap:
                        raise SystemExit(f"calibration would exceed the cap (${spent:.4f} spent of ${cap:.2f})")
                    res = client.chat(model, messages, max_tokens=max_tokens, stream=True, record_events=True,
                                      temperature=cfg["defaults"]["temperature"],
                                      response_format=cfg["defaults"]["response_format"],
                                      meta={"rid": rid, "kind": e["kind"]}, **mcfg.get("chat_kwargs", {}))
                    spent += res.cost_usd if res.cost_estimate_known else reserve
                    tv = first_valid_time(res.content_events, e["legal_ids"]) if res.ok else None
                    f.write(json.dumps({"model": model, "rid": rid, "kind": e["kind"], "ok": res.ok,
                                        "http_status": res.http_status, "error": res.error[:200],
                                        "prompt_bytes": prompt_bytes, "prompt_tokens": res.prompt_tokens,
                                        "completion_tokens": res.completion_tokens, "cached_tokens": res.cached_tokens,
                                        "cache_usage_reported": res.cache_usage_reported, "ttft_s": res.ttft_s,
                                        "ttft_token_s": res.ttft_token_s, "valid_action_s": tv,
                                        "latency_s": res.latency_s, "cost_usd": res.cost_usd,
                                        "cost_estimate_known": res.cost_estimate_known, "reply": res.text[:80],
                                        "live": True}) + "\n")
                    f.flush()
                    time.sleep(1.0)
    finally:
        record_spend(cfg["iteration"], f"interference calibration ({d.name})", spent,
                     len(picks) * len(models or cfg["models"]))
    return d


def accounted_spend(d: Path) -> float:
    """Estimated spend of a run directory from its files alone: recorded request costs, plus the full
    reservation of every dispatched request that never recorded a row (in flight at a crash/interrupt)."""
    rows = [json.loads(l) for l in open(d / "requests.jsonl")] if (d / "requests.jsonl").exists() else []
    dispatched = [json.loads(l) for l in open(d / "dispatch.jsonl")] if (d / "dispatch.jsonl").exists() else []
    done = {(r.get("run_id"), r["rid"]) for r in rows}
    unfinished = sum(float(x["reserved_usd"]) for x in dispatched if (x["run_id"], x["rid"]) not in done)
    return sum(float(r.get("cost_usd") or 0.0) for r in rows) + unfinished


def run_live(cfg: Dict[str, Any], out: str, models: List[str] = None, client=None) -> Path:
    """`client` is injectable for offline tests; an injected client marks the run as mock."""
    mock = client is not None
    d = fresh_run_dir(out, cfg, "live", mock=mock)
    states, entries = load_corpus(cfg["corpus"])
    client = client or _client(cfg, d)
    conds = conditions(cfg)
    models = models or list(cfg["models"])
    cap, spent, n_calls = (float(cfg["budget_usd"]) if mock else _cap(cfg)), 0.0, 0
    stopped = ""
    try:
        with open(d / "blocks.jsonl", "a", encoding="utf-8") as blocks_f:
            for b in block_plan(cfg, entries, models):
                blocks_f.write(json.dumps({k: v for k, v in b.items() if k != "trace"} |
                                          {"trace_sha256": b["trace"]["trace_sha256"],
                                           "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}) + "\n")
                blocks_f.flush()
                mcfg = cfg["models"][b["model"]]
                for name in b["order"]:
                    rcfg = {"model": b["model"], "budget_usd": cap - spent, "max_wall_s": cfg["max_wall_s"],
                            "max_waiting": cfg["max_waiting"], "max_tokens_short": mcfg["max_tokens_short"],
                            "max_tokens_long": mcfg["max_tokens_long"], "temperature": cfg["defaults"]["temperature"],
                            "response_format": cfg["defaults"]["response_format"],
                            "chat_kwargs": mcfg.get("chat_kwargs", {})}
                    meta = {"run_id": f"{_slug(b['model'])}_b{b['block']}_{name}", "model": b["model"],
                            "block": b["block"], "trace_seed": b["trace_seed"],
                            "trace_sha256": b["trace"]["trace_sha256"], "live": not mock, "mock": mock}
                    rows = replay(b["trace"], states, entries, conds[name], client, rcfg, str(d), meta)
                    spent += sum(float(r.get("cost_usd") or 0.0) for r in rows)
                    n_calls += sum(1 for r in rows if r["kind"] != "ping" and r["status"] in ("ok", "error", "exception"))
                    print(f"[{meta['run_id']}] rows={len(rows)} est_spent=${spent:.4f}", flush=True)
                    if any(r["status"] == "not_sent_budget" for r in rows):
                        stopped = "budget"
                        break
                    time.sleep(float(cfg["cooldown_s"]))
                if stopped:
                    break
    except BaseException:
        stopped = "interrupted"
        raise
    finally:
        spent = max(spent, accounted_spend(d))
        if not mock:
            record_spend(cfg["iteration"], f"interference live pilot ({d.name})", spent, n_calls)
        (d / "spend.json").write_text(json.dumps({"estimated_cost_usd": spent, "n_calls": n_calls,
                                                  "cap_usd": cap, "stopped": stopped or "completed"}, indent=2))
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["sim", "calibrate", "live"])
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="*", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.mode == "sim":
        print(run_sim(cfg, args.out))
    elif args.mode == "calibrate":
        print(run_calibrate(cfg, args.out, args.models))
    else:
        print(run_live(cfg, args.out, args.models))


if __name__ == "__main__":
    main()
