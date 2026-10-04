import json

import pytest

from llm.tensormesh_client import CallResult


class CacheClient:
    """Fake endpoint that keeps at most `capacity` prefixes: later warm-ups evict the oldest."""

    def __init__(self, capacity=3):
        self.pricing = {"m": {"input": 0.1, "output": 0.2, "cached": 0.0}}
        self.capacity, self.cache, self.calls = capacity, [], 0
        import threading
        self.lock = threading.Lock()

    def chat(self, model, messages, max_tokens=1, stream=False, meta=None, **kw):
        key = messages[0]["content"]
        with self.lock:
            self.calls += 1
            hit = key in self.cache
            if not hit:
                self.cache.append(key)
                self.cache = self.cache[-self.capacity:]
        prompt = 1000 + 100 * (meta["call"] == "probe")
        r = CallResult(model=model, ok=True, http_status=200, prompt_tokens=prompt, completion_tokens=1,
                       cached_tokens=1000 if (hit and meta["call"] == "probe") else 0, usage_reported=True,
                       cache_usage_reported=True, cost_estimate_known=True, latency_s=0.01, t_start=0.0)
        r.cost_usd = (prompt - r.cached_tokens) * 0.1e-6
        return r


def test_capacity_sweep_accounts_every_agent_and_detects_eviction(tmp_path):
    from llm.capacity_sweep import run

    cfg = {"iteration": 99, "name": "t", "history_turns": 2, "gap_s": 0, "cooldown_s": 0, "loads": [1, 2, 8],
           "parallel_lanes": 1, "max_inflight": 8, "seed": 1, "budget_usd": 1.0, "models": [{"id": "m", "reps": 2}]}
    import time
    client = CacheClient(capacity=3)
    d = run(cfg, str(tmp_path / "cap"), client=client, sleep=lambda s: time.sleep(0.05))  # all warm-ups land first
    rows = [json.loads(l) for l in open(d / "rows.jsonl")]
    assert len(rows) == 2 * (1 + 2 + 8) and client.calls == 2 * len(rows)
    hit = {(r["load"]): [] for r in rows}
    for r in rows:
        hit[r["load"]].append(r["probe_cached_tokens"] > 0)
    assert all(hit[1]) and all(hit[2])          # fits in the cache
    assert sum(hit[8]) < len(hit[8])            # 8 warm-ups overflow a 3-slot cache before the probes
    assert len({r["nonce"] for r in rows}) == len(rows)
    spend = json.loads((d / "spend.json").read_text())
    assert spend["stopped"] == "completed" and spend["calls"] == len(rows) * 2


def test_capacity_sweep_budget_is_fail_closed(tmp_path):
    from llm.capacity_sweep import round_reservation, run

    with pytest.raises(ValueError):
        round_reservation({}, 1000, 2)
    cfg = {"iteration": 99, "name": "t", "history_turns": 2, "gap_s": 0, "cooldown_s": 0, "loads": [1, 64],
           "parallel_lanes": 1, "max_inflight": 8, "seed": 3, "budget_usd": 0.0005, "models": [{"id": "m", "reps": 1}]}
    d = run(cfg, str(tmp_path / "cap"), client=CacheClient(), sleep=lambda s: None)
    spend = json.loads((d / "spend.json").read_text())
    assert spend["stopped"] == "budget" and spend["estimated_cost_usd"] <= 0.0005


def test_capacity_sweep_resume_skips_done_rounds_and_adds_extra(tmp_path):
    from llm.capacity_sweep import plan, run

    base = {"iteration": 99, "name": "t", "history_turns": 2, "gap_s": 0, "cooldown_s": 0, "loads": [1, 2, 32],
            "parallel_lanes": 2, "max_inflight": 64, "exclusive_load": 32, "seed": 1, "budget_usd": 1.0,
            "models": [{"id": "m", "reps": 1}]}
    d = run(base, str(tmp_path / "a"), client=CacheClient(capacity=64), sleep=lambda s: None)
    first = [json.loads(l) for l in open(d / "rows.jsonl")]
    resumed = dict(base, resume_from=str(d), models=[{"id": "m", "reps": 2}], extra=[{"model": "m", "rep": 100, "load": 2}])
    rounds = plan(resumed)[0]["rounds"]
    done = {(r["rep"], r["load"]) for r in first}
    assert all((r["rep"], r["load"]) not in done for r in rounds)
    assert {(r["rep"], r["load"]) for r in rounds} == {(1, 1), (1, 2), (1, 32), (100, 2)}
