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


def test_decision_model_plays_legal_moves_and_logs_every_option(tmp_path):
    """A fake decision model (lowest-hole option wins) plays through the bridge; every call is logged raw."""
    import json
    from types import SimpleNamespace

    from llm.decision_model import play

    class Fake:
        def system_one(self, state, questions):
            crit = questions["move"]["criteria"]
            assert "board" in state and "piece" in state
            score = {k: (0 if "no new holes" in v else -1) + (1 if "clears 1" in v else 0) for k, v in crit.items()}
            z = sum(2.0 ** s for s in score.values())
            probs = {k: 2.0 ** s / z for k, s in score.items()}
            best = max(probs, key=probs.get)
            ans = SimpleNamespace(choice=best, confidence=probs[best], probabilities=probs)
            return SimpleNamespace(choices={"move": ans}, usage=SimpleNamespace(input_tokens=123))

    log = tmp_path / "calls.jsonl"
    rows, ep = play(Fake(), seed=1000, pieces=12, arm="fake/decision", calls_log=log)
    assert len(rows) == ep["pieces"] == 12
    assert all(r["used_id"] in json.loads(r["legal_ids"]) for r in rows)
    recs = [json.loads(l) for l in log.read_text().splitlines()]
    assert len(recs) == 12
    assert abs(sum(o["p"] for o in recs[0]["options"].values()) - 1) < 1e-9
    assert len(recs[0]["options"]) == rows[0]["n_legal"]


def test_jev_backend_against_fake_typesafe_server(tmp_path, monkeypatch):
    """The Jev client speaks POST /v1/systemone with a Bearer key, retries a 429, maps answers to the runner's
    shape, and never writes the key into any log."""
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from llm.decision_model import JevClient, play

    seen = {"auth": set(), "n": 0}

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            seen["n"] += 1
            seen["auth"].add(self.headers.get("Authorization"))
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert self.path == "/v1/systemone" and req["model"] == "jev-1.13.0"
            if seen["n"] == 2:  # one rate-limit response to exercise the retry
                self.send_response(429); self.end_headers(); return
            crit = req["questions"]["move"]["criteria"]
            keys = list(crit)
            probs = {k: (0.5 if i == 0 else 0.5 / max(1, len(keys) - 1)) for i, k in enumerate(keys)}
            body = {"model": "jev-1.13.0", "answers": {"move": {"type": "choice", "choice": keys[0], "confidence": 0.31,
                                                                "probabilities": probs}},
                    "usage": {"input_tokens": 777, "output_tokens": 5}}
            data = json.dumps(body).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("TYPESAFE_API_KEY", "sk-test-secret")
    monkeypatch.setenv("TYPESAFE_BASE_URL", f"http://127.0.0.1:{srv.server_port}")
    monkeypatch.setattr("time.sleep", lambda s: None)
    client = JevClient(model="jev-1.13.0")
    log = tmp_path / "calls.jsonl"
    rows, ep = play(client, seed=1000, pieces=6, arm="jev/decision", calls_log=log, model_label="jev-1.13.0", tag="jev")
    srv.shutdown()
    assert seen["auth"] == {"Bearer sk-test-secret"}
    assert len(rows) == 6 and rows[0]["input_tokens"] == 777
    assert abs(rows[0]["top_prob"] - 0.5) < 1e-9 and rows[0]["confidence"] == 0.31  # confidence kept apart from p
    assert "sk-test-secret" not in log.read_text()
    assert json.loads(log.read_text().splitlines()[0])["served_model"] == "jev-1.13.0"
