import json
import threading
import time

import pandas as pd
import pytest

from llm.admission import FIFO, Condition, DeferLong, Item, make_policy
from llm.endpoint_sim import EndpointProfile, simulate
from llm.mixed_workload import make_trace, task_sha256
from llm.tensormesh_client import CallResult
from llm.trace_replay import first_valid_time, replay, reserve_usd

PRICE = {"m": {"input": 0.1, "output": 0.2, "cached": 0.0}}


def _entries(n_short=12, n_long=4):
    entries = {}
    for k in range(n_short):
        entries[f"s{k:04d}"] = {"rid": f"s{k:04d}", "kind": "short", "legal_ids": [1, 2, 3], "history_turns": 0}
    for k in range(n_long):
        entries[f"l{k:04d}"] = {"rid": f"l{k:04d}", "kind": "long", "legal_ids": [1, 2, 3], "history_turns": 48}
    for e in entries.values():
        e["messages"] = [{"role": "system", "content": "rules"},
                         {"role": "user", "content": e["rid"] * (40 if e["kind"] == "long" else 1)}]
        e["task_sha256"] = task_sha256(e["messages"])
    return entries


@pytest.fixture
def fake_corpus(monkeypatch):
    entries = _entries()
    monkeypatch.setattr("llm.trace_replay.request_messages", lambda states, entry: entry["messages"])
    return entries


class FakeClient:
    """Streams a fixed reply after a kind-dependent delay. Records every request body it receives."""

    def __init__(self, delay=None, fail=(), boom=(), usage=True, reply='{"action_id": 2}'):
        self.pricing = dict(PRICE)
        self.delay = delay or {"short": 0.02, "long": 0.08}
        self.fail, self.boom, self.usage, self.reply = set(fail), set(boom), usage, reply
        self.calls, self.pings = [], 0
        self.lock = threading.Lock()

    def list_models(self):
        with self.lock:
            self.pings += 1
        return []

    def chat(self, model, messages, max_tokens=64, stream=True, record_events=False, meta=None, extra_body=None, **kw):
        rid, kind = meta["rid"], meta["kind"]
        with self.lock:
            self.calls.append({"rid": rid, "messages": messages, "extra_body": extra_body, "max_tokens": max_tokens})
        if rid in self.boom:
            raise RuntimeError("socket closed")
        res = CallResult(model=model, max_tokens=max_tokens, meta=dict(meta))
        res.t0_perf = time.perf_counter()
        res.content_events = [] if record_events else None
        time.sleep(self.delay[kind])
        if rid in self.fail:
            res.http_status, res.error = 503, "HTTP 503"
            return res
        half = len(self.reply) // 2
        for t, part in ((self.delay[kind] * 0.9, self.reply[:half]), (self.delay[kind], self.reply[half:])):
            if res.content_events is not None:
                res.content_events.append((t, part))
        res.ok, res.http_status, res.text = True, 200, self.reply
        res.ttft_s = res.ttft_token_s = self.delay[kind] * 0.9
        if self.usage:
            res.usage_reported = res.cache_usage_reported = res.cost_estimate_known = True
            res.prompt_tokens, res.completion_tokens = 100, 5
            res.cost_usd = 100 * 0.1e-6 + 5 * 0.2e-6
        return res


def _trace(entries, duration=1.0, rs=8.0, rl=2.0, seed=3, ping=0.25):
    return make_trace(entries, duration, rs, rl, seed, ping)


CFG = {"model": "m", "budget_usd": 1.0, "max_tokens_short": 32, "max_tokens_long": 32, "max_wall_s": 30}


def _run(trace, entries, cond, client, tmp_path, cfg=CFG, name="r"):
    return replay(trace, None, entries, cond, client, cfg, str(tmp_path / name), {"run_id": name, "mock": True})


# ---- metric definitions ----------------------------------------------------------------------

def test_first_valid_time_requires_complete_legal_json():
    assert first_valid_time([(0.1, '{"action_'), (0.2, 'id": 1'), (0.3, '2}')], [12]) == 0.3
    assert first_valid_time([(0.1, '{"action_id": 1'), (0.2, '2}')], [1]) is None  # 12 is not legal
    assert first_valid_time([(0.1, '```json\n{"action_id": 3}'), (0.4, '\n```')], [3]) == 0.1
    assert first_valid_time([(0.1, 'I pick 3')], [3]) is None
    assert first_valid_time([], [3]) is None and first_valid_time(None, [3]) is None


def test_reservation_is_an_upper_bound_and_fails_closed():
    assert reserve_usd({"input": 1.0, "output": 2.0, "cached": None}, 1000, 10) == pytest.approx((3048 + 20) / 1e6)
    with pytest.raises(ValueError):
        reserve_usd({"input": None, "output": 2.0}, 10, 10)
    with pytest.raises(ValueError):
        reserve_usd({}, 10, 10)


# ---- policies --------------------------------------------------------------------------------

def _items(spec):
    return [Item(rid, kind, t, t) for rid, kind, t in spec]


def test_fifo_is_strict_and_pings_bypass_the_cap():
    waiting = _items([("a", "long", 0), ("p", "ping", 0.1), ("b", "short", 0.2), ("c", "short", 0.3)])
    inflight = _items([("x", "short", 0)])
    out = FIFO(concurrency=2).select(1.0, waiting, inflight)
    assert [i.rid for i in out] == ["a", "p"]  # cap reached after "a"; "b" may not overtake; ping bypasses


def test_defer_long_holds_longs_until_shorts_drain_or_age():
    pol = DeferLong(concurrency=8, max_long_inflight=1, max_defer_s=5.0)
    waiting = _items([("l1", "long", 0.0), ("s1", "short", 0.5)])
    assert [i.rid for i in pol.select(1.0, waiting, [])] == ["s1"]
    assert pol.select(1.0, _items([("l1", "long", 0.0)]), _items([("s1", "short", 0.5)])) == []
    assert [i.rid for i in pol.select(1.0, _items([("l1", "long", 0.0)]), [])] == ["l1"]
    aged = pol.select(5.0, _items([("l1", "long", 0.0)]), _items([("s1", "short", 4.0)]))
    assert [i.rid for i in aged] == ["l1"]  # starvation bound
    one_at_a_time = pol.select(9.0, _items([("l2", "long", 0.0)]), _items([("l1", "long", 0.0)]))
    assert one_at_a_time == []


def test_defer_long_keeps_fifo_order_among_longs_and_respects_cap():
    pol = DeferLong(concurrency=2, max_long_inflight=2, max_defer_s=1.0)
    out = pol.select(10.0, _items([("l1", "long", 0.0), ("l2", "long", 9.5), ("s1", "short", 9.9)]),
                     _items([("s0", "short", 9.0)]))
    assert [i.rid for i in out] == ["s1"]  # cap 2 is full; l1 (aged) would be next, l2 may never jump l1
    assert make_policy({"policy": "defer_long", "max_defer_s": 3}).max_defer_s == 3


# ---- traces ----------------------------------------------------------------------------------

def test_trace_is_seeded_sorted_and_without_replacement():
    entries = _entries(40, 10)
    a, b = _trace(entries, 3.0, seed=5), _trace(entries, 3.0, seed=5)
    assert a["trace_sha256"] == b["trace_sha256"] and a["trace_sha256"] != _trace(entries, 3.0, seed=6)["trace_sha256"]
    offsets = [i["offset_s"] for i in a["items"]]
    assert offsets == sorted(offsets)
    rids = [i["rid"] for i in a["items"]]
    assert len(rids) == len(set(rids))
    with pytest.raises(ValueError):
        make_trace(_entries(2, 1), 10.0, 5.0, 0.0, 1)


# ---- replay accounting -----------------------------------------------------------------------

def test_every_trace_item_is_accounted_once_and_conditions_get_identical_tasks(fake_corpus, tmp_path):
    trace = _trace(fake_corpus)
    sent = {}
    for cond in (Condition("fifo", FIFO(concurrency=3)), Condition("defer", DeferLong(concurrency=3, max_defer_s=0.2)),
                 Condition("short_only", FIFO(concurrency=3), kinds=["short", "ping"])):
        client = FakeClient()
        rows = _run(trace, fake_corpus, cond, client, tmp_path, name=cond.name)
        expected = [i["rid"] for i in trace["items"] if i["kind"] in cond.kinds]
        assert sorted(r["rid"] for r in rows) == sorted(expected)
        assert all(r["status"] == "ok" for r in rows)
        logged = [json.loads(l) for l in open(tmp_path / cond.name / "requests.jsonl")]
        assert len(logged) == len(rows) and all(r["mock"] and r["condition"] == cond.name for r in logged)
        decisions = [r for r in rows if r["kind"] != "ping"]
        assert all(r["valid_action_s"] is not None and r["valid_action_s"] >= r["admit_s"] >= r["arrival_s"] - 0.05
                   for r in decisions)
        assert all(r["task_sha256"] == fake_corpus[r["rid"]]["task_sha256"] for r in decisions)
        sent[cond.name] = {c["rid"]: c for c in client.calls}
        namespaces = [c["messages"][0]["content"].split("\n")[0] for c in client.calls]
        assert len(set(namespaces)) == len(namespaces)  # a fresh namespace per request
        for c in client.calls:  # ... and nothing else differs from the frozen task
            stripped = [dict(m) for m in c["messages"]]
            stripped[0]["content"] = stripped[0]["content"].split("\n", 1)[1]
            assert task_sha256(stripped) == fake_corpus[c["rid"]]["task_sha256"]
        assert client.pings == sum(1 for i in trace["items"] if i["kind"] == "ping")
    assert set(sent["fifo"]) == set(sent["defer"])
    assert set(sent["short_only"]) == {r for r in sent["fifo"] if r.startswith("s")}


def test_failures_and_exceptions_are_measured_not_retried(fake_corpus, tmp_path):
    trace = _trace(fake_corpus)
    shorts = [i["rid"] for i in trace["items"] if i["kind"] == "short"]
    client = FakeClient(fail=shorts[:2], boom=shorts[2:3])
    rows = {r["rid"]: r for r in _run(trace, fake_corpus, Condition("fifo", FIFO()), client, tmp_path)}
    assert [rows[r]["status"] for r in shorts[:3]] == ["error", "error", "exception"]
    assert rows[shorts[0]]["valid_action_s"] is None and rows[shorts[0]]["http_status"] == 503
    assert sum(1 for c in client.calls if c["rid"] == shorts[0]) == 1
    assert rows[shorts[2]]["cost_usd"] == rows[shorts[2]]["reserved_usd"] > 0  # unknown cost charged at reservation


def test_budget_stop_is_fail_closed_and_accounts_unsent(fake_corpus, tmp_path):
    trace = _trace(fake_corpus)
    client = FakeClient(usage=False)  # missing usage: every call is charged its full reservation
    one = reserve_usd(PRICE["m"], 400, 32)
    cfg = dict(CFG, budget_usd=one * 3.5)
    rows = _run(trace, fake_corpus, Condition("fifo", FIFO()), client, tmp_path, cfg=cfg)
    assert len(rows) == len(trace["items"])
    spent = sum(r.get("cost_usd") or 0 for r in rows)
    assert 0 < spent <= cfg["budget_usd"]
    assert any(r["status"] == "not_sent_budget" for r in rows)
    summary = json.loads(open(tmp_path / "r" / "runs.jsonl").read())
    assert summary["stop"] == "budget" and summary["estimated_cost_usd"] == pytest.approx(spent)


def test_backpressure_rejects_instead_of_growing_the_queue(fake_corpus, tmp_path):
    trace = _trace(fake_corpus, rs=12.0)
    client = FakeClient(delay={"short": 0.3, "long": 0.3})
    rows = _run(trace, fake_corpus, Condition("fifo", FIFO(concurrency=1)), client, tmp_path,
                cfg=dict(CFG, max_waiting=2))
    assert len(rows) == len(trace["items"])
    assert any(r["status"] == "rejected_backpressure" for r in rows)


def test_priority_hints_are_sent_only_by_the_hinted_condition(fake_corpus, tmp_path):
    trace = _trace(fake_corpus)
    plain, hinted = FakeClient(), FakeClient()
    _run(trace, fake_corpus, Condition("fifo", FIFO()), plain, tmp_path, name="a")
    _run(trace, fake_corpus, Condition("fifo_prio", FIFO(priority_hints={"short": 0, "long": 10})), hinted, tmp_path,
         name="b")
    assert all(not c["extra_body"] for c in plain.calls)
    assert all(c["extra_body"]["priority"] == (10 if c["rid"].startswith("l") else 0) for c in hinted.calls)


# ---- simulator (SIMULATED evidence; checks conservation and the direction of the mechanism) ---

def _sim_inputs():
    entries = _entries(80, 20)
    trace = make_trace(entries, 30.0, 1.5, 0.3, 9, 2.0)
    tokens = {r: {"prompt": 15000 if e["kind"] == "long" else 900, "output": 12} for r, e in entries.items()}
    return trace, tokens


def test_simulator_conserves_requests_and_marks_rows_simulated():
    trace, tokens = _sim_inputs()
    prof = EndpointProfile("slow", prefill_tok_s=5000, step_tokens=8192)
    for cond in (Condition("fifo", FIFO()), Condition("defer", DeferLong()),
                 Condition("short_only", FIFO(), kinds=["short", "ping"])):
        rows = simulate(trace, tokens, cond, prof, seed=1, run_meta={"run_id": "t"})
        assert sorted(r["rid"] for r in rows) == sorted(i["rid"] for i in trace["items"] if i["kind"] in cond.kinds)
        assert all(r["simulated"] and r["complete_s"] >= r["arrival_s"] for r in rows)


def test_simulated_interference_and_mitigation_have_the_expected_direction():
    trace, tokens = _sim_inputs()
    prof = EndpointProfile("slow", prefill_tok_s=5000, step_tokens=8192)

    def p95(cond):
        rows = simulate(trace, tokens, cond, prof, seed=1)
        ttva = pd.Series([r["valid_action_s"] - r["arrival_s"] for r in rows if r["kind"] == "short"])
        return ttva.quantile(0.95)

    alone = p95(Condition("short_only", FIFO(), kinds=["short", "ping"]))
    fifo = p95(Condition("fifo", FIFO()))
    assert fifo > alone * 2  # a serialized slow prefill endpoint shows interference
    assert p95(Condition("defer", DeferLong())) <= fifo


# ---- end to end: live runner code path with an injected fake client, then the analysis -------

def test_live_runner_and_analysis_end_to_end_with_mock_client(tmp_path):
    from llm.interference_analysis import analyze
    from llm.interference_pilot import load_config, run_live

    cfg = load_config("configs/interference/pilot.yaml")
    cfg.update(blocks=2, cooldown_s=0, trace={"duration_s": 2.0, "rate_short": 6.0, "rate_long": 1.5, "ping_every_s": 0.5})
    cfg["models"] = {"m": {"max_tokens_short": 16, "max_tokens_long": 16, "chat_kwargs": {}}}
    cfg["analysis"]["bootstrap"]["n"] = 50
    client = FakeClient(delay={"short": 0.01, "long": 0.05})
    d = run_live(cfg, str(tmp_path / "live"), client=client)
    manifest = json.loads((d / "manifest.json").read_text())
    assert manifest["mock"] and not manifest["live"]
    blocks = [json.loads(l) for l in open(d / "blocks.jsonl")]
    assert len(blocks) == 2 and all(sorted(b["order"]) == sorted(cfg["conditions"]) for b in blocks)
    rows = [json.loads(l) for l in open(d / "requests.jsonl")]
    for b in blocks:
        for cond, spec in cfg["conditions"].items():
            got = [r for r in rows if r["block"] == b["block"] and r["condition"] == cond]
            kinds = spec.get("kinds", ["short", "long", "ping"])
            assert len(got) == len({r["rid"] for r in got}) > 0
            assert {r["kind"] for r in got} <= set(kinds)
    out = analyze(str(d))
    assert out["evidence"] == "MOCK" and set(out["verdicts"]["m"]) == {"H1", "H2", "B2"}
    pooled = pd.read_csv(d / "summary.csv")
    assert (pooled["legal_rate"] == 1.0).all()  # the fake always answers a legal id
    with pytest.raises(FileExistsError):
        run_live(cfg, str(d), client=client)


def test_interrupted_runs_charge_in_flight_requests_at_their_reservation(tmp_path):
    from llm.interference_pilot import accounted_spend

    (tmp_path / "requests.jsonl").write_text(json.dumps({"run_id": "a", "rid": "s1", "cost_usd": 0.001}) + "\n")
    (tmp_path / "dispatch.jsonl").write_text("".join(json.dumps(x) + "\n" for x in (
        {"run_id": "a", "rid": "s1", "reserved_usd": 0.01}, {"run_id": "a", "rid": "l1", "reserved_usd": 0.02},
        {"run_id": "b", "rid": "s1", "reserved_usd": 0.01})))
    assert accounted_spend(tmp_path) == pytest.approx(0.001 + 0.02 + 0.01)
