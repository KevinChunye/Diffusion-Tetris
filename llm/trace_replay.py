"""trace_replay.py

Open-loop replay of a frozen mixed trace against an endpoint under one admission condition.

- Arrivals are released at their scheduled offsets whether or not earlier requests finished
  (no coordinated omission). The client queue, admission policy and concurrency cap are the only
  things that differ between conditions.
- Every request gets a fresh namespace at position 0 of the system prompt, so no condition can
  benefit from another's KV cache and every long request costs a real (fresh) prefill. Reported
  cached_tokens is logged to check this.
- No retries: a failure is a measured outcome. Per-request timeout, bounded client queue
  (backpressure), wall-clock stop and a price-card spend reservation (byte-length upper bound on
  prompt tokens, fail-closed when usage is missing) bound the run. Reservations are estimates,
  not invoices.
- Monotonic timestamps (s since run start): scheduled arrival, release into the client queue,
  admission (dispatch), first SSE chunk, first output delta, first valid legal action (complete
  JSON object parsed from streamed content), completion. Time-to-valid-action (TTVA) runs from the
  scheduled arrival, so client queueing counts.
- Probe requests (kind "ping": GET /models) measure the network/gateway path under the same load.
"""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from llm.admission import Condition, Item
from llm.cache_replay import with_namespace
from llm.mixed_workload import request_messages
from llm.prompts import parse_action

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def first_valid_time(events: List, legal: List[int]) -> Optional[float]:
    """Earliest event time at which the accumulated content is a complete JSON object whose
    action_id is legal. Partial prefixes like '{"action_id": 1' never count (they may become 12)."""
    text = ""
    for t, delta in events or []:
        text += delta
        candidate = _FENCE.sub("", text.strip()).strip()
        if not candidate.endswith("}"):
            continue
        try:
            obj = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("action_id"), int) and obj["action_id"] in legal:
            return float(t)
    return None


def reserve_usd(price: Dict, prompt_bytes: int, max_tokens: int) -> float:
    """Upper bound: byte-level BPE tokenizers emit <= 1 token per UTF-8 byte; +2048 for templates."""
    if not price or price.get("input") is None or price.get("output") is None:
        raise ValueError("missing price card: refusing to spend (fail-closed)")
    p_in = max(float(price["input"]), float(price.get("cached") or 0.0))
    return ((prompt_bytes + 2048) * p_in + max_tokens * float(price["output"])) / 1e6


def replay(trace: Dict, states, entries: Dict[str, Dict], condition: Condition, client, cfg: Dict, out_dir: str,
           run_meta: Dict[str, Any], clock: Callable[[], float] = time.perf_counter) -> List[Dict[str, Any]]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / "requests.jsonl"
    policy = condition.policy
    model = cfg["model"]
    price = client.pricing.get(model)
    budget = float(cfg["budget_usd"])
    run_id = run_meta["run_id"]
    items = [i for i in trace["items"] if i["kind"] in condition.kinds]
    cv = threading.Condition()
    state = {"spent": 0.0, "reserved": 0.0, "stop": "", "n_done": 0}
    waiting: List[Item] = []
    inflight: Dict[str, Item] = {}
    rows: List[Dict[str, Any]] = []
    t0 = clock()

    def record(row: Dict[str, Any]) -> None:
        row.update(run_meta)
        row["condition"] = condition.name
        rows.append(row)
        with open(rows_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")

    def unsent(spec: Dict, status: str) -> None:
        record({"rid": spec["rid"], "kind": spec["kind"], "arrival_s": spec["offset_s"], "status": status})

    def worker(item: Item, admit_s: float, reserve: float, messages, legal, max_tokens, extra, audit) -> None:
        row: Dict[str, Any] = {"rid": item.rid, "kind": item.kind, "arrival_s": item.arrival_s,
                               "release_s": item.release_s, "admit_s": admit_s, "reserved_usd": reserve}
        row.update(audit)
        try:
            if item.kind == "ping":
                a = clock()
                client.list_models()
                row.update(status="ok", complete_s=clock() - t0, ping_s=clock() - a)
                cost = 0.0
            else:
                body_extra = dict(cfg.get("chat_kwargs", {}).get("extra_body") or {})
                body_extra.update(extra)
                kw = {k: v for k, v in cfg.get("chat_kwargs", {}).items() if k != "extra_body"}
                res = client.chat(model, messages, max_tokens=max_tokens, stream=True, temperature=cfg.get("temperature", 0),
                                  response_format=cfg.get("response_format"), extra_body=body_extra or None,
                                  record_events=True, meta={"rid": item.rid, "kind": item.kind, "run_id": run_id,
                                                            "condition": condition.name}, **kw)
                done = clock() - t0
                base = (res.t0_perf - t0) if res.t0_perf is not None else admit_s
                tv = first_valid_time(res.content_events, legal) if res.ok else None
                proposed = parse_action(res.text, legal) if res.ok else None
                known = bool(res.cost_estimate_known)
                cost = res.cost_usd if known else reserve  # fail-closed when usage is missing
                row.update(status="ok" if res.ok else "error", http_status=res.http_status, error=res.error[:200],
                           first_chunk_s=None if res.ttft_s is None else base + res.ttft_s,
                           first_output_s=None if res.ttft_token_s is None else base + res.ttft_token_s,
                           valid_action_s=None if tv is None else base + tv, complete_s=done,
                           proposed_id=proposed, legal=proposed in legal, finish_reason=res.finish_reason,
                           prompt_tokens=res.prompt_tokens, completion_tokens=res.completion_tokens,
                           cached_tokens=res.cached_tokens, cache_usage_reported=res.cache_usage_reported,
                           usage_reported=res.usage_reported, cost_usd=cost, cost_estimate_known=known,
                           n_content_events=len(res.content_events or []), reply=res.text[:60])
        except Exception as exc:  # never lose a dispatched request from the accounting
            cost = reserve
            row.update(status="exception", error=f"{type(exc).__name__}: {str(exc)[:160]}", complete_s=clock() - t0,
                       cost_usd=cost, cost_estimate_known=False)
        with cv:
            state["reserved"] -= reserve
            state["spent"] += cost
            inflight.pop(item.rid, None)
            record(row)
            cv.notify_all()

    idx = 0
    max_wall = float(cfg.get("max_wall_s", trace["duration_s"] + 180))
    with cv:
        while True:
            now = clock() - t0
            while idx < len(items) and items[idx]["offset_s"] <= now:
                spec = items[idx]
                idx += 1
                if state["stop"]:
                    unsent(spec, f"not_sent_{state['stop']}")
                elif len(waiting) >= int(cfg.get("max_waiting", 200)) and spec["kind"] != "ping":
                    unsent(spec, "rejected_backpressure")
                else:
                    waiting.append(Item(spec["rid"], spec["kind"], spec["offset_s"], now))
            if now > max_wall and not state["stop"]:
                state["stop"] = "wall"
            if state["stop"]:
                # Account for everything that will never be sent, then wait only for in-flight work.
                for it in waiting:
                    unsent({"rid": it.rid, "kind": it.kind, "offset_s": it.arrival_s}, f"not_sent_{state['stop']}")
                waiting.clear()
                for spec in items[idx:]:
                    unsent(spec, f"not_sent_{state['stop']}")
                idx = len(items)
            else:
                for it in policy.select(now, list(waiting), list(inflight.values())):
                    extra: Dict[str, Any] = {}
                    audit: Dict[str, Any] = {}
                    messages, legal, max_tokens, reserve = None, [], 0, 0.0
                    if it.kind != "ping":
                        entry = entries[it.rid]
                        namespace = uuid.uuid4().hex
                        messages = with_namespace(request_messages(states, entry), namespace)
                        legal = entry["legal_ids"]
                        max_tokens = int(cfg["max_tokens_long" if it.kind == "long" else "max_tokens_short"])
                        prompt_bytes = len(json.dumps(messages).encode())
                        reserve = reserve_usd(price, prompt_bytes, max_tokens)
                        audit = {"task_sha256": entry["task_sha256"], "namespace": namespace,
                                 "prompt_bytes": prompt_bytes, "max_tokens": max_tokens}
                        if state["spent"] + state["reserved"] + reserve > budget:
                            state["stop"] = "budget"
                            break
                        if policy.priority_hints:
                            extra["priority"] = int(policy.priority_hints[it.kind])
                    waiting.remove(it)
                    inflight[it.rid] = it
                    state["reserved"] += reserve
                    threading.Thread(target=worker, args=(it, now, reserve, messages, legal, max_tokens, extra, audit),
                                     daemon=True).start()
            if idx >= len(items) and not waiting and not inflight:
                break
            wait = items[idx]["offset_s"] - (clock() - t0) if idx < len(items) else 0.25
            cv.wait(timeout=max(0.0005, min(wait, 0.25)))
    summary = {"run_id": run_id, "condition": condition.name, "stop": state["stop"] or "completed",
               "estimated_cost_usd": state["spent"], "n_rows": len(rows), "n_items": len(items),
               "wall_s": clock() - t0}
    with open(out / "runs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(summary, **{k: v for k, v in run_meta.items() if k != "run_id"})) + "\n")
    return rows
