"""endpoint_sim.py — SIMULATED evidence only.

Discrete-event model of a hosted endpoint, used to validate the admission policies and the replay
accounting offline and to size the live pilot. It is NOT a model of Tensormesh's actual
deployment. Its parameters are assumptions loosely inspired by historical API timings.

Server model (per replica, vLLM-V1-like):
  - one engine step at a time; per-step token budget `step_tokens` (chunked prefill)
  - each step schedules running decodes first (1 token each), then continues running prefills,
    then admits waiting requests FCFS into the remaining budget (`max_seqs` cap)
  - step duration = step_overhead_s + prefill_tokens / prefill_tok_s + n_decodes * decode_s_per_seq
  - first output token at the end of the step that completes a request's prefill; the answer
    (a short JSON action) is valid when all `output_tokens` are produced
Requests route to the replica with the least outstanding tokens. Network adds rtt_s/2 each way.
The client side uses the same `llm.admission` policies and the same row schema as the live replayer.
"""

from __future__ import annotations

import heapq
import itertools
import random
from dataclasses import dataclass, field
from typing import Dict, List

from llm.admission import Condition, Item


@dataclass
class EndpointProfile:
    name: str
    replicas: int = 1
    step_tokens: int = 8192
    max_seqs: int = 64
    step_overhead_s: float = 0.012
    prefill_tok_s: float = 20000.0
    decode_s_per_seq: float = 0.0004
    rtt_s: float = 0.35
    rtt_jitter_s: float = 0.05


@dataclass
class _Req:
    rid: str
    kind: str
    prompt: int
    output: int
    prefill_left: int = 0
    out_left: int = 0
    first_token_t: float = None
    done_t: float = None


@dataclass
class _Replica:
    waiting: List[_Req] = field(default_factory=list)
    running: List[_Req] = field(default_factory=list)
    busy: bool = False

    def outstanding(self) -> int:
        return sum(r.prefill_left + r.out_left for r in self.waiting + self.running)


def simulate(trace: Dict, tokens: Dict[str, Dict[str, int]], condition: Condition, profile: EndpointProfile,
             seed: int = 0, run_meta: Dict = None) -> List[Dict]:
    """tokens[rid] = {"prompt": P, "output": O}. Returns rows with the live replayer's schema."""
    rng = random.Random(seed)
    policy = condition.policy
    items = [i for i in trace["items"] if i["kind"] in condition.kinds]
    replicas = [_Replica() for _ in range(profile.replicas)]
    events: List = []
    seq = itertools.count()
    waiting: List[Item] = []
    inflight: Dict[str, Item] = {}
    rows: Dict[str, Dict] = {}

    def push(t, kind, payload):
        heapq.heappush(events, (t, next(seq), kind, payload))

    def half_rtt():
        return max(0.0, profile.rtt_s / 2 + rng.uniform(-profile.rtt_jitter_s, profile.rtt_jitter_s) / 2)

    def start_step(ri: int, now: float):
        rep = replicas[ri]
        if rep.busy or not (rep.waiting or rep.running):
            return
        budget = profile.step_tokens
        decodes = [r for r in rep.running if r.prefill_left == 0 and r.out_left > 0]
        budget -= len(decodes)
        plan = []  # (req, prefill tokens this step)
        for r in [r for r in rep.running if r.prefill_left > 0]:
            if budget <= 0:
                break
            take = min(budget, r.prefill_left)
            plan.append((r, take))
            budget -= take
        while rep.waiting and budget > 0 and len(rep.running) < profile.max_seqs:
            r = rep.waiting.pop(0)
            rep.running.append(r)
            take = min(budget, r.prefill_left)
            plan.append((r, take))
            budget -= take
        prefill_tokens = sum(t for _, t in plan)
        dur = profile.step_overhead_s + prefill_tokens / profile.prefill_tok_s + len(decodes) * profile.decode_s_per_seq
        rep.busy = True
        push(now + dur, "step_end", (ri, decodes, plan))

    def dispatch(now: float):
        for it in policy.select(now, list(waiting), list(inflight.values())):
            waiting.remove(it)
            inflight[it.rid] = it
            rows[it.rid]["admit_s"] = now
            if it.kind == "ping":
                rows[it.rid]["ping_s"] = 2 * half_rtt()
                push(now + rows[it.rid]["ping_s"], "client_done", it.rid)
                continue
            push(now + half_rtt(), "server_arrive", it.rid)

    for i in items:
        push(i["offset_s"], "arrive", i)
    while events:
        now, _, kind, payload = heapq.heappop(events)
        if kind == "arrive":
            rows[payload["rid"]] = {"rid": payload["rid"], "kind": payload["kind"], "arrival_s": payload["offset_s"],
                                    "release_s": now, "status": "ok"}
            waiting.append(Item(payload["rid"], payload["kind"], payload["offset_s"], now))
            dispatch(now)
        elif kind == "server_arrive":
            tk = tokens[payload]
            req = _Req(payload, rows[payload]["kind"], tk["prompt"], tk["output"], tk["prompt"], tk["output"])
            ri = min(range(len(replicas)), key=lambda k: (replicas[k].outstanding(), k))
            replicas[ri].waiting.append(req)
            rows[payload]["_req"] = req
            start_step(ri, now)
        elif kind == "step_end":
            ri, decodes, plan = payload
            rep = replicas[ri]
            for r in decodes:
                r.out_left -= 1
            for r, take in plan:
                r.prefill_left -= take
                if r.prefill_left == 0:
                    r.out_left -= 1  # the step that finishes prefill emits the first token
                    r.first_token_t = now
            for r in [r for r in rep.running if r.prefill_left == 0 and r.out_left <= 0]:
                rep.running.remove(r)
                r.done_t = now
                push(now + half_rtt(), "client_done", r.rid)
            rep.busy = False
            start_step(ri, now)
        elif kind == "client_done":
            row = rows[payload]
            row["complete_s"] = now
            req = row.pop("_req", None)
            if req is not None:
                lag = now - req.done_t
                row["first_output_s"] = req.first_token_t + lag
                row["first_chunk_s"] = row["first_output_s"]
                row["valid_action_s"] = now  # short JSON answer is actionable when complete
                row.update(prompt_tokens=req.prompt, completion_tokens=req.output, legal=True)
            inflight.pop(payload, None)
            dispatch(now)
    out = []
    for r in rows.values():
        r.update(run_meta or {})
        r["condition"] = condition.name
        r["simulated"] = True
        out.append(r)
    return out
