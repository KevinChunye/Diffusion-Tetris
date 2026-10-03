"""admission.py

Client-side admission policies for a mixed stream of short decision requests and long-context
requests sent to a black-box endpoint. Policies are pure: given the time, the waiting queue and what
is in flight, they return what to dispatch now. The live replayer (llm/trace_replay.py) and the
offline simulator (llm/endpoint_sim.py) call exactly the same objects.

  fifo        one queue in arrival order, at most `concurrency` requests in flight (client baseline)
  defer_long  shorts first; at most `max_long_inflight` longs in flight; a long is held while any short
              is waiting or in flight, unless it has waited >= `max_defer_s` (starvation bound).
              Known idea (length-aware deferral, cf. arXiv 2604.06970 / HyGen); here it is the
              candidate whose endpoint-level effect is measured, not a claimed new algorithm.

Probe requests (kind == "ping") bypass admission and never count against the cap.
Priority hints (vLLM-style `priority`, lower = earlier) are a request attribute, not a policy:
a policy may carry `priority_hints` that the runner adds to request bodies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Item:
    rid: str
    kind: str                 # "short" | "long" | "ping"
    arrival_s: float          # scheduled arrival offset
    release_s: float          # when it actually entered the client queue
    est_tokens: int = 0


@dataclass
class Policy:
    name: str
    concurrency: int = 8
    priority_hints: Optional[Dict[str, int]] = None

    def select(self, now: float, waiting: List[Item], inflight: List[Item]) -> List[Item]:
        raise NotImplementedError


@dataclass
class FIFO(Policy):
    name: str = "fifo"

    def select(self, now: float, waiting: List[Item], inflight: List[Item]) -> List[Item]:
        load = sum(1 for i in inflight if i.kind != "ping")
        out = []
        for item in waiting:  # waiting is kept in release (arrival) order
            if item.kind == "ping":
                out.append(item)
                continue
            if load >= self.concurrency:
                break  # strict FIFO: no overtaking
            out.append(item)
            load += 1
        return out


@dataclass
class DeferLong(Policy):
    name: str = "defer_long"
    max_long_inflight: int = 1
    max_defer_s: float = 6.0

    def select(self, now: float, waiting: List[Item], inflight: List[Item]) -> List[Item]:
        load = sum(1 for i in inflight if i.kind != "ping")
        longs_inflight = sum(1 for i in inflight if i.kind == "long")
        shorts_inflight = sum(1 for i in inflight if i.kind == "short")
        out = [i for i in waiting if i.kind == "ping"]
        shorts = [i for i in waiting if i.kind == "short"]
        longs = [i for i in waiting if i.kind == "long"]
        for item in shorts:
            if load >= self.concurrency:
                break
            out.append(item)
            load += 1
            shorts_inflight += 1
        shorts_waiting = len(shorts) - sum(1 for i in out if i.kind == "short")
        for item in longs:
            if load >= self.concurrency or longs_inflight >= self.max_long_inflight:
                break
            aged = now - item.release_s >= self.max_defer_s
            if (shorts_inflight or shorts_waiting) and not aged:
                break  # keep FIFO order among longs
            out.append(item)
            load += 1
            longs_inflight += 1
        return out


def make_policy(spec: Dict) -> Policy:
    spec = dict(spec)
    kind = spec.pop("policy")
    if kind == "fifo":
        return FIFO(**spec)
    if kind == "defer_long":
        return DeferLong(**spec)
    raise ValueError(f"unknown policy {kind}")


@dataclass
class Condition:
    """One arm of the experiment: a policy plus which request kinds of the trace it receives."""
    name: str
    policy: Policy
    kinds: List[str] = field(default_factory=lambda: ["short", "long", "ping"])
