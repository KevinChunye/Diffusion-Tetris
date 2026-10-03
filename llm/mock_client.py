"""mock_client.py

Offline stand-in for TensormeshClient (same chat() signature, no network) for dry runs and tests.
It fakes a global prefix cache over rendered prompts so cache accounting can be checked end to end:
cached tokens = longest common prefix with any earlier prompt, in ~4-char tokens, 16-token blocks.
It plays a fixed rule: the listed placement with the lowest resulting height, then fewest holes.
"""

from __future__ import annotations

import json
import re
import threading
import time
from typing import Any, Dict, List, Optional

from llm.tensormesh_client import CallResult, call_cost_usd, load_pricing

_ROW = re.compile(r"^(\d+): rot (\d+), cols (\d+)-(\d+)(?: -> lines (\d+), height (\d+), holes (\d+))?$", re.MULTILINE)


class MockClient:
    def __init__(self, log_path: Optional[str] = None, illegal_every: int = 0, block: int = 16):
        self.pricing = load_pricing()
        self.log_path = log_path
        self.illegal_every = int(illegal_every)
        self.block = int(block)
        self._prompts: List[str] = []
        self._lock = threading.Lock()
        self.total_cost_usd = 0.0
        self.n_calls = 0

    def _cached_tokens(self, prompt: str) -> int:
        best = 0
        for p in self._prompts[-256:]:
            n = min(len(p), len(prompt))
            i = 0
            while i < n and p[i] == prompt[i]:
                i += 1
            best = max(best, i)
        return (best // 4) // self.block * self.block

    def chat(self, model: str, messages: List[Dict[str, str]], max_tokens: int = 64, stream: bool = True,
             reasoning_effort: Optional[str] = "low", response_format: Optional[Dict[str, Any]] = None,
             temperature: Optional[float] = None, seed: Optional[int] = None,
             extra_body: Optional[Dict[str, Any]] = None, meta: Optional[Dict[str, Any]] = None) -> CallResult:
        prompt = "".join(f"<|{m['role']}|>{m['content']}" for m in messages)
        with self._lock:
            cached = self._cached_tokens(prompt)
            self._prompts.append(prompt)
            self.n_calls += 1
            k = self.n_calls
        prompt_tokens = max(1, len(prompt) // 4)
        cached = min(cached, prompt_tokens - 1)
        rows = _ROW.findall(messages[-1]["content"])
        if rows:
            best = min((int(r[5] or 0), int(r[6] or 0), int(r[0])) for r in rows)
            aid = best[-1]
        else:
            aid = 0
        if self.illegal_every and k % self.illegal_every == 0:
            aid = 999
        text = json.dumps({"action_id": aid})
        ttft = 0.001 + 2e-6 * (prompt_tokens - cached)
        res = CallResult(model=model, ok=True, text=text, finish_reason="stop", http_status=200, stream=stream,
                         prompt_tokens=prompt_tokens, completion_tokens=8, cached_tokens=cached,
                         ttft_s=ttft, ttft_token_s=ttft, latency_s=ttft + 0.001, max_tokens=max_tokens,
                         prompt_chars=sum(len(m["content"]) for m in messages), t_start=time.time(), meta=dict(meta or {}))
        res.cost_usd = call_cost_usd(self.pricing.get(model), res.prompt_tokens, res.cached_tokens, res.completion_tokens)
        res.usage_reported = res.cache_usage_reported = True
        res.cost_estimate_known = bool(self.pricing.get(model))
        res.end_to_end_s = res.latency_s
        with self._lock:
            self.total_cost_usd += res.cost_usd
        if self.log_path:
            rec = {"event": "call"}
            rec.update(res.to_row())
            with self._lock, open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, default=str) + "\n")
        return res
