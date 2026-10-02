"""tensormesh_client.py

Minimal client for Tensormesh serverless (OpenAI-compatible chat completions).

Every call is logged as one JSON line with: model, prompt/completion/cached
tokens, TTFT, total latency, retries, $ cost (cached tokens at the cached price
when the price card lists one), plus caller metadata (turn index, arm, seed...).

TTFT definitions (streaming):
  ttft_s        time from request start to the first SSE data chunk. vLLM emits
                its first chunk (the role delta) only after prefill has produced
                the first token, so this is the prefill-dominated latency.
  ttft_token_s  time to the first non-empty reasoning/content delta.
Non-streaming calls cannot measure TTFT but return the per-tier cache split
(vLLM GPU prefix cache vs LMCache) in `kv_transfer_params`.
"""

from __future__ import annotations

import json
import os
import random
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import yaml

BASE_URL = "https://serverless.tensormesh.ai/v1"
RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524}
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRICING = REPO_ROOT / "configs" / "pricing.yaml"
DEFAULT_MODEL_SETTINGS = REPO_ROOT / "configs" / "model_settings.yaml"
ALLOWLIST_HINT = (
    "Tensormesh API unreachable through the egress proxy. Add serverless.tensormesh.ai "
    "to this environment's network allowlist."
)


class TensormeshUnreachable(RuntimeError):
    """Raised when the egress proxy blocks the host (403 from the proxy / not allowlisted)."""


def load_pricing(path: str | os.PathLike = DEFAULT_PRICING) -> Dict[str, Dict[str, Optional[float]]]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return dict(data.get("models", {}))


def load_model_settings(path: str | os.PathLike = DEFAULT_MODEL_SETTINGS) -> Dict[str, Dict[str, Any]]:
    """Per-model chat kwargs (reasoning_effort / extra_body) that keep reasoning low."""
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return dict(data.get("models", {}))


def call_cost_usd(price: Dict[str, Optional[float]] | None, prompt_tokens: int, cached_tokens: int, completion_tokens: int) -> float:
    """$ for one call. Cached tokens use the cached price if listed, else the input price."""
    if not price:
        return 0.0
    p_in = float(price.get("input") or 0.0)
    p_out = float(price.get("output") or 0.0)
    p_cached = price.get("cached")
    p_cached = p_in if p_cached is None else float(p_cached)
    uncached = max(0, int(prompt_tokens) - int(cached_tokens))
    return (uncached * p_in + int(cached_tokens) * p_cached + int(completion_tokens) * p_out) / 1e6


@dataclass
class CallResult:
    model: str
    ok: bool = False
    text: str = ""
    reasoning: str = ""
    finish_reason: str = ""
    http_status: int = 0
    error: str = ""
    stream: bool = True
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    created_cache_tokens: int = 0
    vllm_cached_tokens: Optional[int] = None
    lmcache_cached_tokens: Optional[int] = None
    ttft_s: Optional[float] = None
    ttft_token_s: Optional[float] = None
    latency_s: float = 0.0
    retries: int = 0
    cost_usd: float = 0.0
    prompt_chars: int = 0
    max_tokens: int = 0
    t_start: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> Dict[str, Any]:
        row = asdict(self)
        meta = row.pop("meta") or {}
        row.update({k: v for k, v in meta.items() if k not in row})
        return row


class TensormeshClient:
    """Plain-`requests` client with streaming TTFT, retries with backoff, and JSONL logging."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = BASE_URL,
        pricing_path: str | os.PathLike = DEFAULT_PRICING,
        log_path: Optional[str] = None,
        max_retries: int = 6,
        backoff_base_s: float = 1.0,
        backoff_cap_s: float = 60.0,
        timeout_s: float = 300.0,
        verbose_retries: bool = True,
    ):
        self.api_key = api_key if api_key is not None else os.environ.get("TENSORMESH_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("TENSORMESH_API_KEY is not set in the environment")
        self.base_url = base_url.rstrip("/")
        self.pricing = load_pricing(pricing_path)
        self.log_path = log_path
        self.max_retries = int(max_retries)
        self.backoff_base_s = float(backoff_base_s)
        self.backoff_cap_s = float(backoff_cap_s)
        self.timeout_s = float(timeout_s)
        self.verbose_retries = bool(verbose_retries)
        self._lock = threading.Lock()
        self._local = threading.local()
        self.total_cost_usd = 0.0
        if log_path:
            os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)

    # -- plumbing -----------------------------------------------------------
    def _session(self) -> requests.Session:
        s = getattr(self._local, "session", None)
        if s is None:
            s = requests.Session()
            s.headers.update({"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})
            self._local.session = s
        return s

    def _log(self, record: Dict[str, Any]) -> None:
        if not self.log_path:
            return
        line = json.dumps(record, default=str)
        with self._lock:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def _log_retry(self, model: str, attempt: int, status: int, error: str, sleep_s: float, meta: Dict[str, Any]) -> None:
        rec = {"event": "retry", "t": time.time(), "model": model, "attempt": attempt, "http_status": status,
               "error": error[:300], "sleep_s": round(sleep_s, 3)}
        rec.update({f"meta_{k}": v for k, v in (meta or {}).items()})
        self._log(rec)
        if self.verbose_retries:
            print(f"[tensormesh retry] model={model} attempt={attempt} status={status} sleep={sleep_s:.1f}s err={error[:120]}",
                  file=sys.stderr, flush=True)

    @staticmethod
    def _is_proxy_block(resp: requests.Response) -> bool:
        # The agent egress proxy answers 403 itself (no upstream JSON body) for non-allowlisted hosts.
        if resp.status_code not in (403, 407):
            return False
        body = (resp.text or "")[:500].lower()
        return ("proxy" in body) or ("allowlist" in body) or ("not allowed" in body) or ("blocked" in body)

    def price(self, model: str) -> Dict[str, Optional[float]]:
        return self.pricing.get(model, {})

    # -- API ----------------------------------------------------------------
    def list_models(self) -> List[Dict[str, Any]]:
        try:
            r = self._session().get(f"{self.base_url}/models", timeout=30)
        except requests.exceptions.ProxyError as exc:
            if any(code in str(exc) for code in ("403", "407", "Forbidden")):
                raise TensormeshUnreachable(f"{ALLOWLIST_HINT} ({exc})") from exc
            raise
        if self._is_proxy_block(r):
            raise TensormeshUnreachable(ALLOWLIST_HINT)
        r.raise_for_status()
        return list(r.json().get("data", []))

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 64,
        stream: bool = True,
        reasoning_effort: Optional[str] = "low",
        response_format: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> CallResult:
        body: Dict[str, Any] = {"model": model, "messages": messages, "max_tokens": int(max_tokens), "stream": bool(stream)}
        if stream:
            body["stream_options"] = {"include_usage": True}
        if reasoning_effort:
            body["reasoning_effort"] = reasoning_effort
        if response_format is not None:
            body["response_format"] = response_format
        if temperature is not None:
            body["temperature"] = float(temperature)
        if seed is not None:
            body["seed"] = int(seed)
        if extra_body:
            body.update(extra_body)

        meta = dict(meta or {})
        res = CallResult(model=model, stream=bool(stream), max_tokens=int(max_tokens), meta=meta,
                         prompt_chars=sum(len(m.get("content") or "") for m in messages))
        attempt = 0
        while True:
            res.t_start = time.time()
            t0 = time.perf_counter()
            status, err = 0, ""
            try:
                if stream:
                    status, err = self._do_stream(body, res, t0)
                else:
                    status, err = self._do_plain(body, res, t0)
            except TensormeshUnreachable:
                raise
            except requests.exceptions.ProxyError as exc:
                # Only a 403/407 answered by the egress proxy means "host not allowlisted"; other proxy
                # failures (relay tunnel closed mid-exchange, resets) are transient and retried.
                if any(code in str(exc) for code in ("403", "407", "Forbidden")):
                    raise TensormeshUnreachable(f"{ALLOWLIST_HINT} ({exc})") from exc
                status, err = -1, f"ProxyError: {exc}"
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout,
                    requests.exceptions.ChunkedEncodingError) as exc:
                status, err = -1, f"{type(exc).__name__}: {exc}"
            res.latency_s = time.perf_counter() - t0
            res.http_status = status
            if status == 200 and not err:
                res.ok = True
                break
            retriable = status == -1 or status in RETRY_STATUS
            if not retriable or attempt >= self.max_retries:
                res.ok = False
                res.error = err or f"HTTP {status}"
                break
            sleep_s = min(self.backoff_cap_s, self.backoff_base_s * (2 ** attempt)) * random.uniform(0.75, 1.25)
            attempt += 1
            res.retries = attempt
            self._log_retry(model, attempt, status, err, sleep_s, meta)
            time.sleep(sleep_s)

        res.cost_usd = call_cost_usd(self.price(model), res.prompt_tokens, res.cached_tokens, res.completion_tokens)
        with self._lock:
            self.total_cost_usd += res.cost_usd
        rec = {"event": "call"}
        rec.update(res.to_row())
        self._log(rec)
        return res

    def _read_usage(self, usage: Dict[str, Any] | None, res: CallResult) -> None:
        if not usage:
            return
        res.prompt_tokens = int(usage.get("prompt_tokens") or 0)
        res.completion_tokens = int(usage.get("completion_tokens") or 0)
        details = usage.get("prompt_tokens_details") or {}
        res.cached_tokens = int(details.get("cached_tokens") or 0)
        res.created_cache_tokens = int(details.get("created_cache_tokens") or 0)

    def _do_plain(self, body: Dict[str, Any], res: CallResult, t0: float) -> tuple[int, str]:
        r = self._session().post(f"{self.base_url}/chat/completions", json=body, timeout=self.timeout_s)
        if self._is_proxy_block(r):
            raise TensormeshUnreachable(ALLOWLIST_HINT)
        if r.status_code != 200:
            return r.status_code, (r.text or "")[:500]
        data = r.json()
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        res.text = msg.get("content") or ""
        res.reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
        res.finish_reason = choice.get("finish_reason") or ""
        self._read_usage(data.get("usage"), res)
        stats = ((data.get("kv_transfer_params") or {}).get("cached_token_stats")) or {}
        if stats:
            res.vllm_cached_tokens = int(stats.get("num_vllm_cached_tokens") or 0)
            res.lmcache_cached_tokens = int(stats.get("num_lmcache_cached_tokens") or 0)
        return 200, ""

    def _do_stream(self, body: Dict[str, Any], res: CallResult, t0: float) -> tuple[int, str]:
        with self._session().post(f"{self.base_url}/chat/completions", json=body, timeout=self.timeout_s, stream=True) as r:
            if self._is_proxy_block(r):
                raise TensormeshUnreachable(ALLOWLIST_HINT)
            if r.status_code != 200:
                return r.status_code, (r.text or "")[:500]
            text_parts: List[str] = []
            reasoning_parts: List[str] = []
            res.ttft_s = None
            res.ttft_token_s = None
            for raw in r.iter_lines(chunk_size=256):
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                now = time.perf_counter() - t0
                if res.ttft_s is None:
                    res.ttft_s = now
                chunk = json.loads(payload)
                if chunk.get("error"):
                    return 500, json.dumps(chunk.get("error"))[:500]
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    c = delta.get("content") or ""
                    rz = delta.get("reasoning") or delta.get("reasoning_content") or ""
                    if (c or rz) and res.ttft_token_s is None:
                        res.ttft_token_s = now
                    if c:
                        text_parts.append(c)
                    if rz:
                        reasoning_parts.append(rz)
                    if choice.get("finish_reason"):
                        res.finish_reason = choice["finish_reason"]
                if chunk.get("usage"):
                    self._read_usage(chunk["usage"], res)
            res.text = "".join(text_parts)
            res.reasoning = "".join(reasoning_parts)
        return 200, ""
