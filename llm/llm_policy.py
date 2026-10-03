"""llm_policy.py

LLM Tetris agent: each turn the model picks one legal placement id.

History policies (what the prompt carries besides the static system prompt):
  stateless : [system, state_t]
  append    : [system, state_0, reply_0, ..., state_t]        prefix only grows      -> cache friendly
  window    : [system, last W (state, reply) pairs, state_t]  prefix shifts each turn -> cache hostile
  compact   : append-only inside blocks of N turns; at each block boundary the history is replaced by a
              short summary + current state                     -> sawtooth prefix
Illegal / unparseable / late replies are logged and replaced by the fallback placement (safe_step).
Decision quality is scored offline by llm/oracle.py from the logged states.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from TetrisGym_updated import TetrisGym
from agent.value_dqn import board_props
from llm.prompts import parse_action, render_state, render_summary, system_prompt
from llm.tetris_tools import placement_outcomes, safe_step


@dataclass
class PolicyCfg:
    model: str = "openai/gpt-oss-20b"
    history: str = "stateless"          # stateless | append | window | compact
    window: int = 8                     # turns kept by `window`
    compact_every: int = 16             # block length for `compact`
    annotated: bool = True              # list lines/height/holes per legal placement
    layout: str = "static_first"        # static_first | state_first (rules after the board)
    max_tokens: int = 512
    response_format: str = "none"       # none | json_object | json_schema (enum of legal ids)
    temperature: Optional[float] = None
    idle_s: float = 0.0                 # sleep before each call (cache-lifetime experiments)
    deadline_ms: Optional[float] = None  # reply slower than this -> fallback action
    chat_kwargs: Dict[str, Any] = field(default_factory=dict)  # per-model reasoning settings


def board_to_str(board) -> str:
    return "/".join("".join("#" if v else "." for v in row) for row in board)


class LLMPolicy:
    def __init__(self, client, cfg: PolicyCfg):
        self.client = client
        self.cfg = cfg
        self.system = system_prompt(annotated=cfg.annotated, json_output=True)
        self.reset()

    def reset(self) -> None:
        self.history: List[Dict[str, str]] = []
        self.pending_summary: Optional[str] = None
        self.block_start_stats: Optional[Dict[str, int]] = None

    # -- prompt assembly ----------------------------------------------------------------------
    def _context(self) -> List[Dict[str, str]]:
        if self.cfg.history == "stateless":
            return []
        if self.cfg.history == "window":
            return self.history[-2 * int(self.cfg.window):] if self.cfg.window > 0 else []
        if self.cfg.history in ("append", "compact"):
            return self.history
        raise ValueError(f"unknown history policy {self.cfg.history}")

    def messages_for(self, state_text: str) -> List[Dict[str, str]]:
        user = state_text if self.pending_summary is None else f"{self.pending_summary}\n\n{state_text}"
        if self.cfg.layout == "state_first":
            return self._context() + [{"role": "user", "content": f"{user}\n\n{self.system}"}]
        return [{"role": "system", "content": self.system}] + self._context() + [{"role": "user", "content": user}]

    def _maybe_compact(self, turn: int, stats: Dict[str, int]) -> None:
        if self.cfg.history != "compact":
            return
        if self.block_start_stats is None:
            self.block_start_stats = dict(stats, turn=turn)
            return
        n = int(self.cfg.compact_every)
        if n > 0 and turn > 0 and turn % n == 0:
            b = self.block_start_stats
            self.pending_summary = render_summary(b["turn"], turn - 1, turn - b["turn"], stats["lines"] - b["lines"],
                                                  stats["score"] - b["score"], b["max_height"], stats["max_height"],
                                                  b["holes"], stats["holes"])
            self.history = []
            self.block_start_stats = dict(stats, turn=turn)

    # -- one decision -------------------------------------------------------------------------
    def act(self, env: TetrisGym, turn: int, stats: Dict[str, int], meta: Dict[str, Any]):
        outcomes = placement_outcomes(env)
        _, rotations = env.game.current_piece
        legal = [{"action_id": aid, "rot": o.rot, "x": o.x, "w": rotations[o.rot].shape[1], "lines": o.lines,
                  "max_height": o.max_height, "holes": o.holes} for aid, o in sorted(outcomes.items())]
        legal_ids = [a["action_id"] for a in legal]
        self._maybe_compact(turn, stats)
        state_text = render_state(turn, env.game.board, env.game.current_piece[0], env.game.next_piece[0],
                                  stats["score"], stats["lines"], legal, annotated=self.cfg.annotated)
        msgs = self.messages_for(state_text)
        rf = None
        if self.cfg.response_format == "json_object":
            rf = {"type": "json_object"}
        elif self.cfg.response_format == "json_schema":
            rf = {"type": "json_schema", "json_schema": {"name": "placement", "strict": True, "schema": {
                "type": "object", "properties": {"action_id": {"type": "integer", "enum": legal_ids}},
                "required": ["action_id"], "additionalProperties": False}}}
        if self.cfg.idle_s > 0:
            time.sleep(self.cfg.idle_s)
        n_hist = sum(1 for m in msgs if m["role"] == "assistant")
        res = self.client.chat(self.cfg.model, msgs, max_tokens=self.cfg.max_tokens, stream=True, response_format=rf,
                               temperature=self.cfg.temperature, meta=dict(meta, turn=turn, history_turns=n_hist),
                               **self.cfg.chat_kwargs)
        proposed = parse_action(res.text, legal_ids) if res.ok else None
        late = self.cfg.deadline_ms is not None and res.end_to_end_s * 1000.0 > float(self.cfg.deadline_ms)
        user_msg = msgs[-1]["content"]
        return proposed, res, outcomes, user_msg, late, n_hist

    def record(self, user_msg: str, used_action_id: int) -> None:
        """Append the turn to history with the action actually executed (keeps history consistent)."""
        if self.cfg.layout == "state_first":
            user_msg = user_msg.split(f"\n\n{self.system}")[0]
        self.history.append({"role": "user", "content": user_msg})
        self.history.append({"role": "assistant", "content": json.dumps({"action_id": int(used_action_id)})})
        self.pending_summary = None


def run_episode(policy: LLMPolicy, episode_seed: int, max_pieces: int, meta: Dict[str, Any],
                step_rows: List[Dict[str, Any]], should_stop=lambda: False) -> Dict[str, Any]:
    env = TetrisGym(max_steps=max_pieces)
    env.reset(seed=episode_seed)
    policy.reset()
    lines = 0
    done = False
    turn = 0
    stop_reason = ""
    while not done and turn < max_pieces:
        if should_stop():
            stop_reason = "budget"
            break
        if not env.get_valid_action_ids():
            break
        board_before = board_to_str(env.game.board)
        curr, nxt = env.game.current_piece[0], env.game.next_piece[0]
        feats = board_props(env.game.board.astype("uint8"))
        stats = {"score": int(env.game.score), "lines": lines, "max_height": int(feats[1]), "holes": int(feats[6])}
        proposed, res, outcomes, user_msg, late, n_hist = policy.act(env, turn, stats, meta)
        _, done, info, used, reason = safe_step(env, None if late else proposed)
        if late:
            reason = "late"
        elif not res.ok:
            reason = "api_error"
        lines += int(info["lines_cleared"])
        policy.record(user_msg, used)
        row = dict(meta)
        row.update({
            "episode_seed": episode_seed, "turn": turn, "board": board_before, "curr": curr, "next": nxt,
            "legal_ids": json.dumps(sorted(outcomes.keys())), "proposed_id": proposed, "used_id": used,
            "fallback_reason": reason, "lines_cleared": int(info["lines_cleared"]), "score_after": int(env.game.score),
            "done": bool(done), "history_turns": n_hist, "user_chars": len(user_msg),
            "ok": res.ok, "error": res.error[:200], "http_status": res.http_status, "retries": res.retries,
            "prompt_tokens": res.prompt_tokens, "cached_tokens": res.cached_tokens,
            "created_cache_tokens": res.created_cache_tokens, "completion_tokens": res.completion_tokens,
            "ttft_s": res.ttft_s, "ttft_token_s": res.ttft_token_s, "latency_s": res.latency_s,
            "cost_usd": res.cost_usd, "prompt_chars": res.prompt_chars, "t_start": res.t_start,
            "end_to_end_s": res.end_to_end_s, "usage_reported": res.usage_reported,
            "cache_usage_reported": res.cache_usage_reported, "cost_estimate_known": res.cost_estimate_known,
            "model_action_accepted": not bool(reason),
            "reply": res.text[:80], "reasoning_chars": len(res.reasoning), "finish_reason": res.finish_reason,
            "max_height_before": stats["max_height"], "holes_before": stats["holes"],
        })
        step_rows.append(row)
        turn += 1
    ep = dict(meta)
    ep.update({"episode_seed": episode_seed, "score": float(env.game.score), "lines_cleared": lines,
               "pieces_placed": turn, "topped_out": bool(env.game.game_over), "stop_reason": stop_reason})
    return ep


def cfg_to_dict(cfg: PolicyCfg) -> Dict[str, Any]:
    return asdict(cfg)
