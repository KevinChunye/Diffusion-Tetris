"""bots.py

One interface for every Tetris bot in the repo, so any of them can be deployed, evaluated and recorded
the same way. A bot returns an action id (or None); harness.play applies it with safe_step, so an
illegal or unparseable choice is logged and replaced by the lowest-resulting-height placement.

  make_bot("random") | make_bot("greedy") | make_bot("beam") | make_bot("beam:3x16")
  make_bot("dqn:runs/<run>/checkpoint.pt")                 # trained with `python -m harness.train dqn`
  make_bot("diffusion:runs/<run>/checkpoints/ckpt.pt")     # trained with `python -m harness.train diffusion`
  make_bot("llm:openai/gpt-oss-20b", history="stateless")  # Tensormesh serverless (TENSORMESH_API_KEY)
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / "diffusion"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from agent.value_dqn import board_props  # noqa: E402
from llm.tetris_tools import placement_outcomes  # noqa: E402


class Bot:
    name = "bot"

    def reset(self, episode_seed: int) -> None:
        pass

    def act(self, env, turn: int) -> Optional[int]:
        raise NotImplementedError

    def record(self, used_action_id: int, info: Dict[str, Any]) -> None:
        """Called after the env step with the action actually executed and the step info."""

    def step_info(self) -> Dict[str, Any]:
        """Extra per-step metrics (LLM tokens, latency, $)."""
        return {}


class RandomBot(Bot):
    name = "random"

    def reset(self, episode_seed: int) -> None:
        self.rng = random.Random(10_000 + episode_seed)

    def act(self, env, turn: int) -> Optional[int]:
        return self.rng.choice(env.get_valid_action_ids())


class GreedyBot(Bot):
    """One-ply: the placement whose resulting board has the best heuristic score."""
    name = "greedy"

    def act(self, env, turn: int) -> Optional[int]:
        best = max((o.heuristic, -o.action_id) for o in placement_outcomes(env).values())
        return -best[1]


class BeamBot(Bot):
    def __init__(self, horizon: int = 3, width: int = 16):
        from baselines.beam_search_planner import BeamCfg

        self.cfg = BeamCfg(horizon=horizon, beam_width=width)
        self.name = f"beam{horizon}x{width}"

    def reset(self, episode_seed: int) -> None:
        from baselines.beam_search_planner import BeamSearchPlanner

        self.planner = BeamSearchPlanner(self.cfg, sim_seed=episode_seed)

    def act(self, env, turn: int) -> Optional[int]:
        return int(self.planner.plan(env)[0])


class DQNBot(Bot):
    """Greedy policy of a CNN-DQN checkpoint saved by train_updated.py."""

    def __init__(self, ckpt: str, device: str = "cpu"):
        import torch

        from agent.cnn_dqn_updated import DQNCNN

        self.torch = torch
        ck = torch.load(ckpt, map_location=device, weights_only=False)
        board_h, board_w = ck.get("board_shape", (1, 20, 10))[1:]
        self.model = DQNCNN(int(ck.get("num_actions", 40)), board_h=board_h, board_w=board_w).to(device)
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        self.device = device
        self.name = f"dqn:{Path(ckpt).parent.name}"

    def act(self, env, turn: int) -> Optional[int]:
        from agent.cnn_dqn_updated import tensorize_obs

        board, curr, nxt = (t.unsqueeze(0).to(self.device) for t in tensorize_obs(env._obs()))
        with self.torch.no_grad():
            q = self.model((board, curr, nxt)).squeeze(0)
        valid = env.get_valid_action_ids()
        return int(valid[int(self.torch.argmax(q[valid]).item())])


class DiffusionBot(Bot):
    """Diffusion-MPC planner (MaskGIT denoiser + reranking) from a train_diffusion_updated.py checkpoint."""

    def __init__(self, ckpt: str, num_candidates: int = 16, horizon: int = 8, sampling_constraints: str = "mask_logits",
                 rerank_mode: str = "heuristic", device: str = "cpu"):
        import torch

        from diffusion_model_updated import PlanDenoiser
        from diffusion_planner_updated import PlannerCfg

        self.torch = torch
        ck = torch.load(ckpt, map_location=device, weights_only=False)
        sd = ck["state_dict"]
        n_pos = sd["pos_emb.weight"].shape[0] if "pos_emb.weight" in sd else horizon + 1
        self.model = PlanDenoiser(board_h=20, board_w=10, horizon=n_pos - 1).to(device)
        self.model.load_state_dict(sd, strict=False)
        self.model.eval()
        self.cfg = PlannerCfg(horizon=min(horizon, n_pos - 1), num_candidates=num_candidates,
                              sampling_constraints=sampling_constraints, rerank_mode=rerank_mode)
        self.device = torch.device(device)
        self.name = f"diffusion_K{num_candidates}_H{self.cfg.horizon}"

    def reset(self, episode_seed: int) -> None:
        from diffusion_planner_updated import DiffusionMPCPlanner

        self.planner = DiffusionMPCPlanner(self.model, self.cfg, device=self.device, sim_seed=episode_seed)

    def act(self, env, turn: int) -> Optional[int]:
        obs = env._obs()
        aid, _, _ = self.planner.plan(env, (obs.board, obs.curr_id, obs.next_id))
        return int(aid)


class LLMBot(Bot):
    """An open model on Tensormesh serverless picks a listed placement id (llm/llm_policy.py)."""

    def __init__(self, model: str, history: str = "stateless", window: int = 8, compact_every: int = 16,
                 max_tokens: int = 1024, mock: bool = False, log_path: Optional[str] = None):
        from llm.llm_policy import LLMPolicy, PolicyCfg
        from llm.mock_client import MockClient
        from llm.tensormesh_client import TensormeshClient, load_model_settings

        client = MockClient(log_path=log_path) if mock else TensormeshClient(log_path=log_path)
        cfg = PolicyCfg(model=model, history=history, window=window, compact_every=compact_every,
                        max_tokens=max_tokens, chat_kwargs=dict(load_model_settings().get(model, {})))
        self.client = client
        self.policy = LLMPolicy(client, cfg)
        self.name = f"{model.split('/')[-1]}[{history}]"
        self._info: Dict[str, Any] = {}

    def reset(self, episode_seed: int) -> None:
        self.policy.reset()
        self.lines = 0

    def act(self, env, turn: int) -> Optional[int]:
        feats = board_props(env.game.board.astype("uint8"))
        stats = {"score": int(env.game.score), "lines": self.lines, "max_height": int(feats[1]), "holes": int(feats[6])}
        proposed, res, _, self._user_msg, late, _ = self.policy.act(env, turn, stats, {"bot": self.name})
        self._info = {"prompt_tokens": res.prompt_tokens, "cached_tokens": res.cached_tokens,
                      "completion_tokens": res.completion_tokens, "ttft_s": res.ttft_s, "latency_s": res.latency_s,
                      "cost_usd": res.cost_usd, "reply": res.text[:60]}
        return None if late else proposed

    def record(self, used_action_id: int, info: Dict[str, Any]) -> None:
        self.policy.record(self._user_msg, used_action_id)
        self.lines += int(info.get("lines_cleared", 0))

    def step_info(self) -> Dict[str, Any]:
        return self._info


def make_bot(spec: str, **kw) -> Bot:
    kind, _, arg = spec.partition(":")
    if kind == "random":
        return RandomBot()
    if kind == "greedy":
        return GreedyBot()
    if kind == "beam":
        h, _, w = (arg or "3x16").partition("x")
        return BeamBot(int(h), int(w or 16))
    if kind == "dqn":
        return DQNBot(arg, device=kw.get("device", "cpu"))
    if kind == "diffusion":
        return DiffusionBot(arg, num_candidates=int(kw.get("num_candidates", 16)), horizon=int(kw.get("horizon", 8)),
                            device=kw.get("device", "cpu"))
    if kind == "llm":
        return LLMBot(arg or "openai/gpt-oss-20b", history=kw.get("history", "stateless"),
                      window=int(kw.get("window", 8)), compact_every=int(kw.get("compact_every", 16)),
                      max_tokens=int(kw.get("max_tokens", 1024)), mock=bool(kw.get("mock", False)),
                      log_path=kw.get("log_path"))
    raise ValueError(f"unknown bot spec {spec!r}")


__all__ = ["Bot", "make_bot"]
