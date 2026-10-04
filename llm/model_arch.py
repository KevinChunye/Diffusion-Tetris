"""model_arch.py - KV-cache footprint per token implied by each served model's published config.

Bytes are for a 16-bit cache unless a model's report states otherwise; real engines may store KV in
FP8 (half) or keep extra metadata, so these are architectural estimates for comparison, not
measurements of the provider's memory. Sliding-window layers hold at most `window` tokens, so for a
context of L tokens they contribute min(L, window) entries; linear-attention layers hold a fixed state.

  python -m llm.model_arch            # table for a 6.6k- and a 15k-token context
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

CONFIGS = Path("runs/explore/models/configs")
B16 = 2  # bytes per element

# Parameter counts as stated by official sources (see notes/model_architectures.md).
PARAMS_B = {  # model id: (total, active), billions; MiniMax-M2.5 active from the M2 card (same architecture)
    "openai/gpt-oss-20b": (20.9, 3.6), "openai/gpt-oss-120b": (116.8, 5.1), "google/gemma-4-31B-it": (30.7, 30.7),
    "deepseek-ai/DeepSeek-V4-Flash": (284.0, 13.0), "MiniMaxAI/MiniMax-M2.5": (229.0, 10.0),
    "Qwen/Qwen3.5-397B-A17B-FP8": (397.0, 17.0), "Qwen/Qwen3.8-27B-FP8": (27.0, 27.0),
    "moonshotai/Kimi-K2.7-Code": (1000.0, 32.0), "lukealonso/GLM-5.2-NVFP4": (744.0, 40.0),
}


def _cfg(model: str) -> Dict:
    c = json.loads((CONFIGS / (model.replace("/", "_") + ".json")).read_text())
    return c.get("text_config", c)


def kv_bytes(model: str, context: int) -> float:
    """Approximate KV-cache bytes for one sequence of `context` tokens."""
    c = _cfg(model)
    t = c.get("model_type", "")
    if t == "gpt_oss":
        per = 2 * c["num_key_value_heads"] * c["head_dim"] * B16
        full = sum(1 for x in c["layer_types"] if x == "full_attention")
        slide = len(c["layer_types"]) - full
        return per * (full * context + slide * min(context, c["sliding_window"]))
    if t == "gemma4_text":
        local = 2 * c["num_key_value_heads"] * c["head_dim"] * B16
        glob = (1 if c.get("attention_k_eq_v") else 2) * c["num_global_key_value_heads"] * c["global_head_dim"] * B16
        full = sum(1 for x in c["layer_types"] if x == "full_attention")
        slide = len(c["layer_types"]) - full
        return glob * full * context + local * slide * min(context, c["sliding_window"])
    if t == "minimax_m2":
        return 2 * c["num_key_value_heads"] * c["head_dim"] * B16 * c["num_hidden_layers"] * context
    if t in ("qwen3_5_moe_text", "qwen3_5_text"):
        full = sum(1 for x in c["layer_types"] if x == "full_attention")
        lin = len(c["layer_types"]) - full
        attn = 2 * c["num_key_value_heads"] * c["head_dim"] * B16 * full * context
        state = lin * c["linear_num_value_heads"] * c["linear_key_head_dim"] * c["linear_value_head_dim"] * 4  # fp32 state
        return attn + state
    if t in ("kimi_k2", "glm_moe_dsa") or "DeepseekV3" in str(c.get("architectures")):
        latent = (c["kv_lora_rank"] + c["qk_rope_head_dim"]) * B16
        idx = (c.get("index_head_dim", 0) * 1) if t == "glm_moe_dsa" else 0  # indexer keys (8-bit assumed)
        return (latent + idx) * c["num_hidden_layers"] * context
    if t == "deepseek_v4":
        ratios = c["compress_ratios"][: c["num_hidden_layers"]]
        entry = c["head_dim"] * 1.0 + 64 * 1.0  # mostly FP8 latent with BF16 RoPE part (report): ~576 B
        total = 0.0
        for r in ratios:
            window = min(context, c["sliding_window"]) * entry  # uncompressed local branch
            total += window + (context / r * entry if r else 0.0)
            if r == 4:
                total += context / r * c["index_head_dim"] * 0.5  # FP4 indexer keys
        return total
    raise ValueError(f"no KV model for {model} ({t})")


def main() -> None:
    models = sorted(p.stem.replace("_", "/", 1) for p in CONFIGS.glob("*.json"))
    print(f"{'model':34s} {'KB/token @15k':>14s} {'MB per 6.6k ctx':>16s} {'MB per 15k ctx':>15s}")
    for m in models:
        print(f"{m:34s} {kv_bytes(m, 15000) / 15000 / 1024:14.1f} {kv_bytes(m, 6650) / 2**20:16.0f} {kv_bytes(m, 15000) / 2**20:15.0f}")


if __name__ == "__main__":
    main()
