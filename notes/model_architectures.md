# Served models: architecture facts and sources (verified 2026-10-04)

Every row was read from an official source: a model card (raw README on Hugging Face), the model's
`config.json` (saved in `runs/explore/models/configs/`), an arXiv report, or an official blog.
KV-cache sizes are our arithmetic from the configs (`llm/model_arch.py`), assuming 16-bit KV. They are
architectural estimates, not measurements of the provider's memory.

| Served id | Total / active params | Attention and KV design | Est. KV per token at 15k | Source |
|:--|:--|:--|--:|:--|
| openai/gpt-oss-20b | 20.9B / 3.6B (MoE, 32 experts, top-4) | alternating dense and 128-token banded window; GQA, 8 KV heads × 64 | 24 KB | OpenAI model card, arXiv:2508.10925 |
| openai/gpt-oss-120b | 116.8B / 5.1B (MoE, 128 experts, top-4) | same pattern, 36 layers | 36 KB | same |
| google/gemma-4-31B-it | 30.7B dense | 5:1 sliding (1024) : global; keys reused as values in global layers | 95 KB | Gemma 4 Technical Report, arXiv:2607.02770; HF card |
| deepseek-ai/DeepSeek-V4-Flash | 284B / 13B (MoE, 256+1 experts, top-6) | interleaved Compressed Sparse Attention (m=4, top-512) and Heavily Compressed Attention (m'=128); shared-KV MQA, head dim 512; 128-token uncompressed window; FP8 KV with BF16 RoPE dims | 4 KB | DeepSeek-V4 report, arXiv:2606.19348; HF card |
| MiniMaxAI/MiniMax-M2.5 | ~229B (safetensors count); M2: 230B / 10B | full softmax attention in all 62 layers; GQA 8 KV heads × 128 | 248 KB | HF card and config; MiniMax blog "Why did M2 end up as a full attention model?" |
| Qwen/Qwen3.5-397B-A17B-FP8 | 397B / 17B (MoE, 512 experts, 10+1) | hybrid: 15 × (3 Gated DeltaNet + 1 gated full attention); full layers 2 KV heads × 256 | 42 KB (incl. fixed state) | HF card ("Qwen3.5: Towards Native Multimodal Agents", Qwen Team, 2026) |
| Qwen/Qwen3.8-27B-FP8 | 27B dense | hybrid: 16 × (3 Gated DeltaNet + 1 gated full attention); 4 KV heads × 256 | 74 KB (incl. fixed state) | HF card ("built on the architectural foundation of Qwen3.5") |
| moonshotai/Kimi-K2.7-Code | 1T / 32B (MoE, 384 experts, top-8) | MLA (latent 512 + RoPE 64), 61 layers; INT4 QAT experts | 69 KB | HF card; Kimi K2 report, arXiv:2507.20534 |
| lukealonso/GLM-5.2-NVFP4 (community quantization of zai-org/GLM-5.2) | 744B / 40B (MoE, 256+1 experts, top-8) | MLA + DeepSeek Sparse Attention with IndexShare (one indexer per 4 layers); 78 layers | 98 KB | zai-org GLM-5 GitHub table; GLM-5 report, arXiv:2602.15763; IndexCache, arXiv:2603.12201 |

## Mechanisms relevant to our measurements

- **Prefill cost** grows with the number of active parameters (about 2 × active parameters FLOPs per
  token) plus attention, which grows with context length. Sparse or compressed attention (DSA, CSA/HCA)
  and linear attention reduce the attention term:
  - DeepSeek-V3.2 (arXiv:2512.02556): DSA "reduces the core attention complexity ... from O(L^2) to
    O(Lk)".
  - DeepSeek-V4 report: Flash uses "only 10% of the single-token FLOPs and 7% of the KV cache size
    compared with DeepSeek-V3.2" at 1M tokens.
  - IndexCache (arXiv:2603.12201): the DSA indexer itself "retains O(L^2) complexity".
- **KV-cache size** sets how many contexts fit in a given memory:
  - MLA compresses KV into a latent vector (DeepSeek-V2, arXiv:2405.04434: "reduces the KV cache by
    93.3%").
  - GQA shares KV heads across query heads (Ainslie et al., EMNLP 2023, arXiv:2305.13245).
  - Sliding-window layers cap their KV at the window.
  - Gated DeltaNet layers (Yang, Kautz, Hatamizadeh, ICLR 2025, arXiv:2412.06464) keep a fixed-size
    recurrent state instead of a growing cache.
- **Prefix caching is harder for some designs:**
  - vLLM caches hybrid (Mamba/GDN) state only at block boundaries (`--mamba-cache-mode align`); this
    mode is marked experimental (vLLM docs; PyTorch blog "Hybrid Models as First-Class Citizens in
    vLLM", Nov 2025).
  - LMCache documents a separate configuration for hybrid models (Qwen3.5/3.6/3.8, GLM 5.1/5.2) and
    notes that cached and fresh runs are not bit-exact for GDN models.
  - The DeepSeek-V4 report: its "KV cache sizes vary across different layers", and SWA layers need
    "separate cache hit and eviction policies".
  - MiniMax names prefix caching as one reason it kept full attention in M2: "In real-world
    applications, the cache-hit rate for conversations is very high. A new architecture must handle
    this gracefully."
- **The provider:**
  - Tensormesh states it is "the team behind LMCache" (tensormesh.ai/about), serves through vLLM
    (`owned_by: vllm`; blog: "sits on top of the vLLM inference engine, the Tensormesh LMCache
    stack"), and lists "KV cache CPU offloading and KV cache external storage offloading".
  - LMCache paper: Liu et al., arXiv:2510.09665.

## Pricing caveat (affects how we estimate cost)

- Card prices (tensormesh.ai/pricing, read 2026-10-04): gpt-oss-20b 0.07/0.28, gpt-oss-120b
  0.15/0.60, gemma-4-31B 0.14/0.56 and MiniMax-M2.5 0.30/1.20 list cached input at $0. DeepSeek V4
  Flash (0.14/0.28) and Qwen3.5-397B (0.60/3.60) cards show no cached row.
- Three served ids are not on the page; the page shows earlier versions instead. Qwen3.8-27B →
  "Qwen3.6-27B" 0.32/3.20, Kimi-K2.7-Code → "Kimi K2.6" 0.96/4.00 with cached $0, GLM-5.2 →
  "GLM-5.1" 1.40/4.40.
- The same page says "Tensormesh does not charge for cached tokens", and its calculator uses $0 cached
  for every model. A May 2026 blog says cached input is billed at $0 "across all of Tensormesh's
  serverless deployments".
- Billing data is not available through the inference API key (HTTP 401). We therefore estimate cost
  under the stated policy (cached input $0 for every model) and report a full-price upper bound
  separately. Our budget guard uses the upper bound.
