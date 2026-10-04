"""refs.py - the reference list. `python paper/refs.py` replaces <!--REFS--> ... <!--/REFS--> in paper.html with
the entries actually cited (href="#ref-..."), sorted by first author. Every entry was checked against its
source (arXiv record, venue page, repository or web page); see notes/references_scaleup.md."""

import re
from pathlib import Path

REFS = {
    "sarathi": "Agrawal, A., Kedia, N., Panwar, A., Mohan, J., Kwatra, N., Gulavani, B. S., Tumanov, A., and Ramjee, R. Taming throughput-latency tradeoff in LLM inference with Sarathi-Serve. OSDI 2024.",
    "gqa": "Ainslie, J., Lee-Thorp, J., de Jong, M., Zemlyanskiy, Y., Lebrón, F., and Sanghai, S. GQA: Training generalized multi-query transformer models from multi-head checkpoints. EMNLP 2023. arXiv:2305.13245.",
    "algorta": "Algorta, S. and Şimşek, Ö. The game of Tetris in machine learning. Video Games and Machine Learning Workshop at ICML 2017. arXiv:1905.01652.",
    "typesafe": "Almeida, D. Introducing System One models &amp; Jev. TypeSafe AI blog, 15 September 2026. typesafe.ai/blog/introducing-system-one-models-and-jev, accessed October 2026.",
    "dindex": "apolinario. Decision Index, edition 0.2.1. GitHub repository, 27 September 2026. github.com/apolinario/decision-index, accessed October 2026.",
    "indexcache": "Bai, Y., Dong, Q., Jiang, T., et al. IndexCache: Accelerating sparse attention via cross-layer index reuse. arXiv:2603.12201, 2026.",
    "cai": "Cai, W., Shi, T., Zhao, X., and Song, D. Are you getting what you pay for? Auditing model substitution in LLM APIs. arXiv:2504.04715, 2025.",
    "jev": "Crabtree, M. Jev: TypeSafe’s System One model explained. DataCamp blog, 16 September 2026. datacamp.com/blog/system-one-models-jev, accessed October 2026.",
    "tail": "Dean, J. and Barroso, L. A. The tail at scale. Communications of the ACM, 56(2):74–80, 2013.",
    "dsv2": "DeepSeek-AI. DeepSeek-V2: A strong, economical, and efficient mixture-of-experts language model. arXiv:2405.04434, 2024.",
    "dsv32": "DeepSeek-AI. DeepSeek-V3.2: Pushing the frontier of open large language models. arXiv:2512.02556, 2025.",
    "dsv4": "DeepSeek-AI. DeepSeek-V4: Towards highly efficient million-token context intelligence. arXiv:2606.19348, 2026.",
    "cachedattn": "Gao, B., He, Z., Sharma, P., Kang, Q., Jevdjic, D., Deng, J., Yang, X., Yu, Z., and Zuo, P. Cost-efficient large language model serving for multi-turn conversations with CachedAttention. USENIX ATC 2024.",
    "gemma4": "Gemma Team. Gemma 4 technical report. arXiv:2607.02770, 2026.",
    "glm5": "GLM-5-Team. GLM-5: From vibe coding to agentic engineering. arXiv:2602.15763, 2026.",
    "gu": "Gu, C., Li, X. L., Kuditipudi, R., Liang, P., and Hashimoto, T. Auditing prompt caching in language model APIs. ICML 2025.",
    "lora": "Hu, E. J., Shen, Y., Wallis, P., et al. LoRA: Low-rank adaptation of large language models. arXiv:2106.09685, 2021.",
    "lmgame": "Hu, L., Huo, M., Zhang, Y., Yu, H., Xing, E. P., Stoica, I., Rosing, T., Jin, H., and Zhang, H. lmgame-Bench: How good are LLMs at playing games? ICLR 2026.",
    "kapoor": "Kapoor, S., Bommasani, R., Klyman, K., et al. Position: On the societal impact of open foundation models. ICML 2024.",
    "kimi": "Kimi Team. Kimi K2: Open agentic intelligence. arXiv:2507.20534, 2025.",
    "intelif": "Krishnan, S. Intelif v0.1: Typed probabilistic decisions from a small open model. GitHub repository SkAndMl/intelif (code MIT; weights UserMoonlight/intelif-qwen3-4b, CC BY-NC 4.0), 2026. Accessed October 2026.",
    "vllm": "Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., Gonzalez, J. E., Zhang, H., and Stoica, I. Efficient memory management for large language model serving with PagedAttention. SOSP 2023.",
    "helm": "Liang, P., Bommasani, R., Lee, T., et al. Holistic evaluation of language models. Transactions on Machine Learning Research, 2023.",
    "lost": "Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., and Liang, P. Lost in the middle: How language models use long contexts. Transactions of the ACL, 12:157–173, 2024a.",
    "agentbench": "Liu, X., Yu, H., Zhang, H., et al. AgentBench: Evaluating LLMs as agents. ICLR 2024b.",
    "lmcache": "Liu, Y., Cheng, Y., Yao, J., et al. LMCache: An efficient KV cache layer for enterprise-scale LLM inference. arXiv:2510.09665, 2025.",
    "autellix": "Luo, M., Shi, X., Cai, C., et al. Autellix: An efficient serving engine for LLM agents as general programs. arXiv:2502.13965, 2025.",
    "minimax": "MiniMax. Why did M2 end up as a full attention model? MiniMax news, 29 October 2025. minimax.io/news/why-did-m2-end-up-as-a-full-attention-model, accessed October 2026.",
    "k2": "MoonshotAI. K2-Vendor-Verifier. GitHub repository, github.com/MoonshotAI/K2-Vendor-Verifier, accessed October 2026.",
    "gptoss": "OpenAI. gpt-oss-120b &amp; gpt-oss-20b model card. arXiv:2508.10925, 2025.",
    "balrog": "Paglieri, D., Cupiał, B., Coward, S., et al. BALROG: Benchmarking agentic LLM and VLM reasoning on games. ICLR 2025.",
    "splitwise": "Patel, P., Choukse, E., Zhang, C., Shah, A., Goiri, Í., Maleki, S., and Bianchini, R. Splitwise: Efficient generative LLM inference using phase splitting. ISCA 2024.",
    "mooncake": "Qin, R., Li, Z., He, W., Cui, J., Ren, F., Zhang, M., Wu, Y., Zheng, W., and Xu, X. Mooncake: Trading more storage for less computation, a KVCache-centric architecture for serving LLM chatbot. FAST 2025.",
    "qwen35": "Qwen Team. Qwen3.5: Towards native multimodal agents. Model card, Hugging Face (Qwen/Qwen3.5-397B-A17B-FP8), 2026. Accessed October 2026.",
    "mlperf": "Reddi, V. J., Cheng, C., Kanter, D., et al. MLPerf inference benchmark. ISCA 2020.",
    "schroeder": "Schroeder, B., Wierman, A., and Harchol-Balter, M. Open versus closed: A cautionary tale. NSDI 2006.",
    "vllmhybrid": "vLLM Team at IBM. Hybrid models as first-class citizens in vLLM. PyTorch blog, 5 November 2025. pytorch.org/blog/hybrid-models-as-first-class-citizens-in-vllm, accessed October 2026.",
    "qwen3": "Yang, A., Li, A., Yang, B., et al. Qwen3 technical report. arXiv:2505.09388, 2025.",
    "gdn": "Yang, S., Kautz, J., and Hatamizadeh, A. Gated delta networks: Improving Mamba2 with delta rule. ICLR 2025. arXiv:2412.06464.",
    "orca": "Yu, G.-I., Jeong, J. S., Kim, G.-W., Kim, S., and Chun, B.-G. Orca: A distributed serving system for transformer-based generative models. OSDI 2022.",
    "yuan": "Yuan, R., Zeng, Y., Gao, X., Yu, L., Liao, H., and Wang, H. Scheduling the unschedulable: Taming black-box LLM inference at scale. arXiv:2604.06970, 2026.",
    "sglang": "Zheng, L., Yin, L., Xie, Z., et al. SGLang: Efficient execution of structured language model programs. NeurIPS 2024.",
    "distserve": "Zhong, Y., Liu, S., Chen, J., Hu, J., Zhu, Y., Liu, X., Jin, X., and Zhang, H. DistServe: Disaggregating prefill and decoding for goodput-optimized large language model serving. OSDI 2024.",
}


def main() -> None:
    p = Path(__file__).resolve().parent / "paper.html"
    html = p.read_text(encoding="utf-8")
    cited = set(re.findall(r'href="#ref-([a-z0-9]+)"', html))
    missing = cited - set(REFS)
    if missing:
        raise SystemExit(f"cited but not in REFS: {sorted(missing)}")
    items = sorted(cited, key=lambda k: REFS[k].lower())
    body = "\n".join(f'  <li id="ref-{k}">{REFS[k]}</li>' for k in items)
    html = re.sub(r"<!--REFS-->.*?<!--/REFS-->", f"<!--REFS-->\n{body}\n<!--/REFS-->", html, flags=re.S)
    p.write_text(html, encoding="utf-8")
    print(f"{len(items)} references; unused: {sorted(set(REFS) - cited)}")


if __name__ == "__main__":
    main()
