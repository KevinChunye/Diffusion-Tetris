# Decision models: verified facts (2026-10-04)

## Jev (TypeSafe), proprietary, API-only, not tested by us
- TypeSafe launch post: Almeida, D. "Introducing System One Models & Jev", TypeSafe AI Blog, 15 Sep 2026,
  https://typesafe.ai/blog/introducing-system-one-models-and-jev
  - "new model architecture, parallel sampler"; trained with "Reinforcement Learning for Calibrated Decisions (RLCD)"
  - "End-to-end response time is 70ms-500ms" (vendor); "Jev is neither small nor an LLM"
  - "We deliberately chose not to publish performance against public benchmarks."
  - "available today in early access"
- TypeSafe docs (https://docs.typesafe.ai/api, /models): POST /v1/systemone; question types Choice (≤255 options),
  Score (≤10 levels), Noul (yes/no); `jev-latest` → `jev-1.13.0`; "the same weights serve every account".
- DataCamp explainer: Crabtree, M. "Jev: TypeSafe's System One Model Explained", DataCamp Blog, 16 Sep 2026,
  https://www.datacamp.com/blog/system-one-models-jev — "a proprietary, hosted model reachable through TypeSafe's
  HTTP API"; "in early access as of September 15, 2026, gated behind a waitlist"; vendor-reported "$0.042 per
  million input tokens", output "unmetered".
- Third-party Decision Index (apolinario, GitHub, edition 0.2.1, 27 Sep 2026, "Not affiliated with TypeSafe AI"):
  38 benchmarks in five areas, 150,317 scoreable requests; lists Jev at 57.89. A Hugging Face Space leaderboard
  (multimodalart/jev-decision-index) shows Jev's median latency 524.1 ms through the hosted API.

## Intelif (open weights, self-hosted by us)
- Krishnan, S. intelif v0.1, GitHub SkAndMl/intelif (commit a9f4ae6, 2026-10-03); weights
  UserMoonlight/intelif-qwen3-4b @ v0.1.
- Code MIT; the adapter and scorer weights are CC BY-NC 4.0 per the main-branch model card (SciQ is CC BY-NC).
- Qwen/Qwen3-4B (rev 1cfa9a72) + LoRA (r=16, alpha 32, on q/k/v/o/gate/up/down, merged at load) + linear scorer
  2560→1. "Each option in the prompt is followed by an anchor token; the scorer reads the anchor's final hidden
  state, and the probabilities are a softmax over the options." Anchor `<|endoftext|>`. "One forward pass answers
  a question, whatever the number of options."
- Decision Index 0.2.1: 31.77 (raw 48.62); "median latency 16.5 ms" on "one NVIDIA RTX PRO 6000 Blackwell" over
  150,317 requests. Training data excludes the Decision Index suite (BANKING77, MASSIVE, xLAM, ALFWorld, WebShop,
  MNLI, SNLI, QQP, PAWS, BoolQ, CommonsenseQA, OpenBookQA, SciQ).
- Its Tetris example: state = '#'/'.' board + piece; options "rotation r, column x" described by outcome; the same
  instruction text we use; "about 40 ms on an RTX PRO 6000".
- "Intelif is wire-compatible with TypeSafe's Jev (POST /v1/systemone), but not affiliated with it." Schema
  matches TypeSafe docs (minor differences: optional instructions, no 255-option cap).
