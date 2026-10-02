# Exploration log: Tetris as a long-horizon workload for serverless KV-cache reuse

Workload: an LLM picks one legal placement id per piece; decisions are scored offline with dense
per-step regret (`llm/oracle.py`: heuristic rollout + beam search on non-clairvoyant clones).
Each pilot uses 3 paired episode seeds × ≤100 pieces, with every arm on the same piece sequence and
all arms running concurrently so they share the same time window.
Spend ledger: `runs/explore/spend.csv` (hard cap $25 total, $5 per iteration).

Normalized score: 0 = random legal policy, 1 = `beam_search_planner` default (H=3, W=16), on the
same seeds and piece budget (`runs/explore/reference_scores.csv`).

---

## Iteration 1: history policies vs KV-cache reuse (gpt-oss-20b)

**Hypothesis.** gpt-oss-20b bills cached input at $0. Memory that keeps the prompt prefix stable
(append-only, block compaction) should therefore cost about the same as no memory. Memory that
rewrites the prefix every turn (a sliding window) should pay for every token it re-sends: about
(W+1)× the uncached tokens per call, plus extra prefill latency. Tetris is Markov, so history may
not improve decisions at all, which would mean any cost it adds buys nothing.

**Why it matters for serverless inference.** Agent frameworks default to sliding windows or
summarization for "memory". On a prefix-cached, per-token-billed endpoint, *how* the context is
edited matters more than *how much* context there is. Edits decide which tokens are re-prefilled and
billed. That is a design rule a serving platform could surface to users.

**Config.** `configs/explore/iter01.yaml`. Model `openai/gpt-oss-20b` (`reasoning_effort=low`,
`max_tokens=1024`; low-effort reasoning used up to 432 tokens in a smoke test). Static-first prompt (~530-token system prefix: rules, piece shapes, output
format), annotated legal placements (`id: rot, cols -> lines, height, holes`), plain-text JSON
reply. Arms:
`stateless` | `append` | `window8` (last 8 turns) | `compact16` (append, reset to a summary every 16
turns). Seeds 1000–1002, 100 pieces, 12 concurrent episodes.

**Budget.** Estimate ≈ $0.30 (window8 dominates at ~9k uncached tokens/call); cap $5.
