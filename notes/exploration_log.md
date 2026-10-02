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

### Iteration 1 results (`runs/explore/iter01/`: `summary.md`, `figure.png`, `replay_seed1001.gif`)
Spend $0.131. An attempt that crashed after its API phase (a beam-search bug, now fixed and
regression-tested) cost $0.057 and reproduced the same pattern; its call log is
`attempt1_calls.jsonl`.

| arm | pieces survived | norm score | beam regret / decision | cached share | uncached tok / call | TTFT p50 | $ / 100 decisions |
|:--|--:|--:|--:|--:|--:|--:|--:|
| stateless | 71.7 | 0.42 | 1.06 | 0.49 | 576 | 0.45 s | 0.0064 |
| append | 32.0 | 0.05 | 4.17 | **0.95** | 540 | 0.55 s | 0.0061 |
| window8 | 45.0 | 0.12 | 3.09 | **0.19** | **4570** | 0.62 s | **0.0341** |
| compact16 | 44.0 | 0.13 | 2.43 | 0.90 | 578 | 0.51 s | 0.0066 |

Per-seed differences against stateless (3 seeds, same sign in 3/3 for every line below):

| arm | cost per decision | TTFT p50 | beam regret | pieces survived |
|:--|:--|:--|:--|:--|
| window8 | +$0.00011/decision (5.3×, t=9.7) | +0.16 s (t=7.9) | +1.9 (t=3.4) | |
| append | ≈ same | +0.10 s (t=42) | +3.0 (t=5.0) | −40 |
| compact16 | ≈ same | | +1.5 (t=2.2) | |

**Verdict: keep, then scale.**
- **Cost, confirmed.** With $0 cached input, history that only appends costs the same per decision
  as no history (0.0061 vs 0.0064 $/100). It sends 10× more prompt tokens, but 95% of them are
  cache hits. A sliding window costs **5.3× more**: shifting the window rewrites the prefix right
  after the 530-token system prompt, so ~4.6k tokens are re-prefilled and billed on every turn.
- **Latency is a second-order effect at this size.** TTFT moves +0.10–0.16 s on prompts up to 22k
  tokens, against a ~0.45 s network floor.
- **Quality, unexpected.** History *hurts* gpt-oss-20b on this Markov task. Beam regret per decision
  is 2–4× stateless, and it is already higher at turns 0–4, when every arm's board is nearly empty
  (0.20 stateless vs 0.31–0.57). Bad decisions then compound into earlier top-outs. The arm with the
  longest context (append) is the worst. The cheapest memory to serve is the most harmful to
  decisions.

Rubric: (1) the cost and regret effects are 3/3-seed consistent and far above noise; (2) one figure
shows prefix rewrites → uncached tokens → $; (3) cache/prefill/cost; (4) $0.07 per pilot. →
**Iteration 2 scales this thread across the price card and models.**

---

## Iteration 2: history policy × price card (3 models)

**Hypothesis.** The cost ranking of memory policies depends on whether the endpoint discounts
cached tokens, and the ranking flips when it doesn't. On gpt-oss-20b/120b (cached $0): append ≈
stateless ≪ window. On DeepSeek-V4-Flash (prefix hits are reported but no cached price is listed,
so every token is billed): stateless < window ≪ append, since append's prompt grows linearly and is
fully billed each turn. Caching still helps latency on every model. Second question: does the
quality damage from history hold for a larger model (gpt-oss-120b) and a non-reasoning model
(DeepSeek-V4-Flash)?

**Why it matters.** The best memory policy for an agent is a property of the price card and the
cache as much as of the model. A serverless platform that caches without discounting cached tokens
gives users no economic reason to write cache-friendly prompts.

**Config.** `configs/explore/iter02.yaml`: {gpt-oss-20b, gpt-oss-120b, DeepSeek-V4-Flash} ×
{stateless, append, window8}, 4 paired seeds (1000–1003) × 100 pieces, 16 concurrent episodes.
DeepSeek runs without reasoning (`max_tokens=64`); gpt-oss uses `reasoning_effort=low`
(`max_tokens=1024`).

**Budget.** Projected from iteration-1 token profiles: ≤ $2.8 worst case (if every episode lasts
100 pieces; DeepSeek-append alone ≈ $0.45/episode). Cap $5. Cumulative before this iteration: $0.13.
