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

### Iteration 2 results (`runs/explore/iter02/`: `summary.md`, `figure.png`, `replay_seed1000.gif`)
Spend $0.87 (1999 calls, 0 retries). $ per 100 decisions and cached share:

| model (cached price) | stateless | append | window8 |
|:--|--:|--:|--:|
| gpt-oss-20b ($0) | 0.0065 · 47% | 0.0065 · 95% | **0.0325** · 21% |
| gpt-oss-120b ($0) | 0.0144 · 46% | 0.0153 · 96% | **0.0839** · 13% |
| DeepSeek-V4-Flash (none listed) | 0.0153 · 24% | **0.1596** · 84% | 0.0765 · 14% |

Pieces survived per seed (1000–1003):

| model | stateless | append | window8 |
|:--|:--|:--|:--|
| gpt-oss-20b | 100/76/56/81 | 32/38/23/32 | 42/28/43/44 |
| gpt-oss-120b | 100/55/100/100 | 53/51/39/43 | 96/81/100/100 |
| DeepSeek-V4-Flash | 51/35/35/31 | 41/33/32/32 | 45/44/32/75 |

**Verdict: the cost result scales, and the price card flips the ranking.**
- On both $0-cached models the iteration-1 ordering holds: append costs the same as stateless and
  window8 costs 5.0–5.8× more.
- On DeepSeek-V4-Flash, append is the **most expensive** policy, 10.4× stateless and 2.1× window8.
  84% of its prompt is a reported cache hit, but every token is still billed. The cache works, and
  the price card throws the saving away.
- Quality (oracle regret per decision; normalized score in brackets):

  | model | stateless | append | window8 |
  |:--|:--|:--|:--|
  | gpt-oss-20b | 0.95 [0.45] | 4.37 [0.08] | 2.96 [0.09] |
  | gpt-oss-120b | **0.55 [0.70]** | 3.20 [0.20] | 1.08 [0.65] |
  | DeepSeek-V4-Flash | 3.83 [0.09] | 4.01 [0.09] | 1.90 [0.17] |

  - Append-only history raises regret in 4/4 seeds on both gpt-oss sizes: paired +3.5 (t=7.5) on 20b
    and +2.6 (t=19.5) on 120b, and roughly halves survival.
  - The 8-turn window is nearly harmless for gpt-oss-120b (+0.5 regret, t=1.9, normalized score
    0.65 vs 0.70) but not for 20b.
  - DeepSeek-V4-Flash without reasoning plays poorly in every arm and emits 7–14% illegal/unparseable
    ids. The fallback placements muddy its regret.
  - So the most cache-friendly memory (append) is also the one that damages decisions most, on both
    model sizes.
- TTFT: DeepSeek-V4-Flash sits at a 2.2–2.4 s median in every arm under this load (p90 ≈ 5 s), so
  queueing swamps prefill. On gpt-oss-120b, window8 adds +0.23 s median TTFT over stateless.

The thread is now scaled across seeds, models and price cards. Next, branch to a new serving
question.

---

## Iteration 3: KV-cache lifetime under idle gaps

**Hypothesis.** A cached prefix survives short idle gaps (seconds) but is evicted after minutes on a
shared serverless deployment. Once it is evicted, the next turn pays the full cold prefill again. On
deployments that report the split, LMCache (CPU/remote tier) should keep the prefix longer than the
vLLM GPU prefix cache. The gap where the benefit disappears should differ by model with traffic and
memory pressure.

**Why it matters.** Real agents idle between moves: tool calls, slow environments, humans in the
loop. If the cache dies after ~a minute, a 16k-token agent context costs a full prefill (seconds on
large models, and dollars on no-cached-price models) after every pause. That would favour keep-alive
pings or shorter contexts. It is the serverless analogue of a CPU cache's working-set lifetime.

**Config.** `configs/explore/iter03.yaml`. Prefix: a real 24-turn append-history Tetris prompt from
a greedy-bot game (~16k tokens), with a unique nonce first so the warm-up starts cold. Gaps 0 s, 5 s,
30 s, 2 min, 10 min. 3 trials per gap (the analogue of 3 paired seeds). Models: gpt-oss-20b,
gpt-oss-120b, Kimi-K2.7-Code, DeepSeek-V4-Flash, GLM-5.2-NVFP4. Non-streaming calls with
max_tokens=1, so latency ≈ prefill and the tier split is visible.

**Budget.** ≈ $1 (GLM cold+warm 16k calls dominate at $1.40/M). Cap $2. Cumulative before: $1.00.

### Iteration 3 results (`runs/explore/iter03/`: `summary.md`, `figure.png`)
Spend $0.93 (150 calls). Prompt: 14.8–15.0k tokens. Values are medians of 3 trials, latency of the
next turn with max_tokens=1.

| model | cold | 0 s | 5 s | 30 s | 2 min | 10 min | cached share after gap |
|:--|--:|--:|--:|--:|--:|--:|:--|
| gpt-oss-20b | 1.00 s | 0.16 | 0.18 | 0.25 | 0.19 | 0.57 | 94% at every gap |
| gpt-oss-120b | 1.29 s | 0.21 | 0.21 | 0.24 | 0.21 | 0.70 | 94% (vLLM 14112, LMCache 13824) |
| Kimi-K2.7-Code | 3.34 s | 0.39 | 0.38 | 0.52 | 0.39 | 0.86 | 94% (vLLM 14048, LMCache 13824) |
| **DeepSeek-V4-Flash** | 3.5 s | 0.36 | 0.36 | **3.48** | **3.59** | **3.92** | **93% → 0% between 5 s and 30 s** |
| GLM-5.2-NVFP4 | 34–44 s | 29 | 26 | 30 | 1.0 | 1.5 | never reported |

**Verdict: keep and go deeper. This is the most serverless-specific effect so far.**
- **DeepSeek-V4-Flash forgets a 15k-token agent context in under 30 s of idleness.** After that,
  the next move costs a full cold prefill: 10× the warm latency, and because no cached price is
  listed the bill is the same either way. gpt-oss-20b/120b and Kimi keep the same prefix for ≥ 10 min.
  On the models that report the split, the vLLM GPU tier still holds it at 10 min.
- The 10-min latency uptick on the ≥10-min models (0.2 → 0.6–0.9 s) happens with 94% cache hits, so
  it is not a miss. It is probably the client re-opening an idle keep-alive connection. Iteration 4
  controls for this.
- GLM's short-gap numbers are confounded by queueing we caused ourselves: 15 concurrent 15k-token
  GLM prefills, with cold calls at 34–44 s. Its 2-min and 10-min probes (1.0–1.5 s vs ≥ 34 s cold)
  imply hits that `usage` never reports.

Rubric: (1) DeepSeek's cliff is 10× with 3/3 trials on each side of it; (2) one sentence, one figure;
(3) cache lifetime is a core serving property; (4) about $0.03 per (model, gap) point. → deepen it.

---

## Iteration 4: is it lifetime or capacity? Load sweep + keep-alive + catalog map

**Correction from a control run (`runs/explore/iter04/isolated/`, $0.016).** DeepSeek-V4-Flash was
re-run **one trial at a time**, with no other traffic from us. It kept the prefix across 10 s and
30 s gaps: 13.8k/14.8k tokens cached, about 0.4 s per probe, 4/4 trials. So iteration 3's "< 30 s"
was not an idle timeout. Our own 15 concurrent 15k-token contexts (plus whatever other tenants
did) either **evicted each other** or got the probe **routed to a replica without the prefix**.
The serving property to measure is how many concurrent agent contexts a deployment keeps warm, not
an idle timeout.

**Hypotheses.**
(a) The hit rate after a fixed 30 s gap falls as the number of concurrent distinct contexts we hold
grows (N = 1, 4, 8, 16) on DeepSeek-V4-Flash, but not on gpt-oss-20b.
(b) Under the same load, contexts that re-send their prefix every 4 s keep it (LRU refresh), while
silent contexts lose it. Each refresh is billed at full price on a no-cached-price model.
(c) The remaining catalog models, at iteration 3's load (12 contexts per model), show their own
survival profiles.
(d) GLM, measured sequentially, hits at short gaps too.
(e) The 10-min latency uptick disappears when the HTTP connection is re-warmed.

**Why it matters.** Multi-agent workloads put many long contexts on the same deployment at once:
parallel episodes, fan-out, many users. If a deployment can keep only a few warm, a fleet of agents
thrashes its own cache and pays cold prefill on every move. That is a capacity number a serverless
platform could publish, and a client could respect by capping concurrency or keeping contexts alive.

**Config.** Phases (configs under `configs/explore/`):
- `iter04_load{1,4,8,16}.yaml`: run one after another, DeepSeek-V4-Flash and gpt-oss-20b side by
  side, gap 30 s.
- `iter04_keepalive.yaml`: 16 DeepSeek contexts, gap 60 s; 8 ping every 4 s, 8 stay silent.
- `iter04_map.yaml`: gemma-4, MiniMax-M2.5, Qwen3.8, Qwen3.5 at gaps 5/30/120/600 s with 12
  concurrent contexts per model, plus the gpt-oss-20b 10-min re-warm control.
- `iter04_glm.yaml`: GLM, sequential, gaps 5/30 s.
Same 24-turn Tetris prefix as iteration 3. The HTTP connection is re-warmed before every probe.

**Budget.** ≈ $1.0 (keep-alive pings ≈ $0.002 each); cap $2. Cumulative before: $1.94.

### Iteration 4 results (`runs/explore/iter04/`: `summary.md`, `figure.png`)
Spend $0.84 across the phases (ledger rows tagged `lifetime_*`). "Hit" means the reported
cached_tokens cover ≥ 50% of the prompt. GLM never reports, so for GLM a hit means probe latency
< 30% of its sequential cold prefill.

**(a) Capacity sweep: hit rate after 30 s idle vs concurrent distinct 15k-token contexts.**

| model | 1 | 4 | 8 | 16 |
|:--|:--|:--|:--|:--|
| gpt-oss-20b | 1/1 | 4/4 | 8/8 | 16/16 (probe 0.16 s) |
| **DeepSeek-V4-Flash** | **3/3** (0.43 s) | **0/4** (4.3 s) | **0/8** (3.6 s) | **0/16** (17 s, 429 "server overloaded") |

DeepSeek's cold prefills also **serialize**: the k-th concurrent warm-up finishes at about 3·k s.
Its deployment behaves like one small replica that keeps roughly one 15k-token agent context warm.

**(b) Keep-alive under 16 contexts, 60 s gap: 0/8 hits with pings every 4 s, 0/8 silent.** Each
ping waited ~20 s in the overloaded queue, so only 2 pings per context got through. Keep-alive cannot
rescue a deployment whose cache is too small for the working set. It adds load and full-price
tokens.

**(c) Map at 12 contexts per model (5/30/120/600 s gaps):** MiniMax-M2.5, Qwen3.5-397B and
Qwen3.8-27B hit 100% at every gap, out to 10 min. gemma-4-31B starts to drop: 3/3, 2/3, 2/3, 1/3.
Its warm-ups queue for 8–10 s (a small deployment).
**(d) GLM sequential:** hits at 5 s and 30 s (1.0 s vs 9 s cold), still unreported in `usage`.
**(e) Re-warm control:** gpt-oss-20b at 10 min with the HTTP connection re-warmed has probe 0.18 s,
the same as at 0 s. The iteration-3 uptick (0.57 s) was connection re-establishment, not the cache.

**Verdict: keep. The intuitive one-figure finding is capacity under concurrency, not idle lifetime.**
On a shared serverless deployment, a prefix survives idleness for as long as nothing else evicts it.
How many agents can stay warm at once is the per-deployment number that matters, and on this
catalog it ranges from ~1 (DeepSeek-V4-Flash) to ≥ 16 (gpt-oss-20b).
- This explains iteration 2's DeepSeek append arm (84% cached): back-to-back moves re-used the
  prefix within ~3 s, before eviction.
- It also explains iteration 3's "< 30 s" cliff: 15 concurrent contexts.

Rubric: (1) 0/28 hits vs 3/3 at N=1 on DeepSeek, 29/29 on gpt-oss-20b; (2) one figure; (3) cache
capacity and prefill queueing; (4) ≈ $0.15 per sweep. Scaling it further (a load sweep on every
model) is cheap, but the mechanism is already clear. The open question it raises is what one agent
should do when it needs many calls on the same long context. That is iteration 5.

---

## Iteration 5: fan-out over a shared long prefix (LLM reranker pattern)

**Hypothesis.** An agent that asks K parallel questions about the same ~15k-token context (rating K
beam candidates, best-of-K) races itself. If all K are sent at once while the prefix is cold, each
request prefills the full prefix (K× prefill tokens and K× billed tokens on uncached-price
models). Priming (send one, wait, then fan out K−1) makes the rest cache hits. On DeepSeek, cold
fan-out also serializes, so priming should cut makespan as well. On a fast, large-cache deployment
(gpt-oss-20b) the cold race may cost little latency but still bills K× uncached tokens.

**Why it matters.** Fan-out is the standard pattern for LLM search, reranking and self-consistency.
Whether the server deduplicates in-flight identical prefixes, or the client has to prime, is a
serving property and a cheap client-side fix.

**Config.** `configs/explore/iter05.yaml`. Same 24-turn Tetris prefix with a fresh nonce per trial.
K ∈ {4, 8}. Strategies: cold_fanout / primed_fanout / sequential. 3 trials. gpt-oss-20b,
gpt-oss-120b, DeepSeek-V4-Flash. One trial at a time per model, models side by side. max_tokens=1.

**Budget.** ≈ $0.6 (DeepSeek bills every token: ~$0.002 per request). Cap $2. Cumulative before:
$2.77.
