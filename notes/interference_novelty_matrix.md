# Long-context interference on short agent decisions: overlap audit and go/no-go

Date: 2026-10-03. Scope: **API-only**. This environment has no GPU and no controllable vLLM/SGLang
server, only Tensormesh's public OpenAI-compatible endpoint, which works with the configured
credentials and proxy. Sources below were retrieved in this session. Claims about them restate their
abstracts/documentation and are not re-measured.

## What already exists

| Work | Mechanism and assumptions | Closest experiment | What the Tetris/API setting adds | Overlap verdict |
|:--|:--|:--|:--|:--|
| **vLLM V1** continuous batching, [chunked prefill + decode-first budget](https://raw.githubusercontent.com/vllm-project/vllm/main/docs/configuration/optimization.md), [`--scheduling-policy priority`](https://docs.vllm.ai/en/v0.8.1/serving/engine_args.html) | Server scheduler. Chunked prefill is on by default and pending decodes are scheduled before prefill chunks. `max_num_batched_tokens` trades ITL against TTFT. An optional per-request `priority` (lower = earlier) | Library benchmarks with synthetic prompt/output lengths | Nothing, if you control the server | **Solves the mechanism server-side.** The client cannot see or set any of it except a `priority` field, whose effect on a hosted endpoint is unknown |
| **Sarathi-Serve** ([arXiv 2403.02310](https://arxiv.org/abs/2403.02310)) | Chunked prefills plus stall-free batching to remove prefill/decode interference; full control of batching | Capacity under TBT/TTFT tail SLOs on A100s, synthetic and dataset traces | — | **Solves server-side prefill/decode interference** |
| **DistServe** ([OSDI'24, arXiv 2401.09670](https://arxiv.org/abs/2401.09670)) | Disaggregates prefill and decode onto different GPUs; goodput under TTFT/TPOT SLOs | Goodput/SLO attainment on chatbot, code and summarization traces | — | **Solves it by disaggregation** (server/cluster control) |
| **SGLang** ([arXiv 2312.07104](https://arxiv.org/abs/2312.07104), [scheduler docs](https://mintlify.com/sgl-project/sglang/concepts/continuous-batching)) | RadixAttention prefix reuse; LPM/DFS-weight cache-aware ordering, priority scheduling, chunked prefill | Throughput on agent/RAG/JSON programs | — | Prefix-aware and priority scheduling are known (server-side) |
| **Preble** ([arXiv 2407.00023](https://arxiv.org/abs/2407.00023)), **Llumnix** ([OSDI'24, arXiv 2406.03243](https://arxiv.org/abs/2406.03243)) | Prefix-aware distributed placement; live migration with priorities across instances | Cluster p99 latency | — | Prefix- and priority-aware placement are known (cluster side) |
| **FastServe** ([arXiv 2305.05920](https://arxiv.org/abs/2305.05920)), **SLOs-Serve** ([arXiv 2504.08784](https://arxiv.org/abs/2504.08784)), **HyGen** ([NeurIPS'25, arXiv 2501.14808](https://arxiv.org/abs/2501.14808)) | Length-informed preemptive MLFQ; multi-SLO token allocation; interference-aware co-location of online and offline/long work with starvation prevention | Server-side SLO/goodput | — | **Short-first, length-aware and starvation-bounded co-location are known** |
| **Autellix** ([arXiv 2502.13965](https://arxiv.org/abs/2502.13965)), **AgentServeSim** ([arXiv 2606.09613](https://arxiv.org/abs/2606.09613)) | Program-level (agent) attained-service scheduling; simulated agent serving and policy search | vLLM-backed and simulated agent programs | — | Agent-aware head-of-line mitigation is known (server-side) |
| **Black-box client scheduling** ([arXiv 2604.06970](https://arxiv.org/abs/2604.06970)), [LiteLLM priority queue](https://docs.litellm.ai/docs/scheduler) | Client-side DRR across classes, feasibility ordering, admit/defer/reject in front of a black-box API; gateway priority queues | **Simulation only.** Mock provider with output-token-linear latency calibrated on 18 calls; no prefill or input-length interference modeled; synthetic/ShareGPT lengths; no task-quality constraint | Real endpoint, prefill-dominated long requests, decision-quality constraint | **The client-side algorithm family exists.** Its premise for *input-length* interference on real endpoints is untested |

## Decision

- **No-go on a new scheduler.** Length-aware deferral, short-first queues, priority classes and
  starvation bounds already exist on both the server side (vLLM, SGLang, FastServe, HyGen) and the
  client side (2604.06970, LiteLLM). Renaming one would not be a contribution.
- **Go on a narrower, falsifiable measurement contribution plus an open artifact.**
  - **Gap.** The only client-side study evaluates in simulation, with latency driven by *output*
    tokens and no prefill interference. Server-side papers assume control of the server. Nobody has
    tested, on a production black-box endpoint, whether a tenant's own long-context (prefill-heavy)
    requests delay its short decision requests, how much of that is visible from the client, and
    whether client-side admission can recover it without quality loss.
  - **What we add.**
    - Frozen real agent decisions with an oracle quality score.
    - Open-loop arrival replay, which avoids coordinated omission.
    - Cache warmth removed by construction: a fresh namespace per request, verified from reported
      `cached_tokens`.
    - Time-to-valid-action (TTVA) rather than TTFT as the latency metric.
    - Path-RTT probes, to separate gateway/network delay from model-server delay.

## Hypothesis (preregistered for the pilot)

*H1 (interference).* On a given hosted endpoint, adding a stream of fresh ~16k-token decision
requests (offered at λ_long) to a stream of fresh ~1.2k-token decision requests (λ_short, unchanged
arrival times) **raises p95 TTVA of the short requests**, compared with the same short trace alone.

*H2 (client mitigation).* Under the identical mixed trace, the candidate client policy
**`defer-long`** reduces short-request p95 TTVA versus FIFO admission at the same concurrency cap
C by ≥ 20%, and does not reduce useful decisions per second. The policy keeps at most L long
requests in flight, and holds a long while any short is in flight unless it has waited ≥ τ
(starvation bound).
- Useful decisions = valid legal action within a configured 2.5 s service objective. This is a
  configured SLO, not a Tetris game mechanic.
- Quality is held: short-request legal-action rate within −2 pp and mean oracle regret within +0.15
  of FIFO (noninferiority margins).
- Cost is held: same requests, so the same billed tokens up to output-length noise.

*Baseline B2 (native knob).* FIFO plus vLLM-style `priority` hints (short 0, long 10). The endpoint
accepts the field (HTTP 200 for −5/0/5 on both models), but acceptance does not show it is honored.
If B2 matches `defer-long`, client deferral is unnecessary on that endpoint.

**What refutes it.**
- H1 is refuted if the paired p95 difference (mixed FIFO − short-only) has a bootstrap CI covering 0,
  or is smaller than the path-RTT probe variation. In that case there is nothing for client
  admission to fix, and simulation-only client-side gains do not transfer.
- H2 is refuted if the CI for (defer-long − FIFO) short p95 excludes a 20% reduction, or if useful
  decisions/s or quality fall outside the margins, or if long-request completion/latency degrades
  beyond the starvation bound.

**Claim scope.** Endpoint-observed effects only. Batching, chunk size, GPU KV memory, replica count
and routing are not observable. "Interference" means interference observed at the endpoint, not
proven GPU prefill interference. Any server mechanism discussed is a hypothesis consistent with the
data, not a measurement. Prior evidence that motivates the endpoint choice comes from iteration 4
(historical): DeepSeek-V4-Flash cold prefills of ~15k tokens serialized at ≈3 s each, while
gpt-oss-20b absorbed 16 concurrent contexts.
