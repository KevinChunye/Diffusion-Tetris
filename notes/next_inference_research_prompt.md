# Handoff prompt: inference efficiency beyond the existing Tetris studies

Copy the following into a new coding/research session with this repository and the benchmark PR available.

---

Continue KevinChunye/Diffusion-Tetris as an inference-systems research project. I want to contribute
at the prefill, decode, serving, or efficiency-evaluation layer. Use Tetris as a reproducible agent
workload and quality constraint. Do not simply rerun or enlarge the existing game leaderboard.

First read README.md, notes/exploration_summary.md, notes/exploration_log.md,
notes/tensormesh_probe.md, notes/tetris_benchmark_protocol.md, and the relevant llm/ implementations.
Inspect the current branch and PR changes before editing. Preserve prior experiment artifacts.
Use the existing checkout; do not create another worktree unless requested.

## What exists already

The historical exploration covers model/API capability probes; stateless, append, sliding-window
and compact histories; cache lifetime/capacity and idle/concurrency effects; keep-alive; shared-prefix
fan-out; price-card comparisons; a model/reasoning ladder; and deterministic piece-sequence fixes.
Do not present these as new contributions. Historical results have the caveats in the protocol.

The current benchmark upgrade adds retry-inclusive API lifetime, missing-telemetry flags,
model-versus-fallback accounting, frozen-state fresh/repeated-prefix probes, seed-bootstrap reports,
run provenance, and a small raw/assisted gameplay config. It passed 18 offline tests at handoff;
64 mock gameplay decisions and six mock cache pairs are plumbing validation only. No new live-model
results were collected for that upgrade. The last credential check found no usable runtime binding;
recheck now because the user may have applied environment settings since then.

## Candidate contribution to investigate

Study **prefill–decode interference in mixed agent workloads**, then evaluate a concrete scheduling
or admission-control policy for latency-sensitive short Tetris decisions mixed with long-context
requests. The target is better tail latency and useful decisions per second under matched load,
without changing model weights, game observations, prompts, or action quality.

This is a candidate, not an established novel idea. Search primary papers, official code and current
documentation before implementation. Compare at least vLLM chunked-prefill/continuous batching,
Sarathi-Serve, DistServe, SGLang/RadixAttention and prefix-aware or SLO-aware scheduling. Check
whether token-length-aware admission, separate queues, prefix-local routing, or decode prioritization
already solve the proposed problem. Build a novelty matrix:

- mechanism and assumptions;
- existing implementation and closest experiment;
- what the Tetris workload adds beyond synthetic prompts;
- the precise unresolved hypothesis;
- which measurements would refute it.

If the algorithm is already known, contribute a rigorous reproducible workload/evaluation artifact
or identify a narrower missing mechanism. Do not rename an existing scheduler and call it novel.
Do not make this a history-policy, cache-capacity or reasoning-budget sweep under a new name.

## Capability gates

Inspect only the names/presence of credentials and existing bindings; never print secret values.
Verify the live Tensormesh catalog and a tiny request through the configured proxy. Do not assume
catalog IDs, licenses, quantization, usage fields or historical prices remain current. Reuse the
existing secret binding; do not ask for another key just because a CLI or unrelated variable is unset.

Determine whether the environment has a GPU and a controllable vLLM/SGLang server, or whether only
Tensormesh's public API is accessible. API access alone does not authorize or enable provider backend
changes. Do not claim kernel, batching, GPU KV memory, energy or GPU-utilization improvements from
client-side timing. Do not rent GPU resources or create external infrastructure without authorization.

With API-only access, evaluate client-side admission/scheduling and report endpoint-level effects.
Use terminology such as interference observed at the endpoint, not proven GPU prefill interference.
With an authorized controllable backend, add request queueing, scheduler, prefill/decode and hardware
telemetry, then test the mechanism causally with a matched baseline. Document unavailable capabilities
and continue all useful CPU/offline work.

## Experimental design

1. Freeze real Tetris state/history traces from multiple seeds and generators. Preserve the exact
   same requests, model, output caps and arrival trace across compared schedulers. Separate held-out
   seeds from development. Use the existing frozen-state replay utilities instead of rebuilding them.
2. Construct mixed workloads with short, time-sensitive action requests and long-context requests
   that also have a defined task. Sweep arrival rate, long-request share, and measured input/output
   lengths. Separate warm-prefix and fresh-prefix blocks; cache warmth must not explain a scheduler gain.
3. Start with FIFO client admission at fixed concurrency as the client baseline. On a controlled
   backend, use its documented default scheduler and native chunked-prefill options as stronger
   baselines. Select one candidate policy only after the literature audit. Log queue entry, admission,
   first output and completion using monotonic clocks; bound starvation of long requests.
4. Replay externally scheduled arrivals rather than issuing a new request only when the last finishes;
   otherwise overload is hidden by coordinated omission. Record offered and achieved load, dropped
   work, timeouts, all retries, backoff and unfinished requests. Preserve per-attempt usage/cost when
   exposed. Use backpressure and stop criteria rather than unbounded queues.
5. Distinguish first SSE chunk, first reasoning/content output, first valid actionable answer, and full
   completion. SSE chunks are not necessarily individual tokens. Report stream-chunk gaps as such;
   claim inter-token latency only with actual token timing from the backend. Include parsing and queue
   delay in time-to-action, and report API lifetime separately.
6. Primary outcomes: p95 time-to-valid-action and useful accepted decisions per second at fixed
   offered load, subject to a preregistered task-quality noninferiority margin. Also report p50/p99,
   deadline misses, fairness/starvation, failure rate, total makespan, reported token throughput and
   estimated cost. Distinguish configured SLOs from native Tetris game mechanics.
7. Score identical frozen decisions with legal-action rate and approximate oracle regret, then run a
   small closed-loop confirmation on paired held-out seeds. Fallback-assisted game score is a system
   metric, never pure model accuracy. Do not let fast invalid/empty responses count as throughput wins.
8. Randomize policy block order, repeat at multiple times, and bootstrap paired differences by source
   episode and arrival-trace/time block. Use pilot variance and a minimum useful effect to plan sample
   size; do not treat correlated calls as independent trials. Report negative results and failure counts.
9. On a controllable backend, hold hardware, model revision, quantization, tokenizer, cache settings,
   sampling and serving version constant. Measure queue, prefill and decode separately, and include
   native optimized scheduler baselines. An improvement must survive those controls.

## Execution and spending

Begin with offline trace generation, fake-stream and scheduler tests. Verify that every baseline
receives identical work, no requests disappear, partial streams and retries remain accounted for,
and starvation/deadline behavior matches the stated policy. Use fresh run directories and persist
requests/results incrementally with commit/config hashes and an explicit mock/live marker.

Then run a tiny live smoke if credentials are usable. Keep initial live work within an aggregate
$1 estimated budget, using conservative token reservations across concurrent requests and fail-closed
handling of missing price/usage information. Record that estimates are not invoice guarantees;
use a provider-enforced spend limit if available. Prepare a costed plan before expanding beyond
that initial budget. Do not launch the old all-catalog probe by default.

## Deliverables and acceptance

- A cited novelty/overlap matrix with a go/no-go decision on the proposed contribution.
- One falsifiable hypothesis and a clearly scoped API-only or controlled-backend claim.
- A versioned trace workload, runnable baseline and candidate scheduler/config, meaningful tests,
  and complete timing/error/cost accounting.
- A reproducible pilot report with paired uncertainty, raw evidence, throughput/latency/quality
  tradeoffs and limitations. Clearly separate mocks, historical pilots and new live results.
- A follow-up PR with the implementation, commands and experiment plan; never claim runs that were
  not executed. If external access blocks live work, finish independent work and identify exactly
  what capability remains necessary.

Keep the work focused on the inference-system question. Do not spend the task retraining diffusion
models, creating another model leaderboard, or restating the existing cache-history findings.
