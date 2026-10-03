# Tetris benchmark for open-weight agents and prefix-cache efficiency

Status: research protocol and initial implementation, 2026-10-02 (America/New_York).
The new validation runs are **offline mocks**, not model-quality or serving-performance results.
Existing `runs/explore/` results are historical pilot evidence; their conclusions need replication
with the revised instrumentation. No new paid Tensormesh requests were made in this session.

## Research question and contribution

At a fixed observation/action interface, which open-weight model and inference policy provide
useful Tetris decisions at the lowest latency and estimated cost? How does reuse of agent context
change that tradeoff under concurrency and deadlines?

The contribution should be a reproducible **quality–serving-efficiency benchmark**, with an
independent game-quality track and controlled serving experiments. Tetris itself is not a novel
LLM benchmark. A single familiar game does not establish general intelligence. Exact prefix
caching is expected to preserve model computation; the relevant quality effect is indirect,
through time budgets, context policy, and the number of candidates one can afford.

## Primary-source research

The following public primary sources were retrieved and read. ARC Prize, METR and lmgame websites
were denied by the egress proxy; their official public GitHub sources were accessible. Findings
below concern the retrieved documentation, not an exhaustive survey or verified current leaderboard.

| Source | Verified relevance | What to adopt | What does not transfer automatically |
|---|---|---|---|
| [LMGame Bench / GamingAgent](https://github.com/lmgame-org/GamingAgent), [paper](https://arxiv.org/abs/2505.15146) | README explicitly lists Tetris, Sokoban, 2048, Candy Crush and other games; Gymnasium interfaces; separate single-model and agent-harness evaluations | Closest Tetris comparison; distinguish raw model and harness-assisted tracks, retain episode logs and replays | Scores are not comparable until rules, actions, observations, termination and budgets match |
| [BALROG](https://github.com/balrog-ai/BALROG), [paper](https://arxiv.org/abs/2411.13543) | Long-horizon interactive LLM/VLM evaluation, configurable text/image histories, local vLLM and API clients | Standard agent interface, explicit history budget, local-serving follow-up, cross-game extension | Its cross-environment claims cannot be inherited by Tetris alone; it is not evidence that Tetris is in its suite |
| [ARC-AGI-1](https://github.com/fchollet/ARC-AGI) | Grid-based task data; explicit separation of development and evaluation tasks and warning against iterative tuning on evaluation data | Freeze evaluation seeds/state corpus before model selection; define generalization splits | Known Tetris rules and legal-action lists measure control, not novel rule induction in ARC |
| [ARC-AGI-3 toolkit](https://github.com/arcprize/ARC-AGI), [agent examples](https://github.com/arcprize/ARC-AGI-3-Agents) | Interactive environment API, local/online execution, action spaces, scorecards and example agents | Replayable action/observation trajectories and per-environment success criteria | Do not call Tetris an ARC-AGI score or claim equivalent human-relative efficiency |
| [METR time-horizon analysis](https://github.com/METR/eval-analysis-public), [paper](https://arxiv.org/abs/2503.14499) | Fits success against log2 human completion time; reports success-threshold horizons and bootstrap uncertainty | Preregister success, report reliability versus task difficulty, bootstrap at independent task/seed level | Surviving 100 pieces or taking 60 API seconds is **not** a METR human-time horizon; that requires measured human task times |
| [METR Task Standard](https://github.com/METR/task-standard) | Explicit task setup, agent interaction and scoring contract | Version environment and grader separately from model clients | Implementation compatibility alone does not establish comparable difficulty |
| [AgentBench](https://github.com/THUDM/AgentBench), [paper](https://arxiv.org/abs/2308.03688) | Eight interactive agent environments including a digital card game | Treat tool/API failures and agent failure modes as part of evaluation | Not a verified Tetris-specific baseline |

ARC-AGI-2's linked raw README could not be retrieved at the attempted main/master paths; no new
claims about its current contents are based on those failed requests. Candidate model IDs in
`configs/model_settings.yaml` and the price card are historical; confirm the current `/v1/models`
catalog and each selected model's weight release/license before calling the final suite open-weight.
An API model name does not prove the exact checkpoint, quantization or serving deployment.

## Repository audit

Reuse `TetrisGym_updated`, `harness`, `llm.run_pilot`, `llm.oracle`, `llm.cache_lifetime`,
`llm.fanout` and existing deterministic episode-piece tests. Do not replace the working simulator.
The environment is 10×20, one next-piece preview, a seeded uniform independent piece generator
(not a seven-bag randomizer), rotation/column placement followed by a straight drop, and scores
2/5/15/60 for one/two/three/four cleared lines. There is no real-time gravity control in this track.
These choices must be in a published benchmark card; other Tetris scores are not interchangeable.

The old exploratory notes raise useful hypotheses: append history might reuse cache better than
sliding windows; history might harm play; provider pricing might not pass cache savings through.
Those are not established mechanisms from this audit. In particular:

- Concurrent jobs are not sufficient randomization; fixed scheduling can correlate model/arm with load.
- Fresh-prefix versus repeated-prefix latency does not by itself measure GPU cache capacity, routing,
  cache eviction policy, prefill FLOPs, or billed cost.
- `cached_tokens=0` is a reported zero, not proof that all cache tiers missed. Missing usage is unknown.
- A listed cached-token price produces an estimate, not an invoice. Never describe uncached-token
  arithmetic as directly measured compute.
- The approximate oracle uses heuristic rollouts/search; its top action is not a ground-truth solution.
- Fallback-assisted score measures the whole system. It must not be attributed entirely to the model.

## Three evaluation tracks

### A. Game decision quality

Primary raw track: full current board, current/next piece, legal placements (rotation and columns),
no simulated lines/height/holes annotation; temperature 0, explicit per-model reasoning settings,
JSON-object output, fixed completion-token cap. This is still legal-action-assisted symbolic play,
not a visual end-to-end task. Report legality separately; a schema enum of legal IDs would constrain
away that failure mode. Keep constrained decoding as a separate ablation.

Assisted track: add the existing per-placement simulator features. Label this **model + simulator**.
Compare random legal, deterministic fallback, greedy, beam search, and available DQN/diffusion
checkpoints under the same seeds and preview information. Record search wall time and learned-model
training/data provenance; don't compare pretrained LLM costs to a free trained baseline without
stating what is excluded. The onboarding diffusion checkpoint is a smoke artifact, not a competitive baseline.

Primary outcomes: lines cleared per episode and survival to a fixed piece cap, with the cap and
termination reason explicit. Secondary outcomes: native score, legal proposal rate, API success,
fallback rate, truncation rate, accepted-model-action oracle regret, executed-system regret, and
approximate epsilon-best action agreement (ties count as ties).

Use disjoint development and held-out episode seeds. Select prompts, models and reasoning settings
on development only. A piece cap is administrative censoring; a budget abort or API crash is not a
normal completed game. Report completion/failure counts alongside complete-case scores. Frozen
state accuracy is a separate distribution and cannot replace closed-loop performance.

### B. Frozen-state cache replay

Fix a corpus of recorded Tetris decisions, including intact prior turns from each trajectory.
Sample from multiple generators (random, greedy, beam and development LLMs), heights and hole counts,
with both easy and near-topout states. Reserve a test corpus before tuning. Hash the corpus and prompt
payloads. The implemented `llm.cache_replay` accepts this standard `steps.csv` format; it currently
samples uniformly from its input, so build/stratify the corpus before the final study.

For each selected state, hold model, state, history, output settings and completion cap fixed:

1. Create two unique, fixed-length namespaces at the start of the system message.
2. Prime namespace A once, logging its latency/tokens/cost separately.
3. Randomize the order of A's exact repeat and B's fresh-prefix measurement.
4. Log both request hashes, shared task hash, actions, usage-presence flags, first chunk, first
   content/reasoning output, final-attempt latency and retry-inclusive API lifetime.

This is an **encouragement of prefix reuse**, not a cache-disable intervention. The fresh namespace
can still share a short template prefix and can affect model behavior; inspect reported hit counts
and action agreement, and use multiple nonce assignments. The repeat may also benefit from unspecified
provider response caching. Do not assert internal KV attribution without provider controls or a
self-hosted cache-on/off experiment. Prime costs count in experiment totals; steady-state warm cost
must be shown separately from warm-up-amortized cost. Priming order is necessarily asymmetric;
repeat blocks at different times and report order, load and delays.

First pilot: 6 pairs × 3 calls × one model. Scale only after checking telemetry. Paper study:
matched context-length bins (~1k/4k/16k reported tokens), 30+ state pairs per model and bin, at least
three time blocks. Stratify using actual token counts, never assume four characters per token.
Also compare exact repeats to a shared-prefix/new-state suffix variant before generalizing to
sequential gameplay. Existing fanout/lifetime runners can supply follow-up workloads after their
accounting is audited; the new replay runner is sequential and does not claim concurrency results.

### C. Serving load and deadline effects

Run contexts N={1,2,4,8,16} and idle gaps {0,5,30,120} seconds with matched prefix/token lengths;
start with N≤4, stop on sustained overload, and repeat in time blocks. Keep per-attempt errors and
429s; compare achieved throughput rather than launched concurrency. Idle time is an experimental
condition, not model latency. Distinguish an idle isolated context from eviction pressure caused
by other contexts. Endpoint results are deployment-specific, not intrinsic properties of weights.

Replay deadlines {0.5,1,2,4} seconds using request-lifetime measurements, then evaluate game score
under a declared deterministic fallback. Current `deadline_ms` is **post-response late detection**:
it does not cancel the request or execute fallback at the deadline. A true online real-time study
requires an asynchronous scheduler and wall-clock action dispatch; that extension is outstanding.
Tetris has no natural gravity deadline in the current setup, so call these imposed service budgets.

## Model selection, statistics and staged budget

Start with the historically configured `openai/gpt-oss-20b`; verify catalog and pricing live. Add
`openai/gpt-oss-120b` and one different available family after the smoke run. Select 3–4 verified
open-weight deployments spanning size/reasoning and cache-pricing behavior. Do not assume a provider's
catalog or cached-price policy is unchanged from `notes/tensormesh_probe.md`.

Development pilot: `configs/benchmark/pilot.yaml`, four arms, four paired seeds, 50-piece cap,
concurrency 2, estimated $1 run budget. The old runner's budget is a **soft observed-spend stop**:
in-flight calls, retries and missing billing telemetry can exceed it. It is not a hard billing cap.
New pilots disable retries; the replay runner reserves a full advertised context's price before
starting each three-call pair and stops on failed calls or unknown usage. Price-card reservation
still cannot guarantee the provider's invoice. No large live grid is implied by this initial pilot.

Final sample size is chosen from development variance and a preregistered smallest effect of
interest. Start planning around 30 paired held-out seeds × 200 pieces × 3 time blocks, then use
pilot-based power analysis; 30 is a starting point, not a guarantee of power. Do not tune on those
seeds. Predeclare one primary contrast (append vs stateless within model) and raw/assisted tracks;
apply Holm correction to confirmatory multiple contrasts or label them exploratory.

For games, bootstrap paired differences at the episode-seed level, retaining time blocks as a
second clustering factor in the final analysis. For frozen states, cluster by source episode and
block rather than treating every turn as independent. Report CIs and denominators; mock intervals
only test report plumbing. `llm.benchmark_report` implements single-block seed bootstrap and paired
line differences; hierarchical blocks, power analysis and multiplicity correction remain final-study work.

Report a Pareto plot of lines/success against p50/p95 API lifetime and estimated dollars per completed
100 decisions, with failure rates visible. Also show first-output latency, prompt/output/reasoning
usage when available, reported cached fraction **only for calls with telemetry**, priming cost, and
aggregate game throughput. Do not subtract a guessed network floor to claim exact prefill throughput.
Do not claim measured GPU memory, FLOPs or energy from this API alone.

## Implemented changes and commands

- `llm/tensormesh_client.py`: separate final-attempt and complete request lifetime; reset partial
  measurements on retry; distinguish missing token/cache telemetry; flag unknown cost estimates.
  Historical `latency_s` remains final-attempt latency. Retry billing remains potentially unobserved.
- `llm/llm_policy.py`: deadlines use complete API lifetime; log whether the model action was accepted.
- `llm/run_pilot.py`: resolved-config/provenance/mock manifest; refuse overwriting prior runs; preserve
  a zero-decision failure report; accepted-model regret separate from fallback-assisted regret.
- `llm/cache_replay.py`: immutable state input, randomized measured-pair order, priming accounting,
  no retries, model/catalog checks, raw request hashes, telemetry-aware paired output and spend reservation.
- `llm/benchmark_report.py`: failure/telemetry coverage, episode-bootstrap summaries, paired line differences.
- `tests/test_benchmark_measurement.py`: missing telemetry, retry timing/reset, immutable replay and fallback tests.

Initialize the already installed environment:

```bash
cd /workspace/Diffusion-Tetris
source .venv/bin/activate
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy MPLBACKEND=Agg
export MPLCONFIGDIR=/workspace/.cache/matplotlib XDG_CACHE_HOME=/workspace/.cache
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
python -m pytest tests -q
```

A short offline run (full config's oracle is more expensive than the unit-test oracle):

```bash
python -m llm.run_pilot --config configs/benchmark/pilot.yaml --mock \
  --seeds 2000,2001 --max_pieces 8 --workers 2 --out_dir runs/benchmark/dev_mock
python -m llm.benchmark_report --dir runs/benchmark/dev_mock --baseline raw_stateless
python -m llm.cache_replay --steps runs/benchmark/dev_mock/steps.csv \
  --out_dir runs/benchmark/cache_mock --pairs 6 --mock
```

For live use, first bind `TENSORMESH_API_KEY` securely in environment settings to
`serverless.tensormesh.ai`; the requirement/domain were saved to the draft. No key value belongs
in Git or chat. The inspected runtime and configuration had no binding. After applying settings,
verify `TensormeshClient().list_models()` and one legal action response, then use a **fresh output
directory** and omit `--mock`. Keep initial live experiments within the small pilot budgets above.
Do not call `probe_tensormesh` across the entire catalog before choosing the initial model.

## Outstanding before publication

Validation in this session: **18 tests passed**, with one existing imageio `fps` deprecation
warning. `runs/benchmark/offline_validation/` contains 64 mock decisions (four arms × two seeds ×
eight turns), oracle scoring and a benchmark report. `runs/benchmark/cache_offline_validation/`
contains 18 mock calls, six complete cache pairs and separate priming measurements. These generated
outputs are ignored by Git; no mock latency, quality or cost is empirical evidence about Tensormesh.

Live API validation and new real-model results; current license/checkpoint/price verification;
stratified frozen-state corpus; matched implementation comparison with LMGame Bench Tetris;
randomized time-block scheduling; hard billing-limit support where the provider exposes it;
per-attempt billing logs; token-level decode timing; actual deadline scheduling; independent
human-baseline study if making human-relative claims; held-out replication and power analysis.

The first deliverable is a defensible small pilot and reproducible instrumentation. A full
leaderboard or paper claim follows only after those measurements, not from the offline smoke run.
