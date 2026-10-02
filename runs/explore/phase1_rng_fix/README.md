# Phase 1: v1 diffusion config before/after the piece-RNG fix

Config `configs/eval_run.yaml` with `sampling_constraints=mask_logits`, `rerank_mode=heuristic`,
`num_candidates=64`, `horizon=8`, 100 episodes, seed 0, CPU.

Deviations, needed to run on a CPU-only box:
- **Checkpoint.** The paper's checkpoint is not in the repo. Both runs use the same freshly trained
  H=8 PlanDenoiser: 200 greedy-heuristic-teacher episodes (top 50% kept, 63.6k sequences), 10 epochs.
- **Episode cap.** Episodes are capped at **50 pieces** (v1 used 2000), because one decision costs
  ~2 s on CPU.

| run | code |
|:--|:--|
| before | commit aeea9cc (original), plus only the verified-equivalent O(width) drop height so it runs at the same speed |
| after | this branch: per-episode seeded game RNG, independent simulation RNGs (`clone_for_simulation`) |

Result (`comparison.md`). Mean score is 23.90 [21.92, 25.99] before and 24.12 [22.32, 26.04]
after; pieces 47.96 vs 48.40; top-out before the cap 14% vs 13%. The bug did not bias the mean,
since both runs draw i.i.d. pieces. It broke reproducibility and pairing:
- "Before" draws pieces from the global `random` stream, which the eval runner never seeds, and
  which every simulation deepcopy also consumes. Its piece sequences cannot be reproduced, and two
  agents on the "same" seed play different games (`tests/test_env_rng.py`).
- "After", each episode's pieces are a function of `episode_seed = seed + episode_idx` alone.

Decision time drops 8% (2213 → 2046 ms) from the cheaper simulation clone.
