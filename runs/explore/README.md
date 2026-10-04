# Raw logs of the serving study

Everything the paper *Evaluating Agents as Served, Not as Weights* reports is computed from the files here.
Every model call was logged raw (request metadata, status, reported tokens, cached tokens, latency, response
text), so any number or figure can be recomputed, and new analyses need no new calls.

| Directory | What it holds | In the paper |
|:--|:--|:--|
| [`scaleup/`](scaleup/README.md) | E1 model ladder, E2 history policy, E3 cache capacity, E5 decision model (Intelif) | Figures 2–4, 6, 7; Tables 2–3 |
| `interference/confirm_gemma/`, `interference/confirm_deepseek/` | E4: long requests vs. short decisions on held-out traffic, four admission policies | Figure 5 |
| `interference/` (other folders) | E4's earlier runs, calibration, simulations and the frozen request corpora (`corpus_dev/`, `corpus_heldout/`) | Section 6.4 |
| `iter01/` – `iter06/` | Exploration iterations before the main experiments (history policies, price cards, cache lifetime, load, fan-out); `iter06/` is the nine-configuration ladder on seeds 1000–1002 | `iter06/`: Figure 1, Figure 6b, the three-seed run in Section 6.1 |
| `probe/` | First probe of the platform: models, cache reporting, concurrency, output format | Section 2 |
| `phase1_rng_fix/` | Before/after check of the seeded piece generator that makes every agent see the same pieces | Section 5 |
| `models/configs/` | The published `config.json` of each served model, used for the KV-cache estimates | Table 2 |
| `reference_scores.csv`, `reference_regret.csv` | Random and beam-search reference scores per seed, used to normalize scores | Section 5 |
| `spend.csv` | Cost and call count of every live run, in order | — |

The same-game animations are in [`gallery/`](../../gallery/README.md).
