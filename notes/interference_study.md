# Do long-context requests delay short Tetris decisions on a hosted endpoint? (iteration 8)

Status: **pilot** (dev corpus, 4 blocks per model). Overlap audit, go/no-go and the preregistered
hypotheses are in [`interference_novelty_matrix.md`](interference_novelty_matrix.md). Config:
[`configs/interference/pilot.yaml`](../configs/interference/pilot.yaml).

There are three kinds of evidence below and they are never pooled:

| Label | What it is | Where |
|:--|:--|:--|
| **LIVE** | Real requests to Tensormesh serverless (`serverless.tensormesh.ai`), price-card cost estimates | `runs/explore/interference/{calibration,live_pilot}` |
| **SIMULATED** | `llm/endpoint_sim.py` with assumed replica counts and step budgets; validates the pipeline and shows the two regimes | `runs/explore/interference/{sim_pilot,sim_calibrated}` |
| **MOCK** | Fake streaming client in `tests/test_interference.py`; checks accounting only | pytest |

RESULTS_PLACEHOLDER
