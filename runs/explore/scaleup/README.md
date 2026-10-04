# Scale-up runs (2026-10-04): raw logs and where each figure comes from

Every API call and every decision-model call is logged raw, so any analysis or figure can be redone later.
Plans, hypotheses and every deviation (with UTC timestamps) are in `notes/scaleup_plan.md`.

| Directory | Experiment | Raw logs | Derived files | Paper |
|:--|:--|:--|:--|:--|
| `capacity/`, `capacity_part2/`, `capacity_part3/` | E3 cache capacity (9 models, N = 1–64) | `calls.jsonl` (every HTTP call: status, tokens, cached tokens, latency, text), `rows.jsonl` (one row per agent: warm-up and probe), `console.txt`, `manifest.json` | — | Fig. 4, 7 |
| `capacity_analysis/` | E3 pooled analysis | — | `agents.csv` (all agents, contended flag), `summary.csv`, `capacity.csv`, `excluded_rounds.csv`, `inference_check.json` | Fig. 4, 7 |
| `ladder/` | E1 model ladder (9 configs × 10 seeds) | `calls.jsonl`, `console.txt`, `manifest.json` | `steps.csv` (one row per decision, with oracle regret), `episodes.csv`, `summary_scaleup.csv`, `analysis.json` | Fig. 2 |
| `memory/` | E2 history policy (4 models × 3 policies × 9 seeds) | `calls.jsonl`, `console.txt`, `oracle_console.txt` | `steps.csv`, `episodes.csv` (stop_reason = budget marks excluded games), `summary_scaleup.csv`, `analysis.json` | Fig. 3 |
| `intelif/`, `intelif_b/` | E5 Intelif games (seed 1000; seeds 1001–1002 after a restart) | `calls.jsonl` (every option with description and probability, latency, input tokens), `console.txt`, `manifest.json`, `precision_check_*.txt` | `steps.csv`, `episodes.csv` | — |
| `intelif_all/` | E5 merged view | `calls.jsonl` (merged), `latency_quiet_console.txt` | `steps.csv` (with oracle regret), `episodes_scored.csv`, `analysis.json`, `latency_quiet.csv` | Fig. 6 |

E4 (interference) lives in `runs/explore/interference/confirm_gemma/` and `confirm_deepseek/`: `requests.jsonl` (every request
with arrival, admission, first byte, first valid action), `dispatch.jsonl`, `calls.jsonl`, `console.txt`, and the analysis
outputs (`summary.csv`, `paired.csv`, `overlap.csv`, `verdicts.json`, `report.md`); it is Fig. 5. Spend per run is in `runs/explore/spend.csv`.
The same-game animations (Fig. 1) are in [`gallery/`](../../../gallery/), rebuilt by `python scripts/build_gallery.py`.

Regenerate everything derived: `python paper/build.py` rebuilds all figures and the PDF.
