# Notes

Plans, logs and verified facts behind the two studies in this repository. Files are grouped by study;
nothing here is needed to run the code.

## Evaluating Agents as Served, Not as Weights (`paper/paper.pdf`)

| File | What it is |
|:--|:--|
| [`scaleup_plan.md`](scaleup_plan.md) | Plans and hypotheses for experiments E1–E5, written before any data, with every later deviation and its UTC timestamp |
| [`model_architectures.md`](model_architectures.md) | Architecture facts for the nine served models, each with its source (Table 2) |
| [`decision_models.md`](decision_models.md) | Verified facts on Jev and Intelif: API format, pricing, licenses, reported latency |
| [`references_scaleup.md`](references_scaleup.md) | How each reference was checked |
| [`interference_study.md`](interference_study.md), [`interference_novelty_matrix.md`](interference_novelty_matrix.md) | The long-context interference study (E4): earlier runs, hypotheses fixed before the confirmatory run, overlap with prior work |
| [`tetris_benchmark_protocol.md`](tetris_benchmark_protocol.md) | Benchmark protocol for open-weight agents and prefix-cache efficiency |
| [`tensormesh_probe.md`](tensormesh_probe.md) | First probe of the serverless platform: models, cache reporting, concurrency |
| [`exploration_log.md`](exploration_log.md), [`exploration_summary.md`](exploration_summary.md) | Iterations 1–6 that preceded the paper's experiments, and what they found |
| [`next_inference_research_prompt.md`](next_inference_research_prompt.md) | Handoff prompt for follow-up work |

## Diffusion-MPC in discrete domains (arXiv 2603.02348)

| File | What it is |
|:--|:--|
| [`dataset_pipeline.md`](dataset_pipeline.md) | Expert dataset generation |
| [`phase3_method.md`](phase3_method.md) | Constraint-aware sampling and self-training |
| [`lit_review.md`](lit_review.md), [`related_work_paragraphs.md`](related_work_paragraphs.md) | Literature-to-implementation mapping and related work |
| [`experiment_board.md`](experiment_board.md), [`idea_backlog.md`](idea_backlog.md) | Experiment tracking and prioritized ideas |
| [`repro_workflow.md`](repro_workflow.md), [`workflow_chart.md`](workflow_chart.md), [`lightning_runbook.md`](lightning_runbook.md) | Reproduction workflow and running at scale on Lightning |
