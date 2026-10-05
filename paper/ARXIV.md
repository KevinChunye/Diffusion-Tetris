# arXiv submission

Source: `paper/arxiv_source.zip` (rebuild with `python paper/build.py && python paper/make_arxiv.py`).
It contains `main.tex`, `main.bbl`, the style files and `figures/`; arXiv compiles it with pdflatex
(`\pdfoutput=1` is set) and needs no BibTeX run. The zip was test-compiled that way: 11 pages,
text identical to `paper/paper.pdf`.

## Form fields

**Title**

    Same Pieces, Different Servers: A Tetris Benchmark for AI Agents as Served

**Authors**

    Haochuan Wang

**Abstract** (1,195 characters; limit 1,920)

    An agent meets a model as served: through an endpoint with a price card, a shared cache and other tenants, or on whatever hardware a self-hosted model runs. Benchmarks rank the weights. We introduce a Tetris benchmark that measures what agents get from models as served: every move is scored against an oracle, and every agent receives the same pieces. In five pre-specified experiments with nine open-weight models on one serverless provider, plus an open decision model self-hosted on a CPU, we find that price and size do not predict decision quality; that resending history costs almost nothing when cached input is free, would cost eleven to twelve times more if it were not, and makes play worse; that deployments keep between 4 and more than 64 agent contexts warm, in line with their throughput rather than the model's KV-cache size; that an agent's own long requests slow its slowest short decisions more than tenfold, which a simple admission rule cuts by 59% without hurting play; and that the decision model plays mid-pack but takes 27 s per move on a CPU, about 650 times longer than reported on a GPU. Architecture predicts some of what an agent sees; the deployment sets the rest.

**Comments**

    11 pages, 7 figures, 3 tables. Code, raw logs and animations: https://github.com/KevinChunye/Diffusion-Tetris

**Primary category:** cs.LG (Machine Learning)

**Cross-lists:** cs.AI (Artificial Intelligence), cs.DC (Distributed, Parallel, and Cluster Computing),
cs.PF (Performance)

**License:** CC BY 4.0, or the arXiv perpetual non-exclusive license if a later venue may require it.

**Leave blank:** Report number, Journal reference, DOI, MSC class, ACM class.

## Before announcing

The footnote and the Comments field link the repository's default branch. Merge the paper branch into
`main` first so `gallery/`, `paper/` and `runs/explore/` are there when readers arrive.
