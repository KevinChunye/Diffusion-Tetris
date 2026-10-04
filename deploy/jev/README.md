# Playing the paper's Tetris games with Jev

Jev (TypeSafe) is a hosted decision model: it runs on TypeSafe's servers and is reached through
`POST https://api.typesafe.ai/v1/systemone`. Nothing heavy runs on your machine; this folder packages the
game engine and a small HTTP client, so an 8 GB laptop (or a 1–2 GB VM) is plenty. The runner asks Jev
exactly the questions Intelif answered in the paper (same board, same options, same instruction, same
three games), pins the model to `jev-1.13.0`, and logs every call raw.

You need a TypeSafe API key (early access). Three 100-piece games are about 240 calls of ~800 input tokens,
roughly 0.2M input tokens: under $0.01 at the listed $0.042 per million input tokens (output is free).

## Option A: Docker (recommended)

From the repository root:

```bash
cp deploy/jev/jev.env.example deploy/jev/jev.env      # put your key in jev.env; it is git-ignored
GIT_COMMIT=$(git rev-parse HEAD) docker compose -f deploy/jev/docker-compose.yml run --rm jev-tetris
```

Results are written to `runs/explore/scaleup/jev/` on your machine. A 5-move smoke test first:

```bash
docker compose -f deploy/jev/docker-compose.yml run --rm jev-tetris \
    --seeds 1000 --pieces 5 --out runs/explore/scaleup/jev_smoke
```

## Option B: plain Python (no Docker)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r deploy/jev/requirements.txt
export TYPESAFE_API_KEY=...            # or put it in your shell's secret store; never commit it
python -m llm.decision_model --backend jev --seeds 1000,1001,1002 --out runs/explore/scaleup/jev
```

## Option C: a cloud session

Add `TYPESAFE_API_KEY` as an environment variable in the cloud environment's settings and start a new
session; Option B then runs as-is (the endpoint is reachable from the container).

## What gets written

| File | Content |
|:--|:--|
| `calls.jsonl` | one record per call: every option with its description and Jev's probability, Jev's `confidence` (not a probability), latency including the network, input tokens, the served model id |
| `steps.csv` | one row per move, same schema as the language-model and Intelif games (replayable, GIF-ready) |
| `episodes.csv` | score, lines, pieces per game |
| `manifest.json` | pinned model, endpoint, client platform, git commit, start time (never the key) |

## After the run

Commit `runs/explore/scaleup/jev/` (not `jev.env`) and push. Then, anywhere with the repository:

```bash
python -m llm.scaleup_analysis decision runs/explore/scaleup/jev   # oracle regret and scores
python paper/build.py                                              # the paper adds Jev to Figure 6 and Section 6.5
```

Latency measured this way includes your network round trip to TypeSafe, which is the latency an agent
on your machine actually sees.

## The Jev lab app (optional)

`github.com/cobusgreyling/Jev` is an unofficial companion app (a local web UI on port 7872) for exploring
Jev's Choice/Score/Noul questions with the same key. It is not needed for the paper's measurements; if you
run it on a server, keep it behind a reverse proxy (Caddy or Nginx) and never expose the key.
