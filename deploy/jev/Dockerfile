# Jev Tetris agent: plays the paper's games against TypeSafe's hosted Jev (POST /v1/systemone).
# The model runs at TypeSafe; this image is only the game engine and an HTTP client, so a laptop with
# 8 GB of RAM (or a 1-2 GB VM) is enough. Build from the repository root:
#   docker compose -f deploy/jev/docker-compose.yml run --rm jev-tetris
ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}
WORKDIR /app
COPY deploy/jev/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
COPY TetrisGame_updated.py TetrisGym_updated.py ./
COPY llm/__init__.py llm/decision_model.py ./llm/
ARG GIT_COMMIT=unknown
ENV GIT_COMMIT=${GIT_COMMIT} MPLBACKEND=Agg PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "-m", "llm.decision_model", "--backend", "jev"]
CMD ["--seeds", "1000,1001,1002", "--pieces", "100", "--out", "runs/explore/scaleup/jev"]
