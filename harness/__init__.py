"""Train / deploy / visualize harness for every Tetris bot in the repo.

  python -m harness.train dqn|diffusion ...          train a bot (wraps the existing trainers)
  python -m harness.play --bot SPEC --gif out.gif    deploy a bot on seeded episodes, record a GIF
  python -m harness.replay --steps steps.csv ...     turn logged LLM episodes into (side-by-side) GIFs

GIF frames come from the repo's own renderer (TetrisGym.render(mode="rgb_array")) and
experiments.video_utils.save_video.
"""
