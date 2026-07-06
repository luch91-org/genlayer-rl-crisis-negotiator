# Manifest emitter

This repository publishes a single `manifest.json` that the GenLayer RL Demo
Suite reads to render this agent. It is assembled from real, committed
artifacts only; nothing is invented, and fields that were never captured are
left out.

## Regenerate

From the repository root, in the agent virtualenv:

    python -m agent.emit_manifest

This writes `manifest.json` at the repository root.

## What it contains

- domain metadata, and the deployed contract address and chain
- the reward equivalence principle and a sample reward prompt, read from
  `contracts/logic.py`
- the learning curve and exploration rate, parsed from `logs/training.txt`
- random and greedy-only baselines from `agent/manifest_data/baselines.json`
- one captured on-chain consensus receipt from `agent/manifest_data/capture.json`
- the saved live run, replayed from `logs/training_live_studionet.txt`

The reward function is immutable on-chain. The agent optimizes it and cannot
rewrite it.
