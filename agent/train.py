"""Training entrypoint.

    python -m agent.train --env mock --episodes 500
    python -m agent.train --env genlayer --address 0x... --episodes 30

MockEnv is the default: it costs nothing and needs no network, so this is
also what CI runs. Switch to --env genlayer only once you have a deployed
contract address (see agent/deploy.py or `genlayer deploy`).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from agent.agent import QLearningAgent
from agent.env import DEFAULT_MAX_STEPS, GenLayerEnv, MockEnv

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_Q_TABLE_PATH = REPO_ROOT / "agent" / "q_table.json"
DEFAULT_LOG_PATH = REPO_ROOT / "logs" / "training.txt"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the CrisisNegotiator RL agent.")
    parser.add_argument("--env", choices=["mock", "genlayer"], default="mock")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument(
        "--address",
        default=os.environ.get("GENLAYER_CONTRACT_ADDRESS"),
        help="Deployed contract address (required for --env genlayer). "
        "Falls back to the GENLAYER_CONTRACT_ADDRESS env var.",
    )
    parser.add_argument(
        "--chain",
        default="localnet",
        choices=["localnet", "testnet_asimov", "testnet_bradbury", "studionet"],
    )
    parser.add_argument("--private-key", default=os.environ.get("GENLAYER_PRIVATE_KEY"))
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--epsilon-decay", type=float, default=0.99)
    parser.add_argument("--epsilon-min", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--resume", action="store_true", help="Load agent/q_table.json before training."
    )
    parser.add_argument("--q-table-path", default=str(DEFAULT_Q_TABLE_PATH))
    parser.add_argument("--log-path", default=str(DEFAULT_LOG_PATH))
    return parser.parse_args(argv)


def make_env(args: argparse.Namespace):
    if args.env == "mock":
        return MockEnv(max_steps=args.max_steps, seed=args.seed)
    if not args.address:
        raise SystemExit("--env genlayer requires --address (or GENLAYER_CONTRACT_ADDRESS)")
    return GenLayerEnv(
        address=args.address,
        chain=args.chain,
        private_key=args.private_key,
        max_steps=args.max_steps,
    )


def run_episode(env, agent: QLearningAgent) -> tuple[float, str]:
    state = env.reset()
    episode_reward = 0.0
    last_reason = ""
    while not env.is_episode_done():
        action_idx, action = agent.select_action(state)
        reward, reason, next_state = env.step(action)
        agent.update(state, action_idx, reward, next_state)
        state = next_state
        episode_reward += reward
        last_reason = reason
    return episode_reward, last_reason


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    agent = QLearningAgent(
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        seed=args.seed,
    )
    if args.resume and Path(args.q_table_path).exists():
        agent.load(args.q_table_path)
        print(f"Resumed Q-table from {args.q_table_path} ({len(agent.q_table)} states)")

    env = make_env(args)

    Path(args.log_path).parent.mkdir(parents=True, exist_ok=True)
    log_lines: list[str] = []
    rewards: list[float] = []
    start = time.time()

    for episode in range(1, args.episodes + 1):
        episode_reward, last_reason = run_episode(env, agent)
        agent.decay_epsilon()
        rewards.append(episode_reward)

        window = rewards[-20:]
        rolling_avg = sum(window) / len(window)
        log_lines.append(
            f"episode={episode} reward={episode_reward:.3f} "
            f"rolling_avg={rolling_avg:.3f} epsilon={agent.epsilon:.4f} "
            f"reason={last_reason[:60]!r}"
        )

        if episode % 10 == 0 or episode == args.episodes:
            print(
                f"[{episode}/{args.episodes}] reward={episode_reward:.2f} "
                f"rolling_avg={rolling_avg:.2f} epsilon={agent.epsilon:.4f} "
                f"reason={last_reason[:60]!r}"
            )

    elapsed = time.time() - start
    final_window = rewards[-20:]
    final_avg = sum(final_window) / len(final_window)
    summary = (
        f"Training complete: {args.episodes} episodes in {elapsed:.1f}s, "
        f"env={args.env}, states_seen={len(agent.q_table)}, "
        f"final_rolling_avg={final_avg:.3f}, final_epsilon={agent.epsilon:.4f}"
    )
    log_lines.append(summary)
    print(summary)

    Path(args.log_path).write_text("\n".join(log_lines) + "\n")
    agent.save(args.q_table_path)
    print(f"Saved Q-table to {args.q_table_path}")
    print(f"Wrote training log to {args.log_path}")


if __name__ == "__main__":
    sys.exit(main())
