"""Renders docs/learning_curve.png from logs/training.txt.

python -m agent.plot
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG_PATH = REPO_ROOT / "logs" / "training.txt"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "docs" / "learning_curve.png"

_LINE_RE = re.compile(
    r"episode=(?P<episode>\d+) reward=(?P<reward>-?\d+\.\d+) "
    r"rolling_avg=(?P<rolling_avg>-?\d+\.\d+) epsilon=(?P<epsilon>-?\d+\.\d+)"
)


def parse_log(log_path: str | Path) -> tuple[list[int], list[float], list[float]]:
    episodes: list[int] = []
    rewards: list[float] = []
    rolling_avgs: list[float] = []

    for line in Path(log_path).read_text().splitlines():
        match = _LINE_RE.match(line)
        if not match:
            continue  # skips the trailing "Training complete: ..." summary line
        episodes.append(int(match.group("episode")))
        rewards.append(float(match.group("reward")))
        rolling_avgs.append(float(match.group("rolling_avg")))

    if not episodes:
        raise ValueError(f"No episode lines found in {log_path}. Run agent.train first.")

    return episodes, rewards, rolling_avgs


def plot(log_path: str | Path, output_path: str | Path) -> None:
    episodes, rewards, rolling_avgs = parse_log(log_path)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(episodes, rewards, color="#9ca3af", linewidth=0.8, alpha=0.6, label="episode reward")
    ax.plot(episodes, rolling_avgs, color="#2563eb", linewidth=2.0, label="rolling avg (20 ep)")
    ax.set_xlabel("episode")
    ax.set_ylabel("reward")
    ax.set_title("CrisisNegotiator: RL agent reward vs. LLM-consensus judge score")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"Saved {output_path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plot the training reward curve.")
    parser.add_argument("--log-path", default=str(DEFAULT_LOG_PATH))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    args = parser.parse_args(argv)
    plot(args.log_path, args.output_path)


if __name__ == "__main__":
    main()
