"""Emit manifest.json for the GenLayer RL Demo Suite.

The demo suite is a pure reader: each domain publishes one manifest.json and the
dashboard renders from it. This script assembles that manifest from real,
committed artifacts only:

  - contracts/logic.py            the reward equivalence principle and a sample
                                  reward prompt, pulled from the deployed code
  - the contract source           the pinned GenVM runner (Depends comment)
  - logs/training.txt             per-episode reward (normalized to per-step)
                                  and the exploration rate
  - logs/training_live_studionet.txt  the saved live run, replayed step by step
  - agent/manifest_data/baselines.json  random and greedy-only reward curves
  - agent/manifest_data/capture.json    one captured on-chain consensus receipt
                                  (leader model, score, validator votes, tx)

Nothing is invented. Fields that were never captured are omitted; the suite
schema treats them as optional. Run from the repository root:

    python -m agent.emit_manifest

writing manifest.json next to this repository's README.
"""

from __future__ import annotations

import inspect
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from contracts import logic

# --- per-domain configuration (the only part that differs between repos) ---

CONFIG = {
    "repo_name": "genlayer-rl-crisis-negotiator",
    "contract_filename": "crisis_negotiator.py",
    "max_steps": 8,
    "domain": {
        "id": "crisis",
        "name": "Crisis Negotiator",
        "plain_name": "Crisis Response",
        "plain_blurb": "sends drones and ambulances to emergencies",
        "world": "Disaster response",
    },
    "contract": {
        "address": "0xE0CBc71F7a3e87523F4A3833d4DdBE8a47595220",
        "chain": "studionet",
    },
    "prompt_fn": "build_reward_prompt",
    "prompt_sample": [
        "<the current emergency state>",
        '{"type":"dispatch","zone":"zone_a","resource":"drones"}',
    ],
    "capture_action": {"id": "dispatch_drones_a", "label": "send drones to zone A"},
}

SCHEMA_VERSION = "1.0"
ROLLING_WINDOW = 20

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(__file__).resolve().parent / "manifest_data"
CONTRACT_FILE = REPO_ROOT / "contracts" / CONFIG["contract_filename"]


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _runner_pin() -> str:
    first = CONTRACT_FILE.read_text(encoding="utf-8").splitlines()[0]
    m = re.search(r'"Depends"\s*:\s*"([^"]+)"', first)
    return m.group(1) if m else ""


def _sdk_version() -> str:
    req = REPO_ROOT / "agent" / "requirements.txt"
    for line in req.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "genlayer" in line:
            return line.replace(">=", " >=").replace("==", " ==")
    return "genlayer-py"


def _explorer_contract(address: str) -> str:
    return f"https://explorer-studionet.genlayerlabs.com/contracts/{address}"


def _explorer_tx(tx_hash: str) -> str:
    return f"https://explorer-studionet.genlayerlabs.com/tx/{tx_hash}"


def _prompt_template() -> str:
    fn = getattr(logic, CONFIG["prompt_fn"], None)
    if fn is None:
        return ""
    try:
        n = len(inspect.signature(fn).parameters)
        return str(fn(*CONFIG["prompt_sample"][:n]))
    except Exception:
        return ""


def _tolerance_from(principle: str) -> float:
    m = re.search(r"within ([\d.]+)", principle)
    return float(m.group(1)) if m else 1.5


def parse_training() -> tuple[list[dict], list[dict]]:
    """training.txt: episode=N reward=R rolling_avg=.. epsilon=E reason=.."""
    path = REPO_ROOT / "logs" / "training.txt"
    max_steps = CONFIG["max_steps"]
    episodes: list[dict] = []
    epsilon: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"episode=(\d+)\s+reward=([\d.]+).*epsilon=([\d.]+)", line)
        if m:
            i, reward, eps = int(m.group(1)), float(m.group(2)), float(m.group(3))
            episodes.append({"i": i, "reward": round(reward / max_steps, 3)})
            epsilon.append({"i": i, "value": round(eps, 4)})
    return episodes, epsilon


def load_baselines() -> dict | None:
    path = DATA_DIR / "baselines.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return {"random": data.get("random"), "greedy_only": data.get("greedy_only")}


def load_capture() -> dict | None:
    path = DATA_DIR / "capture.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _deep_json(src) -> dict | None:
    """A receipt payload is often a JSON-encoded string, sometimes doubly."""
    obj = src
    for _ in range(3):
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, str):
            try:
                obj = json.loads(obj)
            except Exception:
                return None
        else:
            return None
    return obj if isinstance(obj, dict) else None


def _reward_from_capture(cap: dict) -> float:
    for src in (cap.get("leader_eq_payload"), cap.get("leader_result")):
        obj = _deep_json(src) if src else None
        if not obj:
            continue
        for key in ("score", "acceptance"):
            if isinstance(obj.get(key), (int, float)):
                return float(obj[key])
        for key in ("last_reward_x100", "acceptance_x100", "reward_x100", "score_x100"):
            if isinstance(obj.get(key), (int, float)):
                return round(float(obj[key]) / 100.0, 2)
    return 0.0


def _reason_from_capture(cap: dict) -> str:
    obj = _deep_json(cap.get("leader_eq_payload") or cap.get("leader_result")) or {}
    return str(obj.get("reason", ""))[:400]


def build_consensus_run(cap: dict, tolerance: float) -> dict | None:
    if not cap:
        return None
    reward = _reward_from_capture(cap)
    reason = _reason_from_capture(cap)
    validators = [
        {"model": v.get("model"), "vote": v.get("vote")}
        for v in cap.get("validators", [])
        if v.get("vote") is not None
    ]
    agrees = sum(1 for v in validators if v["vote"] == "agree")
    consensus = {
        "outcome": "MAJORITY" if cap.get("status") == 5 else "NO_MAJORITY",
        "tolerance": tolerance,
        "leader_model": cap.get("leader_model"),
        "leader_score": reward,
        "leader_reason": reason,
        "validators": validators,
    }
    step = {
        "i": 1,
        "action": CONFIG["capture_action"],
        "reward": reward,
        "reward_kind": "llm",
        "reason": reason,
        "consensus": consensus,
        "tx": {
            "hash": cap["tx_hash"],
            "explorer": _explorer_tx(cap["tx_hash"]),
            "elapsed_s": cap.get("elapsed_s"),
        },
    }
    return {
        "id": "live-capture",
        "mode": "live",
        "label": f"Live studionet step ({agrees} of {len(validators)} agreed)",
        "episodes": [{"i": 1, "steps": [step]}],
    }


# Live logs come in two shapes across the domains: a per-step line (with an
# optional polarization) or a per-episode line (with an exploration rate).
LIVE_STEP_RE = re.compile(
    r"step=(\d+).*?action=(\S+)\s+reward=([\d.]+)(?:\s+polarization=([\d.]+))?"
    r".*?elapsed=(\d+)s\s+reason='(.*)'"
)
LIVE_EP_RE = re.compile(r"episode=(\d+)\s+reward=([\d.]+).*?epsilon=([\d.]+)\s+reason='(.*)'")


def build_replay_run() -> dict | None:
    path = REPO_ROOT / "logs" / "training_live_studionet.txt"
    if not path.exists():
        return None
    steps: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = LIVE_STEP_RE.search(line)
        if m:
            i, action, reward, polarization, _elapsed, reason = m.groups()
            step = {
                "i": int(i),
                "action": {"id": action, "label": action.replace("#", " ")},
                "reward": float(reward),
                "reward_kind": "llm",
                "reason": reason.rstrip("."),
            }
            if polarization:
                step["state_after"] = {"polarization": float(polarization)}
            steps.append(step)
            continue
        m = LIVE_EP_RE.search(line)
        if m:
            i, reward, eps, reason = m.groups()
            steps.append(
                {
                    "i": int(i),
                    "action": {"id": "live-decision", "label": "live decision"},
                    "reward": float(reward),
                    "reward_kind": "llm",
                    "reason": reason.rstrip("."),
                    "epsilon": float(eps),
                }
            )
    if not steps:
        return None
    return {
        "id": "replay-studionet",
        "mode": "replay",
        "label": "Saved studionet run",
        "episodes": [{"i": 1, "steps": steps}],
    }


def build_manifest() -> dict:
    principle = getattr(logic, "REWARD_EQUIVALENCE_PRINCIPLE", "")
    tolerance = _tolerance_from(principle)
    episodes, epsilon = parse_training()
    baselines = load_baselines()
    cap = load_capture()

    runs = []
    consensus_run = build_consensus_run(cap, tolerance)
    if consensus_run:
        runs.append(consensus_run)
    replay_run = build_replay_run()
    if replay_run:
        runs.append(replay_run)

    return {
        "schema_version": SCHEMA_VERSION,
        "domain": CONFIG["domain"],
        "provenance": {
            "repo": CONFIG["repo_name"],
            "commit": _git_commit(),
            "sdk": _sdk_version(),
            "runner_pin": _runner_pin(),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "contract": {
            "address": CONFIG["contract"]["address"],
            "chain": CONFIG["contract"]["chain"],
            "explorer": _explorer_contract(CONFIG["contract"]["address"]),
        },
        "reward": {
            "kind": "llm_comparative",
            "scale": [0, 10],
            "principle": principle,
            "prompt_template": _prompt_template(),
        },
        "learning": {
            "rolling_window": ROLLING_WINDOW,
            "episodes": episodes,
            "epsilon": epsilon,
            "baselines": baselines or {},
        },
        "runs": runs,
    }


def main() -> None:
    manifest = build_manifest()
    out = REPO_ROOT / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    learning = manifest["learning"]
    print(
        f"wrote {out.name}: episodes={len(learning['episodes'])} "
        f"baselines={'yes' if learning['baselines'] else 'no'} "
        f"runs={len(manifest['runs'])} commit={manifest['provenance']['commit']}"
    )


if __name__ == "__main__":
    main()
