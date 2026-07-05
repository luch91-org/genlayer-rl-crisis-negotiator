"""Tabular Q-learning agent for CrisisNegotiator.

State space design note: the environment's raw state includes exact
resource counts (drones 0-5, ambulances 0-3, supply_kits 0-20) and zone
statuses (4 possible values each, across 3 zones). Used raw, that's roughly
32,000 reachable (resources x zones) combinations -- far too many for a
tabular method to see enough repeat visits within a 500-episode budget and
still show a clean learning curve. serialize_state() discretizes each
resource count into an "available" / "empty" bucket, which keeps the state
space small (at most 8 resource buckets x 64 zone combos = 512 states)
while preserving exactly the distinction the reward function actually cares
about: can this dispatch happen at all. This is a deliberate simplification
in the spirit of "tabular Q-learning is intentionally the simplest thing
that works" (see CLAUDE.md, design principle 4) -- function approximation
over the raw counts is a documented future path, not a v1 requirement.
"""

from __future__ import annotations

import ast
import json
import random
from pathlib import Path
from typing import Any

ACTIONS: list[dict[str, Any]] = [
    {"type": "dispatch", "zone": "zone_a", "resource": "drones", "quantity": 1},
    {"type": "dispatch", "zone": "zone_a", "resource": "ambulances", "quantity": 1},
    {"type": "dispatch", "zone": "zone_a", "resource": "supply_kits", "quantity": 1},
    {"type": "dispatch", "zone": "zone_b", "resource": "drones", "quantity": 1},
    {"type": "dispatch", "zone": "zone_b", "resource": "ambulances", "quantity": 1},
    {"type": "dispatch", "zone": "zone_b", "resource": "supply_kits", "quantity": 1},
    {"type": "dispatch", "zone": "zone_c", "resource": "drones", "quantity": 1},
    {"type": "dispatch", "zone": "zone_c", "resource": "ambulances", "quantity": 1},
    {"type": "dispatch", "zone": "zone_c", "resource": "supply_kits", "quantity": 1},
    {"type": "evacuate", "zone": "zone_a"},
    {"type": "evacuate", "zone": "zone_b"},
    {"type": "evacuate", "zone": "zone_c"},
    {"type": "wait"},
]

StateKey = tuple


def serialize_state(state: dict[str, Any]) -> StateKey:
    zone_tuple = tuple(sorted(state["zone_status"].items()))
    resource_tuple = tuple(
        sorted(
            (name, "available" if count > 0 else "empty")
            for name, count in state["resources"].items()
        )
    )
    return (zone_tuple, resource_tuple)


class QLearningAgent:
    def __init__(
        self,
        actions: list[dict[str, Any]] = ACTIONS,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.99,
        seed: int | None = None,
    ):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table: dict[StateKey, list[float]] = {}
        self._rng = random.Random(seed)

    def _ensure_state(self, key: StateKey) -> list[float]:
        if key not in self.q_table:
            self.q_table[key] = [0.0] * len(self.actions)
        return self.q_table[key]

    def select_action(self, state: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        key = serialize_state(state)
        q_values = self._ensure_state(key)

        if self._rng.random() < self.epsilon:
            idx = self._rng.randrange(len(self.actions))
        else:
            best_q = max(q_values)
            best_indices = [i for i, q in enumerate(q_values) if q == best_q]
            idx = self._rng.choice(best_indices)

        return idx, self.actions[idx]

    def update(
        self,
        state: dict[str, Any],
        action_idx: int,
        reward: float,
        next_state: dict[str, Any],
    ) -> None:
        key = serialize_state(state)
        next_key = serialize_state(next_state)
        q_values = self._ensure_state(key)
        next_q_values = self._ensure_state(next_key)

        td_target = reward + self.gamma * max(next_q_values)
        td_error = td_target - q_values[action_idx]
        q_values[action_idx] += self.alpha * td_error

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def best_action(self, state: dict[str, Any]) -> dict[str, Any]:
        key = serialize_state(state)
        q_values = self._ensure_state(key)
        best_q = max(q_values)
        best_indices = [i for i, q in enumerate(q_values) if q == best_q]
        return self.actions[self._rng.choice(best_indices)]

    def save(self, path: str | Path) -> None:
        payload = {
            "q_table": {repr(key): values for key, values in self.q_table.items()},
            "epsilon": self.epsilon,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
        }
        Path(path).write_text(json.dumps(payload, indent=2))

    def load(self, path: str | Path) -> None:
        payload = json.loads(Path(path).read_text())
        self.q_table = {ast.literal_eval(key): values for key, values in payload["q_table"].items()}
        self.epsilon = payload.get("epsilon", self.epsilon)
        self.alpha = payload.get("alpha", self.alpha)
        self.gamma = payload.get("gamma", self.gamma)
        self.epsilon_min = payload.get("epsilon_min", self.epsilon_min)
        self.epsilon_decay = payload.get("epsilon_decay", self.epsilon_decay)
