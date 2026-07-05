"""Tests for the tabular Q-learning agent. Everything here runs against a
scripted stand-in Env (or no Env at all) -- no MockEnv heuristic, no
network -- so these tests isolate agent.py's own logic: action selection,
the Bellman update, epsilon decay, and Q-table save/resume.
"""

from __future__ import annotations

import json

from agent.agent import ACTIONS, QLearningAgent, serialize_state
from agent.train import run_episode

STATE_A = {
    "resources": {"drones": 5, "ambulances": 3, "supply_kits": 20},
    "zone_status": {"zone_a": "critical", "zone_b": "moderate", "zone_c": "stable"},
    "round": 0,
    "total_score": 0.0,
    "last_reward": 0.0,
    "last_reason": "",
}
STATE_B = {
    "resources": {"drones": 4, "ambulances": 3, "supply_kits": 20},
    "zone_status": {"zone_a": "moderate", "zone_b": "moderate", "zone_c": "stable"},
    "round": 1,
    "total_score": 5.0,
    "last_reward": 5.0,
    "last_reason": "",
}


class ScriptedEnv:
    """Ignores whatever action it's given and returns a pre-scripted reward
    per step, transitioning through a fixed sequence of states. Lets tests
    assert on exact, deterministic Q-value arithmetic."""

    def __init__(self, states: list[dict], rewards: list[float]):
        assert len(states) == len(rewards) + 1
        self.states = states
        self.rewards = rewards
        self._step_idx = 0

    def reset(self) -> dict:
        self._step_idx = 0
        return self.states[0]

    def step(self, action):
        reward = self.rewards[self._step_idx]
        self._step_idx += 1
        next_state = self.states[self._step_idx]
        return reward, "scripted", next_state

    def is_episode_done(self) -> bool:
        return self._step_idx >= len(self.rewards)


def test_epsilon_decays_towards_the_floor():
    agent = QLearningAgent(epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.9)
    for _ in range(200):
        agent.decay_epsilon()
    assert agent.epsilon == 0.05  # floors, never goes below epsilon_min


def test_epsilon_decays_monotonically_before_the_floor():
    agent = QLearningAgent(epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.99)
    previous = agent.epsilon
    for _ in range(10):
        agent.decay_epsilon()
        assert agent.epsilon <= previous
        previous = agent.epsilon


def test_update_moves_q_value_towards_a_positive_reward():
    agent = QLearningAgent(alpha=0.5, gamma=0.9)
    key = serialize_state(STATE_A)
    action_idx = 0
    before = agent._ensure_state(key)[action_idx]
    agent.update(STATE_A, action_idx, reward=10.0, next_state=STATE_B)
    after = agent.q_table[key][action_idx]
    assert before == 0.0
    assert after > before  # a positive reward must pull Q upward


def test_update_moves_q_value_towards_a_negative_reward():
    agent = QLearningAgent(alpha=0.5, gamma=0.9)
    key = serialize_state(STATE_A)
    action_idx = 0
    agent.update(STATE_A, action_idx, reward=-5.0, next_state=STATE_B)
    after = agent.q_table[key][action_idx]
    assert after < 0.0


def test_bellman_update_matches_hand_computed_value():
    agent = QLearningAgent(alpha=0.5, gamma=0.9)
    key = serialize_state(STATE_A)
    next_key = serialize_state(STATE_B)
    action_idx = 2
    agent.update(STATE_A, action_idx, reward=10.0, next_state=STATE_B)
    # Q starts at 0; next state's Q-values also start at 0, so
    # td_target = reward + gamma * max(next_q) = 10 + 0.9*0 = 10
    # new_q = old_q + alpha * (td_target - old_q) = 0 + 0.5*(10-0) = 5.0
    assert agent.q_table[key][action_idx] == 5.0
    assert next_key in agent.q_table  # next_state got initialized too


def test_greedy_selection_picks_the_max_q_action():
    agent = QLearningAgent(epsilon_start=0.0, epsilon_min=0.0)
    key = serialize_state(STATE_A)
    agent._ensure_state(key)
    best_idx = 5
    agent.q_table[key][best_idx] = 99.0
    idx, action = agent.select_action(STATE_A)
    assert idx == best_idx
    assert action == ACTIONS[best_idx]


def test_epsilon_zero_never_explores_across_many_draws():
    agent = QLearningAgent(epsilon_start=0.0, epsilon_min=0.0, seed=0)
    key = serialize_state(STATE_A)
    agent._ensure_state(key)
    agent.q_table[key][3] = 42.0
    for _ in range(50):
        idx, _action = agent.select_action(STATE_A)
        assert idx == 3


def test_save_and_load_round_trips_q_table_and_hyperparameters(tmp_path):
    agent = QLearningAgent(
        alpha=0.2, gamma=0.8, epsilon_start=0.7, epsilon_min=0.02, epsilon_decay=0.95
    )
    key = serialize_state(STATE_A)
    agent._ensure_state(key)
    agent.q_table[key][0] = 3.25
    agent.q_table[key][1] = -1.5
    agent.epsilon = 0.42

    path = tmp_path / "q_table.json"
    agent.save(path)

    loaded = QLearningAgent()
    loaded.load(path)

    assert loaded.q_table == agent.q_table
    assert loaded.epsilon == 0.42
    assert loaded.alpha == 0.2
    assert loaded.gamma == 0.8
    assert loaded.epsilon_min == 0.02
    assert loaded.epsilon_decay == 0.95


def test_saved_q_table_file_is_valid_json_with_string_keys(tmp_path):
    agent = QLearningAgent()
    key = serialize_state(STATE_A)
    agent._ensure_state(key)
    path = tmp_path / "q_table.json"
    agent.save(path)

    raw = json.loads(path.read_text())
    assert isinstance(raw["q_table"], dict)
    assert all(isinstance(k, str) for k in raw["q_table"])


def test_run_episode_against_scripted_env_updates_q_table_towards_rewards():
    states = [STATE_A, STATE_B, STATE_A]
    rewards = [8.0, 2.0]
    env = ScriptedEnv(states, rewards)
    agent = QLearningAgent(alpha=0.5, gamma=0.9, epsilon_start=1.0)

    episode_reward, last_reason = run_episode(env, agent)

    assert episode_reward == sum(rewards)
    assert last_reason == "scripted"
    # Two distinct state keys should have been touched (STATE_A visited
    # twice, STATE_B once, but as *states* there are only two distinct keys).
    assert len(agent.q_table) == 2
