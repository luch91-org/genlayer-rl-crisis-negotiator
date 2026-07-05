"""Environment abstraction so the RL loop in agent/agent.py never has to know
whether it's talking to a local heuristic or a deployed GenLayer contract.

MockEnv is the default everywhere (dev, CI, hyperparameter tuning): it is a
pure-Python reimplementation of contracts/crisis_negotiator.py's state
machine with a deterministic-but-noisy heuristic reward that mimics the LLM
judge's rubric, so training a policy costs nothing and takes no network
access. GenLayerEnv talks to a deployed contract for the real demo, where
every step is an actual on-chain LLM-consensus call.
"""

from __future__ import annotations

import random
from typing import Any, Protocol

from contracts.logic import apply_action, initial_resources, initial_zone_status

DEFAULT_MAX_STEPS = 8

# Base (pre-noise) reward mimicking the LLM judge's rubric: lives saved,
# resource efficiency, responsiveness to critical zones.
_DISPATCH_REWARD_BY_ZONE = {
    "critical": 9.5,
    "moderate": 6.5,
    "stable": 2.0,
    "evacuated": 0.5,
}
_DISPATCH_REWARD_NO_RESOURCE = 1.0
_EVACUATE_REWARD_BY_ZONE = {
    "critical": 6.0,
    "moderate": 3.5,
    "stable": 1.0,
    "evacuated": 0.5,
}
_WAIT_REWARD_ALL_DONE = 8.5
_WAIT_REWARD_OTHERWISE = 1.5
_NOISE_STD = 0.6


class Env(Protocol):
    def reset(self) -> dict[str, Any]: ...

    def step(self, action: dict[str, Any]) -> tuple[float, str, dict[str, Any]]:
        """Returns (reward, reason, next_state)."""
        ...


def _clip(value: float, lo: float = 0.0, hi: float = 10.0) -> float:
    return max(lo, min(hi, value))


class MockEnv:
    """Instant, free, local reimplementation of the contract's state machine."""

    def __init__(self, max_steps: int = DEFAULT_MAX_STEPS, seed: int | None = None):
        self.max_steps = max_steps
        self._rng = random.Random(seed)
        self.resources: dict[str, int] = {}
        self.zone_status: dict[str, str] = {}
        self.total_score = 0.0
        self.round = 0
        self.last_reward = 0.0
        self.last_reason = ""
        self.reset()

    def reset(self) -> dict[str, Any]:
        self.resources = initial_resources()
        self.zone_status = initial_zone_status()
        self.total_score = 0.0
        self.round = 0
        self.last_reward = 0.0
        self.last_reason = ""
        return self._state()

    def _state(self) -> dict[str, Any]:
        return {
            "resources": dict(self.resources),
            "zone_status": dict(self.zone_status),
            "round": self.round,
            "total_score": self.total_score,
            "last_reward": self.last_reward,
            "last_reason": self.last_reason,
        }

    def _heuristic_reward(self, action: dict[str, Any]) -> tuple[float, str]:
        """Deterministic-but-noisy stand-in for the LLM judge. Evaluated
        against the state BEFORE the action is applied, same as the real
        contract scores the action that produced the new state."""
        a_type = action.get("type")
        zone = action.get("zone")
        zone_before = self.zone_status.get(zone) if zone else None

        if a_type == "dispatch":
            resource = action.get("resource")
            qty = int(action.get("quantity", 1))
            has_resource = resource in self.resources and self.resources[resource] >= qty
            if not has_resource:
                base, reason = _DISPATCH_REWARD_NO_RESOURCE, "no resource available to dispatch"
            else:
                base = _DISPATCH_REWARD_BY_ZONE.get(zone_before or "", _DISPATCH_REWARD_NO_RESOURCE)
                reason = f"dispatched {resource} to {zone_before or 'unknown'} zone"
        elif a_type == "evacuate":
            base = _EVACUATE_REWARD_BY_ZONE.get(zone_before or "", 0.5)
            reason = f"evacuated a {zone_before or 'unknown'} zone"
        else:  # "wait" or unrecognized
            all_done = all(s in ("stable", "evacuated") for s in self.zone_status.values())
            base = _WAIT_REWARD_ALL_DONE if all_done else _WAIT_REWARD_OTHERWISE
            reason = (
                "waited with no critical zones left"
                if all_done
                else "waited while zones needed help"
            )

        noisy = _clip(base + self._rng.gauss(0.0, _NOISE_STD))
        return noisy, reason

    def step(self, action: dict[str, Any]) -> tuple[float, str, dict[str, Any]]:
        reward, reason = self._heuristic_reward(action)

        new_resources, new_zone_status, _applied = apply_action(
            self.resources, self.zone_status, action
        )
        self.resources = new_resources
        self.zone_status = new_zone_status
        self.round += 1
        self.total_score += reward
        self.last_reward = reward
        self.last_reason = reason

        return reward, reason, self._state()

    def is_episode_done(self) -> bool:
        return self.round >= self.max_steps


class GenLayerEnv:
    """Talks to a deployed CrisisNegotiator contract via the first-party
    genlayer-py SDK. Signatures confirmed against genlayer-py 0.18.0 source
    (genlayer_py/client/genlayer_client.py, genlayer_py/contracts/actions.py,
    genlayer_py/transactions/actions.py) -- NOT the docs.genlayer.com prose
    guide, which shows a write_contract(transaction=..., hash=...) example
    containing typos that don't match the real client. genlayer-py requires
    Python >= 3.12 at import time (collections.abc.Buffer); the import below
    is deferred into __init__ so MockEnv-only workflows never need it
    installed.
    """

    def __init__(
        self,
        address: str,
        chain: str = "localnet",
        private_key: str | None = None,
        max_steps: int = DEFAULT_MAX_STEPS,
    ):
        from genlayer_py import create_account, create_client
        from genlayer_py.chains import localnet, studionet, testnet_asimov, testnet_bradbury
        from genlayer_py.types import TransactionStatus

        # genlayer-py 0.18.0 ships all four chains (0.8.x had no
        # testnet_bradbury) -- confirmed via genlayer_py/chains/__init__.py.
        chains = {
            "localnet": localnet,
            "testnet_asimov": testnet_asimov,
            "testnet_bradbury": testnet_bradbury,
            "studionet": studionet,
        }
        if chain not in chains:
            raise ValueError(f"Unknown chain '{chain}'. Choose from: {sorted(chains)}")

        self.address = address
        self.max_steps = max_steps
        self._TransactionStatus = TransactionStatus
        # create_account(private_key) generates/loads an account; on a
        # public testnet you must import a funded private key. On localnet
        # a fresh account plus fund_account() below is enough.
        self.account = create_account(private_key) if private_key else create_account()
        # create_client() already calls initialize_consensus_smart_contract()
        # internally (confirmed: genlayer_py/client/client.py), so we don't
        # call it again here.
        self.client = create_client(chain=chains[chain], account=self.account)
        # fund_account only refuses when chain.id != localnet.id -- and
        # studionet deliberately shares localnet's chain id 61999 (confirmed:
        # genlayer_py/chains/studionet.py), so funding works on both.
        if chain in ("localnet", "studionet"):
            try:
                self.client.fund_account(address=self.account.address, amount=10**18)
            except Exception as exc:  # best effort: some setups pre-fund accounts
                print(f"[GenLayerEnv] fund_account skipped: {exc}")
        self._round = 0

    def reset(self) -> dict[str, Any]:
        self._round = 0
        return self._get_state()

    def _get_state(self) -> dict[str, Any]:
        raw: Any = self.client.read_contract(
            address=self.address,
            function_name="get_state",
            args=[],
        )
        # The contract stores scores as integers scaled x100 because floats
        # are neither GenVM-storable nor calldata-encodable (confirmed
        # against a live studionet deploy). Convert to the same float shape
        # MockEnv produces so the agent never sees the difference.
        return {
            "resources": dict(raw["resources"]),
            "zone_status": dict(raw["zone_status"]),
            "round": int(raw["round"]),
            "total_score": int(raw["total_score_x100"]) / 100.0,
            "last_reward": int(raw["last_reward_x100"]) / 100.0,
            "last_reason": str(raw.get("last_reason", "")),
        }

    def step(self, action: dict[str, Any]) -> tuple[float, str, dict[str, Any]]:
        tx_hash = self.client.write_contract(
            address=self.address,
            function_name="take_action",
            account=self.account,
            args=[action],
            value=0,
        )
        # wait_for_transaction_receipt raises GenLayerError if the tx never
        # reaches the requested status within retries*interval, so reaching
        # the line after this call already means consensus accepted it. We
        # deliberately do NOT try to parse a pass/fail flag out of the
        # receipt's consensus_data.leader_receipt.execution_result: that
        # field's shape differs between localnet (raw JSON) and testnet
        # (RLP-decoded) in genlayer-py 0.8.1, and the reward the contract
        # recorded in state is the ground truth we actually need.
        self.client.wait_for_transaction_receipt(
            transaction_hash=tx_hash,
            status=self._TransactionStatus.ACCEPTED,
            interval=3000,
            retries=30,
        )
        state = self._get_state()
        self._round += 1
        return float(state["last_reward"]), state.get("last_reason", ""), state

    def is_episode_done(self) -> bool:
        return self._round >= self.max_steps
