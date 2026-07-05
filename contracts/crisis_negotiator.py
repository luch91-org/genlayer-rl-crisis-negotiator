# { "Depends": "py-genlayer:1jb45aa8ynh2a9c9xn3b7qqh8sm5q93hwfp7jqmwsfhh8jpz09h6" }
"""CrisisNegotiator Intelligent Contract.

Disaster-response environment. The agent dispatches drones, ambulances, and
supply kits (or evacuates, or waits) across three zones. Each action is
scored 0-10 by an LLM judge, and validators reach consensus on that score
via a comparative equivalence principle (never strict_eq -- a subjective
judgment call will not come back byte-identical across independent LLM
calls, even from the same model).

This file is deliberately SELF-CONTAINED: deploy_contract(code=...) sends
exactly one source file on-chain, so the contract cannot import sibling
modules like contracts.logic. Off-chain, contracts/logic.py re-exports the
pure helpers below by exec'ing this file's source with a stubbed
`genlayer` module, so pytest exercises the real deployed code.

Two GenVM constraints confirmed the hard way (real failed deploy to
studionet, tx 0x5c612e..., 2026-07):
  - Storage fields cannot be bare `dict`: GenVM's storage generator raises
    "class is not marked for usage within storage" for it. Use
    TreeMap[str, u256] / TreeMap[str, str].
  - Floats are not calldata-encodable (genlayer_py.abi.calldata.encode(1.5)
    raises "invalid type <class 'float'>"), so scores are stored and
    returned as integers scaled x100 (7.5 -> 750). Only the JSON string
    passed between leader and validators inside the equivalence principle
    carries the raw float, which is fine -- it's a string at that point.

API surface confirmed 2026-07 against docs.genlayer.com and the deployed
studionet GenVM:
  - gl.eq_principle.prompt_comparative(fn, principle=...) is the current,
    documented form (gl.exec_prompt / gl.eq_principle_prompt_comparative
    are deprecated flat forms).
  - gl.nondet.exec_prompt(prompt, response_format="json") returns an
    already-parsed dict, not a string.
"""

import json

from genlayer import *

# --- pure deterministic helpers (exec'd off-chain by contracts/logic.py) ----

ZONES = ("zone_a", "zone_b", "zone_c")
RESOURCES = ("drones", "ambulances", "supply_kits")

# Zone severity improves one step at a time when a matching resource is
# dispatched. "stable" and "evacuated" are absorbing states for dispatch.
_ZONE_IMPROVEMENT = {"critical": "moderate", "moderate": "stable"}

REWARD_EQUIVALENCE_PRINCIPLE = (
    "The two evaluations agree if their scores are within 1.5 points of each "
    "other and express the same overall judgment of the action. Wording may differ."
)


def initial_resources() -> dict:
    return {"drones": 5, "ambulances": 3, "supply_kits": 20}


def initial_zone_status() -> dict:
    return {"zone_a": "critical", "zone_b": "moderate", "zone_c": "stable"}


def apply_action(resources: dict, zone_status: dict, action: dict) -> tuple:
    """Deterministic state transition. Returns (new_resources, new_zone_status, applied).

    Operates on and returns PLAIN dicts -- the contract copies storage
    TreeMaps into dicts before calling this, and writes the result back
    key by key. `applied` is True when the action actually changed state.
    """
    new_resources = dict(resources)
    new_zone_status = dict(zone_status)

    a_type = action.get("type")
    zone = action.get("zone")
    resource = action.get("resource")
    qty = int(action.get("quantity", 1))

    applied = False

    if a_type == "dispatch" and resource in new_resources and zone in new_zone_status:
        if new_resources[resource] >= qty:
            new_resources[resource] -= qty
            current = new_zone_status[zone]
            if current != "stable":
                new_zone_status[zone] = _ZONE_IMPROVEMENT.get(current, current)
            applied = True
    elif a_type == "evacuate" and zone in new_zone_status:
        new_zone_status[zone] = "evacuated"
        applied = True
    # "wait" (or any unrecognized type) is a no-op: applied stays False.

    return new_resources, new_zone_status, applied


def build_reward_prompt(
    resources_snap: str, zones_snap: str, action_snap: str, round_snap: int
) -> str:
    return (
        "You are a disaster-response evaluator.\n"
        f"Resources remaining: {resources_snap}\n"
        f"Zone statuses: {zones_snap}\n"
        f"Action taken: {action_snap}\n"
        f"Round: {round_snap}\n\n"
        "Score the action 0-10 on lives saved, resource efficiency, "
        "and responsiveness to critical zones.\n"
        'Return ONLY JSON: {"score": <number 0-10>, "reason": "<short reason>"}'
    )


def parse_reward_output(raw) -> tuple:
    """Parse and clamp an LLM-judge response into (score, reason).

    Accepts either an already-parsed dict (what gl.nondet.exec_prompt with
    response_format="json" returns) or a JSON string, and always returns a
    score clamped to [0, 10] so a malformed or out-of-range LLM response
    cannot corrupt the reward signal or the Q-table.
    """
    data = json.loads(raw) if isinstance(raw, str) else raw
    score = float(data["score"])
    score = max(0.0, min(10.0, score))
    reason = str(data.get("reason", ""))
    return score, reason


def score_to_x100(score: float) -> int:
    """Floats are not calldata-encodable and not GenVM-storable, so scores
    live on-chain as integers scaled x100 (7.5 -> 750)."""
    return int(round(float(score) * 100))


def normalize_reward_for_consensus(score: float, reason: str) -> str:
    """Canonical JSON string the leader function returns to the equivalence
    principle, so all validators compare the same stable shape."""
    return json.dumps({"score": float(score), "reason": str(reason)}, sort_keys=True)


# --- the contract itself -----------------------------------------------------


class CrisisNegotiator(gl.Contract):
    # GenVM storage: bare dict/float are not storable (see module docstring).
    resources: TreeMap[str, u256]
    zone_status: TreeMap[str, str]
    total_score_x100: u256
    round: u256
    last_reward_x100: u256
    last_reason: str

    def __init__(self):
        # TreeMap storage slots pre-exist empty; populate key by key
        # (slot-level assignment of a plain dict is not supported).
        for name, count in initial_resources().items():
            self.resources[name] = u256(count)
        for zone, status in initial_zone_status().items():
            self.zone_status[zone] = status
        self.total_score_x100 = u256(0)
        self.round = u256(0)
        self.last_reward_x100 = u256(0)
        self.last_reason = ""

    @gl.public.view
    def get_state(self) -> dict:
        return {
            "resources": {k: int(v) for k, v in self.resources.items()},
            "zone_status": {k: str(v) for k, v in self.zone_status.items()},
            "round": int(self.round),
            "total_score_x100": int(self.total_score_x100),
            "last_reward_x100": int(self.last_reward_x100),
            "last_reason": self.last_reason,
        }

    @gl.public.write
    def take_action(self, action: dict) -> dict:
        self.round = u256(int(self.round) + 1)

        # 1) Deterministic state update. self IS allowed here -- this runs
        # identically on every validator, so it is not wrapped in a nondet
        # block. Copy storage into plain dicts, transition, write back.
        resources_now = {k: int(v) for k, v in self.resources.items()}
        zones_now = {k: str(v) for k, v in self.zone_status.items()}
        new_resources, new_zone_status, _applied = apply_action(resources_now, zones_now, action)
        for name, count in new_resources.items():
            self.resources[name] = u256(count)
        for zone, status in new_zone_status.items():
            self.zone_status[zone] = status

        # 2) Snapshot state into LOCALS. self is NOT accessible inside the
        # nondet block below.
        resources_snap = json.dumps(new_resources, sort_keys=True)
        zones_snap = json.dumps(new_zone_status, sort_keys=True)
        action_snap = json.dumps(action, sort_keys=True)
        round_snap = int(self.round)

        # 3) Leader actually calls the LLM and returns a canonical JSON
        # string (never just the prompt -- a function that only builds a
        # prompt string runs no inference and is a bug).
        def score_block() -> str:
            prompt = build_reward_prompt(resources_snap, zones_snap, action_snap, round_snap)
            out = gl.nondet.exec_prompt(prompt, response_format="json")
            score, reason = parse_reward_output(out)
            return normalize_reward_for_consensus(score, reason)

        # 4) Validators agree the leader's score is reasonable, not
        # byte-identical -- strict_eq would never pass on a subjective score.
        raw = gl.eq_principle.prompt_comparative(
            score_block, principle=REWARD_EQUIVALENCE_PRINCIPLE
        )
        score, reason = parse_reward_output(raw)
        reward_x100 = score_to_x100(score)

        self.total_score_x100 = u256(int(self.total_score_x100) + reward_x100)
        self.last_reward_x100 = u256(reward_x100)
        self.last_reason = reason
        return {"reward_x100": reward_x100, "reason": reason, "round": int(self.round)}

    @gl.public.view
    def get_score(self) -> int:
        """Total accumulated score, scaled x100 (divide by 100 off-chain)."""
        return int(self.total_score_x100)
