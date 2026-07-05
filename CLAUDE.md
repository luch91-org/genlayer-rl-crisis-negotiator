# genlayer-rl-crisis-negotiator

Domain-scoped build guide. Inherits the shared engineering spec from the
[GenLayer RL Agent Autonomy](https://github.com/luch91-org)
umbrella CLAUDE.md (repository standards, GenLayer contract rules, agent
architecture) - read that first. This file only covers what's specific to
this domain.

## Domain

Disaster response. State = three zones (`zone_a`, `zone_b`, `zone_c`), each
in `{critical, moderate, stable, evacuated}`, plus three resource pools
(`drones`, `ambulances`, `supply_kits`). Actions = `dispatch(resource,
zone, quantity)`, `evacuate(zone)`, `wait`. The agent learns to dispatch
available resources to the zones that need them most, evacuate only when
that's genuinely the right call, and stop burning resources once every zone
is handled.

## Where things live

- `contracts/crisis_negotiator.py` - the actual `gl.Contract`, and the
  single source of truth for ALL contract logic, deterministic helpers
  included. It must stay **fully self-contained**: `deploy_contract`
  sends exactly one source file on-chain, so it can never import sibling
  modules (`tests/test_contract.py` has a regression guard for this - 
  an earlier version imported `contracts.logic` and would have failed on
  a real node while passing every offline test).
- `contracts/logic.py` - does NOT contain logic of its own. It execs the
  contract's real source with a stubbed `genlayer` module and re-exports
  the pure helpers (`apply_action`, `parse_reward_output`, ...), so
  pytest and `MockEnv` exercise the deployed code itself rather than a
  hand-maintained mirror. If you add new deterministic behavior, add it
  as a module-level function in `crisis_negotiator.py` and re-export it
  in `logic.py`'s explicit export list.
- `agent/env.py` - `MockEnv`'s reward heuristic lives here
  (`_DISPATCH_REWARD_BY_ZONE` etc.). It's a deliberate approximation of
  the LLM judge's rubric, not a copy of it - tune it if the learning curve
  stops climbing cleanly, but keep the same qualitative shape (critical >
  moderate > stable > evacuated for dispatch; "wait when done" high,
  "wait while zones need help" low).
- `agent/agent.py` - `serialize_state()` discretizes resource counts into
  `available`/`empty` buckets before they become part of the Q-table key.
  This is required for the state space to stay small enough for tabular
  Q-learning to converge in a few hundred episodes; don't switch this back
  to raw counts without also switching to function approximation.

## Non-negotiable GenLayer rules for this contract

(Full rationale in the umbrella CLAUDE.md; restated here because this is
the file someone edits when they touch `contracts/crisis_negotiator.py`.)

- Never call `gl.nondet.exec_prompt` directly in `take_action`'s body - it
  must be inside the `score_block()` inner function passed to
  `gl.eq_principle.prompt_comparative`.
- Never reference `self` inside `score_block()`. Everything it needs
  (`resources_snap`, `zones_snap`, `action_snap`, `round_snap`) is
  snapshotted into locals before the function is defined.
- Never use `gl.eq_principle.strict_eq` for the reward score - it's a
  subjective 0-10 judgment call, and independent LLM calls will not
  produce byte-identical output even from the same model. Use
  `gl.eq_principle.prompt_comparative` with a stated numeric tolerance
  (currently 1.5 points, in `contracts/logic.py`'s
  `REWARD_EQUIVALENCE_PRINCIPLE`).
- Keep using the namespaced current API (`gl.nondet.*`, `gl.eq_principle.*`,
  `gl.public.view`/`gl.public.write`). Some bundled example contracts in
  older versions of `genlayer-test` still use the deprecated flat forms
  (`gl.exec_prompt`, `gl.eq_principle_prompt_comparative`) - don't copy
  from those.
- Storage fields must be GenVM storage types: `TreeMap[str, u256]`,
  `TreeMap[str, str]`, `u256`, `str` - never bare `dict` (deploy fails
  with "class is not marked for usage within storage"; we hit this on a
  real studionet deploy). Populate TreeMaps key by key in `__init__`.
- No floats anywhere the chain can see them: not in storage, not in
  method returns (`calldata.encode(1.5)` raises). Scores are integers
  scaled ×100 on-chain (`*_x100` fields); `GenLayerEnv` converts back to
  floats for the agent. Floats inside the leader/validator JSON string
  are fine - that's a string by then.

## Testing constraints specific to this repo

`genlayer-test` (the package providing GenVM Direct Mode / `gltest`) pins
`genlayer-py==0.3.0` in its own published dependency metadata, which
conflicts with the `genlayer-py>=0.8.0` the live agent needs. Don't add
`genlayer-test` to `agent/requirements.txt` - if you want to run Direct
Mode contract tests, install it in a separate virtualenv (see
`docs/tutorial.md`). The default test suite (`pytest tests/`) never needs
it; it runs entirely against `contracts/logic.py`'s pure functions and
`agent/env.py`'s `MockEnv`.

## Success bar for this repo

See the README's "Setup" section for the exact commands. In short: 500
mock episodes should show the rolling-average reward climb from roughly 3
to 8 or higher (per-step; episode rewards in the log are summed over
`--max-steps`, default 8), `agent/q_table.json` and
`docs/learning_curve.png` should exist afterward, and 30 episodes against
a deployed contract (`--env genlayer`) should also show climbing - 
noisier, since it's real LLM-judged - rewards.
