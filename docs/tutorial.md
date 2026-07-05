# Tutorial: how CrisisNegotiator actually works

## The contract

`contracts/crisis_negotiator.py` holds three zones (`zone_a`, `zone_b`, `zone_c`),
each in one of four states (`critical`, `moderate`, `stable`, `evacuated`), and
three resource pools (`drones`, `ambulances`, `supply_kits`). Every call to
`take_action(action)`:

1. **Applies the action deterministically.** Dispatch spends a resource and
   nudges the targeted zone one step better (`critical` -> `moderate` ->
   `stable`); evacuate marks a zone `evacuated`; wait does nothing. This part
   runs identically on every validator, so it's plain Python with `self`
   available -- no consensus mechanism needed for it.
2. **Snapshots state into locals.** `resources_snap`, `zones_snap`,
   `action_snap`, `round_snap` are captured as local variables *before* the
   non-deterministic block, because `self` cannot be referenced inside the
   inner function passed to an equivalence principle -- the contract state at
   that point isn't guaranteed to be visible in the sandboxed context each
   validator runs the LLM call in.
3. **Calls an LLM to score the action.** The inner `score_block()` function
   builds a prompt from the snapshot and calls
   `gl.nondet.exec_prompt(prompt, response_format="json")`. This inner
   function *is* the leader function: it has to actually perform the
   inference and return the result, not just build a prompt string.
4. **Reaches consensus on the score via a comparative equivalence
   principle.** `gl.eq_principle.prompt_comparative(score_block,
   principle=REWARD_EQUIVALENCE_PRINCIPLE)` has every validator run
   `score_block()` themselves and compare their own result to the leader's,
   under the stated principle ("scores within 1.5 points, same overall
   judgment"). If validators agree, the leader's score becomes the reward.

## Why a comparative equivalence principle, not `strict_eq`

`strict_eq` requires every validator's output to match byte-for-byte. That's
right for deterministic computation, but an LLM asked "score this 0-10" will
not return the identical number (or even identical wording) twice, even from
the same model on the same input -- it's not that GenLayer is being
imprecise, it's that the question is genuinely subjective. A comparative
equivalence principle instead asks "do these independent judgments agree
closely enough and point the same direction," which is what subjective
scoring actually needs. `contracts/logic.py`'s
`REWARD_EQUIVALENCE_PRINCIPLE` states exactly what "close enough" means (1.5
points) so validators have an unambiguous bar to check against, instead of a
vague "seems reasonable."

## Where the deterministic logic actually lives

`contracts/crisis_negotiator.py` is fully **self-contained**: the
state-transition helpers (`apply_action`), the reward prompt builder, and
the reward parsing/clamping all live inline in the contract file. This is
a hard requirement, not a style choice -- `deploy_contract(code=...)`
sends exactly one source file on-chain, so a contract that imports a
sibling module (`from contracts.logic import ...`) passes every offline
test and then fails at execution time on a real node. (An earlier version
of this repo had exactly that bug; `tests/test_contract.py` now has a
regression guard asserting the contract source contains no sibling
imports.)

Off-chain code still needs those same helpers: pytest tests them and
`agent/env.py`'s `MockEnv` reuses the state machine. `contracts/logic.py`
provides them **without duplicating them** -- it execs the contract's
actual source with a stubbed `genlayer` module and re-exports the
resulting functions. (The real `genlayer` module only exists inside the
GenVM runtime; the package published on PyPI under that name is a 1.3 KB
placeholder, so `from genlayer import *` can't succeed in an ordinary
Python process without the stub.) The payoff: everything the tests
exercise IS the deployed code, byte for byte -- there is no hand-written
mirror to drift out of sync. The stub's `gl.nondet` / `gl.eq_principle`
entry points raise `NotImplementedError`, so nothing off-chain can
accidentally "succeed" at calling the LLM judge.

## GenVM storage and calldata constraints (learned from a real failed deploy)

Two constraints that offline tests cannot catch, both confirmed against a
live studionet deploy (failed deploy tx `0x5c612e...`, then fixed):

1. **Storage fields must use GenVM storage types.** A class-level
   annotation of bare `dict` makes the storage generator raise
   `"class is not marked for usage within storage"` at deploy time. Use
   `TreeMap[str, u256]` / `TreeMap[str, str]` and populate them key by
   key in `__init__` (slot-level assignment of a plain dict is not
   supported either).
2. **Floats are neither GenVM-storable nor calldata-encodable.**
   `genlayer_py.abi.calldata.encode(1.5)` raises
   `invalid type <class 'float'>`. Scores therefore live on-chain as
   integers scaled ×100 (`7.5` → `750`, fields named `*_x100`), and
   `GenLayerEnv._get_state()` divides by 100 so the agent still sees
   floats. The only place a raw float travels is inside the JSON string
   the leader passes to the equivalence principle - which is fine,
   because it's a string at that point.

## Mock vs. live: the tradeoff

`agent/env.py` defines two environments behind the same `Env` protocol
(`reset()` / `step(action)` / `is_episode_done()`):

- **`MockEnv`** reimplements the same state machine in pure Python and
  scores actions with a deterministic-but-noisy heuristic that mimics the
  LLM judge's rubric (high reward for dispatching an available resource to a
  critical zone, low reward for acting on a stable/evacuated zone or wasting
  a resource, small Gaussian noise layered on top). It costs nothing, needs
  no network, and is the default everywhere -- local dev, CI, and
  hyperparameter tuning.
- **`GenLayerEnv`** talks to a real deployed contract. Every `step()` is an
  actual on-chain transaction that triggers LLM inference across validators
  plus consensus -- meaningfully slower (seconds, not microseconds) and
  costs gas. Use it for the real demo, with a modest episode budget (the
  success criteria in the README use 30 episodes against `--env genlayer`,
  vs. 500 against `--env mock`).

Because the reward is LLM-judged, it's also non-stationary: the same state
scored twice will not always come back identical. Treat the training curve
against `MockEnv` as the one to expect a clean upward trend from; the live
curve against `GenLayerEnv` should still trend upward but will be noisier.

## Tuning the hyperparameters

`agent/agent.py`'s `QLearningAgent` exposes everything via `agent/train.py`
CLI flags:

- `--alpha` (default `0.1`): learning rate. Higher values adapt faster to
  new rewards but are noisier; with an LLM-judged (non-stationary) reward,
  going much above `0.2` tends to make the rolling average jumpy rather than
  smoothly climbing.
- `--gamma` (default `0.95`): discount factor. Episodes are short (8 steps
  by default), so this mostly controls how much the agent values setting up
  a good zone 2-3 steps ahead (e.g. dispatching to `critical` before it
  matters) vs. grabbing the best immediate action.
- `--epsilon-decay` (default `0.99`, multiplicative per episode) and
  `--epsilon-min` (default `0.01`): control the explore/exploit schedule.
  With 500 episodes, `0.99` decays past the floor around episode ~460,
  leaving the last ~40 episodes almost fully greedy -- which is what
  produces the "final rolling average" number `agent/train.py` prints.
- `--max-steps` (default `8`): episode length. `agent/env.py`'s reward
  shaping assumes an optimal trajectory clears both non-stable zones in the
  first few steps and then spends the rest of the episode correctly
  `wait`-ing (`zone_a` critical -> moderate -> stable in 2 dispatches,
  `zone_b` moderate -> stable in 1); shortening `--max-steps` below ~4 will
  compress or cut off that convergence and lower the achievable ceiling.

State-space note: the Q-table key discretizes each resource count into
`"available"` / `"empty"` rather than using the exact count (see the module
docstring in `agent/agent.py`) -- this keeps the tabular state space small
enough (≤512 states) to actually converge within a few hundred episodes.
It's a deliberate simplification, not an oversight; moving to function
approximation over the exact counts is a documented future path.

## Running the contract tests against a real GenVM runner (optional)

`tests/test_contract.py` factors all the assertions that don't need GenVM
into plain pytest functions against `contracts/logic.py`. If you also want
to exercise the actual `CrisisNegotiator` contract class in GenVM's Direct
Mode, install `genlayer-test` (`pip install genlayer-test`) -- but do this
in a **separate virtualenv** from the agent's. `genlayer-test==0.1.2` pins
`genlayer-py==0.3.0` exactly in its own dependency metadata, which conflicts
with the `genlayer-py>=0.8.0` this repo's agent needs for the current
`write_contract`/`wait_for_transaction_receipt` API. Mixing both in one
environment will leave you with either a broken agent or a broken test
harness depending on which one `pip` resolves last.
