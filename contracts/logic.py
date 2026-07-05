"""Off-chain access to the REAL contract's pure helpers.

contracts/crisis_negotiator.py must be fully self-contained because
deploy_contract(code=...) sends exactly one source file on-chain -- it
cannot import sibling modules. But its deterministic helpers
(apply_action, parse_reward_output, ...) still need to be testable with
plain pytest, and agent/env.py's MockEnv needs the same state machine.

Rather than maintaining a hand-written mirror that could silently drift
from the deployed code, this module execs the contract's actual source
with a stubbed `genlayer` module (the real one only exists inside the
GenVM runtime; the `genlayer` package on PyPI is a 1.3 KB placeholder).
Everything exported here IS the deployed code, byte for byte.

The stub mimics just enough of the GenVM SDK surface for the contract to
be importable and its deterministic paths runnable off-chain:
  - gl.Contract auto-creates annotated storage fields on instantiation
    (GenVM pre-creates storage slots; plain Python doesn't).
  - TreeMap behaves as a dict; u256 as an int.
  - gl.nondet / gl.eq_principle / gl.vm raise on use, so nothing
    off-chain can accidentally "succeed" at calling an LLM.
"""

from __future__ import annotations

import sys
import types
import typing
from pathlib import Path

_CONTRACT_PATH = Path(__file__).resolve().parent / "crisis_negotiator.py"


class _StubTreeMap(dict):
    """Off-chain stand-in for GenVM's TreeMap storage type."""

    def __class_getitem__(cls, item):
        return types.GenericAlias(cls, item if isinstance(item, tuple) else (item,))


class _StubU256(int):
    """Off-chain stand-in for GenVM's u256 storage type."""


def _default_for_annotation(annotation):
    origin = typing.get_origin(annotation)
    if origin is _StubTreeMap or annotation is _StubTreeMap:
        return _StubTreeMap()
    if annotation is _StubU256:
        return _StubU256(0)
    if annotation is str:
        return ""
    if annotation is int:
        return 0
    if annotation is bool:
        return False
    return None


class _StubContract:
    """Pre-creates annotated storage fields, the way GenVM does for real
    contracts, so contract __init__ methods that only item-assign into
    TreeMaps work identically off-chain."""

    def __new__(cls):
        instance = super().__new__(cls)
        for name, annotation in getattr(cls, "__annotations__", {}).items():
            object.__setattr__(instance, name, _default_for_annotation(annotation))
        return instance


def _make_genlayer_stub() -> types.ModuleType:
    def _gl_runtime_only(*_args, **_kwargs):
        raise NotImplementedError(
            "gl.nondet / gl.eq_principle / gl.vm only exist inside the GenVM "
            "runtime. Off-chain code may only use the contract's deterministic "
            "helpers (apply_action, parse_reward_output, ...)."
        )

    class _Public:
        # @gl.public.view / @gl.public.write become identity decorators
        # off-chain: the methods stay callable, they just aren't registered
        # with any runtime.
        @staticmethod
        def view(fn):
            return fn

        @staticmethod
        def write(fn):
            return fn

    gl = types.SimpleNamespace(
        Contract=_StubContract,
        public=_Public,
        nondet=types.SimpleNamespace(
            exec_prompt=_gl_runtime_only,
            web=types.SimpleNamespace(get=_gl_runtime_only, render=_gl_runtime_only),
        ),
        eq_principle=types.SimpleNamespace(
            prompt_comparative=_gl_runtime_only,
            prompt_non_comparative=_gl_runtime_only,
            strict_eq=_gl_runtime_only,
        ),
        vm=types.SimpleNamespace(run_nondet=_gl_runtime_only),
    )

    module = types.ModuleType("genlayer")
    module.gl = gl  # type: ignore[attr-defined]
    module.TreeMap = _StubTreeMap  # type: ignore[attr-defined]
    module.u256 = _StubU256  # type: ignore[attr-defined]
    module.__all__ = ["gl", "TreeMap", "u256"]  # type: ignore[attr-defined]
    return module


def _load_contract_namespace() -> dict:
    source = _CONTRACT_PATH.read_text(encoding="utf-8")
    original = sys.modules.get("genlayer")
    sys.modules["genlayer"] = _make_genlayer_stub()
    try:
        namespace: dict = {"__name__": "contracts._crisis_negotiator_offchain"}
        # dont_inherit=True: without it, compile() inherits THIS module's
        # `from __future__ import annotations`, which would stringify the
        # contract's class annotations and break the stub's storage-field
        # auto-creation (it inspects the evaluated annotation objects).
        exec(compile(source, str(_CONTRACT_PATH), "exec", dont_inherit=True), namespace)
        return namespace
    finally:
        if original is not None:
            sys.modules["genlayer"] = original
        else:
            del sys.modules["genlayer"]


_ns = _load_contract_namespace()

ZONES = _ns["ZONES"]
RESOURCES = _ns["RESOURCES"]
REWARD_EQUIVALENCE_PRINCIPLE = _ns["REWARD_EQUIVALENCE_PRINCIPLE"]
initial_resources = _ns["initial_resources"]
initial_zone_status = _ns["initial_zone_status"]
apply_action = _ns["apply_action"]
build_reward_prompt = _ns["build_reward_prompt"]
parse_reward_output = _ns["parse_reward_output"]
score_to_x100 = _ns["score_to_x100"]
normalize_reward_for_consensus = _ns["normalize_reward_for_consensus"]
# The contract class itself, with gl stubbed: deterministic methods
# (__init__, get_state, get_score) work off-chain; take_action raises
# NotImplementedError the moment it reaches the equivalence principle.
CrisisNegotiator = _ns["CrisisNegotiator"]

__all__ = [
    "ZONES",
    "RESOURCES",
    "REWARD_EQUIVALENCE_PRINCIPLE",
    "initial_resources",
    "initial_zone_status",
    "apply_action",
    "build_reward_prompt",
    "parse_reward_output",
    "score_to_x100",
    "normalize_reward_for_consensus",
    "CrisisNegotiator",
]
