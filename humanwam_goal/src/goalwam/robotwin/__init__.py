"""RoboTwin 2.0 evaluation layer: observation bridge, released goal cache and session."""

from importlib import import_module

# Exports load on first use, so `python -m goalwam.robotwin.goal_cache` runs that module
# without it being imported twice.
_EXPORTS = {
    "ACTION_MASK": "bridge",
    "ROBOT": "bridge",
    "STATE_MASK": "bridge",
    "encode_observation": "bridge",
    "joint_commands": "bridge",
    "native_observation": "bridge",
    "RPC_PROTOCOL": "goal_cache",
    "OfficialGoalCache": "goal_cache",
    "load_profile": "goal_cache",
    "CUBLASLT_12_8_3_14_SHA256": "runtime",
    "cublaslt_runtime": "runtime",
    "RoboTwinSession": "session",
}
__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
