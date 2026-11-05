"""GoalWAM inference."""

__version__ = "0.1.0"


def __getattr__(name):
    if name == "GoalWAM":
        from .engine import GoalWAM

        return GoalWAM
    raise AttributeError(name)


__all__ = ["GoalWAM"]
