# Modified by Open-HumanScale contributors: removed unused release runtime code.
"""One text contract for training samples and offline cache validation."""

from collections.abc import Mapping


def task_instruction(data: Mapping) -> str:
    """Preserve the complete task, including literal @ signs and whitespace."""
    task = data.get("task")
    return task
