# Modified by Open-HumanScale contributors: retain local training-state management only.
"""Retention of completed local training states; weight files are kept intact."""
import os
from pathlib import Path
import re
import shutil
from typing import Iterable

_STEP_RE = re.compile(r"^step_(\d+)(?:\.(?:adapter|bottleneck))?(?:\.pt)?$")


def checkpoint_step(path: os.PathLike[str] | str) -> int | None:
    match = _STEP_RE.match(Path(path).name)
    return int(match.group(1)) if match else None


def _sorted_steps(paths: Iterable[Path]) -> list[Path]:
    return sorted(
        (path for path in paths if checkpoint_step(path) is not None),
        key=lambda path: checkpoint_step(path) or -1,
    )


def prune_training_states(state_dir: os.PathLike[str] | str, keep_last: int) -> list[Path]:
    """Delete complete trainer states except the newest ``keep_last`` entries."""
    root = Path(state_dir)
    if not root.is_dir():
        return []
    complete = _sorted_steps(
        path
        for path in root.iterdir()
        if path.is_dir() and (path / "trainer_state.json").is_file()
    )
    victims = complete[:-keep_last] if keep_last else complete
    for path in victims:
        shutil.rmtree(path)
    return victims
