"""Uniform runtime sampling from each LeRobot demo's unique prompt set."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


class EpisodeInstructionSampler:
    """Deduplicate ``episodes.jsonl.tasks`` and sample one prompt per access."""

    def __init__(
        self,
        dataset_dirs: Sequence[str | Path],
        selected_episodes: Sequence[Sequence[int] | None],
    ):
        pools_by_dataset: list[dict[int, tuple[str, ...]]] = []
        unique_prompts: set[str] = set()
        pool_sizes: list[int] = []
        source_ids: list[str] = []
        for root_value, selected_value in zip(
            dataset_dirs, selected_episodes, strict=True
        ):
            root = Path(root_value).expanduser().resolve()
            episodes_path = root / "meta" / "episodes.jsonl"
            selected = (
                None
                if selected_value is None
                else {int(episode) for episode in selected_value}
            )
            pools: dict[int, tuple[str, ...]] = {}
            with episodes_path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    episode_index = int(row.get("episode_index", -1))
                    if selected is not None and episode_index not in selected:
                        continue
                    raw_prompts = row.get("tasks")
                    prompts = tuple(
                        dict.fromkeys(str(prompt) for prompt in raw_prompts)
                    )
                    pools[episode_index] = prompts
                    unique_prompts.update(prompts)
                    pool_sizes.append(len(prompts))
            pools_by_dataset.append(pools)
            source_ids.append(root.name)
        self._pools_by_dataset = tuple(pools_by_dataset)
        self.unique_prompt_count = len(unique_prompts)
        self.pool_count = len(pool_sizes)
        self.min_pool_size = min(pool_sizes)
        self.max_pool_size = max(pool_sizes)
        self.source_ids = tuple(source_ids)

    def sample_task(
        self,
        dataset_index: int,
        episode_index: int,
        *,
        rng: Any | None = None,
    ) -> str:
        pools = self._pools_by_dataset[int(dataset_index)]
        prompts = pools[int(episode_index)]
        generator = np.random if rng is None else rng
        return prompts[int(generator.randint(len(prompts)))]

    def prompts_for_episode(
        self, dataset_index: int, episode_index: int
    ) -> tuple[str, ...]:
        return self._pools_by_dataset[int(dataset_index)][int(episode_index)]
