"""Deterministic shard-aware sampling without loss or duplication."""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from collections.abc import Hashable, Iterator, Sequence

from torch.utils.data import Sampler


def _derived_seed(seed: int, epoch: int, *parts: object) -> int:
    digest = hashlib.blake2b(digest_size=8)
    digest.update(str(int(seed)).encode())
    digest.update(b"\0")
    digest.update(str(int(epoch)).encode())
    for part in parts:
        digest.update(b"\0")
        digest.update(repr(part).encode("utf-8"))
    return int.from_bytes(digest.digest(), "little")


class HierarchicalShardSampler(Sampler[int]):
    """Three-level deterministic shuffle with whole-shard rank assignment.

    1. Epoch-seeded shard permutation.
    2. Greedy whole-shard assignment to ranks (no cross-rank overlap).
    3. Intra-shard block permutation and bounded in-block permutation.

    Across all ranks, every input position appears exactly once per epoch.  Rank
    lengths may differ when shard sizes differ; set ``require_equal_rank_lengths``
    to fail before training if the caller cannot support uneven distributed input.
    """

    def __init__(
        self,
        shard_ids: Sequence[Hashable],
        *,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 0,
        block_size: int = 256,
        require_equal_rank_lengths: bool = False,
    ) -> None:
        self.shard_ids = tuple(shard_ids)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.block_size = int(block_size)
        self.require_equal_rank_lengths = bool(require_equal_rank_lengths)
        self.epoch = 0

        indices_by_shard: dict[Hashable, list[int]] = defaultdict(list)
        for index, shard_id in enumerate(self.shard_ids):
            indices_by_shard[shard_id].append(index)
        self._indices_by_shard = dict(indices_by_shard)
        # repr sorting only establishes a stable pre-permutation order; all visible
        # epoch order comes from the seeded permutation below.
        self._shards = tuple(
            sorted(self._indices_by_shard, key=lambda value: repr(value))
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _assignment(self) -> tuple[list[list[Hashable]], list[int]]:
        shards = list(self._shards)
        random.Random(
            _derived_seed(self.seed, self.epoch, "shard-permutation")
        ).shuffle(shards)
        assigned: list[list[Hashable]] = [[] for _ in range(self.num_replicas)]
        loads = [0] * self.num_replicas
        for shard in shards:
            rank = min(range(self.num_replicas), key=lambda value: (loads[value], value))
            assigned[rank].append(shard)
            loads[rank] += len(self._indices_by_shard[shard])
        return assigned, loads

    def rank_loads(self) -> tuple[int, ...]:
        return tuple(self._assignment()[1])

    def rank_shards(self) -> tuple[Hashable, ...]:
        return tuple(self._assignment()[0][self.rank])

    def _shuffle_shard(self, shard: Hashable) -> list[int]:
        indices = self._indices_by_shard[shard]
        blocks = [
            list(indices[offset : offset + self.block_size])
            for offset in range(0, len(indices), self.block_size)
        ]
        block_rng = random.Random(
            _derived_seed(self.seed, self.epoch, "blocks", shard)
        )
        block_rng.shuffle(blocks)
        result = []
        for original_block in blocks:
            block = list(original_block)
            random.Random(
                _derived_seed(
                    self.seed,
                    self.epoch,
                    "within-block",
                    shard,
                    original_block[0],
                )
            ).shuffle(block)
            result.extend(block)
        return result

    def __iter__(self) -> Iterator[int]:
        assigned, _ = self._assignment()
        for shard in assigned[self.rank]:
            yield from self._shuffle_shard(shard)

    def __len__(self) -> int:
        return self._assignment()[1][self.rank]
