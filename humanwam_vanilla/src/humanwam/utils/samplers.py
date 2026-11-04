import bisect
import hashlib
import itertools
import math
import os
from fractions import Fraction
from typing import Iterator, Sized

import torch
from torch.utils.data import Sampler

from ..datasets.locality_sampler import HierarchicalShardSampler


# ``torch.randperm(n).tolist()`` temporarily holds both an int64 tensor and a
# Python-int list.  That is useful for exact shuffle semantics on ordinary
# datasets, but grows to tens of GiB per rank for the 520M-frame ego corpus.
# Above this threshold, sample with replacement in deterministic fixed-size
# blocks instead.  The threshold is environment-overridable so a node with a
# tighter CPU-memory budget can opt into streaming earlier.
DEFAULT_STREAMING_SAMPLER_THRESHOLD = 10_000_000
STREAMING_SAMPLER_BLOCK_SIZE = 65_536


def _positive_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = int(raw)
    return value


def _streaming_block_seed(epoch_seed: int, block_index: int) -> int:
    """Mix an epoch seed and block index into a stable torch RNG seed."""
    # SplitMix64 finalizer.  Explicit masking makes the result independent of
    # Python's unbounded-int behavior and stable across processes/platforms.
    mask64 = (1 << 64) - 1
    x = (
        (int(epoch_seed) & mask64)
        + 0x9E3779B97F4A7C15 * (int(block_index) + 1)
    ) & mask64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & mask64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & mask64
    x ^= x >> 31
    # torch.Generator.manual_seed accepts signed/unsigned 64-bit values, but a
    # non-negative signed-64 seed is the least surprising across torch versions.
    return x & ((1 << 63) - 1)


class ResumableEpochSampler(Sampler[int]):
    def __init__(self, dataset: Sized, seed: int, batch_size: int, num_processes: int):
        self.dataset = dataset
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.num_processes = int(num_processes)
        self.epoch = 0
        self.epoch_offset = 0
        self.resume_batch_offset = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def set_epoch_offset(self, epoch_offset: int):
        self.epoch_offset = int(epoch_offset)

    def set_resume_batch_offset(self, batch_in_epoch: int):
        self.resume_batch_offset = int(batch_in_epoch)

    def clear_resume_batch_offset(self):
        self.resume_batch_offset = 0

    def _sample_offset(self) -> int:
        if self.resume_batch_offset > 0:
            return self.resume_batch_offset * self.batch_size * self.num_processes
        return 0

    def _iter_streaming(self, dataset_size: int, epoch_seed: int, sample_offset: int) -> Iterator[int]:
        """Yield a deterministic, bounded-memory replacement sample stream.

        Fixed independently seeded blocks make resume O(remaining samples in the
        first block), rather than regenerating every sample before the offset.
        All ranks construct the same global stream; Accelerate's batch-sampler
        sharding continues to assign disjoint global-batch slices to each rank.
        """
        block_size = STREAMING_SAMPLER_BLOCK_SIZE
        first_block = sample_offset // block_size
        offset_in_first_block = sample_offset % block_size
        num_blocks = (dataset_size + block_size - 1) // block_size

        for block_index in range(first_block, num_blocks):
            block_start = block_index * block_size
            current_size = min(block_size, dataset_size - block_start)
            generator = torch.Generator(device="cpu")
            generator.manual_seed(_streaming_block_seed(epoch_seed, block_index))
            indices = torch.randint(
                low=0,
                high=dataset_size,
                size=(current_size,),
                generator=generator,
                dtype=torch.int64,
                device="cpu",
            ).tolist()
            if block_index == first_block and offset_in_first_block:
                indices = indices[offset_in_first_block:]
            yield from indices

    def __iter__(self) -> Iterator[int]:
        dataset_size = len(self.dataset)
        g = torch.Generator(device="cpu")
        epoch_seed = self.seed + self.epoch + self.epoch_offset
        g.manual_seed(epoch_seed)
        sample_offset = min(self._sample_offset(), dataset_size)

        streaming_threshold = _positive_int_env(
            "FASTWAM_STREAMING_SAMPLER_THRESHOLD",
            DEFAULT_STREAMING_SAMPLER_THRESHOLD,
        )
        if dataset_size > streaming_threshold:
            return self._iter_streaming(dataset_size, epoch_seed, sample_offset)

        # Preserve the exact historical sequence and eager-list behavior for
        # ordinary datasets, including resume slicing semantics.
        indices = torch.randperm(dataset_size, generator=g).tolist()
        return iter(indices[sample_offset:])

    def __len__(self) -> int:
        return len(self.dataset)


class RankAwareStrictResumableSampler(Sampler[int]):
    """Rank-local deterministic sampler with equal per-rank lengths.

    The DataLoader using this sampler is already rank-local and must not be
    sharded by Accelerate a second time. For explicitly opted-in datasets whose
    length is not divisible by world size,
    ``FASTWAM_STRICT_SAMPLER_TRIM_REMAINDER=1`` shuffles globally and rotates
    the omitted final remainder between epochs without duplicating samples.
    """

    def __init__(
        self,
        dataset: Sized,
        seed: int,
        batch_size: int,
        num_processes: int,
        process_index: int,
    ) -> None:
        self.dataset = dataset
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.num_processes = int(num_processes)
        self.process_index = int(process_index)
        self.epoch = 0
        self.epoch_offset = 0
        self.resume_batch_offset = 0
        self.dataset_size = len(dataset)
        self.trim_remainder = False
        if self.dataset_size % self.num_processes != 0:
            self.trim_remainder = True
        self.samples_per_rank = self.dataset_size // self.num_processes
        self.effective_size = self.samples_per_rank * self.num_processes
        self._shard_ids = tuple(
            index // self.samples_per_rank for index in range(self.effective_size)
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def set_epoch_offset(self, epoch_offset: int) -> None:
        self.epoch_offset = int(epoch_offset)

    def set_resume_batch_offset(self, batch_in_epoch: int) -> None:
        self.resume_batch_offset = int(batch_in_epoch)

    def clear_resume_batch_offset(self) -> None:
        self.resume_batch_offset = 0

    def _sampler(self) -> HierarchicalShardSampler:
        sampler = HierarchicalShardSampler(
            self._shard_ids,
            num_replicas=self.num_processes,
            rank=self.process_index,
            seed=self.seed,
            require_equal_rank_lengths=True,
        )
        sampler.set_epoch(self.epoch + self.epoch_offset)
        return sampler

    def __iter__(self) -> Iterator[int]:
        offset = min(
            self.resume_batch_offset * self.batch_size,
            self.samples_per_rank,
        )
        if self.trim_remainder:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.seed + self.epoch + self.epoch_offset)
            indices = torch.randperm(
                self.dataset_size,
                generator=generator,
                dtype=torch.int64,
                device="cpu",
            )
            start = self.process_index * self.samples_per_rank + offset
            stop = (self.process_index + 1) * self.samples_per_rank
            return iter(indices[start:stop].tolist())
        return itertools.islice(iter(self._sampler()), offset, None)

    def __len__(self) -> int:
        return self.samples_per_rank

    def rank_loads(self) -> tuple[int, ...]:
        if self.trim_remainder:
            return (self.samples_per_rank,) * self.num_processes
        return self._sampler().rank_loads()


def _stable_uint64(*values: object) -> int:
    payload = "\x1f".join(str(value) for value in values).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _affine_permutation_parameters(size: int, seed: int) -> tuple[int, int]:
    if size == 1:
        return 0, 0
    multiplier = int(seed % size) | 1
    while math.gcd(multiplier, size) != 1:
        multiplier = (multiplier + 2) % size
        if multiplier == 0:
            multiplier = 1
    offset = int(_stable_uint64(seed, "offset") % size)
    return multiplier, offset


class EnvironmentUniformResumableSampler(ResumableEpochSampler):
    """Sample environments uniformly without forcing per-batch stratification.

    The dataset exposes compact global-index spans per environment. The first
    ``training_total_samples`` entries contain exactly the same number of samples from
    every environment and no repeated anchor within an environment. Balanced
    environment slots are shuffled in bounded-memory blocks, so individual global
    batches fluctuate naturally instead of containing a fixed quota per environment.
    The existing batch/rank resume contract is retained.
    """

    def __init__(self, dataset: Sized, seed: int, batch_size: int, num_processes: int):
        super().__init__(dataset, seed, batch_size, num_processes)
        contract = getattr(dataset, "environment_sampling", None)
        self.environment_ids = tuple(contract["environment_ids"])
        self.spans = tuple(tuple(value) for value in contract["spans"])
        self.capacities = tuple(int(value) for value in contract["capacities"])
        self.training_total_samples = int(contract["training_total_samples"])
        self.samples_per_environment = int(contract["samples_per_environment"])
        span_ends = []
        for environment_spans in self.spans:
            cumulative = 0
            environment_ends = []
            for start, end in environment_spans:
                cumulative += end - start
                environment_ends.append(cumulative)
            span_ends.append(tuple(environment_ends))
        self._span_ends = tuple(span_ends)

    def _global_index(self, environment: int, local_index: int) -> int:
        ends = self._span_ends[environment]
        span_position = bisect.bisect_right(ends, local_index)
        previous_end = 0 if span_position == 0 else ends[span_position - 1]
        start, _ = self.spans[environment][span_position]
        return int(start + local_index - previous_end)

    def _iter_shuffled_slots(
        self,
        *,
        phase_start: int,
        phase_stop: int,
        output_start: int,
        epoch_seed: int,
        phase_key: str,
    ) -> Iterator[int]:
        """Shuffle a contiguous slot range in deterministic bounded-memory blocks."""
        if output_start >= phase_stop:
            return
        environment_count = len(self.environment_ids)
        # Full blocks contain an integer number of copies of every environment.
        # Consequently each output position has the same marginal environment
        # probability, while batches are not explicitly stratified.
        block_size = max(
            environment_count,
            (STREAMING_SAMPLER_BLOCK_SIZE // environment_count) * environment_count,
        )
        relative_start = output_start - phase_start
        phase_size = phase_stop - phase_start
        first_block = relative_start // block_size
        offset_in_first_block = relative_start % block_size
        num_blocks = (phase_size + block_size - 1) // block_size

        for block_index in range(first_block, num_blocks):
            block_start = block_index * block_size
            current_size = min(block_size, phase_size - block_start)
            generator = torch.Generator(device="cpu")
            generator.manual_seed(
                _streaming_block_seed(
                    _stable_uint64(epoch_seed, phase_key), block_index
                )
            )
            order = torch.randperm(
                current_size,
                generator=generator,
                dtype=torch.int64,
                device="cpu",
            ).tolist()
            if block_index == first_block and offset_in_first_block:
                order = order[offset_in_first_block:]
            for block_offset in order:
                yield phase_start + block_start + block_offset

    def __iter__(self) -> Iterator[int]:
        dataset_size = len(self.dataset)
        sample_offset = min(self._sample_offset(), dataset_size)
        epoch_seed = self.seed + self.epoch + self.epoch_offset
        environment_count = len(self.environment_ids)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(epoch_seed)
        environment_order = torch.randperm(
            environment_count, generator=generator
        ).tolist()
        first_pass_parameters = [
            _affine_permutation_parameters(
                capacity,
                _stable_uint64(epoch_seed, environment_id, 0),
            )
            for environment_id, capacity in zip(
                self.environment_ids, self.capacities, strict=True
            )
        ]
        later_parameters: dict[tuple[int, int], tuple[int, int]] = {}

        def iterator() -> Iterator[int]:
            if sample_offset < self.training_total_samples:
                prefix_slots = self._iter_shuffled_slots(
                    phase_start=0,
                    phase_stop=self.training_total_samples,
                    output_start=sample_offset,
                    epoch_seed=epoch_seed,
                    phase_key="training",
                )
            else:
                prefix_slots = iter(())
            suffix_start = max(sample_offset, self.training_total_samples)
            suffix_slots = self._iter_shuffled_slots(
                phase_start=self.training_total_samples,
                phase_stop=dataset_size,
                output_start=suffix_start,
                epoch_seed=epoch_seed,
                phase_key="post-training",
            )
            for position in itertools.chain(prefix_slots, suffix_slots):
                if position < self.training_total_samples:
                    round_index, cycle_position = divmod(
                        position, environment_count
                    )
                else:
                    post_position = position - self.training_total_samples
                    post_round, cycle_position = divmod(
                        post_position, environment_count
                    )
                    round_index = self.samples_per_environment + post_round
                environment = environment_order[cycle_position]
                capacity = self.capacities[environment]
                pass_index, ordinal = divmod(round_index, capacity)
                if pass_index == 0:
                    multiplier, offset = first_pass_parameters[environment]
                else:
                    key = (environment, pass_index)
                    parameters = later_parameters.get(key)
                    if parameters is None:
                        parameters = _affine_permutation_parameters(
                            capacity,
                            _stable_uint64(
                                epoch_seed,
                                self.environment_ids[environment],
                                pass_index,
                            ),
                        )
                        later_parameters[key] = parameters
                    multiplier, offset = parameters
                local_index = (multiplier * ordinal + offset) % capacity
                yield self._global_index(environment, local_index)

        return iterator()


class TaskWeightedResumableSampler(ResumableEpochSampler):
    """Deterministic replacement sampling from exact per-task integer mass.

    Every rank constructs the same global sample stream. Accelerate keeps its
    historical responsibility for sharding global batches across ranks, matching
    ``ResumableEpochSampler``. The stream is rounded up to a complete global-batch
    multiple so Accelerate never performs resume-dependent ``even_batches`` tail
    recycling. Fixed independently seeded blocks make exact-state resume
    proportional only to the remainder of the first block.
    """

    def __init__(self, dataset: Sized, seed: int, batch_size: int, num_processes: int):
        super().__init__(dataset, seed, batch_size, num_processes)
        contract = getattr(dataset, "task_weighted_sampling", None)
        raw_tasks = contract.get("tasks")

        self.task_ids = []
        self.starts = []
        self.ends = []
        self.capacities = []
        self.integer_weights = []
        self.mass_ends = []
        total_mass = 0
        for position, entry in enumerate(raw_tasks):
            task_id = str(entry.get("task_id", "")).strip()
            start = int(entry.get("start", -1))
            end = int(entry.get("end", -1))
            capacity = int(entry.get("capacity", -1))
            integer_weight = int(entry.get("integer_weight", -1))
            task_mass = capacity * integer_weight
            total_mass += task_mass
            self.task_ids.append(task_id)
            self.starts.append(start)
            self.ends.append(end)
            self.capacities.append(capacity)
            self.integer_weights.append(integer_weight)
            self.mass_ends.append(total_mass)

        self.total_integer_mass = int(contract.get("total_integer_mass", -1))
        self.native_epoch_samples = int(
            contract.get("native_epoch_samples", -1)
        )
        self.global_batch_size = int(contract.get("global_batch_size", -1))
        self.expected_batch_size_per_rank = int(
            contract.get("expected_batch_size_per_rank", -1)
        )
        self.expected_num_processes = int(
            contract.get("expected_num_processes", -1)
        )
        self.effective_epoch_samples = int(
            contract.get("effective_epoch_samples", -1)
        )

    def sampling_report(self) -> dict[str, object]:
        return {
            "mode": "task_weighted_replacement_v1",
            "replacement": True,
            "expected_batch_size_per_rank": self.expected_batch_size_per_rank,
            "expected_num_processes": self.expected_num_processes,
            "global_batch_size": self.global_batch_size,
            "native_epoch_samples": self.native_epoch_samples,
            "effective_epoch_samples": self.effective_epoch_samples,
            "padding_equivalent_extra": (
                self.effective_epoch_samples - self.native_epoch_samples
            ),
            "exact_resume_topology": (
                "exact only when runtime batch_size_per_rank/num_processes/global "
                "batch equal contract topology "
                f"{self.expected_batch_size_per_rank}/"
                f"{self.expected_num_processes}/{self.global_batch_size}"
            ),
            "total_integer_mass": self.total_integer_mass,
            "tasks": [
                {
                    "task_id": task_id,
                    "capacity": capacity,
                    "integer_weight": integer_weight,
                    "expected_task_probability": (
                        capacity * integer_weight / self.total_integer_mass
                    ),
                    "per_anchor_probability": (
                        integer_weight / self.total_integer_mass
                    ),
                    "per_anchor_expected_dose_ratio_vs_uniform_same_effective_slots": (
                        integer_weight
                        * self.native_epoch_samples
                        / self.total_integer_mass
                    ),
                }
                for task_id, capacity, integer_weight in zip(
                    self.task_ids,
                    self.capacities,
                    self.integer_weights,
                    strict=True,
                )
            ],
        }

    def __iter__(self) -> Iterator[int]:
        sample_offset = min(self._sample_offset(), self.effective_epoch_samples)
        epoch_seed = self.seed + self.epoch + self.epoch_offset
        block_size = STREAMING_SAMPLER_BLOCK_SIZE
        first_block = sample_offset // block_size
        offset_in_first_block = sample_offset % block_size
        num_blocks = (
            self.effective_epoch_samples + block_size - 1
        ) // block_size

        def iterator() -> Iterator[int]:
            for block_index in range(first_block, num_blocks):
                block_start = block_index * block_size
                current_size = min(
                    block_size, self.effective_epoch_samples - block_start
                )
                generator = torch.Generator(device="cpu")
                generator.manual_seed(
                    _streaming_block_seed(
                        _stable_uint64(epoch_seed, "task-weighted-replacement"),
                        block_index,
                    )
                )
                mass_slots = torch.randint(
                    low=0,
                    high=self.total_integer_mass,
                    size=(current_size,),
                    generator=generator,
                    dtype=torch.int64,
                    device="cpu",
                ).tolist()
                if block_index == first_block and offset_in_first_block:
                    mass_slots = mass_slots[offset_in_first_block:]
                for mass_slot in mass_slots:
                    task_position = bisect.bisect_right(self.mass_ends, mass_slot)
                    previous_mass_end = (
                        0 if task_position == 0 else self.mass_ends[task_position - 1]
                    )
                    task_mass_slot = mass_slot - previous_mass_end
                    local_index = (
                        task_mass_slot // self.integer_weights[task_position]
                    )
                    yield self.starts[task_position] + local_index

        return iterator()

    def __len__(self) -> int:
        return self.effective_epoch_samples


class SourceMixResumableSampler(ResumableEpochSampler):
    """Route robot/Ego samples using either batch or sample-level mixing.

    Batch-homogeneous routing remains available when the two sources use
    different image sizes. Sample-level routing is intended for aligned source
    contracts whose examples can be collated together. The Ego route additionally
    preserves the exact environment-uniform contract exposed by the K-condition
    dataset.
    """

    def __init__(self, dataset: Sized, seed: int, batch_size: int, num_processes: int):
        super().__init__(dataset, seed, batch_size, num_processes)
        contract = getattr(dataset, "source_mix_sampling", None)
        self.robot_size = int(contract["robot_size"])
        self.ego_size = int(contract["ego_size"])
        self.ego_ratio = float(contract["ego_ratio"])
        self.mix_mode = str(contract.get("mix_mode", "batch_homogeneous")).strip().lower()
        raw_epoch_steps = contract.get("epoch_steps")
        self.epoch_steps = (
            None if raw_epoch_steps is None else int(raw_epoch_steps)
        )
        self.global_micro_batch = self.batch_size * self.num_processes
        fraction = Fraction(self.ego_ratio).limit_denominator(1000)
        self.ego_slots = int(fraction.numerator)
        self.period = int(fraction.denominator)
        self.ego_environment = contract.get("ego_environment_sampling")
        self._ego_span_ends = None
        if isinstance(self.ego_environment, dict) and self.ego_environment.get("mode") == "uniform":
            self.ego_environment_ids = tuple(self.ego_environment["environment_ids"])
            self.ego_spans = tuple(
                tuple((int(start), int(end)) for start, end in spans)
                for spans in self.ego_environment["spans"]
            )
            self.ego_capacities = tuple(
                int(value) for value in self.ego_environment["capacities"]
            )
            span_ends = []
            for spans in self.ego_spans:
                cumulative = 0
                ends = []
                for start, end in spans:
                    cumulative += end - start
                    ends.append(cumulative)
                span_ends.append(tuple(ends))
            self._ego_span_ends = tuple(span_ends)

    def __len__(self) -> int:
        if self.epoch_steps is not None:
            return self.epoch_steps * self.global_micro_batch
        return (
            (len(self.dataset) + self.global_micro_batch - 1)
            // self.global_micro_batch
            * self.global_micro_batch
        )

    def _period_slots(self, epoch_seed: int) -> tuple[bool, ...]:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_stable_uint64(epoch_seed, "source-period") & ((1 << 63) - 1))
        order = torch.randperm(self.period, generator=generator).tolist()
        ego_positions = set(order[: self.ego_slots])
        return tuple(position in ego_positions for position in range(self.period))

    def _source_batches_before(self, batch_index: int, slots: tuple[bool, ...]) -> int:
        full_periods, remainder = divmod(int(batch_index), self.period)
        return full_periods * self.ego_slots + sum(slots[:remainder])

    def _source_samples_before(self, sample_index: int, slots: tuple[bool, ...]) -> int:
        """Count Ego samples before a position in the mixed global stream."""
        full_periods, remainder = divmod(int(sample_index), self.period)
        return full_periods * self.ego_slots + sum(slots[:remainder])

    def _ego_global_index(self, ordinal: int, epoch_seed: int) -> int:
        if self._ego_span_ends is None:
            return int(_stable_uint64(epoch_seed, "ego", ordinal) % self.ego_size)
        environment_count = len(self.ego_environment_ids)
        environment = int(ordinal % environment_count)
        pass_ordinal = int(ordinal // environment_count)
        capacity = self.ego_capacities[environment]
        multiplier, offset = _affine_permutation_parameters(
            capacity,
            _stable_uint64(
                epoch_seed,
                self.ego_environment_ids[environment],
                pass_ordinal // max(capacity, 1),
            ),
        )
        local_index = (multiplier * pass_ordinal + offset) % capacity
        ends = self._ego_span_ends[environment]
        span_position = bisect.bisect_right(ends, local_index)
        previous_end = 0 if span_position == 0 else ends[span_position - 1]
        start, _ = self.ego_spans[environment][span_position]
        return int(start + local_index - previous_end)

    def __iter__(self) -> Iterator[tuple[int, int]]:
        total_samples = len(self)
        sample_offset = min(self._sample_offset(), total_samples)
        epoch_seed = self.seed + self.epoch + self.epoch_offset
        slots = self._period_slots(epoch_seed)

        if self.mix_mode == "sample_mixed":
            def mixed_iterator() -> Iterator[tuple[int, int]]:
                for global_index in range(sample_offset, total_samples):
                    is_ego = slots[global_index % self.period]
                    ego_before = self._source_samples_before(global_index, slots)
                    if is_ego:
                        yield (1, self._ego_global_index(ego_before, epoch_seed))
                    else:
                        robot_ordinal = global_index - ego_before
                        yield (
                            0,
                            int(_stable_uint64(epoch_seed, "robot", robot_ordinal) % self.robot_size),
                        )

            return mixed_iterator()

        start_batch, offset_in_batch = divmod(sample_offset, self.global_micro_batch)

        def iterator() -> Iterator[tuple[int, int]]:
            total_batches = total_samples // self.global_micro_batch
            for batch_index in range(start_batch, total_batches):
                is_ego = slots[batch_index % self.period]
                ego_batches_before = self._source_batches_before(batch_index, slots)
                robot_batches_before = batch_index - ego_batches_before
                batch_ordinal = ego_batches_before if is_ego else robot_batches_before
                start = offset_in_batch if batch_index == start_batch else 0
                for within_batch in range(start, self.global_micro_batch):
                    ordinal = batch_ordinal * self.global_micro_batch + within_batch
                    if is_ego:
                        yield (1, self._ego_global_index(ordinal, epoch_seed))
                    else:
                        local = int(
                            _stable_uint64(epoch_seed, "robot", ordinal)
                            % self.robot_size
                        )
                        yield (0, local)

        return iterator()


def build_resumable_sampler(
    dataset: Sized,
    seed: int,
    batch_size: int,
    num_processes: int,
    process_index: int = 0,
    rank_aware_strict: bool = False,
) -> Sampler[int]:
    task_weighted_sampling = getattr(dataset, "task_weighted_sampling", None)
    source_mix_sampling = getattr(dataset, "source_mix_sampling", None)
    environment_sampling = getattr(dataset, "environment_sampling", None)
    if rank_aware_strict:
        return RankAwareStrictResumableSampler(
            dataset,
            seed=seed,
            batch_size=batch_size,
            num_processes=num_processes,
            process_index=process_index,
        )
    if source_mix_sampling is not None:
        return SourceMixResumableSampler(
            dataset, seed=seed, batch_size=batch_size, num_processes=num_processes
        )
    if environment_sampling is not None:
        return EnvironmentUniformResumableSampler(
            dataset, seed=seed, batch_size=batch_size, num_processes=num_processes
        )
    if task_weighted_sampling is not None:
        return TaskWeightedResumableSampler(
            dataset, seed=seed, batch_size=batch_size, num_processes=num_processes
        )
    return ResumableEpochSampler(
        dataset, seed=seed, batch_size=batch_size, num_processes=num_processes
    )
