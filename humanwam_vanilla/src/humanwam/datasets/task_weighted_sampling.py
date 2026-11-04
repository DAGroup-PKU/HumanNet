"""Task-weighted sampling settings for a single dataset corpus.

The contract changes only how existing global frame-anchor indices are sampled.
It does not duplicate roots, alter examples, create source-specific normalization,
or attach a task-dependent loss weight.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any


TASK_WEIGHTED_SAMPLING_MODE = "task_weighted_replacement_v1"


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _positive_fraction(value: Any, *, location: str) -> Fraction:
    fraction = Fraction(str(value))
    return fraction


def build_task_weighted_sampling_contract(
    dataset_dirs: Sequence[str | Path],
    root_capacities: Sequence[int],
    config: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Return an exact-root, integer-mass sampling contract or ``None``.

    A configured weight applies to every existing frame anchor under its exact
    root. Sampling is with replacement from the expanded integer mass. Thus an
    anchor with integer weight 2 has exactly twice the probability of an anchor
    with integer weight 1. The effective stream length is the native corpus size
    rounded up to a complete configured global batch, preventing Accelerate from
    adding resume-dependent ``even_batches`` samples.
    """

    if config is None:
        return None
    raw_rules = config.get("rules")
    topology = {}
    for field in (
        "expected_batch_size_per_rank",
        "expected_num_processes",
        "global_batch_size",
    ):
        raw_value = config.get(field)
        topology[field] = raw_value
    global_batch_size = topology["global_batch_size"]

    roots = [str(Path(path).expanduser().resolve()) for path in dataset_dirs]
    capacities = [int(value) for value in root_capacities]

    by_root: dict[str, tuple[str, Fraction, str]] = {}
    task_ids: set[str] = set()
    for position, raw_rule in enumerate(raw_rules):
        root = str(Path(str(raw_rule["dataset_root"])).expanduser().resolve())
        task_id = str(raw_rule["task_id"]).strip()
        fraction = _positive_fraction(
            raw_rule["weight"],
            location=f"task_weighted_sampling.rules[{position}].weight",
        )
        by_root[root] = (task_id, fraction, str(raw_rule["weight"]))
        task_ids.add(task_id)

    denominator_lcm = 1
    ordered_fractions = []
    for root in roots:
        fraction = by_root[root][1]
        ordered_fractions.append(fraction)
        denominator_lcm = math.lcm(denominator_lcm, fraction.denominator)
    integer_weights = [
        fraction.numerator * (denominator_lcm // fraction.denominator)
        for fraction in ordered_fractions
    ]
    common_gcd = math.gcd(*integer_weights)
    integer_weights = [value // common_gcd for value in integer_weights]

    entries = []
    start = 0
    native_epoch_samples = sum(capacities)
    total_mass = sum(
        capacity * weight
        for capacity, weight in zip(capacities, integer_weights, strict=True)
    )
    for root, capacity, integer_weight in zip(
        roots, capacities, integer_weights, strict=True
    ):
        task_id, configured_fraction, configured_text = by_root[root]
        end = start + capacity
        probability_numerator = capacity * integer_weight
        entries.append(
            {
                "dataset_root": root,
                "task_id": task_id,
                "start": start,
                "end": end,
                "capacity": capacity,
                "configured_weight": configured_text,
                "configured_weight_fraction": (
                    f"{configured_fraction.numerator}/{configured_fraction.denominator}"
                ),
                "integer_weight": integer_weight,
                "expected_task_probability_numerator": probability_numerator,
                "expected_task_probability_denominator": total_mass,
                "expected_task_probability": probability_numerator / total_mass,
                "per_anchor_probability_numerator": integer_weight,
                "per_anchor_probability_denominator": total_mass,
                "per_anchor_expected_dose_ratio_vs_uniform_same_effective_slots": (
                    integer_weight * native_epoch_samples / total_mass
                ),
            }
        )
        start = end

    effective_epoch_samples = (
        (native_epoch_samples + global_batch_size - 1) // global_batch_size
    ) * global_batch_size
    payload = {
        "mode": TASK_WEIGHTED_SAMPLING_MODE,
        "replacement": True,
        "expected_batch_size_per_rank": topology[
            "expected_batch_size_per_rank"
        ],
        "expected_num_processes": topology["expected_num_processes"],
        "global_batch_size": global_batch_size,
        "native_epoch_samples": native_epoch_samples,
        "effective_epoch_samples": effective_epoch_samples,
        "padding_equivalent_extra": effective_epoch_samples - native_epoch_samples,
        "total_integer_mass": total_mass,
        "weighting": "uniform within exact task root; per-anchor mass proportional to integer_weight",
        "tasks": entries,
    }
    return {**payload, "sha256": _canonical_sha256(payload)}
