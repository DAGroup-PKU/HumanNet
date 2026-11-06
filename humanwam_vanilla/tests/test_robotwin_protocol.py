import importlib.util
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    'robotwin_protocol', Path(__file__).resolve().parents[1] / 'scripts/robotwin_protocol.py')
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
SeedRetries = _module.SeedRetries


class Unstable(Exception):
    pass


def test_only_historical_seed_failures_are_retryable():
    retries = SeedRetries(7, Unstable)
    for error in (Unstable(), TypeError(), IndexError(), np.linalg.LinAlgError()):
        assert retries.retry(error, 'expert')
        assert retries.retry(error, 'policy_setup')
    move = AssertionError('target_pose cannot be None for move action.')
    assert retries.retry(move, 'expert')
    assert not retries.retry(move, 'policy_setup')
    for stage in ('expert', 'policy_setup'):
        for error in (AssertionError('unexpected'), RuntimeError('CUDA failure'), ValueError()):
            assert not retries.retry(error, stage)


def test_overlay_budget_is_shared_across_expert_and_setup():
    retries = SeedRetries(7, Unstable)
    assert retries.retry(TypeError(), 'policy_setup')
    for _ in range(111):
        assert retries.retry(IndexError(), 'expert')
    with pytest.raises(RuntimeError, match='overlay retry budget'):
        retries.retry(TypeError(), 'expert')


def test_policy_setup_and_move_assertion_share_budget():
    retries = SeedRetries(7, Unstable)
    # Ordinary unstable expert scenes did not consume a budget historically.
    for _ in range(200):
        assert retries.retry(Unstable(), 'expert')
    assert retries.retry(AssertionError('target_pose cannot be None for move action.'), 'expert')
    for _ in range(55):
        assert retries.retry(Unstable(), 'policy_setup')
    with pytest.raises(RuntimeError, match='Policy setup'):
        retries.retry(Unstable(), 'policy_setup')
