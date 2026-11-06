"""Seed-failure classification used by the frozen RoboTwin paper evaluator."""


class SeedRetries:
    def __init__(self, episodes, unstable_error):
        self.unstable_error = unstable_error
        self.overlay_count = 0
        self.setup_count = 0
        self.overlay_limit = max(64, episodes * 16)
        self.setup_limit = max(32, episodes * 8)

    def retry(self, error, stage):
        """Return whether a seed can be skipped; unexpected errors remain fatal.

        The historical overlay shares its budget across expert play and both
        setup calls. Policy setup and the specific move assertion share a
        second budget. Ordinary expert failures do not consume either budget.
        """
        import numpy as np

        if stage not in ('expert', 'policy_setup'):
            raise ValueError(f'Unknown retry stage: {stage}')
        converted = isinstance(error, (TypeError, IndexError, np.linalg.LinAlgError))
        if converted:
            self.overlay_count += 1
            if self.overlay_count > self.overlay_limit:
                raise RuntimeError('Expert overlay retry budget exhausted') from error
        unstable = converted or isinstance(error, self.unstable_error)
        move_assertion = (stage == 'expert' and isinstance(error, AssertionError)
                          and str(error) == 'target_pose cannot be None for move action.')
        if (stage == 'policy_setup' and unstable) or move_assertion:
            self.setup_count += 1
            if self.setup_count > self.setup_limit:
                raise RuntimeError('Policy setup / move assertion retry budget exhausted') from error
        return unstable or move_assertion
