# RoboTwin evaluation protocol

Use the [main README evaluation commands](../../README.md#vanilla-evaluation)
to download the checkpoint bundle and run the benchmark.

The archived evaluation uses the **21k post-training checkpoint**, unseen instructions, and
50 tasks × 50 expert-valid episodes for each of Clean and Random. Each task uses
eight shards: `--seed 0` and `1` run 7 episodes each; seeds `2`–`7` run 6 each.
Scene seeds start at `100000 × (1 + shard)`, skipping failed expert checks.
The policy predicts 32 actions and executes 24 before observing again, with
10 video denoising steps followed by 10 action denoising steps.

The evaluation wrapper runs shards sequentially per GPU because online expert
planning has a wall-clock timeout. On Blackwell, follow the
[OIDN compatibility setup](environment.md#blackwell-rendering).

The summary expects all 800 task/phase/shard cells and reports missing coverage.
The scores in the README are archived results, not a new benchmark measurement.

The original evaluation used Python 3.11, PyTorch 2.4.1/CUDA 11.8, and OIDN 2.0.1.
The Blackwell-compatible setup above uses newer GPU libraries. Historical language
generation also left Python's random state unseeded, so scene seeds alone cannot
recover the original instructions. Save the per-episode instructions and selected
scene seeds when comparing runs.
