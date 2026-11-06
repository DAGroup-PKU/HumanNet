# RoboTwin archived evaluation protocol

Start with [checkpoint download and the demo](evaluation.md). Download
the matching config/statistics bundle before attempting this optional benchmark;
the public RoboTwin release includes all three files.

The archived evaluation uses the **21k post-training checkpoint**, unseen instructions, and
50 tasks × 50 expert-valid episodes for each of Clean and Random. Each task uses
eight shards: `--seed 0` and `1` run 7 episodes each; seeds `2`–`7` run 6 each.
Scene seeds start at `100000 × (1 + shard)`, skipping failed expert checks.
The policy predicts 32 actions and executes 24 before observing again, with
10 video denoising steps followed by 10 action denoising steps.

For optional full-benchmark reproduction, run the complete task list from
`humanwam_vanilla/` using the same checkpoint and statistics in every shard.
Run shards sequentially per GPU because online expert planning has a wall-clock
timeout. For Blackwell, use the [OIDN library setting](environment.md#blackwell-rendering)
when calling the evaluator directly.

```bash
for phase in clean random; do
  for shard in {0..7}; do
    episodes=6
    if (( shard < 2 )); then episodes=7; fi
    CUDA_VISIBLE_DEVICES=0 ROBOTWIN_RT_DENOISER=oidn \
      ../.venv/sim/bin/python scripts/eval_robotwin.py \
      --robotwin-root ../RoboTwin --policy-python ../.venv/train/bin/python \
      --checkpoint /path/to/step_021000.pt --stats /path/to/dataset_stats.json \
      --task-file evaluation/robotwin/tasks-50.txt \
      --phase "$phase" --seed "$shard" --episodes "$episodes" \
      --output "../runs/robotwin-$phase-shard$shard" || exit 1
  done
done

python scripts/summarize_robotwin.py ../runs/robotwin-*-shard* \
  --output ../runs/robotwin-summary.json
```

The summary checks all 800 task/phase/shard cells. It rejects duplicate cells and
mixed checkpoints/settings, excludes truncated diagnostics, and reports missing
coverage instead of presenting a partial benchmark percentage.

Local checks verified exact agreement with the frozen evaluation entry for
three-camera preprocessing, state/action transformations, and the complete model's
10-step inference on identical weights and inputs. Native VAE/text-encoder tensors
and tokenizer files also match the archived evaluation bundle. Simulator assets
match too, apart from relocated robot file paths. The fresh environments have
also completed a real optimizer step and full learned-policy episodes.
Local validation covers demo execution; it does not establish full-benchmark
success rates. The published scores remain archived results, not measurements from this demo.

The original evaluation used Python 3.11, PyTorch 2.4.1/CUDA 11.8, and OIDN 2.0.1.
The Blackwell-compatible setup above uses newer GPU libraries. Historical language
generation also left Python's random state unseeded, so scene seeds alone cannot
recover the original instructions. Save the per-episode instructions and selected
scene seeds when comparing runs.
