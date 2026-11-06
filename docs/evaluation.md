# HumanWAM-Vanilla: download and evaluate

## 1. Download the evaluation checkpoint

For the RoboTwin demo, download the **post-trained model**. The Base pretrain and
mid-train checkpoints are not required for evaluation.
From `humanwam_vanilla/`, after the [environment setup](../README.md#vanilla-environment-setup)
has installed `.venv/train`:

```bash
../.venv/train/bin/huggingface-cli download DAGroup-PKU/HumanWAM-Vanilla-RoboTwin \
  model.pt config.yaml dataset_stats.json \
  --local-dir ../checkpoints/vanilla-robotwin
```

The release includes the matching evaluation configuration and normalization
statistics. Keep these files together with `model.pt`.

RoboDojo remains model-card-only; there is no public post50k
download command yet. Exploration downloads and their per-checkpoint config/stats
bindings also remain pending.

## 2. Prepare the matching configuration and resources

The released RoboTwin policy uses W0 three-view conditioning, native wrists
(unalign), and absolute camera-frame EEF16 actions. Use the configuration and
statistics downloaded with the checkpoint.

Follow [environment setup](../README.md#vanilla-environment-setup) for the separate
policy and simulator environments, Wan VAE/text encoder/tokenizer, and RoboTwin
assets. On Blackwell, apply the [OIDN rendering setting](environment.md#blackwell-rendering).
No training corpus, training video/text cache, or optimizer state is needed for
this simulator demo.

## 3. Run one complete RoboTwin episode

From `humanwam_vanilla/`:

```bash
../.venv/sim/bin/python scripts/eval_robotwin.py \
  --robotwin-root ../RoboTwin --policy-python ../.venv/train/bin/python \
  --config ../checkpoints/vanilla-robotwin/config.yaml \
  --checkpoint ../checkpoints/vanilla-robotwin/model.pt \
  --stats ../checkpoints/vanilla-robotwin/dataset_stats.json \
  --tasks beat_block_hammer --phase clean --seed 0 --episodes 1 \
  --output ../runs/robotwin-demo
```

Use a new output directory for each run. Inspect `result.json` and `policy.log`
there for completion, success, the actual scene seed and instruction.
One successful episode verifies the execution chain, not an aggregate benchmark score.

The current adapter is validated for the W0 packed three-view model, and the CLI
supports Clean and Random. It is **not a general Exploration adapter** and does
not implement Scene-OOD. Those releases need their own verified adapter/config
bindings before evaluation commands can be provided.

## Further evaluation

- [Archived RoboTwin protocol](robotwin_archived_evaluation.md): optional full
  benchmark, 50 episodes/task/condition; differs from the manuscript's 100-episode description.
- [RoboDojo evaluation status](../README.md#vanilla-evaluation): public weights and evaluation commands are pending.
- [Post-training and validation record](post_training.md): exact tested scope and
  the remaining resource/environment packaging work.
