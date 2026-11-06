# HumanWAM-Vanilla post-training and evaluation

The recipes below use **50kh pre40k → W0 mid10k** initialization. RoboDojo refers
to the **OpenWAM-style augmentation** experiment, selecting **post50k from a
60k-step run**, not the later task-augmentation experiment.

## Training contracts

| Setting | RoboTwin | RoboDojo |
| --- | --- | --- |
| Config | [`robotwin.yaml`](../configs/robotwin.yaml) | [`robodojo.yaml`](../configs/robodojo.yaml) |
| Path template | [`paths.example.yaml`](../configs/paths.example.yaml) | [`paths.robodojo.example.yaml`](../configs/paths.robodojo.example.yaml) |
| Corpus | 50 tasks × 50 demonstrations = 2,500 | Full35: 35 tasks × 100 demonstrations = 3,500 |
| Holdout | None (`val_set_proportion: 0.0`) | None (`val_set_proportion: 0.0`) |
| Initialization | W0 mid10k weights file | W0 mid10k weights file |
| Schedule / selected step | 24k / 21k | 60k / 50k |
| Wrist definition | Native wrists, unalign | Left `diag(1,-1,-1)`, right identity |
| Action representation | Absolute camera-frame EEF16 | Absolute camera-frame link6 EEF16, quaternion wxyz, gripper 1=open |
| Images | Historical T-canvas MP4 codec path | FP32 resize → compact T canvas → uint8, no MP4 round trip |
| Augmentation | As resolved in the RoboTwin config | PIL brightness/contrast/saturation ±0.2, probability 1, shared across all frames/views; no hue |

The RoboDojo camera transform uses `R_x(30°)` and translation `[0, -0.41, 1.308]`
for camera-to-world. Inference must invert the normalization, wrist transform and
camera transform before executing world-frame poses. Quaternions are standardized
through rotation matrices. Do not apply relative-pose conversion or gripper flipping.

OpenWAM-style denotes the augmentation contract; it does not claim equivalence to
OpenWAM's entire data loader or training stack. Its reference is commit
`3dfc9a7d9230f0e1382e6b22639278f1a86e5eb5`.

The selected RoboDojo training run used batch 64 × 8 GPUs = 512. Changing the
number of GPUs requires matching the effective batch through accumulation if
reproducing that setting. Retain `max_steps: 60000` and `warmup_ratio: 0.05` when
selecting post50k; changing the schedule to 50k changes the learning rate history.

`bash scripts/train.sh --config CONFIG --init-checkpoint WEIGHTS.pt` initializes
a new run with a fresh optimizer and step counter. Use `--resume STATE_DIRECTORY`
to continue an existing run. The two flags are mutually exclusive. The optional
`--paths` file and legacy config `resume` binding remain supported. Local
`configs/local/*.yaml` files can combine the recipe and resource bindings so
normal launches need only a config and the previous-stage weights. Do not
change a resumed run's corpus, sampler, or tokenizer settings.

Both recipes validate normalization metadata against roots, episode/frame counts,
transforms and manifests. The public package name `humanwam` is accepted as an
alias for the historical `fastwam` namespace; numeric transform parameters remain
strictly checked. Relocated training data require a provenance-preserving
preparation step; do not turn off validation to make old absolute paths pass.
RoboDojo also requires a complete compact cache covering all 3,500 episodes
and matching the source transforms.

## Offline checkpoint check

From `humanwam_vanilla/`, with the training environment activated and local
resources filled in:

```bash
python scripts/check_policy.py --config configs/robodojo.yaml \
  --paths configs/paths.robodojo.local.yaml \
  --output ../runs/robodojo-policy-check.json --steps 10 \
  paths.resume=/path/to/robodojo/step_050000.pt \
  model.mot_compile_blocks=false model.vae_compile_encoder=false
```

This loads the full checkpoint, checks parameter names/shapes, and predicts a
32-action chunk from a real dataset sample with its cached text conditioning.
It checks normalized model outputs, not robot success. The simulator-facing
[`RoboDojoPolicy`](../src/humanwam/robodojo_policy.py) performs
RGB preprocessing and returns world-frame EEF16 actions using the same configured
transforms; it requires the Wan text encoder for live instructions.

## Evaluation

Begin with the [checkpoint downloads](../../README.md#vanilla-weight-download)
and [evaluation commands](../../README.md#vanilla-evaluation) in the main README.

RoboTwin uses the isolated SAPIEN environment built by `setup_environment.sh sim`.
See the [README evaluation](../../README.md#vanilla-evaluation) and
[archived evaluation protocol](robotwin_evaluation.md). The archive uses
50 episodes/task/condition; the manuscript's 100-episode description remains a
known discrepancy. The archived scores 92.08 / 68.88 / 80.48 are not a new local
benchmark measurement.

RoboDojo uses a separate Isaac Sim/IsaacLab process and a policy process. Its
RGB camera order is head, left wrist, right wrist. Use the RoboDojo adapter and
full35 statistics, not the RoboTwin native-wrist adapter. Use [`serve_robodojo.py`](../scripts/serve_robodojo.py) for
the policy process and the simulator’s `src/eval_client/main.py` for one episode.
The policy process uses the WebSocket implementation in the supplied
`RoboDojo/XPolicyLab/client_server/ws` checkout, with `websockets==15.0.1` and
`msgpack-numpy==0.4.8`, `msgpack==1.2.4`, and `pydantic==2.13.5` on the policy side.
This is a single-environment adapter, predicting 32 actions
and executing up to 24 before observing again. A portable Isaac installation and
asset preparation script is not yet included. The RoboDojo model card remains
README-only; its manuscript scores are not attributed to the selected post50k
checkpoint without a matching evaluation record.
