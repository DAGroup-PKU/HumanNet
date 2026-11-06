# HumanWAM-Vanilla post-training and evaluation

The recipes below use **50kh pre40k → W0 mid10k** initialization. RoboDojo refers
to the **OpenWAM-style augmentation** experiment, selecting **post50k from a
60k-step run**, not the later task-augmentation experiment.

## Published checkpoints

| Repository | Files | Status |
| --- | --- | --- |
| [Base](https://huggingface.co/DAGroup-PKU/HumanWAM-Vanilla-Base) | `pretrained/step_040000.pt`, `midtrained/step_010000.pt` | Weights available |
| [RoboTwin](https://huggingface.co/DAGroup-PKU/HumanWAM-Vanilla-RoboTwin) | `model.pt` (post21k) | Weights available |
| [RoboDojo](https://huggingface.co/DAGroup-PKU/HumanWAM-Vanilla-RoboDojo) | README only | No downloadable weights |

The RoboTwin repository provides its matching evaluation configuration and
normalization statistics. Training resource preparation and the RoboDojo bundle
are still pending. A weight download
alone is insufficient to run the commands. Never substitute statistics from a
different action convention or from the old 3,465-episode RoboDojo split.

## Training contracts

| Setting | RoboTwin | RoboDojo |
| --- | --- | --- |
| Config | [`robotwin.yaml`](../humanwam_vanilla/configs/robotwin.yaml) | [`robodojo.yaml`](../humanwam_vanilla/configs/robodojo.yaml) |
| Path template | [`paths.example.yaml`](../humanwam_vanilla/configs/paths.example.yaml) | [`paths.robodojo.example.yaml`](../humanwam_vanilla/configs/paths.robodojo.example.yaml) |
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
[`RoboDojoPolicy`](../humanwam_vanilla/src/humanwam/robodojo_policy.py) performs
RGB preprocessing and returns world-frame EEF16 actions using the same configured
transforms; it requires the Wan text encoder for live instructions.

## Evaluation

Begin with [checkpoint download and evaluation setup](evaluation.md), including
the release status of the matching configuration and statistics.

RoboTwin uses the isolated SAPIEN environment built by `setup_environment.sh sim`.
See the [README evaluation](../README.md#vanilla-evaluation) and
[archived evaluation protocol](robotwin_archived_evaluation.md). The archive uses
50 episodes/task/condition; the manuscript's 100-episode description remains a
known discrepancy. The archived scores 92.08 / 68.88 / 80.48 are not a new local
benchmark measurement.

RoboDojo uses a separate Isaac Sim/IsaacLab process and a policy process. Its
RGB camera order is head, left wrist, right wrist. Use the RoboDojo adapter and
full35 statistics, not the RoboTwin native-wrist adapter. Use [`serve_robodojo.py`](../humanwam_vanilla/scripts/serve_robodojo.py) for
the policy process and the simulator’s `src/eval_client/main.py` for one episode.
The policy process uses the WebSocket implementation in the supplied
`RoboDojo/XPolicyLab/client_server/ws` checkout, with `websockets==15.0.1` and
`msgpack-numpy==0.4.8`, `msgpack==1.2.4`, and `pydantic==2.13.5` on the policy side.
This is a single-environment adapter, predicting 32 actions
and executing up to 24 before observing again. A portable Isaac installation and
asset preparation script is not yet included. The RoboDojo model card remains
README-only; its manuscript scores are not attributed to the selected post50k
checkpoint without a matching evaluation record.

## Validation status

[Machine-readable validation record](validation/post_training_20261106.json).

Local validation on 2026-11-06 reuses the training and RoboTwin environments
installed from scratch on 2026-11-05; it is **not another fresh installation**.
Training uses Python 3.10 and PyTorch 2.7.1/CUDA 12.8.
The separate RoboDojo simulator environment is an existing installation; its
relocated editable source paths required an isolated source overlay. No claim of
a fresh RoboDojo simulator installation is made. The original RoboDojo run
used a different compiled training runtime, so these checks establish functional
execution, not numerical identity of a complete retraining run.

| Check | Result |
| --- | --- |
| W0 mid10k → RoboTwin post-training | One real optimizer step; finite loss; cache hit, no cache miss |
| Published RoboTwin post21k → simulator | `beat_block_hammer`, Clean, scene 100002, unseen instruction; success after 110 actions / 5 chunks; not truncated |
| RoboDojo full35 loader + augmentation | 3,500 episodes, 1,859,602 frames; complete cache; strict normalization provenance; uint8 `[3,9,384,352]` video |
| W0 mid10k → RoboDojo post-training | One real optimizer step; loss 0.2842 (action 0.1685, video 0.1157); no cache miss |
| RoboDojo post50k offline inference | Full parameter names/shapes match; checkpoint step 50,000; 10 inference steps; finite `[32,16]` action output; peak allocated GPU memory 13.81 GB |
| RoboDojo Isaac Sim 5.1 / IsaacLab 0.54.3 | Existing Python 3.11 / PyTorch 2.7.0 CUDA 12.8 environment: headless startup/close, 8 physics steps and camera render all pass |
| RoboDojo Isaac closed-loop rollout | `stack_blocks`, `arx_x5`, seed 0, layout 0: complete success; score 1.0; 11 policy chunks in the initial check; successful rerun through the published `serve_robodojo.py` entry; three camera videos saved |

The optimizer checks deliberately use batch 1, zero data-loader workers,
compilation disabled, and zero warmup to exercise a nonzero learning-rate update.
They stop after one step without saving/replacing checkpoints. These diagnostics
retain the intended full corpus and do not establish training convergence.


The local Isaac check uses RoboDojo source based on commit
`25691aa78fb34bbbe798aa3d880d57ef2788f696`, XPolicyLab's WebSocket transport,
and CuRobo source at `d17b54ce32cba095c0b000c4c58777075d11de0e`. The simulator
source includes local compatibility changes; a clean upstream checkout alone is
not claimed equivalent. Relocated editable imports and the X5 CuRobo
`asset_root_path`/`urdf_path` were fixed only in an isolated validation overlay.
The original simulator checkout, robot assets, run configurations and checkpoints
were preserved. These dependencies and fixes need to be packaged before a
from-scratch RoboDojo installation can be advertised.

A single successful RoboDojo episode is a pipeline check, not evidence for the
manuscript's aggregate success/progress scores. The full benchmark was not run.
