# Environment compatibility and troubleshooting

For installation and resource downloads, follow the
[main README](../../README.md#vanilla-environment-setup).
Run the commands below from `humanwam_vanilla/`.

RoboTwin uses SAPIEN. RoboDojo requires a separate Isaac Sim environment.

## System prerequisites

| Component | Training / policy | RoboTwin simulator |
| --- | --- | --- |
| OS | Linux; validated on Ubuntu 22.04 x86_64 | Same |
| Python | 3.10; validated with 3.10.12 | Same, in a separate environment |
| Package installer | `uv` | `uv` |
| GPU runtime | NVIDIA GPU and driver supporting the installed CUDA 12.8 PyTorch build | Same; NVIDIA Vulkan support is also required for rendering |
| Video libraries | FFmpeg executable **and shared libraries**; validated with FFmpeg 4.4.2 | FFmpeg |
| Build tools | A C++ compiler / development tools for native dependencies | Git, C++ compiler, and CUDA toolkit providing `nvcc` |

## Blackwell rendering

The locally tested SAPIEN 3.0.0b1 installation bundled OIDN 2.0.1, whose CUDA
device initialization failed on the validation GPU. Install the tested OIDN 2.3.3
compatibility library once:

```bash
mkdir -p ../.runtime/oidn
curl -fL https://github.com/RenderKit/oidn/releases/download/v2.3.3/oidn-2.3.3.x86_64.linux.tar.gz \
  -o ../.runtime/oidn/oidn.tar.gz
tar -xzf ../.runtime/oidn/oidn.tar.gz -C ../.runtime/oidn
export LD_PRELOAD="$(realpath ../.runtime/oidn/oidn-2.3.3.x86_64.linux/lib/libOpenImageDenoise.so.2.3.3)${LD_PRELOAD:+:$LD_PRELOAD}"
```

OIDN 2.3.3 differs from the paper's OIDN 2.0.1 renderer and does not establish
numerical parity with the archived evaluation.

## Troubleshooting

| Symptom | Next step |
| --- | --- |
| Environment already exists | Use a fresh target path; setup deliberately preserves existing installations |
| TorchCodec cannot load FFmpeg | Check both `ffmpeg -version` and availability of FFmpeg shared libraries in the dynamic linker paths |
| `nvcc` missing / CuRobo build fails | Make the CUDA toolkit and C++ compiler available; inspect `TORCH_CUDA_ARCH_LIST`; keep `MAX_JOBS=1` when memory is constrained |
| No visible GPU during architecture detection | Set `CUDA_VISIBLE_DEVICES` correctly or specify the target `TORCH_CUDA_ARCH_LIST` for an offline build |
| Vulkan or camera initialization fails | Verify the host NVIDIA Vulkan driver and container graphics access; run the standalone render check |
| OIDN device initialization fails on Blackwell | Follow the Blackwell rendering commands above and set `LD_PRELOAD` before evaluation |
| Missing robot meshes / invalid robot paths | Check the pinned asset extraction layout and rerun path relocation on your own asset copy |
| Policy startup fails | Inspect `policy.log`; check checkpoint/statistics pairing, Wan files, offline flags, and available GPU memory |
| Output directory already exists | Choose a new `--output` path; existing rollout records are preserved |
