#!/usr/bin/env python3
"""Check installed training dependencies and video decoding without loading weights."""
import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write a JSON verification receipt")
    args = parser.parse_args()
    # This check is intentionally CPU-only, including on shared training hosts.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    report = {"python": sys.version, "prefix": sys.prefix, "base_prefix": sys.base_prefix,
              "gpu_tested": False, "checkpoint_loading_tested": False, "checks": {}}

    def check(name, fn):
        try:
            report["checks"][name] = {"passed": True, "details": fn()}
        except Exception as exc:
            report["checks"][name] = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}

    def ffmpeg():
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg executable is missing")
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=True)
        return result.stdout.splitlines()[0]

    check("ffmpeg", ffmpeg)
    for module in ("torch", "torchvision", "torchcodec.decoders", "accelerate", "deepspeed",
                   "transformers", "hydra", "humanwam.runtime", "humanwam.trainer",
                   "humanwam.models.wan22.fastwam_idm", "humanwam.datasets.lerobot.robot_video_dataset"):
        check("import:" + module, lambda name=module: str(importlib.import_module(name).__file__))

    def video_roundtrip():
        import av
        import numpy as np
        import torch
        from torchcodec.decoders import VideoDecoder

        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(prefix="humanwam-video-check-") as directory:
            path = Path(directory) / "sample.mp4"
            with av.open(str(path), "w") as container:
                stream = container.add_stream("mpeg4", rate=10)
                stream.width = stream.height = 32
                stream.pix_fmt = "yuv420p"
                stream.thread_count = 1
                for level in (0, 80, 160):
                    frame = av.VideoFrame.from_ndarray(np.full((32, 32, 3), level, dtype=np.uint8), format="rgb24")
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            decoder = VideoDecoder(str(path), device="cpu", num_ffmpeg_threads=1)
            frames = decoder.get_frames_at(indices=[0, 1, 2]).data
            assert tuple(frames.shape) == (3, 3, 32, 32), tuple(frames.shape)
            means = frames.float().mean(dim=(1, 2, 3))
            assert torch.allclose(means, torch.tensor([0., 80., 160.]), atol=5), means
            return {"shape": list(frames.shape), "frame_means": means.tolist()}

    check("pyav_encode_torchcodec_decode", video_roundtrip)
    report["versions"] = {}
    for package in ("torch", "torchvision", "torchcodec", "av", "accelerate", "deepspeed",
                    "transformers", "huggingface-hub", "datasets", "numpy"):
        try:
            report["versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report["versions"][package] = None
    report["passed"] = all(item["passed"] for item in report["checks"].values())
    encoded = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    print(encoded)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
