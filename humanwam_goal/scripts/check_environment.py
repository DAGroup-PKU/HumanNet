#!/usr/bin/env python3
"""Check the selected inference backend's dependencies and CUDA environment."""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import sys
from pathlib import Path

CUBLASLT_SHA256 = "4ef47ce60325811dfce57acbc2ced3000dbb6c50ce65f4ec34340b1897d6911a"
EXPECTED = {
    "native": {
        "torch": "2.10.0",
        "transformers": "5.8.1",
        "Pillow": "13.0.0",
        "flash-attn-3-nv": "1.0.3+cu128.torch210",
    },
    "phyai": {
        "torch": "2.11.0",
        "torchvision": "0.26.0",
        "transformers": "5.8.1",
        "flash-attn-3-nv": "1.0.3+cu128.torch210",
        "phyai": "0.2.0rc3",
        "phyai-kernel": "0.2.0rc3",
        "phyai-utils-tools": "0.2.0rc3",
        "phyai-ext": "0.2.0rc3",
        "phyai-gateway": "0.2.0rc3",
    },
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("native", "phyai"), default="native")
    parser.add_argument("--cublaslt", type=Path, help="Pinned libcublasLt.so.12 to verify")
    parser.add_argument("--no-gpu", action="store_true", help="Skip the GPU check")
    args = parser.parse_args()
    if args.backend == "phyai" and args.cublaslt is not None:
        parser.error("--cublaslt is only used by the native reproduction environment")
    import torch
    import transformers

    errors = []
    report = dict(
        backend=args.backend,
        python=sys.version.split()[0],
        torch=torch.__version__,
        cuda=torch.version.cuda,
        transformers=transformers.__version__,
    )
    if sys.version_info[:2] != (3, 13):
        errors.append("Python 3.13 is required")
    report["packages"] = {}
    for name, expected in EXPECTED[args.backend].items():
        try:
            version = importlib.metadata.version(name)
            report["packages"][name] = version
            actual = version.split("+")[0] if name in ("torch", "torchvision") else version
            if actual != expected:
                errors.append(f"{name}=={expected} is required, found {version}")
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"{name}=={expected} is not installed")
    modules = (
        ("flash_attn_3_nv._C",)
        if args.backend == "native"
        else (
            "flash_attn_3_nv._C",
            "phyai.models.humanwam_goal.main_humanwam_goal",
            "torchvision",
        )
    )
    for name in modules:
        try:
            importlib.import_module(name)
        except (ImportError, OSError, RuntimeError) as error:
            errors.append(f"{name} does not load: {error}")
    if torch.version.cuda != "12.8":
        errors.append("Use the CUDA 12.8 PyTorch build")
    import goalwam
    import goalwam.robotwin  # noqa: F401

    report["goalwam"] = goalwam.__version__
    if not args.no_gpu:
        if not torch.cuda.is_available():
            errors.append("No CUDA GPU is visible")
        else:
            report["gpu"] = torch.cuda.get_device_name(0)
            report["capability"] = list(torch.cuda.get_device_capability(0))
            if tuple(report["capability"]) != (9, 0):
                errors.append("HumanWAM-Goal is validated on Hopper (SM90) GPUs")
    if args.cublaslt is not None:
        digest = hashlib.sha256(args.cublaslt.read_bytes()).hexdigest()
        report["cublaslt_sha256"] = digest
        if digest != CUBLASLT_SHA256:
            errors.append("The cuBLASLt library is not version 12.8.3.14")
    report["errors"] = errors
    print(json.dumps(report, indent=2))
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
