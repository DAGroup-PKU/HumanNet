#!/usr/bin/env python3
"""Check the inference environment: versions, FlashAttention-3, GPU and the pinned cuBLASLt."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

CUBLASLT_SHA256 = "4ef47ce60325811dfce57acbc2ced3000dbb6c50ce65f4ec34340b1897d6911a"
EXPECTED = {"torch": "2.10.0", "transformers": "5.8.1"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cublaslt", type=Path, help="Pinned libcublasLt.so.12 to verify")
    parser.add_argument("--no-gpu", action="store_true", help="Skip the GPU check")
    args = parser.parse_args()
    import torch
    import transformers

    errors = []
    report = dict(
        python=sys.version.split()[0],
        torch=torch.__version__,
        cuda=torch.version.cuda,
        transformers=transformers.__version__,
    )
    if sys.version_info[:2] != (3, 13):
        errors.append("Python 3.13 is required")
    for name, module in (("torch", torch), ("transformers", transformers)):
        if module.__version__.split("+")[0] != EXPECTED[name]:
            errors.append(f"{name}=={EXPECTED[name]} is required")
    try:
        import flash_attn_3_nv._C  # noqa: F401  # needs GLIBC >= 2.34
    except ImportError as error:
        errors.append(f"FlashAttention-3 does not load: {error}")
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
                errors.append("The native backend is validated on Hopper (SM90) GPUs")
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
