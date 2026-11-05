"""Verify the cuBLASLt library mapped into this process.

The released RoboTwin results were served with cuBLASLt 12.8.3.14, preloaded
with ``LD_PRELOAD``; PyTorch 2.10 ships 12.8.4.1, which selects different BF16
GEMM kernels. A missing preload is otherwise silent, so check the mapped file.
"""

import ctypes
import hashlib
from pathlib import Path

CUBLASLT_12_8_3_14_SHA256 = "4ef47ce60325811dfce57acbc2ced3000dbb6c50ce65f4ec34340b1897d6911a"


def loaded_cublaslt(maps="/proc/self/maps"):
    paths = set()
    for line in Path(maps).read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and Path(fields[5]).name.startswith("libcublasLt.so"):
            paths.add(Path(fields[5]).resolve())
    return paths


def cublaslt_runtime(expected_sha256=CUBLASLT_12_8_3_14_SHA256, maps="/proc/self/maps"):
    """Return the one mapped cuBLASLt; fail on none, several, or a different build."""
    paths = loaded_cublaslt(maps)
    if len(paths) != 1:
        raise RuntimeError(
            f"Expected exactly one loaded cuBLASLt library, found: {sorted(map(str, paths))}"
        )
    path = paths.pop()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if expected_sha256 is not None and actual != expected_sha256:
        raise RuntimeError(
            f"cuBLASLt SHA-256 mismatch: {path}: expected {expected_sha256}, got {actual}"
        )
    library = ctypes.CDLL(str(path))
    library.cublasLtGetVersion.restype = ctypes.c_size_t
    return dict(path=str(path), version=library.cublasLtGetVersion(), sha256=actual)
