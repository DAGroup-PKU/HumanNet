"""Strict, streaming safetensors loading."""

import json
from pathlib import Path

from safetensors import safe_open


def load_weights(module, directory, remap=lambda name: name, *, optional=()):
    directory = Path(directory)
    index = directory / "model.safetensors.index.json"
    if index.is_file():
        files = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    else:
        files = ["model.safetensors"]
    expected = set(module.state_dict())
    loaded = set()
    for filename in files:
        path = (directory / filename).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError("Weight shards must be inside the checkpoint directory")
        with safe_open(path, framework="pt", device="cpu") as handle:
            values = {}
            for source in handle.keys():
                key = remap(source)
                if key is None:
                    continue
                if key not in expected or key in loaded:
                    raise ValueError(f"Unexpected or duplicate model tensor: {source}")
                values[key] = handle.get_tensor(source)
                loaded.add(key)
            module.load_state_dict(values, strict=False, assign=True)
    missing = expected - loaded - set(optional)
    if missing:
        raise ValueError(f"Missing model tensors: {sorted(missing)}")
