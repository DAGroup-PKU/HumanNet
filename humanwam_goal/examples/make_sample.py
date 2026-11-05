"""Create synthetic camera inputs for an installation smoke test."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("examples/sample"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    views = {}
    for index, name in enumerate(("head", "left", "right")):
        y, x = np.indices((192, 320))
        pixels = np.stack(
            ((x + index * 40) % 256, (y * 2) % 256, ((x + y) // 2) % 256), axis=-1
        ).astype(np.uint8)
        Image.fromarray(pixels).save(args.output / f"{name}.png")
        views[name] = f"{name}.png"
    sample = {
        "instruction": "Move the object to the position shown in the goal image.",
        "images": views,
        "goal_images": {"head": "head.png"},
        "observation_timestamp": 0.0,
        "history": [],
        "normalized_state": [0.0] * 49,
        "domain_id": -1,
        "state_valid_mask": [False] * 23 + [True] * 19 + [False] * 7,
        "action_valid_mask": [False] * 23 + [True] * 19 + [False] * 7,
        "seed": 0,
    }
    target = args.output / "observation.json"
    target.write_text(json.dumps(sample, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
