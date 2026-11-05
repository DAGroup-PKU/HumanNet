"""Local RGB observation files and portable action outputs."""

import json
from pathlib import Path

import numpy as np
from PIL import Image


def read_observation(path):
    path = Path(path)
    sample = json.loads(path.read_text())

    def images(views):
        result = {}
        for camera, filename in views.items():
            with Image.open(path.parent / filename) as source:
                result[camera] = np.array(source.convert("RGB"), copy=True)
        return result

    sample["images"] = images(sample["images"])
    sample["goal_images"] = images(sample["goal_images"])
    sample["history"] = [
        dict(capture, images=images(capture["images"])) for capture in sample.get("history", [])
    ]
    return sample


def save_actions(path, result):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as target:
        np.savez(target, **result)
