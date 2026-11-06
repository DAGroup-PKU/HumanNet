import numpy as np
import pytest
import torch

from humanwam.robotwin_policy import prepare_images


def test_camera_order_survives_resize_and_codec():
    cameras = {}
    for index, key in enumerate(('head_camera', 'left_camera', 'right_camera')):
        rgb = np.zeros((240, 320, 3), dtype=np.uint8)
        rgb[..., index] = 255
        cameras[key] = rgb
    image = prepare_images(cameras, device='cpu', dtype=torch.bfloat16).float()[0]
    assert image.shape == (3, 384, 352)
    assert torch.isfinite(image).all()
    for index, (row, col) in enumerate(((100, 100), (320, 80), (320, 270))):
        expected = torch.full((3,), -1.)
        expected[index] = 1.
        torch.testing.assert_close(image[:, row, col], expected, atol=0.03, rtol=0)


def test_camera_float_range_is_not_silently_interpreted_as_bytes():
    cameras = {key: np.zeros((240, 320, 3), dtype=np.float32)
               for key in ('head_camera', 'left_camera', 'right_camera')}
    with pytest.raises(ValueError, match='uint8'):
        prepare_images(cameras, device='cpu', dtype=torch.bfloat16)
