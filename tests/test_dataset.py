"""Tests for the shared HAM10000 loader."""
import numpy as np
from PIL import Image

from dataset import SquareCenterCrop


def test_crop_reduces_to_the_short_side():
    img = Image.fromarray(np.zeros((450, 600, 3), np.uint8))
    assert SquareCenterCrop()(img).size == (450, 450)


def test_crop_keeps_the_center():
    # 600 wide: 75px red, 450px green, 75px blue. Only green should survive.
    strip = np.zeros((450, 600, 3), np.uint8)
    strip[:, :75] = (255, 0, 0)
    strip[:, 75:525] = (0, 255, 0)
    strip[:, 525:] = (0, 0, 255)

    out = np.array(SquareCenterCrop()(Image.fromarray(strip)))

    assert out.shape == (450, 450, 3)
    assert (out[:, :, 1] == 255).all()
    assert (out[:, :, 0] == 0).all()
    assert (out[:, :, 2] == 0).all()


def test_crop_is_a_noop_on_square_input():
    # The cached path feeds 64x64 images through the same pipeline, so the
    # geometric steps must leave an already-square image untouched.
    rng = np.random.default_rng(612)
    img = Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))

    assert np.array_equal(np.array(SquareCenterCrop()(img)), np.array(img))
