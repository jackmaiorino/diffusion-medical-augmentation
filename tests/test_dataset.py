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


import json
import os

import pytest
import torch

import dataset
from dataset import HAM10000
from make_cache import build_cache

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELED = os.path.join(REPO_ROOT, 'data', 'labeled')


@pytest.fixture
def mini_split(tmp_path):
    """An eight-row splits.csv with train rows at non-contiguous positions."""
    import pandas

    rows = pandas.read_csv(os.path.join(REPO_ROOT, 'data', 'splits.csv'))
    # Two classes, two images each, so a row/label swap is detectable.
    train = pandas.concat([
        rows[rows['dx'] == 'df'].head(2),
        rows[rows['dx'] == 'vasc'].head(2),
    ]).reset_index(drop=True)
    train['split'] = 'train'

    # Filler rows placed between the train rows so the surviving train rows
    # land at positions 0, 2, 4, 6 in the full file rather than 0, 1, 2, 3.
    # Otherwise the preserved-row index and the plain post-filter positional
    # index would agree by coincidence, and this fixture could not catch a
    # regression to positional indexing in __getitem__.
    filler = rows[~rows['image_id'].isin(train['image_id'])].head(4)
    filler = filler.reset_index(drop=True)
    filler['split'] = 'val'

    picked = pandas.concat([
        train.iloc[[0]], filler.iloc[[0]],
        train.iloc[[1]], filler.iloc[[1]],
        train.iloc[[2]], filler.iloc[[2]],
        train.iloc[[3]], filler.iloc[[3]],
    ]).reset_index(drop=True)

    path = tmp_path / 'splits.csv'
    picked.to_csv(path, index=False)
    return str(path)


def test_cache_has_one_row_per_split_row(mini_split, tmp_path):
    npy = build_cache(mini_split, LABELED, str(tmp_path), 64)
    array = np.load(npy)

    assert array.shape == (8, 64, 64, 3)
    assert array.dtype == np.uint8


def test_cached_and_uncached_tensors_are_identical(mini_split, tmp_path):
    # The alignment guard. If cache rows and CSV rows ever desync, every
    # image gets the wrong label and nothing else would catch it.
    build_cache(mini_split, LABELED, str(tmp_path), 64)

    cached = HAM10000('train', image_size=64, splits_csv=mini_split,
                      labeled_dir=LABELED, cache=True, cache_dir=str(tmp_path))
    raw = HAM10000('train', image_size=64, splits_csv=mini_split,
                   labeled_dir=LABELED, cache=False, cache_dir=str(tmp_path))

    assert len(cached) == len(raw) == 4
    for i in range(len(raw)):
        cached_image, cached_label = cached[i]
        raw_image, raw_label = raw[i]
        assert cached_label == raw_label
        assert torch.equal(cached_image, raw_image)


def test_stale_cache_raises(mini_split, tmp_path):
    build_cache(mini_split, LABELED, str(tmp_path), 64)

    _, sidecar = dataset.cache_paths(str(tmp_path), 64)
    meta = json.load(open(sidecar))
    meta['splits_sha1'] = '0' * 40
    json.dump(meta, open(sidecar, 'w'))

    with pytest.raises(RuntimeError, match='make_cache'):
        HAM10000('train', image_size=64, splits_csv=mini_split,
                 labeled_dir=LABELED, cache=True, cache_dir=str(tmp_path))


def test_missing_cache_falls_back_instead_of_raising(mini_split, tmp_path):
    # No cache built. A missing cache is a speed problem, not a correctness
    # one, so it must degrade to decoding rather than fail. This is what
    # keeps vae_roundtrip.py working at 256 with no 256px cache.
    ds = HAM10000('train', image_size=64, splits_csv=mini_split,
                  labeled_dir=LABELED, cache=True, cache_dir=str(tmp_path))

    assert len(ds) == 4
    assert ds[0][0].shape == (3, 64, 64)
