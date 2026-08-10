"""Tests for the evaluation reference exporter."""
import json
import os

import numpy as np
import pytest
from PIL import Image

import dataset
from export_real64 import export_real64
from make_cache import build_cache

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELED = os.path.join(REPO_ROOT, 'data', 'labeled')


@pytest.fixture
def tiny_setup(tmp_path):
    """A six-row splits.csv with mixed splits, plus its 64px cache."""
    import pandas

    rows = pandas.read_csv(os.path.join(REPO_ROOT, 'data', 'splits.csv'))
    train = pandas.concat([rows[rows['dx'] == 'df'].head(2),
                           rows[rows['dx'] == 'vasc'].head(2)])
    train = train.reset_index(drop=True)
    train['split'] = 'train'
    filler = rows[~rows['image_id'].isin(train['image_id'])].head(2)
    filler = filler.reset_index(drop=True)
    filler['split'] = 'val'

    # val rows interleaved so train cache positions are not 0..3
    picked = pandas.concat([train.iloc[[0]], filler.iloc[[0]],
                            train.iloc[[1]], filler.iloc[[1]],
                            train.iloc[[2]], train.iloc[[3]]])
    picked = picked.reset_index(drop=True)
    splits = tmp_path / 'splits.csv'
    picked.to_csv(splits, index=False)
    build_cache(str(splits), LABELED, str(tmp_path), 64)
    return str(splits), str(tmp_path)


def test_exports_one_png_per_train_row(tiny_setup, tmp_path):
    splits, cache_dir = tiny_setup
    out = str(tmp_path / 'real64')

    count = export_real64(splits, cache_dir, out)

    assert count == 4
    assert sorted(os.listdir(out)) == ['df', 'vasc']
    assert len(os.listdir(os.path.join(out, 'df'))) == 2
    assert len(os.listdir(os.path.join(out, 'vasc'))) == 2


def test_filenames_are_image_ids(tiny_setup, tmp_path):
    import pandas

    splits, cache_dir = tiny_setup
    out = str(tmp_path / 'real64')
    export_real64(splits, cache_dir, out)

    rows = pandas.read_csv(splits)
    train = rows[rows['split'] == 'train']
    for row in train.itertuples():
        assert os.path.exists(os.path.join(out, row.dx,
                                           row.image_id + '.png'))


def test_exported_pixels_equal_cache_rows(tiny_setup, tmp_path):
    import pandas

    splits, cache_dir = tiny_setup
    out = str(tmp_path / 'real64')
    export_real64(splits, cache_dir, out)

    npy_path, _ = dataset.cache_paths(cache_dir, 64)
    array = np.load(npy_path)
    rows = pandas.read_csv(splits)
    train = rows[rows['split'] == 'train']
    for row in train.itertuples():
        saved = np.array(Image.open(
            os.path.join(out, row.dx, row.image_id + '.png')))
        assert np.array_equal(saved, array[row.Index])


def test_missing_cache_raises(tiny_setup, tmp_path):
    splits, _ = tiny_setup
    empty = tmp_path / 'nocache'
    empty.mkdir()

    with pytest.raises(RuntimeError, match='make_cache'):
        export_real64(splits, str(empty), str(tmp_path / 'real64'))


def test_stale_sidecar_raises(tiny_setup, tmp_path):
    splits, cache_dir = tiny_setup
    _, sidecar = dataset.cache_paths(cache_dir, 64)
    meta = json.load(open(sidecar))
    meta['splits_sha1'] = '0' * 40
    json.dump(meta, open(sidecar, 'w'))

    with pytest.raises(RuntimeError, match='make_cache'):
        export_real64(splits, cache_dir, str(tmp_path / 'real64'))


def test_rerun_removes_orphans(tiny_setup, tmp_path):
    splits, cache_dir = tiny_setup
    out = str(tmp_path / 'real64')
    export_real64(splits, cache_dir, out)
    orphan = os.path.join(out, 'df', 'orphan.png')
    Image.fromarray(np.zeros((64, 64, 3), np.uint8)).save(orphan)

    export_real64(splits, cache_dir, out)

    assert not os.path.exists(orphan)
