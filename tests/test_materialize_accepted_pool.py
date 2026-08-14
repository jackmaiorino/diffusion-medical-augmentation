"""Tests for deterministic lesion-accepted pool materialization."""
import csv
import hashlib
import json
from pathlib import Path

import pytest

from materialize_accepted_pool import materialize_accepted_pool


def write_detail(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=['class', 'file', 'lpips_nn', 'inception_nn'])
        writer.writeheader()
        writer.writerows(rows)


def put(root, cls, name, content=None):
    folder = root / cls
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_bytes(content if content is not None else name.encode())
    return path


@pytest.fixture
def pool_inputs(tmp_path):
    splits = tmp_path / 'splits.csv'
    splits.write_text('image_id,lesion_id,dx,split\na,l,df,train\n')
    detail = tmp_path / 'detail.csv'
    rows = [
        {'class': 'df', 'file': 'df_00000.png',
         'lpips_nn': 0.20, 'inception_nn': 0.20},
        {'class': 'df', 'file': 'df_00001.png',
         'lpips_nn': 0.10, 'inception_nn': 0.11},
        {'class': 'df', 'file': 'df_00002.png',
         'lpips_nn': 0.01, 'inception_nn': 0.20},
        {'class': 'vasc', 'file': 'vasc_00000.png',
         'lpips_nn': 0.30, 'inception_nn': 0.30},
    ]
    write_detail(detail, rows)
    thresholds = tmp_path / 'thresholds.json'
    thresholds.write_text(json.dumps({
        'splits_sha1': hashlib.sha1(splits.read_bytes()).hexdigest(),
        'thresholds': {
            'df': {'lpips': 0.10, 'inception': 0.10},
            'vasc': {'lpips': 0.20, 'inception': 0.20},
        },
    }))
    active = tmp_path / 'pool'
    flagged = tmp_path / 'pool_flagged'
    put(active, 'df', 'df_00000.png', b'active survivor')
    put(active, 'df', 'df_00002.png', b'lesion rejected')
    put(flagged, 'df', 'df_00001.png', b'quarantined survivor')
    put(flagged, 'vasc', 'vasc_00000.png', b'vasc survivor')
    return detail, thresholds, splits, active, flagged


def tree_snapshot(root):
    return [
        (path.relative_to(root).as_posix(), path.read_bytes())
        for path in sorted(root.rglob('*')) if path.is_file()
    ]


def materialize(pool_inputs, destination):
    detail, thresholds, splits, active, flagged = pool_inputs
    return materialize_accepted_pool(
        detail, thresholds, active, destination, splits, flagged)


def test_materializes_survivors_from_both_trees_and_cleans_destination(
        pool_inputs, tmp_path):
    destination = tmp_path / 'accepted'
    put(destination, 'stale', 'old.png', b'stale')

    result = materialize(pool_inputs, destination)

    assert result == {
        'total': 4, 'accepted': 3, 'rejected': 1, 'classes': 2,
        'splits_sha1': hashlib.sha1(pool_inputs[2].read_bytes()).hexdigest(),
    }
    assert tree_snapshot(destination) == [
        ('df/df_00000.png', b'active survivor'),
        ('df/df_00001.png', b'quarantined survivor'),
        ('vasc/vasc_00000.png', b'vasc survivor'),
    ]
    assert sorted(path.name for path in destination.iterdir()) == ['df',
                                                                    'vasc']

    first = tree_snapshot(destination)
    materialize(pool_inputs, destination)
    assert tree_snapshot(destination) == first
    assert not list(tmp_path.glob('.accepted.*-*'))


def test_missing_source_fails_before_replacing_destination(pool_inputs,
                                                           tmp_path):
    destination = tmp_path / 'accepted'
    sentinel = put(destination, 'old', 'sentinel.png', b'keep me')
    pool_inputs[4].joinpath('df', 'df_00001.png').unlink()

    with pytest.raises(RuntimeError, match='missing source images'):
        materialize(pool_inputs, destination)

    assert sentinel.read_bytes() == b'keep me'


def test_duplicate_across_active_and_flagged_trees_fails(pool_inputs,
                                                         tmp_path):
    put(pool_inputs[4], 'df', 'df_00000.png', b'duplicate')

    with pytest.raises(RuntimeError, match='duplicate source image'):
        materialize(pool_inputs, tmp_path / 'accepted')


def test_unexpected_source_image_fails(pool_inputs, tmp_path):
    put(pool_inputs[3], 'df', 'df_99999.png', b'unexpected')

    with pytest.raises(RuntimeError, match='unexpected source images'):
        materialize(pool_inputs, tmp_path / 'accepted')


def test_unexpected_non_png_file_fails(pool_inputs, tmp_path):
    pool_inputs[3].joinpath('df', 'notes.txt').write_text('unexpected')

    with pytest.raises(RuntimeError, match='unexpected source file'):
        materialize(pool_inputs, tmp_path / 'accepted')


def test_unexpected_empty_class_directory_fails(pool_inputs, tmp_path):
    pool_inputs[3].joinpath('nv').mkdir()

    with pytest.raises(RuntimeError, match='unexpected source class'):
        materialize(pool_inputs, tmp_path / 'accepted')


def test_stale_split_hash_fails_before_replacing_destination(pool_inputs,
                                                             tmp_path):
    destination = tmp_path / 'accepted'
    sentinel = put(destination, 'old', 'sentinel.png', b'keep me')
    pool_inputs[2].write_text('changed\n')

    with pytest.raises(RuntimeError, match='different splits.csv'):
        materialize(pool_inputs, destination)

    assert sentinel.read_bytes() == b'keep me'


def test_missing_optional_flagged_tree_is_allowed(pool_inputs, tmp_path):
    detail, thresholds, splits, active, flagged = pool_inputs
    for path in list(flagged.rglob('*.png')):
        target = active / path.relative_to(flagged)
        target.parent.mkdir(parents=True, exist_ok=True)
        path.replace(target)
    for path in sorted(flagged.rglob('*'), reverse=True):
        if path.is_dir():
            path.rmdir()
    flagged.rmdir()

    result = materialize_accepted_pool(
        detail, thresholds, active, tmp_path / 'accepted', splits)

    assert result['accepted'] == 3


def test_destination_may_not_overlap_a_source_tree(pool_inputs):
    destination = pool_inputs[3] / 'accepted'

    with pytest.raises(RuntimeError, match='destination overlaps source tree'):
        materialize(pool_inputs, destination)
