"""Tests for the arm manifest builder."""
import os

import numpy as np
import pandas
import pytest
from PIL import Image

from build_arms import build_arms, duplicate_manifest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def tiny_png(path):
    Image.fromarray(np.zeros((64, 64, 3), np.uint8)).save(path)


def fill_pool(pool_dir, dx, names):
    os.makedirs(os.path.join(pool_dir, dx), exist_ok=True)
    for name in names:
        tiny_png(os.path.join(pool_dir, dx, name))


@pytest.fixture
def setup(tmp_path):
    """A df/vasc splits.csv plus matching all/accepted pools."""
    rows = pandas.read_csv(os.path.join(REPO_ROOT, 'data', 'splits.csv'))
    picked = pandas.concat([rows[rows['dx'] == 'df'].head(4),
                            rows[rows['dx'] == 'vasc'].head(3)])
    picked = picked.reset_index(drop=True)
    picked['split'] = ['train', 'train', 'train', 'val',
                       'train', 'train', 'val']
    splits = tmp_path / 'splits.csv'
    picked.to_csv(splits, index=False)

    all_dir = str(tmp_path / 'all')
    accepted_dir = str(tmp_path / 'accepted')
    fill_pool(all_dir, 'df', [f'df_{i:05d}.png' for i in range(4)])
    fill_pool(all_dir, 'vasc', [f'vasc_{i:05d}.png' for i in range(3)])
    fill_pool(accepted_dir, 'df', ['df_00000.png', 'df_00002.png'])
    fill_pool(accepted_dir, 'vasc', ['vasc_00001.png'])
    return str(splits), all_dir, accepted_dir


def test_real_only_is_exactly_the_train_split(setup, tmp_path):
    splits, all_dir, accepted_dir = setup
    arms = build_arms(splits, all_dir, accepted_dir,
                      str(tmp_path / 'arms'), str(tmp_path), seed=612)

    rows = pandas.read_csv(splits)
    train = rows[rows['split'] == 'train']
    real = arms['real_only']
    assert (real['source'] == 'real').all()
    assert sorted(real['ref']) == sorted(train['image_id'])


def test_dup_counts_and_classes_match_the_accepted_pool(setup, tmp_path):
    splits, all_dir, accepted_dir = setup
    arms = build_arms(splits, all_dir, accepted_dir,
                      str(tmp_path / 'arms'), str(tmp_path), seed=612)

    dup = arms['dup_real'][arms['dup_real']['source'] == 'dup']
    assert dup.groupby('dx').size().to_dict() == {'df': 2, 'vasc': 1}

    rows = pandas.read_csv(splits)
    train = rows[rows['split'] == 'train']
    for entry in dup.itertuples():
        pool = set(train.loc[train['dx'] == entry.dx, 'image_id'])
        assert entry.ref in pool


def test_duplicate_manifest_survives_counts_beyond_the_pool():
    train = pandas.DataFrame({'dx': ['df'] * 3,
                              'image_id': ['a', 'b', 'c']})
    dup = duplicate_manifest(train, {'df': 10}, seed=612)
    assert len(dup) == 10
    assert set(dup['ref']) <= {'a', 'b', 'c'}


def test_manifests_are_deterministic(setup, tmp_path):
    splits, all_dir, accepted_dir = setup
    first = build_arms(splits, all_dir, accepted_dir,
                       str(tmp_path / 'one'), str(tmp_path), seed=612)
    second = build_arms(splits, all_dir, accepted_dir,
                        str(tmp_path / 'two'), str(tmp_path), seed=612)

    for name in first:
        pandas.testing.assert_frame_equal(first[name], second[name])


def test_synth_refs_are_portable_paths_that_exist(setup, tmp_path):
    splits, all_dir, accepted_dir = setup
    arms = build_arms(splits, all_dir, accepted_dir,
                      str(tmp_path / 'arms'), str(tmp_path), seed=612)

    synth = arms['synth_all'][arms['synth_all']['source'] == 'synth']
    assert len(synth) == 7
    for ref in synth['ref']:
        assert '\\' not in ref
        assert os.path.exists(os.path.join(str(tmp_path), ref))


def test_arm_totals_add_up(setup, tmp_path):
    splits, all_dir, accepted_dir = setup
    arms = build_arms(splits, all_dir, accepted_dir,
                      str(tmp_path / 'arms'), str(tmp_path), seed=612)

    assert len(arms['real_only']) == 5
    assert len(arms['dup_real']) == 8
    assert len(arms['synth_all']) == 12
    assert len(arms['synth_accepted']) == 8


def test_accepted_image_outside_the_full_pool_raises(setup, tmp_path):
    splits, all_dir, accepted_dir = setup
    tiny_png(os.path.join(accepted_dir, 'df', 'df_99999.png'))

    with pytest.raises(RuntimeError, match='missing from the full pool'):
        build_arms(splits, all_dir, accepted_dir,
                   str(tmp_path / 'arms'), str(tmp_path), seed=612)


def test_pool_with_wrong_classes_raises(setup, tmp_path):
    splits, all_dir, accepted_dir = setup
    fill_pool(all_dir, 'nv', ['nv_00000.png'])

    with pytest.raises(RuntimeError, match='disagree on classes'):
        build_arms(splits, all_dir, accepted_dir,
                   str(tmp_path / 'arms'), str(tmp_path), seed=612)


def test_stray_files_do_not_count_as_images(setup, tmp_path):
    splits, all_dir, accepted_dir = setup
    # Explorer artifacts in both pools must not skew counts or manifests
    for pool in (all_dir, accepted_dir):
        with open(os.path.join(pool, 'df', 'desktop.ini'), 'w') as handle:
            handle.write('[ViewState]')
    with open(os.path.join(all_dir, 'Thumbs.db'), 'w') as handle:
        handle.write('junk')

    arms = build_arms(splits, all_dir, accepted_dir,
                      str(tmp_path / 'arms'), str(tmp_path), seed=612)

    dup = arms['dup_real'][arms['dup_real']['source'] == 'dup']
    assert dup.groupby('dx').size().to_dict() == {'df': 2, 'vasc': 1}
    synth = arms['synth_all'][arms['synth_all']['source'] == 'synth']
    assert all(ref.endswith('.png') for ref in synth['ref'])
