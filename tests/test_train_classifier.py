"""Tests for the classifier arm dataset and training pieces."""
import os

import numpy as np
import pandas
import pytest
import torch
from PIL import Image

import train_classifier
from dataset import CLASS_TO_INDEX, IMAGENET_MEAN, IMAGENET_STD
from make_cache import build_cache
from train_classifier import (ArmDataset, build_resnet18, compute_metrics,
                              evaluate, lr_at)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELED = os.path.join(REPO_ROOT, 'data', 'labeled')


@pytest.fixture
def tiny_arm(tmp_path):
    """A df/vasc manifest with real, dup and synth rows plus its cache."""
    rows = pandas.read_csv(os.path.join(REPO_ROOT, 'data', 'splits.csv'))
    picked = pandas.concat([rows[rows['dx'] == 'df'].head(3),
                            rows[rows['dx'] == 'vasc'].head(2)])
    picked = picked.reset_index(drop=True)
    picked['split'] = ['train', 'train', 'val', 'train', 'train']
    splits = tmp_path / 'splits.csv'
    picked.to_csv(splits, index=False)
    build_cache(str(splits), LABELED, str(tmp_path), 64)

    synth_dir = tmp_path / 'synthetic' / 'df'
    synth_dir.mkdir(parents=True)
    Image.fromarray(np.full((64, 64, 3), 7, np.uint8)).save(
        synth_dir / 'df_00000.png')

    train = picked[picked['split'] == 'train']
    manifest = pandas.DataFrame({
        'dx': list(train['dx']) + ['df', 'df'],
        'source': ['real'] * len(train) + ['dup', 'synth'],
        'ref': list(train['image_id']) + [train['image_id'].iloc[0],
                                          'synthetic/df/df_00000.png']})
    manifest_csv = tmp_path / 'arm.csv'
    manifest.to_csv(manifest_csv, index=False)
    return str(manifest_csv), str(splits), str(tmp_path)


def load_arm(tiny_arm, augment=False):
    manifest_csv, splits, root = tiny_arm
    return ArmDataset(manifest_csv, augment, splits_csv=splits,
                      cache_dir=root, repo_root=root)


def test_arm_dataset_loads_real_dup_and_synth_rows(tiny_arm):
    data = load_arm(tiny_arm)

    assert len(data) == 6
    labels = [CLASS_TO_INDEX[dx] for dx in
              ('df', 'df', 'vasc', 'vasc', 'df', 'df')]
    assert data.labels.tolist() == labels
    # the dup row must be a pixel copy of its source real row
    assert np.array_equal(data.images[4], data.images[0])
    assert (data.images[5] == 7).all()


def test_arm_dataset_rejects_non_train_rows(tiny_arm, tmp_path):
    manifest_csv, splits, root = tiny_arm
    rows = pandas.read_csv(splits)
    val_id = rows.loc[rows['split'] == 'val', 'image_id'].iloc[0]
    bad = pandas.read_csv(manifest_csv)
    bad.loc[0, 'ref'] = val_id
    bad_csv = tmp_path / 'bad.csv'
    bad.to_csv(bad_csv, index=False)

    with pytest.raises(RuntimeError, match='non-train'):
        ArmDataset(str(bad_csv), False, splits_csv=splits,
                   cache_dir=root, repo_root=root)


def test_arm_dataset_rejects_unknown_sources(tiny_arm, tmp_path):
    manifest_csv, splits, root = tiny_arm
    bad = pandas.read_csv(manifest_csv)
    bad.loc[0, 'source'] = 'gan'
    bad_csv = tmp_path / 'bad.csv'
    bad.to_csv(bad_csv, index=False)

    with pytest.raises(RuntimeError, match='unknown manifest sources'):
        ArmDataset(str(bad_csv), False, splits_csv=splits,
                   cache_dir=root, repo_root=root)


def test_items_are_imagenet_normalized(tiny_arm):
    data = load_arm(tiny_arm)
    image, label = data[5]

    assert image.shape == (3, 64, 64)
    assert label == CLASS_TO_INDEX['df']
    expected = (7 / 255 - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
    for channel in range(3):
        assert torch.allclose(image[channel],
                              torch.full((64, 64), expected[channel]).float(),
                              atol=1e-5)


def test_augmentation_actually_varies_the_image(tiny_arm):
    data = load_arm(tiny_arm, augment=True)
    torch.manual_seed(612)
    one, _ = data[0]
    other, _ = data[0]

    assert not torch.equal(one, other)


def test_resnet_stem_keeps_64px_resolution():
    model = build_resnet18()

    assert model.conv1.kernel_size == (3, 3)
    assert model.conv1.stride == (1, 1)
    assert isinstance(model.maxpool, torch.nn.Identity)
    logits = model(torch.randn(2, 3, 64, 64))
    assert logits.shape == (2, 7)


def test_no_pretrained_weights_are_loaded():
    with open(train_classifier.__file__) as handle:
        source = handle.read()
    assert 'weights=None' in source
    assert 'from_pretrained' not in source


def test_lr_warms_up_then_decays_to_zero():
    peak, total, warmup = 1e-3, 1000, 100

    assert lr_at(0, total, warmup, peak) == pytest.approx(peak / warmup)
    assert lr_at(warmup, total, warmup, peak) == pytest.approx(peak)
    assert lr_at(total - 1, total, warmup, peak) < peak / 100
    middle = lr_at(warmup + (total - warmup) // 2, total, warmup, peak)
    assert middle == pytest.approx(peak / 2, rel=0.01)


def test_perfect_predictions_score_one_everywhere():
    labels = torch.arange(7).repeat(3)
    logits = torch.nn.functional.one_hot(labels, 7).float() * 10

    metrics = compute_metrics(logits, labels)

    assert metrics['macro_f1'] == pytest.approx(1.0)
    assert metrics['rare_f1'] == pytest.approx(1.0)
    assert metrics['balanced_acc'] == pytest.approx(1.0)
    assert metrics['auc_ovr'] == pytest.approx(1.0)


def test_rare_f1_averages_the_three_rare_classes():
    torch.manual_seed(612)
    labels = torch.arange(7).repeat(4)
    logits = torch.randn(28, 7)

    metrics = compute_metrics(logits, labels)

    expected = np.mean([metrics['f1_akiec'], metrics['f1_df'],
                        metrics['f1_vasc']])
    assert metrics['rare_f1'] == pytest.approx(expected)


def test_evaluate_covers_every_row_in_batches():
    torch.manual_seed(612)
    model = torch.nn.Sequential(torch.nn.Flatten(),
                                torch.nn.Linear(3 * 8 * 8, 7))
    images = torch.randn(10, 3, 8, 8)
    labels = torch.cat([torch.arange(7), torch.tensor([0, 1, 2])])

    loss, metrics = evaluate(model, images, labels, 'cpu', batch_size=4)

    assert np.isfinite(loss)
    assert set(metrics) >= {'macro_f1', 'rare_f1', 'balanced_acc', 'auc_ovr'}
    assert model.training, 'evaluate must hand the model back in train mode'
