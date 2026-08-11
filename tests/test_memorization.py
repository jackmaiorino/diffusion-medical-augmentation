"""Tests for the memorization check."""
import numpy as np
import torch
from PIL import Image

from memorization import (build_inception, build_lpips, cosine_distances,
                          inception_vectors, load_batch, lpips_distances,
                          lpips_vectors)


def save_pngs(folder, count, side=64):
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(612)
    paths = []
    for i in range(count):
        arr = rng.integers(0, 255, (side, side, 3), dtype=np.uint8)
        path = folder / f'img_{i:03d}.png'
        Image.fromarray(arr).save(path)
        paths.append(str(path))
    return paths


def test_load_batch_shape_and_range(tmp_path):
    paths = save_pngs(tmp_path / 'imgs', 3)

    batch = load_batch(paths)

    assert batch.shape == (3, 3, 64, 64)
    assert 0.0 <= batch.min() and batch.max() <= 1.0


def test_lpips_vectors_reproduce_the_direct_distance(tmp_path):
    # the whole point of the vector path: chunked math must equal the
    # network's own answer, or every threshold downstream is meaningless
    paths = save_pngs(tmp_path / 'imgs', 4)
    model = build_lpips('cpu')

    vectors = lpips_vectors(model, paths, 'cpu', batch_size=2)
    images = load_batch(paths) * 2 - 1
    with torch.no_grad():
        direct = model(images[:2], images[2:]).flatten()
    decomposed = ((vectors[:2] - vectors[2:]) ** 2).sum(dim=1)

    assert torch.allclose(direct, decomposed, atol=1e-6)


def test_lpips_distances_match_the_broadcast_sum():
    a = torch.randn(5, 40)
    b = torch.randn(3, 40)

    distances = lpips_distances(a, b, 'cpu', chunk=2)

    expected = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
    assert distances.shape == (5, 3)
    assert torch.allclose(distances, expected, atol=1e-5)


def test_inception_vectors_are_unit_rows(tmp_path):
    paths = save_pngs(tmp_path / 'imgs', 3)
    model, transform = build_inception('cpu')

    vectors = inception_vectors(model, transform, paths, 'cpu', batch_size=2)

    assert vectors.shape == (3, 2048)
    assert torch.allclose(vectors.norm(dim=1), torch.ones(3), atol=1e-5)


def test_cosine_distances_on_known_vectors():
    a = torch.eye(3)
    b = torch.eye(3)

    distances = cosine_distances(a, b, 'cpu', chunk=2)

    assert torch.allclose(distances.diagonal(), torch.zeros(3), atol=1e-6)
    assert torch.allclose(distances[0, 1], torch.tensor(1.0), atol=1e-6)


from memorization import (class_thresholds, flag_mask, load_thresholds,
                          loo_nn, nn_to_reference, save_thresholds)


def test_loo_nn_excludes_the_diagonal():
    distances = torch.tensor([[0.0, 5.0, 2.0],
                              [5.0, 0.0, 9.0],
                              [2.0, 9.0, 0.0]])

    assert loo_nn(distances).tolist() == [2.0, 5.0, 2.0]


def test_nn_to_reference_returns_ids_with_values():
    distances = torch.tensor([[3.0, 1.0], [0.5, 4.0]])

    values, ids = nn_to_reference(distances, ['a', 'b'])

    assert values.tolist() == [1.0, 0.5]
    assert ids == ['b', 'a']


def test_flag_mask_fires_on_either_space():
    lpips_nn = torch.tensor([0.1, 0.9, 0.9, 0.1])
    incep_nn = torch.tensor([0.9, 0.1, 0.9, 0.1])

    mask = flag_mask(lpips_nn, incep_nn, 0.5, 0.5)

    assert mask.tolist() == [True, True, False, True]


def test_class_thresholds_take_the_5th_percentile():
    values = torch.arange(101, dtype=torch.float32)

    thresholds = class_thresholds(values, values * 2)

    assert abs(thresholds['lpips'] - 5.0) < 1e-6
    assert abs(thresholds['inception'] - 10.0) < 1e-6


def test_threshold_cache_roundtrip(tmp_path):
    path = str(tmp_path / 'thresholds.json')
    data = {'df': {'lpips': 0.1, 'inception': 0.2}}

    save_thresholds(path, data, 'sha-one')

    assert load_thresholds(path, 'sha-one') == data
    assert load_thresholds(path, 'sha-two') is None
    assert load_thresholds(str(tmp_path / 'missing.json'), 'sha-one') is None


import csv
import os

from memorization import quarantine, write_rows


def test_write_rows_overwrites_with_given_columns(tmp_path):
    path = str(tmp_path / 'out.csv')
    with open(path, 'w') as handle:
        handle.write('stale')

    write_rows(path, ['x', 'y'], [{'x': 1, 'y': 2}])

    with open(path, newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == ['x', 'y']
    assert len(rows) == 1


def test_quarantine_moves_only_flagged_files(tmp_path):
    synth = tmp_path / 'setname' / 'df'
    keep = save_pngs(synth, 2)[0]
    flagged = str(synth / 'img_001.png')

    root = quarantine(str(tmp_path / 'setname'), [flagged])

    assert root == str(tmp_path / 'setname_flagged')
    assert os.path.exists(os.path.join(root, 'df', 'img_001.png'))
    assert not os.path.exists(flagged)
    assert os.path.exists(keep)
