"""Tests for the FID/KID orchestration around the metric call."""
import csv
import os

import numpy as np
from PIL import Image

from eval_fid_kid import (CSV_COLUMNS, discover_classes, half_split,
                          kid_subset_size, list_pngs, write_csv)


def png(folder, name):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, name)
    Image.fromarray(np.zeros((64, 64, 3), np.uint8)).save(path)
    return path


def test_list_pngs_ignores_non_png_files(tmp_path):
    folder = str(tmp_path)
    png(folder, 'b.png')
    png(folder, 'a.png')
    (tmp_path / 'Thumbs.db').write_text('junk')

    paths = list_pngs(folder)

    assert [os.path.basename(p) for p in paths] == ['a.png', 'b.png']


def test_list_pngs_of_a_missing_folder_is_empty(tmp_path):
    assert list_pngs(str(tmp_path / 'nope')) == []


def test_list_pngs_returns_absolute_paths_for_relative_folder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    png('subfolder-name', 'a.png')

    paths = list_pngs('subfolder-name')

    assert all(os.path.isabs(p) for p in paths)
    assert [os.path.basename(p) for p in paths] == ['a.png']


def test_discover_classes_partitions_and_flags_unknown(tmp_path):
    real, synth = str(tmp_path / 'real'), str(tmp_path / 'synth')
    for name in ('df', 'vasc'):
        png(os.path.join(real, name), 'r.png')
    png(os.path.join(synth, 'df'), 's.png')
    os.makedirs(os.path.join(synth, 'vasc'))          # empty, must skip
    png(os.path.join(synth, 'melanoma'), 's.png')     # typo, must warn

    scored, skipped, unknown = discover_classes(real, synth)

    assert scored == ['df']
    assert 'vasc' in skipped and 'akiec' in skipped
    assert unknown == ['melanoma']


def test_half_split_is_seeded_and_exact_on_odd_counts():
    paths = [f'p{i}.png' for i in range(7)]

    first_a, first_b = half_split(paths, 612)
    second_a, second_b = half_split(paths, 612)

    assert (first_a, first_b) == (second_a, second_b)
    assert len(first_a) == 3 and len(first_b) == 4
    assert sorted(first_a + first_b) == sorted(paths)


def test_kid_subset_size_clamps_to_the_smallest_side():
    assert kid_subset_size(84, 1000) == 84
    assert kid_subset_size(4679, 1000) == 100
    assert kid_subset_size(42, 42) == 42


def test_write_csv_overwrites_with_fixed_columns(tmp_path):
    path = str(tmp_path / 'scores.csv')
    with open(path, 'w') as handle:
        handle.write('stale content')
    row = dict.fromkeys(CSV_COLUMNS, 0)
    row['class'] = 'df'

    write_csv(path, [row])

    with open(path, newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == CSV_COLUMNS
    assert rows[0]['class'] == 'df'
    assert len(rows) == 1


from eval_fid_kid import floor_metrics, score_class


def fake_metrics(calls):
    """A metrics stand-in that records call-time directory counts and args."""
    def metrics(dir_a, dir_b, subset_size, seed):
        # count while the temp dirs still exist, they are gone on return
        calls.append((len(list_pngs(dir_a)), len(list_pngs(dir_b)),
                      subset_size, seed))
        return 12.0, 0.002, 0.0005
    return metrics


def test_floor_metrics_splits_copies_and_sizes_correctly(tmp_path):
    folder = str(tmp_path / 'real')
    paths = [png(folder, f'r{i}.png') for i in range(7)]
    calls = []

    scores = floor_metrics(paths, 612, metrics=fake_metrics(calls))

    assert scores == (12.0, 0.002, 0.0005)
    assert calls == [(3, 4, 3, 612)]


def test_score_class_builds_a_csv_row_with_kid_x1000(tmp_path):
    real, synth = str(tmp_path / 'real'), str(tmp_path / 'synth')
    for i in range(5):
        png(os.path.join(real, 'df'), f'r{i}.png')
    for i in range(3):
        png(os.path.join(synth, 'df'), f's{i}.png')
    calls = []

    row = score_class('df', real, synth, 612, metrics=fake_metrics(calls))

    assert list(row) == CSV_COLUMNS
    assert (row['n_real'], row['n_synth']) == (5, 3)
    assert row['fid'] == 12.0 and row['fid_floor'] == 12.0
    assert row['kid_mean'] == 2.0 and row['kid_floor_mean'] == 2.0
    assert row['kid_std'] == 0.5
    assert calls[0] == (5, 3, 3, 612)    # main pair, subset min(5, 3, 100)
    assert calls[1] == (2, 3, 2, 612)    # floor halves, subset min(2, 3, 100)
