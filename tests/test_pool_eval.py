"""Tests for the filtered-pool evaluation."""
import pandas
import pytest
import torch

from pool_eval import (accepted_paths, diversity_stats, matched_diversity,
                       sample_indices, validated_pool_paths)


def test_accepted_paths_survivors_found_in_either_tree(tmp_path):
    detail = pandas.DataFrame({
        'class': ['df'] * 3,
        'file': ['a.png', 'b.png', 'c.png'],
        'lpips_nn': [0.5, 0.5, 0.01],
        'inception_nn': [0.5, 0.5, 0.5]})
    thresholds = {'df': {'lpips': 0.05, 'inception': 0.05}}
    synth = tmp_path / 'pool'
    (synth / 'df').mkdir(parents=True)
    (synth / 'df' / 'a.png').touch()
    flagged = tmp_path / 'pool_flagged'
    (flagged / 'df').mkdir(parents=True)
    (flagged / 'df' / 'b.png').touch()

    paths = accepted_paths(detail, 'df', thresholds, str(synth))

    assert [p.split('pool')[-1] for p in paths] == \
        ['\\df\\a.png', '_flagged\\df\\b.png']


def test_accepted_paths_rejects_a_missing_survivor(tmp_path):
    detail = pandas.DataFrame({
        'class': ['df'], 'file': ['missing.png'],
        'lpips_nn': [0.5], 'inception_nn': [0.5]})
    synth = tmp_path / 'pool'
    (synth / 'df').mkdir(parents=True)

    with pytest.raises(RuntimeError, match='expected exactly one'):
        accepted_paths(detail, 'df',
                       {'df': {'lpips': 0.05, 'inception': 0.05}},
                       str(synth))


def test_validated_pool_paths_rejects_partial_pool(tmp_path):
    detail = pandas.DataFrame({
        'class': ['df', 'df'], 'file': ['a.png', 'b.png']})
    synth = tmp_path / 'pool'
    (synth / 'df').mkdir(parents=True)
    (synth / 'df' / 'a.png').touch()

    with pytest.raises(RuntimeError, match='inventory mismatch.*b.png'):
        validated_pool_paths(detail, 'df', str(synth))


def test_sample_indices_rejects_an_oversized_draw():
    with pytest.raises(ValueError, match='cannot draw'):
        sample_indices(3, 4, 612)


def test_diversity_stats_counts_near_duplicates():
    vectors = torch.tensor([[0.0], [0.1], [5.0]])

    mean_pairwise, dup_frac = diversity_stats(vectors, 'cpu', 0.5)

    # nn distances are squared L2: 0.01, 0.01, 24.01 -> two under 0.5
    assert abs(dup_frac - 2 / 3) < 1e-6
    assert mean_pairwise > 0


def test_matched_diversity_uses_the_smaller_pool_size():
    real = torch.arange(10, dtype=torch.float32).unsqueeze(1)
    real_codes = torch.arange(10)
    pool = torch.tensor([[0.0], [0.01], [5.0], [9.0]])

    result = matched_diversity(real, real_codes, pool, 'cpu', cap=8,
                               repeats=3, seed=612)

    assert result['diversity_n'] == 4
    assert result['diversity_repeats'] == 3
    assert 0 <= result['near_dup_frac'] <= 1
    assert 0 <= result['null_near_dup_frac'] <= 1


def test_matched_diversity_is_deterministic():
    torch.manual_seed(612)
    real = torch.rand(12, 3)
    pool = torch.rand(7, 3)
    codes = torch.arange(12)

    one = matched_diversity(real, codes, pool, 'cpu', repeats=4)
    other = matched_diversity(real, codes, pool, 'cpu', repeats=4)

    assert one == other
