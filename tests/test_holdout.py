"""Tests for the held-out distance comparison."""
import torch

from holdout_distance import matched_nn, matched_stats


def test_matched_nn_with_all_columns_equals_full_min():
    distances = torch.rand(6, 4)

    subs = matched_nn(distances, 4, seeds=3)

    for sub in subs:
        assert torch.equal(sub, distances.min(dim=1).values)


def test_matched_nn_never_beats_the_full_min():
    distances = torch.rand(8, 10)
    full = distances.min(dim=1).values

    for sub in matched_nn(distances, 3, seeds=5):
        assert (sub >= full).all()


def test_matched_stats_fraction_bounds():
    distances = torch.zeros(5, 4)
    other = torch.ones(5)

    med, frac = matched_stats(distances, other, 2, seeds=3)

    assert med == 0.0
    assert frac == 1.0
