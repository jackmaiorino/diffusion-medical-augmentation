"""Tests for the lesion-excluded threshold recalibration."""
import pandas
import torch

from memorization import loo_nn
from recalibrate_thresholds import lesion_codes, lesion_loo_nn, reflag


def test_lesion_loo_nn_skips_same_lesion_columns():
    distances = torch.tensor([[0.0, 1.0, 5.0],
                              [1.0, 0.0, 4.0],
                              [5.0, 4.0, 0.0]])
    codes = torch.tensor([0, 0, 1])

    result = lesion_loo_nn(distances, codes)

    # rows 0 and 1 share a lesion, so their sibling distance must be ignored
    assert result.tolist() == [5.0, 4.0, 4.0]
    assert loo_nn(distances).tolist() == [1.0, 1.0, 4.0]


def test_lesion_loo_nn_matches_loo_nn_when_lesions_are_unique():
    distances = torch.rand(5, 5)
    distances = distances + distances.T
    codes = torch.arange(5)

    assert torch.equal(lesion_loo_nn(distances, codes), loo_nn(distances))


def test_lesion_codes_groups_siblings():
    lesion_of = {'a': 'HAM_1', 'b': 'HAM_2', 'c': 'HAM_1'}

    codes = lesion_codes(['x/a.png', 'x/b.png', 'x/c.png'], lesion_of)

    assert codes[0] == codes[2]
    assert codes[0] != codes[1]


def test_reflag_uses_either_space():
    group = pandas.DataFrame({'lpips_nn': [0.01, 0.5, 0.5],
                              'inception_nn': [0.9, 0.02, 0.9]})
    thr = {'lpips': 0.05, 'inception': 0.05}

    assert reflag(group, thr).tolist() == [True, True, False]
