"""Tests for classifier-free guidance arithmetic."""
import pytest
import torch

import sample_ddpm
from sample_ddpm import apply_guidance, resolve_image_size


def test_weight_of_one_is_the_unguided_prediction():
    conditional = torch.tensor([1.0, 2.0])
    unconditional = torch.tensor([5.0, 9.0])

    assert torch.allclose(apply_guidance(conditional, unconditional, 1.0),
                          conditional)


def test_weight_of_zero_is_the_unconditional_prediction():
    conditional = torch.tensor([1.0, 2.0])
    unconditional = torch.tensor([5.0, 9.0])

    assert torch.allclose(apply_guidance(conditional, unconditional, 0.0),
                          unconditional)


def test_higher_weight_extrapolates_past_the_conditional():
    conditional = torch.tensor([1.0])
    unconditional = torch.tensor([0.0])

    assert torch.allclose(apply_guidance(conditional, unconditional, 3.0),
                          torch.tensor([3.0]))


def test_no_pretrained_weights_are_loaded():
    with open(sample_ddpm.__file__) as handle:
        assert 'from_pretrained' not in handle.read()


def test_resolve_image_size_takes_the_checkpoint_value_when_omitted():
    assert resolve_image_size(None, 64) == 64


def test_resolve_image_size_accepts_a_matching_override():
    assert resolve_image_size(64, 64) == 64


def test_resolve_image_size_rejects_a_mismatched_override():
    with pytest.raises(ValueError, match='32.*64|64.*32'):
        resolve_image_size(32, 64)
