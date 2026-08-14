"""Tests for classifier-free guidance arithmetic."""
import pytest
import torch

import sample_ddpm
from sample_ddpm import (apply_guidance, prepare_output_root,
                         resolve_image_size, staged_output_root)


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


def test_prepare_output_root_leaves_a_missing_directory_unwritten(tmp_path):
    target = tmp_path / 'samples'

    assert prepare_output_root(str(target)) == str(target.resolve())
    assert not target.exists()


def test_prepare_output_root_accepts_an_empty_directory(tmp_path):
    target = tmp_path / 'samples'
    target.mkdir()

    prepare_output_root(str(target))


def test_prepare_output_root_rejects_stale_content(tmp_path):
    target = tmp_path / 'samples'
    target.mkdir()
    (target / 'old.png').touch()

    with pytest.raises(RuntimeError, match='must be empty'):
        prepare_output_root(str(target))


def test_staged_output_root_installs_only_after_success(tmp_path):
    target = tmp_path / 'samples'

    with staged_output_root(str(target)) as staged:
        staged_path = tmp_path / staged
        assert not target.exists()
        (staged_path / 'df').mkdir()
        (staged_path / 'df' / 'df_00000.png').touch()

    assert (target / 'df' / 'df_00000.png').is_file()
    assert not staged_path.exists()


def test_staged_output_root_discards_partial_pool_on_error(tmp_path):
    target = tmp_path / 'samples'

    with pytest.raises(RuntimeError, match='generation failed'):
        with staged_output_root(str(target)) as staged:
            staged_path = tmp_path / staged
            (staged_path / 'partial.png').touch()
            raise RuntimeError('generation failed')

    assert not target.exists()
    assert not staged_path.exists()
