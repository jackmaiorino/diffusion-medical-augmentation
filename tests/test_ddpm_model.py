"""Tests for the class-conditional UNet definition."""
import torch

import ddpm_model
import sample_ddpm
import train_ddpm
from ddpm_model import NULL_CLASS, NUM_CLASSES, SCHEDULER_KWARGS, build_unet


def test_parameter_count_matches_the_reported_architecture():
    # The interim report states 37.1M. If this changes, the report is wrong.
    assert sum(p.numel() for p in build_unet().parameters()) == 37_068_803


def test_forward_accepts_every_class_and_the_null_token():
    model = build_unet()
    labels = torch.arange(NUM_CLASSES + 1)
    noise = torch.randn(NUM_CLASSES + 1, 3, 64, 64)
    steps = torch.full((NUM_CLASSES + 1,), 10)

    out = model(noise, steps, class_labels=labels).sample

    assert out.shape == (NUM_CLASSES + 1, 3, 64, 64)


def test_null_class_sits_past_the_real_classes():
    from dataset import CLASSES

    assert NUM_CLASSES == len(CLASSES) == 7
    assert NULL_CLASS == 7


def test_no_pretrained_weights_are_loaded():
    # The project requires training from scratch, not fine-tuning.
    with open(ddpm_model.__file__) as handle:
        assert 'from_pretrained' not in handle.read()


def test_scheduler_kwargs_are_unchanged():
    # Values must stay exactly as they were before the refactor that moved
    # them here from train_ddpm.py and sample_ddpm.py.
    assert SCHEDULER_KWARGS == {'num_train_timesteps': 1000,
                                'beta_schedule': 'squaredcos_cap_v2'}


def test_train_and_sample_share_the_one_scheduler_config():
    # Both modules must build their scheduler from ddpm_model.SCHEDULER_KWARGS
    # instead of repeating the literals, so a future ablation to the training
    # schedule cannot silently desync the sampler.
    with open(train_ddpm.__file__) as handle:
        train_source = handle.read()
    with open(sample_ddpm.__file__) as handle:
        sample_source = handle.read()

    assert 'SCHEDULER_KWARGS' in train_source
    assert 'SCHEDULER_KWARGS' in sample_source
    assert 'squaredcos_cap_v2' not in train_source
    assert 'squaredcos_cap_v2' not in sample_source
