"""Tests for the class-conditional UNet definition."""
import torch

import ddpm_model
import sample_ddpm
import train_ddpm
from ddpm_model import (NULL_CLASS, NUM_CLASSES, SCHEDULER_KWARGS,
                        build_sample_scheduler, build_train_scheduler,
                        build_unet)


def test_parameter_count_matches_the_reported_architecture():
    # the interim report says 37.1M, if this changes the report is wrong
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
    # the project requires training from scratch, not fine-tuning
    with open(ddpm_model.__file__) as handle:
        assert 'from_pretrained' not in handle.read()


def test_scheduler_kwargs_are_unchanged():
    # existing checkpoints were trained under exactly these values
    assert SCHEDULER_KWARGS == {'num_train_timesteps': 1000,
                                'beta_schedule': 'squaredcos_cap_v2'}


def test_train_and_sample_schedulers_agree_on_the_noise_schedule():
    # a mismatched noise schedule degrades samples without ever raising
    train = build_train_scheduler()
    sample = build_sample_scheduler()

    assert train.config.num_train_timesteps == \
        sample.config.num_train_timesteps == 1000
    assert train.config.beta_schedule == \
        sample.config.beta_schedule == 'squaredcos_cap_v2'


def test_neither_script_builds_its_own_scheduler():
    # the agreement above only holds while both scripts use the builders
    for module in (train_ddpm, sample_ddpm):
        with open(module.__file__) as handle:
            source = handle.read()
        assert 'squaredcos_cap_v2' not in source
        assert 'DDPMScheduler(' not in source
        # from_config on the training scheduler's config is still allowed
        assert 'DDIMScheduler(' not in source
