"""Tests for the class-conditional UNet definition."""
import torch

import ddpm_model
import sample_ddpm
import train_ddpm
from ddpm_model import (NULL_CLASS, NUM_CLASSES, SCHEDULER_KWARGS,
                        build_sample_scheduler, build_train_scheduler,
                        build_unet)


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


def test_train_and_sample_schedulers_agree_on_the_noise_schedule():
    # Compares the schedulers the two scripts actually construct, not the
    # source text. A sampler running a different noise schedule than the
    # checkpoint was trained under degrades samples without raising.
    train = build_train_scheduler()
    sample = build_sample_scheduler()

    assert train.config.num_train_timesteps == \
        sample.config.num_train_timesteps == 1000
    assert train.config.beta_schedule == \
        sample.config.beta_schedule == 'squaredcos_cap_v2'


def test_neither_script_builds_its_own_scheduler():
    # The agreement test above is only meaningful while both scripts go
    # through the shared builders, so guard the single call site too.
    for module in (train_ddpm, sample_ddpm):
        with open(module.__file__) as handle:
            source = handle.read()
        assert 'squaredcos_cap_v2' not in source
        assert 'DDPMScheduler(' not in source
        # train_ddpm's preview sampler legitimately derives a DDIMScheduler
        # from the training scheduler's own config, so allow from_config.
        assert 'DDIMScheduler(' not in source
