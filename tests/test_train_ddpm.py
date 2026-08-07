"""Tests for the hand-rolled pieces of the DDPM training loop."""
import torch

import train_ddpm
from ddpm_model import NULL_CLASS
from train_ddpm import (EMA, balanced_sampler, drop_labels, ema_decay,
                        resume_mismatches)


def test_ema_ramps_from_fast_to_the_cap():
    # Without the ramp, decay 0.9999 would keep random init dominant for
    # tens of thousands of steps.
    assert ema_decay(0, 0.9999) == 0.1
    assert ema_decay(90, 0.9999) == 0.91
    assert ema_decay(1_000_000, 0.9999) == 0.9999


def test_ema_converges_to_repeated_weights():
    model = torch.nn.Linear(4, 4, bias=False)
    torch.nn.init.zeros_(model.weight)
    ema = EMA(model, decay=0.9)

    torch.nn.init.ones_(model.weight)
    for step in range(500):
        ema.update(model, step)

    target = torch.nn.Linear(4, 4, bias=False)
    ema.copy_to(target)
    assert torch.allclose(target.weight, torch.ones(4, 4), atol=1e-3)


def test_drop_labels_at_the_extremes():
    labels = torch.zeros(1000, dtype=torch.long)
    generator = torch.Generator().manual_seed(612)

    assert (drop_labels(labels, 1.0, NULL_CLASS, generator) == NULL_CLASS).all()
    assert (drop_labels(labels, 0.0, NULL_CLASS, generator) == 0).all()


def test_drop_labels_hits_the_configured_rate():
    labels = torch.zeros(100_000, dtype=torch.long)
    generator = torch.Generator().manual_seed(612)

    dropped = drop_labels(labels, 0.1, NULL_CLASS, generator)
    rate = (dropped == NULL_CLASS).float().mean().item()

    assert 0.09 < rate < 0.11


def test_drop_labels_leaves_the_input_alone():
    labels = torch.zeros(100, dtype=torch.long)
    generator = torch.Generator().manual_seed(612)

    drop_labels(labels, 1.0, NULL_CLASS, generator)

    assert (labels == 0).all()


def test_balanced_sampler_equalizes_a_9_to_1_imbalance():
    # HAM10000 is worse than this: nv is 56x df in the training split.
    labels = torch.tensor([0] * 900 + [1] * 100)
    sampler = balanced_sampler(labels, num_samples=40_000, seed=612)

    drawn = labels[torch.tensor(list(sampler))]
    minority_rate = (drawn == 1).float().mean().item()

    assert 0.47 < minority_rate < 0.53


def test_no_pretrained_weights_are_loaded():
    with open(train_ddpm.__file__) as handle:
        assert 'from_pretrained' not in handle.read()


def test_resume_mismatches_flags_a_changed_hyperparameter():
    saved = {'lr': 1e-4, 'batch_size': 64, 'seed': 612}
    current = {'lr': 3e-4, 'batch_size': 64, 'seed': 612}

    assert resume_mismatches(saved, current) == [('lr', 1e-4, 3e-4)]


def test_resume_mismatches_is_empty_when_current_values_match():
    saved = {'lr': 1e-4, 'batch_size': 64, 'seed': 612}
    current = {'lr': 1e-4, 'batch_size': 64, 'seed': 612}

    assert resume_mismatches(saved, current) == []


def test_resume_mismatches_ignores_administrative_fields():
    # resume always differs trivially, and these never change what is
    # learned, only where it is logged or how fast the loader runs.
    saved = {'resume': None, 'name': 'a', 'out': '/a', 'workers': 4,
             'log_every': 100, 'sample_every': 2000, 'ckpt_every': 5000,
             'no_cache': False, 'smoke': False}
    current = {'resume': 'ckpt.pt', 'name': 'b', 'out': '/b', 'workers': 8,
              'log_every': 50, 'sample_every': 1000, 'ckpt_every': 500,
              'no_cache': True, 'smoke': True}

    assert resume_mismatches(saved, current) == []
