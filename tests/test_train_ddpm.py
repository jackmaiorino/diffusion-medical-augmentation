"""Tests for the hand-rolled pieces of the DDPM training loop."""
import torch
from diffusers import UNet2DModel

import train_ddpm
from ddpm_model import NULL_CLASS, NUM_CLASSES, build_train_scheduler, build_unet
from train_ddpm import (EMA, accumulate_gradients, balanced_sampler,
                        drop_labels, ema_decay, resume_mismatches)


def tiny_unet(seed):
    """A small UNet2DModel shaped like the real one, for fast gradient tests."""
    torch.manual_seed(seed)
    return UNet2DModel(
        sample_size=16, in_channels=3, out_channels=3, layers_per_block=1,
        # 32 and up so the default 32 GroupNorm groups still divide evenly
        block_out_channels=(32, 64),
        down_block_types=('DownBlock2D', 'AttnDownBlock2D'),
        up_block_types=('AttnUpBlock2D', 'UpBlock2D'),
        num_class_embeds=NUM_CLASSES + 1)


def test_ema_ramps_from_fast_to_the_cap():
    # without the ramp, 0.9999 keeps random init dominant for ~10k steps
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


def test_accumulated_gradients_equal_one_undivided_batch():
    # two passes of 4 with the loss halved must match one pass of 8 exactly
    scheduler = build_train_scheduler()
    torch.manual_seed(612)
    images = torch.randn(8, 3, 16, 16)
    targets = torch.randint(0, NUM_CLASSES, (8,))
    noise = torch.randn(8, 3, 16, 16)
    timesteps = torch.randint(0, 1000, (8,))

    whole = tiny_unet(0)
    split = tiny_unet(0)
    for one, other in zip(whole.parameters(), split.parameters()):
        assert torch.equal(one, other), 'the two models must start identical'

    # grad_accum is the only difference, and both paths call the real
    # accumulate_gradients so a deleted division cannot pass
    undivided = accumulate_gradients(
        whole, scheduler, [(images, targets, noise, timesteps)],
        grad_accum=1, amp=False)
    halves = [(images[half], targets[half], noise[half], timesteps[half])
              for half in (slice(0, 4), slice(4, 8))]
    accumulated = accumulate_gradients(split, scheduler, halves,
                                       grad_accum=2, amp=False)

    for (name, one), (_, other) in zip(whole.named_parameters(),
                                       split.named_parameters()):
        assert torch.allclose(one.grad, other.grad, atol=1e-6), name
    # the reported loss has to match too, so log.csv stays comparable
    assert abs(undivided - accumulated) < 1e-6


def test_the_real_unet_has_no_batch_coupled_normalization():
    # GroupNorm is per-sample, a BatchNorm would break the equivalence above
    for module in build_unet().modules():
        assert not isinstance(module, torch.nn.modules.batchnorm._BatchNorm)


def test_the_loader_supplies_a_micro_batch_for_every_accumulation_pass():
    # a sampler sized in optimizer steps would starve the loop early
    steps, grad_accum, batch_size = 7, 3, 4
    labels = torch.tensor([0] * 20 + [1] * 20)
    data = torch.utils.data.TensorDataset(torch.randn(40, 3, 8, 8), labels)
    sampler = balanced_sampler(labels, steps * grad_accum * batch_size, 612)

    loader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                         sampler=sampler, drop_last=True)

    assert len(list(loader)) == steps * grad_accum


def test_resume_flags_a_changed_accumulation():
    # grad_accum changes the effective batch, so it is not administrative
    assert 'grad_accum' not in train_ddpm.IGNORED_ON_RESUME
    assert resume_mismatches({'grad_accum': 2},
                             {'grad_accum': 4}) == [('grad_accum', 2, 4)]


def test_resume_mismatches_ignores_administrative_fields():
    # these change where things get logged, not what is learned
    saved = {'resume': None, 'name': 'a', 'out': '/a', 'workers': 4,
             'log_every': 100, 'sample_every': 2000, 'ckpt_every': 5000,
             'no_cache': False, 'smoke': False}
    current = {'resume': 'ckpt.pt', 'name': 'b', 'out': '/b', 'workers': 8,
              'log_every': 50, 'sample_every': 1000, 'ckpt_every': 500,
              'no_cache': True, 'smoke': True}

    assert resume_mismatches(saved, current) == []
