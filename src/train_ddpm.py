"""Train the class-conditional DDPM from scratch on HAM10000."""
import argparse
import copy
import csv
import os
import random
import time

import numpy
import torch
from diffusers import DDIMScheduler, DDPMScheduler
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.utils import save_image

from dataset import CLASS_TO_INDEX, HAM10000
from ddpm_model import NULL_CLASS, NUM_CLASSES, SCHEDULER_KWARGS, build_unet

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ema_decay(step, cap):
    """EMA decay with a warmup ramp so early steps are not stuck at init."""
    return min(cap, (1 + step) / (10 + step))


class EMA:
    """Exponential moving average of model weights."""

    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for parameter in self.shadow.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model, step):
        rate = ema_decay(step, self.decay)
        for shadow, live in zip(self.shadow.parameters(), model.parameters()):
            shadow.lerp_(live.detach(), 1 - rate)
        for shadow, live in zip(self.shadow.buffers(), model.buffers()):
            shadow.copy_(live)

    def copy_to(self, model):
        model.load_state_dict(self.shadow.state_dict())

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, state):
        self.shadow.load_state_dict(state)


def drop_labels(labels, p, null_class, generator=None):
    """Replace labels with the null token at probability p, for guidance."""
    mask = torch.rand(labels.shape, generator=generator,
                      device=labels.device) < p
    return torch.where(mask, torch.full_like(labels, null_class), labels)


def balanced_sampler(labels, num_samples, seed):
    """Sample so every class appears with equal expected frequency."""
    counts = torch.bincount(labels, minlength=NUM_CLASSES).float()
    weights = (1.0 / counts.clamp(min=1))[labels]
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(weights, num_samples=num_samples,
                                 replacement=True, generator=generator)


def set_seed(seed):
    """Seed torch, numpy and random so init and data order are reproducible."""
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)


@torch.no_grad()
def sample_grid(model, scheduler, device, guidance, per_class=4, steps=50,
                seed=612):
    """Draw per_class images for every class and tile them into one figure."""
    sampler = DDIMScheduler.from_config(scheduler.config)
    sampler.set_timesteps(steps)

    labels = torch.arange(NUM_CLASSES, device=device).repeat_interleave(
        per_class)
    null = torch.full_like(labels, NULL_CLASS)
    generator = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn(len(labels), 3, model.config.sample_size,
                          model.config.sample_size, device=device,
                          generator=generator)

    for timestep in sampler.timesteps:
        batch = torch.cat([latents, latents])
        both = torch.cat([labels, null])
        with torch.autocast('cuda', dtype=torch.bfloat16,
                            enabled=device.startswith('cuda')):
            predicted = model(batch, timestep, class_labels=both).sample
        conditional, unconditional = predicted.float().chunk(2)
        guided = unconditional + guidance * (conditional - unconditional)
        latents = sampler.step(guided, timestep, latents).prev_sample

    return (latents.clamp(-1, 1) + 1) / 2


# Fields where a difference from the checkpoint is expected and meaningless:
# output paths, dataloader workers, logging/sampling/checkpoint cadence, and
# cache mode never change what gets learned.
IGNORED_ON_RESUME = {'resume', 'name', 'out', 'workers', 'log_every',
                     'sample_every', 'ckpt_every', 'no_cache', 'smoke'}


def resume_mismatches(saved_args, current_args):
    """List (field, checkpoint value, current value) for args that differ."""
    return [(key, saved_args[key], current_args[key]) for key in saved_args
           if key not in IGNORED_ON_RESUME
           and saved_args[key] != current_args[key]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='ddpm64')
    parser.add_argument('--out', default=os.path.join(REPO_ROOT, 'runs'))
    parser.add_argument('--steps', type=int, default=100_000)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--image-size', type=int, default=64)
    parser.add_argument('--cfg-dropout', type=float, default=0.1)
    parser.add_argument('--ema-decay', type=float, default=0.9999)
    parser.add_argument('--seed', type=int, default=612)
    parser.add_argument('--log-every', type=int, default=100)
    parser.add_argument('--sample-every', type=int, default=2000)
    parser.add_argument('--ckpt-every', type=int, default=5000)
    parser.add_argument('--guidance', type=float, default=2.0)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available()
                        else 'cpu')
    parser.add_argument('--resume')
    parser.add_argument('--no-cache', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()

    if args.smoke:
        # Short overfit on a few images. If the grid shows lesion-shaped
        # structure, conditioning and normalization are wired correctly.
        args.steps, args.log_every = 500, 50
        args.sample_every, args.ckpt_every = 100, 500
        args.name = args.name + '_smoke'

    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but not available. '
                           'Pass --device cpu to run without a GPU.')

    # Determined before the sampler and loader exist, so a resumed run can
    # be seeded and sized off of it rather than replaying the original
    # stream from the start.
    resume_state = None
    start_step = 0
    if args.resume:
        resume_state = torch.load(args.resume, map_location=args.device)
        start_step = resume_state['step'] + 1
        print(f"resuming {args.resume} at step {start_step}")
        mismatches = resume_mismatches(resume_state['args'], vars(args))
        if mismatches:
            print("warning: args differ from checkpoint")
            for key, before, after in mismatches:
                print(f"  {key}: checkpoint {before}, now {after}")
            print("continuing with current values")

    remaining_steps = max(args.steps - start_step, 0)
    if remaining_steps == 0:
        print(f"checkpoint is already at step {start_step - 1}, nothing "
              f"to do for --steps {args.steps}")
        return

    # Offsetting by start_step means a fresh run (start_step 0) seeds and
    # samples exactly as before, while a resumed run draws a fresh slice of
    # the data/noise/dropout stream instead of repeating what already ran.
    set_seed(args.seed + start_step)
    run_dir = os.path.join(args.out, args.name)
    os.makedirs(run_dir, exist_ok=True)

    data = HAM10000('train', image_size=args.image_size,
                    normalize='diffusion', cache=not args.no_cache)
    labels = torch.tensor(data.rows['dx'].map(CLASS_TO_INDEX).to_numpy())
    if args.smoke:
        labels = labels[:200]
        data = torch.utils.data.Subset(data, range(200))

    loader = DataLoader(
        data, batch_size=args.batch_size,
        sampler=balanced_sampler(labels, remaining_steps * args.batch_size,
                                 args.seed + start_step),
        num_workers=args.workers, pin_memory=True, drop_last=True)

    model = build_unet(args.image_size).to(args.device)
    parameters = sum(p.numel() for p in model.parameters())
    print(f"{parameters:,} parameters on {args.device}")

    scheduler = DDPMScheduler(**SCHEDULER_KWARGS)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ema = EMA(model, decay=args.ema_decay)

    if resume_state is not None:
        model.load_state_dict(resume_state['model'])
        ema.load_state_dict(resume_state['ema'])
        optimizer.load_state_dict(resume_state['optimizer'])

    log_path = os.path.join(run_dir, 'log.csv')
    if not os.path.exists(log_path):
        with open(log_path, 'w', newline='') as handle:
            csv.writer(handle).writerow(['step', 'loss', 'lr', 'seconds'])

    started = time.time()
    running, counted = 0.0, 0
    for step, (images, targets) in enumerate(loader, start=start_step):
        if step >= args.steps:
            break

        images = images.to(args.device, non_blocking=True)
        targets = drop_labels(targets.to(args.device), args.cfg_dropout,
                              NULL_CLASS)
        noise = torch.randn_like(images)
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps,
                                  (images.shape[0],), device=args.device)
        noisy = scheduler.add_noise(images, noise, timesteps)

        with torch.autocast('cuda', dtype=torch.bfloat16,
                            enabled=args.device.startswith('cuda')):
            predicted = model(noisy, timesteps, class_labels=targets).sample
            loss = torch.nn.functional.mse_loss(predicted.float(), noise)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        ema.update(model, step)

        running += loss.item()
        counted += 1

        if (step + 1) % args.log_every == 0:
            # Log a running mean: single-step diffusion loss mostly reflects
            # which timesteps were drawn, not progress.
            mean = running / counted
            running, counted = 0.0, 0
            elapsed = time.time() - started
            print(f"step {step + 1:>7} loss {mean:.4f} {elapsed:8.0f}s")
            with open(log_path, 'a', newline='') as handle:
                csv.writer(handle).writerow(
                    [step + 1, f"{mean:.6f}", args.lr, f"{elapsed:.1f}"])

        if (step + 1) % args.sample_every == 0:
            preview = build_unet(args.image_size).to(args.device).eval()
            ema.copy_to(preview)
            grid = sample_grid(preview, scheduler, args.device, args.guidance)
            save_image(grid, os.path.join(run_dir, f'samples_{step + 1}.png'),
                       nrow=4)
            del preview

        if (step + 1) % args.ckpt_every == 0 or step + 1 == args.steps:
            state = {'step': step, 'model': model.state_dict(),
                     'ema': ema.state_dict(),
                     'optimizer': optimizer.state_dict(), 'args': vars(args)}
            torch.save(state, os.path.join(run_dir, 'ckpt_last.pt'))
            torch.save(state, os.path.join(run_dir, f'ckpt_{step + 1}.pt'))

    print(f"done in {time.time() - started:.0f}s, run at {run_dir}")


if __name__ == '__main__':
    main()
