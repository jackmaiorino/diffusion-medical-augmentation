"""Score saved DDPM checkpoints by diffusion loss on the validation split."""
import argparse
import csv
import os
import re

import torch

from dataset import HAM10000
from ddpm_model import NULL_CLASS, build_train_scheduler, build_unet
from train_ddpm import diffusion_loss, drop_labels

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def list_checkpoints(run_dir):
    """Numbered (step, path) checkpoint pairs in a run, sorted by step."""
    found = []
    for name in os.listdir(run_dir):
        match = re.fullmatch(r'ckpt_(\d+)\.pt', name)
        if match:
            found.append((int(match.group(1)), os.path.join(run_dir, name)))
    if not found:
        raise RuntimeError(f"no numbered checkpoints under {run_dir}")
    return sorted(found)


def draw_eval_batches(images, labels, scheduler, cfg_dropout, repeats,
                      batch_size, seed):
    """Yield eval batches whose draws are identical on every call."""
    # one generator seeded here, so every checkpoint sees the same
    # noise, timesteps and label drops and the curve is comparable
    generator = torch.Generator().manual_seed(seed)
    for _ in range(repeats):
        for start in range(0, len(images), batch_size):
            batch = images[start:start + batch_size]
            targets = drop_labels(labels[start:start + batch_size],
                                  cfg_dropout, NULL_CLASS, generator)
            noise = torch.randn(batch.shape, generator=generator)
            timesteps = torch.randint(
                0, scheduler.config.num_train_timesteps,
                (batch.shape[0],), generator=generator)
            yield batch, targets, noise, timesteps


@torch.no_grad()
def score_checkpoint(model, scheduler, batches, device, amp):
    """Mean diffusion loss of one loaded model over the eval batches."""
    total, count = 0.0, 0
    for images, targets, noise, timesteps in batches:
        images, targets = images.to(device), targets.to(device)
        noise, timesteps = noise.to(device), timesteps.to(device)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
            loss = diffusion_loss(model, scheduler, images, targets, noise,
                                  timesteps)
        total += loss.item() * images.shape[0]
        count += images.shape[0]
    return total / count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run',
                        default=os.path.join(REPO_ROOT, 'runs', 'ddpm64'))
    parser.add_argument('--out',
                        default=os.path.join(REPO_ROOT, 'reports',
                                             'ddpm_val_loss.csv'))
    parser.add_argument('--repeats', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=612)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    checkpoints = list_checkpoints(args.run)
    first = torch.load(checkpoints[0][1], map_location='cpu')
    image_size = first['args']['image_size']
    cfg_dropout = first['args']['cfg_dropout']

    dataset = HAM10000('val', image_size=image_size, normalize='diffusion')
    images = torch.stack([dataset[i][0] for i in range(len(dataset))])
    labels = torch.tensor([dataset[i][1] for i in range(len(dataset))])

    scheduler = build_train_scheduler()
    model = build_unet(image_size).to(args.device).eval()
    amp = args.device.startswith('cuda')

    with open(args.out, 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['step', 'val_loss'])
        for step, path in checkpoints:
            state = torch.load(path, map_location='cpu')
            if state['args']['image_size'] != image_size:
                raise RuntimeError(f"{path} is {state['args']['image_size']}px"
                                   f", run started at {image_size}px")
            # the raw weights, not the EMA, match the logged training loss
            model.load_state_dict(state['model'])
            batches = draw_eval_batches(images, labels, scheduler,
                                        cfg_dropout, args.repeats,
                                        args.batch_size, args.seed)
            score = score_checkpoint(model, scheduler, batches, args.device,
                                     amp)
            writer.writerow([step, f"{score:.6f}"])
            print(f"step {step}: val loss {score:.4f}")
    print(f"wrote {args.out}")


if __name__ == '__main__':
    main()
