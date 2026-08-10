"""Generate class-conditional samples from a trained DDPM checkpoint."""
import argparse
import os

import matplotlib.pyplot as plt
import torch
from PIL import Image

from dataset import CLASSES
from ddpm_model import NULL_CLASS, build_sample_scheduler, build_unet

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def apply_guidance(conditional, unconditional, weight):
    """Classifier-free guidance: weight 1.0 leaves the prediction unchanged."""
    return unconditional + weight * (conditional - unconditional)


def resolve_image_size(cli_value, checkpoint_value):
    """Use the checkpoint's image size unless the CLI gives a conflicting one."""
    if cli_value is None:
        return checkpoint_value
    if cli_value != checkpoint_value:
        raise ValueError(
            f"--image-size {cli_value} does not match the checkpoint, "
            f"which was trained at {checkpoint_value}. Omit --image-size "
            "to use the checkpoint's value.")
    return cli_value


@torch.no_grad()
def generate(model, scheduler, label, count, guidance, device, generator):
    """Sample count images of one class, returned as uint8 HWC arrays."""
    labels = torch.full((count,), label, device=device, dtype=torch.long)
    null = torch.full_like(labels, NULL_CLASS)
    size = model.config.sample_size
    latents = torch.randn(count, 3, size, size, device=device,
                          generator=generator)

    for timestep in scheduler.timesteps:
        with torch.autocast('cuda', dtype=torch.bfloat16,
                            enabled=device.startswith('cuda')):
            predicted = model(torch.cat([latents, latents]), timestep,
                              class_labels=torch.cat([labels, null])).sample
        conditional, unconditional = predicted.float().chunk(2)
        guided = apply_guidance(conditional, unconditional, guidance)
        latents = scheduler.step(guided, timestep, latents).prev_sample

    images = ((latents.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    return images.permute(0, 2, 3, 1).cpu().numpy()


def save_grid(model, scheduler, args, generator, per_class=6):
    """Write one labeled figure with a row of samples per class."""
    fig, axes = plt.subplots(len(CLASSES), per_class,
                             figsize=(per_class * 1.1, len(CLASSES) * 1.2))

    for row, name in enumerate(CLASSES):
        images = generate(model, scheduler, CLASSES.index(name), per_class,
                          args.guidance, args.device, generator)
        for col in range(per_class):
            axes[row, col].imshow(images[col])
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
        axes[row, 0].set_ylabel(name, fontsize=9, rotation=0, ha='right',
                                va='center')

    fig.suptitle(f"DDPM samples at guidance {args.guidance}", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130, bbox_inches='tight')
    print(f"figure written to {args.out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    # a .png path in --grid mode, a directory otherwise
    parser.add_argument('--out', required=True)
    parser.add_argument('--grid', action='store_true')
    parser.add_argument('--classes', default='df,vasc,akiec')
    parser.add_argument('--per-class', type=int, default=1000)
    parser.add_argument('--guidance', type=float, default=2.0)
    parser.add_argument('--steps', type=int, default=50)
    # guidance doubles the batch through the model, 64 spills VRAM
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=612)
    parser.add_argument('--image-size', type=int, default=None)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available()
                        else 'cpu')
    args = parser.parse_args()

    state = torch.load(args.ckpt, map_location=args.device)
    args.image_size = resolve_image_size(args.image_size,
                                         state['args']['image_size'])
    model = build_unet(args.image_size).to(args.device).eval()
    # the EMA weights sample cleaner than the live ones
    model.load_state_dict(state['ema'])

    scheduler = build_sample_scheduler()
    scheduler.set_timesteps(args.steps)
    generator = torch.Generator(device=args.device).manual_seed(args.seed)

    if args.grid:
        save_grid(model, scheduler, args, generator)
        return

    for name in args.classes.split(','):
        label = CLASSES.index(name)
        folder = os.path.join(args.out, name)
        os.makedirs(folder, exist_ok=True)

        written = 0
        while written < args.per_class:
            count = min(args.batch_size, args.per_class - written)
            for image in generate(model, scheduler, label, count,
                                  args.guidance, args.device, generator):
                Image.fromarray(image).save(
                    os.path.join(folder, f'{name}_{written:05d}.png'))
                written += 1
            print(f"{name}: {written}/{args.per_class}")

    print(f"wrote samples to {args.out}")


if __name__ == '__main__':
    main()
