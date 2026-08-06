"""Measure how well the pretrained SD autoencoder reconstructs dermoscopy."""
import argparse
import os

import matplotlib.pyplot as plt
import torch
from diffusers import AutoencoderKL

from dataset import CLASSES, HAM10000

VAE_NAME = 'stabilityai/sd-vae-ft-mse'
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(REPO_ROOT, 'reports')


def to_display(tensor):
    """[-1, 1] tensor to a [0, 1] HWC array for matplotlib."""
    return ((tensor.float() + 1) / 2).clamp(0, 1).permute(1, 2, 0).cpu().numpy()


def psnr(a, b):
    """PSNR in dB between two [-1, 1] tensors, over a 2.0-wide value range."""
    mse = torch.mean((a.float() - b.float()) ** 2)
    return float(20 * torch.log10(torch.tensor(2.0)) - 10 * torch.log10(mse))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--per-class', type=int, default=8)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available()
                        else 'cpu')
    args = parser.parse_args()

    dtype = torch.float16 if args.device.startswith('cuda') else torch.float32
    vae = AutoencoderKL.from_pretrained(VAE_NAME, torch_dtype=dtype)
    vae = vae.to(args.device).eval()

    os.makedirs(REPORTS_DIR, exist_ok=True)
    fig, axes = plt.subplots(2, len(CLASSES), figsize=(2.2 * len(CLASSES), 4.8))

    print(f"{'class':8s} {'n':>3s} {'PSNR dB':>9s}")
    scores = {}
    for col, dx in enumerate(CLASSES):
        ds = HAM10000('train', image_size=args.image_size, classes=[dx])
        n = min(args.per_class, len(ds))
        batch = torch.stack([ds[i][0] for i in range(n)]).to(args.device, dtype)

        with torch.no_grad():
            # Sample the posterior rather than taking its mean, matching how
            # latents are drawn during diffusion training.
            latents = vae.encode(batch).latent_dist.sample()
            recon = vae.decode(latents).sample

        scores[dx] = psnr(batch, recon)
        print(f"{dx:8s} {n:3d} {scores[dx]:9.2f}")

        axes[0, col].imshow(to_display(batch[0]))
        axes[0, col].set_title(dx, fontsize=10)
        axes[1, col].imshow(to_display(recon[0]))
        axes[1, col].set_xlabel(f"{scores[dx]:.1f} dB", fontsize=9)
        for row in (0, 1):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

    axes[0, 0].set_ylabel('real', fontsize=10)
    axes[1, 0].set_ylabel('reconstructed', fontsize=10)
    mean_psnr = sum(scores.values()) / len(scores)
    fig.suptitle(f"SD autoencoder round-trip at {args.image_size}px, "
                 f"mean {mean_psnr:.1f} dB", fontsize=11)
    fig.tight_layout()

    out = os.path.join(REPORTS_DIR, f'vae_roundtrip_{args.image_size}.png')
    fig.savefig(out, dpi=130, bbox_inches='tight')
    print(f"\nmean PSNR {mean_psnr:.2f} dB")
    print(f"figure written to {out}")


if __name__ == '__main__':
    main()
