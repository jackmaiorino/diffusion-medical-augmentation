"""Live demo: sample from the DDPM and catch the rare-class copies."""
import argparse
import os
import tempfile

import matplotlib.pyplot as plt
import numpy
import torch
from PIL import Image

from dataset import CLASSES
from ddpm_model import build_sample_scheduler, build_unet
from eval_fid_kid import REAL_DIR, list_pngs
from memorization import build_lpips, lpips_distances, lpips_vectors
from sample_ddpm import generate

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CKPT = os.path.join(REPO_ROOT, 'runs', 'ddpm64', 'ckpt_100000.pt')


def sample_class(model, scheduler, dx, count, guidance, device, generator):
    """Generate count images of one class as uint8 HWC arrays."""
    return generate(model, scheduler, CLASSES.index(dx), count, guidance,
                    device, generator)


def nearest_real(samples, dx, lpips_model, device):
    """Nearest train image path and LPIPS distance for each sample."""
    real_paths = list_pngs(os.path.join(REAL_DIR, dx))
    # the path-based helpers score exactly like the memorization analysis did
    with tempfile.TemporaryDirectory() as workdir:
        sample_paths = []
        for i, image in enumerate(samples):
            path = os.path.join(workdir, f'sample_{i}.png')
            Image.fromarray(image).save(path)
            sample_paths.append(path)
        sample_vecs = lpips_vectors(lpips_model, sample_paths, device)
    real_vecs = lpips_vectors(lpips_model, real_paths, device)

    distances = lpips_distances(sample_vecs, real_vecs, device)
    nn = distances.min(dim=1)
    return ([real_paths[i] for i in nn.indices.tolist()], nn.values.tolist())


def show_or_save(fig, out_dir, name):
    """Save the figure when out_dir is set, otherwise block on the window."""
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, name + '.png')
        fig.savefig(path, dpi=130, bbox_inches='tight')
        print(f"figure written to {path}")
        plt.close(fig)
    else:
        plt.show()


def copy_catch(model, scheduler, lpips_model, dx, args, generator):
    """Show generated images above their nearest training images."""
    samples = sample_class(model, scheduler, dx, args.count, args.guidance,
                           args.device, generator)
    paths, distances = nearest_real(samples, dx, lpips_model, args.device)

    fig, axes = plt.subplots(2, args.count,
                             figsize=(args.count * 1.6, 3.8))
    for col in range(args.count):
        axes[0, col].imshow(samples[col])
        axes[1, col].imshow(Image.open(paths[col]))
        axes[1, col].set_title(f'LPIPS {distances[col]:.3f}', fontsize=8)
        for row in (0, 1):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
    axes[0, 0].set_ylabel('generated', fontsize=9)
    axes[1, 0].set_ylabel('nearest\ntrain image', fontsize=9)
    fig.suptitle(f'{dx}: generated (top) vs nearest training image (bottom)')
    fig.tight_layout()
    show_or_save(fig, args.out, f'copy_catch_{dx}')


def game(model, scheduler, dx, args, generator):
    """Real-or-generated guessing grid with a reveal after Enter."""
    half = args.count // 2
    synth = list(sample_class(model, scheduler, dx, half, args.guidance,
                              args.device, generator))
    pool = list_pngs(os.path.join(REAL_DIR, dx))
    rng = numpy.random.default_rng(args.seed)
    real = [numpy.asarray(Image.open(pool[i]))
            for i in rng.choice(len(pool), half, replace=False)]

    images = [(image, False) for image in synth] + \
             [(image, True) for image in real]
    rng.shuffle(images)

    for reveal in (False, True):
        fig, axes = plt.subplots(1, len(images),
                                 figsize=(len(images) * 1.6, 2.2))
        for i, (image, is_real) in enumerate(images):
            axes[i].imshow(image)
            axes[i].set_xticks([])
            axes[i].set_yticks([])
            if reveal:
                axes[i].set_title('real' if is_real else 'generated',
                                  fontsize=9,
                                  color='green' if is_real else 'red')
            else:
                axes[i].set_title(str(i + 1), fontsize=9)
        fig.suptitle(f'{dx}: which are real?' if not reveal
                     else f'{dx}: reveal')
        fig.tight_layout()
        show_or_save(fig, args.out, f'game_{dx}_{"reveal" if reveal else "ask"}')
        if not reveal and not args.out:
            input('press Enter to reveal...')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default=DEFAULT_CKPT)
    parser.add_argument('--classes', default='df,nv')
    parser.add_argument('--count', type=int, default=6)
    parser.add_argument('--guidance', type=float, default=2.0)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--seed', type=int, default=612)
    parser.add_argument('--game', action='store_true')
    parser.add_argument('--game-class', default='bkl')
    parser.add_argument('--out', help='save figures here instead of showing')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available()
                        else 'cpu')
    args = parser.parse_args()

    state = torch.load(args.ckpt, map_location=args.device)
    model = build_unet(state['args']['image_size']).to(args.device).eval()
    model.load_state_dict(state['ema'])
    scheduler = build_sample_scheduler()
    scheduler.set_timesteps(args.steps)
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    lpips_model = build_lpips(args.device)

    if args.game:
        game(model, scheduler, args.game_class, args, generator)

    for dx in args.classes.split(','):
        copy_catch(model, scheduler, lpips_model, dx, args, generator)


if __name__ == '__main__':
    main()
