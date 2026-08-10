"""Per-class FID and KID for a synthetic image tree against the real reference."""
import argparse
import csv
import os
import random
import shutil
import tempfile

import torch
import torch_fidelity

from dataset import CLASSES

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_DIR = os.path.join(REPO_ROOT, 'data', 'eval', 'real64')
KID_SUBSETS = 100
CSV_COLUMNS = ['class', 'n_real', 'n_synth', 'fid', 'kid_mean', 'kid_std',
               'fid_floor', 'kid_floor_mean', 'kid_floor_std']


def list_pngs(folder):
    """Sorted .png paths in folder, or [] if the folder is missing."""
    if not os.path.isdir(folder):
        return []
    return sorted(os.path.abspath(os.path.join(folder, name))
                  for name in os.listdir(folder) if name.endswith('.png'))


def discover_classes(real_dir, synth_dir):
    """Partition CLASSES into (scored, skipped) and list unknown synth subdirs."""
    scored, skipped = [], []
    for name in CLASSES:
        real = list_pngs(os.path.join(real_dir, name))
        synth = list_pngs(os.path.join(synth_dir, name))
        (scored if real and synth else skipped).append(name)
    unknown = sorted(
        name for name in os.listdir(synth_dir)
        if os.path.isdir(os.path.join(synth_dir, name))
        and name not in CLASSES)
    return scored, skipped, unknown


def half_split(paths, seed):
    """Shuffle paths with the seed and split at len // 2."""
    shuffled = list(paths)
    random.Random(seed).shuffle(shuffled)
    middle = len(shuffled) // 2
    return shuffled[:middle], shuffled[middle:]


def kid_subset_size(n_real, n_synth):
    """KID subset size that survives the smallest class."""
    return min(n_real, n_synth, 100)


def write_csv(path, rows):
    """Write score rows to path, overwriting."""
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def compute_metrics(dir_a, dir_b, subset_size, seed):
    """FID and KID between two image directories via torch-fidelity."""
    scores = torch_fidelity.calculate_metrics(
        input1=dir_a, input2=dir_b, fid=True, kid=True,
        kid_subset_size=subset_size, kid_subsets=KID_SUBSETS,
        rng_seed=seed, cuda=torch.cuda.is_available(), verbose=False)
    return (scores['frechet_inception_distance'],
            scores['kernel_inception_distance_mean'],
            scores['kernel_inception_distance_std'])


def floor_metrics(real_paths, seed, metrics=compute_metrics):
    """Real-vs-real scores between seeded halves of one class."""
    half_a, half_b = half_split(real_paths, seed)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as scratch:
        folders = []
        for name, half in (('a', half_a), ('b', half_b)):
            folder = os.path.join(scratch, name)
            os.makedirs(folder)
            for path in half:
                shutil.copy(path, folder)
            folders.append(folder)
        return metrics(folders[0], folders[1],
                       kid_subset_size(len(half_a), len(half_b)), seed)


def score_class(name, real_dir, synth_dir, seed, metrics=compute_metrics):
    """One CSV row of scores for a single class, KID reported x1000."""
    real = list_pngs(os.path.join(real_dir, name))
    synth = list_pngs(os.path.join(synth_dir, name))
    fid, kid_mean, kid_std = metrics(
        os.path.join(real_dir, name), os.path.join(synth_dir, name),
        kid_subset_size(len(real), len(synth)), seed)
    fid_floor, kid_floor_mean, kid_floor_std = floor_metrics(
        real, seed, metrics)
    return {'class': name, 'n_real': len(real), 'n_synth': len(synth),
            'fid': round(fid, 2),
            'kid_mean': round(1000 * kid_mean, 3),
            'kid_std': round(1000 * kid_std, 3),
            'fid_floor': round(fid_floor, 2),
            'kid_floor_mean': round(1000 * kid_floor_mean, 3),
            'kid_floor_std': round(1000 * kid_floor_std, 3)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--synthetic-dir', required=True)
    parser.add_argument('--real-dir', default=REAL_DIR)
    parser.add_argument('--out')
    parser.add_argument('--seed', type=int, default=612)
    args = parser.parse_args()

    if not os.path.isdir(args.synthetic_dir):
        parser.error(f"{args.synthetic_dir} is not a directory")
    if args.out is None:
        name = os.path.basename(os.path.normpath(args.synthetic_dir))
        args.out = os.path.join(REPO_ROOT, 'reports', f'fid_kid_{name}.csv')

    scored, skipped, unknown = discover_classes(args.real_dir,
                                                args.synthetic_dir)
    for name in unknown:
        print(f"warning: {name}/ is not a HAM10000 class, ignored")
    for name in skipped:
        print(f"{name}: missing or empty on one side, skipped")

    rows = []
    for name in scored:
        row = score_class(name, args.real_dir, args.synthetic_dir, args.seed)
        rows.append(row)
        print(f"{name:6s} fid {row['fid']:7.2f} "
              f"(floor {row['fid_floor']:7.2f})  "
              f"kid {row['kid_mean']:7.3f} "
              f"(floor {row['kid_floor_mean']:7.3f})")

    write_csv(args.out, rows)
    print(f"wrote {args.out}")


if __name__ == '__main__':
    main()
