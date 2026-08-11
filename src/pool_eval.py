"""Fidelity, diversity and duplication for filtered synthetic pools."""
import argparse
import json
import os
import shutil
import tempfile

import pandas
import torch

from dataset import SPLITS_CSV
from eval_fid_kid import compute_metrics, kid_subset_size, list_pngs, REAL_DIR
from holdout_distance import VAL_DIR, pool_paths
from memorization import (build_lpips, loo_nn, lpips_distances,
                          lpips_vectors, write_rows)
from recalibrate_thresholds import (LESION_THRESHOLDS_PATH, lesion_codes,
                                    lesion_loo_nn, reflag)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 612
SAMPLE_CAP = 500
COLUMNS = ['class', 'pool', 'n', 'kid_val_mean', 'kid_val_std',
           'pairwise_lpips', 'near_dup_frac']


def accepted_paths(detail, cls, thresholds, synth_dir):
    """Files for one class that survive the lesion-excluded thresholds."""
    group = detail[detail['class'] == cls]
    keep = group.loc[~reflag(group, thresholds[cls]), 'file']
    flagged_dir = os.path.normpath(synth_dir) + '_flagged'
    paths = []
    for name in keep:
        for root in (synth_dir, flagged_dir):
            path = os.path.join(root, cls, name)
            if os.path.exists(path):
                paths.append(path)
                break
    return sorted(paths)


def sample(paths, cap, seed):
    """At most cap paths, drawn without replacement."""
    if len(paths) <= cap:
        return list(paths)
    gen = torch.Generator().manual_seed(seed)
    picked = torch.randperm(len(paths), generator=gen)[:cap]
    return [paths[i] for i in picked.tolist()]


def kid_vs_val(paths, val_folder, seed):
    """KID (x1000 mean, std) of a file pool against the val class folder."""
    n_val = len(list_pngs(val_folder))
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as scratch:
        for path in paths:
            shutil.copy(path, scratch)
        _, kid_mean, kid_std = compute_metrics(
            scratch, val_folder, kid_subset_size(len(paths), n_val), seed)
    return 1000 * kid_mean, 1000 * kid_std


def diversity_stats(vectors, device, threshold, codes=None):
    """Mean pairwise LPIPS and the under-threshold NN fraction."""
    distances = lpips_distances(vectors, vectors, device)
    n = distances.shape[0]
    mean_pairwise = (distances.sum() / (n * (n - 1))).item()
    nn = lesion_loo_nn(distances, codes) if codes is not None \
        else loo_nn(distances)
    return mean_pairwise, (nn < threshold).float().mean().item()


def pool_row(cls, pool, paths, picked, codes, val_folder, model, device,
             batch_size, threshold):
    """One CSV row of metrics for a named pool of images."""
    vectors = lpips_vectors(model, picked, device, batch_size)
    pairwise, dup_frac = diversity_stats(vectors, device, threshold, codes)
    kid_mean, kid_std = kid_vs_val(paths, val_folder, SEED)
    return {'class': cls, 'pool': pool, 'n': len(paths),
            'kid_val_mean': round(kid_mean, 3),
            'kid_val_std': round(kid_std, 3),
            'pairwise_lpips': round(pairwise, 5),
            'near_dup_frac': round(dup_frac, 4)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--synthetic-dir', required=True)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available()
                        else 'cpu')
    args = parser.parse_args()

    name = os.path.basename(os.path.normpath(args.synthetic_dir))
    detail = pandas.read_csv(os.path.join(
        REPO_ROOT, 'data', 'eval', f'memorization_{name}_detail.csv'))
    with open(LESION_THRESHOLDS_PATH) as handle:
        thresholds = json.load(handle)['thresholds']

    splits = pandas.read_csv(SPLITS_CSV)
    lesion_of = dict(zip(splits['image_id'], splits['lesion_id']))
    model = build_lpips(args.device)

    rows = []
    for cls in sorted(os.listdir(REAL_DIR)):
        val_folder = os.path.join(VAL_DIR, cls)
        threshold = thresholds[cls]['lpips']
        pools = [
            ('real_train', list_pngs(os.path.join(REAL_DIR, cls))),
            ('all', pool_paths(args.synthetic_dir, cls)),
            ('accepted',
             accepted_paths(detail, cls, thresholds, args.synthetic_dir))]
        for pool, paths in pools:
            picked = sample(paths, SAMPLE_CAP, SEED)
            # real images need lesion exclusion or siblings count as dups
            codes = lesion_codes(picked, lesion_of) \
                if pool == 'real_train' else None
            row = pool_row(cls, pool, paths, picked, codes, val_folder,
                           model, args.device, args.batch_size, threshold)
            rows.append(row)
            print(f"{cls:6s} {pool:10s} n {row['n']:4d} "
                  f"kid {row['kid_val_mean']:8.3f} "
                  f"pairwise {row['pairwise_lpips']:.4f} "
                  f"dup {100 * row['near_dup_frac']:5.1f}%")

    out = os.path.join(REPO_ROOT, 'reports', f'pool_eval_{name}.csv')
    write_rows(out, COLUMNS, rows)
    print(f"wrote {out}")


if __name__ == '__main__':
    main()
