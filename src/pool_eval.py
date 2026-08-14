"""Fidelity, diversity and duplication for filtered synthetic pools."""
import argparse
import json
import os
import shutil
import tempfile

import numpy
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
DIVERSITY_REPEATS = 20
COLUMNS = ['class', 'pool', 'n', 'kid_val_mean', 'kid_val_std',
           'diversity_n', 'diversity_repeats', 'pairwise_lpips',
           'near_dup_frac', 'null_near_dup_frac']


def accepted_paths(detail, cls, thresholds, synth_dir):
    """Files for one class that survive the lesion-excluded thresholds."""
    group = detail[detail['class'] == cls]
    keep = group.loc[~reflag(group, thresholds[cls]), 'file']
    flagged_dir = os.path.normpath(synth_dir) + '_flagged'
    paths = []
    for name in keep:
        matches = []
        for root in (synth_dir, flagged_dir):
            path = os.path.join(root, cls, name)
            if os.path.isfile(path):
                matches.append(path)
        if len(matches) != 1:
            raise RuntimeError(
                f'accepted image {cls}/{name} has {len(matches)} sources; '
                'expected exactly one across active and flagged trees')
        paths.append(matches[0])
    return sorted(paths)


def validated_pool_paths(detail, cls, synth_dir):
    """Return the exact complete pool described by the detail CSV."""
    expected = detail.loc[detail['class'] == cls, 'file'].tolist()
    if not expected:
        raise RuntimeError(f'memorization detail has no rows for class {cls}')
    if len({name.casefold() for name in expected}) != len(expected):
        raise RuntimeError(f'memorization detail duplicates files for {cls}')

    paths = pool_paths(synth_dir, cls)
    actual = [os.path.basename(path) for path in paths]
    if len({name.casefold() for name in actual}) != len(actual):
        raise RuntimeError(f'synthetic pool duplicates files for {cls}')

    expected_folded = {name.casefold() for name in expected}
    actual_folded = {name.casefold() for name in actual}
    missing = sorted(expected_folded - actual_folded)
    unexpected = sorted(actual_folded - expected_folded)
    if missing or unexpected:
        raise RuntimeError(
            f'synthetic pool inventory mismatch for {cls}: '
            f'missing={missing[:3]}, unexpected={unexpected[:3]}')
    return paths


def sample_indices(total, count, seed):
    """Deterministic indices for a without-replacement subset."""
    if count > total:
        raise ValueError(f'cannot draw {count} rows from a pool of {total}')
    if count == total:
        return torch.arange(total)
    gen = torch.Generator().manual_seed(seed)
    return torch.randperm(total, generator=gen)[:count]


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


def matched_diversity(real_vectors, real_codes, pool_vectors, device,
                      pool_codes=None, cap=SAMPLE_CAP,
                      repeats=DIVERSITY_REPEATS, seed=SEED):
    """Diversity and duplicate rate under a size-matched real null.

    Each repeat draws equally sized real and pool subsets. The duplicate
    threshold is recalibrated on that repeat's real subset with whole-lesion
    exclusion, so a 30-image accepted pool is not compared with a threshold
    derived from hundreds or thousands of candidate neighbors.
    """
    n_eval = min(len(real_vectors), len(pool_vectors), cap)
    if n_eval < 2:
        raise ValueError('diversity evaluation requires at least two images')
    if repeats < 1:
        raise ValueError('repeats must be positive')

    pairwise, duplicate_rates, null_rates = [], [], []
    for offset in range(repeats):
        real_idx = sample_indices(len(real_vectors), n_eval, seed + offset)
        pool_idx = sample_indices(len(pool_vectors), n_eval, seed + offset)

        real_subset = real_vectors[real_idx]
        real_subset_codes = real_codes[real_idx]
        real_distances = lpips_distances(real_subset, real_subset, device)
        real_nn = lesion_loo_nn(real_distances, real_subset_codes)
        finite = real_nn[torch.isfinite(real_nn)]
        if len(finite) != n_eval:
            raise RuntimeError(
                'a size-matched real subset has no cross-lesion neighbor')
        threshold = float(numpy.percentile(finite.numpy(), 5))
        null_rates.append((finite < threshold).float().mean().item())

        subset_codes = pool_codes[pool_idx] if pool_codes is not None else None
        mean, rate = diversity_stats(pool_vectors[pool_idx], device,
                                     threshold, subset_codes)
        pairwise.append(mean)
        duplicate_rates.append(rate)

    return {
        'diversity_n': n_eval,
        'diversity_repeats': repeats,
        'pairwise_lpips': float(numpy.mean(pairwise)),
        'near_dup_frac': float(numpy.mean(duplicate_rates)),
        'null_near_dup_frac': float(numpy.mean(null_rates)),
    }


def pool_row(cls, pool, paths, real_vectors, real_codes, model, val_folder,
             device, batch_size, pool_codes=None):
    """One CSV row of metrics for a named pool of images."""
    vectors = real_vectors if pool == 'real_train' else \
        lpips_vectors(model, paths, device, batch_size)
    diversity = matched_diversity(real_vectors, real_codes, vectors, device,
                                  pool_codes)
    kid_mean, kid_std = kid_vs_val(paths, val_folder, SEED)
    return {'class': cls, 'pool': pool, 'n': len(paths),
            'kid_val_mean': round(kid_mean, 3),
            'kid_val_std': round(kid_std, 3),
            'diversity_n': diversity['diversity_n'],
            'diversity_repeats': diversity['diversity_repeats'],
            'pairwise_lpips': round(diversity['pairwise_lpips'], 5),
            'near_dup_frac': round(diversity['near_dup_frac'], 4),
            'null_near_dup_frac':
                round(diversity['null_near_dup_frac'], 4)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--synthetic-dir', required=True)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available()
                        else 'cpu')
    args = parser.parse_args()

    name = os.path.basename(os.path.normpath(args.synthetic_dir))
    out = os.path.join(REPO_ROOT, 'reports', f'pool_eval_{name}.csv')
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
        real_paths = list_pngs(os.path.join(REAL_DIR, cls))
        real_vectors = lpips_vectors(model, real_paths, args.device,
                                     args.batch_size)
        real_codes = lesion_codes(real_paths, lesion_of)
        pools = [
            ('real_train', real_paths),
            ('all', validated_pool_paths(detail, cls, args.synthetic_dir)),
            ('accepted',
             accepted_paths(detail, cls, thresholds, args.synthetic_dir))]
        for pool, paths in pools:
            # real images need lesion exclusion or siblings count as dups
            codes = lesion_codes(paths, lesion_of) \
                if pool == 'real_train' else None
            row = pool_row(cls, pool, paths, real_vectors, real_codes, model,
                           val_folder, args.device, args.batch_size, codes)
            rows.append(row)
            print(f"{cls:6s} {pool:10s} n {row['n']:4d} "
                  f"eval {row['diversity_n']:4d} "
                  f"kid {row['kid_val_mean']:8.3f} "
                  f"pairwise {row['pairwise_lpips']:.4f} "
                  f"dup {100 * row['near_dup_frac']:5.1f}% "
                  f"(null {100 * row['null_near_dup_frac']:4.1f}%)")

    write_rows(out, COLUMNS, rows)
    print(f"wrote {out}")


if __name__ == '__main__':
    main()
