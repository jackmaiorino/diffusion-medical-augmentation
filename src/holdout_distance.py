"""Compare synthetic NN distances against train and held-out val references."""
import argparse
import os

import pandas
import torch

from dataset import SPLITS_CSV
from eval_fid_kid import REAL_DIR, list_pngs
from memorization import (build_lpips, lpips_distances, lpips_vectors,
                          write_rows)
from recalibrate_thresholds import lesion_codes, lesion_loo_nn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VAL_DIR = os.path.join(REPO_ROOT, 'data', 'eval', 'real64_val')
SEED = 612
COLUMNS = ['class', 'n_train', 'n_val', 'n_synth', 'synth_train_nn',
           'synth_train_nn_matched', 'synth_val_nn', 'frac_synth_closer',
           'val_train_nn_matched', 'val_val_nn', 'frac_val_closer']


def pool_paths(synth_dir, cls):
    """Class PNGs from the pool plus its quarantined _flagged sibling."""
    paths = list_pngs(os.path.join(synth_dir, cls))
    flagged = os.path.normpath(synth_dir) + '_flagged'
    return sorted(paths + list_pngs(os.path.join(flagged, cls)))


def matched_nn(distances, n_sub, seeds):
    """Per-seed NN vectors over size-matched column subsamples."""
    outs = []
    for s in range(seeds):
        gen = torch.Generator().manual_seed(SEED + s)
        cols = torch.randperm(distances.shape[1], generator=gen)[:n_sub]
        outs.append(distances[:, cols].min(dim=1).values)
    return outs


def matched_stats(distances, other_nn, n_sub, seeds):
    """(median matched NN, fraction closer than other_nn), seed-averaged."""
    subs = matched_nn(distances, n_sub, seeds)
    med = sum(s.median().item() for s in subs) / len(subs)
    frac = sum((s < other_nn).float().mean().item() for s in subs) / len(subs)
    return med, frac


def score_class(cls, synth_dir, lesion_of, model, device, batch_size, seeds):
    """One CSV row: is the pool closer to train than fresh data would be?"""
    train = list_pngs(os.path.join(REAL_DIR, cls))
    val = list_pngs(os.path.join(VAL_DIR, cls))
    synth = pool_paths(synth_dir, cls)

    vec_train = lpips_vectors(model, train, device, batch_size)
    vec_val = lpips_vectors(model, val, device, batch_size)
    vec_synth = lpips_vectors(model, synth, device, batch_size)

    d_synth_train = lpips_distances(vec_synth, vec_train, device)
    d_synth_val = lpips_distances(vec_synth, vec_val, device)
    d_val_train = lpips_distances(vec_val, vec_train, device)
    d_val_val = lpips_distances(vec_val, vec_val, device)

    synth_val_nn = d_synth_val.min(dim=1).values
    # baseline: real val images play the role of the generator
    val_val_nn = lesion_loo_nn(d_val_val, lesion_codes(val, lesion_of))

    synth_med, synth_frac = matched_stats(d_synth_train, synth_val_nn,
                                          len(val), seeds)
    val_med, val_frac = matched_stats(d_val_train, val_val_nn,
                                      len(val), seeds)
    return {
        'class': cls, 'n_train': len(train), 'n_val': len(val),
        'n_synth': len(synth),
        'synth_train_nn':
            round(d_synth_train.min(dim=1).values.median().item(), 5),
        'synth_train_nn_matched': round(synth_med, 5),
        'synth_val_nn': round(synth_val_nn.median().item(), 5),
        'frac_synth_closer': round(synth_frac, 4),
        'val_train_nn_matched': round(val_med, 5),
        'val_val_nn': round(val_val_nn.median().item(), 5),
        'frac_val_closer': round(val_frac, 4)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--synthetic-dir', required=True)
    parser.add_argument('--seeds', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available()
                        else 'cpu')
    args = parser.parse_args()

    if not os.path.isdir(VAL_DIR):
        raise RuntimeError(f"no val reference at {VAL_DIR}. "
                           "Run preprocessing/export_real64.py --split val.")

    splits = pandas.read_csv(SPLITS_CSV)
    lesion_of = dict(zip(splits['image_id'], splits['lesion_id']))
    model = build_lpips(args.device)

    rows = []
    for cls in sorted(os.listdir(REAL_DIR)):
        row = score_class(cls, args.synthetic_dir, lesion_of, model,
                          args.device, args.batch_size, args.seeds)
        rows.append(row)
        print(f"{cls:6s} synth->train {row['synth_train_nn_matched']:.4f} "
              f"synth->val {row['synth_val_nn']:.4f} "
              f"closer {100 * row['frac_synth_closer']:5.1f}% "
              f"(real baseline {100 * row['frac_val_closer']:5.1f}%)")

    name = os.path.basename(os.path.normpath(args.synthetic_dir))
    out = os.path.join(REPO_ROOT, 'reports', f'holdout_distance_{name}.csv')
    write_rows(out, COLUMNS, rows)
    print(f"wrote {out}")


if __name__ == '__main__':
    main()
