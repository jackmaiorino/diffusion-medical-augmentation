"""Recalibrate memorization thresholds by excluding same-lesion siblings."""
import argparse
import glob
import json
import os
import re

import pandas
import torch

from dataset import SPLITS_CSV, splits_sha1
from eval_fid_kid import REAL_DIR, list_pngs
from memorization import (build_inception, build_lpips, class_thresholds,
                          cosine_distances, flag_mask, inception_vectors,
                          lpips_distances, lpips_vectors, save_thresholds,
                          write_rows)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(REPO_ROOT, 'data', 'eval')
OLD_THRESHOLDS_PATH = os.path.join(EVAL_DIR, 'memorization_thresholds.json')
LESION_THRESHOLDS_PATH = os.path.join(EVAL_DIR,
                                      'memorization_thresholds_lesion.json')
RECAL_COLUMNS = ['run', 'class', 'n_synth', 'flagged_fraction',
                 'flagged_fraction_lesion', 'lpips_threshold',
                 'lpips_threshold_lesion', 'inception_threshold',
                 'inception_threshold_lesion']
SOURCE_COLUMNS = ['class', 'n_flagged', 'distinct_sources', 'train_images',
                  'distinct_lesions', 'train_lesions', 'top_source_share']


def lesion_codes(paths, lesion_of):
    """Integer lesion codes aligned with the given image paths."""
    ids = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    lesions = pandas.Index(lesion_of[i] for i in ids)
    return torch.tensor(pandas.factorize(lesions)[0])


def lesion_loo_nn(distances, codes):
    """Row-wise NN distance, ignoring every column from the same lesion."""
    masked = distances.clone()
    masked[codes.unsqueeze(1) == codes.unsqueeze(0)] = float('inf')
    return masked.min(dim=1).values


def compute_lesion_thresholds(real_dir, lesion_of, device, batch_size):
    """Per-class 5th-percentile thresholds under whole-lesion exclusion."""
    lpips_model = build_lpips(device)
    inception_model, transform = build_inception(device)
    thresholds = {}
    for cls in sorted(os.listdir(real_dir)):
        paths = list_pngs(os.path.join(real_dir, cls))
        codes = lesion_codes(paths, lesion_of)
        vec_l = lpips_vectors(lpips_model, paths, device, batch_size)
        vec_i = inception_vectors(inception_model, transform, paths, device,
                                  batch_size)
        thresholds[cls] = class_thresholds(
            lesion_loo_nn(lpips_distances(vec_l, vec_l, device), codes),
            lesion_loo_nn(cosine_distances(vec_i, vec_i, device), codes))
    return thresholds


def reflag(group, thr):
    """Boolean flags for one class's detail rows under new thresholds."""
    return flag_mask(group['lpips_nn'], group['inception_nn'],
                     thr['lpips'], thr['inception'])


def recal_rows(detail_paths, old, new):
    """Old-vs-new flagged fractions for every run and class."""
    rows = []
    for path in detail_paths:
        run = re.fullmatch(r'memorization_(.+)_detail\.csv',
                           os.path.basename(path)).group(1)
        detail = pandas.read_csv(path)
        for cls, group in detail.groupby('class'):
            rows.append({
                'run': run, 'class': cls, 'n_synth': len(group),
                'flagged_fraction': round(float(group['flagged'].mean()), 4),
                'flagged_fraction_lesion':
                    round(float(reflag(group, new[cls]).mean()), 4),
                'lpips_threshold': round(old[cls]['lpips'], 6),
                'lpips_threshold_lesion': round(new[cls]['lpips'], 6),
                'inception_threshold': round(old[cls]['inception'], 6),
                'inception_threshold_lesion':
                    round(new[cls]['inception'], 6)})
    return rows


def source_rows(detail, thresholds, splits):
    """How many distinct train images and lesions the flags trace back to."""
    train = splits[splits['split'] == 'train']
    rows = []
    for cls, group in detail.groupby('class'):
        hits = group.loc[reflag(group, thresholds[cls]), 'lpips_nn_id']
        cls_train = train[train['dx'] == cls]
        lesion_of = dict(zip(cls_train['image_id'], cls_train['lesion_id']))
        rows.append({
            'class': cls, 'n_flagged': len(hits),
            'distinct_sources': int(hits.nunique()),
            'train_images': len(cls_train),
            'distinct_lesions': int(hits.map(lesion_of).nunique()),
            'train_lesions': int(cls_train['lesion_id'].nunique()),
            'top_source_share': round(float(
                hits.value_counts().iloc[0] / len(hits)), 4)
            if len(hits) else 0.0})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available()
                        else 'cpu')
    args = parser.parse_args()

    splits = pandas.read_csv(SPLITS_CSV)
    lesion_of = dict(zip(splits['image_id'], splits['lesion_id']))
    with open(OLD_THRESHOLDS_PATH) as handle:
        old = json.load(handle)['thresholds']

    new = compute_lesion_thresholds(REAL_DIR, lesion_of, args.device,
                                    args.batch_size)
    save_thresholds(LESION_THRESHOLDS_PATH, new, splits_sha1(SPLITS_CSV))
    for cls in sorted(new):
        print(f"{cls:6s} lpips {old[cls]['lpips']:.6f} -> "
              f"{new[cls]['lpips']:.6f}  inception "
              f"{old[cls]['inception']:.6f} -> {new[cls]['inception']:.6f}")

    detail_paths = sorted(glob.glob(
        os.path.join(EVAL_DIR, 'memorization_*_detail.csv')))
    recal_path = os.path.join(REPO_ROOT, 'reports',
                              'memorization_lesion_recal.csv')
    write_rows(recal_path, RECAL_COLUMNS, recal_rows(detail_paths, old, new))
    print(f"wrote {recal_path}")

    main_detail = os.path.join(EVAL_DIR, 'memorization_ddpm64_detail.csv')
    source_path = os.path.join(REPO_ROOT, 'reports',
                               'memorization_sources_ddpm64.csv')
    write_rows(source_path, SOURCE_COLUMNS,
               source_rows(pandas.read_csv(main_detail), new, splits))
    print(f"wrote {source_path}")


if __name__ == '__main__':
    main()
