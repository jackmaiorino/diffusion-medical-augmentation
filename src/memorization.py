"""Nearest-neighbor memorization check for synthetic images."""
import argparse
import csv
import json
import os
import shutil

import lpips
import numpy
import torch
from PIL import Image
from torchvision.models import Inception_V3_Weights, inception_v3

from dataset import SPLITS_CSV, splits_sha1
from eval_fid_kid import REAL_DIR, discover_classes, list_pngs

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THRESHOLDS_PATH = os.path.join(REPO_ROOT, 'data', 'eval',
                               'memorization_thresholds.json')
PERCENTILE = 5
SUMMARY_COLUMNS = ['class', 'n_synth', 'n_flagged', 'flagged_fraction',
                   'lpips_threshold', 'inception_threshold']
DETAIL_COLUMNS = ['class', 'file', 'lpips_nn', 'lpips_nn_id', 'inception_nn',
                  'inception_nn_id', 'flagged']


def build_lpips(device):
    """The LPIPS alex model in eval mode."""
    return lpips.LPIPS(net='alex', verbose=False).to(device).eval()


def build_inception(device):
    """InceptionV3 pool-feature extractor and its preprocessing transform."""
    weights = Inception_V3_Weights.IMAGENET1K_V1
    model = inception_v3(weights=weights)
    model.fc = torch.nn.Identity()
    return model.to(device).eval(), weights.transforms()


def load_batch(paths):
    """Stack PNGs as an NCHW float tensor in [0, 1]."""
    arrays = [numpy.asarray(Image.open(p).convert('RGB')) for p in paths]
    stacked = torch.from_numpy(numpy.stack(arrays))
    return stacked.permute(0, 3, 1, 2).float() / 255.0


@torch.no_grad()
def lpips_vectors(model, paths, device, batch_size=256):
    """Vectors whose pairwise squared L2 equals the LPIPS distance."""
    chunks = []
    for start in range(0, len(paths), batch_size):
        batch = load_batch(paths[start:start + batch_size]).to(device)
        scaled = model.scaling_layer(batch * 2 - 1)
        features = model.net.forward(scaled)
        parts = []
        for layer, feats in enumerate(features):
            normed = lpips.normalize_tensor(feats)
            # fold the linear weights and spatial mean into the vector, so
            # distance becomes plain squared L2 over the concatenation
            weight = model.lins[layer].model[-1].weight.squeeze()
            h, w = normed.shape[2], normed.shape[3]
            parts.append((normed * weight.sqrt().view(1, -1, 1, 1)
                          / (h * w) ** 0.5).flatten(1))
        chunks.append(torch.cat(parts, dim=1).cpu())
    return torch.cat(chunks)


@torch.no_grad()
def inception_vectors(model, transform, paths, device, batch_size=256):
    """Unit-normalized InceptionV3 pool features, one row per image."""
    chunks = []
    for start in range(0, len(paths), batch_size):
        batch = transform(load_batch(paths[start:start + batch_size]))
        features = model(batch.to(device))
        chunks.append(torch.nn.functional.normalize(features, dim=1).cpu())
    return torch.cat(chunks)


def lpips_distances(a, b, device, chunk=2048):
    """Pairwise LPIPS distances from flattened vectors."""
    rows = []
    b_dev = b.to(device)
    for start in range(0, len(a), chunk):
        part = a[start:start + chunk].to(device)
        rows.append((torch.cdist(part, b_dev) ** 2).cpu())
    return torch.cat(rows)


def cosine_distances(a, b, device, chunk=4096):
    """Pairwise cosine distances between unit-normalized vector sets."""
    rows = []
    b_dev = b.to(device)
    for start in range(0, len(a), chunk):
        part = a[start:start + chunk].to(device)
        rows.append((1 - part @ b_dev.T).cpu())
    return torch.cat(rows)


def loo_nn(distances):
    """Each row's nearest-neighbor distance, excluding the diagonal."""
    masked = distances.clone()
    masked.fill_diagonal_(float('inf'))
    return masked.min(dim=1).values


def nn_to_reference(distances, reference_ids):
    """(distance, reference id) of each row's nearest reference column."""
    values, indices = distances.min(dim=1)
    return values, [reference_ids[i] for i in indices.tolist()]


def flag_mask(lpips_nn, inception_nn, lpips_threshold, inception_threshold):
    """Flag when either feature space falls below its threshold."""
    return (lpips_nn < lpips_threshold) | (inception_nn < inception_threshold)


def class_thresholds(loo_lpips, loo_inception):
    """The 5th-percentile thresholds for one class."""
    return {
        'lpips': float(numpy.percentile(loo_lpips.numpy(), PERCENTILE)),
        'inception': float(numpy.percentile(loo_inception.numpy(),
                                            PERCENTILE))}


def load_thresholds(path, splits_sha):
    """Cached per-class thresholds, or None when missing or stale."""
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        payload = json.load(handle)
    if payload.get('splits_sha1') != splits_sha:
        return None
    return payload['thresholds']


def save_thresholds(path, thresholds, splits_sha):
    """Write the thresholds next to the splits hash that produced them."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as handle:
        json.dump({'splits_sha1': splits_sha, 'thresholds': thresholds},
                  handle)


def write_rows(path, columns, rows):
    """Write rows to path as CSV, overwriting."""
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def quarantine(synth_dir, flagged_paths):
    """Move flagged files into a sibling _flagged tree, returning its root."""
    root = os.path.normpath(synth_dir) + '_flagged'
    for path in flagged_paths:
        folder = os.path.join(root, os.path.basename(os.path.dirname(path)))
        os.makedirs(folder, exist_ok=True)
        shutil.move(path, os.path.join(folder, os.path.basename(path)))
    return root


def score_one_class(cls, real, synth, models, thresholds, device, batch_size):
    """NN distances, ids and flags for one class, given loaded model handles."""
    lpips_model, inception_model, transform = models
    real_ids = [os.path.splitext(os.path.basename(p))[0] for p in real]

    real_l = lpips_vectors(lpips_model, real, device, batch_size)
    real_i = inception_vectors(inception_model, transform, real, device,
                               batch_size)
    if cls not in thresholds:
        thresholds[cls] = class_thresholds(
            loo_nn(lpips_distances(real_l, real_l, device)),
            loo_nn(cosine_distances(real_i, real_i, device)))

    synth_l = lpips_vectors(lpips_model, synth, device, batch_size)
    synth_i = inception_vectors(inception_model, transform, synth, device,
                                batch_size)
    nn_l, ids_l = nn_to_reference(lpips_distances(synth_l, real_l, device),
                                  real_ids)
    nn_i, ids_i = nn_to_reference(cosine_distances(synth_i, real_i, device),
                                  real_ids)
    mask = flag_mask(nn_l, nn_i, thresholds[cls]['lpips'],
                     thresholds[cls]['inception'])
    return nn_l, ids_l, nn_i, ids_i, mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--synthetic-dir', required=True)
    parser.add_argument('--real-dir', default=REAL_DIR)
    parser.add_argument('--out')
    parser.add_argument('--quarantine', action='store_true')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available()
                        else 'cpu')
    args = parser.parse_args()

    if not os.path.isdir(args.synthetic_dir):
        parser.error(f"{args.synthetic_dir} is not a directory")
    if not os.path.isdir(args.real_dir):
        raise RuntimeError(f"no reference at {args.real_dir}. "
                           "Run preprocessing/export_real64.py first.")

    name = os.path.basename(os.path.normpath(args.synthetic_dir))
    if args.out is None:
        args.out = os.path.join(REPO_ROOT, 'reports',
                                f'memorization_{name}.csv')
    detail_path = os.path.join(REPO_ROOT, 'data', 'eval',
                               f'memorization_{name}_detail.csv')

    scored, skipped, unknown = discover_classes(args.real_dir,
                                                args.synthetic_dir)
    for cls in unknown:
        print(f"warning: {cls}/ is not a HAM10000 class, ignored")
    for cls in skipped:
        print(f"{cls}: missing or empty on one side, skipped")

    models = (build_lpips(args.device),) + build_inception(args.device)
    splits_sha = splits_sha1(SPLITS_CSV)
    thresholds = load_thresholds(THRESHOLDS_PATH, splits_sha) or {}
    known = set(thresholds)

    summary, detail, flagged_paths = [], [], []
    for cls in scored:
        real = list_pngs(os.path.join(args.real_dir, cls))
        synth = list_pngs(os.path.join(args.synthetic_dir, cls))
        nn_l, ids_l, nn_i, ids_i, mask = score_one_class(
            cls, real, synth, models, thresholds, args.device,
            args.batch_size)

        for i, path in enumerate(synth):
            detail.append({
                'class': cls, 'file': os.path.basename(path),
                'lpips_nn': round(float(nn_l[i]), 6),
                'lpips_nn_id': ids_l[i],
                'inception_nn': round(float(nn_i[i]), 6),
                'inception_nn_id': ids_i[i],
                'flagged': int(mask[i])})
            if mask[i]:
                flagged_paths.append(path)
        summary.append({
            'class': cls, 'n_synth': len(synth),
            'n_flagged': int(mask.sum()),
            'flagged_fraction': round(float(mask.float().mean()), 4),
            'lpips_threshold': round(thresholds[cls]['lpips'], 6),
            'inception_threshold': round(thresholds[cls]['inception'], 6)})
        row = summary[-1]
        print(f"{cls:6s} flagged {row['n_flagged']:4d} / {row['n_synth']} "
              f"({100 * row['flagged_fraction']:.2f}%)")

    if set(thresholds) != known:
        save_thresholds(THRESHOLDS_PATH, thresholds, splits_sha)
    write_rows(args.out, SUMMARY_COLUMNS, summary)
    write_rows(detail_path, DETAIL_COLUMNS, detail)
    print(f"wrote {args.out}")

    if args.quarantine and flagged_paths:
        root = quarantine(args.synthetic_dir, flagged_paths)
        print(f"moved {len(flagged_paths)} flagged files to {root}")


if __name__ == '__main__':
    main()
