"""Export per-class train-split PNGs from the ham64 cache for evaluation."""
import argparse
import json
import os
import shutil
import sys

import numpy
import pandas
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'src'))

from dataset import CACHE_DIR, SPLITS_CSV, cache_paths  # noqa: E402
from dataset import validate_cache_meta  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(REPO_ROOT, 'data', 'eval', 'real64')


def export_real64(splits_csv, cache_dir, out_dir, image_size=64,
                  split='train'):
    """Write one split's rows as out_dir/<dx>/<image_id>.png, return count."""
    npy_path, json_path = cache_paths(cache_dir, image_size)
    # no JPEG fallback here, a wrong-geometry reference poisons every score
    if not (os.path.exists(npy_path) and os.path.exists(json_path)):
        raise RuntimeError(
            f"no {image_size}px cache in {cache_dir}. "
            "Run preprocessing/make_cache.py first.")

    rows = pandas.read_csv(splits_csv)
    with open(json_path) as handle:
        meta = json.load(handle)
    validate_cache_meta(meta, image_size, len(rows), splits_csv)

    array = numpy.load(npy_path, mmap_mode='r')
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)  # a re-split must not leave orphans

    picked = rows[rows['split'] == split]
    for row in picked.itertuples():
        folder = os.path.join(out_dir, row.dx)
        os.makedirs(folder, exist_ok=True)
        # row.Index is the position in the full csv, which indexes the cache
        Image.fromarray(numpy.asarray(array[row.Index])).save(
            os.path.join(folder, row.image_id + '.png'))
    return len(picked)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--splits-csv', default=SPLITS_CSV)
    parser.add_argument('--cache-dir', default=CACHE_DIR)
    parser.add_argument('--out')
    parser.add_argument('--image-size', type=int, default=64)
    parser.add_argument('--split', default='train',
                        choices=['train', 'val', 'test'])
    args = parser.parse_args()

    if args.out is None:
        args.out = EVAL_DIR if args.split == 'train' \
            else EVAL_DIR + '_' + args.split
    count = export_real64(args.splits_csv, args.cache_dir, args.out,
                          args.image_size, args.split)
    print(f"wrote {count} images to {args.out}")


if __name__ == '__main__':
    main()
