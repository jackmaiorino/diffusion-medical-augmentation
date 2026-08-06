"""Build a uint8 image cache aligned to splits.csv row order."""
import argparse
import hashlib
import json
import os
import sys

import numpy
import pandas
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'src'))

from dataset import CACHE_DIR, LABELED_DIR, SPLITS_CSV  # noqa: E402
from dataset import cache_paths, geometry_transform, splits_sha1  # noqa: E402


def build_cache(splits_csv, labeled_dir, cache_dir, image_size):
    """Write ham<size>.npy and its sidecar, one row per splits.csv row."""
    rows = pandas.read_csv(splits_csv)
    resize = geometry_transform(image_size)

    array = numpy.zeros((len(rows), image_size, image_size, 3), numpy.uint8)
    for i, (image_id, dx) in enumerate(zip(rows['image_id'], rows['dx'])):
        path = os.path.join(labeled_dir, dx, image_id + '.jpg')
        with Image.open(path) as image:
            array[i] = numpy.asarray(resize(image.convert('RGB')))

    os.makedirs(cache_dir, exist_ok=True)
    npy_path, json_path = cache_paths(cache_dir, image_size)
    numpy.save(npy_path, array)
    with open(json_path, 'w') as handle:
        json.dump({'image_size': image_size, 'rows': len(rows),
                   'splits_sha1': splits_sha1(splits_csv)}, handle)

    return npy_path


def verify(splits_csv, labeled_dir, cache_dir, image_size, sample):
    """Re-decode a random sample of rows and compare against the cache."""
    rows = pandas.read_csv(splits_csv)
    npy_path, _ = cache_paths(cache_dir, image_size)
    array = numpy.load(npy_path, mmap_mode='r')
    resize = geometry_transform(image_size)

    rng = numpy.random.default_rng(612)
    picked = rng.choice(len(rows), size=min(sample, len(rows)), replace=False)
    for i in picked:
        row = rows.iloc[i]
        path = os.path.join(labeled_dir, row['dx'], row['image_id'] + '.jpg')
        with Image.open(path) as image:
            fresh = numpy.asarray(resize(image.convert('RGB')))
        if not numpy.array_equal(fresh, array[i]):
            raise RuntimeError(f"cache row {i} does not match {path}")

    print(f"verified {len(picked)} rows against source images")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-size', type=int, default=64)
    parser.add_argument('--splits-csv', default=SPLITS_CSV)
    parser.add_argument('--labeled-dir', default=LABELED_DIR)
    parser.add_argument('--cache-dir', default=CACHE_DIR)
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--sample', type=int, default=200)
    args = parser.parse_args()

    if not args.verify:
        path = build_cache(args.splits_csv, args.labeled_dir, args.cache_dir,
                           args.image_size)
        size_mb = os.path.getsize(path) / 1e6
        print(f"wrote {path} ({size_mb:.0f} MB)")

    verify(args.splits_csv, args.labeled_dir, args.cache_dir, args.image_size,
           args.sample)


if __name__ == '__main__':
    main()
