import hashlib
import json
import os

import numpy
import pandas
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# do not reorder, checkpoints encode these indices
CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASSES)}

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS_CSV = os.path.join(REPO_ROOT, 'data', 'splits.csv')
LABELED_DIR = os.path.join(REPO_ROOT, 'data', 'labeled')
CACHE_DIR = os.path.join(REPO_ROOT, 'data', 'cache')

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class SquareCenterCrop:
    """Crop a PIL image to a centered square of its shorter side."""

    def __call__(self, image):
        side = min(image.size)
        return transforms.functional.center_crop(image, [side, side])


def cache_paths(cache_dir, image_size):
    """Array and sidecar paths for a cache at one resolution."""
    stem = os.path.join(cache_dir, f'ham{image_size}')
    return stem + '.npy', stem + '.json'


def splits_sha1(splits_csv):
    """SHA-1 of the splits file, so a stale cache is detectable."""
    with open(splits_csv, 'rb') as handle:
        return hashlib.sha1(handle.read()).hexdigest()


def validate_cache_meta(meta, image_size, total_rows, splits_csv):
    """Raise RuntimeError if a cache sidecar is stale or mismatched."""
    # a stale cache silently mislabels every image, so fail hard
    if meta['image_size'] != image_size:
        raise RuntimeError(
            f"cache is {meta['image_size']}px, requested {image_size}px. "
            "Rebuild with preprocessing/make_cache.py.")
    if meta['rows'] != total_rows:
        raise RuntimeError(
            f"cache has {meta['rows']} rows, splits.csv has {total_rows}. "
            "Rebuild with preprocessing/make_cache.py.")
    if meta['splits_sha1'] != splits_sha1(splits_csv):
        raise RuntimeError(
            "cache was built from a different splits.csv. "
            "Rebuild with preprocessing/make_cache.py.")


def geometry_transform(image_size):
    """Crop to square and resize. Shared with the cache builder."""
    return transforms.Compose([
        SquareCenterCrop(),
        transforms.Resize((image_size, image_size)),
    ])


def build_transform(image_size=256, normalize='diffusion', augment=False):
    """Crop to a square, resize, optionally augment, then normalize."""
    # crop first, resizing 600x450 directly squashes lesions 1.33x
    steps = [geometry_transform(image_size)]

    if augment:
        steps += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                   saturation=0.2, hue=0.05),
        ]

    steps.append(transforms.ToTensor())

    if normalize == 'diffusion':
        steps.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    elif normalize == 'imagenet':
        steps.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    elif normalize is not None:
        raise ValueError(f"unknown normalize option: {normalize}")

    return transforms.Compose(steps)


class HAM10000(Dataset):
    """HAM10000 images for one split, read from data/labeled/<dx>/<id>.jpg."""

    def __init__(self, split, image_size=256, normalize='diffusion',
                 augment=False, classes=None, splits_csv=SPLITS_CSV,
                 labeled_dir=LABELED_DIR, cache=True, cache_dir=CACHE_DIR):
        if split not in ('train', 'val', 'test'):
            raise ValueError(f"split must be train, val or test, got {split}")

        self.labeled_dir = labeled_dir
        self.transform = build_transform(image_size, normalize, augment)

        rows = pandas.read_csv(splits_csv)
        total_rows = len(rows)
        # the position in the full file indexes the cache
        rows = rows.assign(row=range(total_rows))
        rows = rows[rows['split'] == split]
        if classes is not None:
            rows = rows[rows['dx'].isin(classes)]
        self.rows = rows.reset_index(drop=True)

        if len(self.rows) == 0:
            raise RuntimeError(f"no rows for split={split}, classes={classes}")

        self.cache = None
        if cache:
            self.cache = self._load_cache(cache_dir, image_size, splits_csv,
                                          total_rows)

        if self.cache is None:
            first = self._path(0)
            if not os.path.exists(first):
                raise RuntimeError(
                    f"image not found: {first}\n"
                    "Run preprocessing/sort_images.py to populate "
                    "data/labeled/.")

    def _path(self, i):
        row = self.rows.iloc[i]
        return os.path.join(self.labeled_dir, row['dx'],
                            row['image_id'] + '.jpg')

    def __len__(self):
        return len(self.rows)

    def _load_cache(self, cache_dir, image_size, splits_csv, total_rows):
        """Load the cache for this resolution, or None to decode JPEGs."""
        npy_path, json_path = cache_paths(cache_dir, image_size)
        if not (os.path.exists(npy_path) and os.path.exists(json_path)):
            print(f"no {image_size}px cache, decoding JPEGs instead. "
                  "Run preprocessing/make_cache.py to speed this up.")
            return None

        with open(json_path) as handle:
            meta = json.load(handle)
        validate_cache_meta(meta, image_size, total_rows, splits_csv)

        return numpy.load(npy_path, mmap_mode='r')

    def __getitem__(self, i):
        row = self.rows.iloc[i]
        if self.cache is not None:
            image = Image.fromarray(numpy.asarray(self.cache[row['row']]))
        else:
            image = Image.open(self._path(i)).convert('RGB')
        return self.transform(image), CLASS_TO_INDEX[row['dx']]

    def class_counts(self):
        counts = self.rows['dx'].value_counts()
        return torch.tensor([int(counts.get(c, 0)) for c in CLASSES])
