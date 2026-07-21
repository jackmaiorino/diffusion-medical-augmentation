import os

import pandas
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Fixed alphabetical order so class indices are identical for every run and
# for both team members. Do not reorder: checkpoints encode these indices.
CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASSES)}

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS_CSV = os.path.join(REPO_ROOT, 'data', 'splits.csv')
LABELED_DIR = os.path.join(REPO_ROOT, 'data', 'labeled')

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(image_size=256, normalize='diffusion', augment=False):
    """Resize to a square and normalize."""
    steps = [transforms.Resize((image_size, image_size))]

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
                 labeled_dir=LABELED_DIR):
        if split not in ('train', 'val', 'test'):
            raise ValueError(f"split must be train, val or test, got {split}")

        self.labeled_dir = labeled_dir
        self.transform = build_transform(image_size, normalize, augment)

        rows = pandas.read_csv(splits_csv)
        rows = rows[rows['split'] == split]
        if classes is not None:
            rows = rows[rows['dx'].isin(classes)]
        self.rows = rows.reset_index(drop=True)

        if len(self.rows) == 0:
            raise RuntimeError(f"no rows for split={split}, classes={classes}")

        first = self._path(0)
        if not os.path.exists(first):
            raise RuntimeError(
                f"image not found: {first}\n"
                "Run preprocessing/sort_images.py to populate data/labeled/.")

    def _path(self, i):
        row = self.rows.iloc[i]
        return os.path.join(self.labeled_dir, row['dx'],
                            row['image_id'] + '.jpg')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        image = Image.open(self._path(i)).convert('RGB')
        label = CLASS_TO_INDEX[self.rows.iloc[i]['dx']]
        return self.transform(image), label

    def class_counts(self):
        counts = self.rows['dx'].value_counts()
        return torch.tensor([int(counts.get(c, 0)) for c in CLASSES])
