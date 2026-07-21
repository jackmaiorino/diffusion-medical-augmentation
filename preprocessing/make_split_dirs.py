"""Materialize splits.csv as train/val/test directory trees."""
import os
import shutil

import pandas

base = os.path.dirname(os.path.abspath(__file__))
labeled_dir = os.path.join(base, '..', 'data', 'labeled')
splits_dir = os.path.join(base, '..', 'data', 'splits')

splits = pandas.read_csv(os.path.join(base, '..', 'data', 'splits.csv'))

linked = 0
copied = 0
missing = []
for image_id, dx, split in zip(splits['image_id'], splits['dx'],
                               splits['split']):
    filename = image_id + '.jpg'
    src = os.path.join(labeled_dir, dx, filename)
    if not os.path.exists(src):
        missing.append(filename)
        continue

    dst_dir = os.path.join(splits_dir, split, dx)
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, filename)
    if os.path.exists(dst):
        continue

    # Hard links keep this near-free on disk; fall back to copying if the
    # destination is on another volume.
    try:
        os.link(src, dst)
        linked += 1
    except OSError:
        shutil.copy2(src, dst)
        copied += 1

print(f"linked {linked}, copied {copied}, missing {len(missing)}")
if missing:
    print(f"first missing: {missing[:3]}")
    print("Run preprocessing/sort_images.py first.")

for split in ('train', 'val', 'test'):
    root = os.path.join(splits_dir, split)
    if not os.path.isdir(root):
        continue
    counts = {d: len(os.listdir(os.path.join(root, d)))
              for d in sorted(os.listdir(root))}
    print(f"{split}: {sum(counts.values())} images {counts}")
