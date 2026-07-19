import os
import shutil

import pandas

base = os.path.dirname(os.path.abspath(__file__))
raw_dir = os.path.join(base, '..', 'data', 'raw')
labeled_dir = os.path.join(base, '..', 'data', 'labeled')

# All images are kept, not one per lesion. Leakage is handled in
# make_splits.py, which assigns whole lesions to a split.
raw = pandas.read_csv(os.path.join(raw_dir, 'HAM10000_metadata.csv'))
part1 = set(os.listdir(os.path.join(raw_dir, 'HAM10000_images_part_1')))
part2 = set(os.listdir(os.path.join(raw_dir, 'HAM10000_images_part_2')))

copied = 0
missing = 0
for image_id, label in zip(raw['image_id'], raw['dx']):
    filename = image_id + '.jpg'
    if filename in part1:
        src = os.path.join(raw_dir, 'HAM10000_images_part_1', filename)
    elif filename in part2:
        src = os.path.join(raw_dir, 'HAM10000_images_part_2', filename)
    else:
        missing += 1
        continue
    os.makedirs(os.path.join(labeled_dir, label), exist_ok=True)
    shutil.copy2(src, os.path.join(labeled_dir, label, filename))
    copied += 1

print(f"copied {copied} images, {missing} not found in raw folders")
