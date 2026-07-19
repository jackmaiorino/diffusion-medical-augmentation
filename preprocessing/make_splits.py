import os

import pandas
from sklearn.model_selection import train_test_split

SEED = 612
VAL_FRAC = 0.15
TEST_FRAC = 0.15

base = os.path.dirname(os.path.abspath(__file__))
raw = pandas.read_csv(os.path.join(base, '..', 'data',
                      'raw', 'HAM10000_metadata.csv'))

# Split on lesions rather than images. HAM10000 has 1-2 images per lesion, so
# assigning images independently would put the same lesion in both train and
# test. Every image of a lesion inherits that lesion's split, which prevents
# the leakage while keeping all 10,015 images.
lesions = raw[['lesion_id', 'dx']].drop_duplicates(ignore_index=True)

train, holdout = train_test_split(
    lesions,
    test_size=VAL_FRAC + TEST_FRAC,
    stratify=lesions['dx'],
    random_state=SEED,
)
val, test = train_test_split(
    holdout,
    test_size=TEST_FRAC / (VAL_FRAC + TEST_FRAC),
    stratify=holdout['dx'],
    random_state=SEED,
)

assignments = pandas.concat([
    train.assign(split='train'),
    val.assign(split='val'),
    test.assign(split='test'),
])[['lesion_id', 'split']]

splits = raw[['image_id', 'lesion_id', 'dx']].merge(
    assignments, on='lesion_id', how='left'
).sort_values('image_id', ignore_index=True)

assert len(splits) == len(raw), 'lost or duplicated rows during splitting'
assert splits['split'].notna().all(), 'image with no split assigned'
assert (splits.groupby('lesion_id')['split'].nunique() == 1).all(), \
    'lesion appears in more than one split'

out_path = os.path.join(base, '..', 'data', 'splits.csv')
splits.to_csv(out_path, index=False)

print(f"wrote {len(splits)} rows to {out_path}")
print(splits.groupby(['dx', 'split']).size().unstack(
    fill_value=0)[['train', 'val', 'test']])
