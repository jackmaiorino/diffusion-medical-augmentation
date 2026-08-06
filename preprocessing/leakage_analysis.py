"""Quantify lesion leakage from naive image-level splitting vs our lesion-level split."""
import collections
import os

import numpy
import pandas

CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15
N_TRIALS = 200

base = os.path.dirname(os.path.abspath(__file__))
meta = pandas.read_csv(os.path.join(base, '..', 'data', 'raw', 'HAM10000_metadata.csv'))

lesion = meta['lesion_id'].to_numpy()
dx = meta['dx'].to_numpy()
n = len(meta)

# Exact, method-independent facts.
sizes = meta.groupby('lesion_id').size()
shared = int((meta['lesion_id'].map(sizes) > 1).sum())
print(f"images {n}, unique lesions {sizes.size}, "
      f"images sharing a lesion {shared} ({100 * shared / n:.1f}%)")

# Expected leakage under a naive image-level split, averaged over many random seeds
# so the estimate does not depend on any single shuffle.
overall = []
per_class = collections.defaultdict(list)
n_test_start = int((TRAIN_FRAC + VAL_FRAC) * n)
for seed in range(N_TRIALS):
    order = numpy.random.default_rng(seed).permutation(n)
    train_les = set(lesion[order[:int(TRAIN_FRAC * n)]])
    test = order[n_test_start:]
    leaked = numpy.array([lesion[i] in train_les for i in test])
    overall.append(100 * leaked.mean())
    for c in CLASSES:
        m = dx[test] == c
        if m.any():
            per_class[c].append(100 * leaked[m].mean())

print(f"\nnaive image-level split leakage over {N_TRIALS} seeds: "
      f"mean {numpy.mean(overall):.1f}% (std {numpy.std(overall):.1f})")
print(f"{'class':6s} {'mean %':>7s} {'std':>5s}")
for c in CLASSES:
    v = per_class[c]
    print(f"{c:6s} {numpy.mean(v):7.1f} {numpy.std(v):5.1f}")
