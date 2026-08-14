"""Build the classifier arm manifests from splits.csv and the synthetic pools."""
import argparse
import os

import numpy
import pandas

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS_CSV = os.path.join(REPO_ROOT, 'data', 'splits.csv')
SYNTH_ROOT = os.path.join(REPO_ROOT, 'data', 'synthetic')
ARMS_DIR = os.path.join(REPO_ROOT, 'data', 'arms')


def pool_files(pool_dir):
    """Per-class sorted .png filename lists for one synthetic pool."""
    # stray files like desktop.ini must not count as accepted images
    classes = sorted(dx for dx in os.listdir(pool_dir)
                     if os.path.isdir(os.path.join(pool_dir, dx)))
    return {dx: sorted(name for name in os.listdir(os.path.join(pool_dir, dx))
                       if name.endswith('.png'))
            for dx in classes}


def synth_manifest(pool_dir, repo_root):
    """Manifest rows for every image in a synthetic pool."""
    rows = [{'dx': dx, 'source': 'synth',
             'ref': os.path.relpath(os.path.join(pool_dir, dx, name),
                                    repo_root).replace(os.sep, '/')}
            for dx, names in pool_files(pool_dir).items() for name in names]
    return pandas.DataFrame(rows)


def duplicate_manifest(train, counts, seed):
    """Per-class duplicates of train image_ids, sampled with replacement."""
    # mel needs 831 duplicates from a 776-image pool, so replacement it is
    rng = numpy.random.default_rng(seed)
    parts = []
    for dx in sorted(counts):
        pool = train.loc[train['dx'] == dx, 'image_id'].to_numpy()
        parts.append(pandas.DataFrame({
            'dx': dx, 'source': 'dup',
            'ref': rng.choice(pool, size=counts[dx], replace=True)}))
    return pandas.concat(parts, ignore_index=True)


def build_arms(splits_csv, all_dir, accepted_dir, out_dir, repo_root, seed):
    """Write the four manifests and return them as {name: DataFrame}."""
    splits = pandas.read_csv(splits_csv)
    train = splits[splits['split'] == 'train']
    real = pandas.DataFrame({'dx': train['dx'], 'source': 'real',
                             'ref': train['image_id']})

    all_files = pool_files(all_dir)
    accepted_files = pool_files(accepted_dir)
    classes = sorted(train['dx'].unique())
    if sorted(all_files) != classes or sorted(accepted_files) != classes:
        raise RuntimeError(
            'synthetic pools and the train split disagree on classes')
    for dx in classes:
        stray = set(accepted_files[dx]) - set(all_files[dx])
        if stray:
            raise RuntimeError(f"accepted {dx} images missing from the "
                               f"full pool: {sorted(stray)[:3]}")

    # dup_real matches the accepted pool's volume and class distribution
    counts = {dx: len(names) for dx, names in accepted_files.items()}
    arms = {
        'real_only': real,
        'dup_real': pandas.concat(
            [real, duplicate_manifest(train, counts, seed)],
            ignore_index=True),
        'synth_all': pandas.concat(
            [real, synth_manifest(all_dir, repo_root)], ignore_index=True),
        'synth_accepted': pandas.concat(
            [real, synth_manifest(accepted_dir, repo_root)],
            ignore_index=True),
    }

    os.makedirs(out_dir, exist_ok=True)
    for name, manifest in arms.items():
        manifest.to_csv(os.path.join(out_dir, name + '.csv'), index=False)
    return arms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--splits-csv', default=SPLITS_CSV)
    parser.add_argument('--all-dir',
                        default=os.path.join(SYNTH_ROOT, 'ddpm64_all'))
    parser.add_argument('--accepted-dir',
                        default=os.path.join(SYNTH_ROOT, 'ddpm64_accepted'))
    parser.add_argument('--out', default=ARMS_DIR)
    parser.add_argument('--seed', type=int, default=612)
    args = parser.parse_args()

    arms = build_arms(args.splits_csv, args.all_dir, args.accepted_dir,
                      args.out, REPO_ROOT, args.seed)
    for name, manifest in arms.items():
        by_class = manifest.groupby('dx').size()
        detail = ' '.join(f'{dx} {n}' for dx, n in by_class.items())
        print(f"{name}: {len(manifest)} rows ({detail})")


if __name__ == '__main__':
    main()
