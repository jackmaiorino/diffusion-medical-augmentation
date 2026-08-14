"""Aggregate classifier run results into the report CSVs."""
import argparse
import glob
import json
import math
import os

import pandas

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.path.join(REPO_ROOT, 'runs', 'classifier')
OUT_RUNS = os.path.join(REPO_ROOT, 'reports', 'classifier_runs.csv')
OUT_SUMMARY = os.path.join(REPO_ROOT, 'reports', 'classifier_summary.csv')

ARM_ORDER = ['real_only', 'classical_aug', 'dup_real', 'synth_all',
             'synth_accepted']
FORMAL_SEEDS = (612, 613, 614)
# duplicated from train_classifier.py defaults; new flags must be added here too
FORMAL_HYPERPARAMETERS = {
    'steps': 6000,
    'batch_size': 128,
    'lr': 1e-3,
    'weight_decay': 0.05,
    'warmup': 250,
    'eval_every': 250,
    'smoke': False,
}
METRICS = ['test_rare_f1', 'test_macro_f1', 'test_balanced_acc',
           'test_auc_ovr', 'test_f1_akiec', 'test_f1_df', 'test_f1_vasc']
REQUIRED_TEST_METRICS = {
    'macro_f1', 'rare_f1', 'balanced_acc', 'auc_ovr',
    *(f'f1_{cls}' for cls in ('akiec', 'bcc', 'bkl', 'df', 'mel', 'nv',
                              'vasc')),
}


def run_label(arm, seed):
    """Label for one arm and seed run."""
    return f'{arm}/seed {seed}'


def finite_metric(value):
    """Whether a JSON value is a finite numeric metric, excluding booleans."""
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def validate_formal_runs(records):
    """Reject incomplete, duplicated, or non-comparable formal runs."""
    paths_by_run = {}
    for path, result in records:
        run = (result.get('arm'), result.get('seed'))
        paths_by_run.setdefault(run, []).append(path)

    duplicates = {run: paths for run, paths in paths_by_run.items()
                  if len(paths) > 1}
    if duplicates:
        details = []
        for (arm, seed), paths in sorted(
                duplicates.items(), key=lambda item: str(item[0])):
            details.append(f"{run_label(arm, seed)} ({', '.join(paths)})")
        raise RuntimeError('duplicate classifier runs: ' +
                           '; '.join(details))

    expected = {(arm, seed) for arm in ARM_ORDER for seed in FORMAL_SEEDS}
    actual = set(paths_by_run)
    matrix_errors = []
    missing = sorted(expected - actual, key=str)
    unexpected = sorted(actual - expected, key=str)
    if missing:
        matrix_errors.append('missing classifier runs: ' +
                             ', '.join(run_label(*run) for run in missing))
    if unexpected:
        matrix_errors.append('unexpected classifier runs: ' +
                             ', '.join(run_label(*run)
                                       for run in unexpected))
    if matrix_errors:
        raise RuntimeError('; '.join(matrix_errors))

    mismatches = []
    for path, result in records:
        arm, seed = result['arm'], result['seed']
        label = run_label(arm, seed)
        args = result.get('args')
        if not isinstance(args, dict):
            mismatches.append(f'{label}: args expected an object, got '
                              f'{type(args).__name__} ({path})')
            continue
        if args.get('arm') != arm:
            mismatches.append(f"{label}: args.arm expected {arm!r}, got "
                              f"{args.get('arm', '<missing>')!r} ({path})")
        if args.get('seed') != seed:
            mismatches.append(f'{label}: args.seed expected {seed!r}, got '
                              f"{args.get('seed', '<missing>')!r} ({path})")
        for name, expected_value in FORMAL_HYPERPARAMETERS.items():
            actual_value = args.get(name, '<missing>')
            matches = actual_value == expected_value
            if isinstance(expected_value, bool):
                matches = actual_value is expected_value
            if not matches:
                mismatches.append(
                    f'{label}: {name} expected {expected_value!r}, got '
                    f'{actual_value!r} ({path})')

        val = result.get('val')
        if not isinstance(val, dict):
            mismatches.append(f'{label}: val expected an object ({path})')
        elif not finite_metric(val.get('macro_f1')):
            mismatches.append(
                f"{label}: val.macro_f1 must be finite, got "
                f"{val.get('macro_f1', '<missing>')!r} ({path})")

        test = result.get('test')
        if not isinstance(test, dict):
            mismatches.append(f'{label}: test expected an object ({path})')
        else:
            for name in sorted(REQUIRED_TEST_METRICS):
                value = test.get(name, '<missing>')
                if not finite_metric(value):
                    mismatches.append(
                        f'{label}: test.{name} must be finite, got '
                        f'{value!r} ({path})')
    if mismatches:
        raise RuntimeError('formal classifier run mismatch: ' +
                           '; '.join(mismatches))


def load_runs(runs_dir):
    """Load the complete formal run matrix, excluding named smoke runs."""
    records = []
    for path in sorted(glob.glob(os.path.join(runs_dir, '*', 'result.json'))):
        if os.path.basename(os.path.dirname(path)).endswith('_smoke'):
            continue
        with open(path) as handle:
            result = json.load(handle)
        records.append((path, result))

    validate_formal_runs(records)
    rows = []
    for _, result in records:
        rows.append({'arm': result['arm'], 'seed': result['seed'],
                     'best_step': result['best_step'],
                     'val_macro_f1': result['val']['macro_f1'],
                     **{f'test_{k}': v for k, v in result['test'].items()}})
    return pandas.DataFrame(rows)


def summarize(runs):
    """Per-arm mean and std, with rare-F1 delta against real_only."""
    grouped = runs.groupby('arm')[METRICS]
    summary = grouped.mean().join(grouped.std(), rsuffix='_std')
    summary['seeds'] = runs.groupby('arm')['seed'].count()
    # NaN until the reference arm has results, so partial matrices still print
    summary['rare_f1_delta'] = float('nan')
    if 'real_only' in summary.index:
        reference = summary.loc['real_only', 'test_rare_f1']
        summary['rare_f1_delta'] = summary['test_rare_f1'] - reference
    order = [arm for arm in ARM_ORDER if arm in summary.index]
    order += [arm for arm in summary.index if arm not in ARM_ORDER]
    return summary.loc[order].reset_index()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs-dir', default=RUNS_DIR)
    args = parser.parse_args()

    runs = load_runs(args.runs_dir)
    if runs.empty:
        raise SystemExit(f"no result.json files under {args.runs_dir}")
    runs = runs.sort_values(['arm', 'seed'], ignore_index=True)
    runs.to_csv(OUT_RUNS, index=False)

    summary = summarize(runs)
    summary.to_csv(OUT_SUMMARY, index=False)

    shown = summary[['arm', 'seeds', 'rare_f1_delta', 'test_rare_f1',
                     'test_macro_f1', 'test_balanced_acc', 'test_auc_ovr']]
    print(shown.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    print(f"wrote {OUT_RUNS} and {OUT_SUMMARY}")


if __name__ == '__main__':
    main()
