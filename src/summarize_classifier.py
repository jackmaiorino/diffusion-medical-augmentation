"""Aggregate classifier run results into the report CSVs."""
import argparse
import glob
import json
import os

import pandas

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.path.join(REPO_ROOT, 'runs', 'classifier')
OUT_RUNS = os.path.join(REPO_ROOT, 'reports', 'classifier_runs.csv')
OUT_SUMMARY = os.path.join(REPO_ROOT, 'reports', 'classifier_summary.csv')

ARM_ORDER = ['real_only', 'classical_aug', 'dup_real', 'synth_all',
             'synth_accepted']
METRICS = ['test_rare_f1', 'test_macro_f1', 'test_balanced_acc',
           'test_auc_ovr', 'test_f1_akiec', 'test_f1_df', 'test_f1_vasc']


def load_runs(runs_dir):
    """One row per result.json, smoke runs excluded."""
    rows = []
    for path in sorted(glob.glob(os.path.join(runs_dir, '*', 'result.json'))):
        if os.path.basename(os.path.dirname(path)).endswith('_smoke'):
            continue
        with open(path) as handle:
            result = json.load(handle)
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
