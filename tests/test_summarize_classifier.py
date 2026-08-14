"""Tests for strict aggregation of the formal classifier matrix."""
import json

import pytest

from summarize_classifier import (ARM_ORDER, FORMAL_HYPERPARAMETERS,
                                  FORMAL_SEEDS, load_runs, summarize)


def result_payload(arm, seed):
    """Minimal valid result.json payload for one formal run."""
    test = {
        'f1_akiec': 0.1,
        'f1_bcc': 0.2,
        'f1_bkl': 0.3,
        'f1_df': 0.4,
        'f1_mel': 0.5,
        'f1_nv': 0.6,
        'f1_vasc': 0.7,
        'macro_f1': 0.4,
        'rare_f1': 0.4,
        'balanced_acc': 0.41,
        'auc_ovr': 0.8,
    }
    return {
        'arm': arm,
        'seed': seed,
        'best_step': 1000,
        'val': {'macro_f1': 0.35},
        'test': test,
        'args': {'arm': arm, 'seed': seed, **FORMAL_HYPERPARAMETERS},
    }


def write_result(root, directory, payload):
    run_dir = root / directory
    run_dir.mkdir()
    (run_dir / 'result.json').write_text(json.dumps(payload),
                                         encoding='utf-8')


def write_formal_matrix(root, omit=None):
    for arm in ARM_ORDER:
        for seed in FORMAL_SEEDS:
            if (arm, seed) != omit:
                write_result(root, f'{arm}_s{seed}',
                             result_payload(arm, seed))


def test_complete_formal_matrix_preserves_summary_shape(tmp_path):
    write_formal_matrix(tmp_path)
    smoke = result_payload('real_only', 612)
    smoke['args']['smoke'] = True
    write_result(tmp_path, 'real_only_s612_smoke', smoke)

    runs = load_runs(tmp_path)
    summary = summarize(runs)

    assert len(runs) == len(ARM_ORDER) * len(FORMAL_SEEDS)
    assert summary['arm'].tolist() == ARM_ORDER
    assert summary['seeds'].tolist() == [len(FORMAL_SEEDS)] * len(ARM_ORDER)
    assert summary.loc[0, 'rare_f1_delta'] == pytest.approx(0.0)


def test_duplicate_arm_seed_run_is_rejected(tmp_path):
    write_formal_matrix(tmp_path)
    write_result(tmp_path, 'copied_real_only_s612',
                 result_payload('real_only', 612))

    with pytest.raises(RuntimeError, match=(
            r'duplicate classifier runs: real_only/seed 612')):
        load_runs(tmp_path)


def test_missing_arm_seed_run_is_rejected(tmp_path):
    write_formal_matrix(tmp_path, omit=('synth_all', 614))

    with pytest.raises(RuntimeError, match=(
            r'missing classifier runs: synth_all/seed 614')):
        load_runs(tmp_path)


def test_formal_hyperparameter_mismatch_names_the_run(tmp_path):
    write_formal_matrix(tmp_path)
    path = tmp_path / 'synth_accepted_s613' / 'result.json'
    payload = json.loads(path.read_text(encoding='utf-8'))
    payload['args']['steps'] = 5000
    path.write_text(json.dumps(payload), encoding='utf-8')

    with pytest.raises(RuntimeError, match=(
            r'formal classifier run mismatch: synth_accepted/seed 613: '
            r'steps expected 6000, got 5000')):
        load_runs(tmp_path)


def test_top_level_and_args_identity_must_match(tmp_path):
    write_formal_matrix(tmp_path)
    path = tmp_path / 'dup_real_s614' / 'result.json'
    payload = json.loads(path.read_text(encoding='utf-8'))
    payload['args']['seed'] = 613
    path.write_text(json.dumps(payload), encoding='utf-8')

    with pytest.raises(RuntimeError, match=(
            r'dup_real/seed 614: args.seed expected 614, got 613')):
        load_runs(tmp_path)


def test_missing_test_metric_is_rejected(tmp_path):
    write_formal_matrix(tmp_path)
    path = tmp_path / 'real_only_s612' / 'result.json'
    payload = json.loads(path.read_text(encoding='utf-8'))
    del payload['test']['rare_f1']
    path.write_text(json.dumps(payload), encoding='utf-8')

    with pytest.raises(RuntimeError, match=(
            r"real_only/seed 612: test.rare_f1 must be finite, got "
            r"'<missing>'")):
        load_runs(tmp_path)


def test_nonfinite_test_metric_is_rejected(tmp_path):
    write_formal_matrix(tmp_path)
    path = tmp_path / 'synth_all_s613' / 'result.json'
    payload = json.loads(path.read_text(encoding='utf-8'))
    payload['test']['f1_df'] = float('nan')
    path.write_text(json.dumps(payload), encoding='utf-8')

    with pytest.raises(RuntimeError, match=(
            r'synth_all/seed 613: test.f1_df must be finite, got nan')):
        load_runs(tmp_path)
