"""Tests for the checkpoint validation-loss evaluation."""
import pytest
import torch

import ddpm_val_loss
from ddpm_model import build_train_scheduler
from ddpm_val_loss import draw_eval_batches, list_checkpoints


def make_checkpoints(tmp_path, names):
    for name in names:
        (tmp_path / name).touch()
    return str(tmp_path)


def test_list_checkpoints_sorts_numerically(tmp_path):
    run = make_checkpoints(tmp_path, ['ckpt_100000.pt', 'ckpt_5000.pt',
                                      'ckpt_20000.pt'])

    steps = [step for step, _ in list_checkpoints(run)]

    assert steps == [5000, 20000, 100000]


def test_list_checkpoints_ignores_ckpt_last_and_logs(tmp_path):
    run = make_checkpoints(tmp_path, ['ckpt_5000.pt', 'ckpt_last.pt',
                                      'log.csv'])

    assert [step for step, _ in list_checkpoints(run)] == [5000]


def test_list_checkpoints_rejects_an_empty_run(tmp_path):
    run = make_checkpoints(tmp_path, ['ckpt_last.pt'])

    with pytest.raises(RuntimeError, match='no numbered checkpoints'):
        list_checkpoints(run)


def test_draw_eval_batches_is_identical_across_calls():
    images = torch.randn(10, 3, 8, 8)
    labels = torch.randint(0, 7, (10,))
    scheduler = build_train_scheduler()

    def draw():
        return list(draw_eval_batches(images, labels, scheduler,
                                      cfg_dropout=0.1, repeats=2,
                                      batch_size=4, seed=612))

    for first, second in zip(draw(), draw()):
        for a, b in zip(first, second):
            assert torch.equal(a, b)


def test_draw_eval_batches_covers_repeats_times_the_split():
    images = torch.randn(10, 3, 8, 8)
    labels = torch.randint(0, 7, (10,))
    scheduler = build_train_scheduler()

    batches = list(draw_eval_batches(images, labels, scheduler,
                                     cfg_dropout=0.1, repeats=3,
                                     batch_size=4, seed=612))

    assert sum(batch[0].shape[0] for batch in batches) == 30


def test_no_pretrained_weights_are_loaded():
    with open(ddpm_val_loss.__file__) as handle:
        assert 'from_pretrained' not in handle.read()
