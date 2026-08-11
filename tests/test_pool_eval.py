"""Tests for the filtered-pool evaluation."""
import pandas
import torch

from pool_eval import accepted_paths, diversity_stats, sample


def test_accepted_paths_survivors_found_in_either_tree(tmp_path):
    detail = pandas.DataFrame({
        'class': ['df'] * 3,
        'file': ['a.png', 'b.png', 'c.png'],
        'lpips_nn': [0.5, 0.5, 0.01],
        'inception_nn': [0.5, 0.5, 0.5]})
    thresholds = {'df': {'lpips': 0.05, 'inception': 0.05}}
    synth = tmp_path / 'pool'
    (synth / 'df').mkdir(parents=True)
    (synth / 'df' / 'a.png').touch()
    flagged = tmp_path / 'pool_flagged'
    (flagged / 'df').mkdir(parents=True)
    (flagged / 'df' / 'b.png').touch()

    paths = accepted_paths(detail, 'df', thresholds, str(synth))

    assert [p.split('pool')[-1] for p in paths] == \
        ['\\df\\a.png', '_flagged\\df\\b.png']


def test_sample_is_deterministic_and_capped():
    paths = [str(i) for i in range(10)]

    assert sample(paths, 20, 612) == paths
    assert sample(paths, 4, 612) == sample(paths, 4, 612)
    assert len(sample(paths, 4, 612)) == 4


def test_diversity_stats_counts_near_duplicates():
    vectors = torch.tensor([[0.0], [0.1], [5.0]])

    mean_pairwise, dup_frac = diversity_stats(vectors, 'cpu', 0.5)

    # nn distances are squared L2: 0.01, 0.01, 24.01 -> two under 0.5
    assert abs(dup_frac - 2 / 3) < 1e-6
    assert mean_pairwise > 0
