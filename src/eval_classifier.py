"""Re-evaluate a saved classifier checkpoint on the held-out test split."""
import argparse
import json
import os

import torch

from dataset import CLASSES
from train_classifier import build_resnet18, evaluate, split_tensors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True,
                        help='run directory, e.g. runs/classifier/real_only_s612')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available()
                        else 'cpu')
    args = parser.parse_args()

    ckpt = torch.load(os.path.join(args.run, 'best.pt'), map_location='cpu')
    model = build_resnet18().to(args.device)
    model.load_state_dict(ckpt['model'])
    images, labels = split_tensors('test')
    _, metrics = evaluate(model, images, labels, args.device)

    stored = None
    result_path = os.path.join(args.run, 'result.json')
    if os.path.exists(result_path):
        with open(result_path) as handle:
            stored = json.load(handle)['test']

    print(f"checkpoint from step {ckpt['step']}, {len(labels)} test images")
    keys = [f'f1_{name}' for name in CLASSES]
    keys += ['macro_f1', 'rare_f1', 'balanced_acc', 'auc_ovr']
    for key in keys:
        line = f"{key:>13} {metrics[key]:.4f}"
        if stored:
            line += f"   reported {stored[key]:.4f}"
        print(line)
    if stored:
        same = all(abs(metrics[k] - stored[k]) < 1e-6 for k in keys)
        print('matches result.json' if same else 'DIFFERS from result.json')


if __name__ == '__main__':
    main()
