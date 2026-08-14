"""Train the fixed ResNet-18 instrument on one classifier arm."""
import argparse
import csv
import json
import math
import os
import time

import numpy
import pandas
import torch
from PIL import Image
from sklearn.metrics import (balanced_accuracy_score, f1_score,
                             roc_auc_score)
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18

from dataset import (CACHE_DIR, CLASSES, CLASS_TO_INDEX, HAM10000, SPLITS_CSV,
                     build_transform, cache_paths, validate_cache_meta)
from train_ddpm import balanced_sampler, set_seed

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARMS_DIR = os.path.join(REPO_ROOT, 'data', 'arms')
RARE = ('akiec', 'df', 'vasc')

# arm name -> (manifest, train-time augmentation)
ARMS = {'real_only': ('real_only.csv', False),
        'classical_aug': ('real_only.csv', True),
        'dup_real': ('dup_real.csv', False),
        'synth_all': ('synth_all.csv', False),
        'synth_accepted': ('synth_accepted.csv', False)}


class ArmDataset(Dataset):
    """One arm's training set, real rows from the ham64 cache, synth from PNGs."""

    def __init__(self, manifest_csv, augment, image_size=64,
                 splits_csv=SPLITS_CSV, cache_dir=CACHE_DIR,
                 repo_root=REPO_ROOT):
        manifest = pandas.read_csv(manifest_csv)
        unknown = set(manifest['source']) - {'real', 'dup', 'synth'}
        if unknown:
            raise RuntimeError(f"unknown manifest sources: {sorted(unknown)}")
        self.transform = build_transform(image_size, 'imagenet', augment)

        rows = pandas.read_csv(splits_csv)
        split_of = dict(zip(rows['image_id'], rows['split']))
        row_of = {image_id: i for i, image_id in enumerate(rows['image_id'])}

        # only the synthetic pools may add images beyond the train split
        real_refs = manifest.loc[manifest['source'] != 'synth', 'ref']
        leaked = sorted({r for r in real_refs if split_of.get(r) != 'train'})
        if leaked:
            raise RuntimeError(f"non-train images in manifest: {leaked[:3]}")

        npy_path, json_path = cache_paths(cache_dir, image_size)
        if not (os.path.exists(npy_path) and os.path.exists(json_path)):
            raise RuntimeError(
                f"no {image_size}px cache in {cache_dir}. "
                "Run preprocessing/make_cache.py first.")
        with open(json_path) as handle:
            meta = json.load(handle)
        validate_cache_meta(meta, image_size, len(rows), splits_csv)
        cache = numpy.load(npy_path, mmap_mode='r')

        # materialize the whole arm, 14k images at 64px is ~170MB
        self.images = numpy.empty((len(manifest), image_size, image_size, 3),
                                  numpy.uint8)
        for i, entry in enumerate(manifest.itertuples()):
            if entry.source == 'synth':
                image = Image.open(os.path.join(repo_root, entry.ref))
                self.images[i] = numpy.asarray(image.convert('RGB'))
            else:
                self.images[i] = cache[row_of[entry.ref]]
        self.labels = torch.tensor(
            [CLASS_TO_INDEX[dx] for dx in manifest['dx']])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        image = Image.fromarray(self.images[i])
        return self.transform(image), int(self.labels[i])


def build_resnet18():
    """ResNet-18 from scratch, stem adapted for 64px inputs."""
    model = resnet18(weights=None, num_classes=len(CLASSES))
    # the stock 7x7 stride-2 stem plus maxpool leaves 2x2 features at 64px
    model.conv1 = torch.nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
    model.maxpool = torch.nn.Identity()
    return model


def lr_at(step, total, warmup, peak):
    """Linear warmup then cosine decay to zero."""
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(total - warmup, 1)
    return peak * 0.5 * (1 + math.cos(math.pi * progress))


def split_tensors(split, image_size=64):
    """One real split as a normalized image stack plus labels."""
    data = HAM10000(split, image_size=image_size, normalize='imagenet')
    loader = DataLoader(data, batch_size=256)
    batches = list(loader)
    return (torch.cat([images for images, _ in batches]),
            torch.cat([labels for _, labels in batches]))


def compute_metrics(logits, labels):
    """Per-class F1, macro F1, rare F1, balanced accuracy and macro OvR AUC."""
    y = labels.numpy()
    predicted = logits.argmax(1).numpy()
    probabilities = torch.softmax(logits, 1).numpy()
    per_class = f1_score(y, predicted, labels=range(len(CLASSES)),
                         average=None, zero_division=0)
    metrics = {f'f1_{name}': float(score)
               for name, score in zip(CLASSES, per_class)}
    metrics['macro_f1'] = float(numpy.mean(per_class))
    metrics['rare_f1'] = float(numpy.mean([metrics[f'f1_{c}'] for c in RARE]))
    metrics['balanced_acc'] = float(balanced_accuracy_score(y, predicted))
    metrics['auc_ovr'] = float(roc_auc_score(
        y, probabilities, multi_class='ovr', average='macro'))
    return metrics


@torch.no_grad()
def evaluate(model, images, labels, device, batch_size=512):
    """Loss and metrics over a full split, model returned to train mode."""
    model.eval()
    logits = []
    for start in range(0, len(images), batch_size):
        batch = images[start:start + batch_size].to(device)
        logits.append(model(batch).cpu())
    logits = torch.cat(logits)
    loss = torch.nn.functional.cross_entropy(logits, labels).item()
    model.train()
    return loss, compute_metrics(logits, labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', required=True, choices=sorted(ARMS))
    parser.add_argument('--seed', type=int, default=612)
    parser.add_argument('--steps', type=int, default=6000)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=0.05)
    parser.add_argument('--warmup', type=int, default=250)
    parser.add_argument('--eval-every', type=int, default=250)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--out',
                        default=os.path.join(REPO_ROOT, 'runs', 'classifier'))
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available()
                        else 'cpu')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()

    if args.smoke:
        args.steps, args.warmup, args.eval_every = 100, 20, 50

    if args.steps % args.eval_every != 0:
        raise SystemExit(f"--steps {args.steps} must be a multiple of "
                         f"--eval-every {args.eval_every}, or no checkpoint "
                         "gets selected")

    run_name = f'{args.arm}_s{args.seed}' + ('_smoke' if args.smoke else '')
    run_dir = os.path.join(args.out, run_name)
    os.makedirs(run_dir, exist_ok=True)

    set_seed(args.seed)
    if args.device.startswith('cuda'):
        # same seed, same GPU model must mean the same run
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    manifest, augment = ARMS[args.arm]
    data = ArmDataset(os.path.join(ARMS_DIR, manifest), augment)
    counts = torch.bincount(data.labels, minlength=len(CLASSES))
    print(f"{args.arm} seed {args.seed}: {len(data)} images, " +
          ' '.join(f"{name} {int(n)}" for name, n in zip(CLASSES, counts)))

    loader = DataLoader(
        data, batch_size=args.batch_size,
        sampler=balanced_sampler(data.labels,
                                 args.steps * args.batch_size, args.seed),
        num_workers=args.workers, pin_memory=True, drop_last=True,
        persistent_workers=args.workers > 0)

    model = build_resnet18().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    val_images, val_labels = split_tensors('val')

    log_path = os.path.join(run_dir, 'log.csv')
    with open(log_path, 'w', newline='') as handle:
        csv.writer(handle).writerow(
            ['step', 'loss', 'lr', 'val_macro_f1', 'seconds'])

    best = None
    running, counted = 0.0, 0
    started = time.time()
    model.train()

    for step, (images, targets) in enumerate(loader):
        lr = lr_at(step, args.steps, args.warmup, args.lr)
        for group in optimizer.param_groups:
            group['lr'] = lr

        images = images.to(args.device, non_blocking=True)
        targets = targets.to(args.device, non_blocking=True)
        loss = torch.nn.functional.cross_entropy(model(images), targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        running += loss.item()
        counted += 1

        if (step + 1) % args.eval_every == 0:
            _, val_metrics = evaluate(model, val_images, val_labels,
                                      args.device)
            mean = running / counted
            running, counted = 0.0, 0
            elapsed = time.time() - started
            print(f"step {step + 1:>5} loss {mean:.4f} "
                  f"val macro-F1 {val_metrics['macro_f1']:.4f} "
                  f"{elapsed:6.0f}s")
            with open(log_path, 'a', newline='') as handle:
                csv.writer(handle).writerow(
                    [step + 1, f"{mean:.6f}", f"{lr:.6f}",
                     f"{val_metrics['macro_f1']:.6f}", f"{elapsed:.1f}"])
            if best is None or val_metrics['macro_f1'] > best['val']['macro_f1']:
                state = {key: value.detach().cpu().clone()
                         for key, value in model.state_dict().items()}
                best = {'step': step + 1, 'state': state, 'val': val_metrics}

    # the single test evaluation, at the val-selected checkpoint
    model.load_state_dict(best['state'])
    test_images, test_labels = split_tensors('test')
    _, test_metrics = evaluate(model, test_images, test_labels, args.device)

    torch.save({'step': best['step'], 'model': best['state'],
                'args': vars(args)}, os.path.join(run_dir, 'best.pt'))
    result = {'arm': args.arm, 'seed': args.seed, 'best_step': best['step'],
              'val': best['val'], 'test': test_metrics, 'args': vars(args)}
    with open(os.path.join(run_dir, 'result.json'), 'w') as handle:
        json.dump(result, handle, indent=2)

    print(f"best step {best['step']} "
          f"val macro-F1 {best['val']['macro_f1']:.4f} "
          f"test macro-F1 {test_metrics['macro_f1']:.4f}, run at {run_dir}")


if __name__ == '__main__':
    main()
