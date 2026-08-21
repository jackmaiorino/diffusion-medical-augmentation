"""Plot DDPM and classifier training loss curves from the run logs."""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(os.path.dirname(HERE), 'runs')
# distinct dashes keep the overlapping arms readable in grayscale
ARMS = [('real_only', '-'), ('classical_aug', '-'), ('dup_real', '--'),
        ('synth_all', '-.'), ('synth_accepted', ':')]
SEEDS = [612, 613, 614]

# effective batch 64 (batch 32, 2-step accumulation) over 6,981 train images
STEPS_PER_EPOCH = 6981 / 64

ddpm = pandas.read_csv(os.path.join(RUNS, 'ddpm64', 'log.csv'))
val = pandas.read_csv(os.path.join(HERE, 'ddpm_val_loss.csv'))

fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.4))
left.plot(ddpm['step'] / STEPS_PER_EPOCH, ddpm['loss'], lw=0.9,
          label='train')
left.plot(val['step'] / STEPS_PER_EPOCH, val['val_loss'], 'o-', ms=3,
          lw=1.2, label='validation')
left.set_yscale('log')
left.set_xlabel('epoch')
left.set_ylabel('DDPM loss')
left.legend(fontsize=8)

for arm, style in ARMS:
    logs = [pandas.read_csv(os.path.join(
        RUNS, 'classifier', f'{arm}_s{seed}', 'log.csv')) for seed in SEEDS]
    steps = logs[0]['step']
    mean = sum(log['loss'] for log in logs) / len(logs)
    right.plot(steps, mean, style, lw=1.2, label=arm)
right.set_yscale('log')
right.set_xlabel('training step')
right.set_ylabel('classifier training loss (3-seed mean)')
right.legend(fontsize=8)

fig.tight_layout()
fig.savefig(os.path.join(HERE, 'loss_curves.png'), dpi=150)
print('wrote loss_curves.png')
