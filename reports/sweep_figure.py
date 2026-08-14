"""Plot rare-class copy rates and train-KID across the checkpoint sweep."""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas

HERE = os.path.dirname(os.path.abspath(__file__))
RARE = ['df', 'vasc', 'akiec']
STEPS = [10000, 20000, 30000, 50000, 70000, 90000, 100000]


def run_name(step):
    return 'ddpm64' if step == 100000 else f'ddpm64_s{step}'


def kid_of(run):
    name = f'fid_kid_{run}.csv'
    frame = pandas.read_csv(os.path.join(HERE, name))
    return frame.set_index('class')['kid_mean']


recal = pandas.read_csv(os.path.join(HERE, 'memorization_lesion_recal.csv'))
flags = recal.set_index(['run', 'class'])['flagged_fraction_lesion']

fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.4))
for cls in RARE:
    left.plot(STEPS, [100 * flags[(run_name(s), cls)] for s in STEPS],
              marker='o', ms=3.5, label=cls)
    right.plot(STEPS, [kid_of(run_name(s))[cls] for s in STEPS],
               marker='o', ms=3.5, label=cls)
for i, cls in enumerate(RARE):
    # open markers: guidance 1.0 at the final checkpoint
    color = left.lines[i].get_color()
    left.plot(100000, 100 * flags[('ddpm64_g1', cls)],
              marker='o', mfc='none', color=color)
    right.plot(100000, kid_of('ddpm64_g1')[cls],
               marker='o', mfc='none', color=color)

left.set_xlabel('training step')
left.set_ylabel('flagged as copies (%)')
left.legend()
right.set_xlabel('training step')
right.set_ylabel('KID x1000 vs train')
fig.tight_layout()
fig.savefig(os.path.join(HERE, 'memorization_sweep.png'), dpi=150)
print('wrote memorization_sweep.png')
