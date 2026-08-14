# Diffusion-Based Data Augmentation for Imbalanced Medical Image Classification

**MSML612 Group Project, Final Report**  
Team (2 members): Jack Maiorino, Rithvik Kommareddy  
Date: August 2026

---

## Abstract

We asked whether diffusion-generated images can improve a downstream classifier on the rare
classes of HAM10000, a severely imbalanced dermoscopic benchmark. We trained a
class-conditional 37.1M-parameter pixel-space DDPM from scratch on a leak-free lesion-level
split and generated 1,000 images per class. The central finding is negative, and we argue it
is more informative than the comparison we originally planned: under class-balanced training
the model does not learn its rare classes, it memorizes them. A nearest-neighbor check in two
perceptual spaces, calibrated on real-to-real distances with whole-lesion exclusion, flags
97% of generated dermatofibroma images as near-copies of specific training images, and the
copies cover 81 of the 84 training images. Copying rises monotonically with training while
similarity to held-out data never materializes: the generated pool sits at median LPIPS
0.008 from the training set but 0.155 from the validation set, where real validation images
sit 0.160 from each other. Filtering the flagged images does not rescue the pools; the
survivors score worse against held-out data than the unfiltered sets. The downstream
comparison across five regimes (real only, classical augmentation, duplicated-real
oversampling, unfiltered synthetic, filtered synthetic) quantifies what these copies are
worth to a classifier. [pending: one-sentence summary of classifier results, Section 7]

## 1. Problem Statement and Research Question

Clinical image datasets are long-tailed, and classifiers underperform exactly where reliable
detection matters most. Classical augmentation only perturbs existing pixels; generative
augmentation promises new samples. Diffusion models are the state of the art in image
generation and provide native class-conditional control, which motivated our original
question: can diffusion-generated images meaningfully improve a rare-class classifier? The
project's findings sharpened that question into the one this report answers:

> When a diffusion model is trained from scratch on tens of images per class, are its
> "synthetic" rare-class samples new images at all, and does anything they add to a
> classifier survive controlling for trivial oversampling?

## 2. Data

HAM10000 (Tschandl et al., 2018) contains 10,015 dermoscopic images in 7 classes, from 66.9%
melanocytic nevi (nv) down to 1.1% dermatofibroma (df). Many lesions are photographed more
than once (4,501 of 10,015 images share a lesion with another image), so a naive image-level
split leaks near-duplicates across the train/test boundary: averaged over 200 random
image-level splits, 34.9% of test images share a lesion with a training image, and the rate
is worst on the minority classes (mel 62.1%, df 51.1%). We therefore split at the lesion
level, stratified by class, 70/15/15, seed 612, keeping all images; an assertion verifies
that no lesion crosses splits. Full details are in the interim report; the resulting counts:

| Class | Train | Val | Test | Total |
|-------|------:|----:|-----:|------:|
| nv    | 4,679 | 1,015 | 1,011 | 6,705 |
| mel   |   776 |   163 |   174 | 1,113 |
| bkl   |   765 |   165 |   169 | 1,099 |
| bcc   |   354 |    77 |    83 |   514 |
| akiec |   227 |    55 |    45 |   327 |
| vasc  |    96 |    23 |    23 |   142 |
| df    |    84 |    15 |    16 |   115 |
| **All** | **6,981** | **1,513** | **1,521** | **10,015** |

df, with 84 training images from 51 distinct lesions, is the stress test the study turns on.

## 3. Generative Model

The generator is a class-conditional DDPM (Ho et al., 2020) built on the diffusers
`UNet2DModel` (von Platen et al., 2022), trained from scratch in pixel space at 64x64:
37.1M parameters, channels 128/256/256/256 with self-attention at 16x16 and 8x8, a cosine
noise schedule with T = 1000 (Nichol and Dhariwal, 2021), and classifier-free guidance
(Ho and Salimans, 2021) with label dropout 0.1. One shared model serves all classes so the
rare classes can reuse representations learned from the common ones, and training uses
class-balanced sampling so they receive equal gradient exposure. As run: 100,000 steps of
AdamW at 1e-4, effective batch 64 (batch 32 with 2-step gradient accumulation, which avoids
VRAM spill on our 12 GB GPU), EMA decay 0.9999, 13.0 hours on a single GPU. Sampling uses
DDIM at 50 steps with guidance 2.0 and seed 612; we generated 1,000 images per class.

![Samples from the final checkpoint, one row per class](ddpm_samples_100k.png)

*Figure 1. EMA samples at 100k steps, guidance 2.0. Legible lesion morphology in every
class; the question the rest of the report answers is where that morphology comes from.*

## 4. Generative Quality by the Standard Metrics

Per-class FID and KID against the real training images, with a real-vs-real "reference
floor" (each class's train set split in half and scored against itself). KID is reported
x1000. Because four classes have so few images that their FID floor exceeds any plausible
synthetic score, FID is uninformative there and we rank classes by KID only.

| Class | FID | FID floor | KID | KID floor |
|-------|----:|----------:|----:|----------:|
| akiec | 46.52 | 70.96 | 14.10 | -0.35 |
| bcc   | 53.91 | 62.39 | 24.04 | 0.53 |
| bkl   | 59.58 | 41.07 | 29.42 | -0.15 |
| df    | 38.33 | 107.42 | 5.68 | 3.75 |
| mel   | 61.61 | 42.12 | 33.65 | 1.07 |
| nv    | 52.30 | 11.14 | 24.56 | -0.04 |
| vasc  | 43.40 | 105.67 | 4.14 | -5.27 |

Taken at face value the two rarest classes, df and vasc, are the model's best. That reading
is wrong, and the reason is the subject of Section 5: a generator that copies its training
set scores near-perfect train-referenced KID by construction. These numbers are kept here
because they are the metrics the field defaults to, and because the gap between this table
and Section 5 is itself a finding: train-referenced fidelity metrics cannot be read as
generative quality without a memorization check next to them.

## 5. Memorization

### 5.1 Detection method

For every generated image we compute the nearest-neighbor distance to its class's real
training images in two spaces: LPIPS (Zhang et al., 2018), computed through an exact
algebraic decomposition that reduces it to squared Euclidean distance on precomputed
vectors, and cosine distance on unit-normalized InceptionV3 pool features. An image is
flagged when it falls below the 5th percentile of that class's real-to-real leave-one-out
nearest-neighbor distribution in either space. Flagged images are quarantined, not deleted.

Calibration matters more than the metric. HAM10000's repeated lesion views mean a plain
leave-one-out null still contains sibling images of the same lesion, which drags the null
distances down and makes the detector too lenient. We therefore recalibrated by excluding
the entire lesion, not just the identical image, from each real image's neighbor set. This
roughly doubles most thresholds (akiec LPIPS 0.033 to 0.092). All headline numbers below
use the lesion-calibrated thresholds; the union of two 5% rules has a nominal false-positive
rate between 5 and 10%, which is the baseline to read the table against.

### 5.2 The model copies its rare classes

| Class | Flagged (image null) | Flagged (lesion null) | Surviving images |
|-------|---------------------:|----------------------:|-----------------:|
| df    | 93.4% | 97.0% | 30 |
| vasc  | 89.8% | 92.3% | 77 |
| akiec | 50.1% | 79.1% | 209 |
| bcc   | 35.4% | 56.2% | 438 |
| bkl   |  4.1% | 25.0% | 750 |
| mel   |  4.6% | 16.9% | 831 |
| nv    |  0.7% |  2.8% | 972 |

The flagged fraction is near-monotonic in inverse class size, the signature of repeated
exposure: under class-balanced sampling over 100k steps, each df training image was
presented roughly 11,000 times. Visual inspection confirms the flags are real copies, not a
threshold artifact (Figure 2), and near-threshold pairs are still lightly perturbed versions
of a specific training image, so if anything the detector under-flags. Only nv, with 4,679
training images, stays below the nominal false-positive band.

![Flagged generated images beside their nearest training images](memcheck_worst_pairs.png)

*Figure 2. The lowest-distance flagged pairs: each column pairs a generated image with its
nearest real training image. These are near-pixel copies.*

The copying is broad, not collapse onto a few prototypes: df's flags trace back to 81 of its
84 training images (50 of 51 lesions), vasc's to 88 of 96, akiec's to 163 of 227, and the
most-copied df source accounts for only 8% of flags. The model has, in effect, learned to
resample its rare-class training sets.

### 5.3 More training buys copying, not quality

We scored checkpoints from 10k to 100k steps, plus a guidance 1.0 probe at 100k (Figure 3).
Flag rates rise steadily with training while train-referenced KID falls. But these two
curves are not independent: copies are close to the empirical training distribution by
definition, so train-KID improves as copying worsens. The trajectory is also not perfectly
monotonic at the point level (the 70k checkpoint beats 90k on both axes, and guidance 1.0
cuts df's flag rate from 97% to 84% at similar KID), so sampling choices move the numbers.
What no tested setting produced is an operating point that is simultaneously novel and
faithful: early checkpoints are novel but visibly unconverged (KID 52 to 106), late ones
are faithful by copying.

![Copy rate and train-KID across checkpoints](memorization_sweep.png)

*Figure 3. Left: lesion-calibrated flag rate. Right: KID x1000 vs train. Open markers:
guidance 1.0 at 100k. Copying and train-referenced "quality" improve together, which is
exactly why train-referenced quality cannot be trusted here.*

### 5.4 The held-out test: close to train, not to fresh data

A generator that learned the distribution should be about as close to a held-out sample of
that distribution as to its training set. A copier should not. Median LPIPS nearest-neighbor
distances, with the validation set as held-out reference:

| Class | Pool to train | Pool to val | Real val to val |
|-------|--------------:|------------:|----------------:|
| df    | 0.008 | 0.155 | 0.160 |
| vasc  | 0.007 | 0.153 | 0.173 |
| akiec | 0.037 | 0.135 | 0.145 |
| bcc   | 0.080 | 0.145 | 0.143 |
| bkl   | 0.116 | 0.148 | 0.148 |
| mel   | 0.125 | 0.160 | 0.149 |
| nv    | 0.110 | 0.125 | 0.096 |

df and vasc sit twenty times closer to specific training images than to the validation set,
while their distance to validation images is no better than the distance between two real
validation images. That is the copying signature in one row: intimacy with the training set
without any corresponding closeness to fresh data. nv, by contrast, is roughly equidistant
to both, which is what generalization looks like (its higher val-to-val floor reflects the
class's size and diversity).

### 5.5 Filtering does not rescue the pools

The obvious salvage is to keep only the unflagged images. Measured against the validation
set, that makes things worse. KID x1000 vs val, with the real train set scored against val
as the reference:

| Class | Real train | All synthetic | Accepted only | Near-dup within pool (all / accepted) |
|-------|-----------:|--------------:|--------------:|:-------------------------------------:|
| akiec | 1.5 | 24.0 | 59.4 | 91% / 78% |
| bcc   | 4.1 | 43.6 | 69.9 | 78% / 70% |
| bkl   | 0.9 | 36.6 | 38.8 | 51% / 49% |
| df    | 6.7 | 15.0 | 26.1 | 97% / 30% |
| mel   | 1.3 | 40.5 | 43.0 | 41% / 41% |
| nv    | -0.1 | 26.6 | 28.3 | 1% / 0% |
| vasc  | 8.7 | 21.1 | 33.7 | 94% / 64% |

Two conclusions. First, the unfiltered rare-class pools owe their apparent fidelity to the
copies: against held-out data, df's KID is 15.0, not the 5.7 the train-referenced table
suggested. Second, the accepted pools are not hidden gems: removing the copies removes the
most realistic images and leaves a lower-fidelity tail, and even the survivors duplicate
each other internally (near-duplicate rates against a real-data calibration of about 5%).
The generator repeats itself on every class except nv, whether or not the repetition matches
a training image.

## 6. Downstream Comparison

The generative analysis predicts that rare-class synthetic data should behave like
duplicated real images, not like new information. The classifier experiment tests that
directly. The measurement instrument is unchanged from the interim report: a fixed
ResNet-18 trained from scratch at 64x64 with class-balanced sampling, identical
architecture, optimizer, schedule, and seeds across regimes; only the training-set
composition changes. Five regimes, same training budget each:

1. Real data only.
2. Real + classical augmentation (flips, rotation, color jitter).
3. Real + duplicated-real oversampling (matching regime 5's added volume with copies of
   real images: the control the memorization result makes essential).
4. Real + unfiltered synthetic (1,000 per class).
5. Real + filtered synthetic (the accepted pools of Section 5.5).

Validation selects checkpoints and mixing ratios; the test set is evaluated once per
regime. The primary metric is per-class F1 on the rare classes as a delta over regime 1,
with macro-F1 and balanced accuracy secondary. The decisive comparison is regime 5 vs
regime 3: if filtered synthetic data carries information beyond its training sources, it
should beat duplicated real images; if it is repackaged copies, the two should tie.

| Regime | Rare-class F1 delta | Macro F1 | Balanced acc. |
|--------|:-------------------:|:--------:|:-------------:|
| 1 Real only | reference (0.343) | 0.439 +/- 0.014 | 0.422 |
| 2 Classical aug | +0.231 | 0.604 +/- 0.025 | 0.611 |
| 3 Duplicated-real | -0.054 | 0.420 +/- 0.002 | 0.446 |
| 4 Unfiltered synthetic | -0.044 | 0.432 +/- 0.012 | 0.414 |
| 5 Filtered synthetic | -0.052 | 0.416 +/- 0.029 | 0.399 |

Each regime is the mean over three seeds (612, 613, 614); rare-class F1 averages
akiec, df, and vasc. Per-run numbers are in reports/classifier_runs.csv and the
per-arm spread in reports/classifier_summary.csv.

[pending: one paragraph of interpretation from the classifier runs]

## 7. Discussion

**Mechanism.** Class-balanced sampling, our deliberate defense against rare-class neglect,
is also what drove the memorization: it multiplied each df image's exposure by roughly 12x
relative to natural frequency, to about 11,000 presentations over the run. Exposure and class size are coupled by that design, so we
cannot fully separate "too few lesions" from "seen too often", but the practical lesson
holds either way: at tens of images per class, a 37M-parameter from-scratch DDPM converges
to its rare-class training data well before it produces novel morphology.

**Why we did not retrain.** The candidate mitigations, in the order we would try them, are
capping the oversampling weight (e.g. 90% empirical + 10% balanced sampling, which cuts
per-image exposure by 5 to 10x), dihedral train-time augmentation with an
augmentation-aware detector, a smaller model, scheduled guidance, and memorization-based
early stopping. Each costs a full 13-hour training run plus re-scoring, and a single
unreplicated attempt would support weaker claims than the analysis above. We chose to spend
the remaining budget characterizing the negative result and measuring its downstream cost.
The honest statement of scope: across the tested checkpoints, guidance scales, and sampler,
no acceptable operating point was observed; the experiment does not estimate true copy
prevalence (the detector's null rate is 5 to 10%, and it under-flags visually confirmed
near-copies), identify a single cause, or rule out untested mitigations.

**Limitations.** Everything is established at 64x64 on one training run. The validation
references for the rarest classes are small (df 15, vasc 23 images), so the held-out KID
values carry wide error bars, though the direction of every comparison is consistent across
classes. Flag rates are detection rates under a calibrated threshold, not prevalence
estimates. The ethical stake is worth stating plainly: in a medical setting, synthetic
images that are lightly perturbed copies of real patient images are a privacy problem
masquerading as augmentation, and a memorization check like ours is cheap insurance any
such pipeline should carry.

## 8. Deviations from the Interim Plan

- **The StyleGAN2-ADA baseline was dropped.** Once the memorization result landed, the
  informative comparison became synthetic-vs-duplicated-real, not diffusion-vs-GAN. A GAN
  trained on 84 images would face the same memorization question without changing the
  conclusion, and the compute went to the checkpoint sweep and held-out analysis instead.
- **Four regimes became five.** The replication control the interim report mentioned in
  passing (duplicated-real oversampling) was promoted to the decisive comparator, and the
  diffusion regime split into unfiltered and filtered variants.
- **The guidance sweep narrowed.** Instead of {1.0, 2.0, 3.0} for sample quality, we probed
  guidance 1.0 specifically for its effect on copying (Figure 3).

## 9. Related Work

The diffusion foundations are as in the interim report: DDPM (Ho et al., 2020), the cosine
schedule (Nichol and Dhariwal, 2021), classifier-free guidance (Ho and Salimans, 2021), and
the diffusers implementation (von Platen et al., 2022). Frid-Adar et al. (2018) and
Trabucco et al. (2024) report generative-augmentation gains; our result is a caution about
the regime where those gains are claimed with the least data. On measurement, FID is due to
Heusel et al. (2017), KID to Binkowski et al. (2018), and LPIPS to Zhang et al. (2018).
Meehan et al. (2020) formalized the three-sample data-copying test our held-out comparison
follows. Carlini et al. (2023) extracted training images from large diffusion models;
Dar et al. (2025) documented the same phenomenon in medical diffusion models, including
copies that survive augmented training, which informed our choice not to treat augmentation
alone as a fix.

## 10. References

1. Ho, J., Jain, A., and Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models.* NeurIPS.
2. Nichol, A., and Dhariwal, P. (2021). *Improved Denoising Diffusion Probabilistic Models.* ICML.
3. Ho, J., and Salimans, T. (2021). *Classifier-Free Diffusion Guidance.* NeurIPS Workshop.
4. Tschandl, P., Rosendahl, C., and Kittler, H. (2018). *The HAM10000 Dataset.* Scientific Data.
5. Frid-Adar, M., et al. (2018). *GAN-based Synthetic Medical Image Augmentation for Increased CNN Performance in Liver Lesion Classification.* Neurocomputing.
6. Trabucco, B., et al. (2024). *Effective Data Augmentation with Diffusion Models.* ICLR.
7. Heusel, M., et al. (2017). *GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium.* NeurIPS.
8. Binkowski, M., et al. (2018). *Demystifying MMD GANs.* ICLR.
9. Zhang, R., et al. (2018). *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric.* CVPR.
10. Meehan, C., Chaudhuri, K., and Dasgupta, S. (2020). *A Non-Parametric Test to Detect Data-Copying in Generative Models.* AISTATS.
11. Carlini, N., et al. (2023). *Extracting Training Data from Diffusion Models.* USENIX Security.
12. Dar, S. U. H., et al. (2025). *Unconditional Latent Diffusion Models Memorize Patient Imaging Data.* Nature Biomedical Engineering.
13. Karras, T., et al. (2020). *Training Generative Adversarial Networks with Limited Data (StyleGAN2-ADA).* NeurIPS.
14. von Platen, P., et al. (2022). *Diffusers: State-of-the-Art Diffusion Models.* GitHub. https://github.com/huggingface/diffusers
