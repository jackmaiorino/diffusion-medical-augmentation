# Diffusion Medical Augmentation

This repository evaluates whether a class-conditional 64 x 64 DDPM improves
rare-class classification on HAM10000. The formal comparison uses five
ResNet-18 training arms and seeds 612, 613, and 614.

Run every command below from the repository root in Windows PowerShell.

## Environment

The project was developed with Python 3.12 and CUDA 12.6 wheels. Google Chrome
is required only to build the PDF report.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.13.0+cu126 torchvision==0.28.0+cu126 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Source data and preprocessing

Download HAM10000 and place the two image archives after extraction at:

```text
data/raw/HAM10000_images_part_1/*.jpg
data/raw/HAM10000_images_part_2/*.jpg
```

`data/raw/HAM10000_metadata.csv` is versioned in this repository. Create the
lesion-disjoint split, image cache, and evaluation references with:

```powershell
.\.venv\Scripts\python.exe preprocessing\sort_images.py
.\.venv\Scripts\python.exe preprocessing\make_splits.py
.\.venv\Scripts\python.exe preprocessing\make_split_dirs.py
.\.venv\Scripts\python.exe preprocessing\make_cache.py --image-size 64
.\.venv\Scripts\python.exe preprocessing\export_real64.py --split train
.\.venv\Scripts\python.exe preprocessing\export_real64.py --split val
.\.venv\Scripts\python.exe -m pytest -q
```

The checked-in `data/splits.csv` is the formal seed-612 split. Recreating it
should produce the same 10,015 rows and keep every `lesion_id` in one split.

## DDPM training and sampling

The formal DDPM command is explicit here even where it matches script defaults:

```powershell
.\.venv\Scripts\python.exe src\train_ddpm.py --name ddpm64 --steps 100000 --batch-size 32 --grad-accum 2 --lr 0.0001 --image-size 64 --cfg-dropout 0.1 --ema-decay 0.9999 --seed 612 --log-every 100 --sample-every 2000 --ckpt-every 5000 --guidance 2.0 --workers 4 --device cuda
```

Generate the 1,000-image-per-class formal pool. Samples are staged and installed
atomically, so `--out` must be new or empty. The second command preserves a
complete pool for the checked-in classifier manifests.

```powershell
.\.venv\Scripts\python.exe src\sample_ddpm.py --ckpt runs\ddpm64\ckpt_100000.pt --out data\synthetic\ddpm64 --classes akiec,bcc,bkl,df,mel,nv,vasc --per-class 1000 --guidance 2.0 --steps 50 --batch-size 32 --seed 612 --image-size 64 --device cuda
Copy-Item -Recurse data\synthetic\ddpm64 data\synthetic\ddpm64_all
```

## Generative evaluation

First create the final-checkpoint memorization, held-out-distance, and FID/KID
results:

```powershell
.\.venv\Scripts\python.exe src\memorization.py --synthetic-dir data\synthetic\ddpm64 --out reports\memorization_ddpm64.csv --device cuda
.\.venv\Scripts\python.exe src\holdout_distance.py --synthetic-dir data\synthetic\ddpm64 --seeds 20 --device cuda
.\.venv\Scripts\python.exe src\eval_fid_kid.py --synthetic-dir data\synthetic\ddpm64 --out reports\fid_kid_ddpm64.csv --seed 612
```

For the full checkpoint sweep, generate and score the three rare classes at
the six earlier checkpoints, then score the guidance control:

```powershell
$sweepSteps = @(10000, 20000, 30000, 50000, 70000, 90000)
foreach ($sweepStep in $sweepSteps) {
    $sweepName = "ddpm64_s$sweepStep"
    .\.venv\Scripts\python.exe src\sample_ddpm.py --ckpt "runs\ddpm64\ckpt_$sweepStep.pt" --out "data\synthetic\$sweepName" --classes akiec,df,vasc --per-class 1000 --guidance 2.0 --steps 50 --batch-size 32 --seed 612 --image-size 64 --device cuda
    .\.venv\Scripts\python.exe src\memorization.py --synthetic-dir "data\synthetic\$sweepName" --device cuda
    .\.venv\Scripts\python.exe src\eval_fid_kid.py --synthetic-dir "data\synthetic\$sweepName" --seed 612
}
.\.venv\Scripts\python.exe src\sample_ddpm.py --ckpt runs\ddpm64\ckpt_100000.pt --out data\synthetic\ddpm64_g1 --classes akiec,df,vasc --per-class 1000 --guidance 1.0 --steps 50 --batch-size 32 --seed 612 --image-size 64 --device cuda
.\.venv\Scripts\python.exe src\memorization.py --synthetic-dir data\synthetic\ddpm64_g1 --device cuda
.\.venv\Scripts\python.exe src\eval_fid_kid.py --synthetic-dir data\synthetic\ddpm64_g1 --seed 612
```

After all desired memorization detail CSVs exist, apply the lesion-level
calibration, materialize the accepted pool, and compute the size-matched pool
statistics. Materialization reads the complete copy and does not alter it.

```powershell
.\.venv\Scripts\python.exe src\recalibrate_thresholds.py --device cuda
.\.venv\Scripts\python.exe preprocessing\materialize_accepted_pool.py --synthetic-dir data\synthetic\ddpm64_all --detail-csv data\eval\memorization_ddpm64_detail.csv
.\.venv\Scripts\python.exe src\pool_eval.py --synthetic-dir data\synthetic\ddpm64 --device cuda
.\.venv\Scripts\python.exe reports\sweep_figure.py
```

## Classifier experiment

Build the four data manifests. `classical_aug` reuses `real_only.csv` and turns
on training-time augmentation in code.

```powershell
.\.venv\Scripts\python.exe preprocessing\build_arms.py --seed 612
```

Run the complete formal 5 x 3 matrix and aggregate it:

```powershell
$classifierArms = @("real_only", "classical_aug", "dup_real", "synth_all", "synth_accepted")
$classifierSeeds = @(612, 613, 614)
foreach ($classifierArm in $classifierArms) { foreach ($classifierSeed in $classifierSeeds) { .\.venv\Scripts\python.exe src\train_classifier.py --arm $classifierArm --seed $classifierSeed --steps 6000 --batch-size 128 --lr 0.001 --weight-decay 0.05 --warmup 250 --eval-every 250 --workers 2 --device cuda:0 } }
.\.venv\Scripts\python.exe src\summarize_classifier.py --runs-dir runs\classifier
```

The summarizer refuses duplicate arm-seed runs, an incomplete 5 x 3 matrix,
or results whose formal hyperparameters differ from the command above. GPU
ordinal, output path, and data-loader worker count are execution settings, not
aggregation keys.

## Build the report

The PDF builder needs Google Chrome and network access to load Mermaid when the
source contains a Mermaid diagram.

```powershell
.\.venv\Scripts\python.exe reports\build_pdf.py reports\final_report.md -o reports\final_report.pdf
```

## Versioned and external artifacts

The repository versions source code, `data/splits.csv`, classifier arm
manifests, report-level CSVs, figures, Markdown, the built report PDFs, and
`reports/run_manifest.json`, which records SHA-256 hashes for the gitignored
artifacts behind the reported numbers (cache, checkpoint, synthetic pools,
classifier result files). Tracked files are already pinned by git history.
Large or readily derived artifacts are intentionally ignored:

- HAM10000 JPEGs under `data/raw`, `data/labeled`, and `data/splits`
- resized arrays under `data/cache`
- generated images under `data/synthetic`
- per-image evaluation details and exported reference PNGs under `data/eval`
- DDPM checkpoints, classifier checkpoints, logs, and `result.json` files under
  `runs`

A clean checkout can regenerate these files with the commands above. To audit
the exact completed run without rerunning GPU work, obtain the ignored
checkpoints, synthetic pools, evaluation details, and classifier run folders
from the project authors and restore them at the paths listed above.
