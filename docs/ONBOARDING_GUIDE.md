# Endotheliosis Quantifier Onboarding Guide

This guide keeps the more explanatory, beginner-friendly walkthrough for the project. The main [`README.md`](../README.md) stays focused on the current WSL/CUDA operating baseline, while this document is meant to help non-technical collaborators, future-you, or anyone returning to the repo after a long break.

## What This Project Is

Endotheliosis Quantifier (`eq`) is a deep learning pipeline for segmentation-first analysis of glomeruli histology images, with a maintained image-level endotheliosis scoring baseline built on Label Studio annotations.

At a high level, the project uses a two-stage idea:

1. Train a segmentation model on a mitochondria dataset to learn useful visual features such as edges, substructure, and boundaries.
2. Transfer that knowledge to glomeruli segmentation in kidney histology images.

That segmentation output now supports a first frozen-embedding ordinal scoring workflow for preeclampsia data.

## Key Ideas

- **Two-stage training**: mitochondria pretraining followed by glomeruli fine-tuning
- **Dynamic patching**: train on full images with augmentations and on-the-fly crops instead of permanently patchifying everything upfront
- **ROI identification**: segment glomeruli regions before later quantification steps
- **Label Studio-first scoring**: for the current preeclampsia workflow, the image-level grade attached in Label Studio is the primary supervised target
- **Union ROI semantics**: image-level scoring uses the full multi-component mask region rather than only the largest connected component

## What Data You Need

There are two main image data sources:

- **Mitochondria data**: electron microscopy images with mitochondria annotations
- **Glomeruli data**: H&E kidney histology images with binary glomeruli masks

You may also have subject-level metadata such as a scoring spreadsheet:

- **Subject metadata**: usually something like `subject_metadata.xlsx`
- This is still useful for audit trails, summaries, or legacy workflows, but it is not the default score source for the current preeclampsia quantification baseline

### Example Metadata Layout

| Glomerulus # | T19-1 | T19-2 | T30-1 | T30-2 | T30-3 |
|--------------|-------|-------|-------|-------|-------|
| 1            | 0.5   | 1     | 0     | 0.5   | 0     |
| 2            | 0     | 1     | 0.5   | 0     | 0     |
| 3            | 0     | 0.5   | 1     | 0     | 0     |
| 4            | 0.5   | 0.5   | 0     | 0     | 0     |

For the current scoring workflow, think of that spreadsheet as optional legacy metadata rather than the canonical supervised label table.

## Naming Convention

The recommended image naming pattern is:

```text
{SUBJECT_ID}-{IMAGE_NUMBER}
```

Examples:

- `T19-1`
- `T19-2`
- `Mouse_A-1`
- `Patient_001-1`

Supported image formats are typically `tif`, `tiff`, `png`, `jpg`, or `jpeg`. Masks should be binary masks with values `0` for background and `255` for foreground.

## Creating Masks With Label Studio

If masks are being created manually, Label Studio is a practical workflow. For the current preeclampsia quantification path, the Label Studio export is more than a mask source: it is also the canonical image-level grading source.

### 1. Install Label Studio

```bash
conda create -n label-studio python=3.9
conda activate label-studio
pip install label-studio
label-studio start
```

Then follow the [Label Studio Quick Start Guide](https://labelstud.io/guide/quick_start).

### 2. Export Masks And Annotation JSON

Inside Label Studio:

1. Open the project.
2. Click `Export`.
3. Choose `Brush labels to PNG`.
4. Also export the task annotations as JSON.
5. Download both the masks and the annotation export.

This is convenient because:

- the PNG export gives you one binary mask per image
- the JSON export preserves the image-level `grade` choice attached to that image/mask pair
- the quantification pipeline can recover duplicate annotations explicitly and prefer the latest graded annotation

### 3. Organize The Data

Recommended project structure:

```text
$EQ_RUNTIME_ROOT/raw_data/cohorts/<cohort_id>/
├── images/
│   ├── T19/
│   │   └── T19-1.jpg
│   └── T30/
│       └── T30-1.jpg
├── masks/
│   ├── T19/
│   │   └── T19-1_mask.png
│   └── T30/
│       └── T30-1_mask.png
├── scores/
│   └── labelstudio_annotations.json
└── metadata/
    └── subject_metadata.xlsx
```

## Suggested Raw Data Layout

At the runtime-root level, a practical layout looks like this:

```text
$EQ_RUNTIME_ROOT/raw_data/
├── lucchi/
│   ├── img/
│   └── label/
└── cohorts/
    └── <cohort_id>/
        ├── images/
        ├── masks/
        ├── scores/
        │   └── labelstudio_annotations.json
        └── metadata/
            └── subject_metadata.xlsx
```

This is cleaner than older hardcoded machine-specific paths and works better across WSL, Windows, and macOS.

## Before Training

After activating the environment, check what hardware the current machine exposes:

```bash
eq capabilities
eq mode --show
```

This is especially useful if you are moving between a CUDA desktop, a CPU-only machine, and a Mac.

## Validate Naming Early

```bash
eq validate-naming --data-dir "$EQ_RUNTIME_ROOT/raw_data/cohorts/lauren_preeclampsia"
eq validate-naming --data-dir "$EQ_RUNTIME_ROOT/raw_data/cohorts/lauren_preeclampsia" --strict
```

This helps catch avoidable problems before you launch a long training job.

## Why Dynamic Patching Matters

Dynamic patching is one of the more important design choices in this repo.

Instead of creating a permanent set of tiny training patches ahead of time, the pipeline can:

- keep full-resolution images intact
- apply augmentation to the full image
- sample training crops on the fly

Benefits:

- better augmentation diversity
- better preservation of image context
- less rigid preprocessing

For glomeruli data, that means your source layout can stay simple:

```text
$EQ_RUNTIME_ROOT/raw_data/cohorts/<cohort_id>/
├── images/
└── masks/
```

For mitochondria data installed with physical training and testing roots, keep those roots separate:

```text
$EQ_RUNTIME_ROOT/raw_data/mitochondria_data/
├── training/
│   ├── images/
│   └── masks/
└── testing/
    ├── images/
    └── masks/
```

Training uses the `training/` root and creates its internal dynamic train/validation split there. The `testing/` root is held out for explicit evaluation.

For quantification, the current contract is also intentionally simple:

- one scored example per image/mask pair
- score recovered from the Label Studio annotation export
- one union ROI built from all positive pixels in the mask
- frozen segmentation-encoder embeddings extracted from that union ROI

External scored cohorts use a separate runtime manifest contract. The active cohort table is:

```text
$EQ_RUNTIME_ROOT/raw_data/cohorts/manifest.csv
```

Each cohort has one localized working directory under:

```text
$EQ_RUNTIME_ROOT/raw_data/cohorts/<cohort_id>/
```

The manifest is image-level and records runtime-local asset paths, score linkage, optional mask path, treatment group, lane assignment, verification state, admission state, and file hashes. Original PhD or cloud source folders remain provenance sources; normal cohort work uses the localized runtime cohort directories.

Current external cohort rules:

- `lauren_preeclampsia` currently contributes 88 admitted `manual_mask_core` rows from Lauren's preeclampsia Label Studio workflow.
- `vegfri_dox` currently contributes 864 Label Studio export rows: 619 decoded `manual_mask_external` rows admitted after mask-quality review, 7 decoded rows missing scores, 228 foreign mixed-export rows, and 10 scored-only rows without decoded runtime images.
- `vegfri_mr` currently contributes 127 workbook image-level rows from the external-drive whole-field TIFF batches. Of these, 126 are localized for concordance/evaluation only and one workbook row, `8570-5`, remains unresolved because the matching TIFF was not found.
- Lucchi and other segmentation-install datasets are not part of the scored cohort manifest.

Refresh the cohort manifest with:

```bash
eq cohort-manifest
```

## Example Training Flow

### 1. Prepare Lucchi Images

```bash
eq organize-lucchi \
  --input-dir "$EQ_RUNTIME_ROOT/raw_data/lucchi" \
  --output-dir "$EQ_RUNTIME_ROOT/raw_data/mitochondria_data"
```

### 2. Train The Mitochondria Model

```bash
python -m eq.training.train_mitochondria \
  --data-dir "$EQ_RUNTIME_ROOT/raw_data/mitochondria_data/training" \
  --model-dir "$EQ_RUNTIME_ROOT/models/segmentation/mitochondria" \
  --epochs 50 \
  --batch-size 24 \
  --learning-rate 1e-3 \
  --image-size 256
```

On the powerful Apple Silicon MPS machine class, `24` is the current starting batch-size recommendation for `256x256` mitochondria training. Override it when throughput or stability requires a different value.

### 3. Train The Glomeruli Model

Use `--model-name` as a base name only. The trainer appends the descriptive run suffix automatically when it creates the artifact directory and exported `.pkl`. For example, `--model-name glomeruli_transfer_candidate` produces an artifact directory like `glomeruli_transfer_candidate-transfer_s1lr1e-3_s2lr_lrfind_e30_b12_lr1e-3_sz256/`; you do not pass that full suffixed name back into `--model-name`, and you should not hardcode the predicted final artifact path before training.

Transfer candidate:

```bash
python -m eq.training.train_glomeruli \
  --data-dir /absolute/path/to/raw_data/cohorts \
  --model-dir /absolute/path/to/glomeruli_models \
  --base-model /absolute/path/to/mito_supported_base.pkl \
  --epochs 30 \
  --batch-size 12 \
  --learning-rate 1e-3 \
  --image-size 256 \
  --crop-size 512 \
  --seed 42
```

Scratch candidate:

```bash
python -m eq.training.train_glomeruli \
  --data-dir "$EQ_RUNTIME_ROOT/raw_data/cohorts" \
  --model-dir "$EQ_RUNTIME_ROOT/models/segmentation/glomeruli" \
  --from-scratch \
  --epochs 50 \
  --batch-size 12 \
  --learning-rate 1e-3 \
  --image-size 256 \
  --crop-size 512 \
  --seed 42
```

On the powerful Apple Silicon MPS machine class, `12` is the current starting batch-size recommendation for `512x512` glomeruli crops. Override it when throughput or stability requires a different value.

For all-data glomeruli training, use the manifest-backed `raw_data/cohorts` registry root. It trains from admitted manifest rows in the `manual_mask_core` and `manual_mask_external` lanes, so unresolved, foreign, MR concordance-only, and scored-only rows stay out. For Lauren-only training, use `raw_data/cohorts/lauren_preeclampsia`. Raw backup trees are source material, not direct training roots. Generated manifests, audits, caches, and metrics belong under `derived_data` or `output`.

The dedicated training module CLI is the authoritative control surface. Optional YAML files are overlays, not the promotion contract, and the later artifact path is derived from the base `--model-name` plus the auto-generated run suffix. Transfer training with `--base-model` must load that artifact and copy compatible weights or the run stops. The `--from-scratch` candidate is the no-mitochondria-base comparator with an ImageNet-pretrained ResNet34 encoder, not a literal all-random initialization baseline. After training, inspect the produced `.pkl` path and reuse that exact path in downstream comparison or quantification commands.

### 4. Run The Current Quantification Baseline

```bash
eq prepare-quant-contract \
  --data-dir "$EQ_RUNTIME_ROOT/raw_data/cohorts/lauren_preeclampsia" \
  --segmentation-model "$EQ_RUNTIME_ROOT/models/segmentation/glomeruli/<your_model>.pkl" \
  --score-source labelstudio \
  --annotation-source "$EQ_RUNTIME_ROOT/raw_data/cohorts/lauren_preeclampsia/scores/labelstudio_annotations.json"

eq quant-endo \
  --data-dir "$EQ_RUNTIME_ROOT/raw_data/cohorts/lauren_preeclampsia" \
  --segmentation-model "$EQ_RUNTIME_ROOT/models/segmentation/glomeruli/<your_model>.pkl" \
  --score-source labelstudio \
  --annotation-source "$EQ_RUNTIME_ROOT/raw_data/cohorts/lauren_preeclampsia/scores/labelstudio_annotations.json" \
  --output-dir "$EQ_RUNTIME_ROOT/output/quantification/lauren_preeclampsia"
```

This writes:

- recovered Label Studio score tables and duplicate-resolution audit outputs
- union-ROI image and mask crops
- frozen segmentation encoder embeddings
- ordinal predictions with class probabilities and confidence proxies
- an HTML review report with 5-7 mixed example cases

## A Good Mental Model For The Repo

Think of the repository in two layers:

- **Code and configuration**: this belongs in Git
- **Data, trained models, logs, and outputs**: these stay local and are intentionally ignored

That split makes the repo much easier to maintain over time, especially when work happens across WSL, Windows, and macOS.

## If You Are Returning To The Repo After A Long Break

Start with this order:

1. Read the main [`README.md`](../README.md) for the current operational baseline.
2. Use this guide if you want the more explanatory version of the workflow.
3. Run:

```bash
conda activate eq
python -m eq --help
eq capabilities
eq mode --show
```

4. Only then start comparing branches, datasets, or historical experiments.
