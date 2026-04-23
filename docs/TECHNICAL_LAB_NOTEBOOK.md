# Technical Lab Notebook: Endotheliosis Quantifier

**Updated**: April 2, 2026  
**Branch**: `master`  
**Project**: Endotheliosis Quantifier (`eq`)  
**Purpose**: Technical status notebook for the checked-in `master` branch

## Scope Note

This notebook describes what the current `master` branch actually implements today.

It is not a project vision document. In particular, it distinguishes between:

- implemented segmentation and data-preparation workflows
- partially implemented or legacy inference code
- still-unimplemented learned quantification and feature-extraction work

## Executive Summary

The current `master` branch is best described as a **segmentation-first FastAI/PyTorch codebase** for:

1. mitochondria pretraining on EM-style data
2. glomeruli segmentation in histology images
3. supporting utilities for data preparation, metadata processing, mask auditing, visualization, and environment detection

The strongest maintained path in this branch is the segmentation workflow. The repository does **not** currently ship a completed learned regression pipeline for endotheliosis scoring. Quantification code exists, but what is checked in today is mainly a heuristic openness/grade calculation rather than a trained feature-extraction plus regression stack.

## Current Baseline

The current branch baseline matches the main repository docs:

- primary development target: WSL on Windows with CUDA-capable PyTorch
- package source: `src/eq/`
- raw datasets: `data/raw_data/`
- generated artifacts: `data/derived_data/`, `models/`, `logs/`, `output/`
- current operational branch: `master`

For the higher-level repo orientation, see:

- [README.md](../README.md)
- [ONBOARDING_GUIDE.md](ONBOARDING_GUIDE.md)
- [OUTPUT_STRUCTURE.md](OUTPUT_STRUCTURE.md)
- [SEGMENTATION_ENGINEERING_GUIDE.md](SEGMENTATION_ENGINEERING_GUIDE.md)

## Repository Layout

The current codebase standardizes on repo-relative paths:

```text
endotheliosis_quantifier/
├── configs/
├── data/
│   ├── raw_data/
│   └── derived_data/
├── logs/
├── models/
│   └── segmentation/
│       ├── mitochondria/
│       └── glomeruli/
├── output/
├── src/eq/
└── tests/
```

Path defaults are defined in [`src/eq/utils/paths.py`](../src/eq/utils/paths.py):

- raw data: `data/raw_data`
- derived data: `data/derived_data`
- cache: `data/derived_data/cache`
- models: `models`
- logs: `logs`

## Problem Framing

The project remains oriented around automated analysis of glomerular histology for endotheliosis-related work, with mitochondria pretraining used as a transfer-learning stage for segmentation.

What the current branch supports directly:

- binary segmentation of mitochondria or glomeruli
- dynamic patching from full images
- metadata standardization for glomeruli scoring spreadsheets
- mask-pair auditing and visualization

What it does **not** currently support as a completed, production-ready workflow:

- learned feature extraction for downstream quantification
- trained regression models such as Random Forest, SVR, XGBoost, or neural regressors for endotheliosis scoring
- a clean, fully maintained end-to-end quantification pipeline from raw image to validated learned score

## Data Model And Input Layouts

The supported segmentation training layout is a full-image root:

```text
<data_root>/
├── images/
└── masks/
```

Full images are loaded directly and crops are sampled during training.

For mitochondria, the installed full-image layout uses separate physical roots:

```text
data/derived_data/mitochondria_data/
├── training/
│   ├── images/
│   └── masks/
└── testing/
    ├── images/
    └── masks/
```

The `training/` root is the training input; the dynamic dataloader creates the train/validation split internally. The `testing/` root is held out for explicit evaluation.

## Data Preparation Workflow

### Raw Data Validation

The CLI includes a naming validator for raw glomeruli projects:

```bash
eq validate-naming --data-dir data/raw_data/<your_project>
```

### Lucchi Preparation

The Lucchi organizer and image extraction flow still exist:

```bash
eq extract-images \
  --input-dir data/raw_data/lucchi \
  --output-dir data/derived_data/mito
```

### Patchification

The main derived-data builder is:

```bash
eq organize-lucchi \
  --input-dir data/raw_data/lucchi \
  --output-dir data/derived_data/mitochondria_data
```

Current behavior of `organize-lucchi`:

- creates `training/images`, `training/masks`, `testing/images`, and `testing/masks`
- preserves the physical held-out `testing/` root for explicit evaluation
- produces the mitochondria full-image training root used by the training examples

The checked-in branch does **not** use the older bare repo-root `derived_data/` convention as its primary documentation target.

## Segmentation Architecture

### Core Contract

The current branch follows a consistent binary segmentation contract:

- class `0`: background
- class `1`: foreground
- `n_out=2`
- masks normalized to `0/1`
- FastAI default loss selection retained for this output format

This aligns with the engineering guidance in [SEGMENTATION_ENGINEERING_GUIDE.md](SEGMENTATION_ENGINEERING_GUIDE.md).

### Model Choice

Current training code uses:

- FastAI v2 with PyTorch
- `unet_learner(...)`
- `resnet34` encoder backbone
- metrics including `Dice` and `JaccardCoeff()`

### Transform Pipeline

The current loader behavior is:

- `item_tfms`: primarily resize/geometry placement
- `batch_tfms`: `IntToFloatTensor()`, augmentation, ImageNet normalization, and mask preprocessing

This matters because the older documentation pattern that put `aug_transforms(...)` in `item_tfms` is no longer the branch truth.

## Training Strategy

### Stage 1: Mitochondria Pretraining

Primary entrypoint:

```bash
python -m eq.training.train_mitochondria \
  --data-dir data/derived_data/mitochondria_data/training \
  --model-dir models/segmentation/mitochondria \
  --epochs 50 \
  --batch-size 24 \
  --learning-rate 1e-3 \
  --image-size 256
```

Current defaults in the training module:

- epochs: `50`
- batch size: machine-aware; currently `24` on the powerful Apple Silicon MPS machine class when using `256x256` crops
- learning rate: `1e-3`
- image size: `256`
- training mode: `dynamic_full_image_patching`

### Stage 2: Glomeruli Training

Primary entrypoint:

```bash
python -m eq.training.train_glomeruli \
  --data-dir data/raw_data/<your_glomeruli_project>/training_pairs \
  --model-dir models/segmentation/glomeruli \
  --epochs 50 \
  --batch-size 12 \
  --learning-rate 1e-3 \
  --image-size 256 \
  --crop-size 512
```

The glomeruli training root must contain paired full-image `images/` and `masks/` directories under `raw_data`. Raw project backups are source material; curate paired files into `training_pairs` before running model training. Generated manifests, audits, caches, and metrics belong under `derived_data`.

Important nuance:

- the README example above is the recommended workflow documentation
- the `train_glomeruli.py` module resolves a machine-aware default batch size and currently starts at `12` on the powerful Apple Silicon MPS machine class when using `512x512` crops
- `configs/glomeruli_finetuning_config.yaml` documents `epochs=30`, `batch_size=12`, and `learning_rate=5e-4`

So there is still some configuration drift inside `master`, and the notebook should not claim a single universal glomeruli default beyond what each entrypoint actually sets.

### Dynamic Patching

Dynamic patching is a real current feature, not just a future idea.

The checked-in loader stack supports:

- loading full images from `images/`
- resolving masks from `masks/`
- on-the-fly crops
- positive-aware cropping for sparse targets

Key positive-aware cropping controls:

- `positive_focus_p`
- `min_pos_pixels`
- `pos_crop_attempts`

## Data Validation And Loader Behavior

Current loader behavior is stricter than the older notebook described.

Implemented validation includes:

- early image-mask pairing checks for full-image dynamic training roots
- failure when expected masks are missing
- basic sampled mask-content sanity checks on validation items
- static patch loaders retained only for legacy audit/conversion inspection

This is one of the more mature parts of the current branch.

## Output Structure

### Training Artifacts

The current training scripts create per-run folders under the model directory.

Typical mitochondria pattern:

```text
models/segmentation/mitochondria/
└── <model_name>-pretrain_e<epochs>_b<batch>_lr<lr>_sz<size>/
```

Typical glomeruli pattern:

```text
models/segmentation/glomeruli/
├── transfer/
│   └── <model_name>-transfer_e<epochs>_b<batch>_lr<lr>_sz<size>/
└── scratch/
    └── <model_name>-scratch_e<epochs>_b<batch>_lr<lr>_sz<size>/
```

Current artifact filenames are prefixed by the run folder name, for example:

- `<model>_training_loss.png`
- `<model>_lr_schedule.png`
- `<model>_metrics.png`
- `<model>_validation_predictions.png`
- `<model>_training_history.tsv`
- `<model>_splits.json`
- `<model>_run_metadata.txt`

### General Output Manager

Separate from training-artifact folders, the repository also has an `OutputManager` that creates:

```text
output/<data_source>/
├── models/
├── plots/
├── results/
└── cache/
```

This path is used by the older production pipeline code.

## Metadata And Spreadsheet Processing

Metadata processing is implemented and useful today.

The current metadata processor can:

- read glomeruli scoring matrices from Excel
- clean summary rows and unnamed columns
- convert wide subject-by-column data into long format
- produce standardized columns:
  - `subject_id`
  - `glomerulus_id`
  - `score`
- create subject summaries
- run metadata-quality validation

Primary CLI entrypoint:

```bash
eq metadata-process \
  --input-file data/raw_data/<project>/subject_metadata.xlsx \
  --output-dir data/derived_data/<project>/metadata
```

## Quantification Status

### What Exists

The current `master` branch now contains a maintained baseline quantification path under [`src/eq/quantification/pipeline.py`](../src/eq/quantification/pipeline.py) plus supporting contract and score-recovery utilities.

What exists today:

- Label Studio-first score recovery from image-level annotation exports
- explicit duplicate-annotation reconciliation with audit outputs
- image-level scored-example tables joined to raw image/mask pairs
- union-ROI extraction over the full multi-component mask
- frozen segmentation-encoder embedding extraction
- grouped ordinal image-level endotheliosis prediction
- prediction exports with class probabilities, expected score, top-two margin, and entropy
- an HTML review artifact with selected example cases
- the older openness heuristic in [`src/eq/evaluation/quantification_metrics.py`](../src/eq/evaluation/quantification_metrics.py), now best treated as an audit feature rather than the primary learned model

### What Does Not Yet Exist As A Matured Workflow

The current `master` branch still does **not** provide:

- calibrated uncertainty estimates
- per-glomerulus labels inside multi-glomerulus images
- validated subject-level endotheliosis prediction as the primary target
- a fully production-hardened deployment path from predicted masks to final score
- faithful attribution methods for the embedding model

The CLI commands `extract-features` and `quantify` still print that they are not yet implemented, so the maintained quantification surface is currently `prepare-quant-contract` plus `quant-endo`.

## Inference Status

Inference support is mixed.

### Present In The Repo

The branch includes:

- glomeruli inference modules
- mitochondria inference modules
- a production-pipeline module intended to load existing models and generate outputs

### Caveat

This code should be treated as **partially maintained** rather than a fully trusted production surface.

Reasons:

- some inference paths still reference older assumptions
- some modules import code that is not present in the tracked source tree
- the production pipeline contains legacy dependencies and undefined classes

So the presence of inference files should not be interpreted as proof of a clean end-to-end supported workflow.

## Recommended Entry Points On `master`

For the current branch, the safest supported workflow is:

1. validate raw naming with `eq validate-naming`
2. prepare derived data with `eq extract-images` or `eq process-data`
3. export Label Studio PNG masks plus annotation JSON when image-level quantification labels are needed
4. process metadata with `eq metadata-process` if spreadsheets are present for legacy support or audit context
5. run `eq prepare-quant-contract` to recover score-linked image/mask pairs
6. run `eq quant-endo` for the current embedding-first image-level baseline
7. audit masks with `eq audit-derived`
8. train mitochondria via `python -m eq.training.train_mitochondria`
9. train glomeruli via `python -m eq.training.train_glomeruli`

The dedicated training modules are more trustworthy than the older orchestration-style CLI routes for heavy model training.

## Current Known Gaps

As of April 2, 2026 on `master`, the main known gaps are:

- the maintained learned quantification path is still an image-level baseline rather than a biologically explicit per-glomerulus model
- uncertainty outputs are confidence proxies, not calibrated probabilities
- interpretation in the HTML review report is descriptive rather than attribution-faithful
- inference exists but is not uniformly current
- some configs, README examples, and code defaults still disagree on exact training hyperparameters

## Current Status

The current `master` branch should be described as:

**A maintained segmentation repository with useful data-preparation utilities, a working Label Studio-first image-level learned quantification baseline, partial inference code, and remaining scientific/production gaps around calibration, deployment, and per-glomerulus labeling.**

That is more accurate than either older extreme: the branch is no longer heuristic-only for quantification, but it is also not yet a finished clinically trustworthy scoring system.
