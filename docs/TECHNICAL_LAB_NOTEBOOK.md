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

- [README.md](/home/ncamarda/endotheliosis_quantifier/README.md)
- [ONBOARDING_GUIDE.md](/home/ncamarda/endotheliosis_quantifier/docs/ONBOARDING_GUIDE.md)
- [OUTPUT_STRUCTURE.md](/home/ncamarda/endotheliosis_quantifier/docs/OUTPUT_STRUCTURE.md)
- [SEGMENTATION_ENGINEERING_GUIDE.md](/home/ncamarda/endotheliosis_quantifier/docs/SEGMENTATION_ENGINEERING_GUIDE.md)

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

Path defaults are defined in [`src/eq/utils/paths.py`](/home/ncamarda/endotheliosis_quantifier/src/eq/utils/paths.py):

- raw data: `data/raw_data`
- derived data: `data/derived_data`
- cache: `data/derived_data/cache`
- models: `models`
- logs: `logs`

## Problem Framing

The project remains oriented around automated analysis of glomerular histology for endotheliosis-related work, with mitochondria pretraining used as a transfer-learning stage for segmentation.

What the current branch supports directly:

- binary segmentation of mitochondria or glomeruli
- preparation of raw image and mask data into derived patch datasets
- dynamic patching from full images
- metadata standardization for glomeruli scoring spreadsheets
- mask-pair auditing and visualization

What it does **not** currently support as a completed, production-ready workflow:

- learned feature extraction for downstream quantification
- trained regression models such as Random Forest, SVR, XGBoost, or neural regressors for endotheliosis scoring
- a clean, fully maintained end-to-end quantification pipeline from raw image to validated learned score

## Data Model And Input Layouts

Two dataset layouts are supported by the loader stack.

### 1. Static Patch Datasets

For pre-generated patches:

```text
<data_root>/
├── image_patches/
├── mask_patches/
└── cache/
```

This is the layout expected after `eq process-data` runs on a raw project directory.

### 2. Dynamic Patching From Full Images

For on-the-fly cropping:

```text
<data_root>/
├── images/
└── masks/
```

In this mode, full images are loaded directly and crops are sampled during training.

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
eq process-data \
  --input-dir data/raw_data/<project> \
  --output-dir data/derived_data/<project>
```

Current behavior of `process-data`:

- creates `image_patches/`, `mask_patches/`, and `cache/`
- calls the unified `patchify_dataset(...)` path
- writes `processing_metadata.json`

The checked-in branch does **not** use the older bare repo-root `derived_data/` convention as its primary documentation target.

## Segmentation Architecture

### Core Contract

The current branch follows a consistent binary segmentation contract:

- class `0`: background
- class `1`: foreground
- `n_out=2`
- masks normalized to `0/1`
- FastAI default loss selection retained for this output format

This aligns with the engineering guidance in [SEGMENTATION_ENGINEERING_GUIDE.md](/home/ncamarda/endotheliosis_quantifier/docs/SEGMENTATION_ENGINEERING_GUIDE.md).

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
  --data-dir data/derived_data/mito \
  --model-dir models/segmentation/mitochondria \
  --epochs 50 \
  --batch-size 16 \
  --learning-rate 1e-3 \
  --image-size 256
```

Current defaults in the training module:

- epochs: `50`
- batch size: `8` at the shared constant level, but README and config examples commonly use `16` for mitochondria
- learning rate: `1e-3`
- image size: `256`
- dynamic patching: enabled by default

### Stage 2: Glomeruli Training

Primary entrypoint:

```bash
python -m eq.training.train_glomeruli \
  --data-dir data/raw_data/<your_glomeruli_project> \
  --model-dir models/segmentation/glomeruli \
  --epochs 50 \
  --batch-size 8 \
  --learning-rate 1e-3 \
  --image-size 256 \
  --crop-size 512
```

Important nuance:

- the README example above is the recommended workflow documentation
- the `train_glomeruli.py` module itself currently defaults to transfer-learning-oriented values of `epochs=30`, `batch_size=16`, and `learning_rate=1e-3`
- `configs/glomeruli_finetuning_config.yaml` documents `epochs=30`, `batch_size=16`, and `learning_rate=5e-4`

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

- early image-mask pairing checks for static patch datasets
- failure when expected masks are missing
- basic sampled mask-content sanity checks on validation items
- support for both static patch and full-image layouts

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

The branch contains quantification-related code in [`src/eq/evaluation/quantification_metrics.py`](/home/ncamarda/endotheliosis_quantifier/src/eq/evaluation/quantification_metrics.py).

What that module currently implements:

- openness score from thresholded bright regions inside a glomerulus mask
- coarse grade assignment from openness thresholds
- summary statistics over batches
- a rule-based severity label

### What Does Not Yet Exist As A Completed Branch Workflow

The current `master` branch does **not** provide a completed learned quantification stack with:

- dedicated feature-extraction modules
- trained regression models
- validated subject-level endotheliosis score prediction
- a maintained end-to-end learned quantification CLI path

The CLI commands `extract-features` and `quantify` still print that they are not yet implemented.

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
3. process metadata with `eq metadata-process` if scoring spreadsheets are present
4. audit masks with `eq audit-derived`
5. train mitochondria via `python -m eq.training.train_mitochondria`
6. train glomeruli via `python -m eq.training.train_glomeruli`

The dedicated training modules are more trustworthy than the older orchestration-style CLI routes for heavy model training.

## Current Known Gaps

As of April 2, 2026 on `master`, the main known gaps are:

- learned feature extraction is not implemented as a maintained pipeline
- learned regression-based quantification is not implemented as a maintained pipeline
- inference exists but is not uniformly current
- some configs, README examples, and code defaults still disagree on exact training hyperparameters

## Current Status

The current `master` branch should be described as:

**A maintained segmentation repository with useful data-preparation and metadata utilities, partial inference code, and only heuristic rather than learned quantification.**

That is a more accurate summary of the checked-in branch than the older claim that the full endotheliosis scoring pipeline is already implemented.
