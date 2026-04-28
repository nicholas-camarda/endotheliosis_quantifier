# Endotheliosis Quantifier Onboarding Guide

This guide keeps the more explanatory, beginner-friendly walkthrough for the project. The main [`README.md`](../README.md) covers the shared workflow and environment contract for both WSL/CUDA and macOS/MPS development, while this document is meant to help non-technical collaborators, future-you, or anyone returning to the repo after a long break.

## What This Project Is

Endotheliosis Quantifier (`eq`) is a deep learning pipeline for glomeruli segmentation and image-level endotheliosis quantification in kidney histology, with scored-cohort workflows built around a shared runtime contract.

At a high level, the project uses a two-stage idea:

1. Train a segmentation model on a mitochondria dataset to learn useful visual features such as edges, substructure, and boundaries.
2. Transfer that knowledge to glomeruli segmentation in kidney histology images.

That segmentation output supports image-level endotheliosis quantification workflows across scored cohorts rather than one single project-specific dataset.

## Key Ideas

- **Two-stage training**: mitochondria pretraining followed by glomeruli fine-tuning
- **Dynamic patching**: train on full images with augmentations and on-the-fly crops instead of permanently patchifying everything upfront
- **ROI identification**: segment glomeruli regions before later quantification steps
- **Current image-level supervision**: the maintained quantification path uses image-level grades joined to image or mask pairs
- **Union ROI semantics**: image-level scoring uses the full multi-component mask region rather than only the largest connected component

## Current Training Snapshot

The current checked-in segmentation results come from the April 25, 2026 P0 workflow artifacts under `$EQ_RUNTIME_ROOT/models/segmentation/` and `$EQ_RUNTIME_ROOT/output/segmentation_evaluation/glomeruli_candidate_comparison/production_glomeruli_candidate_p0_contract_20260425_adjudicated/`.

- Current candidate artifacts are identified by the comparison report and model sidecars under the runtime model root.
- Deterministic glomeruli review panel: `30` crops across `27` images and `5` subjects
- Category balance: `10` background, `10` boundary, `10` positive

The current glomeruli candidates are research-use artifacts, not current defaults. The internal deterministic panel is useful for debugging and comparison, but onboarding does not present one model as the default because transfer and scratch remain within the configured practical tie margin. The checked-in internal figures live in [TECHNICAL_LAB_NOTEBOOK.md](TECHNICAL_LAB_NOTEBOOK.md#current-segmentation-training-snapshot).

## What Data You Need

There are two main image data sources:

- **Mitochondria data**: electron microscopy images with mitochondria annotations
- **Glomeruli data**: H&E kidney histology images with binary glomeruli masks

Some cohorts may also include spreadsheets or other metadata files for audit context. Those are useful for summaries, joins, and migration work, but they are not the front-door control surface for the maintained segmentation workflow.

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

If masks are being created manually, Label Studio is a practical workflow. In the current maintained quantification path, the annotation export can also serve as the image-level grading source.

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
│   ├── SUBJECT_A/
│   │   └── SUBJECT_A-1.jpg
│   └── SUBJECT_B/
│       └── SUBJECT_B-1.jpg
├── masks/
│   ├── SUBJECT_A/
│   │   └── SUBJECT_A-1_mask.png
│   └── SUBJECT_B/
│       └── SUBJECT_B-1_mask.png
├── scores/
│   └── labelstudio_annotations.json
└── metadata/  # optional audit context
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
        └── metadata/  # optional
```

This layout keeps cohort inputs portable across WSL, Windows, and macOS.

## Before Training

After activating the environment, check what hardware the current machine exposes:

```bash
eq capabilities
eq mode --show
```

This is especially useful if you are moving between a CUDA desktop, a CPU-only machine, and a Mac.

## Validate Naming Early

```bash
eq validate-naming --data-dir "$EQ_RUNTIME_ROOT/raw_data/cohorts/<cohort_id>"
eq validate-naming --data-dir "$EQ_RUNTIME_ROOT/raw_data/cohorts/<cohort_id>" --strict
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
- score recovered from the active image-level annotation source
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

The manifest is image-level and records runtime-local asset paths, score linkage, optional mask path, treatment group, lane assignment, verification state, admission state, and file hashes. Original source folders remain provenance sources; normal cohort work uses the localized runtime cohort directories.

The cohort manifest is a project-local data contract. It is useful for this lab workflow because it makes image paths, mask paths, scores, admission state, and file hashes auditable in one place. It should not be read as a generic public dataset requirement. Current local cohort counts and unresolved-source notes are recorded in the [technical lab notebook](TECHNICAL_LAB_NOTEBOOK.md#local-cohort-manifest-snapshot).

Lucchi and other segmentation-install datasets are not part of the scored cohort manifest.

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

For all-data glomeruli training, use the manifest-backed `raw_data/cohorts` registry root. It trains from admitted manifest rows in the `manual_mask_core` and `manual_mask_external` lanes, so unresolved, foreign, evaluation-only, and scored-only rows stay out. For a single-cohort run, use `raw_data/cohorts/<cohort_id>`. Raw backup trees are source material, not direct training roots. Generated manifests, audits, caches, and metrics belong under `derived_data` or `output`.

The YAML workflow is the normal control surface for full candidate training and comparison. Direct training module commands remain useful for targeted runs, and the later artifact path is derived from the base model name plus the auto-generated run suffix. Transfer training with `--base-model` must load that artifact and copy compatible weights or the run stops. The `--from-scratch` candidate is the no-mitochondria-base comparator with an ImageNet-pretrained ResNet34 encoder, not a literal all-random initialization baseline. `eq run-config` writes workflow logs under `$EQ_RUNTIME_ROOT/logs/run_config/<run_id>/`, model artifacts under `$EQ_RUNTIME_ROOT/models/segmentation/`, and comparison reports under `$EQ_RUNTIME_ROOT/output/segmentation_evaluation/`. Supported direct module entrypoints write logs under `$EQ_RUNTIME_ROOT/logs/direct/<surface>/<run_id>/`; imported functions use the caller's logger and do not create log files by themselves. Generic `eq` utility subcommands use `eq --log-file <path>` when file capture is needed. After training, inspect the produced `.pkl` path and reuse that exact path in downstream comparison or quantification commands.

### 4. Run The Current Quantification Workflow

```bash
eq prepare-quant-contract \
  --data-dir "$EQ_RUNTIME_ROOT/raw_data/cohorts/<cohort_id>" \
  --segmentation-model "$EQ_RUNTIME_ROOT/models/segmentation/glomeruli/<transfer_or_scratch>/<model_run>/<your_model>.pkl" \
  --score-source labelstudio \
  --annotation-source "$EQ_RUNTIME_ROOT/raw_data/cohorts/<cohort_id>/scores/labelstudio_annotations.json"

eq quant-endo \
  --data-dir "$EQ_RUNTIME_ROOT/raw_data/cohorts/<cohort_id>" \
  --segmentation-model "$EQ_RUNTIME_ROOT/models/segmentation/glomeruli/<transfer_or_scratch>/<model_run>/<your_model>.pkl" \
  --score-source labelstudio \
  --annotation-source "$EQ_RUNTIME_ROOT/raw_data/cohorts/<cohort_id>/scores/labelstudio_annotations.json" \
  --output-dir "$EQ_RUNTIME_ROOT/output/quantification_results/<cohort_id>"
```

This writes:

- recovered Label Studio score tables and duplicate-resolution audit outputs
- union-ROI image and mask crops
- frozen segmentation encoder embeddings
- `burden_model/INDEX.md` as the first-read map for the burden-model subtree
- `burden_model/primary_burden_index/model/` artifacts with `endotheliosis_burden_0_100`, threshold probabilities, prediction sets, uncertainty intervals, and the serialized exploratory burden model
- `burden_model/primary_burden_index/validation/`, `burden_model/primary_burden_index/calibration/`, `burden_model/primary_burden_index/summaries/`, and `burden_model/primary_burden_index/evidence/` artifacts with support gates, grouping audit, calibration, nearest scored examples, cohort metrics, summary intervals, and prediction explanations
- `burden_model/primary_burden_index/feature_sets/` and `burden_model/primary_burden_index/diagnostics/` morphology feature artifacts for open lumina, collapsed/slit-like structures, RBC-like patent-lumen confounding, and ROI quality
- `burden_model/primary_burden_index/candidates/` candidate-screen artifacts such as `signal_comparator_metrics.csv`, `subject_level_candidate_predictions.csv`, `precision_candidate_summary.json`, `morphology_candidate_metrics.csv`, and `morphology_candidate_summary.json`; these are review artifacts, not deployed models
- `burden_model/primary_burden_index/evidence/morphology_feature_review/` with the visual feature review page, selected cases, overlay assets, and operator adjudication template
- `burden_model/learned_roi/` with `INDEX.md`, `summary/` first-read verdict artifacts, the capped learned ROI phase-1 screen, provider audit, learned feature table, candidate metrics, calibration summary, cohort-confounding diagnostics, nearest examples, and learned ROI evidence review
- `ordinal_model/` artifacts with ordinal comparator predictions, probabilities, metrics, confusion matrix, and the comparator-specific HTML review
- `quantification_review/` artifacts with the combined burden/comparator HTML review, reviewer examples, concrete results summaries, and a README/docs snippet generated from the current run

The burden score is a predictive ordinal stage-burden index from image-level grades. It is not a pixel-level tissue-area percent and should be interpreted with the generated support, calibration, and uncertainty artifacts.

The current full-cohort burden result is exploratory rather than a deployed model claim. The generated `quantification_review/readme_results_snippet.md` is not automatically shareable; reuse it only when the review report marks the selected track as README/docs-ready. Current candidate screens keep subject/cohort burden, per-image burden, morphology-aware feature evidence, and learned ROI evidence separate so feature QA, uncertainty, and cohort-confounding gates can be reviewed before any public claim.

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
