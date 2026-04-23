# Segmentation Engineering Guide

This document combines the practical debugging notes and implementation rules that matter most for segmentation work in this repository.

## Binary Segmentation Contract

In this codebase, binary segmentation should be treated as a two-class problem:

- class `0`: background
- class `1`: foreground

That implies:

- use `n_out=2`
- keep masks as `0/1` class indices
- let FastAI configure the learner loss for that output format unless there is a strong, validated reason to override it

Using `n_out=1` in a setup that still expects class-index behavior tends to produce misleading metrics and unstable training behavior.

## Transform Pipeline Rules

A reliable transform organization is:

- item transforms for resize and geometry
- batch transforms for tensor conversion and normalization

Important implementation points:

- `IntToFloatTensor()` should run before transforms that require float tensors
- ImageNet normalization should remain in place when using pretrained encoders
- masks should be normalized into a clear binary target contract before training

## Image And Mask Contract

Images and masks should be paired explicitly and validated before training starts.

Recommended assumptions:

- every training image has a resolvable corresponding mask
- masks are binary on disk
- masks are converted into `0/1` targets in the loader path
- loaders should fail early when expected image-mask pairs are missing

Supported training roots contain full-image `images/` and `masks/` directories. The dataloader performs dynamic patching from that selected root. Do not train supported mitochondria or glomeruli models from `image_patches/` or `mask_patches/`.

For mitochondria, keep the installed physical split explicit:

- `derived_data/mitochondria_data/training` is the dynamic training root
- `derived_data/mitochondria_data/testing` is held out for explicit evaluation

The physical `testing/` root is not the FastAI validation split. Training-time validation is created inside the selected `training/` root.

For glomeruli, train from a curated paired full-image root under `raw_data`, such as `raw_data/preeclampsia_project/training_pairs`. Raw project backup trees are source material and may contain images without masks; they should not be passed directly to training unless every image has a matching mask.

Path contract:

- `raw_data/...`: source images, source masks, curated paired training roots
- `derived_data/...`: generated manifests, audits, caches, metrics, evaluation artifacts

The current mitochondria runtime under `derived_data/mitochondria_data/{training,testing}` is a Lucchi-installed exception, not the generic naming rule for curated glomeruli training pairs.

Local Apple Silicon starting points on the powerful MPS machine class:

- mitochondria: `batch_size=24` with `256x256` crops
- glomeruli: `batch_size=12` with `512x512` crops

These are throughput/stability starting points, not scientific constants. Override them if actual MPS performance or stability requires it.

## Model Artifact Compatibility Gate

FastAI learner exports are executable Python pickle artifacts, not neutral model-weight files. A `.pkl` can depend on the exact project module paths, FastAI objects, NumPy pickle namespaces, and package versions that existed when it was exported.

A supported segmentation model artifact requires:

- export from the current `src/eq` namespace
- loadability in the current certified environment without legacy module shims
- recorded package versions for Python, PyTorch, torchvision, FastAI, and NumPy
- recorded training command, data root, training mode, and code version
- a current pipeline or inference test that depends on the supported artifact path

Legacy `.pkl` artifacts that reference removed modules such as `eq.segmentation...`, old FastAI transform namespaces, or incompatible NumPy pickle namespaces are historical artifacts. Do not point new tests, specs, or default commands at them unless the work is explicitly a legacy-artifact compatibility change.

## Model Promotion Gate

Training completion, MPS execution, pickle loading, and successful pipeline execution are runtime compatibility checks. They do not establish that a segmentation model is ready for scientific use.

A promoted glomeruli segmentation model requires:

- a current-namespace exported model artifact
- a training-data audit showing both foreground and background coverage
- validation metrics that improve beyond a trivial foreground-only or background-only policy
- prediction-review images showing non-degenerate masks across representative validation examples
- deterministic validation examples covering positive, boundary, and background crops

Static patch datasets that contain only foreground-positive glomeruli patches are not sufficient for promotion. A model trained on that data can pass runtime checks while learning all-positive predictions. Treat that as a data-contract failure, not as a usable segmentation model.

The 2026-04-22 static-patch glomeruli artifact illustrates the failure mode:

- `3766 / 3766` static training patches contained foreground pixels
- validation contained no empty/background-only masks
- validation foreground fraction was high: median `29.7%`, 75th percentile `65.9%`, 95th percentile `98.2%`
- all-foreground validation baseline was approximately `dice=0.5659`, `jaccard=0.3946`
- the trained artifact reported `dice=0.5560`, `jaccard=0.3851`, effectively matching the all-foreground baseline

Promotion review must compare candidate predictions against all-background and all-foreground baselines and block all-foreground or all-background prediction panels.

## Why Displayed Inputs Can Look Washed Out

The model consumes normalized tensors based on ImageNet statistics. If those tensors are displayed directly, or after aggressive contrast stretching, they may appear desaturated compared with the raw RGB image.

For human-readable visualizations:

- display the raw image from disk for fidelity
- display a de-normalized model input when validating the preprocessing pipeline
- clamp de-normalized images to `[0, 1]` before rendering

If a visualization path applies additional contrast stretching, treat that as a debugging aid rather than a faithful representation of the original sample.

## Improving Segmentation Quality

### Losses And Thresholding

- Consider Dice loss or a BCE-plus-Dice combination when optimizing overlap quality directly.
- Consider Tversky-style losses when false negatives are especially costly.
- Sweep the decision threshold on a validation set rather than assuming `0.5` is optimal.

### Sampling And Positive Coverage

- Increase positive-focused cropping when the target object is sparse.
- Revisit:
  - `positive_focus_p`
  - `min_pos_pixels`
  - `pos_crop_attempts`
- Re-check image-mask alignment whenever Dice drops unexpectedly.

### Augmentation And Domain Alignment

- Use geometric augmentation conservatively but consistently.
- Use stain-aware or color-robust augmentation for H&E data when appropriate.
- Avoid lighting changes that materially distort label semantics.

### Architecture And Scale

- If structures are larger than the default crop, test larger crop sizes or multi-scale training.
- Stronger encoders may help, but data quality and mask correctness usually matter more than architectural churn.

### Validation And Post-Processing

- Test-time augmentation can improve validation-time robustness.
- Simple morphology can reduce small false-positive islands and fill tiny holes.
- Any post-processing step should be validated quantitatively, not only visually.

## Common Failure Modes

### Incorrect Transform Ordering

If tensor conversion happens too late, augmentation code may still be operating on byte tensors and fail at runtime.

### Images Without Masks

If loaders do not validate the image-mask mapping early, training tends to fail later and less clearly.

### Manual Loss Overrides

If FastAI is already configuring the learner correctly for the chosen output format, manually overriding the loss can introduce shape mismatches and metric confusion.

### Metric Misinterpretation

When the model output format and metric assumptions do not match, validation numbers can look much worse than the actual predictions.

## Diagnosing Out-Of-Memory Errors

OOM problems often appear after a loader fix because training finally starts processing real image tensors.

Common causes:

- unexpectedly large images bypassing a resize or crop invariant
- batch size too large for the available GPU memory
- validation or visualization code retaining tensors longer than expected

Typical mitigations:

- enforce a final size constraint before the model forward pass
- reduce batch size
- use mixed precision where appropriate
- confirm that dynamic patching always yields the intended crop size

## Review Checklist

When adding or reviewing segmentation code in this repo, prefer this checklist:

1. `n_out=2`
2. masks represented as `0/1`
3. item transforms handle geometry and resize
4. batch transforms handle tensor conversion and normalization
5. image-mask pairing is validated before training
6. training data includes foreground and background coverage
7. validation predictions are not all foreground or all background
8. visualization paths distinguish raw images from model-space tensors
