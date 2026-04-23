## Why

Segmentation training currently has two competing data contracts: the intended full-image dynamic-patching path and a legacy pre-generated `image_patches/` / `mask_patches/` path. The legacy path already produced a current-namespace but scientifically unusable glomeruli compatibility artifact, and it remains easy for code, configs, or runtime directories to accidentally select that path again.

This change makes full-image `images/` + `masks/` dynamic patching the supported segmentation training contract for mitochondria and glomeruli, retires active static patch training inputs, and requires model artifact provenance that distinguishes legacy pickles, compatibility artifacts, supported runtime artifacts, and scientifically promoted artifacts.

## What Changes

- **BREAKING**: Remove static pre-generated patch datasets as a supported training input for mitochondria and glomeruli training.
- **BREAKING**: Remove or disable CLI/config paths that let supported training run from `image_patches/` / `mask_patches/`.
- Require segmentation training data roots to contain full-size `images/` and `masks/` directories.
- Keep dynamic patching as the single supported training mode for segmentation models.
- Preserve existing physical `training/` and `testing/` full-image layouts where present; dynamic patching creates the internal train/validation split from the selected training root, while physical `testing/` roots are held-out evaluation inputs.
- Retire remaining active mitochondria static patch directories into ignored ProjectsRuntime retired storage, matching the already-retired glomeruli static patch datasets. This runtime move has been completed at `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/_retired/mitochondria_static_patch_datasets_2026-04-22`.
- Update glomeruli and mitochondria training configs so they point to full-image training roots rather than derived patch directories.
- Update docs and repo instructions to describe static patch datasets as retired legacy artifacts, not active training inputs.
- Add tests that supported training rejects static patch roots and accepts full-image roots.
- Add provenance requirements for newly exported supported segmentation artifacts, including training mode, data root, package versions, code version, and command.
- Add a glomeruli promotion gate that requires deterministic validation crops, trivial all-foreground/all-background baselines, training-data foreground/background audits, and non-degenerate prediction review.
- Preserve legacy patchification/audit utilities only where they are explicitly scoped as conversion, audit, or historical inspection tools, not supported training paths.

## Capabilities

### New Capabilities
- `segmentation-training-contract`: Defines the supported segmentation training data contract, dynamic-patching-only training behavior, static patch retirement, and model artifact provenance requirements.

### Modified Capabilities

None.

## Impact

- Affected source modules:
  - `src/eq/training/train_mitochondria.py`
  - `src/eq/training/train_glomeruli.py`
  - `src/eq/training/transfer_learning.py`
  - `src/eq/data_management/datablock_loader.py`
  - `src/eq/data_management/standard_getters.py`
  - `src/eq/utils/run_io.py`
  - CLI surfaces in `src/eq/__main__.py` if training or processing command help still presents static patch training as current behavior.
- Affected configs:
  - `configs/mito_pretraining_config.yaml`
  - `configs/glomeruli_finetuning_config.yaml`
- Affected docs and instructions:
  - `AGENTS.md`
  - `README.md`
  - `docs/ONBOARDING_GUIDE.md`
  - `docs/TECHNICAL_LAB_NOTEBOOK.md`
  - `docs/OUTPUT_STRUCTURE.md`
  - `docs/SEGMENTATION_ENGINEERING_GUIDE.md`
- Affected tests:
  - Existing DataBlock and training smoke tests need to reflect full-image dynamic patching as the supported path.
  - New tests should reject static patch roots for supported training entrypoints.
- Affected runtime artifact directories:
  - Keep active raw/full-image roots such as `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/raw_data/preeclampsia_project/clean_backup`.
  - Keep full-image mitochondria roots under `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/derived_data/mitochondria_data/{training,testing}/{images,masks}` unless a better raw runtime source is identified. Use `training/` as the training root with an internal dynamic train/validation split; use `testing/` only as a deliberate held-out evaluation root.
  - Preserve retired mitochondria static patches at `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/_retired/mitochondria_static_patch_datasets_2026-04-22`.
  - Preserve already-retired glomeruli static patches at `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/_retired/glomeruli_static_patch_datasets_2026-04-22`.
- Compatibility risks:
  - Existing ad hoc commands that point at `derived_data/.../image_patches` will fail and must be rewritten to full-image roots.
  - Old `.pkl` model artifacts remain historical unless re-exported or retrained from current namespace and current training contract.
  - Static patch utilities may still be needed for legacy audit or conversion tasks, so this change should remove them from supported training without deleting unrelated historical data outright.
