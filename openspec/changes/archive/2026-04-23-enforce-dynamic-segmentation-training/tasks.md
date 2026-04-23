## 1. Runtime Data Retirement

- [x] 1.1 Inventory active mitochondria static patch directories under `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/derived_data/mitochondria_data/{training,testing}` and record path, size, and file count before moving.
- [x] 1.2 Create a dated ignored runtime retirement directory under `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/_retired/mitochondria_static_patch_datasets_2026-04-22`.
- [x] 1.3 Move mitochondria `image_patches/`, `mask_patches/`, `image_patch_validation/`, and `mask_patch_validation/` from active `training/` and `testing/` directories into the retirement directory.
- [x] 1.4 Verify active mitochondria `training/` and `testing/` directories still contain full-image `images/` and `masks/` and no static patch directories.
- [x] 1.5 Verify active glomeruli `training/`, `testing/`, and `prediction/` directories still have no `image_patches/` or `mask_patches/` directories.
- [x] 1.6 Record the runtime retirement result in the implementation notes, including moved paths, final retired size, and final retired file count.
- [x] 1.7 Verify the physical mitochondria `training/` and `testing/` full-image roots remain separate after static patch retirement.

## 2. Training Contract Implementation

- [x] 2.1 Add a shared training data-root validation helper that accepts only roots with full-image `images/` and `masks/` directories for supported segmentation training.
- [x] 2.2 Make the validation helper fail clearly when a root contains static `image_patches/` / `mask_patches/` instead of full-image `images/` / `masks/`.
- [x] 2.3 Update `src/eq/training/train_mitochondria.py` to always use full-image dynamic patching and remove `--no-dynamic-patching`.
- [x] 2.4 Update `src/eq/training/train_glomeruli.py` to always use full-image dynamic patching and remove `--no-dynamic-patching`.
- [x] 2.5 Update `src/eq/training/transfer_learning.py` so glomeruli transfer learning always builds full-image dynamic-patching dataloaders and does not expose a static-patch training path.
- [x] 2.6 Preserve static patch DataBlock utilities only as legacy/audit/conversion helpers, with names or docstrings that do not present them as supported training builders.
- [x] 2.7 Update training logs so each segmentation run records `training_mode=dynamic_full_image_patching` and the validated full-image data root.
- [x] 2.8 Ensure training entrypoints use the selected full-image data root only and do not auto-merge sibling `testing/` roots into training or validation.
- [x] 2.9 Update the glomeruli training-root contract so canonical curated paired roots live under `raw_data/.../training_pairs` rather than `derived_data` placeholders.
- [x] 2.10 Document the current mitochondria full-image runtime under `derived_data/mitochondria_data/{training,testing}` as an explicit Lucchi-installed exception rather than the generic path rule.

## 3. Artifact Provenance

- [x] 3.1 Extend training metadata output to record the exact training command, code version or dirty-worktree marker, Python version, PyTorch version, torchvision version, FastAI version, NumPy version, data root, training mode, and output artifact paths.
- [x] 3.2 Ensure newly exported supported segmentation artifacts have sidecar metadata sufficient to distinguish supported runtime artifacts from legacy FastAI pickles and compatibility artifacts.
- [x] 3.3 Mark the current Mac static-patch-trained mitochondria and glomeruli exports as compatibility artifacts in docs or implementation notes rather than promoted model artifacts.
- [x] 3.4 Keep legacy FastAI pickle loading behavior out of scope unless a separate compatibility change explicitly adds tests for it.

## 4. Config And CLI Surface Cleanup

- [x] 4.1 Update `configs/mito_pretraining_config.yaml` so training paths point at full-image mitochondria `images/` and `masks/` roots rather than patch directories.
- [x] 4.2 Update `configs/glomeruli_finetuning_config.yaml` so training paths point at full-image glomeruli `images/` and `masks/` roots rather than patch directories.
- [x] 4.3 Remove patch-generation-as-training language from segmentation training config comments.
- [x] 4.4 Update relevant CLI help text so supported training commands describe full-image dynamic patching and do not advertise static patch training.
- [x] 4.5 Audit `src/eq/__main__.py` processing and audit commands and relabel any retained patchification behavior as legacy conversion/audit behavior rather than supported training input generation.
- [x] 4.6 For mitochondria configs, point training at the physical `training/` full-image root and represent the physical `testing/` full-image root only as held-out evaluation data.
- [x] 4.7 Update glomeruli configs and examples so canonical trainable image/mask roots are expressed under `raw_data/.../training_pairs`.
- [x] 4.8 Encode machine-aware Apple Silicon MPS starting batch defaults: mitochondria `24` at `256x256`, glomeruli `12` at `512x512`, with explicit overrides retained.

## 5. Tests

- [x] 5.1 Add unit tests for the training data-root validator accepting full-image `images/` / `masks/` fixtures.
- [x] 5.2 Add unit tests for the training data-root validator rejecting static `image_patches/` / `mask_patches/` fixtures.
- [x] 5.3 Update segmentation training smoke tests to use full-image dynamic-patching fixtures.
- [x] 5.4 Add tests that `train_mitochondria.py --help` and `train_glomeruli.py --help` do not expose `--no-dynamic-patching`.
- [x] 5.5 Add tests or focused assertions that transfer learning constructs dynamic full-image dataloaders for glomeruli training.
- [x] 5.6 Add tests or fixtures for glomeruli data audits that report foreground fraction distribution, background-only crop rate, full-foreground crop rate, and subject or image split coverage.
- [x] 5.7 Add tests for trivial all-background and all-foreground baseline metric computation on fixed validation examples.
- [x] 5.8 Add tests that promotion validation uses deterministic examples rather than changing selected validation crops across repeated evaluations.
- [x] 5.9 Add tests that degenerate all-foreground or all-background prediction reviews block glomeruli model promotion.
- [x] 5.10 Update or remove tests that assume static patch datasets are the supported segmentation training input.
- [x] 5.11 Add tests that mitochondria dynamic training uses the selected `training/` root only and does not include the sibling `testing/` root unless an explicit held-out evaluation path requests it.
- [x] 5.12 Add tests that docs/config-selected glomeruli training roots use the raw-data paired-root contract rather than derived-data placeholders.
- [x] 5.13 Add tests or focused assertions for the powerful Apple Silicon MPS batch defaults and their override behavior.

## 6. Documentation

- [x] 6.1 Update `AGENTS.md` to reflect that mitochondria and glomeruli supported training both use full-image dynamic patching.
- [x] 6.2 Update `README.md` training examples to use full-image roots and remove static patch training examples.
- [x] 6.3 Update `docs/ONBOARDING_GUIDE.md` to describe dynamic patching as the supported training path.
- [x] 6.4 Update `docs/TECHNICAL_LAB_NOTEBOOK.md` to remove present-tense claims that patch datasets are the active segmentation training pipeline.
- [x] 6.5 Update `docs/OUTPUT_STRUCTURE.md` to document retired static patch datasets and current full-image training roots.
- [x] 6.6 Update `docs/SEGMENTATION_ENGINEERING_GUIDE.md` with the final dynamic-only training contract and artifact-provenance gate.
- [x] 6.7 Document the 2026-04-22 glomeruli static-patch failure evidence: `3766/3766` foreground-positive patches, no empty validation masks, all-foreground baseline metrics near the trained artifact metrics, and the validation prediction panel.
- [x] 6.8 Document that glomeruli promotion requires fixed positive, boundary, and background validation examples plus all-background and all-foreground baseline comparisons.
- [x] 6.9 Document the distinction between physical installed `training/` and `testing/` full-image roots and the internal dynamic train/validation split used during model training.
- [x] 6.10 Document the canonical raw-data vs derived-data boundary: curated trainable image/mask roots under `raw_data`, generated manifests/audits/caches/metrics under `derived_data`.
- [x] 6.11 Document the powerful Apple Silicon MPS starting batch defaults and that they are throughput/stability starting points rather than scientific constants.

## 7. Validation

- [x] 7.1 Run `env OPENSPEC_TELEMETRY=0 openspec validate enforce-dynamic-segmentation-training --strict`.
- [x] 7.2 Run `mamba run -n eq-mac python -m py_compile src/eq/training/train_mitochondria.py src/eq/training/train_glomeruli.py src/eq/training/transfer_learning.py`.
- [x] 7.3 Run focused tests covering data-root validation, dynamic-patching dataloaders, CLI help, and artifact metadata.
- [x] 7.4 Run `mamba run -n eq-mac python -m eq --help`.
- [x] 7.5 Run `mamba run -n eq-mac python -m eq capabilities`.
- [x] 7.6 Run a bounded mitochondria dynamic-patching training smoke on full-image fixtures or the approved runtime full-image root.
- [x] 7.7 Run a bounded glomeruli dynamic-patching transfer-training smoke on full-image fixtures or the approved runtime full-image root.
- [x] 7.8 Confirm no active ProjectsRuntime segmentation training directory contains `image_patches/` or `mask_patches/` after implementation.
- [x] 7.9 Run the glomeruli training-data audit on the approved full-image runtime root and record foreground/background crop coverage.
- [x] 7.10 Run fixed validation-manifest or deterministic validation-example checks for glomeruli promotion readiness if a candidate glomeruli artifact is produced during this change.
- [x] 7.11 Confirm the mitochondria physical `testing/` root remains held out from training-time validation and is only used through an explicit evaluation path.
- [x] 7.12 Validate that glomeruli config and docs point to canonical raw-data paired roots rather than derived-data placeholders.
- [x] 7.13 Run a bounded local MPS throughput/stability check at the new starting batch defaults and record whether overrides are needed on this Mac.
- [x] 7.14 Run the whole intended pipeline from the beginning under the updated path contract, dynamic training contract, and local MPS defaults.
- [x] 7.15 If the end-to-end run fails, fix the blocker, restart validation from the beginning, and repeat until the pipeline completes end to end.
- [x] 7.16 Record the end-to-end execution path, every issue found during iteration, each fix applied, and any final residual limitations inside this OpenSpec change only.
