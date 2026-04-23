## Context

The repository currently contains two segmentation training paths:

- full-image training roots with `images/` and `masks/`, loaded through dynamic patching
- legacy pre-generated `image_patches/` and `mask_patches/`, loaded as static patch datasets

Both `train_mitochondria.py` and `train_glomeruli.py` expose `--no-dynamic-patching`, and both can still call `build_segmentation_dls(...)` for static patch datasets. The repository configs still point at patch directories:

- `configs/mito_pretraining_config.yaml` references `derived_data/mitochondria_data/training/image_patches` and `mask_patches`.
- `configs/glomeruli_finetuning_config.yaml` references `derived_data/glomeruli_data/training/image_patches`, `mask_patches`, and `testing/image_patches`.

The active runtime data proves static patch training is not required for either current segmentation stage:

- Glomeruli full-image pairs exist at `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/raw_data/preeclampsia_project/clean_backup/images` and `masks`.
- Mitochondria full-image pairs exist under `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/derived_data/mitochondria_data/{training,testing}/images` and `masks`.
- Old glomeruli static patch directories have already been moved to `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/_retired/glomeruli_static_patch_datasets_2026-04-22`.
- Old mitochondria static patch directories have already been moved to `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/_retired/mitochondria_static_patch_datasets_2026-04-22`.

The latest Mac compatibility retraining accidentally used static patch data for both mitochondria and glomeruli. The mitochondria artifact trained successfully, but it was still produced through the legacy path. The glomeruli artifact trained on foreground-only static patches, produced flat validation metrics close to an all-foreground baseline, and must remain a compatibility artifact rather than a promoted scientific model.

The glomeruli failure is empirically supported by the current runtime data:

- Static glomeruli training data at `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/derived_data/glomeruli_data/training` contained `3766 / 3766` patches with foreground mask pixels and `0` background-only masks.
- The validation split also contained `0` empty/background-only masks.
- Validation foreground fraction was high: median `29.7%`, 75th percentile `65.9%`, 95th percentile `98.2%`, with `14` validation patches at `100%` foreground.
- An all-foreground prediction policy on that validation split would score approximately `dice=0.5659` and `jaccard=0.3946`.
- The actual static-patch-trained glomeruli model scored `dice=0.5560` and `jaccard=0.3851`, effectively matching the trivial all-foreground baseline.
- The validation prediction panel showed broad foreground predictions rather than clean glomerulus segmentation:
  `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/models/segmentation/glomeruli/transfer/glom_dynamic_unet_mps-transfer_loss-dice_s1lr1e-3_s2lr_lrfind_e30_b16_lr1e-3_sz256/glom_dynamic_unet_mps-transfer_loss-dice_s1lr1e-3_s2lr_lrfind_e30_b16_lr1e-3_sz256_validation_predictions.png`.

The raw full-image glomeruli data has a different failure profile. Among valid full-image pairs, `89 / 89` masks contain foreground, but the foreground occupies a small fraction of each full image. A simulation of `1780` random `256x256` crops from raw masks found foreground in `18.7%` of crops and at least `64` foreground pixels in `18.2%` of crops. Dynamic full-image patching is therefore the right training source because it can expose real background tissue, but random validation crops alone are not enough for scientific promotion.

Runtime retirement already completed for static patch datasets:

- Glomeruli static patches are retired at `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/_retired/glomeruli_static_patch_datasets_2026-04-22`, verified at `184M` and `16558` files.
- Mitochondria static patches are retired at `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/_retired/mitochondria_static_patch_datasets_2026-04-22`, verified at `132M` and `6564` files.
- Active mitochondria full-image roots remain present with `660` files across `training/images`, `training/masks`, `testing/images`, and `testing/masks`.

The physical installed layout is separate from the internal model-training split. For mitochondria, the existing full-image `training/` and `testing/` directories should remain in place. Dynamic patching should build its train/validation split from the selected training root, normally `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/derived_data/mitochondria_data/training`; the sibling `testing/` root is held-out evaluation data and must not be silently merged into training or treated as the FastAI validation split. For glomeruli, the active full-image source is currently the single clean backup root, so dynamic training may split that root internally unless a separate held-out full-image root is later defined.

This change is therefore a training-contract cleanup. It is not a model-quality optimization by itself; it prevents the wrong data contract from being selected again and sets up future retraining on the intended full-image dynamic-patching path.

## Goals / Non-Goals

**Goals:**

- Make dynamic patching from full `images/` and `masks/` the only supported segmentation training path for mitochondria and glomeruli.
- Remove `--no-dynamic-patching` and equivalent supported config paths from training entrypoints.
- Make training fail clearly when pointed at a static patch root.
- Retire remaining active mitochondria static patch directories into ignored runtime retired storage before retraining from full images.
- Keep already-retired glomeruli patch datasets out of active training paths.
- Preserve existing physical full-image `training/` and `testing/` installed layouts, and document that dynamic patching creates the internal train/validation split from the selected training root.
- Update configs, docs, tests, and OpenSpec-facing instructions so they describe full-image dynamic patching as current behavior.
- Record model artifact provenance for supported segmentation exports, including training mode, data root, command, package versions, and code version.
- Add tests that distinguish supported full-image dynamic training from retired static patch data.
- Add the validation and promotion contract needed to avoid declaring success solely because dynamic training runs.

**Non-Goals:**

- Do not delete historical static patch datasets outright; move active static patch datasets into `_retired/`.
- Do not remove generic patchification or static patch inspection utilities if they remain useful for legacy audit, conversion, or debugging.
- Do not add silent fallback from dynamic full-image training to static patch training.
- Do not merge, rename, or otherwise reorganize physical full-image `training/` and `testing/` directories as part of this change.
- Do not promote a new glomeruli model solely because it trains, loads, or passes the contract-first pipeline test.
- Do not change quantification contracts, Label Studio score ingestion, union ROI semantics, embedding schemas, ordinal model outputs, or review artifact schemas.
- Do not restore legacy FastAI pickle compatibility as part of this change.

## Decisions

- Use full-image `images/` and `masks/` as the segmentation training input contract.
  - Rationale: both current segmentation stages have full image/mask pairs, and full-image dynamic crops can expose real background context that foreground-only static patches cannot.
  - Alternative considered: keep static patch input but add warnings. Rejected because the accidental static-patch retrain already shows warnings are too weak for this failure mode.

- Remove static patch training from supported CLI/config surfaces.
  - Rationale: as long as `--no-dynamic-patching` and patch-dir configs remain normal options, the wrong path can be selected during urgent retraining work.
  - Alternative considered: keep the flag for expert use. Rejected because this repository treats silent scientific invalidity as a higher risk than temporary inconvenience.

- Preserve static patch utility code only behind legacy, audit, or conversion semantics.
  - Rationale: patchification and static patch loaders may still be useful for inspecting historical artifacts or converting datasets, but that is not the same as supported training.
  - Alternative considered: delete all patch-related code immediately. Rejected because the blast radius includes processing commands, audit commands, and historical outputs that should be reviewed before removal.

- Retire active static patch directories instead of deleting them.
  - Rationale: runtime data is outside Git, may be needed for provenance, and should remain recoverable while no longer being discoverable as active training input.
  - Alternative considered: leave directories in place and rely on code rejection. Rejected because physical retirement makes accidental path selection less likely and clarifies the runtime contract.

- Preserve physical train/test layouts and split training internally.
  - Rationale: the installed mitochondria data already has full-image `training/` and `testing/` roots. Dynamic patching should split the chosen training root into train/validation examples, while the physical `testing/` root remains an explicit held-out evaluation set.
  - Alternative considered: merge physical `training/` and `testing/` roots into one full-image pool. Rejected because it would change the installed data contract and risk leaking held-out examples into model selection.

- Add artifact provenance as part of training output.
  - Rationale: FastAI `.pkl` files are executable pickle artifacts whose loadability depends on current package and project namespaces; supported artifacts need traceable environment and training context.
  - Alternative considered: rely on directory names and training history TSVs. Rejected because they do not reliably encode command, namespace, code version, and training mode.

- Require deterministic validation evidence before glomeruli model promotion.
  - Rationale: dynamic training is the correct source of background context, but random validation crops can make metrics non-comparable across runs. Promotion needs fixed positive, boundary, and background validation examples.
  - Alternative considered: rerun dynamic training and treat the normal validation dataloader metrics as sufficient. Rejected because stochastic validation can change the measured metric without a real model improvement.

- Use dynamic sampling for training and fixed manifests for promotion validation.
  - Rationale: training benefits from online crop variation, while promotion needs repeatable, auditable metrics and examples.
  - Alternative considered: regenerate a static patch dataset with positives and negatives as the primary training source. Rejected as the supported training path because it reintroduces a permanent patch dataset as active training input, though manifest-based fixed validation examples remain acceptable.

## Risks / Trade-offs

- [Risk] Existing ad hoc commands and docs that point at `image_patches/` will fail. -> Mitigation: update configs/docs and make failures explicit with instructions to use full-image roots.
- [Risk] Removing `--no-dynamic-patching` could break a legitimate legacy inspection workflow. -> Mitigation: keep legacy utilities where needed, but move them out of supported training entrypoints.
- [Risk] Retiring mitochondria static patch directories is a runtime filesystem mutation outside Git. -> Mitigation: move directories to `_retired/` with a dated name, verify file counts and sizes, and record the move in implementation notes.
- [Risk] Dynamic training can still produce unstable metrics if validation crops are random. -> Mitigation: require fixed validation manifests or equivalent deterministic validation evidence before glomeruli model promotion.
- [Risk] A newly trained current-namespace model can still be scientifically poor. -> Mitigation: keep model artifact compatibility, runtime compatibility, and scientific promotion as separate gates.
- [Risk] Patch-related code remains and could be misused later. -> Mitigation: tests, CLI help, configs, and docs must state that static patches are retired for supported training; future code cleanup can remove leftover legacy utilities after references are gone.

## Migration Plan

1. Confirm the already-completed mitochondria static patch retirement at `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/_retired/mitochondria_static_patch_datasets_2026-04-22`.
2. Verify glomeruli static patches remain retired and are not recreated under active `derived_data/glomeruli_data`.
3. Update mitochondria and glomeruli training entrypoints to require full-image dynamic patching and remove `--no-dynamic-patching`.
4. Update transfer-learning helpers so glomeruli transfer learning always builds full-image dynamic-patching dataloaders.
5. Update configs to reference full-image roots and remove patch-generation-as-training sections. For mitochondria, point training at the physical `training/` full-image root and keep the physical `testing/` full-image root as held-out evaluation data.
6. Update tests to cover accepted full-image roots and rejected static patch roots.
7. Update docs and AGENTS/OpenSpec guidance to describe static patch datasets as retired legacy artifacts.
8. Add glomeruli promotion validation requirements covering training-data audit, fixed validation examples, trivial baselines, and prediction review.
9. Validate focused tests, CLI help, and OpenSpec strict validation.

Rollback is filesystem-level for runtime data: move retired directories back from `_retired/` only if a separate legacy workflow explicitly requires them. Code rollback should restore training flags only through a deliberate compatibility change, not as an implicit fallback.

## Resolved Questions

- Mitochondria full-image data will remain under `derived_data/mitochondria_data/{training,testing}/images` and `masks` for this change. Use `training/` as the selected dynamic-training root, with an internal train/validation split created by the dataloader; keep `testing/` as a held-out evaluation root. A cleaner raw mitochondria source root can be proposed separately if needed; it should not block removing active static patch training.
- Deterministic validation evidence is part of this change's promotion contract. Implementation may create a fixed validation manifest now or define the manifest-producing testable interface, but a glomeruli artifact cannot be promoted without fixed positive, boundary, and background validation examples.
- Patchification commands may remain exposed only if relabeled as legacy conversion/audit behavior. They must not be documented as producing supported training inputs.
- Existing static-patch-trained current-namespace model artifacts are compatibility artifacts. They do not need to be moved in this change, but metadata/docs must not present them as promoted or supported scientific artifacts.
