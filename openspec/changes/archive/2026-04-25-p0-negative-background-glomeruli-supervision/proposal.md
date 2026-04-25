## Why

The p3 quick run proved that the current glomeruli candidate-comparison workflow executes end-to-end on MPS, but both quick candidates still failed background review. The transfer candidate had background Dice `0.0` with median predicted foreground fraction `0.0513` on background crops, and the scratch candidate had background Dice `0.0` with median predicted foreground fraction `0.4889`. The failure mode is therefore not just insufficient epochs; it is insufficient negative/background supervision and insufficient evidence that the current augmentation/resize choices are helping.

The completed p1 change defined the negative-crop contract and recorded that unmasked MR/TIFF source images cannot be treated as true negatives without explicit crop-level review. It deliberately stopped before implementation. This change implements the missing training and evidence surfaces while preserving the existing rule: no unlabeled MR/TIFF crop may silently become negative supervision.

## What Changes

- Add an automatic **mask-derived background crop** generator for admitted masked glomeruli images. These crops are supported true negatives because their source masks can prove zero foreground overlap in the selected crop box.
- Add a **curated MR/TIFF negative crop review-batch** generator that proposes candidate crop boxes and review assets but does not mark them trainable until a reviewer writes a valid manifest row.
- Add a negative-crop manifest validator for `derived_data/glomeruli_negative_crops/manifests/<curation_id>.csv`.
- Extend glomeruli dynamic training so validated negative crop manifests can contribute supervised background crop samples without creating active static patch dataset roots.
- Add provenance fields to model metadata, split manifests, and candidate-comparison reports for negative/background supervision.
- Add config controls under `configs/glomeruli_candidate_comparison.yaml` for mask-derived background sampling, optional curated negative manifest use, and sampler weights.
- Add an explicit augmentation audit instead of treating the existing augmentation settings as proven. The audit must record what augmentation is actually applied, whether YAML augmentation fields are wired, and which augmentation variants are allowed in quick ablation runs.
- Add tests that fail if unreviewed MR/TIFF crops are used as negative supervision, if a malformed negative manifest is accepted, or if negative supervision provenance disappears from reports.

## Explicit Decisions

- Change ID: `p0-negative-background-glomeruli-supervision`.
- Mask-derived negative source: admitted masked glomeruli rows from `raw_data/cohorts/manifest.csv` with crop boxes whose corresponding mask crop has zero foreground pixels.
- Curated unmasked source: `raw_data/cohorts/vegfri_mr/images/`, with source provenance from `raw_data/cohorts/vegfri_mr/metadata/source_audit.csv`.
- Canonical generated roots:
  - `derived_data/glomeruli_negative_crops/manifests/<curation_id>.csv`
  - `derived_data/glomeruli_negative_crops/audits/<curation_id>.json`
  - `derived_data/glomeruli_negative_crops/review_assets/<curation_id>/`
- Runtime training remains dynamic full-image patching. Negative/background crop manifests are additional sampler inputs, not a replacement training root and not a revived `image_patches/` / `mask_patches/` workflow.
- Supported negative labels:
  - `mask_derived_background` for zero-mask-overlap crops from paired image/mask rows
  - `negative_glomerulus` for explicitly reviewed unmasked-source crops
- Unreviewed MR/TIFF proposals have status `proposed_review_only` and are never trainable.
- Training provenance must include `negative_crop_supervision_status`, `negative_crop_manifest_path`, `negative_crop_manifest_sha256`, `negative_crop_count`, `mask_derived_background_crop_count`, `curated_negative_crop_count`, `negative_crop_source_image_count`, `negative_crop_review_protocol_version`, and `negative_crop_sampler_weight`.
- Augmentation is not considered solved by this change. It is an audited secondary axis. The primary first retrain after implementation should change negative/background supervision while keeping augmentation and resize policy fixed unless an explicit ablation config is selected.

## Open Questions

- [audit_first_then_decide] What default mask-derived background sampler weight should the production candidate comparison use? Evidence source: quick smoke runs that compare foreground false positives and positive recall at weights `0.25`, `0.5`, and `1.0` before full training.
- [audit_first_then_decide] Should curated MR/TIFF negative crops be included in the first production retrain or only after human review? Evidence source: validated manifest row count and review provenance. If no reviewed manifest exists, production retrain must proceed with mask-derived background only.
- [audit_first_then_decide] Which augmentation variants are worth testing after negative/background supervision is wired? Evidence source: augmentation audit and short ablation runs. Candidate variants are current FastAI defaults, spatial-only, and current-plus-lighting; gaussian noise must not be claimed as active unless implemented and recorded.

## Capabilities

### Modified Capabilities

- `segmentation-training-contract`: add mask-derived and curated negative crop manifest ingestion while preserving dynamic full-image training.
- `glomeruli-candidate-comparison`: require negative/background supervision provenance and category-level background metrics in comparison reports.
- `workflow-config-entrypoints`: expose negative/background supervision and augmentation audit controls through `configs/glomeruli_candidate_comparison.yaml`.

## Impact

- Affected code: `src/eq/data_management/datablock_loader.py`, new negative-crop curation/validation helpers under `src/eq/data_management/` or `src/eq/training/`, `src/eq/training/train_glomeruli.py`, `src/eq/training/transfer_learning.py`, `src/eq/training/compare_glomeruli_candidates.py`, `src/eq/training/segmentation_validation_audit.py`, `src/eq/training/run_glomeruli_candidate_comparison_workflow.py`, and `src/eq/run_config.py`.
- Affected configs: `configs/glomeruli_candidate_comparison.yaml` and the p3 quick-test config pattern.
- Affected tests: data contract tests, training sampler tests, candidate-comparison report tests, CLI/config dry-run tests, and OpenSpec validation.
- Affected runtime artifacts: negative crop manifests/audits/review assets under `derived_data/glomeruli_negative_crops/...`, model metadata under `models/segmentation/glomeruli/...`, and comparison reports under `output/segmentation_evaluation/glomeruli_candidate_comparison/...`.
