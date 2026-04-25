## 1. Audit And Config Baseline

- [x] 1.1 Document the current augmentation reality in the change review notes: DataBlock FastAI transforms are active, `configs/glomeruli_finetuning_config.yaml` augmentation fields are not the candidate-comparison control surface, and no augmentation ablation has been completed.
- [x] 1.2 Add `negative_background_supervision` and `augmentation_audit` sections to `configs/glomeruli_candidate_comparison.yaml` with explicit defaults and comments.
- [x] 1.3 Update the p3 quick-test config pattern or add a new quick-test config for short negative/background supervision validation.

## 2. Negative Crop Manifest And Review Tooling

- [x] 2.1 Implement a manifest schema/validator for supported negative crop manifests with hard failures for missing fields, unsupported labels, unreviewed rows, invalid crop boxes, or missing source files.
- [x] 2.2 Implement mask-derived background crop manifest generation from admitted paired image/mask rows with zero-foreground crop validation.
- [x] 2.3 Implement MR/TIFF review-batch proposal generation under `derived_data/glomeruli_negative_crops/review_assets/<curation_id>/` with proposal manifests marked non-trainable until reviewed.
- [x] 2.4 Add CLI or workflow-accessible entrypoints for generating mask-derived background manifests and MR/TIFF review batches without creating static patch training roots.

## 3. Training Integration

- [x] 3.1 Extend the dynamic-patching DataBlock/sampler to consume validated negative crop manifests as additional supervised crop specs.
- [x] 3.2 Preserve standard full-image dynamic patching and positive-aware sampling; do not replace it with a static patch dataset.
- [x] 3.3 Thread negative/background supervision config through `train_glomeruli.py`, `transfer_learning.py`, `compare_glomeruli_candidates.py`, and `run_glomeruli_candidate_comparison_workflow.py`.
- [x] 3.4 Record negative/background supervision fields in split manifests, training histories, run metadata, and exported model metadata.

## 4. Candidate Comparison And Augmentation Evidence

- [x] 4.1 Add negative/background supervision provenance to `candidate_summary.csv`, `promotion_report.json`, `promotion_report.md`, and review artifacts.
- [x] 4.2 Add augmentation policy evidence to training provenance and candidate reports, including whether gaussian noise or config-defined augmentation was actually active.
- [x] 4.3 Ensure promotion gates still require background category performance and do not promote a candidate solely because aggregate Dice improves.

## 5. Tests

- [x] 5.1 Add unit tests for manifest validation, including rejection of unreviewed MR/TIFF proposals.
- [x] 5.2 Add unit tests for mask-derived background crop generation and zero-overlap guarantees.
- [x] 5.3 Add sampler/DataBlock tests proving negative crop samples produce all-zero masks and do not require active static patch roots.
- [x] 5.4 Add config/CLI dry-run tests for the new negative/background supervision controls.
- [x] 5.5 Add candidate-comparison report tests for negative supervision and augmentation provenance fields.

## 6. Validation

- [x] 6.1 Run `env OPENSPEC_TELEMETRY=0 openspec validate p0-negative-background-glomeruli-supervision --strict`.
- [x] 6.2 Run `python3 scripts/check_openspec_explicitness.py openspec/changes/p0-negative-background-glomeruli-supervision`.
- [x] 6.3 Run focused pytest for negative crop validation, sampler integration, config dry-runs, and candidate report provenance.
- [x] 6.4 Run `/Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m pytest -q`.
- [x] 6.5 Run a short MPS candidate-comparison smoke test after implementation with negative/background supervision enabled and record the output path, background false-positive fractions, and promotion status in implementation notes.
