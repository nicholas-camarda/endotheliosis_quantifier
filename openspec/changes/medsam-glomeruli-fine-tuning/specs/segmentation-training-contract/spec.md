## ADDED Requirements

### Requirement: MedSAM domain adaptation uses admitted manual-mask references
Supported MedSAM/SAM glomeruli domain adaptation SHALL use admitted manual-mask cohort rows as reference supervision and SHALL NOT train from generated masks as if they were manual truth.

#### Scenario: Adaptation training data is selected
- **WHEN** MedSAM/SAM glomeruli domain adaptation builds its training split
- **THEN** it SHALL enumerate admitted `manual_mask_core` and `manual_mask_external` rows from `raw_data/cohorts/manifest.csv`
- **AND** it SHALL record each image and manual mask source path in the split manifest
- **AND** it SHALL NOT include `derived_data/generated_masks/**` rows as manual-reference supervision unless a future explicit semi-supervised contract supports that mode

#### Scenario: Generated-mask release is inspected
- **WHEN** a generated-mask release under `derived_data/generated_masks/glomeruli/**` is inspected
- **THEN** it SHALL be labeled as generated data with `mask_source`
- **AND** reusable generated glomeruli masks SHALL be discoverable through `derived_data/generated_masks/glomeruli/manifest.csv`
- **AND** it SHALL NOT be presented as the canonical raw/manual training-data root

### Requirement: MedSAM fine-tuned checkpoints record supported provenance
Supported MedSAM/SAM glomeruli adapted checkpoints SHALL record enough provenance to identify training inputs, base checkpoint, code state, environment, adaptation mode, frozen/unfrozen parameter policy, and generated-mask release eligibility.

#### Scenario: Fine-tuned checkpoint is exported
- **WHEN** a MedSAM/SAM glomeruli fine-tuning run exports a checkpoint
- **THEN** its checkpoint directory under `models/medsam_glomeruli/<checkpoint_id>/` SHALL include provenance for training command, code version, package versions, MedSAM repository path, base checkpoint path and hash when readable, split manifest paths and hashes, adaptation mode, frozen and trainable component names, training hyperparameters, and output checkpoint files

#### Scenario: Fine-tuned checkpoint lacks provenance
- **WHEN** a MedSAM/SAM checkpoint lacks required provenance or cannot identify its admitted manual-mask split manifests
- **THEN** it SHALL be treated as historical, audit-only, or unsupported until rerun or documented through the supported fine-tuning workflow

### Requirement: MedSAM generated-mask adoption remains separate from scientific promotion
MedSAM/SAM fine-tuning completion and generated-mask release creation SHALL NOT by themselves promote the model scientifically.

#### Scenario: Fine-tuned checkpoint runs successfully
- **WHEN** a fine-tuned MedSAM/SAM checkpoint loads and completes automatic-prompt inference
- **THEN** that result establishes runtime compatibility and generated-mask candidate evidence only
- **AND** it SHALL NOT be treated as scientific promotion without downstream grading stability and non-degenerate prediction review evidence

#### Scenario: Generated-mask release is used downstream
- **WHEN** a downstream workflow consumes `derived_data/generated_masks/glomeruli/manifest.csv` or `derived_data/generated_masks/glomeruli/medsam_finetuned/<mask_release_id>/manifest.csv`
- **THEN** downstream provenance SHALL record `mask_source=medsam_finetuned_glomeruli`, central registry or release manifest path, adoption tier, checkpoint ID, and release provenance path
- **AND** downstream interpretation SHALL distinguish generated-mask input evidence from manual-reference truth
