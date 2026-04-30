## ADDED Requirements

### Requirement: Fine-tuning workflow uses fixed admitted manual-mask splits
The system SHALL fine-tune and evaluate MedSAM/SAM glomeruli checkpoints using deterministic admitted manual-mask train, validation, and test split manifests.

#### Scenario: Split manifests are built
- **WHEN** `medsam_glomeruli_fine_tuning` runs without explicit split manifests
- **THEN** it SHALL read `raw_data/cohorts/manifest.csv` under the active runtime root
- **AND** it SHALL include only admitted rows with `lane_assignment` in `manual_mask_core` or `manual_mask_external`, non-empty image and mask paths, and existing image and mask files
- **AND** it SHALL assign splits by grouping on `source_sample_id` when present, or the strongest available subject/source identifier otherwise, so related rows do not cross train, validation, and test splits
- **AND** it SHALL write `splits/train.csv`, `splits/validation.csv`, and `splits/test.csv` under `output/segmentation_evaluation/medsam_glomeruli_fine_tuning/<run_id>/`
- **AND** each split row SHALL record `manifest_row_id`, `cohort_id`, `lane_assignment`, `source_sample_id`, `split_group_id`, `image_path`, `mask_path`, `split`, `selection_rank`, and `selection_reason`

#### Scenario: Split manifests already exist
- **WHEN** the workflow is configured with explicit train, validation, and test split manifest paths
- **THEN** it SHALL validate that each referenced image and mask exists under the active runtime root or as an allowed absolute path
- **AND** it SHALL record the split manifest paths and hashes in run provenance

### Requirement: Baseline evaluation precedes fine-tuning adoption
The workflow SHALL evaluate current automatic MedSAM, oracle-prompt MedSAM reference metrics when available, current segmenter candidates, and trivial baselines on the fixed validation/test examples before adopting a fine-tuned checkpoint.

#### Scenario: Baselines are evaluated
- **WHEN** fixed validation or test split rows are available
- **THEN** the workflow SHALL evaluate current automatic MedSAM using the configured proposal-box source
- **AND** it SHALL include oracle-prompt MedSAM reference metrics from configured completed oracle evidence or compute oracle-prompt metrics for the fixed examples when configured
- **AND** it SHALL evaluate configured current segmenter candidates using tiled full-field inference
- **AND** it SHALL compute all-background and all-foreground trivial mask baselines
- **AND** it SHALL write comparable metrics and overlays under `output/segmentation_evaluation/medsam_glomeruli_fine_tuning/<run_id>/`

#### Scenario: Baseline evidence is missing
- **WHEN** a fine-tuned checkpoint is evaluated without required baseline metrics for the same fixed examples
- **THEN** the workflow SHALL fail closed before marking the checkpoint or generated masks as adoptable

### Requirement: MedSAM/SAM domain adaptation records complete checkpoint provenance
The workflow SHALL store adapted MedSAM/SAM checkpoint artifacts under the runtime model root with enough provenance to reproduce the training run, adaptation mode, frozen/unfrozen parameter policy, and data contract.

#### Scenario: Constrained adaptation runs
- **WHEN** the configured MedSAM/SAM adaptation command completes successfully
- **THEN** the workflow SHALL write checkpoint artifacts under `models/medsam_glomeruli/<checkpoint_id>/`
- **AND** it SHALL record training command, environment, MedSAM repository path, base checkpoint path and hash when readable, code version, package versions, split manifest paths and hashes, adaptation mode, frozen and trainable component names, training hyperparameters, output checkpoint path, and training status

#### Scenario: Adaptation mode is selected
- **WHEN** the workflow prepares a MedSAM/SAM glomeruli adaptation run
- **THEN** it SHALL prefer frozen-image-encoder mask-decoder or prompt-related tuning when supported by the audited MedSAM code
- **AND** it MAY use adapter-style or similarly lightweight updates when mask-decoder-only tuning is unavailable or insufficient
- **AND** it SHALL NOT run full original-scale MedSAM retraining unless a separate future change explicitly authorizes that scope

#### Scenario: Local hardware feasibility is checked before full pilot
- **WHEN** the workflow is configured for local domain adaptation on Apple Silicon or another constrained local accelerator
- **THEN** it SHALL run a tiny adaptation smoke run (for example, one to two examples) before the full pilot
- **AND** it SHALL record backend, device, image size, batch size, elapsed time, and memory-related failure evidence in run provenance
- **AND** it SHALL set a feasibility status that is either `local_feasible` or `requires_external_accelerator`
- **AND** it SHALL NOT mark local full-pilot execution as required when feasibility status is `requires_external_accelerator`

#### Scenario: Fine-tuning dependency is unavailable
- **WHEN** the configured MedSAM/SAM repository, Python environment, base checkpoint, or fine-tuning entrypoint is missing
- **THEN** the workflow SHALL audit the official MedSAM training path documented by `https://github.com/bowang-lab/MedSAM`, including `pre_CT_MR.py`, `train_one_gpu.py`, `train_multi_gpus.sh`, and `utils/ckpt_convert.py` where applicable
- **AND** it SHALL either adapt that official training path to the constrained glomeruli domain-adaptation mode with explicit provenance or fail closed with a clear dependency error
- **AND** it SHALL NOT write checkpoint provenance that implies training completed unless a checkpoint was produced and validated

### Requirement: Fine-tuned checkpoints are evaluated through automatic prompts
The workflow SHALL evaluate fine-tuned checkpoints using the same automatic proposal-box mechanism used for practical generated-mask production.

#### Scenario: Fine-tuned automatic inference runs
- **WHEN** a fine-tuned checkpoint is available and fixed validation/test examples are selected
- **THEN** the workflow SHALL run automatic proposal-box MedSAM inference with that checkpoint
- **AND** it SHALL write fine-tuned automatic masks, prompt provenance, metrics, and overlays under the run evaluation root
- **AND** it SHALL compare fine-tuned results with current automatic MedSAM, current segmenters, and trivial baselines on the same fixed examples

#### Scenario: Fine-tuned prompt fails
- **WHEN** automatic inference fails for a fine-tuned checkpoint prompt or image
- **THEN** the workflow SHALL record the failure in `prompt_failures.csv`
- **AND** it SHALL NOT substitute manual, current-segmenter, current MedSAM, or empty masks as successful fine-tuned outputs

### Requirement: Generated-mask release is reusable derived data
The workflow SHALL package reusable fine-tuned MedSAM generated masks under the active runtime `derived_data` tree rather than under raw data or run diagnostics.

#### Scenario: Fine-tuned checkpoint passes generated-mask gates
- **WHEN** a fine-tuned checkpoint passes configured generated-mask adoption gates
- **THEN** the workflow MAY write a reusable generated-mask release under `derived_data/generated_masks/glomeruli/medsam_finetuned/<mask_release_id>/`
- **AND** the release SHALL contain `masks/`, `manifest.csv`, `INDEX.md`, and `provenance.json`
- **AND** the release manifest SHALL record `generated_mask_id`, `mask_release_id`, source image path, reference manual mask path when available, generated mask path, `mask_source=medsam_finetuned_glomeruli`, checkpoint ID, checkpoint path, proposal source, proposal threshold, run ID, generation status, adoption tier, and metric links when available

#### Scenario: Central generated-mask registry is updated
- **WHEN** a reusable generated-mask release is written
- **THEN** the workflow SHALL update or create `derived_data/generated_masks/glomeruli/manifest.csv`
- **AND** the central registry SHALL include `generated_mask_id`, `mask_release_id`, `mask_source`, `adoption_tier`, `cohort_id`, `lane_assignment`, `source_sample_id`, `source_image_path`, `reference_mask_path`, `generated_mask_path`, `release_manifest_path`, `checkpoint_id`, `checkpoint_path`, `proposal_source`, `proposal_threshold`, `run_id`, `generation_status`, and `provenance_path`
- **AND** downstream workflows SHALL prefer the central registry or a release-local manifest path over scanning `output/segmentation_evaluation/**` directories

#### Scenario: Generated masks would enter raw data
- **WHEN** any generated-mask output path resolves under `raw_data/cohorts/**/images` or `raw_data/cohorts/**/masks`
- **THEN** the workflow SHALL fail closed with a clear path-isolation error
- **AND** it SHALL NOT write generated masks, overlays, metrics, or provenance into raw cohort directories

### Requirement: Practical generated-mask adoption is gated separately from scientific promotion
The workflow SHALL distinguish practical generated-mask adoption from scientific model promotion.

#### Scenario: Oracle-level target is evaluated
- **WHEN** fine-tuned automatic MedSAM metrics are computed on fixed validation/test examples
- **THEN** `summary.json` SHALL report the fine-tuned checkpoint's gap from oracle-prompt MedSAM reference metrics when oracle reference metrics are available
- **AND** it SHALL report whether the checkpoint passed configured oracle-level gates including minimum Dice, minimum Jaccard, and maximum oracle gap
- **AND** the initial configurable oracle-level gate defaults SHALL be `min_dice=0.90`, `min_jaccard=0.82`, and `max_oracle_dice_gap=0.05`

#### Scenario: Generated-mask adoption gates pass
- **WHEN** fine-tuned automatic MedSAM reaches configured oracle-level gates on fixed validation/test examples
- **AND** it improves over current automatic MedSAM and configured current segmenters on fixed validation/test examples
- **AND** prompt failures, foreground fraction, area ratio, trivial-baseline comparisons, and overlay review gates pass
- **THEN** `summary.json` SHALL set `recommended_generated_mask_source` to `medsam_finetuned_glomeruli`
- **AND** it SHALL set `primary_generated_mask_transition_status` to `oracle_level_preferred`
- **AND** reusable release manifests and the central generated-mask registry SHALL record `adoption_tier=oracle_level_preferred`

#### Scenario: Fine-tuned checkpoint improves but is not oracle-level
- **WHEN** fine-tuned automatic MedSAM improves over current automatic MedSAM and configured current segmenters
- **AND** prompt failures, foreground fraction, area ratio, trivial-baseline comparisons, and overlay review gates pass
- **AND** configured oracle-level gates do not pass
- **THEN** `summary.json` SHALL set `primary_generated_mask_transition_status` to `improved_candidate_not_oracle`
- **AND** any reusable release manifest and central generated-mask registry rows SHALL record `adoption_tier=improved_candidate_not_oracle`
- **AND** the release SHALL be eligible only for explicit downstream comparison or audit opt-in, not as the preferred generated-mask source

#### Scenario: Scientific promotion is requested
- **WHEN** the fine-tuned checkpoint or generated-mask release is proposed as scientifically promoted
- **THEN** the promotion evidence SHALL include downstream grading stability, non-degenerate prediction review, fixed validation/test metrics, and generated-mask release provenance
- **AND** segmentation metrics alone SHALL NOT be treated as sufficient scientific promotion

#### Scenario: Gates fail
- **WHEN** fine-tuned automatic MedSAM does not pass generated-mask adoption gates
- **THEN** `summary.json` SHALL classify the failure mode as `oracle_gap`, `training_quality`, `prompt_geometry`, `data_split_limit`, `downstream_integration`, or `none_detected`
- **AND** the workflow SHALL set `primary_generated_mask_transition_status` to `blocked`
- **AND** the workflow SHALL NOT write a reusable generated-mask release unless reliability gates pass and the release is explicitly labeled with a non-preferred candidate or audit adoption tier
