# segmentation-training-contract Specification

## Purpose
Define the supported full-image dynamic-patching training contract, artifact provenance requirements, and promotion-related training constraints for segmentation models in this repository.
## Requirements
### Requirement: Segmentation training uses full-image data roots
Supported segmentation training SHALL use data roots that contain full-image `images/` and corresponding `masks/` directories.

#### Scenario: Mitochondria training receives a full-image root
- **WHEN** mitochondria training is started with a data root containing `images/` and `masks/`
- **THEN** training builds dataloaders from the full-image pairs using dynamic patching

#### Scenario: Glomeruli training receives a full-image root
- **WHEN** glomeruli training is started with a data root containing `images/` and `masks/`
- **THEN** training builds dataloaders from the full-image pairs using dynamic patching

#### Scenario: Required full-image directories are missing
- **WHEN** supported segmentation training is started with a data root that does not contain both `images/` and `masks/`
- **THEN** training fails before model construction with an error that identifies the missing full-image training contract

### Requirement: Curated trainable image-mask roots use the raw-data contract
When segmentation training consumes paired image/mask files directly or through the unified admitted-mask cohort registry, those trainable roots SHALL be treated as `raw_data` assets rather than `derived_data` artifacts.

#### Scenario: Glomeruli all-data training root is defined
- **WHEN** glomeruli training should use all currently admitted masked rows
- **THEN** the canonical root is the manifest-backed `raw_data/cohorts` registry root
- **AND** the loader enumerates admitted `manual_mask_core` and `manual_mask_external` rows with runtime-local image and mask paths

#### Scenario: Glomeruli project-only paired root is defined
- **WHEN** glomeruli full-image pairs are curated for a project-only training run
- **THEN** the supported root lives under `raw_data/...` and contains direct paired `images/` and `masks/`, such as `raw_data/cohorts/lauren_preeclampsia`
- **AND** that root contains only trainable paired images and masks

#### Scenario: Raw backup source tree contains unpaired files
- **WHEN** a raw backup tree such as `clean_backup` contains images without matching masks
- **THEN** it is treated as source material to curate or localize from, not as the canonical supported training root

#### Scenario: Derived data contract is inspected
- **WHEN** manifests, audits, caches, metrics, or evaluation artifacts are produced from segmentation training data
- **THEN** those generated outputs live under `derived_data/...`
- **AND** they are not presented as the canonical trainable glomeruli image/mask root
- **AND** `derived_data/glomeruli_data` and `derived_data/mitochondria_data` are not supported active training-data roots

### Requirement: Physical installed splits remain distinct from dynamic train/validation splits
Supported segmentation training SHALL preserve existing physical full-image `training/` and `testing/` layouts, while dynamic patching creates the internal train/validation split from the selected training root.

#### Scenario: Mitochondria installed layout is preserved
- **WHEN** `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/raw_data/mitochondria_data` is inspected after static patch retirement
- **THEN** `training/images`, `training/masks`, `testing/images`, and `testing/masks` remain active full-image directories
- **AND** those directories are not merged, renamed, or flattened by this change

#### Scenario: Mitochondria dynamic training uses the training root only
- **WHEN** mitochondria training is started with `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/raw_data/mitochondria_data/training` as the data root
- **THEN** dynamic patching creates the internal train/validation split from that selected `training/` root
- **AND** the sibling `testing/` root is not silently included in training or validation

#### Scenario: Mitochondria held-out testing is deliberate
- **WHEN** mitochondria held-out evaluation is run
- **THEN** it uses `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/raw_data/mitochondria_data/testing` only through an explicit evaluation path
- **AND** held-out testing examples do not affect training-time validation metrics

#### Scenario: Glomeruli curated paired root splits internally
- **WHEN** glomeruli training uses a curated paired full-image root under `raw_data`
- **THEN** dynamic patching may create an internal train/validation split from that selected root
- **AND** this does not imply that retired static `training/`, `testing/`, or `prediction/` patch directories are active model-training inputs

#### Scenario: Mitochondria installed runtime uses raw data
- **WHEN** the current Lucchi-derived mitochondria runtime is inspected
- **THEN** its installed full-image `training/` and `testing/` roots live under `raw_data/mitochondria_data`
- **AND** the mitochondria dataset remains outside the scored-cohort registry at `raw_data/cohorts/manifest.csv`

### Requirement: Static patch roots are not supported training inputs
Supported segmentation training SHALL NOT accept pre-generated `image_patches/` and `mask_patches/` directories as training inputs.

#### Scenario: Static patch root is supplied to mitochondria training
- **WHEN** mitochondria training is started with a data root whose active training inputs are `image_patches/` and `mask_patches/`
- **THEN** training fails before model construction and instructs the user to use the full-image `images/` and `masks/` root

#### Scenario: Static patch root is supplied to glomeruli training
- **WHEN** glomeruli training is started with a data root whose active training inputs are `image_patches/` and `mask_patches/`
- **THEN** training fails before model construction and instructs the user to use the full-image `images/` and `masks/` root

#### Scenario: Static patch utilities are used outside supported training
- **WHEN** a legacy audit, conversion, or historical-inspection workflow uses static patch utilities
- **THEN** that workflow is explicitly labeled legacy or audit-only and is not presented as a supported model-training path

### Requirement: Dynamic patching is the only supported segmentation training mode
The segmentation training CLIs and configs SHALL NOT expose a supported option to disable dynamic patching.

#### Scenario: Training CLI help is displayed
- **WHEN** the mitochondria or glomeruli training CLI help is displayed
- **THEN** the help does not present `--no-dynamic-patching` or an equivalent supported static-patch training option

#### Scenario: Config examples are inspected
- **WHEN** segmentation training config files are inspected
- **THEN** they point to full-image roots and do not configure `image_patches/` or `mask_patches/` as model-training inputs

#### Scenario: Transfer learning creates glomeruli dataloaders
- **WHEN** glomeruli transfer learning builds target dataloaders
- **THEN** it uses full-image dynamic patching and does not silently fall back to static patch dataloaders

### Requirement: Powerful Apple Silicon MPS uses higher starting batch defaults
On powerful Apple Silicon systems using MPS, local segmentation training SHALL use machine-aware starting batch defaults while keeping explicit overrides available.

#### Scenario: Mitochondria local training starts on a powerful Apple Silicon MPS machine
- **WHEN** mitochondria training starts with `256x256` crops on the certified powerful Apple Silicon MPS machine class
- **THEN** the default starting batch size is `24`
- **AND** CLI or config overrides may still change the batch size

#### Scenario: Glomeruli local training starts on a powerful Apple Silicon MPS machine
- **WHEN** glomeruli training starts with `512x512` crops on the certified powerful Apple Silicon MPS machine class
- **THEN** the default starting batch size is `12`
- **AND** CLI or config overrides may still change the batch size

#### Scenario: Throughput or stability requires a different batch
- **WHEN** MPS throughput, memory pressure, or stability indicates the machine-aware starting batch should change
- **THEN** the operator may override the batch size without changing the underlying training contract

### Requirement: Static patch datasets are retired from active runtime training locations
Active ProjectsRuntime segmentation training directories SHALL NOT contain static patch directories for supported training.

#### Scenario: Glomeruli runtime data is inspected
- **WHEN** `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/derived_data/glomeruli_data` is inspected after this change
- **THEN** it is absent from active runtime training locations
- **AND** glomeruli training uses `raw_data/cohorts` or a direct `raw_data/cohorts/<cohort_id>` image/mask root

#### Scenario: Mitochondria runtime data is inspected
- **WHEN** `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/derived_data/mitochondria_data` is inspected after this change
- **THEN** it is absent from active runtime training locations
- **AND** mitochondria training uses `raw_data/mitochondria_data/training`

#### Scenario: Retired static patch data is preserved
- **WHEN** static patch directories are retired from active runtime locations
- **THEN** they are moved to a dated directory under `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/_retired/` and the implementation records moved paths, size, and file count

### Requirement: Supported segmentation artifacts record training provenance
Supported segmentation model artifacts SHALL record enough provenance to determine whether they were trained with the current full-image dynamic-patching contract.

#### Scenario: Supported model artifact is exported
- **WHEN** a supported segmentation model artifact is exported
- **THEN** its sidecar metadata records the training command, code version, package versions, data root, training mode, and output artifact paths

#### Scenario: Static-patch-trained artifact is encountered
- **WHEN** a segmentation artifact provenance indicates static patch training or lacks training-mode provenance
- **THEN** it is treated as historical or compatibility-only until retrained or re-exported through the supported full-image dynamic-patching contract

#### Scenario: Legacy FastAI pickle artifact is encountered
- **WHEN** a segmentation artifact requires removed project modules, old FastAI transform namespaces, incompatible NumPy pickle namespaces, or legacy namespace shims to load
- **THEN** it is treated as a historical artifact unless a separate compatibility change explicitly supports it

### Requirement: Scientific promotion remains separate from runtime support
Runtime support SHALL NOT be treated as scientific model promotion.

#### Scenario: Current-namespace artifact loads and runs
- **WHEN** a current-namespace segmentation artifact loads and completes a pipeline or training smoke check
- **THEN** that result establishes runtime compatibility only, not scientific promotion

#### Scenario: Glomeruli model is proposed for promotion
- **WHEN** a glomeruli segmentation artifact is proposed as a promoted scientific model
- **THEN** the promotion evidence includes training-data foreground/background coverage, validation metrics against trivial all-foreground and all-background baselines, and non-degenerate prediction review artifacts

#### Scenario: Dynamic training completes
- **WHEN** segmentation training completes with dynamic patching
- **THEN** completion alone does not promote the artifact without the required model-quality evidence

### Requirement: Glomeruli training-data audit is required for promotion
Glomeruli model promotion SHALL include an audit of the training and validation data used to produce the artifact.

#### Scenario: Glomeruli training data is audited
- **WHEN** a glomeruli segmentation artifact is proposed for scientific promotion
- **THEN** the audit reports foreground fraction distribution, background-only crop rate, full-foreground crop rate, and subject or image split coverage for the training and validation data

#### Scenario: Positive-only static patch data is detected
- **WHEN** a glomeruli training or validation set contains no background-only examples and rewards all-foreground predictions
- **THEN** the resulting artifact is blocked from promotion regardless of training completion or current-namespace loadability

#### Scenario: Runtime evidence from the 2026-04-22 static patch artifact is reviewed
- **WHEN** the static-patch-trained glomeruli compatibility artifact is used as comparison evidence
- **THEN** its metrics are compared against the all-foreground baseline and treated as evidence of the retired static patch failure mode, not as promotion evidence

### Requirement: Glomeruli promotion validation uses deterministic examples
Glomeruli model promotion SHALL use deterministic validation examples rather than relying only on stochastic dynamic validation crops.

#### Scenario: Fixed validation manifest is created
- **WHEN** a glomeruli model is evaluated for promotion
- **THEN** validation uses a fixed manifest or equivalent deterministic selection of positive, boundary, and background examples

#### Scenario: Promotion validation is rerun
- **WHEN** the same promoted-candidate artifact is evaluated multiple times against the fixed validation manifest
- **THEN** the selected validation examples and trivial-baseline comparisons remain stable across runs

#### Scenario: Dynamic training uses random crops
- **WHEN** dynamic patching uses stochastic crops during training
- **THEN** the training sampler may remain stochastic, but promotion validation still uses fixed validation examples

### Requirement: Glomeruli promotion compares against trivial baselines
Glomeruli model promotion SHALL compare candidate predictions against trivial all-background and all-foreground baselines.

#### Scenario: Candidate metrics are computed
- **WHEN** glomeruli promotion metrics are reported
- **THEN** Dice and Jaccard are reported for the candidate model, an all-background baseline, an all-foreground baseline, and the current compatibility artifact when available

#### Scenario: Candidate fails to beat trivial baselines
- **WHEN** a candidate model does not beat the relevant trivial baselines on the fixed validation examples
- **THEN** the candidate is blocked from scientific promotion

#### Scenario: Candidate only matches all-foreground behavior
- **WHEN** candidate Dice or Jaccard is close to the all-foreground baseline and prediction review shows broad foreground masks
- **THEN** the candidate is treated as degenerate and blocked from promotion

### Requirement: Glomeruli promotion includes prediction review artifacts
Glomeruli model promotion SHALL include quantitative and visual prediction-review artifacts.

#### Scenario: Prediction review is generated
- **WHEN** a glomeruli model is evaluated for promotion
- **THEN** review artifacts include sampled validation panels, prediction foreground fraction summaries, and evidence that predictions are not all foreground or all background

#### Scenario: Degenerate prediction review is detected
- **WHEN** prediction review shows all-foreground or all-background behavior on representative validation examples
- **THEN** the candidate is blocked from scientific promotion even if aggregate metrics appear acceptable

### Requirement: The updated contract must run end to end
This change SHALL NOT be considered complete until the repository can run the intended pipeline from the beginning under the updated contract and complete end to end.

#### Scenario: End-to-end validation is executed
- **WHEN** final validation for this change is performed
- **THEN** the operator runs the whole pipeline from the beginning using the updated path contract, dynamic training contract, and current local hardware defaults
- **AND** the run covers the real entrypoints needed to prove the updated workflow is executable rather than only isolated smoke tests

#### Scenario: End-to-end run finds a blocker
- **WHEN** the full pipeline fails at any step during final validation
- **THEN** the blocker is treated as a required issue to fix for this change rather than a deferred note
- **AND** validation is rerun from the beginning after the fix until the pipeline completes end to end

#### Scenario: End-to-end iteration is documented
- **WHEN** the full pipeline has been iterated to completion
- **THEN** the implementation records the executed path, every issue discovered during iteration, the fix applied for each issue, and any remaining residual limitations that do not block completion
- **AND** that record lives inside this OpenSpec change's artifacts rather than in separate implementation notes, side documents, or external logs

### Requirement: Glomeruli promotion uses a concrete candidate-comparison workflow
Promoting a glomeruli segmentation model SHALL require a concrete comparison of supported candidate artifacts rather than evaluating a single newly trained artifact in isolation.

#### Scenario: Promotion workflow is executed
- **WHEN** glomeruli promotion is attempted under the supported training contract
- **THEN** the workflow compares at least two supported candidate families under a shared deterministic validation manifest
- **AND** the compared families include mitochondria transfer learning and no-mitochondria-base ImageNet-initialized training unless one family is explicitly unavailable and reported as such

#### Scenario: Promotion workflow control surface is evaluated
- **WHEN** glomeruli candidate comparison is defined or documented
- **THEN** the supported top-level control surface is the dedicated candidate-comparison workflow config `configs/glomeruli_candidate_comparison.yaml` executed through `eq run-config`
- **AND** stale mixed workflow names such as `segmentation_fixedloader_full_retrain` and `fixedloader_full` are retired if they conflict with the supported training contract
- **AND** the underlying training-module commands remain recorded in provenance rather than serving as competing orchestration contracts

#### Scenario: Scratch glomeruli candidate requests larger crop context
- **WHEN** the canonical glomeruli training path is run with `--from-scratch`, `--image-size 256`, and `--crop-size 512`
- **THEN** the scratch training path SHALL preserve the requested `512` crop size through batch-size sizing, dynamic patching, and exported provenance
- **AND** it SHALL NOT silently replace the requested crop size with `256`
- **AND** the exported provenance SHALL identify the candidate as the no-mitochondria-base ImageNet-pretrained ResNet34 baseline rather than a literal all-random initialization baseline

#### Scenario: Promotion decision is recorded
- **WHEN** candidate comparison completes
- **THEN** the resulting promotion report records whether one candidate is promoted, no candidate is promoted, or the evidence is insufficient
- **AND** runtime-compatible but non-promoted artifacts remain labeled as non-promoted in provenance and documentation

#### Scenario: Candidate comparison validation is declared complete
- **WHEN** this change is treated as implementation-complete
- **THEN** completion evidence SHALL include an unsandboxed `eq-mac` candidate-comparison validation run where both transfer and scratch execute successfully under the supported runtime contract
- **AND** a report generated only from structured candidate-family failures SHALL NOT be treated as final completion evidence

### Requirement: Static patch roots are retired and not active code paths
Supported segmentation training SHALL use full-image `images/` and `masks/` roots with dynamic patching, and SHALL NOT expose active source-code paths that create, load, audit, or train from pre-generated `image_patches/` and `mask_patches/` trees.

#### Scenario: Static patch root is supplied to training
- **WHEN** mitochondria or glomeruli training is started with a data root whose active inputs are `image_patches/` and `mask_patches/`
- **THEN** training fails before model construction and instructs the user to use the full-image `images/` and `masks/` root

#### Scenario: Static patch utility is requested from the active CLI or package
- **WHEN** CLI help or active package exports are inspected
- **THEN** no command or public helper is exposed for patchifying, loading, or auditing static patch datasets

#### Scenario: Retired static patch code or data is preserved
- **WHEN** stale static patch code or runtime data is retired
- **THEN** it is moved to a dated `_retired/` location outside the active source/runtime input tree
- **AND** the implementation records the original path, retired path, size, and move timestamp

### Requirement: Dynamic patching is the only supported segmentation loading path
The active segmentation data loader SHALL provide only full-image dynamic-patching builders for supported model training.

#### Scenario: Active dataloader exports are inspected
- **WHEN** `eq.data_management.datablock_loader` or `eq.data_management` exports are inspected
- **THEN** they expose dynamic full-image training helpers and validators, not static patch dataloaders

### Requirement: Unmasked large-image crops are not implicit negative supervision
Segmentation training SHALL NOT silently treat crops from larger unmasked glomeruli source images as supported negative supervision.

#### Scenario: Training pipeline inspects an unlabeled large-image crop source
- **WHEN** glomeruli training or curation code encounters larger MR/TIFF source images without full masks
- **THEN** those images are treated as source material only
- **AND** their unlabeled crops SHALL NOT be treated as true negative glomeruli examples by default

#### Scenario: Curated negative crop supervision is added later
- **WHEN** glomeruli training uses negative crop supervision from larger unmasked source images
- **THEN** those negative crops must come from an explicit annotation manifest or equivalent provenance-backed source mapping
- **AND** the resulting training provenance records that curated negative crop supervision was used

### Requirement: Curated negative crop manifests are additional sampler inputs
Segmentation training SHALL consume curated negative glomeruli crop manifests only as additional supervised sampler inputs while preserving full-image dynamic patching as the canonical glomeruli training contract.

#### Scenario: Training is configured with curated negative crops
- **WHEN** glomeruli training receives a supported negative crop manifest
- **THEN** the primary positive and mask-bearing data source remains the selected full-image root or manifest-backed `raw_data/cohorts` registry
- **AND** the negative crop manifest contributes reviewed crop boxes to the sampler without creating or requiring active `image_patches/` or `mask_patches/` directories

#### Scenario: Training provenance is written
- **WHEN** glomeruli training writes model provenance or run metadata
- **THEN** it records `negative_crop_supervision_status`, `negative_crop_manifest_path`, `negative_crop_manifest_sha256`, `negative_crop_count`, `negative_crop_source_image_count`, `negative_crop_review_protocol_version`, and `negative_crop_sampler_weight`
- **AND** absence of a curated manifest is recorded as `negative_crop_supervision_status=absent`

### Requirement: Mask-derived background crops are supported negative supervision
Glomeruli training SHALL support background crop boxes from paired image/mask rows when the corresponding mask crop contains zero foreground pixels.

#### Scenario: Mask-derived background crop is accepted
- **WHEN** a crop box is generated from an admitted paired glomeruli image/mask row
- **AND** the mask crop contains zero foreground pixels
- **THEN** the crop SHALL be eligible for the `mask_derived_background` label
- **AND** it SHALL be eligible for supervised negative/background training evidence
- **AND** the source image, source mask, crop box, and zero-foreground validation result are recorded in a manifest or training provenance

#### Scenario: Mask-derived background crop overlaps foreground
- **WHEN** a proposed mask-derived background crop has any foreground pixels in the paired mask crop
- **THEN** it SHALL NOT be accepted as negative/background supervision
- **AND** the audit records the rejection count

### Requirement: Unreviewed MR/TIFF proposals are not trainable
Glomeruli training SHALL NOT use unmasked MR/TIFF crop proposals as negative supervision unless those rows have reviewed negative annotation status.

#### Scenario: MR/TIFF crop proposal is generated
- **WHEN** a crop proposal is generated from `raw_data/cohorts/vegfri_mr/images/`
- **THEN** it is recorded as `proposed_review_only`
- **AND** it SHALL NOT be consumed by training

#### Scenario: Reviewed MR/TIFF crop is accepted
- **WHEN** a curated negative manifest row has `label=negative_glomerulus`, `annotation_status=reviewed_negative`, and `negative_scope=crop_only`
- **THEN** training MAY consume that crop as supervised negative/background evidence
- **AND** the manifest path and hash are recorded in training provenance

### Requirement: Negative crop manifests are additional sampler inputs
Negative/background crop manifests SHALL be consumed as additional supervised sampler inputs while preserving full-image dynamic patching as the canonical glomeruli training mode.

#### Scenario: Training uses negative crop manifests
- **WHEN** glomeruli training is configured with a valid negative crop manifest
- **THEN** the DataBlock or sampler returns image crops and all-zero masks for those negative crop samples
- **AND** training still reads source pixels from canonical source image paths
- **AND** no active static `image_patches/` or `mask_patches/` training root is required

### Requirement: Training provenance records negative supervision state
Glomeruli training metadata SHALL disclose whether negative/background crop supervision was present.

#### Scenario: Training completes with negative supervision enabled
- **WHEN** a glomeruli model artifact is exported after using negative/background crop supervision
- **THEN** metadata records `negative_crop_supervision_status`, `negative_crop_manifest_path`, `negative_crop_manifest_sha256`, `negative_crop_count`, `mask_derived_background_crop_count`, `curated_negative_crop_count`, `negative_crop_source_image_count`, `negative_crop_review_protocol_version`, and `negative_crop_sampler_weight`

#### Scenario: Training completes without negative supervision
- **WHEN** no validated negative/background manifest is supplied
- **THEN** metadata records `negative_crop_supervision_status=absent`

### Requirement: Augmentation policy is explicit provenance
Glomeruli training metadata SHALL record the actual augmentation policy used by the DataBlock.

#### Scenario: Training uses the default FastAI augmentation policy
- **WHEN** glomeruli training builds DataLoaders with default batch transforms
- **THEN** metadata records the FastAI augmentation settings and repo constants
- **AND** it does not claim config-defined gaussian noise or brightness/contrast settings were active unless code actually applied them

### Requirement: Mitochondria transfer base records training-scope provenance
Mitochondria artifacts used as glomeruli transfer bases SHALL record whether they preserved the physical mitochondria testing split or used all available mitochondria pairs for representation pretraining.

#### Scenario: Mitochondria base artifact is exported
- **WHEN** `src/eq/training/train_mitochondria.py` exports a mitochondria base artifact
- **THEN** its sidecar metadata SHALL record `mitochondria_training_scope`, `mitochondria_inference_claim_status`, physical `training/` image count, physical `testing/` image count, actual pretraining image paths, actual pretraining mask paths, split policy, resize/preprocessing policy, training command, code version, and package versions
- **AND** the artifact SHALL record whether the physical `raw_data/mitochondria_data/testing` root was included in model fitting

#### Scenario: All mitochondria data are used for representation pretraining
- **WHEN** a mitochondria base uses both physical `training/` and `testing/` roots for model fitting
- **THEN** it SHALL set `mitochondria_training_scope=all_available_pretraining`
- **AND** it SHALL set `mitochondria_inference_claim_status=not_applicable_for_inference_claim`
- **AND** it MAY be used as a glomeruli transfer base only when glomeruli promotion evidence remains held-out and audit-passing

#### Scenario: Mitochondria testing split is preserved
- **WHEN** a workflow reports mitochondria held-out segmentation performance or uses mitochondria held-out metrics for model selection
- **THEN** the physical `raw_data/mitochondria_data/testing` root SHALL remain excluded from mitochondria training
- **AND** the artifact SHALL set `mitochondria_training_scope=heldout_test_preserved`
- **AND** it SHALL set `mitochondria_inference_claim_status=heldout_evaluable`

#### Scenario: Mitochondria scope is missing for transfer
- **WHEN** a glomeruli transfer candidate references a mitochondria base artifact with missing or inconsistent mitochondria training-scope provenance
- **THEN** the transfer candidate MAY remain `runtime_use_status=available_research_use` if it loads and runs
- **AND** its promotion evidence SHALL be `audit_missing` until transfer-base provenance is resolved

### Requirement: Glomeruli artifacts record split and sampler provenance
Supported glomeruli segmentation artifacts SHALL record enough split, sampler, crop, augmentation, and preprocessing provenance to audit whether model training and promotion evaluation were statistically separable.

#### Scenario: Glomeruli model artifact is exported
- **WHEN** `src/eq/training/train_glomeruli.py` or `src/eq/training/transfer_learning.py` exports a glomeruli candidate artifact
- **THEN** its sidecar metadata SHALL record `data_root`, `training_mode`, `candidate_family`, `seed`, `split_seed`, `splitter_name`, `train_images`, `valid_images`, `source_image_size_summary`, `source_mask_size_summary`, `crop_size`, `image_size`, `output_size`, `crop_to_output_resize_ratio`, `aspect_ratio_policy`, `resize_method`, image interpolation, mask interpolation, mask-binarization-after-resize semantics, prediction resize-back assumptions, threshold/resize ordering assumptions, `positive_focus_p`, `min_pos_pixels`, `pos_crop_attempts`, augmentation settings, learner preprocessing, transfer-base artifact path when applicable, transfer-base `mitochondria_training_scope` when applicable, training command, code version, and package versions
- **AND** the split sidecar SHALL be machine-readable enough for `segmentation_validation_audit.py` to compare candidate splits against deterministic promotion manifests

#### Scenario: Split sidecar cannot be audited
- **WHEN** an exported glomeruli artifact lacks train/validation image identifiers or the identifiers cannot be resolved
- **THEN** the artifact SHALL be classified as `runtime_use_status=available_research_use` if it loads and runs in the supported environment
- **AND** it SHALL be classified as `promotion_evidence_status=audit_missing`
- **AND** it SHALL NOT be treated as scientifically promoted or used for README-facing current-performance claims

### Requirement: DataBlock sampling audit is available for supported training roots
Supported segmentation training roots SHALL be auditable through the same DataBlock construction path used for training.

#### Scenario: DataBlock audit samples a supported glomeruli root
- **WHEN** `segmentation_validation_audit.py` audits `$EQ_RUNTIME_ROOT/raw_data/cohorts` or `$EQ_RUNTIME_ROOT/raw_data/cohorts/<cohort_id>`
- **THEN** it SHALL build DataLoaders through `build_segmentation_dls_dynamic_patching`
- **AND** it SHALL report crop foreground distributions for train and validation batches without writing generated training data into the repository

#### Scenario: Static patch root is encountered during audit
- **WHEN** the audit target is a retired or active static patch root such as `image_patches/` or `mask_patches/`
- **THEN** the audit SHALL fail closed with the same unsupported-root policy as training
- **AND** it SHALL NOT convert that root into an active training or validation input

### Requirement: Resize policy is auditable for supported training roots
Supported segmentation training SHALL expose the crop-to-network resize policy clearly enough to test whether the policy is technically aligned and promotion-supporting.

#### Scenario: Dynamic-patching DataLoaders resize crops
- **WHEN** `build_segmentation_dls_dynamic_patching` builds glomeruli DataLoaders with a `crop_size` different from `output_size`
- **THEN** the training provenance SHALL identify source image/mask size summaries, selected crop size, final network input size, crop-to-output resize ratio, aspect-ratio policy, resize method, image interpolation, mask interpolation, mask binarization semantics, and threshold/resize ordering assumptions
- **AND** the audit SHALL be able to distinguish the current `512 -> 256` downsampling policy from no-downsample or less-downsample sensitivity runs

#### Scenario: Source-resolution distributions are promotion-relevant
- **WHEN** glomeruli artifacts are compared for promotion
- **THEN** their provenance SHALL be sufficient to compare train, validation, and deterministic promotion source-resolution distributions
- **AND** the candidate SHALL NOT be `promotion_eligible` when resolution distribution imbalance is unresolved

#### Scenario: Resize policy benefit is not established
- **WHEN** a training artifact uses downsampling but lacks held-out resize-sensitivity evidence
- **THEN** the artifact MAY remain `runtime_use_status=available_research_use`
- **AND** it SHALL NOT be classified as `promotion_eligible` on resize-dependent performance claims

### Requirement: Dynamic validation split is not promotion evidence by itself
Training-time validation metrics from stochastic dynamic crops SHALL NOT be sufficient evidence for scientific promotion.

#### Scenario: Training completes with validation metrics
- **WHEN** glomeruli training completes and records training-time validation Dice or Jaccard
- **THEN** those metrics SHALL be treated as optimization diagnostics
- **AND** scientific promotion SHALL still require the held-out deterministic promotion manifest and validation audit gates

#### Scenario: Training and promotion use the same data root
- **WHEN** candidate training and candidate promotion both reference the admitted cohort registry root
- **THEN** the promotion workflow SHALL use recorded split provenance to select held-out evaluation images only
- **AND** it SHALL mark promotion evidence as `not_promotion_eligible` if held-out selection cannot be verified

### Requirement: Training provenance distinguishes runtime support from scientific promotion
The artifact provenance contract SHALL continue to separate current-namespace runtime support from scientific model promotion.

#### Scenario: Artifact loads successfully
- **WHEN** a glomeruli artifact loads in the certified environment and can run inference
- **THEN** the sidecar SHALL allow `artifact_status=supported_runtime`
- **AND** `runtime_use_status` SHALL allow `available_research_use`
- **AND** `promotion_evidence_status` SHALL remain `audit_missing`, `insufficient_evidence_for_promotion`, `not_promotion_eligible`, or `promotion_eligible` according to the hardened validation audit rather than loadability alone

#### Scenario: Artifact passes hardened audit
- **WHEN** a glomeruli artifact clears split integrity, DataBlock audit, deterministic held-out metrics, prediction-shape gates, and documentation-claim gates
- **THEN** it MAY be marked as scientifically promoted by the promotion report
- **AND** the sidecar SHALL reference the exact promotion report and validation audit payload that justified the status

