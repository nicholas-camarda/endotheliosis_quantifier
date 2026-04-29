# endotheliosis-grade-model Specification

## Purpose
Define the P3 autonomous quantification workflow that attempts to produce the best supportable current-data endotheliosis product from existing ROI, mask, score, morphology, learned-feature, and adjudication artifacts: severe-risk triage, ordinal burden bands, diagnostic ordinal output, or a clear current-data-insufficient verdict.

## Requirements
### Requirement: P3 uses existing evidence and writes a contained product subtree
The workflow SHALL consume existing quantification artifacts and write all P3 outputs under a contained runtime subtree.

#### Scenario: P3 output root is contained
- **WHEN** P3 runs from the full quantification workflow
- **THEN** it SHALL write artifacts under `burden_model/endotheliosis_grade_model/`
- **AND** it SHALL not mutate archived P2 artifacts
- **AND** it SHALL preserve row identity, subject identity, cohort identity, score, ROI paths, raw image paths, mask paths, and adjudication provenance where available

#### Scenario: Existing burden-model artifacts are referenced not duplicated
- **WHEN** P3 consumes feature tables, candidate screens, comparator predictions, or evidence from existing burden-model subtrees
- **THEN** it SHALL read them from their existing locations under `primary_burden_index/`, `learned_roi/`, `source_aware_estimator/`, or `severe_aware_ordinal_estimator/`
- **AND** it SHALL write `summary/input_artifact_index.json` with each consumed artifact path, role, source subtree, and read-only/copy status
- **AND** rerun outputs for those existing families SHALL remain in their existing family subtrees

#### Scenario: Row-level feature table joins preserve ROI identity
- **WHEN** P3 consumes row-level optional feature tables from `primary_burden_index/feature_sets/morphology_features.csv` or `learned_roi/feature_sets/learned_roi_features.csv`
- **THEN** it SHALL join those tables to candidate rows by `subject_image_id` and `glomerulus_id`
- **AND** `learned_roi/feature_sets/learned_roi_features.csv` SHALL preserve `glomerulus_id` when it is present in the input embedding or ROI rows
- **AND** P3 SHALL NOT silently fall back to a `subject_image_id`-only join for ROI-level tables
- **AND** missing required row-level join keys or duplicate optional-table join keys SHALL be recorded as hard blockers
- **AND** the workflow SHALL continue to write diagnostics describing the unusable join rather than crashing with an unhandled pandas merge error

#### Scenario: Newly fit model families are first-class subtrees
- **WHEN** P3 fits a new committed model family
- **THEN** it SHALL write that family under a first-class `burden_model/<family_id>/` subtree
- **AND** expected family subtrees SHALL include `three_band_ordinal_model/`, `four_band_ordinal_model/` when supported, `severe_triage_model/`, `aggregate_grade_model/`, and `embedding_grade_model/`
- **AND** each first-class family subtree SHALL contain `INDEX.md`, `summary/`, `diagnostics/`, `predictions/`, `model/`, `evidence/`, and `internal/`
- **AND** no required newly fit family SHALL be represented only in `burden_model/endotheliosis_grade_model/internal/`

#### Scenario: Model-family diagnostics are first-class
- **WHEN** a first-class model-family subtree is written
- **THEN** it SHALL include `diagnostics/input_support.json`, `diagnostics/feature_diagnostics.json`, `diagnostics/fold_diagnostics.json`, `diagnostics/source_sensitivity.json`, and `diagnostics/gate_diagnostics.json`
- **AND** it SHALL include `diagnostics/embedding_source_predictability.json` when embedding-heavy features are evaluated
- **AND** it SHALL include `diagnostics/aggregate_label_diagnostics.json` when aggregate-aware features are evaluated
- **AND** it SHALL include `diagnostics/calibration_diagnostics.json` when calibration is used
- **AND** it SHALL include `diagnostics/threshold_selection_diagnostics.json` when thresholds are selected
- **AND** it SHALL include `diagnostics/hard_blockers.json` when expected inputs or required runtime conditions are missing

#### Scenario: Expected upstream inputs are hard requirements
- **WHEN** a required candidate family depends on existing morphology, ROI/QC, learned ROI, source-aware, severe-aware, scored/mask-paired, adjudication, or MR TIFF artifacts declared by P3
- **AND** the expected artifact is missing or unusable
- **THEN** P3 SHALL record a hard blocker
- **AND** it SHALL not silently downgrade that family to skipped or best-effort status

#### Scenario: Candidate family names are source-truthful
- **WHEN** P3 registers learned-feature or embedding-heavy candidate families
- **THEN** learned-feature candidates SHALL require actual learned-feature columns
- **AND** embedding-heavy candidates SHALL require actual embedding columns
- **AND** a missing source-specific feature set SHALL be recorded as unavailable rather than silently replacing the source-specific candidate with morphology-only or ROI/QC-only columns

#### Scenario: Common modeling mechanics are centralized
- **WHEN** P3 performs finite feature preprocessing, recall-targeted threshold selection, warning capture, JSON writing, artifact manifest writing, or grouped-development metric labeling
- **THEN** it SHALL use shared helpers from `src/eq/quantification/modeling_contracts.py`
- **AND** it SHALL not create another local implementation of those repeated contracts inside the P3 selector

#### Scenario: Existing adjudications are consumed
- **WHEN** `burden_model/severe_aware_ordinal_estimator/evidence/severe_false_negative_adjudications.json` exists
- **THEN** P3 SHALL use it to define the primary adjudicated `score >= 2` severe target
- **AND** it SHALL record how many labels came from original scores versus reviewer adjudication
- **AND** it SHALL keep the original score available for ordinal targets

#### Scenario: Reviewed rubric label overrides are first-class inputs
- **WHEN** `configs/endotheliosis_quantification.yaml` declares `inputs.label_overrides`
- **THEN** the quantification workflow SHALL apply explicit reviewer scores before ROI extraction, embedding extraction, burden modeling, learned ROI modeling, source-aware modeling, severe-aware modeling, and P3 selection
- **AND** the override file SHALL be keyed by `subject_image_id`
- **AND** the override score SHALL come from one of `rubric_score`, `reviewer_score`, `reviewer_grade`, or `score`
- **AND** the workflow SHALL write `scored_examples/score_label_overrides_audit.csv` and `scored_examples/score_label_overrides_summary.json`
- **AND** the summary SHALL state that only explicit reviewed label overrides were applied and that no inferred labels were applied
- **AND** missing override files, duplicate override rows, unsupported scores, nonnumeric scores, or override rows absent from scored examples SHALL fail the workflow rather than silently skipping rows

#### Scenario: Image-level aggregate labels are explicit
- **WHEN** an input image or mask contains multiple glomerulus components
- **THEN** P3 SHALL treat the available score as an image-level aggregate label
- **AND** it SHALL NOT treat that score as a true per-glomerulus label for every component
- **AND** it SHALL record multi-component counts and aggregate-label status in a compact diagnostics artifact under `internal/` or `summary/`

### Requirement: P3 uses grouped development validation and final refit
The workflow SHALL use current labeled scored/mask-paired data for subject-heldout grouped development validation and final refit rather than reserving a fixed internal locked test set by default.

#### Scenario: Development folds are written before fitting
- **WHEN** P3 starts candidate evaluation
- **THEN** it SHALL write `splits/development_folds.csv`
- **AND** the file SHALL identify subjects, fold assignments, row counts, score counts, severe-positive counts, and cohort counts
- **AND** the folds SHALL be generated before candidate fitting or threshold selection

#### Scenario: All eligible labeled current data are used
- **WHEN** enough subjects and severe positives are available
- **THEN** P3 SHALL include all eligible labeled scored/mask-paired subjects in grouped development validation
- **AND** it SHALL not reserve a fixed internal locked test split by default
- **AND** current-data metrics SHALL be labeled grouped out-of-fold development estimates, not independent heldout estimates

#### Scenario: Development folds report score coverage
- **WHEN** development folds are created
- **THEN** it SHALL attempt to include rows from all supported score values `[0, 0.5, 1, 1.5, 2, 3]`
- **AND** if subject grouping makes any score absent from a fold, it SHALL record the absent score and reason

#### Scenario: No fold leakage is allowed
- **WHEN** candidate fitting, feature selection, threshold selection, calibration, or model selection runs
- **THEN** validation-fold labels SHALL NOT be used to fit transforms, select features, choose thresholds, calibrate probabilities, or select candidate families for that fold
- **AND** no `subject_id` SHALL appear in more than one development fold

#### Scenario: Final model is refit after recipe selection
- **WHEN** the selected model recipe, feature set, dimensionality strategy, calibration strategy, and threshold are fixed from grouped development evidence
- **THEN** P3 MAY refit the final selected model on all eligible labeled scored/mask-paired data
- **AND** the final refit SHALL be applied to MR TIFF deployment data without changing the selected recipe based on MR results

### Requirement: P3 uses Dox scored-no-mask smoke before MR TIFF deployment
The workflow SHALL build a governed Dox scored-no-mask smoke surface and run it before the larger MR TIFF deployment test when clean candidates exist.

#### Scenario: Dox scored-only resolution audit is written
- **WHEN** P3 prepares deployment-smoke inputs
- **THEN** it SHALL write `raw_data/cohorts/vegfri_dox/metadata/dox_scored_only_resolution_audit.csv`
- **AND** each Dox `scored_only` manifest row SHALL be resolved against the Dox Label Studio upload image tree
- **AND** clean resolved images SHALL be copied under `raw_data/cohorts/vegfri_dox/scored_no_mask_smoke/images/`
- **AND** the audit SHALL record the source Label Studio image path, localized runtime image path, image hash, match count, duplicate source-image-name flag, conflicting-score flag, missing-score flag, and clean-smoke eligibility status
- **AND** the master manifest SHALL store the localized smoke image in canonical `image_path`, keep `mask_path` empty, and add Dox-smoke eligibility/status columns including `eligible_dox_scored_no_mask_smoke`

#### Scenario: Clean Dox scored-no-mask smoke manifest is conservative
- **WHEN** the Dox scored-only resolution audit completes
- **THEN** it SHALL write `raw_data/cohorts/vegfri_dox/metadata/dox_scored_no_mask_smoke_manifest.csv`
- **AND** the clean manifest SHALL include only rows with exactly one resolved Label Studio upload image, a localized runtime image copy, a nonmissing score, no duplicate source image name, and no conflicting score for that source image name
- **AND** excluded rows SHALL remain in the audit with an explicit exclusion reason

#### Scenario: Dox smoke precedes MR TIFF smoke
- **WHEN** a final P3 model artifact exists and the master manifest contains rows with `eligible_dox_scored_no_mask_smoke = true`
- **THEN** P3 SHALL run the supported segmentation-to-quantification path on the localized clean Dox smoke images before MR TIFF deployment
- **AND** it SHALL write Dox smoke predictions, image-level summaries, and human-grade comparison evidence under `burden_model/endotheliosis_grade_model/deployment/`
- **AND** passing Dox smoke SHALL NOT by itself permit README-facing deployment
- **AND** failing or unavailable Dox smoke SHALL keep MR TIFF deployment blocked until the failure is resolved or explicitly recorded as a hard blocker

#### Scenario: Dox overcall triage is bounded
- **WHEN** Dox smoke preserves severe recall but fails severe precision
- **THEN** P3 SHALL write `dox_scored_no_mask_overcall_triage_queue.csv` and `dox_scored_no_mask_overcall_triage_report.html`
- **AND** the queue SHALL include representative false-positive clusters, highest-confidence false positives, threshold-boundary false positives, human severe references, and segmentation misses when those buckets exist
- **AND** the queue SHALL be built from deployment-computable prediction fields without rerunning segmentation or requiring broad manual relabeling

#### Scenario: Reviewed Dox overcalls reject the selected model
- **WHEN** reviewed cluster-representative Dox false positives show usable non-severe ROIs are being called severe
- **THEN** P3 SHALL write `dox_scored_no_mask_overcall_review_diagnostic.json` and `dox_scored_no_mask_first12_review_interpretation.csv`
- **AND** P3 SHALL set `dox_overcall_confirmed = true` in `summary/final_product_verdict.json`
- **AND** P3 SHALL downgrade the selected severe-risk model to `diagnostic_only_current_data_model`
- **AND** P3 SHALL remove final deployable model artifacts including `model/final_model.joblib`, `model/final_model_metadata.json`, and `model/inference_schema.json`
- **AND** P3 SHALL keep MR TIFF deployment blocked until a less nonspecific candidate clears the Dox smoke gate

### Requirement: P3 runs an autonomous severe-risk model loop
The workflow SHALL test high-sensitivity severe-risk candidates with threshold selection performed inside development data.

#### Scenario: Severe candidates include high-sensitivity objectives
- **WHEN** severe candidates are evaluated
- **THEN** candidates SHALL include class-balanced or recall-targeted logistic severe gates using ROI/QC, morphology, and ROI/QC plus morphology features
- **AND** candidates SHALL include learned ROI, reduced embedding, learned-plus-morphology, and embedding-plus-morphology severe gates with dimensionality control where needed
- **AND** candidates MAY include tree-based comparators when diagnostics label them exploratory
- **AND** each candidate SHALL report severe recall, precision, false negatives, false positives, AUROC, average precision, and finite-output status

#### Scenario: Learned and embedding-heavy severe candidates are gated not excluded
- **WHEN** learned ROI or embedding-heavy severe candidates are evaluated
- **THEN** dimensionality reduction, feature selection, calibration, and threshold selection SHALL be learned inside grouped development folds
- **AND** validation-fold labels SHALL NOT be used to choose learned feature subsets, principal components, thresholds, or candidate families
- **AND** learned/embedding-heavy candidates SHALL be eligible for final selection when they pass grouped out-of-fold severe gates, finite-output checks, leakage checks, and source-sensitivity checks
- **AND** if they fail, the failure SHALL be recorded as overfit, source-sensitive, numerically unstable, or underperforming rather than omitted

#### Scenario: Threshold selection is development-only
- **WHEN** a severe candidate requires an operating threshold
- **THEN** the threshold SHALL be selected from development folds or inner training splits only
- **AND** the selected threshold, target recall, and achieved development metrics SHALL be written to `summary/severe_threshold_selection.json`

#### Scenario: Severe triage promotion gate is explicit
- **WHEN** grouped out-of-fold severe recall is at least 0.80, grouped out-of-fold severe precision is at least 0.25, outputs are finite, and no leakage is detected
- **THEN** P3 MAY pass the severe safety gate
- **AND** severe triage SHALL become a README-facing output only if richer non-triage grade outputs fail and the MR TIFF deployment test passes

#### Scenario: High-sensitivity low-precision triage is allowed with narrower claims
- **WHEN** grouped out-of-fold severe recall is at least 0.90 but precision is below 0.25 and at least 0.15
- **THEN** P3 MAY pass a lower-precision high-sensitivity severe safety gate
- **AND** the verdict SHALL state that the product is designed to reduce missed severe images and will produce reviewable false positives

### Requirement: P3 tests ordinal burden feasibility before giving up
The workflow SHALL attempt ordinal outputs after and alongside severe-risk modeling.

#### Scenario: Three-band ordinal output is evaluated
- **WHEN** P3 evaluates ordinal burden
- **THEN** it SHALL evaluate `none_low=[0,0.5]`, `mild_mod=[1,1.5]`, and `severe=[2,3]`
- **AND** it SHALL report balanced accuracy, severe-band recall, adjacent-band error rate, non-adjacent error rate, and confusion matrix

#### Scenario: Learned and embedding-heavy ordinal bands are evaluated
- **WHEN** three-band or four-band ordinal outputs are evaluated
- **THEN** P3 SHALL include learned ROI, reduced embedding, learned-plus-morphology, embedding-plus-morphology, and hybrid learned/ROI/morphology candidate families
- **AND** the workflow SHALL report whether learned or embedding-heavy features improve grouped out-of-fold ordinal-band behavior over ROI/QC and morphology-only candidates
- **AND** learned/embedding-heavy ordinal candidates SHALL be eligible for final selection only when they pass grouped out-of-fold ordinal gates and source-sensitivity checks

#### Scenario: Four-band ordinal output is evaluated when supported
- **WHEN** score support permits
- **THEN** P3 SHALL evaluate `0`, `0.5`, `1/1.5`, and `2/3` bands
- **AND** it SHALL report whether the four-band output improves over three-band output without increasing severe false negatives

#### Scenario: Six-bin ordinal output is diagnostic unless gates pass
- **WHEN** six-bin ordinal models are fit
- **THEN** they SHALL preserve the supported rubric `[0, 0.5, 1, 1.5, 2, 3]`
- **AND** they SHALL report exact accuracy, adjacent accuracy, balanced accuracy, calibration diagnostics, and prediction-set width
- **AND** six-bin output SHALL remain diagnostic unless it beats naive and adjacent-baseline comparators with finite grouped out-of-fold evidence

#### Scenario: Ordinal bands can be the final product
- **WHEN** an ordinal band candidate passes grouped out-of-fold balanced accuracy at least 0.50, severe-band recall at least 0.80, finite-output checks, and leakage checks
- **THEN** P3 MAY select ordinal bands as the preferred grade-model output
- **AND** the final product SHALL include both ordinal band prediction and severe-risk flag

### Requirement: P3 distinguishes predictive features from biological proof
The workflow SHALL allow morphology features as predictive covariates without claiming validated closed-lumen biology.

#### Scenario: Blocked morphology remains predictive only
- **WHEN** morphology readiness is `blocked_by_visual_feature_readiness`
- **THEN** P3 MAY still use morphology features for prediction
- **AND** reports SHALL state that morphology features are predictive covariates/QC evidence, not validated mechanistic closed-lumen measurements

#### Scenario: Feature ablations are written
- **WHEN** candidate evaluation completes
- **THEN** P3 SHALL write feature-family ablations comparing ROI/QC, morphology, learned ROI, reduced embeddings, and combined families where available
- **AND** the final verdict SHALL identify which feature families materially improved severe recall or ordinal-band behavior

#### Scenario: Learned feature source-sensitivity is diagnosed
- **WHEN** learned ROI or embedding-heavy features are included in any candidate family
- **THEN** P3 SHALL diagnose cohort/source predictability, fold-to-fold performance instability, and severe/ordinal residuals by cohort where estimable
- **AND** the final verdict SHALL state whether learned features were selected, rejected for source/overfit risk, or retained as diagnostic-only

### Requirement: P3 handles multi-glomerulus aggregate labels without new manual scoring
The workflow SHALL test aggregate-aware image-level candidates for multi-glomerulus inputs without inventing per-glomerulus labels.

#### Scenario: Aggregate-aware features are generated
- **WHEN** masks contain multiple connected glomerulus components
- **THEN** P3 SHALL compute component count, component area summaries, and component-level feature summaries where available
- **AND** summaries SHALL include mean, median, maximum, upper quantile, and spread for eligible component-level morphology, QC, learned, or embedding features

#### Scenario: Aggregate candidates use image-level targets
- **WHEN** aggregate-aware severe or ordinal candidates are evaluated
- **THEN** they SHALL train and evaluate against the image-level severe or ordinal target
- **AND** component-level predictions SHALL be aggregated to image-level outputs before gate evaluation
- **AND** per-component predictions SHALL be labeled diagnostic unless future per-glomerulus labels exist

#### Scenario: Aggregate-label limitation is reported
- **WHEN** P3 writes the final verdict
- **THEN** it SHALL state whether aggregate-aware features improved, worsened, or did not materially change grouped out-of-fold performance
- **AND** if they do not solve the problem, the report SHALL list image-level average labels on multi-glomerulus inputs as a limitation rather than requiring recropping or new per-glomerulus scoring during P3

### Requirement: P3 generates reviewable error evidence automatically
The workflow SHALL produce review artifacts for errors without requiring manual review during execution.

#### Scenario: Severe error review is written
- **WHEN** P3 selects or rejects a severe-risk candidate
- **THEN** it SHALL write `evidence/severe_false_negative_review.html`
- **AND** it SHALL include grouped out-of-fold development false negatives, severe true positives, and high-confidence false positives where available

#### Scenario: Ordinal confusion review is written
- **WHEN** ordinal candidates are evaluated
- **THEN** P3 SHALL write `evidence/ordinal_confusion_review.html`
- **AND** it SHALL show representative adjacent errors, non-adjacent errors, severe-band misses, and correct predictions

### Requirement: P3 final verdict is one of six explicit states
The workflow SHALL end with a bounded final product decision.

#### Scenario: README-facing MR TIFF grade model is selected
- **WHEN** a non-triage grade-output quantification gate passes
- **AND** final model artifacts and inference schema are written
- **AND** the MR TIFF segmentation-to-quantification deployment test passes
- **AND** the severe safety gate passes
- **THEN** `summary/final_product_verdict.json` SHALL set `overall_status` to `readme_facing_deployable_mr_tiff_grade_model`
- **AND** the verdict SHALL state the supported output type, grouped out-of-fold development metrics, MR deployment-test inputs, and current-data/source-sensitive claim boundary

#### Scenario: README-facing MR TIFF severe triage fallback is selected
- **WHEN** no non-triage grade-output gate passes
- **AND** severe triage gates pass
- **AND** final model artifacts and inference schema are written
- **AND** the MR TIFF segmentation-to-quantification deployment test passes
- **THEN** `summary/final_product_verdict.json` SHALL set `overall_status` to `readme_facing_deployable_mr_tiff_severe_triage`
- **AND** the verdict SHALL state that the deployed output is a high-sensitivity review triage model rather than ordinal burden quantification

#### Scenario: Quantification model is ready but MR TIFF deployment test is blocked
- **WHEN** a severe-risk or ordinal-band quantification gate passes
- **AND** final model artifacts and inference schema are written
- **AND** the MR TIFF segmentation-to-quantification deployment test cannot be completed because a supported segmentation artifact or MR TIFF input is missing or fails
- **THEN** `summary/final_product_verdict.json` SHALL set `overall_status` to `model_ready_pending_mr_tiff_deployment_smoke`
- **AND** README-facing deployment language SHALL remain blocked

#### Scenario: Model remains diagnostic only
- **WHEN** grouped development metrics are promising but MR deployment evidence fails or is too incomplete
- **THEN** `summary/final_product_verdict.json` SHALL set `overall_status` to `diagnostic_only_current_data_model`
- **AND** it SHALL explain which gate failed

#### Scenario: Current data are insufficient
- **WHEN** no severe-risk or ordinal candidate beats baseline gates after bounded attempts
- **THEN** `summary/final_product_verdict.json` SHALL set `overall_status` to `current_data_insufficient`
- **AND** it SHALL report the strongest failed candidate, failure mode, and minimum additional data or annotation most likely to change the conclusion

### Requirement: P3 writes deployment artifacts only when gates pass
The workflow SHALL create deployable artifacts only for supported final product states.

#### Scenario: Model artifact is written for passing product
- **WHEN** P3 selects `readme_facing_deployable_mr_tiff_grade_model`, `readme_facing_deployable_mr_tiff_severe_triage`, or `model_ready_pending_mr_tiff_deployment_smoke`
- **THEN** it SHALL write `model/final_model.joblib`, `model/final_model_metadata.json`, `model/inference_schema.json`, and `model/deployment_smoke_predictions.csv`
- **AND** metadata SHALL include feature columns, preprocessing steps, selected threshold, target definition, grouped out-of-fold metrics, package versions, and claim boundary

#### Scenario: No deployable model is written for failed quantification gates
- **WHEN** P3 selects `diagnostic_only_current_data_model` or `current_data_insufficient`
- **THEN** it SHALL NOT write a deployable final model artifact
- **AND** it SHALL write diagnostics sufficient to resume from the strongest failed candidate
- **AND** it SHALL remove stale deployable model artifacts from prior runs under `burden_model/endotheliosis_grade_model/model/`

### Requirement: P3 proves end-to-end MR TIFF deployment before README promotion
The workflow SHALL require a segmentation-to-quantification deployment test on MR whole-field TIFF data before any README-facing deployment claim.

#### Scenario: MR deployment test uses supported segmentation output
- **WHEN** P3 attempts README-facing promotion
- **THEN** it SHALL identify a supported current-namespace glomerulus segmentation artifact
- **AND** it SHALL use `vegfri_mr` whole-field TIFF input images from the runtime cohort manifest when available
- **AND** it SHALL tile whole-field TIFFs, run segmentation inference or a supported segmentation-output loading path, merge tile predictions into whole-field coordinates, filter components by area and quality, and generate accepted glomerulus ROI/mask records for P3 inference

#### Scenario: MR deployment test runs the selected quantification model
- **WHEN** segmentation or supported segmentation outputs produce ROI/mask records
- **THEN** P3 SHALL compute the exact columns required by `model/inference_schema.json`
- **AND** it SHALL load `model/final_model.joblib`
- **AND** it SHALL write `deployment/mr_tiff_smoke_predictions.csv`
- **AND** it SHALL write `deployment/segmentation_quantification_contract.json` describing the required image, segmentation, ROI, feature, model, and prediction fields

#### Scenario: MR deployment test aggregates image-level predictions
- **WHEN** ROI-level predictions are written for MR TIFF inputs
- **THEN** P3 SHALL aggregate accepted ROI predictions to image-level median and summary outputs
- **AND** rows with zero accepted inferred ROIs SHALL be marked non-evaluable, not silently admitted
- **AND** when human image-level median scores or replicate summaries are available, P3 SHALL report human-vs-inferred concordance
- **AND** when labels are absent, P3 SHALL mark the MR result as technical deployment smoke only rather than accuracy evidence

#### Scenario: MR deployment report is reviewable
- **WHEN** MR deployment test completes
- **THEN** P3 SHALL write `deployment/mr_tiff_smoke_report.html`
- **AND** the report SHALL show input image identity, segmentation artifact identity, tile/coordinate assumptions, ROI count, failed ROI count, accepted ROI count, prediction count, image-level aggregate, selected output type, human-vs-inferred concordance when available, and any nonfatal exclusions

#### Scenario: Feature-table-only predictions are insufficient
- **WHEN** P3 writes predictions directly from precomputed feature tables without proving segmentation-to-quantification execution
- **THEN** P3 SHALL NOT select `readme_facing_deployable_mr_tiff_grade_model` or `readme_facing_deployable_mr_tiff_severe_triage`
- **AND** it SHALL set `model_ready_pending_mr_tiff_deployment_smoke` when quantification gates otherwise pass

#### Scenario: Missing deployment inputs block README-facing status
- **WHEN** no supported current-namespace segmentation artifact or no MR TIFF input is available
- **THEN** P3 SHALL record the missing input in `summary/final_product_verdict.json`
- **AND** it SHALL NOT update README-facing deployment language

### Requirement: P3 artifacts are manifest-listed and indexed
The workflow SHALL make the output subtree navigable and complete.

#### Scenario: Manifest and index are written
- **WHEN** P3 completes
- **THEN** it SHALL write `INDEX.md`
- **AND** it SHALL write `summary/artifact_manifest.json`
- **AND** it SHALL write `summary/candidate_coverage_matrix.csv`
- **AND** the manifest SHALL list every P3 artifact by path, role, consumer, required status, reportability, and existence
- **AND** no unmanifested files SHALL remain under the grade-model root except explicitly ignored hidden filesystem files

#### Scenario: Candidate coverage matrix is complete
- **WHEN** P3 completes
- **THEN** `summary/candidate_coverage_matrix.csv` SHALL include every committed candidate family
- **AND** each row SHALL include `family_id`, `subtree_path`, `required_or_exploratory`, `run_status`, `candidate_ids`, `metrics_path`, `diagnostics_path`, `predictions_path`, `gate_status`, `selected`, and `failure_or_exclusion_reason`
- **AND** the selected row SHALL link to the source family subtree that produced the deployed model
- **AND** missing expected upstream artifacts for required families SHALL be represented as hard blockers, not ordinary skipped rows

#### Scenario: Executive summary is written
- **WHEN** P3 completes
- **THEN** it SHALL write `summary/executive_summary.md`
- **AND** the summary SHALL state what was performed, which data and artifacts were used, which candidate families were tested, the selected verdict state, key severe and ordinal results, MR deployment-test result, limitations, and concrete next steps
- **AND** the summary SHALL explicitly state whether any next step is required before README-facing deployment
