## ADDED Requirements

### Requirement: Severe-aware estimator uses the current quantification workflow
The severe-aware ordinal estimator SHALL run inside the existing quantification workflow and SHALL use the current scored MR TIFF/ROI evidence contract.

#### Scenario: Estimator runs from main quantification config
- **WHEN** `eq run-config --config configs/endotheliosis_quantification.yaml` completes burden, learned ROI, and source-aware evaluation
- **THEN** the workflow SHALL call `evaluate_severe_aware_ordinal_endotheliosis_estimator`
- **AND** the estimator SHALL use `subject_id` as the validation grouping key
- **AND** the estimator SHALL use `cohort_id` as the primary source/context field
- **AND** the estimator SHALL preserve `subject_id`, `sample_id`, `image_id`, `subject_image_id`, `cohort_id`, `score`, and ROI/image path provenance in prediction and review artifacts where present

#### Scenario: Source-aware handoff is consumed
- **WHEN** `burden_model/source_aware_estimator/summary/estimator_verdict.json` exists
- **THEN** the severe-aware estimator SHALL read its verdict as upstream current-data evidence
- **AND** it SHALL record selected P1 image candidate, selected P1 subject candidate, hard blockers, scope limiters, reportable scopes, and testing status in its own diagnostics or verdict
- **AND** it SHALL NOT treat P1 source-aware outputs as external validation or promoted truth

### Requirement: Severe-aware evaluator participates in repo-wide execution logging
The severe-aware evaluator SHALL emit operational logger events as a high-level function while durable log capture remains owned by the existing quantification execution surface.

#### Scenario: Evaluator emits function-level events
- **WHEN** `evaluate_severe_aware_ordinal_endotheliosis_estimator` runs inside the quantification workflow or under caller-owned logging
- **THEN** it SHALL emit logger events for start, resolved input artifact roots, output root, row count, subject count, source count, threshold-support summary, selected feature families, candidate IDs, hard blockers, scope limiters, verdict path, artifact manifest path, completion status, and elapsed time where available
- **AND** these events SHALL be capturable by the repo-wide execution logging context when the existing `endotheliosis_quantification` workflow is run through `eq run-config`

#### Scenario: Evaluator failure is logged and re-raised
- **WHEN** the severe-aware evaluator fails after execution begins
- **THEN** it SHALL log failure context including the surface name, failing step where known, available input or output roots, exception message, and verdict/artifact status where known
- **AND** it SHALL re-raise rather than converting the failure into a successful limited verdict unless the failure is an explicitly modeled scientific scope limiter

#### Scenario: Evaluator does not own durable execution logging
- **WHEN** the severe-aware evaluator is imported or called by another Python function
- **THEN** it SHALL NOT call `setup_logging(...)`, attach durable file handlers, create `$EQ_RUNTIME_ROOT/logs/...`, create repo-root `logs/`, or implement custom subprocess stdout/stderr teeing
- **AND** durable capture SHALL remain the responsibility of the existing `endotheliosis_quantification` workflow and repo-wide execution logging contract

#### Scenario: Logging docs impact is bounded
- **WHEN** P2 adds the severe-aware evaluator without adding a new CLI command or log root
- **THEN** P2 documentation SHALL describe the severe-aware artifact subtree and review links
- **AND** it SHALL NOT duplicate the repo-wide logging documentation unless P2 changes log roots, automatic durable logging behavior, or public execution commands

### Requirement: Severe separability audit precedes final candidate selection
The estimator SHALL audit whether severe cases are separable in current features before selecting severe-aware candidates.

#### Scenario: Severe separability audit is written
- **WHEN** severe-aware estimator evaluation begins
- **THEN** the workflow SHALL write `burden_model/severe_aware_ordinal_estimator/diagnostics/severe_separability_audit.json`
- **AND** the artifact SHALL include row count, subject count, source support, and positive/negative support for `score >= 1.5`, `score >= 2`, and `score >= 3`
- **AND** the artifact SHALL summarize available ROI/QC, morphology, learned ROI, and embedding-derived feature families
- **AND** the artifact SHALL report feature count, nonfinite count, zero-variance count, near-zero-variance count, and numerical-warning or rank-diagnostic status where feasible

#### Scenario: Non-separable severe cases are reported
- **WHEN** the audit finds that a severe threshold is underpowered, source-confounded beyond estimation, or not separable under current features
- **THEN** the estimator SHALL mark that threshold as `non_estimable` or `exploratory`
- **AND** it SHALL record the concrete reason
- **AND** it SHALL NOT silently fit and promote a candidate for that threshold

### Requirement: Annotation and segmentation-backbone escalation are evidence gated
The severe-aware workflow SHALL decide whether new manual annotation or upstream segmentation-backbone work is justified from observed failure localization rather than assuming more masks or a different segmenter are required.

#### Scenario: Severe false negatives are localized before annotation is recommended
- **WHEN** severe false negatives or high-uncertainty severe cases are reviewed
- **THEN** the evidence review SHALL record whether each reviewed failure appears driven by bad ROI extraction, missed glomerulus segmentation, mask/ROI geometry error, feature/model limitation, or visually ambiguous grading signal where determinable
- **AND** the verdict SHALL state whether new Label Studio patch/mask annotation is recommended, deferred, or not supported by current evidence

#### Scenario: Better segmenter recommendation requires upstream failure evidence
- **WHEN** the severe-aware verdict recommends a future segmentation-backbone comparison
- **THEN** the recommendation SHALL identify the observed upstream failure mode it is intended to address
- **AND** it SHALL distinguish glomerulus/ROI segmentation improvement from severe-grade prediction improvement
- **AND** it SHALL not imply that a stronger segmenter will solve score `2/3` underprediction without evidence that ROI extraction is the limiting factor

#### Scenario: Promptable segmenter audit separates oracle and automatic prompts
- **WHEN** the severe-aware verdict recommends auditing MedSAM, SAM, or another promptable segmenter
- **THEN** the recommendation SHALL require separate oracle-prompt and automatic-prompt evaluations
- **AND** oracle prompts SHALL be labeled as upper-bound evidence when boxes or points are derived from manual masks
- **AND** automatic prompts SHALL be generated from deployable sources such as current glomerulus proposals, connected components, tiling proposals, or ROI candidates
- **AND** oracle-prompt performance SHALL NOT be treated as deployable segmentation performance

#### Scenario: Promptable segmenter audit measures downstream relevance
- **WHEN** MedSAM, SAM, or another promptable segmenter is compared against the current segmentation path
- **THEN** the comparison SHALL report segmentation metrics against existing masks where available
- **AND** it SHALL report prompt-generation failures, missed glomeruli, duplicate prompts, mask area changes, ROI-feature changes, and severe false-negative changes where estimable
- **AND** it SHALL state whether any segmentation improvement changes severe-aware quantification behavior

#### Scenario: Segmentation-derived features are tested for severe signal
- **WHEN** alternative segmentation masks or component decompositions are available for comparison
- **THEN** the severe-aware audit SHALL recompute severity-relevant ROI and morphology features from those masks where feasible
- **AND** it SHALL test whether those feature changes improve severe separability, `score >= 2` false-negative behavior, or ordinal prediction-set usefulness
- **AND** it SHALL distinguish a pure mask-geometry improvement from an improvement in severity-correlated feature extraction

#### Scenario: Segmentation baseline feasibility is inventoried before comparison
- **WHEN** the severe-aware verdict recommends exploring nnU-Net, DeepLab, Mask2Former-style, MedSAM, SAM, or other upstream segmentation baselines
- **THEN** the recommendation SHALL be backed by a feasibility inventory
- **AND** the inventory SHALL record whether each package or model family is installed in the certified environment, whether installation is required, whether weights or code require network download, whether macOS/MPS or WSL/CUDA is the intended runtime, expected training or inference cost, required dataset conversion, and whether existing masks are sufficient
- **AND** unavailable model families SHALL NOT be described as current supported runtime dependencies

#### Scenario: Existing masks are the first comparison substrate
- **WHEN** upstream segmentation baselines are explored
- **THEN** first-pass comparisons SHALL use existing image/mask pairs where possible
- **AND** new manual annotation SHALL NOT be required for the first feasibility comparison unless the inventory records that existing masks cannot answer the comparison question
- **AND** outputs SHALL distinguish architecture feasibility from scientific improvement in endotheliosis quantification

#### Scenario: Annotation pilot remains targeted
- **WHEN** the verdict recommends new manual annotation
- **THEN** the recommendation SHALL specify a targeted pilot population such as severe false negatives, uncertain MR TIFF ROIs, or suspected ROI extraction failures
- **AND** it SHALL avoid recommending broad large-TIFF patch masking unless a smaller targeted pilot is insufficient or infeasible

### Requirement: Severe-threshold support is explicit
The estimator SHALL evaluate severe thresholds only when support is sufficient and SHALL record non-estimable thresholds honestly.

#### Scenario: Threshold support artifact is written
- **WHEN** severe-aware estimator evaluation completes
- **THEN** the workflow SHALL write `burden_model/severe_aware_ordinal_estimator/diagnostics/threshold_support.json`
- **AND** the artifact SHALL report positive row count, negative row count, positive subject count, negative subject count, and source support for `score >= 1.5`, `score >= 2`, and `score >= 3`
- **AND** the artifact SHALL identify which thresholds are eligible for candidate fitting, exploratory only, or non-estimable

#### Scenario: Score greater than or equal to two is primary
- **WHEN** the current score rubric contains observed score `2` or `3` rows
- **THEN** `score >= 2` SHALL be the primary severe threshold for false-negative review and severe-risk reporting
- **AND** candidate metrics SHALL include severe recall, severe false-negative count, severe false-negative rate, and severe precision where estimable

#### Scenario: Score greater than or equal to three is support-gated
- **WHEN** `score >= 3` has insufficient independent subject support or insufficient source support
- **THEN** the estimator SHALL report `score >= 3` as exploratory or non-estimable
- **AND** it SHALL not select a `score >= 3` threshold model as a reportable current-data result

### Requirement: Candidate set is severe-aware and bounded
The severe-aware estimator SHALL evaluate a small fixed first-pass candidate set unless a later OpenSpec change expands it.

#### Scenario: Candidate IDs are capped
- **WHEN** severe-aware candidates are fit
- **THEN** allowed first-pass candidate IDs SHALL be limited to `severe_roi_qc_threshold`, `severe_morphology_threshold`, `severe_roi_qc_morphology_threshold`, `ordinal_roi_qc_thresholds`, `ordinal_roi_qc_morphology_thresholds`, `two_stage_severe_gate_roi_qc`, `two_stage_severe_gate_roi_qc_morphology`, and `subject_severe_aware_ordinal`
- **AND** learned or embedding-heavy candidates SHALL be included only as audit comparators unless the severe separability audit records them as eligible for final candidate selection
- **AND** new fitted foundation or backbone feature providers SHALL NOT be added by this change

#### Scenario: Candidate metrics include severe behavior
- **WHEN** candidate metrics are written
- **THEN** `burden_model/severe_aware_ordinal_estimator/internal/candidate_metrics.csv` SHALL include candidate ID, target level, feature family, model family, threshold target where applicable, row count, subject count, source scope, validation split, stage-index MAE, grade-scale MAE, severe recall, severe precision, severe false-negative count, severe false-negative rate, finite-output status, warning count, and intended use
- **AND** every candidate row SHALL state whether it is eligible for model selection

### Requirement: Ordinal score structure is preserved
The estimator SHALL model the ordered score rubric explicitly rather than treating the target only as an unconstrained scalar.

#### Scenario: Ordinal threshold predictions are written
- **WHEN** an ordinal/cumulative-threshold candidate is selected or retained for review
- **THEN** image predictions SHALL include threshold probabilities or threshold decisions for ordered cutpoints derived from `[0, 0.5, 1, 1.5, 2, 3]`
- **AND** public threshold probabilities SHALL be finite and monotonic after any deterministic correction
- **AND** the correction method SHALL be recorded in the candidate metadata

#### Scenario: Ordinal prediction sets are written
- **WHEN** ordinal prediction sets are estimable
- **THEN** `predictions/image_predictions.csv` SHALL include an ordinal prediction-set field containing only supported rubric values
- **AND** the set construction method SHALL be recorded in `summary/estimator_verdict.json` or `diagnostics/reliability_labels.json`
- **AND** ordinal prediction sets SHALL not be described as independent testing

### Requirement: Severe false negatives are primary review evidence
The estimator SHALL make severe underprediction visible as a first-class failure mode.

#### Scenario: Severe false-negative review is written
- **WHEN** severe-aware estimator evaluation completes
- **THEN** the workflow SHALL write `burden_model/severe_aware_ordinal_estimator/evidence/severe_false_negative_review.html`
- **AND** the review SHALL include representative severe true positives, severe false negatives, low/mid false positives, high-uncertainty severe cases, and source-stratified severe examples where available
- **AND** each example SHALL show observed score, predicted severe risk or ordinal set, predicted burden where available, reliability label, `cohort_id`, `subject_id`, and ROI/image provenance

#### Scenario: Severe false negatives affect verdict
- **WHEN** selected candidates have high severe false-negative rates for `score >= 2`
- **THEN** `summary/estimator_verdict.json` SHALL list severe underprediction as a scope limiter or hard blocker depending on severity and output claim
- **AND** the combined review SHALL not describe scalar burden estimates as reliable for severe endotheliosis

### Requirement: Split labels distinguish training validation and testing
The severe-aware estimator SHALL report training/apparent, validation, and testing metrics with explicit split labels.

#### Scenario: Metrics by split are written
- **WHEN** severe-aware estimator evaluation completes
- **THEN** the workflow SHALL write `burden_model/severe_aware_ordinal_estimator/summary/metrics_by_split.csv`
- **AND** the workflow SHALL write `burden_model/severe_aware_ordinal_estimator/summary/metrics_by_split.json`
- **AND** each metrics row SHALL include split label, split description, candidate ID, target level, row count, subject count, source scope, primary severe threshold, grade-scale metrics, stage-index metrics, severe-threshold metrics where applicable, finite-output status, warning count, intended use, and model-selection eligibility

#### Scenario: Apparent metrics are not model-selection evidence
- **WHEN** full-data or in-fold metrics are reported
- **THEN** they SHALL use `training_apparent`
- **AND** they SHALL be marked ineligible for model selection
- **AND** reports SHALL state that they are optimistic diagnostics

#### Scenario: Subject-heldout validation is primary
- **WHEN** grouped out-of-fold validation is performed
- **THEN** metrics SHALL use `validation_subject_heldout`
- **AND** no `subject_id` SHALL appear in both train and validation partitions for a candidate fold
- **AND** this split SHALL be the primary current-data model-selection split unless an explicit held-out test partition is predeclared

#### Scenario: No explicit test set is labeled honestly
- **WHEN** no predeclared independent held-out test partition exists
- **THEN** the metrics artifact SHALL include `testing_not_available_current_data_sensitivity`
- **AND** source-stratified, leave-source-out, severe-tail, and full-dataset checks SHALL NOT be labeled as independent testing

### Requirement: Source behavior is scoped for severe thresholds
The estimator SHALL report source-specific severe-threshold behavior without presenting it as external validation.

#### Scenario: Source severe sensitivity artifact is written
- **WHEN** severe-aware estimator evaluation completes
- **THEN** the workflow SHALL write `burden_model/severe_aware_ordinal_estimator/diagnostics/source_severe_sensitivity.json`
- **AND** the artifact SHALL include threshold support by `cohort_id`, severe false-negative behavior by `cohort_id`, threshold metrics by `cohort_id` where estimable, and leave-source-out severe-threshold behavior where estimable
- **AND** non-estimable source/threshold cells SHALL be labeled with reasons

#### Scenario: Source-confounded severe support limits claims
- **WHEN** severe positives are present in only one source or severe threshold support is materially source-confounded
- **THEN** the verdict SHALL include a source-sensitivity scope limiter
- **AND** the report SHALL state that severe-threshold evidence is current-data/source-sensitive rather than externally validated

### Requirement: Artifacts are contained indexed and capped
The severe-aware estimator SHALL write all artifacts under one indexed output subtree and SHALL avoid duplicate flat aliases.

#### Scenario: Indexed output subtree is written
- **WHEN** severe-aware estimator evaluation completes
- **THEN** the workflow SHALL write `burden_model/severe_aware_ordinal_estimator/INDEX.md`
- **AND** all severe-aware estimator artifacts SHALL live under `summary/`, `predictions/`, `diagnostics/`, `evidence/`, or `internal/` within `burden_model/severe_aware_ordinal_estimator/`
- **AND** the workflow SHALL NOT write duplicate severe-aware estimator artifacts to flat `burden_model/*` locations

#### Scenario: Artifact manifest caps first-pass outputs
- **WHEN** severe-aware estimator evaluation completes
- **THEN** the workflow SHALL write `burden_model/severe_aware_ordinal_estimator/summary/artifact_manifest.json`
- **AND** the manifest SHALL list every severe-aware estimator artifact by relative path, role, consumer, reportability, and required status
- **AND** first-pass outputs SHALL be limited to the manifest-listed artifacts unless a named consumer is added to the manifest

### Requirement: Severe-aware verdict is reader-first
The estimator SHALL produce a top-level verdict that explains what happened, what can be trusted, and what remains limited.

#### Scenario: Verdict artifacts are written
- **WHEN** severe-aware estimator evaluation completes
- **THEN** the workflow SHALL write `burden_model/severe_aware_ordinal_estimator/summary/estimator_verdict.json`
- **AND** the workflow SHALL write `burden_model/severe_aware_ordinal_estimator/summary/estimator_verdict.md`
- **AND** the verdict SHALL identify selected image candidate if any, selected subject candidate if any, selected severe threshold if any, selected ordinal candidate if any, hard blockers, scope limiters, reportable scopes, non-reportable scopes, testing status, README eligibility, next action, and claim boundary

#### Scenario: Verdict distinguishes output types
- **WHEN** the estimator has enough evidence to report a result
- **THEN** the verdict SHALL state whether the reportable output is scalar burden, severe-risk label, ordinal prediction set, subject-level aggregate, aggregate current-data summary, or limited/non-reportable evidence
- **AND** it SHALL not imply that success for one output type validates the others

#### Scenario: README snippets remain opt-in
- **WHEN** `readme_snippet_eligible` is false
- **THEN** `quantification_review/readme_results_snippet.md` SHALL NOT include severe-aware estimator results
- **AND** the combined review SHALL still link to the severe-aware estimator index for runtime review

### Requirement: Severe-aware summary figures are capped and diagnostic
The estimator SHALL write a capped set of human-facing figures focused on severe behavior, ordinal behavior, source sensitivity, and uncertainty.

#### Scenario: Summary figures are written
- **WHEN** severe-aware estimator evaluation completes
- **THEN** the workflow SHALL write no more than eight first-pass summary figures under `burden_model/severe_aware_ordinal_estimator/summary/figures/`
- **AND** the figures SHALL include severe threshold metrics, predicted versus observed severity, severe false-negative review summary, calibration by score or threshold, source severe performance, uncertainty or ordinal prediction-set width, and reliability-label counts where estimable
- **AND** every figure SHALL be listed in `summary/artifact_manifest.json`

### Requirement: Hard blockers and scope limiters are separated
The estimator SHALL separate invalidating failures from limitations that constrain interpretation.

#### Scenario: Reliability diagnostics are written
- **WHEN** severe-aware estimator evaluation completes
- **THEN** the workflow SHALL write `burden_model/severe_aware_ordinal_estimator/diagnostics/reliability_labels.json`
- **AND** the artifact SHALL define emitted prediction labels and verdict labels
- **AND** hard blockers SHALL include unsupported scores, broken joins, nonfinite selected predictions, subject validation leakage, missing required identity fields, missing required verdict/index artifacts, and claims outside predictive grade-equivalent or severe-risk evidence

#### Scenario: Severe limitations do not disappear
- **WHEN** score-2/3 underprediction, source-confounded severe support, broad ordinal prediction sets, underpowered thresholds, or nonfatal numerical warnings occur without a hard blocker
- **THEN** the verdict SHALL classify those as scope limiters
- **AND** predictions SHALL remain present with reliability labels
- **AND** reports SHALL avoid presenting limited scopes as solved
